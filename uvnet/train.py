"""
uvnet.train  —  학습/평가 유틸리티

루트의 train.py / test.py 에서 이 모듈의 함수를 import해서 사용합니다.

공개 API
--------
split_dataset(data_dir, train_ratio, val_ratio, seed)
    → (train_ds, val_ds, test_ds)

run_epoch(model, loader, face_loss_fn, graph_loss_fn, alpha, beta, optimizer=None)
    → metrics dict

evaluate_detail(model, loader, face_loss_fn, graph_loss_fn, alpha, beta)
    → metrics dict  +  per-class accuracy 콘솔 출력
"""

import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from uvnet.model import (
    NUM_FACE_CLASSES,
    NUM_GRAPH_CLASSES,
    FACE_LABEL_NAMES,
    GRAPH_LABEL_NAMES,
)
from uvnet.dataset import GraphDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 split
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float   = 0.1,
    seed: int          = 42,
) -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
    """
    디렉토리의 .pt 파일을 랜덤 셔플 후 train / val / test 로 분할.

    test 비율 = 1 - train_ratio - val_ratio
    """
    dataset = GraphDataset.from_directory(data_dir)
    n       = len(dataset)

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_files = [dataset._files[i] for i in indices[:n_train]]
    val_files   = [dataset._files[i] for i in indices[n_train : n_train + n_val]]
    test_files  = [dataset._files[i] for i in indices[n_train + n_val :]]

    print(
        f"데이터 split  total={n}  "
        f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )
    return GraphDataset(train_files), GraphDataset(val_files), GraphDataset(test_files)


# ──────────────────────────────────────────────────────────────────────────────
# epoch 단위 학습 / 평가
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    face_loss_fn: nn.Module,
    graph_loss_fn: nn.Module,
    alpha: float,
    beta: float,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    """
    한 epoch를 학습(optimizer 제공) 또는 평가(optimizer=None)합니다.

    Returns
    -------
    dict with keys: loss, face_loss, graph_loss, face_acc, graph_acc
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss  = face_total = graph_total = 0.0
    n_face_ok   = n_face_all = 0
    n_graph_ok  = n_graph_all = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)

            face_logits, graph_logits = model(batch)

            l_face  = face_loss_fn(face_logits, batch.face_y)
            l_graph = graph_loss_fn(graph_logits, batch.graph_y)
            loss    = alpha * l_face + beta * l_graph

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item()
            face_total  += l_face.item()
            graph_total += l_graph.item()

            # face accuracy (ignore_index = -1)
            valid = batch.face_y >= 0
            if valid.any():
                pred_f = face_logits[valid].argmax(dim=1)
                n_face_ok  += (pred_f == batch.face_y[valid]).sum().item()
                n_face_all += valid.sum().item()

            # graph accuracy
            pred_g = graph_logits.argmax(dim=1)
            n_graph_ok  += (pred_g == batch.graph_y).sum().item()
            n_graph_all += batch.graph_y.size(0)

    n = max(len(loader), 1)
    return {
        "loss"      : total_loss  / n,
        "face_loss" : face_total  / n,
        "graph_loss": graph_total / n,
        "face_acc"  : n_face_ok  / max(n_face_all,  1),
        "graph_acc" : n_graph_ok / max(n_graph_all, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 상세 평가 (클래스별 accuracy 포함)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_detail(
    model: nn.Module,
    loader: DataLoader,
    face_loss_fn: nn.Module,
    graph_loss_fn: nn.Module,
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    """
    전체 metrics + 클래스별 accuracy를 단일 순회로 계산합니다.

    Returns
    -------
    dict with keys: loss, face_loss, graph_loss, face_acc, graph_acc
    """
    from collections import defaultdict

    model.eval()

    total_loss  = face_total = graph_total = 0.0
    n_face_ok   = n_face_all = 0
    n_graph_ok  = n_graph_all = 0
    face_ok     = defaultdict(int)
    face_cnt    = defaultdict(int)
    graph_ok    = defaultdict(int)
    graph_cnt   = defaultdict(int)
    all_graph_gt:   list = []
    all_graph_pred: list = []
    all_face_gt:    list = []
    all_face_pred:  list = []

    for batch in loader:
        batch = batch.to(DEVICE)
        face_logits, graph_logits = model(batch)

        l_face  = face_loss_fn(face_logits, batch.face_y)
        l_graph = graph_loss_fn(graph_logits, batch.graph_y)
        loss    = alpha * l_face + beta * l_graph

        total_loss  += loss.item()
        face_total  += l_face.item()
        graph_total += l_graph.item()

        valid = batch.face_y >= 0
        if valid.any():
            pred_f = face_logits[valid].argmax(1)
            gt_f   = batch.face_y[valid]
            n_face_ok  += (pred_f == gt_f).sum().item()
            n_face_all += valid.sum().item()
            for p, g in zip(pred_f.cpu().tolist(), gt_f.cpu().tolist()):
                face_cnt[g] += 1
                if p == g:
                    face_ok[g] += 1
            all_face_gt.extend(gt_f.cpu().tolist())
            all_face_pred.extend(pred_f.cpu().tolist())

        pred_g = graph_logits.argmax(1)
        n_graph_ok  += (pred_g == batch.graph_y).sum().item()
        n_graph_all += batch.graph_y.size(0)
        all_graph_gt.extend(batch.graph_y.cpu().tolist())
        all_graph_pred.extend(pred_g.cpu().tolist())
        for p, g in zip(pred_g.cpu().tolist(), batch.graph_y.cpu().tolist()):
            graph_cnt[g] += 1
            if p == g:
                graph_ok[g] += 1

    n = max(len(loader), 1)

    face_per_class  = {c: face_ok[c]  / max(face_cnt[c],  1) for c in face_cnt}
    graph_per_class = {c: graph_ok[c] / max(graph_cnt[c], 1) for c in graph_cnt}

    metrics = {
        "loss"           : total_loss  / n,
        "face_loss"      : face_total  / n,
        "graph_loss"     : graph_total / n,
        "face_acc"       : n_face_ok  / max(n_face_all,  1),
        "graph_acc"      : n_graph_ok / max(n_graph_all, 1),
        "face_per_class" : face_per_class,
        "graph_per_class": graph_per_class,
        "graph_gt"       : all_graph_gt,
        "graph_pred"     : all_graph_pred,
        "face_gt"        : all_face_gt,
        "face_pred"      : all_face_pred,
    }

    print("\n  [Face segmentation — per-class accuracy]")
    for c in range(NUM_FACE_CLASSES):
        total = face_cnt[c]
        if total == 0:
            continue
        name = FACE_LABEL_NAMES[c] if c < len(FACE_LABEL_NAMES) else str(c)
        print(f"    {name:25s}: {face_ok[c]:4d}/{total:4d}  ({face_ok[c]/total:.3f})")

    print("\n  [Graph classification — per-class accuracy]")
    for c in range(NUM_GRAPH_CLASSES):
        total = graph_cnt[c]
        if total == 0:
            continue
        name = GRAPH_LABEL_NAMES[c] if c < len(GRAPH_LABEL_NAMES) else str(c)
        print(f"    {name:25s}: {graph_ok[c]:4d}/{total:4d}  ({graph_ok[c]/total:.3f})")

    return metrics
