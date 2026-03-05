"""
UVNet Multi-task 테스트 스크립트

학습된 체크포인트를 로드하고 테스트 데이터셋에서 상세 평가를 수행합니다.

실행 예시
---------
    # 기본 수치 평가
    python test.py

    # 특정 체크포인트 지정
    python test.py --ckpt data/checkpoints/best_model.pt

    # 혼동 행렬 이미지 저장 (results/ 폴더)
    python test.py --save_cm

    # GT vs 예측 face 라벨 OCC Viewer 시각화 (기본 5개)
    python test.py --viz

    # 시각화할 케이스 수 지정
    python test.py --viz --n_viz 10

    # 전부 함께
    python test.py --save_cm --viz --n_viz 8 --results_dir results
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # GUI 없는 백엔드 (이미지 저장 전용)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from uvnet.model import UVNetMultiTask
from uvnet.train import split_dataset, evaluate_detail

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CONFIG = "config/train_config.json"

GRAPH_LABEL_NAMES = {0: "turning", 1: "milling", 2: "milling+turning"}
FACE_LABEL_NAMES  = [
    "stock", "step", "groove", "chamfer", "fillet",
    "blind_hole", "through_hole", "rect_pocket", "rect_passage",
]

# 정답 / 오답 표시 색상 (RGB 0-255)
_COLOR_CORRECT = [50,  205,  50]   # limegreen
_COLOR_WRONG   = [220,  50,  50]   # crimson
_COLOR_UNKNOWN = [160, 160, 160]   # gray


# ──────────────────────────────────────────────────────────────────────────────
# Config + argparse
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if Path(pre_args.config).exists():
        with open(pre_args.config, encoding="utf-8") as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    p = argparse.ArgumentParser(description="UVNet Multi-task 테스트")
    p.add_argument("--config",      default=DEFAULT_CONFIG)
    p.add_argument("--ckpt",        default=str(Path(cfg.get("ckpt_dir", "data/checkpoints")) / "best_model.pt"))
    p.add_argument("--data_dir",    default=cfg.get("data_dir",    "data/graphs"))
    p.add_argument("--batch_size",  type=int,   default=cfg.get("batch_size", 8))
    p.add_argument("--alpha",       type=float, default=cfg.get("alpha",      1.0))
    p.add_argument("--beta",        type=float, default=cfg.get("beta",       1.0))
    p.add_argument("--train_ratio", type=float, default=cfg.get("train_ratio", 0.8))
    p.add_argument("--val_ratio",   type=float, default=cfg.get("val_ratio",   0.1))
    p.add_argument("--seed",        type=int,   default=cfg.get("seed",        42))
    p.add_argument("--save_cm",     action="store_true",
                   help="혼동 행렬 이미지를 results_dir에 저장")
    p.add_argument("--results_dir", default="results",
                   help="혼동 행렬 이미지 저장 폴더 (기본: results)")
    p.add_argument("--viz",         action="store_true",
                   help="OCC Viewer 로 GT / 예측 / 정오답 face 라벨 시각화")
    p.add_argument("--n_viz",       type=int, default=5,
                   help="시각화할 테스트 케이스 수 (기본 5)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix — 텍스트 (콘솔)
# ──────────────────────────────────────────────────────────────────────────────

def _print_confusion_matrix(all_gt: List[int], all_pred: List[int], labels: List[str]):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for g, p in zip(all_gt, all_pred):
        if 0 <= g < n and 0 <= p < n:
            mat[g][p] += 1

    col_w = max(len(l) for l in labels) + 2
    idx_w = max(len(l) for l in labels) + 2

    header = " " * idx_w + "".join(l.center(col_w) for l in labels) + "  <- 예측"
    print(header)
    print(" " * idx_w + "-" * (col_w * n))
    for i, row in enumerate(mat):
        row_str = labels[i].ljust(idx_w) + "".join(str(v).center(col_w) for v in row)
        print(row_str)


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix — 이미지 저장
# ──────────────────────────────────────────────────────────────────────────────

def _save_confusion_matrix_image(
    all_gt: List[int],
    all_pred: List[int],
    labels: List[str],
    title: str,
    save_path: str,
) -> None:
    """
    혼동 행렬을 matplotlib 히트맵으로 그려 PNG 저장.

    셀 배경 강도: Blues colormap (정규화 없이 절대 빈도)
    셀 텍스트: 절대 빈도 + 비율(%)
    대각선: 정답 → 파란 계열, 오차 셀 → 강조
    """
    n   = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for g, p in zip(all_gt, all_pred):
        if 0 <= g < n and 0 <= p < n:
            mat[g][p] += 1

    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    mat_pct  = mat / row_sums * 100.0          # 행 단위 비율(%)

    cell_size = max(1.0, 7.0 / max(n, 1))
    fig_w = max(6, n * cell_size + 2)
    fig_h = max(5, n * cell_size + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    thresh = 50.0
    for i in range(n):
        for j in range(n):
            cnt = mat[i, j]
            pct = mat_pct[i, j]
            txt = f"{cnt}\n({pct:.1f}%)"
            fg  = "white" if pct >= thresh else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=fg, fontweight="bold" if i == j else "normal")

    ax.set_xlabel("예측 (Predicted)", fontsize=10)
    ax.set_ylabel("정답 (GT)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("행 비율 (%)", fontsize=9)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  혼동 행렬 저장 → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 예측 결과 수집
# ──────────────────────────────────────────────────────────────────────────────

def _collect_predictions(
    model: nn.Module,
    test_files: List[str],
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    각 .pt 파일에 대해 예측 수행 후 결과 dict 목록 반환.

    반환 dict 키
    ------------
    name        : 파일 스템
    step_path   : labeled STEP 경로 (없으면 None)
    gt_face     : GT face 라벨 리스트
    pred_face   : 예측 face 라벨 리스트
    gt_graph    : GT graph 라벨 (int)
    pred_graph  : 예측 graph 라벨 (int)
    correct     : graph 예측 정답 여부
    face_correct: face별 정답 여부 리스트 (bool)
    """
    rng = random.Random(seed)
    files = test_files[:]
    if n_samples is not None:
        files = rng.sample(files, min(n_samples, len(files)))

    model.eval()
    results = []

    for f in files:
        data = torch.load(f, weights_only=False)
        step_path = getattr(data, "step_path", None)

        with torch.no_grad():
            d = data.clone().to(DEVICE)
            if not hasattr(d, "batch") or d.batch is None:
                d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=DEVICE)
            face_logits, graph_logits = model(d)

        pred_face  = face_logits.argmax(1).cpu().tolist()
        pred_graph = int(graph_logits.argmax(1).cpu().item())
        gt_face    = data.face_y.tolist()
        gt_graph   = int(data.graph_y.item())

        face_correct = [
            (gf == pf) if gf >= 0 else None
            for gf, pf in zip(gt_face, pred_face)
        ]

        results.append({
            "name":         Path(f).stem,
            "step_path":    step_path,
            "gt_face":      gt_face,
            "pred_face":    pred_face,
            "gt_graph":     gt_graph,
            "pred_graph":   pred_graph,
            "correct":      (pred_graph == gt_graph),
            "face_correct": face_correct,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# OCC Viewer 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _make_occ_color(rgb):
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    return Quantity_Color(
        rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0,
        Quantity_TOC_RGB,
    )


def _render_faces_with_labels(disp, faces_topo, labels: List[int], label_names, color_table):
    """TopoDS_Face 리스트 + face 라벨 → OCC Viewer 렌더링."""
    disp.EraseAll()
    for face, lbl in zip(faces_topo, labels):
        if lbl < 0:
            name = "unknown"
        elif lbl < len(label_names):
            name = label_names[lbl]
        else:
            name = f"label_{lbl}"
        rgb   = color_table.get(name, [160, 160, 160])
        color = _make_occ_color(rgb)
        disp.DisplayShape(face, color=color, transparency=0.0)
    disp.FitAll()
    disp.Repaint()


def _render_face_accuracy(disp, faces_topo, face_correct: List[Optional[bool]]):
    """
    face별 정답/오답을 색으로 표시.

    정답  → limegreen  (50, 205, 50)
    오답  → crimson    (220, 50,  50)
    무시  → gray       (160, 160, 160)
    """
    disp.EraseAll()
    for face, ok in zip(faces_topo, face_correct):
        if ok is None:
            rgb = _COLOR_UNKNOWN
        elif ok:
            rgb = _COLOR_CORRECT
        else:
            rgb = _COLOR_WRONG
        disp.DisplayShape(face, color=_make_occ_color(rgb), transparency=0.0)
    disp.FitAll()
    disp.Repaint()


def _render_graph_accuracy(disp, faces_topo, graph_correct: bool):
    """
    graph 예측 결과를 전체 형상 단색으로 표시.

    정답  → limegreen  (50, 205, 50)
    오답  → crimson    (220, 50,  50)
    """
    disp.EraseAll()
    rgb   = _COLOR_CORRECT if graph_correct else _COLOR_WRONG
    color = _make_occ_color(rgb)
    for face in faces_topo:
        disp.DisplayShape(face, color=color, transparency=0.0)
    disp.FitAll()
    disp.Repaint()


def _load_faces_from_step(step_path: str):
    """STEP → TopoDS_Face 리스트 (face_adjacency 순서)."""
    from core.step_io import load_step
    from occwl.solid import Solid
    from occwl.graph import face_adjacency
    from OCC.Core.TopoDS import topods_Face

    shape = load_step(step_path)
    if shape is None or shape.IsNull():
        return None

    solid      = Solid(shape)
    adj        = face_adjacency(solid)
    nodes_data = list(adj.nodes.data())
    occwl_faces = [nodes_data[i][1]["face"] for i in range(len(adj.nodes))]

    topo_faces = []
    for f in occwl_faces:
        try:
            topo_faces.append(topods_Face(f.topods_shape()))
        except Exception:
            topo_faces.append(None)
    return topo_faces


# ──────────────────────────────────────────────────────────────────────────────
# OCC Viewer 시각화 (메인)
# ──────────────────────────────────────────────────────────────────────────────

def _visualize_with_occ(results: List[Dict], label_props_path: str = "config/LABEL_PROPS.json"):
    """
    OCC Viewer 메뉴:
      GT Labels       — GT face 라벨 색상
      Predicted Labels — 예측 face 라벨 색상
      Face Accuracy   — face별 정답(초록) / 오답(빨간) 표시
      Graph Accuracy  — graph 예측 정답(초록) / 오답(빨간) 단색 표시
    """
    from OCC.Display.SimpleGui import init_display
    from viz.label_viz import load_label_props

    label_names, color_table = load_label_props(label_props_path)

    viz_cases = [r for r in results if r["step_path"] and Path(r["step_path"]).exists()]
    if not viz_cases:
        print("\n[경고] 시각화할 수 있는 케이스가 없습니다.")
        print("  generate_dataset.py 실행 시 --no_step 옵션 없이 실행해야 합니다.")
        return

    print(f"\nSTEP 파일 로드 중 ({len(viz_cases)}개)...")
    for case in viz_cases:
        case["faces_topo"] = _load_faces_from_step(case["step_path"])
        status = "OK" if case["faces_topo"] else "FAIL"
        mark   = "O" if case["correct"] else "X"
        print(f"  [{mark}] {case['name']}  "
              f"GT={GRAPH_LABEL_NAMES.get(case['gt_graph'], '?')}  "
              f"Pred={GRAPH_LABEL_NAMES.get(case['pred_graph'], '?')}  [{status}]")

    disp, start_display, add_menu, add_function_to_menu = init_display()
    disp.set_bg_gradient_color([255, 255, 255], [255, 255, 255])

    add_menu("GT Labels")
    add_menu("Predicted Labels")
    add_menu("Face Accuracy")
    add_menu("Graph Accuracy")

    for case in viz_cases:
        if case["faces_topo"] is None:
            continue

        faces      = [f for f in case["faces_topo"] if f is not None]
        n          = len(faces)
        gt_lbl     = case["gt_face"][:n]
        pred_lbl   = case["pred_face"][:n]
        face_corr  = case["face_correct"][:n]
        graph_corr = case["correct"]
        name       = case["name"]

        # ── GT Labels ──────────────────────────────────────────────
        def _make_gt(f=faces, gl=gt_lbl, c=case):
            def cb():
                _render_faces_with_labels(disp, f, gl, label_names, color_table)
                g_name = GRAPH_LABEL_NAMES.get(c["gt_graph"], "?")
                print(f"[GT]   {c['name']}  graph_label={g_name}")
            return cb

        # ── Predicted Labels ────────────────────────────────────────
        def _make_pred(f=faces, pl=pred_lbl, c=case):
            def cb():
                _render_faces_with_labels(disp, f, pl, label_names, color_table)
                gt_n = GRAPH_LABEL_NAMES.get(c["gt_graph"],   "?")
                pr_n = GRAPH_LABEL_NAMES.get(c["pred_graph"],  "?")
                mark = "O" if c["correct"] else "X"
                print(f"[Pred] {c['name']}  GT={gt_n}  Pred={pr_n}  [{mark}]")
            return cb

        # ── Face Accuracy ───────────────────────────────────────────
        def _make_face_acc(f=faces, fc=face_corr, c=case):
            def cb():
                _render_face_accuracy(disp, f, fc)
                n_ok  = sum(1 for x in fc if x is True)
                n_tot = sum(1 for x in fc if x is not None)
                pct   = n_ok / max(n_tot, 1) * 100
                print(f"[FaceAcc] {c['name']}  {n_ok}/{n_tot} ({pct:.1f}%)"
                      f"  Green=correct  Red=wrong")
            return cb

        # ── Graph Accuracy ──────────────────────────────────────────
        def _make_graph_acc(f=faces, gc=graph_corr, c=case):
            def cb():
                _render_graph_accuracy(disp, f, gc)
                gt_n = GRAPH_LABEL_NAMES.get(c["gt_graph"],   "?")
                pr_n = GRAPH_LABEL_NAMES.get(c["pred_graph"],  "?")
                mark = "O" if gc else "X"
                print(f"[GraphAcc] {c['name']}  GT={gt_n}  Pred={pr_n}  [{mark}]"
                      f"  Green=correct  Red=wrong")
            return cb

        add_function_to_menu("GT Labels",        _make_gt())
        add_function_to_menu("Predicted Labels", _make_pred())
        add_function_to_menu("Face Accuracy",    _make_face_acc())
        add_function_to_menu("Graph Accuracy",   _make_graph_acc())

    # 첫 번째 케이스 자동 표시 (GT)
    first = next((c for c in viz_cases if c.get("faces_topo")), None)
    if first:
        faces = [f for f in first["faces_topo"] if f is not None]
        _render_faces_with_labels(disp, faces, first["gt_face"][:len(faces)],
                                  label_names, color_table)

    start_display()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[오류] 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        return

    # ── 데이터 (학습과 동일한 split) ─────────────────────────────────────
    _, _, test_ds = split_dataset(
        args.data_dir, args.train_ratio, args.val_ratio, args.seed
    )

    if len(test_ds) == 0:
        print("[경고] 테스트 데이터가 없습니다.")
        return

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── 모델 복원 ──────────────────────────────────────────────────────
    ckpt       = torch.load(str(ckpt_path), weights_only=False)
    saved_args = ckpt.get("args", {})

    model = UVNetMultiTask(
        gnn_type    = saved_args.get("gnn",    "gcn"),
        hidden_size = saved_args.get("hidden", 64),
        dropout     = 0.0,
        num_uv      = saved_args.get("num_uv", 10),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    face_loss_fn  = nn.CrossEntropyLoss(ignore_index=-1)
    graph_loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"UVNet Multi-task 테스트")
    print(f"  체크포인트 : {ckpt_path}")
    print(f"  학습 epoch : {ckpt.get('epoch', '?')}")
    print(f"  val loss   : {ckpt.get('val_loss', float('inf')):.4f}")
    print(f"  테스트 샘플: {len(test_ds)}")
    print(f"  device     : {DEVICE}")
    print(f"{'='*60}")

    # ── 수치 평가 ──────────────────────────────────────────────────────
    ts = evaluate_detail(
        model, test_loader, face_loss_fn, graph_loss_fn,
        args.alpha, args.beta,
    )

    print(f"\n{'='*60}")
    print(f"  [전체 결과]")
    print(f"  Loss       : {ts['loss']:.4f}")
    print(f"  Face  Acc  : {ts['face_acc']:.4f}")
    print(f"  Graph Acc  : {ts['graph_acc']:.4f}")
    print(f"{'='*60}")

    # Face label per-class accuracy
    if "face_per_class" in ts:
        print(f"\n  [Face 클래스별 정확도]")
        for i, name in enumerate(FACE_LABEL_NAMES):
            acc = ts["face_per_class"].get(i, float("nan"))
            print(f"    {name:<20s}: {acc:.4f}")

    # Graph label per-class accuracy
    if "graph_per_class" in ts:
        print(f"\n  [Graph 클래스별 정확도]")
        for i, name in GRAPH_LABEL_NAMES.items():
            acc = ts["graph_per_class"].get(i, float("nan"))
            print(f"    {name:<20s}: {acc:.4f}")

    # Graph confusion matrix (텍스트)
    if "graph_gt" in ts and "graph_pred" in ts:
        gnames = [GRAPH_LABEL_NAMES[i] for i in sorted(GRAPH_LABEL_NAMES)]
        print(f"\n  [Graph Confusion Matrix]  (행=GT, 열=예측)")
        _print_confusion_matrix(ts["graph_gt"], ts["graph_pred"], gnames)

    # ── 혼동 행렬 이미지 저장 ──────────────────────────────────────────
    if args.save_cm:
        results_dir = args.results_dir
        print(f"\n  [혼동 행렬 이미지 저장]  →  {results_dir}/")

        if "face_gt" in ts and "face_pred" in ts:
            _save_confusion_matrix_image(
                ts["face_gt"], ts["face_pred"],
                FACE_LABEL_NAMES,
                "Face Label Confusion Matrix",
                str(Path(results_dir) / "confusion_matrix_face.png"),
            )

        if "graph_gt" in ts and "graph_pred" in ts:
            gnames = [GRAPH_LABEL_NAMES[i] for i in sorted(GRAPH_LABEL_NAMES)]
            _save_confusion_matrix_image(
                ts["graph_gt"], ts["graph_pred"],
                gnames,
                "Graph Label Confusion Matrix",
                str(Path(results_dir) / "confusion_matrix_graph.png"),
            )

    # ── OCC Viewer 시각화 ───────────────────────────────────────────────
    if args.viz:
        test_files = [str(f) for f in test_ds.files]
        results    = _collect_predictions(model, test_files, n_samples=args.n_viz,
                                          seed=args.seed)
        _visualize_with_occ(results)


if __name__ == "__main__":
    main()
