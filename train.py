"""
UVNet Multi-task 학습 스크립트  (train / val)

하이퍼파라미터는 config/train_config.json 에서 읽으며,
CLI 인자로 개별 값을 덮어쓸 수 있습니다.

실행 예시
---------
    # config 기본값으로 실행
    python train.py

    # 일부 값만 override
    python train.py --gnn gat --epochs 150 --dropout 0.1

    # 다른 config 파일 사용
    python train.py --config config/my_config.json
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from uvnet.model import UVNetMultiTask
from uvnet.train import split_dataset, run_epoch, evaluate_detail

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CONFIG = "config/train_config.json"


# ──────────────────────────────────────────────────────────────────────────────
# Config + argparse
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    # 1단계: config 파일 경로만 먼저 파악
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()

    # 2단계: config 파일 로드
    cfg = {}
    if Path(pre_args.config).exists():
        with open(pre_args.config, encoding="utf-8") as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    else:
        print(f"[경고] config 파일 없음: {pre_args.config}  —  기본값 사용")

    # 3단계: 전체 파서 (config를 기본값으로)
    p = argparse.ArgumentParser(description="UVNet Multi-task 학습")
    p.add_argument("--config",      default=DEFAULT_CONFIG)
    p.add_argument("--data_dir",    default=cfg.get("data_dir",    "data/graphs"))
    p.add_argument("--log_dir",     default=cfg.get("log_dir",     "data/logs"))
    p.add_argument("--ckpt_dir",    default=cfg.get("ckpt_dir",    "data/checkpoints"))
    p.add_argument("--gnn",         default=cfg.get("gnn",         "gcn"),
                   choices=["gcn", "sage", "gat"])
    p.add_argument("--hidden",      type=int,   default=cfg.get("hidden",     64))
    p.add_argument("--num_uv",      type=int,   default=cfg.get("num_uv",     10))
    p.add_argument("--dropout",     type=float, default=cfg.get("dropout",    0.0))
    p.add_argument("--epochs",      type=int,   default=cfg.get("epochs",     100))
    p.add_argument("--batch_size",  type=int,   default=cfg.get("batch_size", 8))
    p.add_argument("--lr",          type=float, default=cfg.get("lr",         1e-3))
    p.add_argument("--alpha",       type=float, default=cfg.get("alpha",      1.0),
                   help="face segmentation loss 가중치")
    p.add_argument("--beta",        type=float, default=cfg.get("beta",       1.0),
                   help="graph classification loss 가중치")
    p.add_argument("--train_ratio", type=float, default=cfg.get("train_ratio", 0.8))
    p.add_argument("--val_ratio",   type=float, default=cfg.get("val_ratio",   0.1))
    p.add_argument("--seed",        type=int,   default=cfg.get("seed",        42))
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 데이터 ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = split_dataset(
        args.data_dir, args.train_ratio, args.val_ratio, args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = UVNetMultiTask(
        gnn_type    = args.gnn,
        hidden_size = args.hidden,
        dropout     = args.dropout,
        num_uv      = args.num_uv,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )
    face_loss_fn  = nn.CrossEntropyLoss(ignore_index=-1)
    graph_loss_fn = nn.CrossEntropyLoss()

    # ── 로깅 / 저장 경로 ─────────────────────────────────────────────────
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.gnn}_h{args.hidden}"
    run_log_dir  = str(Path(args.log_dir)  / run_name)
    run_ckpt_dir = str(Path(args.ckpt_dir) / run_name)
    Path(run_log_dir).mkdir(parents=True, exist_ok=True)
    Path(run_ckpt_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_log_dir)

    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"\n{'='*60}")
    print(f"UVNet Multi-task 학습")
    print(f"  run       : {run_name}")
    print(f"  config    : {args.config}")
    print(f"  GNN       : {args.gnn}  hidden={args.hidden}  dropout={args.dropout}")
    print(f"  epochs    : {args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  loss w    : face(α)={args.alpha}  graph(β)={args.beta}")
    print(f"  device    : {DEVICE}")
    print(f"{'='*60}\n")

    # ── 학습 루프 ───────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, face_loss_fn, graph_loss_fn,
            args.alpha, args.beta, optimizer=optimizer,
        )
        vl = run_epoch(
            model, val_loader, face_loss_fn, graph_loss_fn,
            args.alpha, args.beta,
        )

        scheduler.step(vl["loss"])

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        for tag, val in tr.items():
            writer.add_scalar(f"train/{tag}", val, epoch)
        for tag, val in vl.items():
            writer.add_scalar(f"val/{tag}", val, epoch)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {tr['loss']:.4f}/{vl['loss']:.4f} | "
            f"FaceAcc {tr['face_acc']:.3f}/{vl['face_acc']:.3f} | "
            f"GraphAcc {tr['graph_acc']:.3f}/{vl['graph_acc']:.3f}"
        )

        # best 체크포인트 저장
        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            best_epoch    = epoch
            torch.save(
                {
                    "epoch"          : epoch,
                    "model_state"    : model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss"       : best_val_loss,
                    "args"           : vars(args),
                },
                str(Path(run_ckpt_dir) / "best_model.pt"),
            )

    # ── 테스트 평가 (best model) ─────────────────────────────────────────
    if len(test_ds) > 0:
        best_ckpt = torch.load(
            str(Path(run_ckpt_dir) / "best_model.pt"), weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state"])

        ts = evaluate_detail(
            model, test_loader, face_loss_fn, graph_loss_fn,
            args.alpha, args.beta,
        )
        for tag in ("loss", "face_loss", "graph_loss", "face_acc", "graph_acc"):
            writer.add_scalar(f"test/{tag}", ts[tag], best_epoch)

        print(
            f"\n  [Test 결과]  "
            f"Loss={ts['loss']:.4f}  "
            f"FaceAcc={ts['face_acc']:.4f}  "
            f"GraphAcc={ts['graph_acc']:.4f}"
        )

    writer.close()
    print(
        f"\n학습 완료 — best epoch: {best_epoch}, "
        f"best val loss: {best_val_loss:.4f}"
    )
    print(f"체크포인트 저장: {run_ckpt_dir}/best_model.pt")
    print(f"TensorBoard 로그: {run_log_dir}")


if __name__ == "__main__":
    main()
