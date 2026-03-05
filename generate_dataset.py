"""
학습 데이터셋 생성 스크립트

원하는 수만큼 터닝·밀링 복합 형상을 합성하고 UV 그래프(.pt)로 변환합니다.
run_pipeline.py 없이 단독으로 실행할 수 있습니다.

흐름
----
  트리 열거/로드
    → turning-only / milling+turning 각각 트리 샘플링
    → 형상 합성 (TurningMillingGenerator, 라벨 포함)
    → UV feature 추출 + face/graph 라벨 부여
    → PyG Data(.pt) 저장 → data/graphs/

Graph label
-----------
  0 = turning          (enable_milling=False 로 생성)
  2 = milling+turning  (enable_milling=True  로 생성)

실행 예시
---------
    # 기본: 100개 생성 (turning 40% / milling+turning 60%)
    python generate_dataset.py

    # 총 200개, turning 30% 비율
    python generate_dataset.py --n_graphs 200 --milling_ratio 0.7

    # 기존 트리 파일 사용
    python generate_dataset.py --trees results/trees/trees_N6_H3.json --n_graphs 150

    # 이미 생성된 데이터셋 통계만 확인
    python generate_dataset.py --summary
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict

from core import generate_trees
from core.tree.io import load_trees
from uvnet.preprocess import preprocess_dataset, default_params
from uvnet.dataset import GraphDataset

DEFAULT_SAVE_DIR  = "data/graphs"
DEFAULT_STEP_DIR  = "data/labeled_steps"
DEFAULT_N_GRAPHS  = 100
DEFAULT_MILLING_R = 0.6   # milling+turning 비율


# ──────────────────────────────────────────────────────────────────────────────
# 트리 샘플링
# ──────────────────────────────────────────────────────────────────────────────

def _sample_trees(all_trees: List[Dict], n: int, seed: int) -> List[Dict]:
    """
    all_trees 에서 n개를 랜덤 샘플링합니다.
    n > len(all_trees) 이면 반복 샘플링을 허용합니다.
    """
    rng = random.Random(seed)
    if n <= len(all_trees):
        return rng.sample(all_trees, n)
    # 반복 샘플링
    result: List[Dict] = []
    while len(result) < n:
        batch = rng.sample(all_trees, min(len(all_trees), n - len(result)))
        result.extend(batch)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="터닝·밀링 학습 데이터셋 생성")

    # 트리 소스
    p.add_argument("--trees",          type=str,   default=None,
                   help="기존 트리 JSON 경로 (없으면 자동 생성)")
    p.add_argument("--n_nodes",        type=int,   default=6,
                   help="트리 노드 수 (--trees 미지정 시 사용)")
    p.add_argument("--max_depth",      type=int,   default=3,
                   help="트리 최대 깊이 (--trees 미지정 시 사용)")

    # 생성 수량 / 비율
    p.add_argument("--n_graphs",       type=int,   default=DEFAULT_N_GRAPHS,
                   help=f"총 생성할 그래프 수 (기본 {DEFAULT_N_GRAPHS})")
    p.add_argument("--milling_ratio",  type=float, default=DEFAULT_MILLING_R,
                   help=f"milling+turning 비율 0~1 (기본 {DEFAULT_MILLING_R})")

    # 출력
    p.add_argument("--save_dir",       type=str,   default=DEFAULT_SAVE_DIR)
    p.add_argument("--step_dir",       type=str,   default=DEFAULT_STEP_DIR,
                   help="labeled STEP 파일 저장 경로 (test.py 시각화용)")
    p.add_argument("--no_step",        action="store_true",
                   help="labeled STEP 파일을 저장하지 않음")
    p.add_argument("--num_uv",         type=int,   default=10,
                   help="UV grid 해상도 (train.py 의 num_uv 와 동일해야 함)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_skip",        action="store_true",
                   help="기존 .pt 파일도 덮어씀")

    # 통계만 출력
    p.add_argument("--summary",        action="store_true",
                   help="이미 생성된 데이터셋 통계만 출력하고 종료")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 통계 모드 ─────────────────────────────────────────────────────────
    if args.summary:
        pt_files = sorted(str(p) for p in Path(args.save_dir).glob("*.pt"))
        if not pt_files:
            print(f"[경고] .pt 파일이 없습니다: {args.save_dir}")
            print(f"  먼저 python generate_dataset.py 를 실행하세요.")
            return
        GraphDataset(pt_files).print_statistics(
            title=f"데이터셋 통계  ({args.save_dir})"
        )
        return

    # ── 트리 열거 ─────────────────────────────────────────────────────────
    if args.trees:
        all_trees = load_trees(args.trees)
        print(f"트리 로드: {args.trees}  ({len(all_trees)}개)")
    else:
        all_trees = generate_trees(args.n_nodes, args.max_depth)
        print(f"트리 열거: N={args.n_nodes}, depth={args.max_depth}  ({len(all_trees)}개)")

    # ── 수량 계산 ─────────────────────────────────────────────────────────
    n_total   = args.n_graphs
    n_milling = round(n_total * args.milling_ratio)
    n_turning = n_total - n_milling

    print(f"\n생성 계획")
    print(f"  총 그래프        : {n_total}개")
    print(f"  turning-only     : {n_turning}개  ({n_turning/n_total*100:.1f}%)")
    print(f"  milling+turning  : {n_milling}개  ({n_milling/n_total*100:.1f}%)")
    print(f"  저장 경로        : {Path(args.save_dir).resolve()}")

    # ── 파라미터 준비 ─────────────────────────────────────────────────────
    params_turning = default_params()
    params_turning.enable_milling = False

    params_milling = default_params()
    params_milling.enable_milling = True

    skip     = not args.no_skip
    step_dir = None if args.no_step else args.step_dir

    if step_dir:
        print(f"  labeled STEP     : {Path(step_dir).resolve()}")

    # ── turning-only 생성 ─────────────────────────────────────────────────
    if n_turning > 0:
        trees_turn = _sample_trees(all_trees, n_turning, seed=args.seed)
        print(f"\n[1/2] turning-only  ({n_turning}개)")
        saved_turn = preprocess_dataset(
            trees         = trees_turn,
            params        = params_turning,
            save_dir      = args.save_dir,
            num_uv        = args.num_uv,
            seed          = args.seed,
            skip_existing = skip,
            start_idx     = 0,
            desc          = "turning-only",
            save_step_dir = step_dir,
        )
    else:
        saved_turn = []

    # ── milling+turning 생성 ──────────────────────────────────────────────
    if n_milling > 0:
        trees_mill = _sample_trees(all_trees, n_milling, seed=args.seed + 1000)
        print(f"\n[2/2] milling+turning  ({n_milling}개)")
        saved_mill = preprocess_dataset(
            trees         = trees_mill,
            params        = params_milling,
            save_dir      = args.save_dir,
            num_uv        = args.num_uv,
            seed          = args.seed + 1000,
            skip_existing = skip,
            start_idx     = n_turning,
            desc          = "milling+turning",
            save_step_dir = step_dir,
        )
    else:
        saved_mill = []

    total_saved = len(saved_turn) + len(saved_mill)

    # ── 생성 후 통계 ──────────────────────────────────────────────────────
    pt_files = sorted(str(p) for p in Path(args.save_dir).glob("*.pt"))
    if pt_files:
        GraphDataset(pt_files).print_statistics(
            title=f"생성 완료  ({args.save_dir})"
        )

    print(f"이번 실행에서 저장된 파일: {total_saved}개")
    print(f"\n다음 단계:")
    print(f"  학습  : python train.py")
    print(f"  테스트: python test.py")


if __name__ == "__main__":
    main()
