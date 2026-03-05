"""
형상 생성 + UV 그래프 전처리 통합 스크립트

TurningMillingGenerator 로 형상 생성 → face 라벨 추출 → PyG Data 변환 → .pt 저장.

실행 방법:
    # 트리 자동 생성 후 전처리
    python -m uvnet.preprocess --n_nodes 6 --max_depth 3 --save_dir data/graphs

    # 기존 트리 파일 사용
    python -m uvnet.preprocess --trees results/trees/trees_N6_H3.json --save_dir data/graphs
"""

import argparse
import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Extend.TopologyUtils import TopologyExplorer
from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.uvgrid import uvgrid, ugrid

from pipeline import TurningMillingGenerator, TurningMillingParams
from core import TurningParams, MillingParams
from core.tree.io import get_tree_stats, load_trees
from core.tree.generator import generate_trees
from core.step_io import save_labeled_step


# ──────────────────────────────────────────────────────────────────────────────
# Graph label 상수
# ──────────────────────────────────────────────────────────────────────────────

GRAPH_LABEL_TURNING         = 0   # 터닝만 있는 형상
GRAPH_LABEL_MILLING         = 1   # 밀링만 있는 형상 (현재 파이프라인에서 미생성)
GRAPH_LABEL_MILLING_TURNING = 2   # 밀링+터닝 복합 형상


# ──────────────────────────────────────────────────────────────────────────────
# occwl Face → TopoDS_Face 변환 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _occwl_to_topods(occwl_face) -> Optional[TopoDS_Face]:
    """
    occwl.Face 객체에서 OCC TopoDS_Face를 추출.

    occwl.Shape base class는 topods_shape() 메서드 또는 _shape 속성으로
    내부 TopoDS_Shape를 노출합니다.  topods_Face()로 다운캐스팅하여 반환합니다.
    """
    from OCC.Core.TopoDS import topods_Face

    # 방법 1: topods_shape() 메서드 (occwl 표준 API)
    if hasattr(occwl_face, "topods_shape"):
        try:
            raw = occwl_face.topods_shape()
            return topods_Face(raw)
        except Exception:
            pass

    # 방법 2: _shape private attribute
    for attr in ("_shape", "_face", "face", "shape"):
        raw = getattr(occwl_face, attr, None)
        if raw is None:
            continue
        try:
            return topods_Face(raw)
        except Exception:
            if isinstance(raw, TopoDS_Face):
                return raw

    return None


def _lookup_face_label(
    occwl_face,
    labeled_faces: Dict[TopoDS_Face, int],
) -> int:
    """
    occwl.Face 에 대응하는 특징형상 라벨 반환.
    찾지 못하면 -1 (CrossEntropyLoss 의 ignore_index 로 처리).
    """
    tf = _occwl_to_topods(occwl_face)
    if tf is None:
        return -1

    # 1차: dict direct lookup
    label = labeled_faces.get(tf)
    if label is not None:
        return label

    # 2차: IsSame 비교 (hash 충돌 방어)
    for lf, lv in labeled_faces.items():
        if lf.IsSame(tf):
            return lv

    return -1


# ──────────────────────────────────────────────────────────────────────────────
# UV feature 추출 (UVNetGraphClassification 과 동일 로직 재구현)
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_center_scale(
    min_pt, max_pt, eps: float = 1e-8
) -> Tuple[np.ndarray, float]:
    mn = np.asarray(min_pt, dtype=float)
    mx = np.asarray(max_pt, dtype=float)
    center = 0.5 * (mn + mx)
    scale = float(np.max(mx - mn)) + eps
    return center, scale


def _normalize(pts: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    orig = pts.shape
    p = pts.reshape(-1, 3)
    p = np.clip((p - center) * (2.0 / scale), -1.0, 1.0)
    return p.reshape(orig)


def _extract_surface_features(
    min_pt, max_pt, faces: list, num_uv: int
) -> np.ndarray:
    """Returns (N, num_uv, num_uv, 7)"""
    center, scale = _bbox_center_scale(min_pt, max_pt)
    feats = []
    zero_feat = np.zeros((num_uv, num_uv, 7), dtype=np.float32)

    for face in faces:
        try:
            pts = _normalize(
                np.asarray(uvgrid(face, num_uv, num_uv, method="point"), dtype=float),
                center, scale,
            )  # (num_uv, num_uv, 3)
            nrm = np.asarray(uvgrid(face, num_uv, num_uv, method="normal"), dtype=float)
            nrm = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-6)
            # (num_uv, num_uv, 3)

            vis = np.asarray(
                uvgrid(face, num_uv, num_uv, method="visibility_status"), dtype=float
            )
            # occwl 버전에 따라 (num_uv, num_uv) 또는 (num_uv, num_uv, 1) 반환
            vis = vis.reshape(num_uv, num_uv)  # 항상 2-D 로 통일
            mask = np.logical_or(vis == 0, vis == 2).astype(np.float32)[..., np.newaxis]
            # (num_uv, num_uv, 1)

            feats.append(np.concatenate([pts, nrm, mask], axis=-1))
        except Exception:
            feats.append(zero_feat)

    return np.array(feats, dtype=np.float32)


def _extract_curve_features(
    min_pt, max_pt, edges: list, num_uv: int
) -> np.ndarray:
    """Returns (E, num_uv, 6)"""
    center, scale = _bbox_center_scale(min_pt, max_pt)
    feats = []
    zero_feat = np.zeros((num_uv, 6), dtype=np.float32)

    for edge in edges:
        try:
            pts = _normalize(
                np.asarray(ugrid(edge, num_uv, method="point"), dtype=float),
                center, scale,
            )
            tan = np.asarray(ugrid(edge, num_uv, method="tangent"), dtype=float)
            tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6)
            feats.append(np.concatenate([pts, tan], axis=-1))
        except Exception:
            feats.append(zero_feat)

    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Shape → PyG Data
# ──────────────────────────────────────────────────────────────────────────────

def shape_to_data(
    shape,
    labeled_faces: Dict[TopoDS_Face, int],
    n_milling: int,
    num_uv: int = 10,
) -> Optional[Data]:
    """
    OCC TopoDS_Shape + face-label dict → PyG Data.

    Data fields
    -----------
    x          : (N, 7, uv, uv)   surface features
    edge_index : (2, E)
    edge_attr  : (E, 6, uv)       curve features
    face_y     : (N,)             face-level label (long, -1 = unknown)
    graph_y    : ()               graph-level label (long)
    """
    # TopoDS_Shape → occwl.Solid 목록
    # Boolean Cut 결과가 COMPOUND로 나올 수 있으므로 모든 SOLID를 열거
    if shape.ShapeType() == TopAbs_SOLID:
        solid_list = [Solid(shape)]
    else:
        te = TopologyExplorer(shape)
        solid_list = [Solid(s) for s in te.solids()]

    if not solid_list:
        return None

    # 각 solid의 face adjacency를 합산 (인덱스 offset 처리)
    occwl_faces: list = []
    occwl_edges: list = []
    raw_edges:   list = []   # (u, v) 정수 쌍

    for solid in solid_list:
        adj = face_adjacency(solid)
        if len(adj.nodes) == 0:
            continue

        offset      = len(occwl_faces)
        nodes_data  = list(adj.nodes.data())   # [(node_id, attr_dict), ...]
        edges_data  = list(adj.edges.data())   # [(u, v, attr_dict), ...]

        occwl_faces.extend(nodes_data[i][1]["face"] for i in range(len(adj.nodes)))
        occwl_edges.extend(
            edges_data[i][2]["edge"]
            for i in range(len(adj.edges))
            if "edge" in edges_data[i][2]
        )
        raw_edges.extend((u + offset, v + offset) for u, v in adj.edges())

    if not occwl_faces:
        return None

    # bounding box (첫 번째 solid 기준)
    min_pt = solid_list[0].box().min_point()
    max_pt = solid_list[0].box().max_point()

    # --- UV feature 추출 ---
    node_feat = _extract_surface_features(min_pt, max_pt, occwl_faces, num_uv)
    node_feat = np.transpose(node_feat, (0, 3, 1, 2))  # (N, 7, uv, uv)

    if occwl_edges:
        edge_feat = _extract_curve_features(min_pt, max_pt, occwl_edges, num_uv)
        edge_feat = np.transpose(edge_feat, (0, 2, 1))  # (E, 6, uv)
    else:
        edge_feat = np.zeros((0, 6, num_uv), dtype=np.float32)

    # edge_index
    if raw_edges:
        edge_index = torch.tensor(raw_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # --- face labels ---
    face_labels = [_lookup_face_label(f, labeled_faces) for f in occwl_faces]

    # --- graph label ---
    graph_label = GRAPH_LABEL_MILLING_TURNING if n_milling > 0 else GRAPH_LABEL_TURNING

    return Data(
        x          = torch.as_tensor(node_feat, dtype=torch.float32),
        edge_index = edge_index,
        edge_attr  = torch.as_tensor(edge_feat, dtype=torch.float32),
        face_y     = torch.tensor(face_labels, dtype=torch.long),
        graph_y    = torch.tensor(graph_label, dtype=torch.long),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 기본 생성 파라미터
# ──────────────────────────────────────────────────────────────────────────────

def default_params() -> TurningMillingParams:
    return TurningMillingParams(
        turning=TurningParams(
            stock_height_margin=(3.0, 8.0),
            stock_radius_margin=(2.0, 5.0),
            step_depth_range=(0.8, 1.5),
            step_height_range=(2.0, 4.0),
            step_margin=0.5,
            groove_depth_range=(0.4, 0.8),
            groove_width_range=(1.5, 3.0),
            groove_margin=0.5,
            chamfer_range=(0.3, 0.8),
            fillet_range=(0.3, 0.8),
            edge_feature_prob=0.3,
        ),
        milling=MillingParams(
            diameter_min=1.0,
            diameter_max_ratio=0.85,
            clearance=0.15,
            depth_ratio=2.0,
            min_spacing=1.0,
            max_features_per_face=3,
            rect_aspect_min=0.4,
            rect_aspect_max=2.5,
        ),
        enable_milling=True,
        target_face_types=["Cylinder", "Cone"],
        max_features=5,
        features_per_face=1,
        enable_labeling=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 메인 전처리 함수
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    trees: List[Dict],
    params: TurningMillingParams,
    save_dir: str,
    num_uv: int = 10,
    seed: int = 42,
    skip_existing: bool = True,
    start_idx: int = 0,
    desc: str = "Preprocessing",
    save_step_dir: Optional[str] = None,
) -> List[str]:
    """
    트리 목록에서 형상 생성 후 .pt 파일로 저장.

    Args:
        trees         : 트리 딕셔너리 리스트
        params        : TurningMillingParams (enable_labeling=True 필수)
        save_dir      : .pt 파일 저장 디렉토리
        num_uv        : UV grid 해상도
        seed          : 랜덤 시드
        skip_existing : 이미 .pt 파일이 있으면 건너뜀
        start_idx     : 파일명 인덱스 시작값 (여러 배치를 이어서 생성할 때 사용)
        desc          : tqdm 진행바 설명
        save_step_dir : labeled STEP 파일 저장 경로 (None이면 저장 안 함)

    Returns:
        저장된 .pt 파일 경로 리스트
    """
    assert params.enable_labeling, "enable_labeling=True 로 설정하세요."

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    step_path_dir: Optional[Path] = None
    if save_step_dir:
        step_path_dir = Path(save_step_dir)
        step_path_dir.mkdir(parents=True, exist_ok=True)

    saved, skipped, failed = [], 0, 0

    for i, tree in enumerate(tqdm(trees, desc=desc)):
        stats      = get_tree_stats(tree)
        global_idx = start_idx + i
        stem       = f"graph_{global_idx:05d}_S{stats['s_count']}_G{stats['g_count']}"
        fpath      = save_path / f"{stem}.pt"

        if skip_existing and fpath.exists():
            saved.append(str(fpath))
            skipped += 1
            continue

        random.seed(seed + i)
        generator = TurningMillingGenerator(params)

        try:
            shape, milling_reqs = generator.generate_from_tree(
                tree, apply_edge_feats=True
            )
        except Exception as e:
            print(f"  [{i:04d}] 생성 오류: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
            continue

        if shape is None or shape.IsNull():
            print(f"  [{i:04d}] 형상 생성 실패")
            failed += 1
            continue

        lm = generator.label_maker
        if lm is None or len(lm.labeled_faces) == 0:
            print(f"  [{i:04d}] 라벨 정보 없음")
            failed += 1
            continue

        try:
            data = shape_to_data(shape, lm.labeled_faces, len(milling_reqs), num_uv)
        except Exception as e:
            print(f"  [{i:04d}] 그래프 변환 오류: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
            continue

        if data is None:
            print(f"  [{i:04d}] 노드가 없는 그래프 — 스킵")
            failed += 1
            continue

        # labeled STEP 저장 (시각화 용도)
        if step_path_dir is not None:
            step_out = step_path_dir / f"{stem}.step"
            try:
                save_labeled_step(shape, lm.labeled_faces, str(step_out))
                data.step_path = str(step_out)
            except Exception as e:
                print(f"  [{i:04d}] STEP 저장 오류: {e}")

        torch.save(data, str(fpath))
        saved.append(str(fpath))

    print(
        f"\n전처리 완료: 저장={len(saved) - skipped}, "
        f"스킵={skipped}, 실패={failed} / 총={len(trees)}"
    )
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="형상 생성 + UV 그래프 전처리")
    parser.add_argument("--trees",     type=str, default=None,
                        help="기존 트리 JSON 파일 경로 (없으면 새로 생성)")
    parser.add_argument("--n_nodes",   type=int, default=6,
                        help="트리 노드 수 (trees 미지정 시 사용)")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="트리 최대 깊이 (trees 미지정 시 사용)")
    parser.add_argument("--save_dir",  type=str, default="data/graphs",
                        help=".pt 파일 저장 경로")
    parser.add_argument("--num_uv",    type=int, default=10,
                        help="UV grid 해상도")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no_skip",   action="store_true",
                        help="이미 생성된 .pt도 덮어씀")
    args = parser.parse_args()

    # 트리 로드 또는 생성
    if args.trees:
        trees = load_trees(args.trees)
        print(f"트리 로드: {args.trees} ({len(trees)}개)")
    else:
        trees = generate_trees(args.n_nodes, args.max_depth)
        print(f"트리 생성: N={args.n_nodes}, depth={args.max_depth} ({len(trees)}개)")

    params = default_params()

    preprocess_dataset(
        trees        = trees,
        params       = params,
        save_dir     = args.save_dir,
        num_uv       = args.num_uv,
        seed         = args.seed,
        skip_existing= not args.no_skip,
    )
