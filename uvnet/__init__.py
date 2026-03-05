"""
uvnet 패키지

UVNet 기반 Multi-task 학습 모듈.

구성 요소
---------
encoder    : SurfaceEncoder, CurveEncoder
gnn        : GCN, SAGE, GAT
model      : UVNetMultiTask (face segmentation + graph classification)
dataset    : GraphDataset
preprocess : 형상 생성 + .pt 전처리 통합 스크립트
train      : 학습/검증/테스트 메인 스크립트

사용 흐름
---------
1. 전처리:
       python -m uvnet.preprocess --n_nodes 6 --max_depth 3 --save_dir data/graphs

2. 학습:
       python -m uvnet.train --data_dir data/graphs --epochs 100 --gnn gcn
"""

from uvnet.encoder import SurfaceEncoder, CurveEncoder
from uvnet.gnn     import GCN, SAGE, GAT, GNN_REGISTRY
from uvnet.model   import UVNetMultiTask, NUM_FACE_CLASSES, NUM_GRAPH_CLASSES
from uvnet.dataset import GraphDataset

__all__ = [
    "SurfaceEncoder",
    "CurveEncoder",
    "GCN",
    "SAGE",
    "GAT",
    "GNN_REGISTRY",
    "UVNetMultiTask",
    "NUM_FACE_CLASSES",
    "NUM_GRAPH_CLASSES",
    "GraphDataset",
]
