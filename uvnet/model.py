"""
UVNet Multi-task 모델

두 태스크를 동시에 학습:
1. Face segmentation  : 각 face의 특징형상 예측 (NUM_FACE_CLASSES = 9)
2. Graph classification: 전체 형상 종류 예측  (NUM_GRAPH_CLASSES = 3)

아키텍처:
  x: (N, 7, uv, uv)   → SurfaceEncoder → (N, 64)
  edge_attr: (E, 6, uv)→ CurveEncoder   → (E, 64)
  GNN → node_feat: (N, 64)
  face_head(node_feat) → face_logits: (N, 9)
  global_mean_pool → graph_feat: (B, 64)
  graph_head(graph_feat) → graph_logits: (B, 3)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from uvnet.encoder import SurfaceEncoder, CurveEncoder
from uvnet.gnn import GNN_REGISTRY

NUM_FACE_CLASSES  = 9   # stock, step, groove, chamfer, fillet,
                         # blind_hole, through_hole, rect_pocket, rect_passage
NUM_GRAPH_CLASSES = 3   # 0=turning, 1=milling, 2=milling+turning

FACE_LABEL_NAMES = [
    "stock", "step", "groove", "chamfer", "fillet",
    "blind_hole", "through_hole", "rect_pocket", "rect_passage",
]
GRAPH_LABEL_NAMES = ["turning", "milling", "milling+turning"]


def _clf_head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(32, out_dim),
    )


class UVNetMultiTask(nn.Module):
    """
    Multi-task UVNet.

    Args:
        gnn_type  : "gcn" | "sage" | "gat"
        hidden_size: GNN 중간 채널 수
        dropout   : dropout 비율
        num_uv    : UV grid 해상도 (학습 시와 전처리 시 동일해야 함)
    """

    def __init__(
        self,
        gnn_type: str = "gcn",
        hidden_size: int = 64,
        dropout: float = 0.0,
        num_uv: int = 10,
    ):
        super().__init__()
        self.srf_encoder = SurfaceEncoder(num_uv=num_uv)
        self.crv_encoder = CurveEncoder(num_uv=num_uv)

        gnn_cls = GNN_REGISTRY[gnn_type.lower()]
        self.gnn = gnn_cls(hidden_size=hidden_size, dropout=dropout)

        self.face_head  = _clf_head(64, NUM_FACE_CLASSES,  dropout)
        self.graph_head = _clf_head(64, NUM_GRAPH_CLASSES, dropout)

    def forward(self, data: Data):
        """
        Returns:
            face_logits  : (N_total, 9)   — 배치 내 전체 노드
            graph_logits : (batch_size, 3)
        """
        srf_feat = self.srf_encoder(data.x)          # (N, 64)

        if data.edge_attr is not None and data.edge_attr.size(0) > 0:
            crv_feat = self.crv_encoder(data.edge_attr)  # (E, 64)
        else:
            crv_feat = data.x.new_zeros((0, 64))

        # GNN에 넘기기 위해 복사본 수정 (원본 batch data 보호)
        gnn_data = data.clone()
        gnn_data.x        = srf_feat
        gnn_data.edge_attr = crv_feat

        node_feat = self.gnn(gnn_data)               # (N, 64)

        face_logits  = self.face_head(node_feat)      # (N, 9)

        graph_feat   = global_mean_pool(node_feat, data.batch)  # (B, 64)
        graph_logits = self.graph_head(graph_feat)    # (B, 3)

        return face_logits, graph_logits
