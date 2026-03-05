"""
GNN 모듈

GCN, SAGE, GAT 중 하나를 선택해서 사용.
모두 64-dim 입력 → 64-dim 출력으로 통일.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    def __init__(self, hidden_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(64, hidden_size)
        self.bn1   = nn.BatchNorm1d(hidden_size)
        self.conv2 = GCNConv(hidden_size, 64)
        self.bn2   = nn.BatchNorm1d(64)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index)
        x = F.leaky_relu(self.bn1(x))
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(x, data.edge_index)
        x = F.leaky_relu(self.bn2(x))
        x = F.dropout(x, self.dropout, self.training)
        return x


class SAGE(nn.Module):
    def __init__(self, hidden_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.conv1 = SAGEConv(64, hidden_size)
        self.bn1   = nn.BatchNorm1d(hidden_size)
        self.conv2 = SAGEConv(hidden_size, 64)
        self.bn2   = nn.BatchNorm1d(64)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index)
        x = F.leaky_relu(self.bn1(x))
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(x, data.edge_index)
        x = F.leaky_relu(self.bn2(x))
        x = F.dropout(x, self.dropout, self.training)
        return x


class GAT(nn.Module):
    """GAT는 edge_attr(64-dim)을 활용합니다."""

    def __init__(self, hidden_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GATConv(64, hidden_size, edge_dim=64, heads=8, concat=False)
        self.bn1   = nn.BatchNorm1d(hidden_size)
        self.conv2 = GATConv(hidden_size, 64, edge_dim=64, heads=8, concat=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        edge_attr = data.edge_attr if (data.edge_attr is not None and data.edge_attr.size(0) > 0) else None
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(self.bn1(x))
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(self.bn2(x))
        x = F.dropout(x, self.dropout, self.training)
        return x


GNN_REGISTRY = {
    "gcn":  GCN,
    "sage": SAGE,
    "gat":  GAT,
}
