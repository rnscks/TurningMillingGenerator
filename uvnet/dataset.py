"""
PyTorch Geometric Dataset for preprocessed .pt graph files.

각 .pt 파일은 Data 객체를 담고 있으며, 다음 필드를 포함합니다:
    x          : (N, 7, uv, uv)   face UV feature
    edge_index : (2, E)
    edge_attr  : (E, 6, uv)       edge UV feature
    face_y     : (N,)             face-level label  (long, -1 = unknown)
    graph_y    : ()               graph-level label (long)
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torch_geometric.data import Dataset, Data

from uvnet.model import FACE_LABEL_NAMES, GRAPH_LABEL_NAMES


class GraphDataset(Dataset):
    """
    .pt 파일 목록을 받아 그래프 데이터를 제공하는 Dataset.

    Args:
        pt_files: .pt 파일 경로 리스트
    """

    def __init__(self, pt_files: List[str]):
        super().__init__()
        self._files = pt_files

    @property
    def files(self) -> List[str]:
        return self._files

    def len(self) -> int:
        return len(self._files)

    def get(self, idx: int) -> Data:
        return torch.load(self._files[idx], weights_only=False)

    @staticmethod
    def from_directory(directory: str) -> "GraphDataset":
        """디렉토리에서 모든 .pt 파일을 로드."""
        files = sorted(str(p) for p in Path(directory).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f".pt 파일이 없습니다: {directory}")
        return GraphDataset(files)

    def compute_statistics(self) -> Dict:
        """
        전체 데이터셋의 통계를 계산합니다.

        Returns
        -------
        dict with keys:
            n_graphs        : 총 그래프 수
            avg_faces       : 그래프당 평균 face 수
            avg_edges       : 그래프당 평균 edge 수
            graph_label_cnt : {label_id: count}
            face_label_cnt  : {label_id: count}  (-1 unknown 포함)
        """
        graph_cnt  = Counter()
        face_cnt   = Counter()
        total_faces = 0
        total_edges = 0

        for f in self._files:
            data = torch.load(f, weights_only=False)
            graph_cnt[int(data.graph_y)] += 1
            total_faces += data.x.size(0)
            total_edges += data.edge_index.size(1) // 2  # undirected → 실제 edge 수
            for lbl in data.face_y.tolist():
                face_cnt[lbl] += 1

        n = len(self._files)
        return {
            "n_graphs"       : n,
            "avg_faces"      : total_faces / max(n, 1),
            "avg_edges"      : total_edges / max(n, 1),
            "graph_label_cnt": dict(graph_cnt),
            "face_label_cnt" : dict(face_cnt),
        }

    def print_statistics(self, title: str = "데이터셋 통계") -> None:
        """compute_statistics() 결과를 콘솔에 출력합니다."""
        stats = self.compute_statistics()
        n     = stats["n_graphs"]
        W     = 54

        print(f"\n{'='*W}")
        print(f" {title}")
        print(f"{'─'*W}")
        print(f"  총 그래프 수  : {n:,}개")
        print(f"  평균 face 수  : {stats['avg_faces']:.1f}")
        print(f"  평균 edge 수  : {stats['avg_edges']:.1f}")

        # Graph-level
        print(f"\n  [Graph-level 분포]")
        gcnt = stats["graph_label_cnt"]
        total_g = sum(gcnt.values())
        for lid, name in enumerate(GRAPH_LABEL_NAMES):
            c = gcnt.get(lid, 0)
            bar = "█" * int(c / max(total_g, 1) * 20)
            print(f"    {name:22s}: {c:4d}개  ({c/max(total_g,1)*100:5.1f}%)  {bar}")

        # Face-level
        print(f"\n  [Face-level 분포]")
        fcnt = stats["face_label_cnt"]
        total_f = sum(v for k, v in fcnt.items() if k >= 0)
        for lid, name in enumerate(FACE_LABEL_NAMES):
            c = fcnt.get(lid, 0)
            bar = "█" * int(c / max(total_f, 1) * 20)
            print(f"    {name:22s}: {c:6d}  ({c/max(total_f,1)*100:5.1f}%)  {bar}")
        unknown = fcnt.get(-1, 0)
        if unknown:
            print(f"    {'unknown(-1)':22s}: {unknown:6d}")

        print(f"{'='*W}\n")
