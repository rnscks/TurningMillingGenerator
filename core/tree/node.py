"""
트리 노드 데이터 클래스

- Region: z 범위와 반경으로 표현되는 가공 가능 영역
- TreeNode: 터닝 트리 구조의 단일 노드
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Region:
    """가공 가능 영역 - z 범위와 반경"""
    z_min: float
    z_max: float
    radius: float
    direction: Optional[str] = None  # 'top' or 'bottom' for step

    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    def __repr__(self):
        return (
            f"Region(z=[{self.z_min:.2f}, {self.z_max:.2f}], "
            f"r={self.radius:.2f}, dir={self.direction})"
        )


class TreeNode:
    """트리 노드 클래스"""

    def __init__(self, node_id: int, label: str, parent_id: Optional[int], depth: int):
        self.id = node_id
        self.label = label
        self.parent_id = parent_id
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.region: Optional[Region] = None
        self.parent_node: Optional['TreeNode'] = None

    def __repr__(self):
        return f"TreeNode(id={self.id}, label='{self.label}', depth={self.depth})"


def load_tree(tree_data: dict) -> TreeNode:
    """JSON 트리 데이터를 TreeNode 구조로 변환."""
    nodes_data = tree_data['nodes']

    nodes: dict = {}
    for nd in nodes_data:
        node = TreeNode(
            node_id=nd['id'],
            label=nd['label'],
            parent_id=nd['parent'],
            depth=nd['depth']
        )
        nodes[node.id] = node

    root = None
    for node in nodes.values():
        if node.parent_id is not None:
            parent = nodes[node.parent_id]
            parent.children.append(node)
            node.parent_node = parent
        else:
            root = node

    return root
