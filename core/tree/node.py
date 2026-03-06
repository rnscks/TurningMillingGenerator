"""
트리 노드 데이터 클래스

- Region: z 범위와 반경으로 표현되는 가공 가능 영역
- TreeNode: 터닝 트리 구조의 단일 노드
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Region:
    """가공 가능 영역 - 높이와 반경"""
    height: float
    radius: float

    def __repr__(self):
        return f"Region(height={self.height:.2f}, r={self.radius:.2f})"


class TreeNode:
    """트리 노드 클래스"""

    def __init__(self, node_id: int, label: str, parent_id: Optional[int], depth: int):
        self.id = node_id
        self.label = label
        self.parent_id = parent_id
        self.depth = depth
        self.direction: Optional[str] = None  # step 노드: 'top' or 'bottom', 그 외: None
        self.children: List['TreeNode'] = []
        self.region: Optional[Region] = None
        self.parent_node: Optional['TreeNode'] = None

    def children_by(self, label: str) -> List['TreeNode']:
        """특정 label의 자식 노드 목록 반환."""
        return [c for c in self.children if c.label == label]

    def __repr__(self):
        return f"TreeNode(id={self.id}, label='{self.label}', depth={self.depth}, dir={self.direction})"


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

    # direction 필드 로드 (step 노드)
    for nd in nodes_data:
        if 'direction' in nd:
            nodes[nd['id']].direction = nd['direction']

    return root
