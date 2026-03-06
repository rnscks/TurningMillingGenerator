"""
TurningTreePlanner: Top-Down BFS로 트리 각 노드의 region을 확정

- step 노드의 direction은 TreeNode.direction에 미리 저장되어 있음 (tree 생성 시 결정)
- BFS에서 각 step 노드의 direction에 따라 z 슬롯을 균등 분배
- Groove는 부모 region 내부에 배치

TurningPlanner: TurningTreePlanner + TurningShapeBuilder 조합 편의 인터페이스
"""

import random
from collections import deque
from typing import List, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Shape

from core.tree.node import TreeNode, Region
from core.turning.params import TurningParams
from core.turning.builder import TurningShapeBuilder


class TurningTreePlanner:

    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self.stock_height: float = 0.0
        self.stock_radius: float = 0.0

    def plan(self, root: TreeNode) -> Tuple[float, float]:
        self.stock_height = random.uniform(*self.params.stock_height_range)
        self.stock_radius = random.uniform(*self.params.stock_radius_range)

        root.region = Region(
            height=self.stock_height,
            radius=self.stock_radius,
        )
        self._assign_regions_bfs(root)
        return self.stock_height, self.stock_radius


    def _assign_regions_bfs(self, root: TreeNode):
        """BFS로 각 노드의 region을 확정."""
        queue: List[Tuple[TreeNode, Region]] = []
        queue.append((root, root.region))
        total_child_cnt: int = self._count_nodes(root)
        root_height: float = root.region.height

        while queue:
            node, parent_region = queue.pop(0)
            
            groove_cnt: int = len(node.children_by('g'))

            for child in node.children:
                child_cnt: int = self._count_nodes(child)
                if child.region is None and child.label == 's':
                    
                    if child_cnt != 0:
                        height: float = ((root_height / total_child_cnt) / 2) * child_cnt
                        radius: float = parent_region.radius - (parent_region.radius - self.params.min_remaining_radius) / child_cnt
                    else:
                        height: float = parent_region.height / 2
                        radius: float = parent_region.radius / 2

                    child.region = Region(height=height, radius=radius)
                    queue.append((child, child.region))
                if child.region is None and child.label == 'g':
                    if child_cnt != 0:
                        height: float = (root_height / total_child_cnt) * child_cnt
                        height /= groove_cnt
                        radius: float = parent_region.radius - (parent_region.radius - self.params.min_remaining_radius) / child_cnt
                    else:
                        height: float = parent_region.height / 2
                        height /= groove_cnt
                        radius: float = parent_region.radius / 2

                    child.region = Region(height=height, radius=radius)
                    queue.append((child, child.region))
        
    def _count_nodes(self, node: TreeNode) -> int:
        cnt: int = 0
        queue: List[TreeNode] = list(node.children)
        while queue:
            n = queue.pop(0)
            cnt += 1
            queue.extend(n.children)
        return cnt

class TurningPlanner:

    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self._tree_planner: Optional[TurningTreePlanner] = None
        self._builder: Optional[TurningShapeBuilder] = None

    def plan_and_apply(
        self,
        root: TreeNode,
        label_maker=None,
    ) -> Tuple[float, float, TopoDS_Shape]:
        self._tree_planner = TurningTreePlanner(self.params)
        stock_height, stock_radius = self._tree_planner.plan(root)

        self._builder = TurningShapeBuilder(stock_height, stock_radius, self.params)
        shape = self._builder.build(root, label_maker)

        return stock_height, stock_radius, shape
