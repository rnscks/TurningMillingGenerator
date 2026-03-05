"""
TurningShapeBuilder: node.region이 확정된 트리를 받아 실제 3D 형상 생성

TurningTreePlanner가 확정한 node.region을 그대로 사용해
stock을 생성하고 BFS Top-Down으로 Step/Groove Boolean Cut을 적용.
"""

from collections import deque
from typing import List, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Shape

from core.tree.node import TreeNode
from core.turning.params import TurningParams
from core.turning.features import create_stock, apply_step_cut, apply_groove_cut
from core.label_maker import Labels


class TurningShapeBuilder:
    """node.region이 확정된 트리를 받아 실제 3D 형상을 생성.

    사용법:
        builder = TurningShapeBuilder(stock_height, stock_radius, params)
        shape   = builder.build(root_node, label_maker)
    """

    def __init__(
        self,
        stock_height: float,
        stock_radius: float,
        params: TurningParams = None,
    ):
        self._stock_height = stock_height
        self._stock_radius = stock_radius
        self.params = params or TurningParams()
        self._occupied_ranges: List[Tuple[float, float]] = []

    def build(
        self,
        root: TreeNode,
        label_maker=None,
    ) -> TopoDS_Shape:
        """stock을 생성하고 트리에 따라 Step/Groove를 적용한 최종 형상을 반환."""
        self._occupied_ranges = []

        shape = create_stock(self._stock_height, self._stock_radius)

        if label_maker is not None:
            label_maker.initialize(shape, base_label=Labels.STOCK)

        shape = self._apply_cuts_bfs(root, shape, label_maker)

        return shape

    # -------------------------------------------------------------------------
    # BFS Top-Down: Step/Groove Boolean Cut 적용
    # -------------------------------------------------------------------------

    def _apply_cuts_bfs(
        self,
        root: TreeNode,
        shape: TopoDS_Shape,
        label_maker,
    ) -> TopoDS_Shape:
        """BFS로 루트 → 리프 방향으로 순회하며 Boolean Cut을 적용.

        TreePlanner가 미리 확정한 node.region을 그대로 사용.
        충돌 감지 또는 Boolean Cut 실패 시 해당 노드와 그 자식은 스킵.
        """
        # (node, apply_parent_succeeded)
        queue: deque = deque()
        queue.append((root, True))

        while queue:
            node, parent_ok = queue.popleft()

            if node.label == 'b':
                for child in node.children:
                    queue.append((child, True))
                continue

            if not parent_ok:
                for child in node.children:
                    queue.append((child, False))
                continue

            region = node.region
            if region is None:
                print(f"    [Warning] region 미설정, 스킵 (node_id={node.id})")
                for child in node.children:
                    queue.append((child, False))
                continue

            applied = False

            if node.label == 's':
                result = self._apply_step(node, shape, label_maker)
                if result is not None:
                    shape = result
                    self._register_z_range(region.z_min, region.z_max)
                    applied = True

            elif node.label == 'g':
                result = self._apply_groove(node, shape, label_maker)
                if result is not None:
                    shape = result
                    self._register_z_range(region.z_min, region.z_max)
                    applied = True

            for child in node.children:
                queue.append((child, applied))

        return shape

    def _apply_step(
        self,
        node: TreeNode,
        shape: TopoDS_Shape,
        label_maker,
    ) -> Optional[TopoDS_Shape]:
        """node.region을 기반으로 Step Boolean Cut 적용."""
        region = node.region
        direction = region.direction or 'top'

        if self._check_z_overlap(region.z_min, region.z_max):
            print(f"    [Warning] Step z 범위 [{region.z_min:.2f}, {region.z_max:.2f}] 충돌, 스킵 (node_id={node.id})")
            return None

        # direction=top    → zpos = region.z_min (region=[zpos, stock_height])
        # direction=bottom → zpos = region.z_max (region=[0, zpos])
        zpos = region.z_min if direction == 'top' else region.z_max
        outer_r = self._get_parent_radius(node)
        inner_r = region.radius

        result = apply_step_cut(shape, zpos, direction, outer_r, inner_r, self._stock_height, label_maker)
        if result is None:
            print(f"    [Warning] Step Boolean Cut 실패 (node_id={node.id})")
        return result

    def _apply_groove(
        self,
        node: TreeNode,
        shape: TopoDS_Shape,
        label_maker,
    ) -> Optional[TopoDS_Shape]:
        """node.region을 기반으로 Groove Boolean Cut 적용."""
        region = node.region
        z_min = region.z_min
        z_max = region.z_max
        width = z_max - z_min

        if self._check_z_overlap(z_min, z_max):
            print(f"    [Warning] Groove z 범위 [{z_min:.2f}, {z_max:.2f}] 충돌, 스킵 (node_id={node.id})")
            return None

        outer_r = self._get_parent_radius(node)
        inner_r = region.radius

        result = apply_groove_cut(shape, z_min, width, outer_r, inner_r, label_maker)
        if result is None:
            print(f"    [Warning] Groove Boolean Cut 실패 (node_id={node.id})")
        return result

    def _get_parent_radius(self, node: TreeNode) -> float:
        """부모 노드의 radius를 반환. 부모가 없으면 stock radius 사용."""
        if node.parent_node is not None and node.parent_node.region is not None:
            return node.parent_node.region.radius
        return self._stock_radius

    def _check_z_overlap(self, z_min: float, z_max: float) -> bool:
        """z 범위가 기존 점유 범위와 겹치는지 검사."""
        for occ_min, occ_max in self._occupied_ranges:
            if z_min < occ_max and z_max > occ_min:
                return True
        return False

    def _register_z_range(self, z_min: float, z_max: float):
        """점유 범위 등록."""
        self._occupied_ranges.append((z_min, z_max))
