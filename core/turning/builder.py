"""
TurningShapeBuilder: node.region이 확정된 트리를 받아 실제 3D 형상 생성

TurningTreePlanner가 확정한 node.region을 그대로 사용해
stock을 생성하고 BFS Top-Down으로 Step/Groove Boolean Cut을 적용.
"""

from collections import deque
from typing import Optional

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

    def build(
        self,
        root: TreeNode,
        label_maker=None,
    ) -> TopoDS_Shape:
        """stock을 생성하고 트리에 따라 Step/Groove를 적용한 최종 형상을 반환."""
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
                    applied = True

            elif node.label == 'g':
                result = self._apply_groove(node, shape, label_maker)
                if result is not None:
                    shape = result
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
        """node.region(height, radius)과 node.direction으로 Step Boolean Cut 적용.

        top:    stock 상단에서 height만큼 내려온 구간 [z_max - height, z_max]
        bottom: stock 하단에서 height만큼 올라온 구간 [0, height]
        """
        region = node.region
        outer_r = self._get_parent_radius(node)
        inner_r = region.radius

        direction = node.direction or 'top'
        if direction == 'top':
            z_cut_max = self._stock_height
            z_cut_min = z_cut_max - region.height
        else:
            z_cut_min = 0.0
            z_cut_max = region.height

        print(f"    [Step 적용] node_id={node.id}, dir={direction}, "
              f"z=[{z_cut_min:.2f}, {z_cut_max:.2f}], "
              f"outer_r={outer_r:.2f}, inner_r={inner_r:.2f}")

        result = apply_step_cut(shape, z_cut_min, z_cut_max, outer_r, inner_r, label_maker)
        if result is None:
            print(f"    [Warning] Step Boolean Cut 실패 (node_id={node.id})")
        return result

    def _apply_groove(
        self,
        node: TreeNode,
        shape: TopoDS_Shape,
        label_maker,
    ) -> Optional[TopoDS_Shape]:
        """node.region(height, radius)으로 Groove Boolean Cut 적용.

        형제 groove 중 몇 번째인지(slot_index)와 총 개수(slot_total)로
        부모 z 범위를 균등 분할하여 배치.
        """
        region = node.region
        width = region.height
        outer_r = self._get_parent_radius(node)
        inner_r = region.radius

        parent = node.parent_node
        if parent is not None and parent.region is not None:
            p_region = parent.region
            p_direction = parent.direction or 'top'
            if p_direction == 'top':
                p_z_max = self._stock_height
                p_z_min = p_z_max - p_region.height
            else:
                p_z_min = 0.0
                p_z_max = p_region.height
        else:
            p_z_min = 0.0
            p_z_max = self._stock_height

        # 형제 groove 중 내 슬롯 index 계산
        if parent is not None:
            sibling_grooves = [c for c in parent.children if c.label == 'g']
            slot_total = len(sibling_grooves)
            slot_index = sibling_grooves.index(node) if node in sibling_grooves else 0
        else:
            slot_total = 1
            slot_index = 0

        p_span = p_z_max - p_z_min
        slot_size = p_span / slot_total
        slot_start = p_z_min + slot_index * slot_size
        slot_center = slot_start + slot_size / 2.0
        z_min = slot_center - width / 2.0
        z_min = max(z_min, slot_start)

        print(f"    [Groove 적용] node_id={node.id}, slot={slot_index+1}/{slot_total}, "
              f"z=[{z_min:.2f}, {z_min + width:.2f}], "
              f"width={width:.2f}, outer_r={outer_r:.2f}, inner_r={inner_r:.2f}")

        result = apply_groove_cut(shape, z_min, width, outer_r, inner_r, label_maker)
        if result is None:
            print(f"    [Warning] Groove Boolean Cut 실패 (node_id={node.id})")
        return result

    def _get_parent_radius(self, node: TreeNode) -> float:
        """부모 노드의 radius를 반환. 부모가 없으면 stock radius 사용."""
        if node.parent_node is not None and node.parent_node.region is not None:
            return node.parent_node.region.radius
        return self._stock_radius
