"""
TurningTreePlanner: Top-Down BFS로 트리 각 노드의 region을 확정

1. stock 크기(height, radius) 먼저 결정
2. BFS로 루트 → 리프 방향 레벨별 순회하며 각 노드의 region을 확정
   - Step : 부모 region의 한쪽 끝(top/bottom)을 잘라서 node.region으로 가져감
   - Groove: 부모 region 내부에 포함되는 위치에 node.region 확정

마진 적용 방식:
- Step  : zpos 경계면 양쪽(진행 방향·반대 방향) 모두 margin 적용
- Groove: z 범위 상단/하단 양쪽 모두 margin 적용

TurningPlanner: TurningTreePlanner + TurningShapeBuilder 조합 편의 인터페이스
"""

import random
from collections import deque
from typing import List, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Shape

from core.tree.node import TreeNode, Region
from core.turning.params import TurningParams
from core.turning.builder import TurningShapeBuilder


MAX_RECURSION_DEPTH = 50


class TurningTreePlanner:
    """Top-Down BFS로 각 노드의 region을 확정하고 stock 크기를 결정.

    사용법:
        tree_planner = TurningTreePlanner(TurningParams())
        stock_height, stock_radius = tree_planner.plan(root_node)

    plan() 완료 후 각 노드의 node.region이 채워진 상태가 됨.
    TurningShapeBuilder는 이 region을 그대로 사용해 형상을 생성.
    """

    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self.stock_height: float = 0.0
        self.stock_radius: float = 0.0

    def plan(self, root: TreeNode) -> Tuple[float, float]:
        """stock 크기를 결정하고 BFS로 각 노드의 region을 확정.

        Returns:
            (stock_height, stock_radius)
        """
        self.stock_height = random.uniform(*self.params.stock_height_range)
        self.stock_radius = random.uniform(*self.params.stock_radius_range)

        root.region = Region(
            z_min=0.0,
            z_max=self.stock_height,
            radius=self.stock_radius,
        )

        print(
            f"  Stock: height={self.stock_height:.2f}, radius={self.stock_radius:.2f}"
        )

        self._assign_regions_bfs(root)

        return self.stock_height, self.stock_radius

    # -------------------------------------------------------------------------
    # BFS Top-Down: 각 노드의 region 확정
    # -------------------------------------------------------------------------

    def _assign_regions_bfs(self, root: TreeNode):
        """BFS로 루트 → 리프 방향으로 순회하며 각 노드의 region을 확정.

        큐 원소: (node, parent_region, step_direction)
        - step_direction: step 노드가 어느 방향을 잘라갈지 ('top' or 'bottom')
        """
        queue: deque = deque()
        queue.append((root, root.region, None))

        while queue:
            node, parent_region, step_direction = queue.popleft()

            if node.label == 'b':
                step_children = [c for c in node.children if c.label == 's']
                for i, child in enumerate(step_children):
                    child_dir = 'top' if i % 2 == 0 else 'bottom'
                    queue.append((child, node.region, child_dir))

                groove_children = [c for c in node.children if c.label == 'g']
                for child in groove_children:
                    queue.append((child, node.region, None))

            elif node.label == 's':
                region = self._sample_step_region(node, parent_region, step_direction or 'top')
                if region is None:
                    print(f"    [Warning] Step region 샘플링 실패, 스킵 (node_id={node.id})")
                    node.region = parent_region
                    for child in node.children:
                        if child.label == 's':
                            queue.append((child, parent_region, step_direction))
                        elif child.label == 'g':
                            queue.append((child, parent_region, None))
                    continue

                node.region = region

                step_children = [c for c in node.children if c.label == 's']
                for i, child in enumerate(step_children):
                    child_dir = 'top' if i % 2 == 0 else 'bottom'
                    queue.append((child, node.region, child_dir))

                groove_children = [c for c in node.children if c.label == 'g']
                for child in groove_children:
                    queue.append((child, node.region, None))

            elif node.label == 'g':
                siblings = []
                if node.parent_node is not None:
                    siblings = [c for c in node.parent_node.children if c.label == 'g']
                groove_index = next(
                    (i for i, s in enumerate(siblings) if s.id == node.id), 0
                )
                total_grooves = len(siblings) if siblings else 1

                region = self._sample_groove_region(
                    node, parent_region, groove_index, total_grooves
                )
                if region is None:
                    print(f"    [Warning] Groove region 샘플링 실패, 스킵 (node_id={node.id})")
                    node.region = parent_region
                    for child in node.children:
                        if child.label == 'g':
                            queue.append((child, parent_region, None))
                    continue

                node.region = region

                groove_children = [c for c in node.children if c.label == 'g']
                for child in groove_children:
                    queue.append((child, node.region, None))

    # -------------------------------------------------------------------------
    # Step region 샘플링
    # -------------------------------------------------------------------------

    def _sample_step_region(
        self,
        node: TreeNode,
        parent_region: Region,
        direction: str,
    ) -> Optional[Region]:
        """부모 region의 한쪽 끝을 잘라서 step의 region을 결정.

        direction='top'   : [zpos, parent.z_max] 구간을 step이 가져감
        direction='bottom': [parent.z_min, zpos] 구간을 step이 가져감

        zpos는 부모 region 안에서 margin을 양쪽에 두고 균등 샘플링.
        depth는 반경 방향 깎기 깊이.
        """
        margin = self.params.step_margin

        if parent_region.height - 2 * margin <= 0:
            return None

        depth = random.uniform(*self.params.step_depth_range)
        depth, ok = self._validate_depth(depth, parent_region.radius, node.id, 'Step')
        if not ok:
            return None

        new_radius = parent_region.radius - depth
        z_inner_min = parent_region.z_min + margin
        z_inner_max = parent_region.z_max - margin

        if direction == 'top':
            zpos = random.uniform(z_inner_min, z_inner_max)
            cut_z_min, cut_z_max = zpos, self.stock_height
        else:
            zpos = random.uniform(z_inner_min, z_inner_max)
            cut_z_min, cut_z_max = 0.0, zpos

        if cut_z_max - cut_z_min <= 0:
            return None

        print(
            f"    Step ({direction}): zpos={zpos:.2f}, z=[{cut_z_min:.2f}, {cut_z_max:.2f}], "
            f"r={new_radius:.2f}, margin(both)={margin:.2f}"
        )

        return Region(
            z_min=cut_z_min,
            z_max=cut_z_max,
            radius=new_radius,
            direction=direction,
        )

    # -------------------------------------------------------------------------
    # Groove region 샘플링
    # -------------------------------------------------------------------------

    def _sample_groove_region(
        self,
        node: TreeNode,
        parent_region: Region,
        groove_index: int = 0,
        total_grooves: int = 1,
    ) -> Optional[Region]:
        """부모 region 내부에 groove의 region을 결정.

        groove는 부모 region을 소비하지 않고 내부에 포함.
        margin은 groove 상단/하단 양쪽 적용.
        형제 groove가 여러 개일 때 가용 구간을 균등 분할해 배치.
        """
        margin = self.params.groove_margin

        depth = random.uniform(*self.params.groove_depth_range)
        depth, ok = self._validate_depth(depth, parent_region.radius, node.id, 'Groove')
        if not ok:
            return None

        new_radius = parent_region.radius - depth
        width = random.uniform(*self.params.groove_width_range)
        if width <= 0:
            width = self.params.groove_width_range[0]

        avail_z_min = parent_region.z_min + margin
        avail_z_max = parent_region.z_max - margin
        avail_span = avail_z_max - avail_z_min

        if avail_span <= 0:
            return None

        if total_grooves == 1:
            max_zpos = avail_z_max - width
            if max_zpos < avail_z_min:
                width = avail_span
                max_zpos = avail_z_min
            zpos = random.uniform(avail_z_min, max_zpos)
        else:
            slot_size = avail_span / total_grooves
            slot_start = avail_z_min + groove_index * slot_size
            slot_end = slot_start + slot_size

            max_zpos = slot_end - width
            if max_zpos < slot_start:
                width = slot_size
                max_zpos = slot_start
            zpos = random.uniform(slot_start, max_zpos)

        zpos = max(zpos, avail_z_min)
        zpos = min(zpos, avail_z_max - width)

        if zpos + width > avail_z_max or zpos < avail_z_min:
            return None

        print(
            f"    Groove: zpos={zpos:.2f}, width={width:.2f}, "
            f"z=[{zpos:.2f}, {zpos + width:.2f}], r={new_radius:.2f}, "
            f"margin(both)={margin:.2f}"
        )

        return Region(
            z_min=zpos,
            z_max=zpos + width,
            radius=new_radius,
            direction=parent_region.direction,
        )

    # -------------------------------------------------------------------------
    # 공통 검증
    # -------------------------------------------------------------------------

    def _validate_depth(
        self,
        depth: float,
        parent_radius: float,
        node_id: int,
        feature_name: str,
    ) -> Tuple[float, bool]:
        """depth가 min_remaining_radius를 침범하지 않도록 클램핑.

        Returns:
            (depth, ok) — ok=False이면 배치 불가
        """
        max_depth = parent_radius - self.params.min_remaining_radius
        if max_depth <= 0:
            print(f"    [Warning] {feature_name} 반경 부족으로 배치 불가 (node_id={node_id})")
            return depth, False

        if depth > max_depth:
            depth = max_depth * 0.9
            print(f"    [Warning] {feature_name} depth 클램핑 → {depth:.2f} (node_id={node_id})")

        return depth, True


# ============================================================================
# TurningPlanner: TurningTreePlanner + TurningShapeBuilder 조합 편의 인터페이스
# ============================================================================

class TurningPlanner:
    """TurningTreePlanner와 TurningShapeBuilder를 조합한 편의 인터페이스.

    사용법:
        planner = TurningPlanner(TurningParams())
        stock_height, stock_radius, shape = planner.plan_and_apply(root_node)
    """

    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self._tree_planner: Optional[TurningTreePlanner] = None
        self._builder: Optional[TurningShapeBuilder] = None

    def plan_and_apply(
        self,
        root: TreeNode,
        label_maker=None,
    ) -> Tuple[float, float, TopoDS_Shape]:
        """트리 파라미터 구체화 + 형상 생성을 순서대로 실행.

        Returns:
            (stock_height, stock_radius, 최종 형상)
        """
        self._tree_planner = TurningTreePlanner(self.params)
        stock_height, stock_radius = self._tree_planner.plan(root)

        self._builder = TurningShapeBuilder(stock_height, stock_radius, self.params)
        shape = self._builder.build(root, label_maker)

        return stock_height, stock_radius, shape

    @property
    def _occupied_ranges(self) -> List[Tuple[float, float]]:
        if self._builder is not None:
            return self._builder._occupied_ranges
        return []
