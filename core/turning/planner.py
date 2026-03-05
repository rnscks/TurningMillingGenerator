"""
트리 기반 터닝 특징형상 배치 계획 (TurningPlanner)

Bottom-Up 방식으로 트리를 순회하여:
1. 리프 → 루트 방향으로 필요 공간(RequiredSpace) 계산
2. Stock 크기 결정 (StockInfo)
3. 루트 → 리프 방향으로 Step 배치 좌표 결정 및 TurningFeatureRequest 생성
4. Step 완료 후 Groove 배치 좌표 결정

클래스인 이유: params 상태 보유 + 재귀 순회 중 내부 상태 관리 필요
트리 구조를 읽기만 함 (TreeNode.region 외부 할당은 플래너 내부 임시 상태)

TurningParams: 이 플래너가 사용하는 모든 제약조건 파라미터를 정의.
plan_and_apply(): plan() 결과를 내부에서 즉시 형상에 적용하여 반환.
  TurningFeatureRequest는 plan()과 apply 사이의 내부 계약으로, 외부 노출이 불필요.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.tree.node import TreeNode, Region, RequiredSpace
from core.turning.features import StockInfo, TurningFeatureRequest, create_stock, apply_turning_requests
from core.label_maker import Labels


# ============================================================================
# TurningParams
# ============================================================================

@dataclass
class TurningParams:
    """터닝 플래너 제약조건 파라미터"""
    stock_height_margin: Tuple[float, float] = (3.0, 8.0)
    stock_radius_margin: Tuple[float, float] = (2.0, 5.0)

    step_depth_range: Tuple[float, float] = (0.8, 1.5)
    step_height_range: Tuple[float, float] = (2.0, 4.0)
    step_margin: float = 0.5

    groove_depth_range: Tuple[float, float] = (0.4, 0.8)
    groove_width_range: Tuple[float, float] = (1.5, 3.0)
    groove_margin: float = 0.5

    min_remaining_radius: float = 2.0

    chamfer_range: Tuple[float, float] = (0.3, 0.8)
    fillet_range: Tuple[float, float] = (0.3, 0.8)
    edge_feature_prob: float = 0.3

MAX_RECURSION_DEPTH = 50


class TurningPlanner:
    """트리 기반 터닝 특징형상 배치 계획.

    사용법:
        planner = TurningPlanner(TurningParams())
        stock_info, requests = planner.plan(root_node)
    """

    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self._stock_height: float = 0.0
        self._stock_radius: float = 0.0

    def plan(self, root: TreeNode) -> Tuple[StockInfo, List[TurningFeatureRequest]]:
        """트리를 순회하여 Stock 정보와 TurningFeatureRequest 목록 생성.

        Args:
            root: 트리 루트 노드 (label == 'b')

        Returns:
            (StockInfo, List[TurningFeatureRequest])
        """
        self._calculate_required_space(root)

        required = root.required_space
        height_margin = random.uniform(*self.params.stock_height_margin)
        radius_margin = random.uniform(*self.params.stock_radius_margin)

        self._stock_height = required.height + height_margin
        self._stock_radius = (
            required.depth + self.params.min_remaining_radius + radius_margin
        )

        stock_info = StockInfo(
            height=self._stock_height,
            radius=self._stock_radius,
        )

        print(
            f"  Stock: height={self._stock_height:.2f}, radius={self._stock_radius:.2f} "
            f"(required: h={required.height:.2f}, d={required.depth:.2f})"
        )

        requests: List[TurningFeatureRequest] = []

        self._collect_step_requests(root, None, requests)
        self._collect_groove_requests(root, requests)

        return stock_info, requests

    def plan_and_apply(
        self,
        root: TreeNode,
        label_maker=None,
    ):
        """트리 계획 후 즉시 Boolean Cut 형상까지 적용하여 반환.

        plan()의 결과를 외부에 노출하지 않고 내부에서 처리합니다.

        Args:
            root: 트리 루트 노드
            label_maker: Face 라벨 관리자 (None이면 라벨링 비활성)

        Returns:
            (StockInfo, TopoDS_Shape)
        """
        stock_info, requests = self.plan(root)
        shape = create_stock(stock_info)

        if label_maker is not None:
            from core.label_maker import Labels
            label_maker.initialize(shape, base_label=Labels.STOCK)

        shape = apply_turning_requests(shape, requests, label_maker)
        return stock_info, shape

    # =========================================================================
    # Bottom-Up: 필요 공간 계산 (리프 → 루트)
    # =========================================================================

    def _calculate_required_space(self, node: TreeNode, _depth: int = 0) -> RequiredSpace:
        if _depth > MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"최대 재귀 깊이({MAX_RECURSION_DEPTH}) 초과: "
                f"node={node.label}(id={node.id})"
            )

        for child in node.children:
            self._calculate_required_space(child, _depth + 1)

        step_children = [c for c in node.children if c.label == 's']
        groove_children = [c for c in node.children if c.label == 'g']

        step_children_height = (
            sum(c.required_space.height for c in step_children)
            if step_children else 0.0
        )
        groove_children_height = (
            sum(c.required_space.height for c in groove_children)
            if groove_children else 0.0
        )
        children_max_depth = max(
            (c.required_space.depth for c in node.children), default=0.0
        )

        if node.label == 'b':
            feature_height = 0.0
            feature_depth = 0.0
            node_margin = 0.0
            total_height = max(step_children_height, groove_children_height)
            total_depth = children_max_depth

        elif node.label == 's':
            feature_height = random.uniform(*self.params.step_height_range)
            feature_depth = random.uniform(*self.params.step_depth_range)
            node_margin = self.params.step_margin

            step_based_height = step_children_height + feature_height + 2 * node_margin
            total_height = max(step_based_height, groove_children_height)
            total_depth = children_max_depth + feature_depth

        elif node.label == 'g':
            random_width = random.uniform(*self.params.groove_width_range)
            feature_depth = random.uniform(*self.params.groove_depth_range)
            node_margin = self.params.groove_margin

            if groove_children:
                min_width = groove_children_height + 2 * node_margin
                feature_height = max(random_width, min_width)
            else:
                feature_height = random_width

            total_height = feature_height + 2 * node_margin
            total_depth = children_max_depth + feature_depth

        else:
            feature_height = 0.0
            feature_depth = 0.0
            node_margin = 0.0
            total_height = step_children_height + groove_children_height
            total_depth = children_max_depth

        node.required_space = RequiredSpace(
            height=total_height,
            depth=total_depth,
            feature_height=feature_height,
            feature_depth=feature_depth,
            margin=node_margin,
        )

        return node.required_space

    # =========================================================================
    # Top-Down: Step 배치 (루트 → 리프)
    # =========================================================================

    def _collect_step_requests(
        self,
        node: TreeNode,
        parent_region: Optional[Region],
        requests: List[TurningFeatureRequest],
        direction: str = None,
        _depth: int = 0,
    ):
        if _depth > MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"_collect_step_requests: 최대 재귀 깊이 초과: "
                f"node={node.label}(id={node.id})"
            )

        if node.label == 'b':
            node.region = Region(
                z_min=0,
                z_max=self._stock_height,
                radius=self._stock_radius,
            )
            step_children = [c for c in node.children if c.label == 's']
            for i, child in enumerate(step_children):
                child_dir = 'top' if i % 2 == 0 else 'bottom'
                self._collect_step_requests(
                    child, node.region, requests, child_dir, _depth + 1
                )

            groove_children = [c for c in node.children if c.label == 'g']
            for child in groove_children:
                child.region = node.region
                self._assign_groove_regions(child)

        elif node.label == 's':
            req = self._make_step_request(node, parent_region, direction or 'top')
            if req is not None:
                requests.append(req)
            else:
                node.region = parent_region
                print(f"    [Warning] Step 배치 실패 (node_id={node.id})")

            step_children = [c for c in node.children if c.label == 's']
            for child in step_children:
                self._collect_step_requests(
                    child, node.region, requests, direction, _depth + 1
                )

            groove_children = [c for c in node.children if c.label == 'g']
            for child in groove_children:
                child.region = node.region
                self._assign_groove_regions(child)

    def _make_step_request(
        self,
        node: TreeNode,
        parent_region: Region,
        direction: str,
    ) -> Optional[TurningFeatureRequest]:
        required = node.required_space
        step_height = required.feature_height
        step_depth = required.feature_depth

        if parent_region.radius - step_depth < self.params.min_remaining_radius:
            step_depth = parent_region.radius - self.params.min_remaining_radius - 0.3
            if step_depth <= 0:
                return None

        new_radius = parent_region.radius - step_depth
        cut_size = required.height

        if direction == 'top':
            cut_z_max = parent_region.z_max
            cut_z_min = max(cut_z_max - cut_size, parent_region.z_min)
        else:
            cut_z_min = parent_region.z_min
            cut_z_max = min(cut_z_min + cut_size, parent_region.z_max)

        actual_height = cut_z_max - cut_z_min
        if actual_height <= 0:
            return None

        node.region = Region(
            z_min=cut_z_min,
            z_max=cut_z_max,
            radius=new_radius,
            direction=direction,
        )

        print(
            f"    Step ({direction}): z=[{cut_z_min:.2f}, {cut_z_max:.2f}], "
            f"r={new_radius:.2f}, h={actual_height:.2f}"
        )

        return TurningFeatureRequest(
            feature_type='step',
            z_min=cut_z_min,
            z_max=cut_z_max,
            outer_radius=parent_region.radius,
            inner_radius=new_radius,
            label=Labels.STEP,
        )

    def _assign_groove_regions(self, node: TreeNode, _depth: int = 0):
        """Groove 노드와 자식들에게 부모 region 임시 할당 (groove 처리 전 준비)."""
        if _depth > MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"_assign_groove_regions: 최대 재귀 깊이 초과: "
                f"node={node.label}(id={node.id})"
            )
        for child in node.children:
            if child.label == 'g':
                child.region = node.region
                self._assign_groove_regions(child, _depth + 1)

    # =========================================================================
    # Top-Down: Groove 배치 (Step 완료 후)
    # =========================================================================

    def _collect_groove_requests(
        self,
        node: TreeNode,
        requests: List[TurningFeatureRequest],
        parent_groove_region: Optional[Region] = None,
        _depth: int = 0,
    ) -> int:
        if _depth > MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"_collect_groove_requests: 최대 재귀 깊이 초과: "
                f"node={node.label}(id={node.id})"
            )

        count = 0

        if node.label == 'g':
            parent_region = parent_groove_region or node.region

            groove_index = 0
            total_grooves = 1
            if node.parent_node is not None:
                siblings = [c for c in node.parent_node.children if c.label == 'g']
                total_grooves = len(siblings)
                for i, sib in enumerate(siblings):
                    if sib.id == node.id:
                        groove_index = i
                        break

            req = self._make_groove_request(
                node, parent_region, groove_index, total_grooves
            )
            if req is not None:
                requests.append(req)
                count += 1
            else:
                print(f"    [Warning] Groove 배치 실패 (node_id={node.id})")

            for child in node.children:
                if child.label == 'g':
                    count += self._collect_groove_requests(
                        child, requests, node.region, _depth + 1
                    )
        else:
            for child in node.children:
                if child.label == 'g':
                    count += self._collect_groove_requests(
                        child, requests, node.region, _depth + 1
                    )
                else:
                    count += self._collect_groove_requests(
                        child, requests, None, _depth + 1
                    )

        return count

    def _make_groove_request(
        self,
        node: TreeNode,
        parent_region: Region,
        groove_index: int = 0,
        total_grooves: int = 1,
        max_retries: int = 3,
    ) -> Optional[TurningFeatureRequest]:
        required = node.required_space
        groove_width = required.feature_height
        groove_depth = required.feature_depth

        if parent_region.radius - groove_depth < self.params.min_remaining_radius:
            groove_depth = parent_region.radius - self.params.min_remaining_radius - 0.2
            if groove_depth <= 0:
                return None

        new_radius = parent_region.radius - groove_depth
        groove_margin = required.margin

        for attempt in range(max_retries):
            if total_grooves == 1:
                center_z = (parent_region.z_min + parent_region.z_max) / 2
                zpos = center_z - groove_width / 2
                zpos = max(zpos, parent_region.z_min + groove_margin)
                zpos = min(zpos, parent_region.z_max - groove_width - groove_margin)
            else:
                if node.parent_node is not None:
                    siblings = [c for c in node.parent_node.children if c.label == 'g']
                    sibling_zones = [sib.required_space.height for sib in siblings]
                    sibling_margins = [sib.required_space.margin for sib in siblings]
                    sibling_widths = [sib.required_space.feature_height for sib in siblings]
                else:
                    sibling_zones = [required.height] * total_grooves
                    sibling_margins = [groove_margin] * total_grooves
                    sibling_widths = [groove_width] * total_grooves

                total_zone = sum(sibling_zones)

                if total_zone > parent_region.height:
                    scale = parent_region.height / total_zone
                    sibling_zones = [z * scale for z in sibling_zones]
                    sibling_margins = [m * scale for m in sibling_margins]
                    sibling_widths = [w * scale for w in sibling_widths]
                    groove_width = sibling_widths[groove_index]
                    groove_margin = sibling_margins[groove_index]
                    total_zone = parent_region.height

                remaining = parent_region.height - total_zone
                gap = remaining / (total_grooves + 1) if total_grooves > 0 else 0

                zpos = parent_region.z_min + gap
                for i in range(groove_index):
                    zpos += sibling_zones[i] + gap
                zpos += sibling_margins[groove_index]

                if attempt > 0 and gap > 0:
                    zpos += random.uniform(-gap * 0.3, gap * 0.3)

            zpos = max(zpos, parent_region.z_min + groove_margin)
            zpos = min(zpos, parent_region.z_max - groove_width - groove_margin)

            if zpos < parent_region.z_min or zpos + groove_width > parent_region.z_max:
                continue

            node.region = Region(
                z_min=zpos,
                z_max=zpos + groove_width,
                radius=new_radius,
                direction=parent_region.direction,
            )

            print(
                f"    Groove: z=[{zpos:.2f}, {zpos + groove_width:.2f}], "
                f"r={new_radius:.2f}, w={groove_width:.2f}"
            )

            return TurningFeatureRequest(
                feature_type='groove',
                z_min=zpos,
                z_max=zpos + groove_width,
                outer_radius=parent_region.radius,
                inner_radius=new_radius,
                label=Labels.GROOVE,
            )

        return None
