"""
트리 기반 터닝 모델 생성기 (Bottom-Up 방식)

트리 구조로 터닝 형상의 계층적 포함 관계를 표현:
- b (base): 루트 노드 - Stock 원기둥
- s (step): 단차 가공 - 자식 노드는 step 영역 내에 포함
- g (groove): 홈 가공 - 자식 노드는 groove 영역 내에 포함

Bottom-Up 방식:
1. 리프 노드부터 필요한 크기 계산 (위로 전파)
2. 루트에서 Stock 크기 결정 (자식들을 수용할 수 있는 크기)
3. 루트에서 아래로 실제 영역 할당 및 형상 생성
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Extend.TopologyUtils import TopologyExplorer

from utils.step_io import save_step


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TurningParams:
    """터닝 형상 파라미터"""
    # Stock 기본 범위 (Bottom-Up에서는 필요 크기에 맞춰 조정됨)
    stock_height_margin: Tuple[float, float] = (3.0, 8.0)    # 필요 높이에 추가할 여유
    stock_radius_margin: Tuple[float, float] = (2.0, 5.0)    # 필요 반경에 추가할 여유
    
    # Step 파라미터
    step_depth_range: Tuple[float, float] = (0.8, 1.5)       # step 깊이 (반경 방향)
    step_height_range: Tuple[float, float] = (2.0, 4.0)      # step 자체 높이
    step_margin: float = 0.5                                  # step 위아래 여유
    
    # Groove 파라미터
    groove_depth_range: Tuple[float, float] = (0.4, 0.8)     # groove 깊이
    groove_width_range: Tuple[float, float] = (1.5, 3.0)     # groove 폭
    groove_margin_ratio: float = 0.15                         # groove 양쪽 여유 비율 (부모 영역 대비)
    
    min_remaining_radius: float = 2.0
    
    # 챔퍼/라운드 파라미터
    chamfer_range: Tuple[float, float] = (0.3, 0.8)
    fillet_range: Tuple[float, float] = (0.3, 0.8)
    edge_feature_prob: float = 0.3  # 각 엣지에 챔퍼/라운드 적용 확률


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
        return f"Region(z=[{self.z_min:.2f}, {self.z_max:.2f}], r={self.radius:.2f}, dir={self.direction})"


@dataclass
class RequiredSpace:
    """Bottom-Up으로 계산된 필요 공간 (margin 포함, 한 곳에서만 계산)"""
    height: float           # 필요한 z 높이 (margin 포함 총 크기)
    depth: float            # 필요한 반경 깊이 (누적)
    feature_height: float   # 자신의 피처 높이 (step height 또는 groove 커팅 폭)
    feature_depth: float    # 자신의 피처 깊이
    margin: float = 0.0     # 자신의 z방향 margin (양쪽 각각)


class TreeNode:
    """트리 노드 클래스"""
    def __init__(self, node_id: int, label: str, parent_id: Optional[int], depth: int):
        self.id = node_id
        self.label = label
        self.parent_id = parent_id
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.region: Optional[Region] = None
        self.required_space: Optional[RequiredSpace] = None  # Bottom-Up 계산된 필요 공간
        
    def __repr__(self):
        return f"TreeNode(id={self.id}, label='{self.label}', depth={self.depth})"


# ============================================================================
# TreeTurningGenerator Class
# ============================================================================

class TreeTurningGenerator:
    """
    트리 기반 터닝 모델 생성기 - 계층적 포함 관계 구현.
    
    사용법:
        generator = TreeTurningGenerator(TurningParams())
        shape = generator.generate_from_tree(tree_data)
        generator.save("output.step")
    """
    
    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self.shape: Optional[TopoDS_Shape] = None
        self.stock_height: float = 0.0
        self.stock_radius: float = 0.0
        
    def load_tree(self, tree_data: Dict) -> TreeNode:
        """JSON 트리 데이터를 TreeNode 구조로 변환."""
        nodes_data = tree_data['nodes']
        
        nodes: Dict[int, TreeNode] = {}
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
            else:
                root = node
                
        return root
    
    def _create_stock(self) -> TopoDS_Shape:
        """Stock 원기둥 생성."""
        self.stock_height = random.uniform(*self.params.stock_height_range)
        self.stock_radius = random.uniform(*self.params.stock_radius_range)
        
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        self.shape = BRepPrimAPI_MakeCylinder(axis, self.stock_radius, self.stock_height).Shape()
        
        return self.shape
    
    def _collect_circular_edges(self) -> List[TopoDS_Edge]:
        """원형 엣지들 수집 (터닝에서 챔퍼/라운드 적용 가능한 엣지)"""
        circular_edges = []
        
        try:
            topo_explorer = TopologyExplorer(self.shape)
            for edge in topo_explorer.edges():
                try:
                    adaptor = BRepAdaptor_Curve(edge)
                    if adaptor.GetType() == GeomAbs_Circle:
                        circular_edges.append(edge)
                except (RuntimeError, TypeError) as e:
                    # OCC 라이브러리에서 발생할 수 있는 예외만 처리
                    pass
        except Exception as e:
            print(f"    엣지 수집 실패: {e}")
        
        return circular_edges
    
    def _apply_chamfer(self, edge: TopoDS_Edge, distance: float) -> bool:
        """엣지에 챔퍼 적용."""
        try:
            builder = BRepFilletAPI_MakeChamfer(self.shape)
            builder.Add(distance, edge)
            builder.Build()
            
            if builder.IsDone():
                new_shape = builder.Shape()
                if new_shape and not new_shape.IsNull():
                    self.shape = new_shape
                    return True
        except Exception:
            pass
        return False
    
    def _apply_fillet(self, edge: TopoDS_Edge, radius: float) -> bool:
        """엣지에 라운드(필렛) 적용."""
        try:
            builder = BRepFilletAPI_MakeFillet(self.shape)
            builder.Add(radius, edge)
            builder.Build()
            
            if builder.IsDone():
                new_shape = builder.Shape()
                if new_shape and not new_shape.IsNull():
                    self.shape = new_shape
                    return True
        except Exception:
            pass
        return False
    
    def _apply_edge_features(self):
        """원형 엣지들에 랜덤하게 챔퍼/라운드 적용."""
        try:
            circular_edges = self._collect_circular_edges()
            
            if not circular_edges:
                print(f"    원형 엣지 없음, 스킵")
                return
            
            print(f"    원형 엣지 {len(circular_edges)}개 발견")
            
            applied_count = 0
            for edge in circular_edges:
                if random.random() < self.params.edge_feature_prob:
                    feature_type = random.choice(['chamfer', 'fillet'])
                    
                    if feature_type == 'chamfer':
                        distance = random.uniform(*self.params.chamfer_range)
                        if self._apply_chamfer(edge, distance):
                            applied_count += 1
                            print(f"    Chamfer: d={distance:.2f}")
                    else:
                        radius = random.uniform(*self.params.fillet_range)
                        if self._apply_fillet(edge, radius):
                            applied_count += 1
                            print(f"    Fillet: r={radius:.2f}")
            
            if applied_count > 0:
                print(f"    총 {applied_count}개 엣지 피처 적용")
        except Exception as e:
            print(f"    엣지 피처 적용 중 오류: {e}")
    
    # =========================================================================
    # Bottom-Up: 필요 공간 계산 (리프 → 루트)
    # =========================================================================
    
    def _calculate_required_space(self, node: TreeNode) -> RequiredSpace:
        """
        Bottom-Up으로 노드가 필요로 하는 공간 계산.
        리프 노드부터 시작해서 위로 전파.
        
        Args:
            node: 계산할 노드
            
        Returns:
            필요한 공간 정보
        """
        # 1. 먼저 모든 자식의 필요 공간을 재귀적으로 계산
        for child in node.children:
            self._calculate_required_space(child)
        
        # 2. 자식들의 필요 높이/깊이 합산
        # Step 자식과 Groove 자식을 구분 (Groove는 부모 영역 내부에 배치되므로 추가 공간 불필요)
        step_children = [c for c in node.children if c.label == 's']
        groove_children = [c for c in node.children if c.label == 'g']
        
        step_children_height = sum(
            c.required_space.height for c in step_children
        ) if step_children else 0.0
        
        groove_children_height = sum(
            c.required_space.height for c in groove_children
        ) if groove_children else 0.0
        
        # 깊이는 모든 자식의 최대값 (반경 방향 누적)
        children_max_depth = max(
            (c.required_space.depth for c in node.children), default=0.0
        )
        
        # 3. 노드 타입에 따른 자신의 피처 크기와 margin 결정
        #    ※ margin은 여기서 한 번만 계산하고 RequiredSpace에 저장
        #    ※ 배치 시에는 저장된 margin을 그대로 사용 (재계산 없음)
        
        if node.label == 'b':
            # Base (Stock): margin 없음 (Stock 자체가 최외곽)
            feature_height = 0.0
            feature_depth = 0.0
            node_margin = 0.0
            total_height = max(step_children_height, groove_children_height)
            total_depth = children_max_depth
            
        elif node.label == 's':
            # Step: 고정 margin
            feature_height = random.uniform(*self.params.step_height_range)
            feature_depth = random.uniform(*self.params.step_depth_range)
            node_margin = self.params.step_margin
            
            step_based_height = step_children_height + feature_height + 2 * node_margin
            # Groove 자식도 수용 가능하도록 보장
            total_height = max(step_based_height, groove_children_height)
            total_depth = children_max_depth + feature_depth
            
        elif node.label == 'g':
            # Groove: 비율 기반 margin (feature_height의 groove_margin_ratio)
            random_width = random.uniform(*self.params.groove_width_range)
            feature_depth = random.uniform(*self.params.groove_depth_range)
            ratio = self.params.groove_margin_ratio
            
            # 자식 groove들을 내부에 수용할 수 있도록 커팅 폭 결정
            if groove_children:
                # feature_height 내에서 자식 배치: 사용 가능 = feature_height - 2*margin
                # margin = feature_height * ratio이므로 사용 가능 = feature_height * (1-2r)
                # 자식 총 필요 = groove_children_height
                # → feature_height >= groove_children_height / (1-2r)
                usable_ratio = 1.0 - 2.0 * ratio
                if usable_ratio <= 0:
                    usable_ratio = 0.5
                min_width = groove_children_height / usable_ratio
                feature_height = max(random_width, min_width)
            else:
                feature_height = random_width
            
            node_margin = feature_height * ratio
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
            margin=node_margin
        )
        
        return node.required_space
    
    def _create_stock_from_requirements(self, root: TreeNode) -> TopoDS_Shape:
        """
        Bottom-Up 계산 결과를 기반으로 Stock 생성.
        """
        required = root.required_space
        
        # 필요 크기 + 여유 공간
        height_margin = random.uniform(*self.params.stock_height_margin)
        radius_margin = random.uniform(*self.params.stock_radius_margin)
        
        self.stock_height = required.height + height_margin
        self.stock_radius = required.depth + self.params.min_remaining_radius + radius_margin
        
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        self.shape = BRepPrimAPI_MakeCylinder(axis, self.stock_radius, self.stock_height).Shape()
        
        return self.shape
    
    # =========================================================================
    # Bottom-Up: 형상 생성 (루트 → 리프, 공간 보장됨)
    # =========================================================================
    
    def _apply_step_bottomup(
        self, 
        node: TreeNode, 
        parent_region: Region, 
        direction: str
    ) -> Optional[Region]:
        """
        Bottom-Up 방식으로 Step 적용.
        자식들의 필요 공간이 이미 계산되어 있으므로 항상 성공.
        
        Args:
            node: Step 노드
            parent_region: 부모 영역
            direction: 'top' or 'bottom'
            
        Returns:
            Step 영역 (자식들이 사용할 영역)
        """
        required = node.required_space
        step_height = required.feature_height
        step_depth = required.feature_depth
        
        # 반경 검증
        if parent_region.radius - step_depth < self.params.min_remaining_radius:
            step_depth = parent_region.radius - self.params.min_remaining_radius - 0.3
            if step_depth <= 0:
                return None
        
        new_radius = parent_region.radius - step_depth
        
        # 방향에 따른 z 위치 결정
        # required.height는 Step 자식 + Groove 자식 모두 수용하는 크기
        # (Bottom-Up에서 max(step_based, groove_children)로 계산됨)
        cut_size = required.height
        
        if direction == 'top':
            # Top: 부모 상단에서 시작, 아래로 확장
            cut_z_max = parent_region.z_max
            cut_z_min = cut_z_max - cut_size
            cut_z_min = max(cut_z_min, parent_region.z_min)  # 범위 제한
            
            actual_step_height = cut_z_max - cut_z_min
        else:
            # Bottom: 부모 하단에서 시작, 위로 확장
            cut_z_min = parent_region.z_min
            cut_z_max = cut_z_min + cut_size
            cut_z_max = min(cut_z_max, parent_region.z_max)  # 범위 제한
            
            actual_step_height = cut_z_max - cut_z_min
        
        if actual_step_height <= 0:
            return None
        
        # Boolean Cut 수행
        axis = gp_Ax2(gp_Pnt(0, 0, cut_z_min), gp_Dir(0, 0, 1))
        outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, actual_step_height).Shape()
        inner = BRepPrimAPI_MakeCylinder(axis, new_radius, actual_step_height).Shape()
        
        cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
        self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
        
        new_region = Region(
            z_min=cut_z_min,
            z_max=cut_z_max,
            radius=new_radius,
            direction=direction
        )
        
        print(f"    Step ({direction}): z=[{new_region.z_min:.2f}, {new_region.z_max:.2f}], "
              f"r={new_radius:.2f}, h={actual_step_height:.2f}")
        return new_region
    
    def _apply_groove_bottomup(
        self, 
        node: TreeNode, 
        parent_region: Region,
        groove_index: int = 0,
        total_grooves: int = 1
    ) -> Optional[Region]:
        """
        Bottom-Up 방식으로 Groove 적용.
        
        Args:
            node: Groove 노드
            parent_region: 부모 영역
            groove_index: 형제 groove 중 인덱스 (0부터 시작)
            total_grooves: 형제 groove 총 개수
            
        Returns:
            Groove 영역 (자식들이 사용할 영역)
        """
        required = node.required_space
        groove_width = required.feature_height  # groove에서는 width가 feature_height
        groove_depth = required.feature_depth
        
        # 반경 검증
        if parent_region.radius - groove_depth < self.params.min_remaining_radius:
            groove_depth = parent_region.radius - self.params.min_remaining_radius - 0.2
            if groove_depth <= 0:
                return None
        
        new_radius = parent_region.radius - groove_depth
        
        # z 위치: 형제 groove들을 Z축으로 분산 배치
        # margin은 bottom-up에서 이미 계산됨 (재계산 없음)
        margin = required.margin
        available_height = parent_region.height - 2 * margin
        
        if total_grooves == 1:
            # 단일 groove: 중앙 배치
            center_z = (parent_region.z_min + parent_region.z_max) / 2
            zpos = center_z - groove_width / 2
        else:
            # 여러 groove: Z축으로 분산 배치
            total_groove_height = sum(
                c.required_space.height for c in node.parent_node.children 
                if c.label == 'g'
            ) if hasattr(node, 'parent_node') and node.parent_node else required.height
            
            spacing = available_height / total_grooves
            zpos = parent_region.z_min + margin + groove_index * spacing
        
        # 범위 제한
        zpos = max(zpos, parent_region.z_min + margin)
        zpos = min(zpos, parent_region.z_max - groove_width - margin)
        
        # Boolean Cut 수행
        axis = gp_Ax2(gp_Pnt(0, 0, zpos), gp_Dir(0, 0, 1))
        outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, groove_width).Shape()
        inner = BRepPrimAPI_MakeCylinder(axis, new_radius, groove_width).Shape()
        
        cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
        self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
        
        new_region = Region(
            z_min=zpos,
            z_max=zpos + groove_width,
            radius=new_radius,
            direction=parent_region.direction
        )
        
        print(f"    Groove: z=[{zpos:.2f}, {zpos + groove_width:.2f}], r={new_radius:.2f}")
        return new_region
    
    def _process_node_bottomup(
        self, 
        node: TreeNode, 
        parent_region: Region,
        direction: str = None,
        groove_index: int = 0,
        total_grooves: int = 1
    ):
        """
        Bottom-Up 방식으로 노드 처리.
        공간이 이미 계산되어 있으므로 스킵 없이 처리.
        
        Args:
            node: 처리할 노드
            parent_region: 부모 영역 (None이면 Stock)
            direction: Step의 경우 방향
            groove_index: Groove의 경우 형제 인덱스
            total_grooves: Groove의 경우 형제 총 개수
        """
        if node.label == 'b':
            # Base (Stock) 노드
            node.region = Region(
                z_min=0,
                z_max=self.stock_height,
                radius=self.stock_radius,
                direction=None
            )
            
            # 자식들 처리 (Step은 top/bottom 번갈아, Groove는 분산 배치)
            step_children = [c for c in node.children if c.label == 's']
            groove_children = [c for c in node.children if c.label == 'g']
            
            for i, child in enumerate(step_children):
                child_dir = 'top' if i % 2 == 0 else 'bottom'
                self._process_node_bottomup(child, node.region, child_dir)
            
            # Groove는 인덱스 정보와 함께 처리
            for i, child in enumerate(groove_children):
                child.parent_node = node  # 부모 참조 저장
                self._process_node_bottomup(child, node.region, None, i, len(groove_children))
                
        elif node.label == 's':
            # Step 적용
            new_region = self._apply_step_bottomup(node, parent_region, direction or 'top')
            
            if new_region is None:
                node.region = parent_region
                print(f"    [Warning] Step 적용 실패")
            else:
                node.region = new_region
            
            # 자식들 처리
            step_children = [c for c in node.children if c.label == 's']
            groove_children = [c for c in node.children if c.label == 'g']
            
            for child in step_children:
                # Step 자식은 같은 방향으로
                self._process_node_bottomup(child, node.region, direction)
            
            # Groove는 인덱스 정보와 함께 처리
            for i, child in enumerate(groove_children):
                child.parent_node = node  # 부모 참조 저장
                self._process_node_bottomup(child, node.region, None, i, len(groove_children))
                
        elif node.label == 'g':
            # Groove 적용 (인덱스 정보 사용)
            new_region = self._apply_groove_bottomup(node, parent_region, groove_index, total_grooves)
            
            if new_region is None:
                node.region = parent_region
                print(f"    [Warning] Groove 적용 실패")
            else:
                node.region = new_region
            
            # Groove의 자식은 Groove만 가능 (트리 생성기 규칙)
            # 자식 groove들도 인덱스 정보로 분산 배치
            groove_children = [c for c in node.children if c.label == 'g']
            for i, child in enumerate(groove_children):
                child.parent_node = node  # 부모 참조 저장
                self._process_node_bottomup(child, node.region, None, i, len(groove_children))
    
    # =========================================================================
    # Legacy functions (하위 호환성)
    # =========================================================================
    
    def _apply_step(self, parent_region: Region, direction: str) -> Optional[Region]:
        """
        Step 적용 - 부모 영역 내에서 새로운 step 영역 생성.
        (하위 호환성을 위해 유지, 내부적으로 allocation 사용)
        
        Args:
            parent_region: 부모 노드의 영역
            direction: 'top' (위에서 깎음) or 'bottom' (아래에서 깎음)
            
        Returns:
            새로 생성된 step 영역 (자식 노드들이 사용할 영역)
        """
        max_step_height = parent_region.height - self.params.min_base_height
        if max_step_height < self.params.step_height_range[0]:
            return None
        
        step_height = random.uniform(
            self.params.step_height_range[0],
            min(self.params.step_height_range[1], max_step_height)
        )
        step_depth = random.uniform(*self.params.step_depth_range)
        
        if parent_region.radius - step_depth < self.params.min_remaining_radius:
            step_depth = parent_region.radius - self.params.min_remaining_radius - 0.5
            # 음수 또는 너무 작은 깊이 방지
            if step_depth <= 0 or step_depth <= 0.3:
                return None
        
        new_radius = parent_region.radius - step_depth
        
        if direction == 'top':
            cut_z_min = parent_region.z_max - step_height
            cut_z_max = parent_region.z_max
            
            axis = gp_Ax2(gp_Pnt(0, 0, cut_z_min), gp_Dir(0, 0, 1))
            outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, step_height).Shape()
            inner = BRepPrimAPI_MakeCylinder(axis, new_radius, step_height).Shape()
            
            cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
            self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
            
            new_region = Region(
                z_min=cut_z_min,
                z_max=cut_z_max,
                radius=new_radius,
                direction='top'
            )
            
        else:  # bottom
            cut_z_min = parent_region.z_min
            cut_z_max = parent_region.z_min + step_height
            
            axis = gp_Ax2(gp_Pnt(0, 0, cut_z_min), gp_Dir(0, 0, 1))
            outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, step_height).Shape()
            inner = BRepPrimAPI_MakeCylinder(axis, new_radius, step_height).Shape()
            
            cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
            self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
            
            new_region = Region(
                z_min=cut_z_min,
                z_max=cut_z_max,
                radius=new_radius,
                direction='bottom'
            )
        
        print(f"    Step ({direction}): z=[{new_region.z_min:.2f}, {new_region.z_max:.2f}], r={new_radius:.2f}")
        return new_region
    
    def _apply_groove(self, parent_region: Region) -> Optional[Region]:
        """
        Groove 적용 - 부모 영역 내에서 groove 생성.
        (하위 호환성을 위해 유지)
        
        Returns:
            groove가 차지하는 영역 (자식 노드들이 사용할 영역)
        """
        margin = self.params.min_base_height
        max_groove_width = parent_region.height - 2 * margin
        if max_groove_width < self.params.groove_width_range[0]:
            return None
        
        groove_width = random.uniform(
            self.params.groove_width_range[0],
            min(self.params.groove_width_range[1], max_groove_width)
        )
        groove_depth = random.uniform(*self.params.groove_depth_range)
        if parent_region.radius - groove_depth < self.params.min_remaining_radius:
            groove_depth = parent_region.radius - self.params.min_remaining_radius - 0.3
            # 음수 또는 너무 작은 깊이 방지
            if groove_depth <= 0 or groove_depth <= 0.2:
                return None
        
        available_range = parent_region.height - groove_width - 2 * margin
        if available_range <= 0:
            zpos = parent_region.z_min + margin
        else:
            zpos = parent_region.z_min + margin + random.uniform(0, available_range)
        
        new_radius = parent_region.radius - groove_depth
        
        axis = gp_Ax2(gp_Pnt(0, 0, zpos), gp_Dir(0, 0, 1))
        outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, groove_width).Shape()
        inner = BRepPrimAPI_MakeCylinder(axis, new_radius, groove_width).Shape()
        
        cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
        self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
        
        new_region = Region(
            z_min=zpos,
            z_max=zpos + groove_width,
            radius=new_radius,
            direction=parent_region.direction
        )
        
        print(f"    Groove: z=[{zpos:.2f}, {zpos + groove_width:.2f}], r={new_radius:.2f}")
        return new_region
    
    def generate_from_tree(self, tree_data: Dict, apply_edge_features: bool = True) -> TopoDS_Shape:
        """
        트리 데이터로부터 터닝 모델 생성 (Bottom-Up 방식, 2단계 처리).
        
        처리 순서:
        1. 필요 공간 계산 (Bottom-Up)
        2. Stock 생성
        3. Step만 먼저 처리 (Boolean Cut이 충돌하지 않도록)
        4. Groove 처리 (Step 완료 후)
        5. 챔퍼/라운드 적용
        
        Args:
            tree_data: 트리 JSON 데이터
            apply_edge_features: 챔퍼/라운드 적용 여부
            
        Returns:
            생성된 TopoDS_Shape
        """
        root = self.load_tree(tree_data)
        
        # 1. Bottom-Up: 필요한 공간 계산 (리프 → 루트)
        self._calculate_required_space(root)
        
        # 2. 계산된 필요 크기로 Stock 생성
        self._create_stock_from_requirements(root)
        print(f"  Stock: height={self.stock_height:.2f}, radius={self.stock_radius:.2f} "
              f"(required: h={root.required_space.height:.2f}, d={root.required_space.depth:.2f})")
        
        # 3. 2단계 처리: Step 먼저 → Groove 나중
        # 3-1. Step만 먼저 처리하여 모든 region 확정
        self._process_steps_only(root, None)
        
        # 3-2. Groove 처리 (Step 완료 후, 형상이 안정된 상태에서)
        groove_count = self._process_grooves_only(root)
        expected_grooves = self._count_grooves_in_tree(root)
        
        if groove_count < expected_grooves:
            print(f"    [Warning] Groove 생성: {groove_count}/{expected_grooves}")
        
        # 4. 챔퍼/라운드 적용 (옵션)
        if apply_edge_features:
            self._apply_edge_features()
        
        return self.shape
    
    def _process_steps_only(self, node: TreeNode, parent_region: Region, direction: str = None):
        """
        Step 노드만 처리하여 모든 region 확정.
        Groove는 건너뛰고 나중에 별도 처리.
        """
        if node.label == 'b':
            # Base (Stock) 노드
            node.region = Region(
                z_min=0,
                z_max=self.stock_height,
                radius=self.stock_radius,
                direction=None
            )
            
            # Step 자식만 처리 (top/bottom 번갈아)
            step_children = [c for c in node.children if c.label == 's']
            for i, child in enumerate(step_children):
                child_dir = 'top' if i % 2 == 0 else 'bottom'
                self._process_steps_only(child, node.region, child_dir)
            
            # Groove 자식의 region만 부모로 설정 (나중에 처리)
            groove_children = [c for c in node.children if c.label == 'g']
            for child in groove_children:
                child.region = node.region  # 임시로 부모 region 할당
                child.parent_node = node
                self._assign_groove_regions(child)
                
        elif node.label == 's':
            # Step 적용
            new_region = self._apply_step_bottomup(node, parent_region, direction or 'top')
            
            if new_region is None:
                node.region = parent_region
                print(f"    [Warning] Step 적용 실패")
            else:
                node.region = new_region
            
            # Step 자식 재귀 처리
            step_children = [c for c in node.children if c.label == 's']
            for child in step_children:
                self._process_steps_only(child, node.region, direction)
            
            # Groove 자식의 region만 부모로 설정 (나중에 처리)
            groove_children = [c for c in node.children if c.label == 'g']
            for child in groove_children:
                child.region = node.region  # 임시로 Step의 region 할당
                child.parent_node = node
                self._assign_groove_regions(child)
    
    def _assign_groove_regions(self, node: TreeNode):
        """Groove 노드와 그 자식들에게 부모 region 할당 (나중 처리용)"""
        for child in node.children:
            if child.label == 'g':
                child.region = node.region
                child.parent_node = node
                self._assign_groove_regions(child)
    
    def _process_grooves_only(self, node: TreeNode, parent_groove_region: Region = None) -> int:
        """
        모든 Groove 노드 처리 (Step 처리 완료 후).
        DFS로 모든 노드 순회하며 Groove만 처리.
        
        Returns:
            성공적으로 생성된 Groove 개수
        """
        groove_count = 0
        
        if node.label == 'g':
            # Groove 적용 (부모의 region 사용)
            parent_region = parent_groove_region if parent_groove_region else node.region
            
            # Groove 인덱스 계산 (형제들 중에서)
            groove_index = 0
            total_grooves = 1
            if hasattr(node, 'parent_node') and node.parent_node:
                siblings = [c for c in node.parent_node.children if c.label == 'g']
                total_grooves = len(siblings)
                for i, sib in enumerate(siblings):
                    if sib.id == node.id:
                        groove_index = i
                        break
            
            new_region = self._apply_groove_with_validation(
                node, parent_region, groove_index, total_grooves
            )
            
            if new_region is not None:
                node.region = new_region
                groove_count += 1
            else:
                print(f"    [Warning] Groove 적용 실패 (node_id={node.id})")
            
            # Groove 자식 처리 (중첩 Groove)
            for child in node.children:
                if child.label == 'g':
                    groove_count += self._process_grooves_only(child, node.region)
        else:
            # Base나 Step인 경우, 자식들 중 Groove 처리
            for child in node.children:
                if child.label == 'g':
                    groove_count += self._process_grooves_only(child, node.region)
                elif child.label == 's':
                    groove_count += self._process_grooves_only(child, None)
                elif child.label == 'b':
                    groove_count += self._process_grooves_only(child, None)
        
        return groove_count
    
    def _apply_groove_with_validation(
        self, 
        node: TreeNode, 
        parent_region: Region,
        groove_index: int = 0,
        total_grooves: int = 1,
        max_retries: int = 3
    ) -> Optional[Region]:
        """
        Groove 적용 (검증 포함, 실패 시 재시도).
        
        형제 groove들은 순차적으로 배치되어 겹치지 않도록 보장.
        groove_width가 부모 영역을 초과하면 축소 적용.
        """
        required = node.required_space
        groove_width = required.feature_height
        groove_depth = required.feature_depth
        
        # 반경 검증
        if parent_region.radius - groove_depth < self.params.min_remaining_radius:
            groove_depth = parent_region.radius - self.params.min_remaining_radius - 0.2
            if groove_depth <= 0:
                return None
        
        new_radius = parent_region.radius - groove_depth
        # margin은 bottom-up에서 이미 계산되어 저장됨 (재계산 없음)
        groove_margin = required.margin
        
        for attempt in range(max_retries):
            # Z 위치 계산
            if total_grooves == 1:
                # 단일 groove: 중앙 배치
                center_z = (parent_region.z_min + parent_region.z_max) / 2
                zpos = center_z - groove_width / 2
                zpos = max(zpos, parent_region.z_min + groove_margin)
                zpos = min(zpos, parent_region.z_max - groove_width - groove_margin)
            else:
                # 형제 groove: zone 기반 순차 배치 (겹침 + 단차쌍 누락 방지)
                # 각 groove의 zone = margin + width + margin = height (bottom-up에서 계산됨)
                if hasattr(node, 'parent_node') and node.parent_node:
                    siblings = [c for c in node.parent_node.children if c.label == 'g']
                    sibling_zones = [sib.required_space.height for sib in siblings]
                    sibling_margins = [sib.required_space.margin for sib in siblings]
                    sibling_widths = [sib.required_space.feature_height for sib in siblings]
                else:
                    sibling_zones = [required.height] * total_grooves
                    sibling_margins = [groove_margin] * total_grooves
                    sibling_widths = [groove_width] * total_grooves
                
                total_zone = sum(sibling_zones)
                
                # 전체 zone이 부모 영역 초과 시 비례 축소
                if total_zone > parent_region.height:
                    scale = parent_region.height / total_zone
                    sibling_zones = [z * scale for z in sibling_zones]
                    sibling_margins = [m * scale for m in sibling_margins]
                    sibling_widths = [w * scale for w in sibling_widths]
                    groove_width = sibling_widths[groove_index]
                    groove_margin = sibling_margins[groove_index]
                    total_zone = parent_region.height
                
                # 여유 공간을 균등 gap으로 분배
                remaining = parent_region.height - total_zone
                gap = remaining / (total_grooves + 1) if total_grooves > 0 else 0
                
                # 순차 누적: gap + [margin + width + margin] + gap + ...
                zpos = parent_region.z_min + gap
                for i in range(groove_index):
                    zpos += sibling_zones[i] + gap
                zpos += sibling_margins[groove_index]  # leading margin
                
                # 재시도 시 gap 범위 내에서 미세 조정
                if attempt > 0 and gap > 0:
                    offset = random.uniform(-gap * 0.3, gap * 0.3)
                    zpos += offset
            
            # 범위 검증
            zpos = max(zpos, parent_region.z_min + groove_margin)
            zpos = min(zpos, parent_region.z_max - groove_width - groove_margin)
            
            if zpos < parent_region.z_min or zpos + groove_width > parent_region.z_max:
                continue
            
            # Boolean Cut 수행
            try:
                axis = gp_Ax2(gp_Pnt(0, 0, zpos), gp_Dir(0, 0, 1))
                outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, groove_width).Shape()
                inner = BRepPrimAPI_MakeCylinder(axis, new_radius, groove_width).Shape()
                
                cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
                
                if cut_shape.IsNull():
                    continue
                
                new_shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
                
                if new_shape.IsNull():
                    continue
                
                # 형상 업데이트
                self.shape = new_shape
                
                new_region = Region(
                    z_min=zpos,
                    z_max=zpos + groove_width,
                    radius=new_radius,
                    direction=parent_region.direction
                )
                
                print(f"    Groove: z=[{zpos:.2f}, {zpos + groove_width:.2f}], "
                      f"r={new_radius:.2f}, w={groove_width:.2f}")
                return new_region
                
            except Exception as e:
                print(f"    [Retry {attempt+1}] Groove Boolean Cut 실패: {e}")
                continue
        
        return None
    
    def _count_grooves_in_tree(self, node: TreeNode) -> int:
        """트리에서 Groove 노드 개수 계산"""
        count = 1 if node.label == 'g' else 0
        for child in node.children:
            count += self._count_grooves_in_tree(child)
        return count
    
    def save(self, filepath: str) -> bool:
        """생성된 형상을 STEP 파일로 저장."""
        if self.shape is None or self.shape.IsNull():
            print("저장할 형상이 없습니다.")
            return False
        return save_step(self.shape, filepath)
