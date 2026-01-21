"""
트리 기반 터닝 모델 생성기
- b (base): 루트 노드 - Stock 원기둥
- s (step): 단차 가공 - 자식 노드는 step 영역 내에 포함
- g (groove): 홈 가공 - 자식 노드는 groove 영역 내에 포함

트리 구조 = 계층적 포함 관계
+ 챔퍼/라운드 엣지 처리
"""
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# OCC imports
from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Extend.TopologyUtils import TopologyExplorer


@dataclass
class TurningParams:
    """터닝 형상 파라미터"""
    stock_height_range: Tuple[float, float] = (15.0, 20.0)
    stock_radius_range: Tuple[float, float] = (6.5, 8.5)
    
    step_depth_range: Tuple[float, float] = (0.8, 1.5)      # step 깊이 (반경 방향)
    step_height_range: Tuple[float, float] = (2.5, 6.5)     # step 높이 (z 방향)
    
    groove_depth_range: Tuple[float, float] = (0.5, 1.0)    # groove 깊이
    groove_width_range: Tuple[float, float] = (2.5, 6.5)    # groove 폭
    
    min_remaining_radius: float = 1.5
    min_base_height: float = 1.0  # Step 적용 후 남아야 하는 최소 높이
    
    # 챔퍼/라운드 파라미터 (안전한 범위로 조정)
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


class TreeNode:
    """트리 노드 클래스"""
    def __init__(self, node_id: int, label: str, parent_id: Optional[int], depth: int):
        self.id = node_id
        self.label = label
        self.parent_id = parent_id
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.region: Optional[Region] = None  # 이 노드가 차지하는 영역
        
    def __repr__(self):
        return f"TreeNode(id={self.id}, label='{self.label}', depth={self.depth})"


class TreeTurningGenerator:
    """트리 기반 터닝 모델 생성기 - 계층적 포함 관계 구현"""
    
    def __init__(self, params: TurningParams = None):
        self.params = params or TurningParams()
        self.shape: Optional[TopoDS_Shape] = None
        self.stock_height: float = 0.0
        self.stock_radius: float = 0.0
        
    def load_tree(self, tree_data: Dict) -> TreeNode:
        """JSON 트리 데이터를 TreeNode 구조로 변환"""
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
        """Stock 원기둥 생성"""
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
                except:
                    pass
        except Exception as e:
            print(f"    엣지 수집 실패: {e}")
        
        return circular_edges
    
    def _apply_chamfer(self, edge: TopoDS_Edge, distance: float) -> bool:
        """엣지에 챔퍼 적용"""
        try:
            builder = BRepFilletAPI_MakeChamfer(self.shape)
            builder.Add(distance, edge)
            builder.Build()
            
            if builder.IsDone():
                new_shape = builder.Shape()
                if new_shape and not new_shape.IsNull():
                    self.shape = new_shape
                    return True
        except Exception as e:
            pass
        return False
    
    def _apply_fillet(self, edge: TopoDS_Edge, radius: float) -> bool:
        """엣지에 라운드(필렛) 적용"""
        try:
            builder = BRepFilletAPI_MakeFillet(self.shape)
            builder.Add(radius, edge)
            builder.Build()
            
            if builder.IsDone():
                new_shape = builder.Shape()
                if new_shape and not new_shape.IsNull():
                    self.shape = new_shape
                    return True
        except Exception as e:
            pass
        return False
    
    def _apply_edge_features(self):
        """원형 엣지들에 랜덤하게 챔퍼/라운드 적용"""
        try:
            circular_edges = self._collect_circular_edges()
            
            if not circular_edges:
                print(f"    원형 엣지 없음, 스킵")
                return
            
            print(f"    원형 엣지 {len(circular_edges)}개 발견")
            
            # 일부 엣지에만 적용
            applied_count = 0
            for edge in circular_edges:
                if random.random() < self.params.edge_feature_prob:
                    # 챔퍼 또는 라운드 랜덤 선택
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
    
    def _apply_step(self, parent_region: Region, direction: str) -> Optional[Region]:
        """Step 적용 - 부모 영역 내에서 새로운 step 영역 생성
        
        Args:
            parent_region: 부모 노드의 영역
            direction: 'top' (위에서 깎음) or 'bottom' (아래에서 깎음)
            
        Returns:
            새로 생성된 step 영역 (자식 노드들이 사용할 영역)
        """
        # 최대 step 높이: 부모 영역에서 min_base_height를 남겨야 함
        max_step_height = parent_region.height - self.params.min_base_height
        if max_step_height < self.params.step_height_range[0]:
            return None  # 공간 부족
        
        step_height = random.uniform(
            self.params.step_height_range[0],
            min(self.params.step_height_range[1], max_step_height)
        )
        step_depth = random.uniform(*self.params.step_depth_range)
        
        # 반경 유효성 검사
        if parent_region.radius - step_depth < self.params.min_remaining_radius:
            step_depth = parent_region.radius - self.params.min_remaining_radius - 0.5
            if step_depth <= 0.3:
                return None
        
        new_radius = parent_region.radius - step_depth
        
        if direction == 'top':
            # 상단에서 깎음: z_max 쪽에서 step_height만큼
            cut_z_min = parent_region.z_max - step_height
            cut_z_max = parent_region.z_max
            
            # Cut shape: 도넛 형태
            axis = gp_Ax2(gp_Pnt(0, 0, cut_z_min), gp_Dir(0, 0, 1))
            outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, step_height).Shape()
            inner = BRepPrimAPI_MakeCylinder(axis, new_radius, step_height).Shape()
            
            cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
            self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
            
            # 새 영역: step이 깎인 상단 부분
            new_region = Region(
                z_min=cut_z_min,
                z_max=cut_z_max,
                radius=new_radius,
                direction='top'
            )
            
        else:  # bottom
            # 하단에서 깎음: z_min 쪽에서 step_height만큼
            cut_z_min = parent_region.z_min
            cut_z_max = parent_region.z_min + step_height
            
            axis = gp_Ax2(gp_Pnt(0, 0, cut_z_min), gp_Dir(0, 0, 1))
            outer = BRepPrimAPI_MakeCylinder(axis, parent_region.radius, step_height).Shape()
            inner = BRepPrimAPI_MakeCylinder(axis, new_radius, step_height).Shape()
            
            cut_shape = BRepAlgoAPI_Cut(outer, inner).Shape()
            self.shape = BRepAlgoAPI_Cut(self.shape, cut_shape).Shape()
            
            # 새 영역: step이 깎인 하단 부분
            new_region = Region(
                z_min=cut_z_min,
                z_max=cut_z_max,
                radius=new_radius,
                direction='bottom'
            )
        
        print(f"    Step ({direction}): z=[{new_region.z_min:.2f}, {new_region.z_max:.2f}], r={new_radius:.2f}")
        return new_region
    
    def _apply_groove(self, parent_region: Region) -> Optional[Region]:
        """Groove 적용 - 부모 영역 내에서 groove 생성
        
        Returns:
            groove가 차지하는 영역 (자식 노드들이 사용할 영역)
        """
        # 최대 groove 폭: 부모 영역에서 양쪽 margin(min_base_height) 남겨야 함
        margin = self.params.min_base_height
        max_groove_width = parent_region.height - 2 * margin
        if max_groove_width < self.params.groove_width_range[0]:
            return None  # 공간 부족
        
        groove_width = random.uniform(
            self.params.groove_width_range[0],
            min(self.params.groove_width_range[1], max_groove_width)
        )
        groove_depth = random.uniform(*self.params.groove_depth_range)
        if parent_region.radius - groove_depth < self.params.min_remaining_radius:
            groove_depth = parent_region.radius - self.params.min_remaining_radius - 0.3
            if groove_depth <= 0.2:
                return None
        
        # groove 위치: 부모 영역 내에서 랜덤 (양쪽 margin 유지)
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
        
        # 새 영역: groove 내부
        new_region = Region(
            z_min=zpos,
            z_max=zpos + groove_width,
            radius=new_radius,
            direction=parent_region.direction
        )
        
        print(f"    Groove: z=[{zpos:.2f}, {zpos + groove_width:.2f}], r={new_radius:.2f}")
        return new_region
    
    def _process_node(self, node: TreeNode, parent_region: Region, step_direction_hint: str = None):
        """노드를 재귀적으로 처리 - 트리 구조에 따른 계층적 포함 관계 구현
        
        Args:
            node: 현재 처리할 노드
            parent_region: 부모 노드의 영역 (이 안에서 특징형상 생성)
            step_direction_hint: step 방향 힌트 ('top' or 'bottom')
        """
        if node.label == 'b':
            # Base 노드: 전체 stock 영역
            node.region = Region(
                z_min=0,
                z_max=self.stock_height,
                radius=self.stock_radius,
                direction=None
            )
            
            # 자식 노드들 처리 - step 방향 분배
            step_children = [c for c in node.children if c.label == 's']
            groove_children = [c for c in node.children if c.label == 'g']
            
            # Step 자식들: 번갈아가며 top/bottom
            for i, child in enumerate(step_children):
                direction = 'top' if i % 2 == 0 else 'bottom'
                self._process_node(child, node.region, step_direction_hint=direction)
            
            # Groove 자식들: base 영역 내에서 생성
            for child in groove_children:
                self._process_node(child, node.region)
                
        elif node.label == 's':
            # Step 노드: 부모 영역 내에서 step 생성
            # 방향 결정: 부모가 step이면 같은 방향, 아니면 힌트 사용
            if parent_region.direction is not None:
                direction = parent_region.direction
            elif step_direction_hint is not None:
                direction = step_direction_hint
            else:
                direction = random.choice(['top', 'bottom'])
            
            new_region = self._apply_step(parent_region, direction)
            
            if new_region is None:
                # Step 생성 실패 - 자식들은 부모 영역 사용
                node.region = parent_region
            else:
                node.region = new_region
            
            # 자식 노드들 처리 - 모두 같은 방향 유지
            for child in node.children:
                self._process_node(child, node.region, step_direction_hint=direction)
                
        elif node.label == 'g':
            # Groove 노드: 부모 영역 내에서 groove 생성
            new_region = self._apply_groove(parent_region)
            
            if new_region is None:
                node.region = parent_region
            else:
                node.region = new_region
            
            # 자식 노드들 처리 - groove 내부 영역에서
            for child in node.children:
                self._process_node(child, node.region)
    
    def generate_from_tree(self, tree_data: Dict, apply_edge_features: bool = True) -> TopoDS_Shape:
        """트리 데이터로부터 터닝 모델 생성
        
        Args:
            tree_data: 트리 JSON 데이터
            apply_edge_features: 챔퍼/라운드 적용 여부
        """
        root = self.load_tree(tree_data)
        
        # 1. Stock 생성
        self._create_stock()
        print(f"  Stock: height={self.stock_height:.2f}, radius={self.stock_radius:.2f}")
        
        # 2. 트리 구조에 따라 재귀적으로 특징형상 생성
        self._process_node(root, None)
        
        # 3. 챔퍼/라운드 적용 (옵션)
        if apply_edge_features:
            self._apply_edge_features()
        
        return self.shape
    
    @staticmethod
    def save_step(shape: TopoDS_Shape, filepath: str) -> bool:
        """STEP 파일로 저장"""
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        
        status = writer.Write(filepath)
        if status == IFSelect_RetDone:
            print(f"  저장: {filepath}")
            return True
        else:
            print(f"  저장 실패: {filepath}")
            return False


def main():
    random.seed(42)
    
    json_path = Path("trees_N6_H3.json")
    if not json_path.exists():
        print(f"Error: {json_path} 파일을 찾을 수 없습니다.")
        return
    
    with open(json_path, 'r') as f:
        trees = json.load(f)
    
    print(f"총 {len(trees)}개의 트리 로드됨")
    
    output_dir = Path("generated_turning_models")
    output_dir.mkdir(exist_ok=True)
    
    # Step 개수별로 분류
    step_count_map: Dict[int, List[int]] = {}
    for i, tree in enumerate(trees):
        step_count = sum(1 for n in tree['nodes'] if n['label'] == 's')
        if step_count not in step_count_map:
            step_count_map[step_count] = []
        step_count_map[step_count].append(i)
    
    print("\nStep 개수별 트리 분포:")
    for count, indices in sorted(step_count_map.items()):
        print(f"  {count}개 step: {len(indices)}개 트리")
    
    # 다양한 구조 선택
    selected_indices = []
    for step_count in range(6):
        if step_count in step_count_map and step_count_map[step_count]:
            candidates = step_count_map[step_count]
            selected = candidates[:min(2, len(candidates))]
            selected_indices.extend(selected)
    
    print(f"\n선택된 트리: {selected_indices}")
    
    for idx in selected_indices:
        tree = trees[idx]
        n_nodes = tree['N']
        max_depth = tree['max_depth_constraint']
        canonical = tree['canonical']
        
        s_count = sum(1 for n in tree['nodes'] if n['label'] == 's')
        g_count = sum(1 for n in tree['nodes'] if n['label'] == 'g')
        
        print(f"\n=== ID:{idx}, S:{s_count}, G:{g_count}, {canonical} ===")
        
        generator = TreeTurningGenerator()
        shape = generator.generate_from_tree(tree, apply_edge_features=True)
        
        if shape and not shape.IsNull():
            filename = f"turning_N{n_nodes}_H{max_depth}_S{s_count}_G{g_count}_{idx:03d}.step"
            output_path = output_dir / filename
            TreeTurningGenerator.save_step(shape, str(output_path))
        else:
            print(f"  모델 생성 실패")
    
    print(f"\n완료! 모델들이 '{output_dir}'에 저장됨")


if __name__ == "__main__":
    main()
