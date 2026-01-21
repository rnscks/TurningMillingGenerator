"""
터닝 형상에 밀링 특징형상 추가 모듈

핵심 로직:
1. FaceAnalyzer의 폭/너비 계산 결과 사용
2. 홀 반경 R ≤ min(W, H) / 2 - clearance 조건으로 유효 면 판단
3. 유효 UV 범위 계산 및 홀 배치

면 타입별 폭/너비 정의:
- Cylinder: width=2R(직경), height=Δz(축방향)
- Cone: width=Δr, height=Δz
- Torus: width=Δr, height=Δz
- Ring: width=thickness(r_out-r_in), height=N/A (둘레 기반)
- Disk: width=height=직경
"""

import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.face_analyzer import FaceAnalyzer, FaceDimensionResult


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HoleParams:
    """홀 특징형상 파라미터"""
    diameter_min: float = 0.5           # 최소 직경 (mm)
    diameter_max_ratio: float = 0.6     # 유효 치수 대비 최대 직경 비율
    clearance: float = 0.3              # 경계로부터의 최소 여유 거리 (mm)
    depth_ratio: float = 1.5            # 홀 깊이 = 직경 * depth_ratio
    min_spacing: float = 1.5            # 홀 간 최소 간격 (mm)
    max_holes_per_face: int = 3         # 면당 최대 홀 수


@dataclass
class ValidFaceInfo:
    """유효한 면 정보"""
    face_id: int
    face: TopoDS_Face
    dimension: FaceDimensionResult
    
    # 홀 스케일 범위
    hole_d_min: float = 0.0
    hole_d_max: float = 0.0
    
    # 유효성
    is_valid: bool = False
    
    # 홀 배치용 정보
    r_outer: float = 0.0
    r_inner: float = 0.0
    z_position: float = 0.0
    radius: float = 0.0  # Cylinder/Cone 반경


@dataclass
class HolePlacement:
    """홀 배치 정보"""
    center_3d: gp_Pnt           # 3D 중심점
    direction: gp_Dir           # 홀 방향 (관통 방향)
    diameter: float             # 홀 직경
    depth: float                # 홀 깊이
    face_id: int                # 배치된 면 ID
    face_type: str              # 면 타입


# ============================================================================
# Hole Scale Calculation
# ============================================================================

def compute_hole_scale_range(
    dim: FaceDimensionResult, 
    params: HoleParams
) -> Tuple[float, float]:
    """
    면의 폭/너비를 기반으로 가능한 홀 직경 범위 계산.
    
    핵심 조건: 
    - 홀 배치 가능 영역 = min(W, H) - 2 * clearance
    - 최대 홀 직경 D_max = ratio * 배치_가능_영역
    
    Returns:
        (D_min, D_max): 가능한 홀 직경 범위
    """
    W = dim.width if dim.width is not None else 0.0
    H = dim.height if dim.height is not None else W
    
    if W <= 0:
        return 0.0, 0.0
    
    min_dim = min(W, H) if H > 0 else W
    effective_dim = min_dim - 2 * params.clearance
    
    if effective_dim <= 0:
        return 0.0, 0.0
    
    D_max = params.diameter_max_ratio * effective_dim
    
    # Ring의 경우 추가 제약
    if dim.is_ring:
        ring_thickness = dim.ring_thickness
        effective_ring = ring_thickness - 2 * params.clearance
        if effective_ring > 0:
            D_max_ring = params.diameter_max_ratio * effective_ring
            D_max = min(D_max, D_max_ring)
        else:
            D_max = 0.0
    
    D_min = params.diameter_min
    D_max = max(0.0, D_max)
    
    return D_min, D_max


# ============================================================================
# Hole Center Calculation by Face Type
# ============================================================================

def _compute_hole_center_plane_ring(
    info: ValidFaceInfo,
    diameter: float,
    params: HoleParams
) -> Optional[gp_Pnt]:
    """Ring 평면에서 유효한 홀 중심점 계산."""
    margin = diameter / 2 + params.clearance
    
    r_valid_min = info.r_inner + margin
    r_valid_max = info.r_outer - margin
    
    if r_valid_min >= r_valid_max:
        return None
    
    r = random.uniform(r_valid_min, r_valid_max)
    theta = random.uniform(0, 2 * math.pi)
    
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = info.z_position
    
    return gp_Pnt(x, y, z)


def _compute_hole_center_plane_disk(
    info: ValidFaceInfo,
    diameter: float,
    params: HoleParams
) -> Optional[gp_Pnt]:
    """Disk 평면에서 유효한 홀 중심점 계산."""
    margin = diameter / 2 + params.clearance
    r_valid_max = info.r_outer - margin
    
    if r_valid_max <= 0:
        return None
    
    r = random.uniform(0, r_valid_max)
    theta = random.uniform(0, 2 * math.pi)
    
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = info.z_position
    
    return gp_Pnt(x, y, z)


def _compute_hole_center_cylinder(
    info: ValidFaceInfo,
    diameter: float,
    params: HoleParams
) -> Optional[Tuple[gp_Pnt, gp_Dir]]:
    """원통면에서 유효한 홀 중심점과 방향 계산."""
    dim = info.dimension
    margin = diameter / 2 + params.clearance
    
    z_valid_min = dim.z_min + margin
    z_valid_max = dim.z_max - margin
    
    if z_valid_min >= z_valid_max:
        return None
    
    z = random.uniform(z_valid_min, z_valid_max)
    theta = random.uniform(0, 2 * math.pi)
    
    radius = info.radius
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    
    center = gp_Pnt(x, y, z)
    direction = gp_Dir(-math.cos(theta), -math.sin(theta), 0)
    
    return center, direction


def _compute_hole_center_cone(
    info: ValidFaceInfo,
    diameter: float,
    params: HoleParams
) -> Optional[Tuple[gp_Pnt, gp_Dir]]:
    """원추면 (챔퍼)에서 유효한 홀 중심점과 방향 계산."""
    dim = info.dimension
    margin = diameter / 2 + params.clearance
    
    z_valid_min = dim.z_min + margin
    z_valid_max = dim.z_max - margin
    
    if z_valid_min >= z_valid_max:
        return None
    
    z = random.uniform(z_valid_min, z_valid_max)
    theta = random.uniform(0, 2 * math.pi)
    
    # Z에 따른 반경 (선형 보간)
    t = (z - dim.z_min) / max(0.001, dim.z_max - dim.z_min)
    r = dim.r_min + t * (dim.r_max - dim.r_min)
    
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    
    center = gp_Pnt(x, y, z)
    direction = gp_Dir(-math.cos(theta), -math.sin(theta), 0)
    
    return center, direction


# ============================================================================
# Hole Creation
# ============================================================================

def create_hole(shape: TopoDS_Shape, center: gp_Pnt, direction: gp_Dir,
                diameter: float, depth: float) -> Optional[TopoDS_Shape]:
    """형상에 홀 추가 (Boolean Cut)."""
    try:
        radius = diameter / 2
        axis = gp_Ax2(center, direction)
        hole_cyl = BRepPrimAPI_MakeCylinder(axis, radius, depth).Shape()
        
        cut_op = BRepAlgoAPI_Cut(shape, hole_cyl)
        cut_op.Build()
        
        if cut_op.IsDone():
            result = cut_op.Shape()
            if result and not result.IsNull():
                return result
    except Exception as e:
        print(f"    홀 생성 실패: {e}")
    
    return None


# ============================================================================
# MillingFeatureAdder Class
# ============================================================================

class MillingFeatureAdder:
    """
    터닝 형상에 밀링 특징형상을 추가하는 클래스.
    
    사용법:
        adder = MillingFeatureAdder(HoleParams())
        new_shape, placements = adder.add_milling_features(
            shape, 
            target_face_types=["Plane", "Cylinder"],
            max_total_holes=5
        )
    """
    
    def __init__(self, params: HoleParams = None):
        self.params = params or HoleParams()
        self.placements: List[HolePlacement] = []
        self.face_infos: List[ValidFaceInfo] = []
        self.analyzer = FaceAnalyzer()
        
    def _analyze_face_for_milling(
        self,
        face: TopoDS_Face,
        face_id: int,
        dim: FaceDimensionResult
    ) -> ValidFaceInfo:
        """면을 분석하여 밀링 가능 여부 판단."""
        # 홀 스케일 범위 계산
        d_min, d_max = compute_hole_scale_range(dim, self.params)
        
        # 유효성 판단
        is_valid = d_max >= d_min and d_max > 0
        
        # 추가 정보 추출
        adaptor = BRepAdaptor_Surface(face, True)
        surf_type = adaptor.GetType()
        
        radius = 0.0
        if surf_type == GeomAbs_Cylinder:
            radius = adaptor.Cylinder().Radius()
        elif surf_type == GeomAbs_Cone:
            radius = (dim.r_max + dim.r_min) / 2
        
        return ValidFaceInfo(
            face_id=face_id,
            face=face,
            dimension=dim,
            hole_d_min=d_min,
            hole_d_max=d_max,
            is_valid=is_valid,
            r_outer=dim.r_outer if dim.r_outer > 0 else dim.r_max,
            r_inner=dim.r_inner if dim.r_inner > 0 else dim.r_min,
            z_position=(dim.z_max + dim.z_min) / 2,
            radius=radius,
        )
        
    def analyze_faces(self, shape: TopoDS_Shape) -> List[ValidFaceInfo]:
        """형상의 모든 면을 분석하여 밀링 가능한 면 목록 반환."""
        topo = TopologyExplorer(shape)
        faces = list(topo.faces())
        
        # 먼저 FaceAnalyzer로 치수 분석
        dim_results = self.analyzer.analyze_shape(shape)
        
        self.face_infos = []
        for i, face in enumerate(faces):
            info = self._analyze_face_for_milling(face, i, dim_results[i])
            self.face_infos.append(info)
        
        return self.face_infos
    
    def get_valid_faces(self, target_types: List[str] = None) -> List[ValidFaceInfo]:
        """유효한 면만 필터링하여 반환."""
        if target_types is None:
            target_types = ["Plane (Ring)", "Plane (Disk)", "Cylinder", "Cone"]
        
        valid = []
        for info in self.face_infos:
            if not info.is_valid:
                continue
            
            face_type = info.dimension.surface_type
            if any(t in face_type for t in target_types):
                valid.append(info)
        
        return valid
    
    def add_hole_to_face(
        self,
        shape: TopoDS_Shape,
        info: ValidFaceInfo,
    ) -> Tuple[TopoDS_Shape, Optional[HolePlacement]]:
        """단일 면에 홀 추가."""
        # 직경 결정
        diameter = random.uniform(info.hole_d_min, info.hole_d_max)
        depth = diameter * self.params.depth_ratio
        
        # 면 타입에 따른 중심점/방향 계산
        face_type = info.dimension.surface_type
        center = None
        direction = gp_Dir(0, 0, -1)  # 기본: -Z
        
        if "Ring" in face_type:
            center = _compute_hole_center_plane_ring(info, diameter, self.params)
            if center and info.z_position > 0:
                direction = gp_Dir(0, 0, -1)
            else:
                direction = gp_Dir(0, 0, 1)
                
        elif "Disk" in face_type:
            center = _compute_hole_center_plane_disk(info, diameter, self.params)
            if center and info.z_position > 0:
                direction = gp_Dir(0, 0, -1)
            else:
                direction = gp_Dir(0, 0, 1)
                
        elif "Cylinder" in face_type:
            result = _compute_hole_center_cylinder(info, diameter, self.params)
            if result:
                center, direction = result
                
        elif "Cone" in face_type:
            result = _compute_hole_center_cone(info, diameter, self.params)
            if result:
                center, direction = result
        
        if center is None:
            return shape, None
        
        # 기존 홀과의 간격 확인
        for existing in self.placements:
            dist = center.Distance(existing.center_3d)
            min_dist = (diameter + existing.diameter) / 2 + self.params.min_spacing
            if dist < min_dist:
                return shape, None
        
        # 홀 생성
        new_shape = create_hole(shape, center, direction, diameter, depth)
        
        if new_shape:
            placement = HolePlacement(
                center_3d=center,
                direction=direction,
                diameter=diameter,
                depth=depth,
                face_id=info.face_id,
                face_type=face_type
            )
            return new_shape, placement
        
        return shape, None
    
    def add_milling_features(
        self,
        shape: TopoDS_Shape,
        target_face_types: List[str] = None,
        max_total_holes: int = 5,
        holes_per_face: int = 1
    ) -> Tuple[TopoDS_Shape, List[HolePlacement]]:
        """
        형상에 밀링 특징형상(홀) 추가.
        
        Args:
            shape: 원본 터닝 형상
            target_face_types: 대상 면 타입 목록
            max_total_holes: 총 최대 홀 수
            holes_per_face: 면당 홀 수
            
        Returns:
            (수정된 형상, 전체 배치 정보)
        """
        if target_face_types is None:
            target_face_types = ["Plane", "Cylinder"]
        
        # 면 분석
        self.analyze_faces(shape)
        print(f"  밀링 대상 면: {len(self.face_infos)}개")
        
        # 유효 면 필터링
        valid_faces = self.get_valid_faces(target_face_types)
        
        if not valid_faces:
            print("  밀링 가능한 면 없음")
            return shape, []
        
        print(f"  필터링된 대상 면: {len(valid_faces)}개")
        
        # 치수 정보 출력
        for info in valid_faces:
            dim = info.dimension
            w = f"{dim.width:.2f}" if dim.width else "N/A"
            h = f"{dim.height:.2f}" if dim.height else "N/A"
            d_range = f"[{info.hole_d_min:.2f}-{info.hole_d_max:.2f}]"
            print(f"    Face {info.face_id} ({dim.surface_type}): W={w}, H={h}, D={d_range}")
        
        current_shape = shape
        self.placements = []
        
        for info in valid_faces:
            if len(self.placements) >= max_total_holes:
                break
            
            for _ in range(holes_per_face):
                if len(self.placements) >= max_total_holes:
                    break
                
                current_shape, placement = self.add_hole_to_face(current_shape, info)
                
                if placement:
                    self.placements.append(placement)
                    print(f"    홀 추가: Face {info.face_id} ({info.dimension.surface_type}), "
                          f"D={placement.diameter:.2f}mm, depth={placement.depth:.2f}mm")
        
        return current_shape, self.placements
