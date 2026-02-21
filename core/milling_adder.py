"""
터닝 형상에 밀링 특징형상 추가 모듈

피처 타입:
- BLIND_HOLE: 블라인드 홀 (막힌 홀)
- THROUGH_HOLE: 관통 홀
- RECTANGULAR_POCKET: 사각 포켓 (블라인드)
- RECTANGULAR_PASSAGE: 사각 통로 (관통)

핵심 로직:
1. FaceAnalyzer의 폭/너비 계산 결과 사용
2. 피처 스케일 ≤ min(W, H) - 2 * clearance 조건으로 유효 면 판단
3. 실린더 측면만 유효 면으로 인식 (상/하 평면은 제외)
4. 유효 범위 계산 및 피처 배치
"""

import math
import random
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.face_analyzer import FaceAnalyzer, FaceDimensionResult
from core.design_operation import DesignOperation
from core.label_maker import LabelMaker, Labels


# ============================================================================
# Enums and Constants
# ============================================================================

class FeatureType(Enum):
    """밀링 피처 타입"""
    BLIND_HOLE = "blind_hole"           # 블라인드 홀
    THROUGH_HOLE = "through_hole"       # 관통 홀
    RECTANGULAR_POCKET = "rect_pocket"  # 사각 포켓 (블라인드)
    RECTANGULAR_PASSAGE = "rect_passage"  # 사각 통로 (관통)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FeatureParams:
    """밀링 피처 파라미터 (공통)"""
    # 스케일 제약
    diameter_min: float = 0.5           # 최소 직경/폭 (mm)
    diameter_max_ratio: float = 0.6     # 유효 치수 대비 최대 스케일 비율
    clearance: float = 0.3              # 경계로부터의 최소 여유 거리 (mm)
    
    # 깊이 설정
    depth_ratio: float = 1.5            # 깊이 = 직경 * depth_ratio (블라인드)
    through_extra: float = 2.0          # 관통 시 추가 깊이 (형상 두께 + extra)
    
    # 배치 설정
    min_spacing: float = 1.5            # 피처 간 최소 간격 (mm)
    max_features_per_face: int = 3      # 면당 최대 피처 수
    
    # 사각 피처 전용
    rect_aspect_min: float = 0.5        # 사각 피처 최소 종횡비 (width/length)
    rect_aspect_max: float = 2.0        # 사각 피처 최대 종횡비



@dataclass
class ValidFaceInfo:
    """유효한 면 정보"""
    face_id: int
    face: TopoDS_Face
    dimension: FaceDimensionResult
    
    # 피처 스케일 범위
    hole_d_min: float = 0.0
    hole_d_max: float = 0.0
    
    # 유효성
    is_valid: bool = False
    
    # 배치용 정보
    r_outer: float = 0.0
    r_inner: float = 0.0
    z_position: float = 0.0
    radius: float = 0.0  # Cylinder/Cone 반경


@dataclass
class FeaturePlacement:
    """피처 배치 정보"""
    feature_type: FeatureType           # 피처 타입
    center_3d: gp_Pnt                   # 3D 중심점
    direction: gp_Dir                   # 피처 방향 (관통 방향)
    
    # 홀 전용
    diameter: float = 0.0               # 홀 직경
    
    # 사각 피처 전용
    width: float = 0.0                  # 사각 폭 (원주 방향)
    length: float = 0.0                 # 사각 길이 (축 방향)
    
    # 공통
    depth: float = 0.0                  # 피처 깊이
    face_id: int = 0                    # 배치된 면 ID
    face_type: str = ""                 # 면 타입
    is_through: bool = False            # 관통 여부



# ============================================================================
# Feature Scale Calculation
# ============================================================================

def compute_hole_scale_range(
    dim: FaceDimensionResult, 
    params: FeatureParams
) -> Tuple[float, float]:
    """
    면의 폭/너비를 기반으로 가능한 피처 스케일 범위 계산.
    
    Returns:
        (D_min, D_max): 가능한 직경/폭 범위
    """
    # None 값 처리 (명시적 검사)
    W = dim.width if dim.width is not None and dim.width > 0 else 0.0
    H = dim.height if dim.height is not None and dim.height > 0 else 0.0
    
    # W가 유효하지 않으면 피처 배치 불가
    if W <= 0:
        return 0.0, 0.0
    
    # H가 유효하지 않으면 W만 사용 (1D 제약)
    if H <= 0:
        min_dim = W
    else:
        min_dim = min(W, H)
    
    effective_dim = min_dim - 2 * params.clearance
    
    if effective_dim <= 0:
        return 0.0, 0.0
    
    D_max = params.diameter_max_ratio * effective_dim
    D_min = params.diameter_min
    
    # D_max가 D_min보다 작으면 유효하지 않은 범위
    if D_max < D_min:
        return 0.0, 0.0
    
    return D_min, D_max


# ============================================================================
# Feature Center/Direction Calculation by Face Type
# ============================================================================

def _compute_feature_center_cylinder(
    info: ValidFaceInfo,
    feature_size: float,
    params: FeatureParams
) -> Optional[Tuple[gp_Pnt, gp_Dir, float]]:
    """
    원통면에서 유효한 피처 중심점, 방향, 가용 깊이 계산.
    
    Returns:
        (center, direction, available_depth) 또는 None
    """
    dim = info.dimension
    margin = feature_size / 2 + params.clearance
    
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
    # 방향: 중심을 향해 (바깥에서 안으로)
    direction = gp_Dir(-math.cos(theta), -math.sin(theta), 0)
    
    # 가용 깊이: 실린더 반경 (중심까지)
    available_depth = radius
    
    return center, direction, available_depth


def _compute_feature_center_cone(
    info: ValidFaceInfo,
    feature_size: float,
    params: FeatureParams
) -> Optional[Tuple[gp_Pnt, gp_Dir, float]]:
    """원추면 (챔퍼)에서 유효한 피처 중심점, 방향, 가용 깊이 계산."""
    dim = info.dimension
    margin = feature_size / 2 + params.clearance
    
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
    available_depth = r
    
    return center, direction, available_depth


# ============================================================================
# Feature Creation Functions
# ============================================================================

def create_blind_hole(
    shape: TopoDS_Shape, 
    center: gp_Pnt, 
    direction: gp_Dir,
    diameter: float, 
    depth: float
) -> Tuple[Optional[TopoDS_Shape], Optional[DesignOperation]]:
    """블라인드 홀 생성 (Boolean Cut + History 추적)."""
    try:
        radius = diameter / 2
        axis = gp_Ax2(center, direction)
        hole_cyl = BRepPrimAPI_MakeCylinder(axis, radius, depth).Shape()
        
        op = DesignOperation(shape)
        result = op.cut(hole_cyl)
        if result is not None:
            return result, op
    except Exception as e:
        print(f"    블라인드 홀 생성 실패: {e}")
    
    return None, None


def create_through_hole(
    shape: TopoDS_Shape, 
    center: gp_Pnt, 
    direction: gp_Dir,
    diameter: float, 
    available_depth: float,
    extra: float = 2.0
) -> Tuple[Optional[TopoDS_Shape], Optional[DesignOperation]]:
    """관통 홀 생성 (Boolean Cut + History 추적)."""
    try:
        radius = diameter / 2
        through_depth = available_depth + extra
        
        axis = gp_Ax2(center, direction)
        hole_cyl = BRepPrimAPI_MakeCylinder(axis, radius, through_depth).Shape()
        
        op = DesignOperation(shape)
        result = op.cut(hole_cyl)
        if result is not None:
            return result, op
    except Exception as e:
        print(f"    관통 홀 생성 실패: {e}")
    
    return None, None


def create_rectangular_pocket(
    shape: TopoDS_Shape,
    center: gp_Pnt,
    direction: gp_Dir,
    width: float,
    length: float,
    depth: float,
    theta: float = 0.0
) -> Tuple[Optional[TopoDS_Shape], Optional[DesignOperation]]:
    """
    사각 포켓 (블라인드) 생성.
    
    원통면에서:
    - width: 원주 방향 (theta 방향)
    - length: 축 방향 (Z 방향)
    """
    try:
        half_w = width / 2
        half_l = length / 2
        
        dir_vec = gp_Vec(direction)
        
        if abs(direction.Z()) > 0.9:
            local_x = gp_Dir(1, 0, 0)
        else:
            z_axis = gp_Dir(0, 0, 1)
            local_x_vec = gp_Vec(direction).Crossed(gp_Vec(z_axis))
            local_x = gp_Dir(local_x_vec)
        
        local_y_vec = gp_Vec(direction).Crossed(gp_Vec(local_x))
        local_y = gp_Dir(local_y_vec)
        
        start_pt = gp_Pnt(
            center.X() - half_w * local_x.X() - half_l * local_y.X(),
            center.Y() - half_w * local_x.Y() - half_l * local_y.Y(),
            center.Z() - half_w * local_x.Z() - half_l * local_y.Z()
        )
        
        box_maker = BRepPrimAPI_MakeBox(
            gp_Ax2(start_pt, direction, local_x),
            width, length, depth
        )
        pocket_box = box_maker.Shape()
        
        op = DesignOperation(shape)
        result = op.cut(pocket_box)
        if result is not None:
            return result, op
    except Exception as e:
        print(f"    사각 포켓 생성 실패: {e}")
    
    return None, None


def create_rectangular_passage(
    shape: TopoDS_Shape,
    center: gp_Pnt,
    direction: gp_Dir,
    width: float,
    length: float,
    available_depth: float,
    extra: float = 2.0,
    theta: float = 0.0
) -> Tuple[Optional[TopoDS_Shape], Optional[DesignOperation]]:
    """사각 통로 (관통) 생성."""
    through_depth = available_depth + extra
    return create_rectangular_pocket(
        shape, center, direction, 
        width, length, through_depth, theta
    )


# ============================================================================
# MillingFeatureAdder Class
# ============================================================================

class MillingFeatureAdder:
    """
    터닝 형상에 밀링 특징형상을 추가하는 클래스.
    
    사용법:
        adder = MillingFeatureAdder(FeatureParams())
        new_shape, placements = adder.add_milling_features(
            shape, 
            target_face_types=["Cylinder"],  # 실린더 측면만
            max_total_features=5
        )
    """
    
    def __init__(self, params: FeatureParams = None):
        self.params = params or FeatureParams()
        self.placements: List[FeaturePlacement] = []
        self.face_infos: List[ValidFaceInfo] = []
        self.analyzer = FaceAnalyzer()
        self._label_maker: Optional[LabelMaker] = None
        
    def _analyze_face_for_milling(
        self,
        face: TopoDS_Face,
        face_id: int,
        dim: FaceDimensionResult
    ) -> ValidFaceInfo:
        """면을 분석하여 밀링 가능 여부 판단."""
        # 스케일 범위 계산
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
        """
        유효한 면만 필터링하여 반환.
        
        기본: Cylinder, Cone만 (Plane 제외 - 상/하 뚜껑면 제외)
        """
        if target_types is None:
            # 기본: 실린더 측면만 (Plane 제외)
            target_types = ["Cylinder", "Cone"]
        
        valid = []
        for info in self.face_infos:
            if not info.is_valid:
                continue
            
            face_type = info.dimension.surface_type
            
            # Plane (Ring/Disk)은 제외 (상/하 뚜껑면)
            if "Plane" in face_type:
                continue
            
            if any(t in face_type for t in target_types):
                valid.append(info)
        
        return valid
    
    def add_feature_to_face(
        self,
        shape: TopoDS_Shape,
        info: ValidFaceInfo,
        feature_type: FeatureType = None
    ) -> Tuple[TopoDS_Shape, Optional[FeaturePlacement]]:
        """
        단일 면에 밀링 피처 추가.
        
        Args:
            shape: 원본 형상
            info: 면 정보
            feature_type: 피처 타입 (None이면 랜덤 선택)
        """
        # 피처 타입 결정
        if feature_type is None:
            feature_type = random.choice(list(FeatureType))
        
        # 스케일 결정
        diameter = random.uniform(info.hole_d_min, info.hole_d_max)
        
        # 면 타입에 따른 중심점/방향 계산
        face_type = info.dimension.surface_type
        result = None
        
        if "Cylinder" in face_type:
            result = _compute_feature_center_cylinder(info, diameter, self.params)
        elif "Cone" in face_type:
            result = _compute_feature_center_cone(info, diameter, self.params)
        
        if result is None:
            return shape, None
        
        center, direction, available_depth = result
        
        # 기존 피처와의 간격 확인
        for existing in self.placements:
            dist = center.Distance(existing.center_3d)
            existing_size = existing.diameter if existing.diameter > 0 else max(existing.width, existing.length)
            min_dist = (diameter + existing_size) / 2 + self.params.min_spacing
            if dist < min_dist:
                return shape, None
        
        # 피처 생성
        new_shape = None
        operation = None
        placement = None
        
        if feature_type == FeatureType.BLIND_HOLE:
            depth = diameter * self.params.depth_ratio
            new_shape, operation = create_blind_hole(shape, center, direction, diameter, depth)
            if new_shape:
                label = Labels.BLIND_HOLE
                placement = FeaturePlacement(
                    feature_type=feature_type,
                    center_3d=center,
                    direction=direction,
                    diameter=diameter,
                    depth=depth,
                    face_id=info.face_id,
                    face_type=face_type,
                    is_through=False
                )
                
        elif feature_type == FeatureType.THROUGH_HOLE:
            new_shape, operation = create_through_hole(
                shape, center, direction, diameter, 
                available_depth, self.params.through_extra
            )
            if new_shape:
                label = Labels.THROUGH_HOLE
                placement = FeaturePlacement(
                    feature_type=feature_type,
                    center_3d=center,
                    direction=direction,
                    diameter=diameter,
                    depth=available_depth + self.params.through_extra,
                    face_id=info.face_id,
                    face_type=face_type,
                    is_through=True
                )
                
        elif feature_type == FeatureType.RECTANGULAR_POCKET:
            aspect = random.uniform(self.params.rect_aspect_min, self.params.rect_aspect_max)
            width = diameter
            length = diameter * aspect
            depth = diameter * self.params.depth_ratio
            
            new_shape, operation = create_rectangular_pocket(
                shape, center, direction, width, length, depth
            )
            if new_shape:
                label = Labels.RECTANGULAR_POCKET
                placement = FeaturePlacement(
                    feature_type=feature_type,
                    center_3d=center,
                    direction=direction,
                    width=width,
                    length=length,
                    depth=depth,
                    face_id=info.face_id,
                    face_type=face_type,
                    is_through=False
                )
                
        elif feature_type == FeatureType.RECTANGULAR_PASSAGE:
            aspect = random.uniform(self.params.rect_aspect_min, self.params.rect_aspect_max)
            width = diameter
            length = diameter * aspect
            
            new_shape, operation = create_rectangular_passage(
                shape, center, direction, width, length,
                available_depth, self.params.through_extra
            )
            if new_shape:
                label = Labels.RECTANGULAR_PASSAGE
                placement = FeaturePlacement(
                    feature_type=feature_type,
                    center_3d=center,
                    direction=direction,
                    width=width,
                    length=length,
                    depth=available_depth + self.params.through_extra,
                    face_id=info.face_id,
                    face_type=face_type,
                    is_through=True
                )
        
        if new_shape and placement:
            # 라벨 추적
            if self._label_maker is not None and operation is not None:
                self._label_maker.update_label(operation, label)
            return new_shape, placement
        
        return shape, None
    
    def add_hole_to_face(
        self,
        shape: TopoDS_Shape,
        info: ValidFaceInfo,
    ) -> Tuple[TopoDS_Shape, Optional[FeaturePlacement]]:
        """단일 면에 홀 추가 (하위 호환성 - 랜덤 홀 타입)."""
        hole_type = random.choice([FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE])
        return self.add_feature_to_face(shape, info, hole_type)
    
    def add_milling_features(
        self,
        shape: TopoDS_Shape,
        target_face_types: List[str] = None,
        max_total_holes: int = 5,
        holes_per_face: int = 1,
        feature_types: List[FeatureType] = None,
        label_maker: Optional[LabelMaker] = None
    ) -> Tuple[TopoDS_Shape, List[FeaturePlacement]]:
        """
        형상에 밀링 특징형상 추가.
        
        Args:
            shape: 원본 터닝 형상
            target_face_types: 대상 면 타입 목록 (기본: ["Cylinder"])
            max_total_holes: 총 최대 피처 수
            holes_per_face: 면당 피처 수
            feature_types: 사용할 피처 타입 목록 (None이면 모든 타입)
            
        Returns:
            (수정된 형상, 전체 배치 정보)
        """
        if target_face_types is None:
            # 기본: 실린더 측면만 (Plane 제외)
            target_face_types = ["Cylinder"]
        
        if feature_types is None:
            feature_types = list(FeatureType)
        
        self._label_maker = label_maker
        
        # 면 분석
        self.analyze_faces(shape)
        print(f"  밀링 대상 면: {len(self.face_infos)}개")
        
        # 유효 면 필터링 (Plane 제외됨)
        valid_faces = self.get_valid_faces(target_face_types)
        
        if not valid_faces:
            print("  밀링 가능한 면 없음 (실린더 측면만 대상)")
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
        face_usage_count: dict = {}
        
        for info in valid_faces:
            if len(self.placements) >= max_total_holes:
                break
            
            current_usage = face_usage_count.get(info.face_id, 0)
            if current_usage >= self.params.max_features_per_face:
                continue
            
            for _ in range(holes_per_face):
                if len(self.placements) >= max_total_holes:
                    break
                
                if face_usage_count.get(info.face_id, 0) >= self.params.max_features_per_face:
                    break
                
                feature_type = random.choice(feature_types)
                
                current_shape, placement = self.add_feature_to_face(
                    current_shape, info, feature_type
                )
                
                if placement:
                    self.placements.append(placement)
                    face_usage_count[info.face_id] = face_usage_count.get(info.face_id, 0) + 1
                    
                    # 로그 출력
                    if placement.diameter > 0:
                        size_str = f"D={placement.diameter:.2f}mm"
                    else:
                        size_str = f"W={placement.width:.2f}mm, L={placement.length:.2f}mm"
                    
                    through_str = "(관통)" if placement.is_through else "(블라인드)"
                    print(f"    피처 추가: Face {info.face_id}, {placement.feature_type.value} {through_str}, "
                          f"{size_str}, depth={placement.depth:.2f}mm")
        
        return current_shape, self.placements
