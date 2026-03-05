"""
밀링 특징형상 모듈

- MillingParams: 밀링 제약조건 파라미터 (기존 FeatureParams 개명)
- MillingFeatureRequest: Analyzer → Applier 계약 (기존 FeaturePlacement 흡수)
- create_blind_hole / create_through_hole: 홀 도구 형상 생성
- create_rectangular_pocket / create_rectangular_passage: 포켓 도구 형상 생성
- compute_feature_center_cylinder / compute_feature_center_cone: 배치 중심 계산
- compute_hole_scale_range: 피처 스케일 범위 계산
- apply_milling_requests: request 목록을 순서대로 Boolean Cut 적용
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

from core.design_operation import DesignOperation
from core.label_maker import LabelMaker, Labels


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MillingParams:
    """밀링 제약조건 파라미터"""
    diameter_min: float = 0.5
    diameter_max_ratio: float = 0.6
    clearance: float = 0.3

    depth_ratio: float = 1.5
    through_extra: float = 2.0

    min_spacing: float = 1.5
    max_features_per_face: int = 3

    rect_aspect_min: float = 0.5
    rect_aspect_max: float = 2.0


@dataclass
class MillingFeatureRequest:
    """밀링 특징형상 적용 요청 (Analyzer → apply_milling_requests 계약)"""
    feature_type: str                   # 'blind_hole', 'through_hole', 'rect_pocket', 'rect_passage'
    center: gp_Pnt
    direction: gp_Dir
    diameter: float = 0.0
    width: float = 0.0
    length: float = 0.0
    depth: float = 0.0
    is_through: bool = False
    label: int = 0
    face_id: int = 0
    face_type: str = ""


# ============================================================================
# 피처 스케일 계산
# ============================================================================

def compute_hole_scale_range(
    dim,
    params: MillingParams,
) -> Tuple[float, float]:
    """면의 폭/너비를 기반으로 가능한 피처 스케일 범위 계산.

    Returns:
        (D_min, D_max): 가능한 직경/폭 범위
    """
    W = dim.width if dim.width is not None and dim.width > 0 else 0.0
    H = dim.height if dim.height is not None and dim.height > 0 else 0.0

    if W <= 0:
        return 0.0, 0.0

    min_dim = min(W, H) if H > 0 else W

    effective_dim = min_dim - 2 * params.clearance
    if effective_dim <= 0:
        return 0.0, 0.0

    D_max = params.diameter_max_ratio * effective_dim
    D_min = params.diameter_min

    if D_max < D_min:
        return 0.0, 0.0

    return D_min, D_max


# ============================================================================
# 배치 중심/방향 계산
# ============================================================================

def compute_feature_center_cylinder(
    info: dict,
    feature_size: float,
    params: MillingParams,
) -> Optional[Tuple[gp_Pnt, gp_Dir, float]]:
    """원통면에서 유효한 피처 중심점, 방향, 가용 깊이 계산.

    Args:
        info: _filter_valid_faces가 반환하는 dict
              {'dim': FaceDimensionResult, 'radius': float, ...}

    Returns:
        (center, direction, available_depth) 또는 None
    """
    dim = info['dim']
    margin = feature_size / 2 + params.clearance

    z_valid_min = dim.z_min + margin
    z_valid_max = dim.z_max - margin

    if z_valid_min >= z_valid_max:
        return None

    z = random.uniform(z_valid_min, z_valid_max)
    theta = random.uniform(0, 2 * math.pi)

    radius = info['radius']
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)

    center = gp_Pnt(x, y, z)
    direction = gp_Dir(-math.cos(theta), -math.sin(theta), 0)
    available_depth = radius

    return center, direction, available_depth


def compute_feature_center_cone(
    info: dict,
    feature_size: float,
    params: MillingParams,
) -> Optional[Tuple[gp_Pnt, gp_Dir, float]]:
    """원추면(챔퍼)에서 유효한 피처 중심점, 방향, 가용 깊이 계산.

    Args:
        info: _filter_valid_faces가 반환하는 dict
    """
    dim = info['dim']
    margin = feature_size / 2 + params.clearance

    z_valid_min = dim.z_min + margin
    z_valid_max = dim.z_max - margin

    if z_valid_min >= z_valid_max:
        return None

    z = random.uniform(z_valid_min, z_valid_max)
    theta = random.uniform(0, 2 * math.pi)

    t = (z - dim.z_min) / max(0.001, dim.z_max - dim.z_min)
    r = dim.r_min + t * (dim.r_max - dim.r_min)

    x = r * math.cos(theta)
    y = r * math.sin(theta)

    center = gp_Pnt(x, y, z)
    direction = gp_Dir(-math.cos(theta), -math.sin(theta), 0)
    available_depth = r

    return center, direction, available_depth


# ============================================================================
# 도구 형상 생성 (순수 함수)
# ============================================================================

def create_blind_hole(
    center: gp_Pnt,
    direction: gp_Dir,
    diameter: float,
    depth: float,
) -> TopoDS_Shape:
    """블라인드 홀 커팅 도구 형상(원통) 생성."""
    radius = diameter / 2
    axis = gp_Ax2(center, direction)
    return BRepPrimAPI_MakeCylinder(axis, radius, depth).Shape()


def create_through_hole(
    center: gp_Pnt,
    direction: gp_Dir,
    diameter: float,
    available_depth: float,
    extra: float = 2.0,
) -> TopoDS_Shape:
    """관통 홀 커팅 도구 형상(원통) 생성."""
    radius = diameter / 2
    through_depth = available_depth + extra
    axis = gp_Ax2(center, direction)
    return BRepPrimAPI_MakeCylinder(axis, radius, through_depth).Shape()


def create_rectangular_pocket(
    center: gp_Pnt,
    direction: gp_Dir,
    width: float,
    length: float,
    depth: float,
) -> TopoDS_Shape:
    """사각 포켓/통로 커팅 도구 형상(박스) 생성.

    원통면에서:
    - width: 원주 방향 (theta 방향)
    - length: 축 방향 (Z 방향)
    """
    half_w = width / 2
    half_l = length / 2

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

    return BRepPrimAPI_MakeBox(
        gp_Ax2(start_pt, direction, local_x),
        width, length, depth
    ).Shape()


def create_rectangular_passage(
    center: gp_Pnt,
    direction: gp_Dir,
    width: float,
    length: float,
    available_depth: float,
    extra: float = 2.0,
) -> TopoDS_Shape:
    """사각 통로 커팅 도구 형상 생성 (관통)."""
    through_depth = available_depth + extra
    return create_rectangular_pocket(center, direction, width, length, through_depth)


# ============================================================================
# Request 적용 (상태 없는 함수)
# ============================================================================

def apply_milling_requests(
    shape: TopoDS_Shape,
    requests: List[MillingFeatureRequest],
    label_maker: Optional[LabelMaker] = None,
) -> TopoDS_Shape:
    """MillingFeatureRequest 목록을 순서대로 Boolean Cut으로 적용.

    Args:
        shape: 원본 터닝 형상
        requests: Analyzer가 생성한 feature request 목록
        label_maker: Face 라벨 관리자 (None이면 라벨링 비활성)

    Returns:
        모든 요청이 적용된 TopoDS_Shape
    """
    for req in requests:
        try:
            if req.feature_type == 'blind_hole':
                tool = create_blind_hole(req.center, req.direction, req.diameter, req.depth)
            elif req.feature_type == 'through_hole':
                tool = create_through_hole(req.center, req.direction, req.diameter, req.depth)
            elif req.feature_type in ('rect_pocket', 'rect_passage'):
                tool = create_rectangular_pocket(
                    req.center, req.direction, req.width, req.length, req.depth
                )
            else:
                print(f"    [Warning] 알 수 없는 feature_type: {req.feature_type}, 스킵")
                continue

            if tool is None or tool.IsNull():
                print(f"    [Warning] {req.feature_type} 도구 형상 생성 실패, 스킵")
                continue

            if label_maker is not None:
                op = DesignOperation(shape)
                result = op.cut(tool)
                if result is None:
                    print(f"    [Warning] {req.feature_type} Boolean Cut 실패, 스킵")
                    continue
                shape = result
                label_maker.update_label(op, req.label)
            else:
                result = BRepAlgoAPI_Cut(shape, tool).Shape()
                if result is None or result.IsNull():
                    print(f"    [Warning] {req.feature_type} Boolean Cut 실패, 스킵")
                    continue
                shape = result

        except Exception as e:
            print(f"    [Warning] {req.feature_type} 적용 중 오류: {e}, 스킵")
            continue

    return shape
