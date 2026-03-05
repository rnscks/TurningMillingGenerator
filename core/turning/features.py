"""
터닝 특징형상 모듈

- StockInfo: Stock 생성 정보
- TurningFeatureRequest: Planner 내부 계획 결과 → Boolean Cut 입력 계약
- create_step_cut / create_groove_cut: 순수 형상 생성 (annular ring)
- create_stock: 원통 스톡 생성
- apply_chamfer / apply_fillet: 단일 엣지 피처 적용
- apply_edge_features: 원형 엣지 전체 후처리
- apply_turning_requests: request 목록을 순서대로 Boolean Cut 적용

TurningParams는 TurningPlanner의 설정 객체로 planner.py에 정의됩니다.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.design_operation import DesignOperation
from core.label_maker import LabelMaker, Labels


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StockInfo:
    """Stock 생성 정보 (Planner 출력 → create_stock 입력)"""
    height: float
    radius: float


@dataclass
class TurningFeatureRequest:
    """터닝 특징형상 적용 요청 (Planner → apply_turning_requests 계약)"""
    feature_type: str       # 'step' or 'groove'
    z_min: float
    z_max: float
    outer_radius: float
    inner_radius: float
    label: int              # Labels.STEP or Labels.GROOVE


# ============================================================================
# 형상 생성 (순수 함수)
# ============================================================================

def create_step_cut(
    z_min: float,
    z_max: float,
    outer_r: float,
    inner_r: float,
) -> TopoDS_Shape:
    """Step 가공용 annular ring 형상 생성 (Boolean Cut 도구).

    outer_r 원통에서 inner_r 원통을 뺀 링 형상을 반환한다.
    """
    height = z_max - z_min
    if height <= 0:
        raise ValueError(f"z_max({z_max}) must be greater than z_min({z_min})")

    axis = gp_Ax2(gp_Pnt(0, 0, z_min), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(axis, outer_r, height).Shape()
    inner = BRepPrimAPI_MakeCylinder(axis, inner_r, height).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


def create_groove_cut(
    z_min: float,
    z_max: float,
    outer_r: float,
    inner_r: float,
) -> TopoDS_Shape:
    """Groove 가공용 annular ring 형상 생성 (Boolean Cut 도구).

    outer_r 원통에서 inner_r 원통을 뺀 링 형상을 반환한다.
    """
    height = z_max - z_min
    if height <= 0:
        raise ValueError(f"z_max({z_max}) must be greater than z_min({z_min})")

    axis = gp_Ax2(gp_Pnt(0, 0, z_min), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(axis, outer_r, height).Shape()
    inner = BRepPrimAPI_MakeCylinder(axis, inner_r, height).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


def create_stock(info: StockInfo) -> TopoDS_Shape:
    """원통 스톡 생성."""
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, info.radius, info.height).Shape()


# ============================================================================
# 엣지 피처 (챔퍼/필렛)
# ============================================================================

def collect_circular_edges(shape: TopoDS_Shape) -> List[TopoDS_Edge]:
    """형상에서 원형 엣지(circular edge)만 수집."""
    circular_edges = []
    try:
        for edge in TopologyExplorer(shape).edges():
            try:
                adaptor = BRepAdaptor_Curve(edge)
                if adaptor.GetType() == GeomAbs_Circle:
                    circular_edges.append(edge)
            except (RuntimeError, TypeError):
                pass
    except Exception as e:
        print(f"    엣지 수집 실패: {e}")
    return circular_edges


def apply_chamfer(
    shape: TopoDS_Shape,
    edge: TopoDS_Edge,
    distance: float,
    label_maker: Optional[LabelMaker] = None,
) -> Tuple[TopoDS_Shape, bool]:
    """엣지에 챔퍼 적용. 성공 시 (새 형상, True), 실패 시 (원본 형상, False)."""
    try:
        if label_maker is not None:
            op = DesignOperation(shape)
            new_shape = op.chamfer(edge, distance)
            if new_shape:
                label_maker.update_label(op, Labels.CHAMFER)
                return new_shape, True
        else:
            builder = BRepFilletAPI_MakeChamfer(shape)
            builder.Add(distance, edge)
            builder.Build()
            if builder.IsDone():
                new_shape = builder.Shape()
                if new_shape and not new_shape.IsNull():
                    return new_shape, True
    except Exception:
        pass
    return shape, False


def apply_fillet(
    shape: TopoDS_Shape,
    edge: TopoDS_Edge,
    radius: float,
    label_maker: Optional[LabelMaker] = None,
) -> Tuple[TopoDS_Shape, bool]:
    """엣지에 필렛 적용. 성공 시 (새 형상, True), 실패 시 (원본 형상, False)."""
    try:
        if label_maker is not None:
            op = DesignOperation(shape)
            new_shape = op.fillet(edge, radius)
            if new_shape:
                label_maker.update_label(op, Labels.FILLET)
                return new_shape, True
        else:
            builder = BRepFilletAPI_MakeFillet(shape)
            builder.Add(radius, edge)
            builder.Build()
            if builder.IsDone():
                new_shape = builder.Shape()
                if new_shape and not new_shape.IsNull():
                    return new_shape, True
    except Exception:
        pass
    return shape, False


def apply_edge_features(
    shape: TopoDS_Shape,
    edge_feature_prob: float,
    chamfer_range: Tuple[float, float],
    fillet_range: Tuple[float, float],
    label_maker: Optional[LabelMaker] = None,
) -> TopoDS_Shape:
    """원형 엣지들에 랜덤하게 챔퍼/필렛 적용."""
    try:
        circular_edges = collect_circular_edges(shape)
        if not circular_edges:
            return shape

        print(f"    원형 엣지 {len(circular_edges)}개 발견")
        applied_count = 0

        for edge in circular_edges:
            if random.random() < edge_feature_prob:
                feature_type = random.choice(['chamfer', 'fillet'])
                if feature_type == 'chamfer':
                    distance = random.uniform(*chamfer_range)
                    shape, success = apply_chamfer(shape, edge, distance, label_maker)
                    if success:
                        applied_count += 1
                        print(f"    Chamfer: d={distance:.2f}")
                else:
                    radius = random.uniform(*fillet_range)
                    shape, success = apply_fillet(shape, edge, radius, label_maker)
                    if success:
                        applied_count += 1
                        print(f"    Fillet: r={radius:.2f}")

        if applied_count > 0:
            print(f"    총 {applied_count}개 엣지 피처 적용")
    except Exception as e:
        print(f"    엣지 피처 적용 중 오류: {e}")

    return shape


# ============================================================================
# Request 적용 (상태 없는 함수)
# ============================================================================

def apply_turning_requests(
    shape: TopoDS_Shape,
    requests: List[TurningFeatureRequest],
    label_maker: Optional[LabelMaker] = None,
) -> TopoDS_Shape:
    """TurningFeatureRequest 목록을 순서대로 Boolean Cut으로 적용.

    Args:
        shape: 원본 스톡 형상
        requests: Planner가 생성한 feature request 목록
        label_maker: Face 라벨 관리자 (None이면 라벨링 비활성)

    Returns:
        모든 요청이 적용된 TopoDS_Shape
    """
    for req in requests:
        if req.feature_type == 'step':
            cut_shape = create_step_cut(req.z_min, req.z_max, req.outer_radius, req.inner_radius)
        elif req.feature_type == 'groove':
            cut_shape = create_groove_cut(req.z_min, req.z_max, req.outer_radius, req.inner_radius)
        else:
            print(f"    [Warning] 알 수 없는 feature_type: {req.feature_type}, 스킵")
            continue

        if cut_shape is None or cut_shape.IsNull():
            print(f"    [Warning] {req.feature_type} 도구 형상 생성 실패, 스킵")
            continue

        if label_maker is not None:
            op = DesignOperation(shape)
            result = op.cut(cut_shape)
            if result is None:
                print(f"    [Warning] {req.feature_type} Boolean Cut 실패, 스킵")
                continue
            shape = result
            label_maker.update_label(op, req.label)
        else:
            result = BRepAlgoAPI_Cut(shape, cut_shape).Shape()
            if result is None or result.IsNull():
                print(f"    [Warning] {req.feature_type} Boolean Cut 실패, 스킵")
                continue
            shape = result

    return shape
