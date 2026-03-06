"""
터닝 특징형상 모듈

- create_stock: 원통 스톡 생성
- create_step_cut / create_groove_cut: 순수 형상 생성 (annular ring)
- apply_step_cut / apply_groove_cut: Boolean Cut 적용 (단일 피처)
- apply_chamfer / apply_fillet: 단일 엣지 피처 적용
- apply_edge_features: 원형 엣지 전체 후처리

TurningParams는 TurningPlanner의 설정 객체로 planner.py에 정의됩니다.
"""

import random
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
# 형상 생성 (순수 함수)
# ============================================================================

def create_stock(height: float, radius: float) -> TopoDS_Shape:
    """원통 스톡 생성."""
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def create_step_cut(
    z_cut_min: float,
    z_cut_max: float,
    outer_r: float,
    inner_r: float,
) -> TopoDS_Shape:
    """Step 가공용 annular ring 형상 생성 (Boolean Cut 도구).

    지정된 z 범위 [z_cut_min, z_cut_max]에서 outer_r → inner_r 링으로 제거.

    Args:
        z_cut_min: 잘라낼 z 범위 시작
        z_cut_max: 잘라낼 z 범위 끝
        outer_r: 바깥 반경 (부모 반경)
        inner_r: 안쪽 반경 (깎인 후 반경)
    """
    z_min, z_max = z_cut_min, z_cut_max

    height = z_max - z_min
    if height <= 0:
        raise ValueError(
            f"Step 높이가 0 이하: z=[{z_min:.2f}, {z_max:.2f}], height={height:.2f}"
        )

    axis = gp_Ax2(gp_Pnt(0, 0, z_min), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(axis, outer_r, height).Shape()
    inner = BRepPrimAPI_MakeCylinder(axis, inner_r, height).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


def create_groove_cut(
    zpos: float,
    width: float,
    outer_r: float,
    inner_r: float,
) -> TopoDS_Shape:
    """Groove 가공용 annular ring 형상 생성 (Boolean Cut 도구).

    zpos를 groove 시작점(z_min)으로, zpos+width를 끝점(z_max)으로 사용한다.

    Args:
        zpos: groove 시작 z 위치 (z_min)
        width: groove 폭 (z 방향 길이)
        outer_r: groove 바깥 반경 (부모 반경)
        inner_r: groove 안쪽 반경 (깎인 후 반경)
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    z_min, z_max = zpos, zpos + width
    axis = gp_Ax2(gp_Pnt(0, 0, z_min), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(axis, outer_r, width).Shape()
    inner = BRepPrimAPI_MakeCylinder(axis, inner_r, width).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


# ============================================================================
# Boolean Cut 적용
# ============================================================================

def apply_step_cut(
    shape: TopoDS_Shape,
    z_cut_min: float,
    z_cut_max: float,
    outer_radius: float,
    inner_radius: float,
    label_maker: Optional[LabelMaker] = None,
) -> Optional[TopoDS_Shape]:
    """Step Boolean Cut 적용. 성공 시 새 형상, 실패 시 None.

    Args:
        z_cut_min: 잘라낼 z 범위 시작
        z_cut_max: 잘라낼 z 범위 끝
        outer_radius: 깎아내기 전 반경 (부모 반경)
        inner_radius: 깎아낸 후 반경 (새 반경)
    """
    cut_shape = create_step_cut(z_cut_min, z_cut_max, outer_radius, inner_radius)
    if cut_shape is None or cut_shape.IsNull():
        print("    [Warning] Step 도구 형상 생성 실패")
        return None
    return _apply_cut(shape, cut_shape, Labels.STEP, label_maker, expected_new_faces=2)


def apply_groove_cut(
    shape: TopoDS_Shape,
    zpos: float,
    width: float,
    outer_radius: float,
    inner_radius: float,
    label_maker: Optional[LabelMaker] = None,
) -> Optional[TopoDS_Shape]:
    """Groove Boolean Cut 적용. 성공 시 새 형상, 실패 시 None.

    Args:
        zpos: groove 시작 z 위치 (z_min)
        width: groove 폭 (z 방향 길이)
        outer_radius: groove 바깥 반경 (부모 반경)
        inner_radius: groove 안쪽 반경 (깎인 후 반경)
    """
    cut_shape = create_groove_cut(zpos, width, outer_radius, inner_radius)
    if cut_shape is None or cut_shape.IsNull():
        print("    [Warning] Groove 도구 형상 생성 실패")
        return None
    return _apply_cut(shape, cut_shape, Labels.GROOVE, label_maker, expected_new_faces=3)


def _apply_cut(
    shape: TopoDS_Shape,
    cut_shape: TopoDS_Shape,
    label: int,
    label_maker: Optional[LabelMaker],
    expected_new_faces: Optional[int] = None,
) -> Optional[TopoDS_Shape]:
    """Boolean Cut 공통 로직.

    Args:
        expected_new_faces: 정상 적용 시 생성되어야 할 새 face 수.
            None이면 검증 생략. Step=2, Groove=3.
    """
    op = DesignOperation(shape)
    result = op.cut(cut_shape)
    if result is None:
        return None

    if expected_new_faces is not None:
        actual = len(op.generated_faces)
        if actual != expected_new_faces:
            feature_name = {Labels.STEP: "Step", Labels.GROOVE: "Groove"}.get(label, "Feature")
            print(
                f"    [Warning] {feature_name} 생성 face 수 불일치: "
                f"expected={expected_new_faces}, actual={actual}"
            )
            return None

    if label_maker is not None:
        label_maker.update_label(op, label)

    return result


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
