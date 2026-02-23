# -*- coding: utf-8 -*-
"""
design_operation.py 테스트 모듈

Boolean Cut / Chamfer / Fillet 연산의 Face History 추적 검증.

테스트 실행:
    pytest tests/test_design_operation.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.design_operation import DesignOperation, collect_faces, search_same_face


# ============================================================================
# 헬퍼: 테스트용 형상 생성
# ============================================================================

def make_cylinder(radius: float = 10.0, height: float = 20.0) -> "TopoDS_Shape":
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def make_small_cylinder(radius: float = 3.0, height: float = 5.0, z_offset: float = 7.5) -> "TopoDS_Shape":
    """Boolean Cut 도구용 작은 원기둥 (옆에서 파는 홀 시뮬레이션)"""
    axis = gp_Ax2(gp_Pnt(10, 0, z_offset), gp_Dir(-1, 0, 0))
    return BRepPrimAPI_MakeCylinder(axis, radius, 15.0).Shape()


def make_box(dx=5.0, dy=5.0, dz=5.0) -> "TopoDS_Shape":
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def get_circular_edge(shape):
    """형상에서 첫 번째 원형 엣지를 반환."""
    for edge in TopologyExplorer(shape).edges():
        try:
            adaptor = BRepAdaptor_Curve(edge)
            if adaptor.GetType() == GeomAbs_Circle:
                return edge
        except Exception:
            pass
    return None


# ============================================================================
# collect_faces / search_same_face 테스트
# ============================================================================

class TestCollectFaces:
    def test_cylinder_has_3_faces(self):
        """원기둥 = 측면 1 + 상면 1 + 하면 1 = 3개"""
        cyl = make_cylinder()
        faces = collect_faces(cyl)
        assert len(faces) == 3

    def test_box_has_6_faces(self):
        """박스 = 6면"""
        box = make_box()
        faces = collect_faces(box)
        assert len(faces) == 6

    def test_returns_list(self):
        cyl = make_cylinder()
        faces = collect_faces(cyl)
        assert isinstance(faces, list)


class TestSearchSameFace:
    def test_find_existing_face(self):
        """face 목록에서 자기 자신을 찾을 수 있는지"""
        cyl = make_cylinder()
        faces = collect_faces(cyl)
        target = faces[0]
        found = search_same_face(target, faces)
        assert found is not None
        assert found.IsSame(target)

    def test_not_found_in_different_shape(self):
        """다른 형상의 face는 찾을 수 없음"""
        cyl1 = make_cylinder(radius=10.0)
        cyl2 = make_cylinder(radius=5.0)
        faces1 = collect_faces(cyl1)
        faces2 = collect_faces(cyl2)
        found = search_same_face(faces1[0], faces2)
        assert found is None

    def test_empty_list_returns_none(self):
        cyl = make_cylinder()
        faces = collect_faces(cyl)
        found = search_same_face(faces[0], [])
        assert found is None


# ============================================================================
# DesignOperation.cut() 테스트
# ============================================================================

class TestDesignOperationCut:
    def test_cut_returns_shape(self):
        """Boolean Cut이 유효한 형상을 반환"""
        stock = make_cylinder(radius=10.0, height=20.0)
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        result = op.cut(tool)

        assert result is not None
        assert not result.IsNull()

    def test_cut_tracks_origin_faces(self):
        """Cut 전 원본 Face 목록이 기록됨"""
        stock = make_cylinder()
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        origin_count = len(op.origin_faces)
        op.cut(tool)

        assert origin_count == 3

    def test_cut_processed_faces_differ(self):
        """Cut 후 processed_faces가 원본과 다름 (면 수 변화)"""
        stock = make_cylinder()
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        op.cut(tool)

        assert len(op.processed_faces) != len(op.origin_faces)

    def test_cut_generates_new_faces(self):
        """Boolean Cut이 새로운 면을 생성"""
        stock = make_cylinder()
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        op.cut(tool)

        assert len(op.generated_faces) > 0

    def test_cut_modified_plus_generated_covers_processed(self):
        """modified + generated ≥ processed (모든 결과 face가 추적됨)"""
        stock = make_cylinder()
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        op.cut(tool)

        modified_count = sum(len(v) for v in op.modified_faces.values())
        generated_count = len(op.generated_faces)
        processed_count = len(op.processed_faces)

        assert modified_count + generated_count == processed_count

    def test_get_modified_faces_for_origin(self):
        """원본 face에 대해 get_modified_faces 호출 가능"""
        stock = make_cylinder()
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        op.cut(tool)

        for origin_face in op.origin_faces:
            modified = op.get_modified_faces(origin_face)
            assert isinstance(modified, list)

    def test_get_generated_faces_returns_list(self):
        stock = make_cylinder()
        tool = make_small_cylinder()

        op = DesignOperation(stock)
        op.cut(tool)

        generated = op.get_generated_faces()
        assert isinstance(generated, list)


# ============================================================================
# DesignOperation.chamfer() / fillet() 테스트
# ============================================================================

class TestDesignOperationEdgeFeatures:
    def test_chamfer_returns_shape(self):
        """Chamfer가 유효한 형상을 반환"""
        cyl = make_cylinder()
        edge = get_circular_edge(cyl)
        if edge is None:
            pytest.skip("원형 엣지를 찾을 수 없음")

        op = DesignOperation(cyl)
        result = op.chamfer(edge, 0.5)

        assert result is not None
        assert not result.IsNull()

    def test_chamfer_generates_new_faces(self):
        """Chamfer가 새로운 면 생성"""
        cyl = make_cylinder()
        edge = get_circular_edge(cyl)
        if edge is None:
            pytest.skip("원형 엣지를 찾을 수 없음")

        op = DesignOperation(cyl)
        op.chamfer(edge, 0.5)

        assert len(op.generated_faces) > 0

    def test_fillet_returns_shape(self):
        """Fillet이 유효한 형상을 반환"""
        cyl = make_cylinder()
        edge = get_circular_edge(cyl)
        if edge is None:
            pytest.skip("원형 엣지를 찾을 수 없음")

        op = DesignOperation(cyl)
        result = op.fillet(edge, 0.5)

        assert result is not None
        assert not result.IsNull()

    def test_fillet_generates_new_faces(self):
        """Fillet이 새로운 면 생성"""
        cyl = make_cylinder()
        edge = get_circular_edge(cyl)
        if edge is None:
            pytest.skip("원형 엣지를 찾을 수 없음")

        op = DesignOperation(cyl)
        op.fillet(edge, 0.5)

        assert len(op.generated_faces) > 0

    def test_chamfer_face_tracking_completeness(self):
        """Chamfer 후 modified + generated = processed"""
        cyl = make_cylinder()
        edge = get_circular_edge(cyl)
        if edge is None:
            pytest.skip("원형 엣지를 찾을 수 없음")

        op = DesignOperation(cyl)
        op.chamfer(edge, 0.5)

        modified_count = sum(len(v) for v in op.modified_faces.values())
        generated_count = len(op.generated_faces)
        processed_count = len(op.processed_faces)

        assert modified_count + generated_count == processed_count


# ============================================================================
# 실패 케이스 테스트
# ============================================================================

class TestDesignOperationFailure:
    def test_cut_with_non_intersecting_tool(self):
        """교차하지 않는 도구로 Cut — 결과는 원본과 동일"""
        stock = make_cylinder(radius=10.0, height=20.0)
        far_tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(100, 100, 100), gp_Dir(0, 0, 1)),
            1.0, 1.0
        ).Shape()

        op = DesignOperation(stock)
        result = op.cut(far_tool)

        assert result is not None
        assert len(op.processed_faces) == len(op.origin_faces)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
