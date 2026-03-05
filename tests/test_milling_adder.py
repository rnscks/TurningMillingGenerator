# -*- coding: utf-8 -*-
"""
milling 모듈 테스트 (구 milling_adder.py → 신 milling/features + milling/analyzer)

새 API:
- MillingParams, MillingFeatureRequest, compute_hole_scale_range (core.milling.features)
- create_blind_hole, create_through_hole, create_rectangular_pocket (core.milling.features)
- apply_milling_requests (core.milling.features)
- MillingAnalyzer (core.milling.analyzer)

테스트 실행:
    pytest tests/test_milling_adder.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

from core.milling.face_analyzer import FaceDimensionResult
from core.milling.features import (
    MillingParams,
    MillingFeatureRequest,
    compute_hole_scale_range,
    create_blind_hole,
    create_through_hole,
    create_rectangular_pocket,
    create_rectangular_passage,
    apply_milling_requests,
)
from core.milling.analyzer import MillingAnalyzer


# ============================================================================
# 헬퍼
# ============================================================================

def make_turning_shape(stock_radius=10.0, height=20.0, step_radius=7.0, step_height=5.0):
    """Step이 있는 터닝 형상 (원기둥 + 상단 Step)"""
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    stock = BRepPrimAPI_MakeCylinder(axis, stock_radius, height).Shape()

    step_axis = gp_Ax2(gp_Pnt(0, 0, height - step_height), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(step_axis, stock_radius, step_height).Shape()
    inner = BRepPrimAPI_MakeCylinder(step_axis, step_radius, step_height).Shape()
    ring = BRepAlgoAPI_Cut(outer, inner).Shape()

    return BRepAlgoAPI_Cut(stock, ring).Shape()


def make_simple_cylinder(radius=10.0, height=20.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


# ============================================================================
# compute_hole_scale_range 테스트
# ============================================================================

class TestComputeHoleScaleRange:
    def test_valid_range(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=10.0, height=5.0)
        params = MillingParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)

        assert d_min == 0.5
        assert d_max > d_min
        assert d_max == pytest.approx(0.6 * (5.0 - 2 * 0.3))

    def test_zero_width_returns_zero(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=0.0, height=5.0)
        params = MillingParams()
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_none_width_returns_zero(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=None, height=5.0)
        params = MillingParams()
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_none_height_uses_width_only(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=10.0, height=None)
        params = MillingParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_max == pytest.approx(0.6 * (10.0 - 2 * 0.3))

    def test_too_small_effective_dim(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=0.5, height=0.5)
        params = MillingParams(clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_d_max_less_than_d_min(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=2.0, height=2.0)
        params = MillingParams(diameter_min=5.0, diameter_max_ratio=0.6, clearance=0.1)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_uses_min_of_width_height(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=20.0, height=3.0)
        params = MillingParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_max == pytest.approx(0.6 * (3.0 - 2 * 0.3))


# ============================================================================
# 피처 생성 함수 테스트 (순수 도구 형상 생성)
# ============================================================================

class TestFeatureCreation:
    def test_create_blind_hole(self):
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)
        tool = create_blind_hole(center, direction, diameter=3.0, depth=5.0)
        assert tool is not None
        assert not tool.IsNull()

    def test_create_through_hole(self):
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)
        tool = create_through_hole(center, direction, diameter=3.0, available_depth=10.0)
        assert tool is not None
        assert not tool.IsNull()

    def test_create_rectangular_pocket(self):
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)
        tool = create_rectangular_pocket(center, direction, width=3.0, length=4.0, depth=2.0)
        assert tool is not None
        assert not tool.IsNull()

    def test_create_rectangular_passage(self):
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)
        tool = create_rectangular_passage(
            center, direction, width=3.0, length=4.0, available_depth=10.0
        )
        assert tool is not None
        assert not tool.IsNull()

    def test_blind_hole_can_cut_shape(self):
        stock = make_simple_cylinder()
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)
        tool = create_blind_hole(center, direction, diameter=3.0, depth=5.0)
        result = BRepAlgoAPI_Cut(stock, tool).Shape()
        assert result is not None
        assert not result.IsNull()


# ============================================================================
# MillingAnalyzer 테스트
# ============================================================================

class TestMillingAnalyzer:
    def test_analyze_returns_list(self):
        shape = make_turning_shape()
        analyzer = MillingAnalyzer(MillingParams())
        requests = analyzer.analyze(shape)
        assert isinstance(requests, list)

    def test_analyze_respects_max_features(self):
        shape = make_turning_shape()
        analyzer = MillingAnalyzer(MillingParams(
            diameter_min=0.5, diameter_max_ratio=0.5, clearance=0.2
        ))
        requests = analyzer.analyze(shape, max_features=2)
        assert len(requests) <= 2

    def test_analyze_returns_valid_requests(self):
        shape = make_turning_shape()
        analyzer = MillingAnalyzer(MillingParams(
            diameter_min=0.5, diameter_max_ratio=0.5, clearance=0.2
        ))
        requests = analyzer.analyze(shape, max_features=3)

        for req in requests:
            assert isinstance(req, MillingFeatureRequest)
            assert req.feature_type in (
                'blind_hole', 'through_hole', 'rect_pocket', 'rect_passage'
            )
            assert req.depth > 0

    def test_analyze_cylinder_only_no_planes(self):
        shape = make_turning_shape()
        analyzer = MillingAnalyzer()
        requests = analyzer.analyze(shape, target_face_types=["Cylinder"])
        for req in requests:
            assert "Plane" not in req.face_type

    def test_no_valid_faces_returns_empty(self):
        shape = make_simple_cylinder(radius=0.3, height=0.5)
        analyzer = MillingAnalyzer(MillingParams(diameter_min=5.0, clearance=1.0))
        requests = analyzer.analyze(shape)
        assert requests == []

    def test_apply_requests_produces_valid_shape(self):
        shape = make_turning_shape()
        analyzer = MillingAnalyzer(MillingParams(
            diameter_min=0.5, diameter_max_ratio=0.5, clearance=0.2
        ))
        requests = analyzer.analyze(shape, max_features=2)
        result = apply_milling_requests(shape, requests)
        assert result is not None
        assert not result.IsNull()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
