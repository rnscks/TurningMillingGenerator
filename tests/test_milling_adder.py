# -*- coding: utf-8 -*-
"""
milling_adder.py 테스트 모듈

피처 스케일 계산, 유효 면 판단, 피처 생성 함수 검증.

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

from core.face_analyzer import FaceDimensionResult
from core.milling_adder import (
    FeatureParams, FeatureType, FeaturePlacement,
    MillingFeatureAdder, compute_hole_scale_range,
    create_blind_hole, create_through_hole,
    create_rectangular_pocket, create_rectangular_passage,
)


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
        params = FeatureParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)

        assert d_min == 0.5
        assert d_max > d_min
        assert d_max == pytest.approx(0.6 * (5.0 - 2 * 0.3))

    def test_zero_width_returns_zero(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=0.0, height=5.0)
        params = FeatureParams()
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_none_width_returns_zero(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=None, height=5.0)
        params = FeatureParams()
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_none_height_uses_width_only(self):
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=10.0, height=None)
        params = FeatureParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_max == pytest.approx(0.6 * (10.0 - 2 * 0.3))

    def test_too_small_effective_dim(self):
        """clearance가 너무 커서 유효 치수 ≤ 0"""
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=0.5, height=0.5)
        params = FeatureParams(clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_d_max_less_than_d_min(self):
        """diameter_min이 계산된 d_max보다 큰 경우"""
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=2.0, height=2.0)
        params = FeatureParams(diameter_min=5.0, diameter_max_ratio=0.6, clearance=0.1)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_uses_min_of_width_height(self):
        """min(W, H)를 사용하는지 확인"""
        dim = FaceDimensionResult(face_id=0, surface_type="Cylinder", width=20.0, height=3.0)
        params = FeatureParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_max == pytest.approx(0.6 * (3.0 - 2 * 0.3))


# ============================================================================
# 피처 생성 함수 테스트
# ============================================================================

class TestFeatureCreation:
    def test_create_blind_hole(self):
        shape = make_simple_cylinder()
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)

        result, op = create_blind_hole(shape, center, direction, diameter=3.0, depth=5.0)
        assert result is not None
        assert op is not None

    def test_create_through_hole(self):
        shape = make_simple_cylinder()
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)

        result, op = create_through_hole(shape, center, direction, diameter=3.0, available_depth=10.0)
        assert result is not None
        assert op is not None

    def test_create_rectangular_pocket(self):
        shape = make_simple_cylinder()
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)

        result, op = create_rectangular_pocket(
            shape, center, direction, width=3.0, length=4.0, depth=2.0
        )
        assert result is not None
        assert op is not None

    def test_create_rectangular_passage(self):
        shape = make_simple_cylinder()
        center = gp_Pnt(10, 0, 10)
        direction = gp_Dir(-1, 0, 0)

        result, op = create_rectangular_passage(
            shape, center, direction, width=3.0, length=4.0, available_depth=10.0
        )
        assert result is not None
        assert op is not None


# ============================================================================
# MillingFeatureAdder 테스트
# ============================================================================

class TestMillingFeatureAdder:
    def test_analyze_faces_returns_infos(self):
        shape = make_turning_shape()
        adder = MillingFeatureAdder(FeatureParams())
        infos = adder.analyze_faces(shape)

        assert len(infos) > 0
        for info in infos:
            assert hasattr(info, 'face_id')
            assert hasattr(info, 'is_valid')

    def test_get_valid_faces_filters_planes(self):
        """유효 면 필터링 시 Plane 제외"""
        shape = make_turning_shape()
        adder = MillingFeatureAdder(FeatureParams())
        adder.analyze_faces(shape)

        valid = adder.get_valid_faces(target_types=["Cylinder"])
        for info in valid:
            assert "Plane" not in info.dimension.surface_type

    def test_add_milling_features_basic(self):
        """기본 밀링 피처 추가"""
        shape = make_turning_shape()
        adder = MillingFeatureAdder(FeatureParams(
            diameter_min=0.5,
            diameter_max_ratio=0.5,
            clearance=0.2,
        ))

        result_shape, placements = adder.add_milling_features(
            shape,
            target_face_types=["Cylinder"],
            max_total_holes=2,
            holes_per_face=1,
        )

        assert result_shape is not None
        assert isinstance(placements, list)

    def test_max_total_holes_respected(self):
        """max_total_holes 제한 준수"""
        shape = make_turning_shape()
        adder = MillingFeatureAdder(FeatureParams(
            diameter_min=0.5,
            diameter_max_ratio=0.5,
            clearance=0.2,
        ))

        _, placements = adder.add_milling_features(
            shape,
            target_face_types=["Cylinder"],
            max_total_holes=1,
            holes_per_face=5,
        )

        assert len(placements) <= 1

    def test_placement_has_required_fields(self):
        """FeaturePlacement에 필수 필드 존재"""
        shape = make_turning_shape()
        adder = MillingFeatureAdder(FeatureParams(
            diameter_min=0.5,
            diameter_max_ratio=0.5,
            clearance=0.2,
        ))

        _, placements = adder.add_milling_features(
            shape,
            target_face_types=["Cylinder"],
            max_total_holes=3,
        )

        for p in placements:
            assert isinstance(p, FeaturePlacement)
            assert isinstance(p.feature_type, FeatureType)
            assert p.depth > 0

    def test_feature_types_filter(self):
        """feature_types 지정 시 해당 타입만 생성"""
        shape = make_turning_shape()
        adder = MillingFeatureAdder(FeatureParams(
            diameter_min=0.5,
            diameter_max_ratio=0.5,
            clearance=0.2,
        ))

        _, placements = adder.add_milling_features(
            shape,
            target_face_types=["Cylinder"],
            max_total_holes=5,
            feature_types=[FeatureType.BLIND_HOLE],
        )

        for p in placements:
            assert p.feature_type == FeatureType.BLIND_HOLE

    def test_no_valid_faces_returns_empty(self):
        """유효한 면이 없으면 빈 리스트"""
        shape = make_simple_cylinder(radius=0.3, height=0.5)
        adder = MillingFeatureAdder(FeatureParams(
            diameter_min=5.0,
            clearance=1.0,
        ))

        result, placements = adder.add_milling_features(
            shape, max_total_holes=5
        )
        assert placements == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
