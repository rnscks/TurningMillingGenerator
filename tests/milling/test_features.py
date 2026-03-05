# -*- coding: utf-8 -*-
"""
core/milling/features.py 유닛 테스트

- MillingParams, MillingFeatureRequest 데이터 클래스 검증
- compute_hole_scale_range 정확성
- create_blind_hole / create_through_hole / create_rectangular_pocket 형상 유효성
- apply_milling_requests 실행 정확성

테스트 실행:
    pytest tests/milling/test_features.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Extend.TopologyUtils import TopologyExplorer

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
from core.label_maker import Labels


# ============================================================================
# 헬퍼
# ============================================================================

def make_cylinder(radius=10.0, height=20.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def make_turning_shape(stock_radius=10.0, height=20.0, step_radius=7.0, step_height=5.0):
    """Step이 있는 터닝 형상"""
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    stock = BRepPrimAPI_MakeCylinder(axis, stock_radius, height).Shape()

    step_axis = gp_Ax2(gp_Pnt(0, 0, height - step_height), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(step_axis, stock_radius, step_height).Shape()
    inner = BRepPrimAPI_MakeCylinder(step_axis, step_radius, step_height).Shape()
    ring = BRepAlgoAPI_Cut(outer, inner).Shape()

    return BRepAlgoAPI_Cut(stock, ring).Shape()


def count_faces(shape):
    return len(list(TopologyExplorer(shape).faces()))


# ============================================================================
# MillingParams 테스트
# ============================================================================

class TestMillingParams:
    def test_default_params(self):
        p = MillingParams()
        assert p.diameter_min == 0.5
        assert p.diameter_max_ratio == 0.6
        assert p.clearance == 0.3
        assert p.depth_ratio == 1.5
        assert p.through_extra == 2.0
        assert p.min_spacing == 1.5
        assert p.max_features_per_face == 3
        assert p.rect_aspect_min == 0.5
        assert p.rect_aspect_max == 2.0

    def test_custom_params(self):
        p = MillingParams(diameter_min=1.0, clearance=0.5)
        assert p.diameter_min == 1.0
        assert p.clearance == 0.5


# ============================================================================
# MillingFeatureRequest 테스트
# ============================================================================

class TestMillingFeatureRequest:
    def test_blind_hole_request(self):
        req = MillingFeatureRequest(
            feature_type='blind_hole',
            center=gp_Pnt(10.0, 0.0, 10.0),
            direction=gp_Dir(-1, 0, 0),
            diameter=2.0,
            depth=3.0,
            label=Labels.BLIND_HOLE,
        )
        assert req.feature_type == 'blind_hole'
        assert req.diameter == 2.0
        assert req.is_through is False

    def test_through_hole_request(self):
        req = MillingFeatureRequest(
            feature_type='through_hole',
            center=gp_Pnt(10.0, 0.0, 10.0),
            direction=gp_Dir(-1, 0, 0),
            diameter=2.0,
            depth=12.0,
            is_through=True,
            label=Labels.THROUGH_HOLE,
        )
        assert req.is_through is True


# ============================================================================
# compute_hole_scale_range 테스트
# ============================================================================

class _FakeDim:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class TestComputeHoleScaleRange:
    def test_valid_range(self):
        dim = _FakeDim(10.0, 5.0)
        params = MillingParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.5
        assert d_max > 0

    def test_zero_width(self):
        dim = _FakeDim(0.0, 5.0)
        params = MillingParams()
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_too_small_for_clearance(self):
        dim = _FakeDim(0.4, 0.4)
        params = MillingParams(clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_max == 0.0

    def test_none_width(self):
        dim = _FakeDim(None, 5.0)
        params = MillingParams()
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_min == 0.0
        assert d_max == 0.0

    def test_no_height_uses_width_only(self):
        dim = _FakeDim(10.0, 0.0)
        params = MillingParams(diameter_min=0.5, diameter_max_ratio=0.6, clearance=0.3)
        d_min, d_max = compute_hole_scale_range(dim, params)
        assert d_max > 0


# ============================================================================
# create_blind_hole 테스트
# ============================================================================

class TestCreateBlindHole:
    def test_returns_valid_shape(self):
        center = gp_Pnt(10.0, 0.0, 10.0)
        direction = gp_Dir(-1, 0, 0)
        shape = create_blind_hole(center, direction, 2.0, 3.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_can_be_used_for_cut(self):
        stock = make_cylinder(10.0, 20.0)
        center = gp_Pnt(10.0, 0.0, 10.0)
        direction = gp_Dir(-1, 0, 0)
        tool = create_blind_hole(center, direction, 2.0, 3.0)
        result = BRepAlgoAPI_Cut(stock, tool).Shape()
        assert result is not None
        assert not result.IsNull()


# ============================================================================
# create_through_hole 테스트
# ============================================================================

class TestCreateThroughHole:
    def test_returns_valid_shape(self):
        center = gp_Pnt(10.0, 0.0, 10.0)
        direction = gp_Dir(-1, 0, 0)
        shape = create_through_hole(center, direction, 2.0, available_depth=10.0)
        assert shape is not None
        assert not shape.IsNull()


# ============================================================================
# create_rectangular_pocket 테스트
# ============================================================================

class TestCreateRectangularPocket:
    def test_returns_valid_shape(self):
        center = gp_Pnt(10.0, 0.0, 10.0)
        direction = gp_Dir(-1, 0, 0)
        shape = create_rectangular_pocket(center, direction, 2.0, 3.0, 2.0)
        assert shape is not None
        assert not shape.IsNull()


# ============================================================================
# create_rectangular_passage 테스트
# ============================================================================

class TestCreateRectangularPassage:
    def test_returns_valid_shape(self):
        center = gp_Pnt(10.0, 0.0, 10.0)
        direction = gp_Dir(-1, 0, 0)
        shape = create_rectangular_passage(center, direction, 2.0, 3.0, 10.0)
        assert shape is not None
        assert not shape.IsNull()


# ============================================================================
# apply_milling_requests 테스트
# ============================================================================

class TestApplyMillingRequests:
    def test_empty_requests(self):
        stock = make_turning_shape()
        result = apply_milling_requests(stock, [])
        assert result is not None
        assert not result.IsNull()

    def test_blind_hole(self):
        stock = make_turning_shape()
        req = MillingFeatureRequest(
            feature_type='blind_hole',
            center=gp_Pnt(10.0, 0.0, 10.0),
            direction=gp_Dir(-1, 0, 0),
            diameter=2.0,
            depth=3.0,
            label=Labels.BLIND_HOLE,
        )
        result = apply_milling_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()
        assert count_faces(result) > count_faces(stock)

    def test_through_hole(self):
        stock = make_turning_shape()
        req = MillingFeatureRequest(
            feature_type='through_hole',
            center=gp_Pnt(10.0, 0.0, 10.0),
            direction=gp_Dir(-1, 0, 0),
            diameter=2.0,
            depth=12.0,
            is_through=True,
            label=Labels.THROUGH_HOLE,
        )
        result = apply_milling_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()

    def test_rect_pocket(self):
        stock = make_turning_shape()
        req = MillingFeatureRequest(
            feature_type='rect_pocket',
            center=gp_Pnt(10.0, 0.0, 10.0),
            direction=gp_Dir(-1, 0, 0),
            width=2.0,
            length=3.0,
            depth=2.0,
            label=Labels.RECTANGULAR_POCKET,
        )
        result = apply_milling_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()

    def test_unknown_type_skipped(self):
        stock = make_turning_shape()
        req = MillingFeatureRequest(
            feature_type='unknown',
            center=gp_Pnt(10.0, 0.0, 10.0),
            direction=gp_Dir(-1, 0, 0),
            label=0,
        )
        result = apply_milling_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()

    def test_multiple_requests(self):
        stock = make_turning_shape()
        requests = [
            MillingFeatureRequest(
                feature_type='blind_hole',
                center=gp_Pnt(10.0, 0.0, 10.0),
                direction=gp_Dir(-1, 0, 0),
                diameter=2.0, depth=3.0,
                label=Labels.BLIND_HOLE,
            ),
            MillingFeatureRequest(
                feature_type='rect_pocket',
                center=gp_Pnt(-10.0, 0.0, 10.0),
                direction=gp_Dir(1, 0, 0),
                width=2.0, length=3.0, depth=2.0,
                label=Labels.RECTANGULAR_POCKET,
            ),
        ]
        result = apply_milling_requests(stock, requests)
        assert result is not None
        assert not result.IsNull()
