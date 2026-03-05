# -*- coding: utf-8 -*-
"""
core/turning/features.py 유닛 테스트

- TurningParams, StockInfo, TurningFeatureRequest 데이터 클래스 검증
- create_step_cut / create_groove_cut 형상 유효성
- create_stock 형상 유효성
- apply_turning_requests 실행 정확성
- apply_edge_features 동작 검증

테스트 실행:
    pytest tests/turning/test_features.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.turning.planner import TurningParams
from core.turning.features import (
    StockInfo,
    TurningFeatureRequest,
    create_step_cut,
    create_groove_cut,
    create_stock,
    apply_turning_requests,
    apply_edge_features,
    collect_circular_edges,
)
from core.label_maker import Labels


# ============================================================================
# 헬퍼
# ============================================================================

def make_cylinder(radius=10.0, height=20.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def count_faces(shape):
    return len(list(TopologyExplorer(shape).faces()))


# ============================================================================
# TurningParams 테스트
# ============================================================================

class TestTurningParams:
    def test_default_params(self):
        p = TurningParams()
        assert p.stock_height_margin == (3.0, 8.0)
        assert p.stock_radius_margin == (2.0, 5.0)
        assert p.step_depth_range == (0.8, 1.5)
        assert p.step_height_range == (2.0, 4.0)
        assert p.step_margin == 0.5
        assert p.groove_depth_range == (0.4, 0.8)
        assert p.groove_width_range == (1.5, 3.0)
        assert p.groove_margin == 0.5
        assert p.min_remaining_radius == 2.0
        assert p.chamfer_range == (0.3, 0.8)
        assert p.fillet_range == (0.3, 0.8)
        assert p.edge_feature_prob == 0.3

    def test_custom_params(self):
        p = TurningParams(step_depth_range=(1.0, 2.0), groove_margin=0.8)
        assert p.step_depth_range == (1.0, 2.0)
        assert p.groove_margin == 0.8


# ============================================================================
# StockInfo 테스트
# ============================================================================

class TestStockInfo:
    def test_fields(self):
        info = StockInfo(height=30.0, radius=15.0)
        assert info.height == 30.0
        assert info.radius == 15.0


# ============================================================================
# TurningFeatureRequest 테스트
# ============================================================================

class TestTurningFeatureRequest:
    def test_step_request(self):
        req = TurningFeatureRequest(
            feature_type='step',
            z_min=5.0, z_max=10.0,
            outer_radius=10.0, inner_radius=8.0,
            label=Labels.STEP
        )
        assert req.feature_type == 'step'
        assert req.z_min == 5.0
        assert req.label == Labels.STEP

    def test_groove_request(self):
        req = TurningFeatureRequest(
            feature_type='groove',
            z_min=3.0, z_max=5.0,
            outer_radius=10.0, inner_radius=9.0,
            label=Labels.GROOVE
        )
        assert req.feature_type == 'groove'


# ============================================================================
# create_stock 테스트
# ============================================================================

class TestCreateStock:
    def test_basic(self):
        info = StockInfo(height=20.0, radius=10.0)
        shape = create_stock(info)
        assert shape is not None
        assert not shape.IsNull()

    def test_face_count(self):
        info = StockInfo(height=20.0, radius=10.0)
        shape = create_stock(info)
        n_faces = count_faces(shape)
        assert n_faces == 3  # 원통: 측면 + 상면 + 하면


# ============================================================================
# create_step_cut 테스트
# ============================================================================

class TestCreateStepCut:
    def test_basic_shape(self):
        shape = create_step_cut(5.0, 10.0, 10.0, 8.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_invalid_z_range(self):
        with pytest.raises(ValueError):
            create_step_cut(10.0, 5.0, 10.0, 8.0)

    def test_equal_z_raises(self):
        with pytest.raises(ValueError):
            create_step_cut(5.0, 5.0, 10.0, 8.0)


# ============================================================================
# create_groove_cut 테스트
# ============================================================================

class TestCreateGrooveCut:
    def test_basic_shape(self):
        shape = create_groove_cut(5.0, 7.0, 10.0, 9.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_invalid_z_range(self):
        with pytest.raises(ValueError):
            create_groove_cut(7.0, 5.0, 10.0, 9.0)


# ============================================================================
# apply_turning_requests 테스트
# ============================================================================

class TestApplyTurningRequests:
    def test_empty_requests(self):
        stock = make_cylinder(10.0, 20.0)
        result = apply_turning_requests(stock, [])
        assert result is not None
        assert not result.IsNull()

    def test_single_step(self):
        stock = make_cylinder(10.0, 20.0)
        req = TurningFeatureRequest(
            feature_type='step',
            z_min=15.0, z_max=20.0,
            outer_radius=10.0, inner_radius=7.0,
            label=Labels.STEP
        )
        result = apply_turning_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()
        assert count_faces(result) > count_faces(stock)

    def test_single_groove(self):
        stock = make_cylinder(10.0, 20.0)
        req = TurningFeatureRequest(
            feature_type='groove',
            z_min=8.0, z_max=10.0,
            outer_radius=10.0, inner_radius=8.5,
            label=Labels.GROOVE
        )
        result = apply_turning_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()

    def test_multiple_requests(self):
        stock = make_cylinder(10.0, 30.0)
        requests = [
            TurningFeatureRequest('step', 25.0, 30.0, 10.0, 7.0, Labels.STEP),
            TurningFeatureRequest('groove', 10.0, 12.0, 10.0, 8.5, Labels.GROOVE),
        ]
        result = apply_turning_requests(stock, requests)
        assert result is not None
        assert not result.IsNull()

    def test_unknown_type_skipped(self):
        stock = make_cylinder(10.0, 20.0)
        req = TurningFeatureRequest(
            feature_type='unknown',
            z_min=5.0, z_max=10.0,
            outer_radius=10.0, inner_radius=8.0,
            label=0
        )
        result = apply_turning_requests(stock, [req])
        assert result is not None
        assert not result.IsNull()


# ============================================================================
# collect_circular_edges 테스트
# ============================================================================

class TestCollectCircularEdges:
    def test_cylinder_has_circular_edges(self):
        shape = make_cylinder(10.0, 20.0)
        edges = collect_circular_edges(shape)
        assert len(edges) > 0

    def test_after_step_cut(self):
        stock = make_cylinder(10.0, 20.0)
        req = TurningFeatureRequest('step', 15.0, 20.0, 10.0, 7.0, Labels.STEP)
        shape = apply_turning_requests(stock, [req])
        edges = collect_circular_edges(shape)
        assert len(edges) > 0


# ============================================================================
# apply_edge_features 테스트
# ============================================================================

class TestApplyEdgeFeatures:
    def test_returns_valid_shape(self):
        stock = make_cylinder(10.0, 20.0)
        req = TurningFeatureRequest('step', 15.0, 20.0, 10.0, 7.0, Labels.STEP)
        shape = apply_turning_requests(stock, [req])

        result = apply_edge_features(shape, edge_feature_prob=1.0,
                                     chamfer_range=(0.3, 0.8), fillet_range=(0.3, 0.8))
        assert result is not None
        assert not result.IsNull()

    def test_no_crash_when_prob_zero(self):
        shape = make_cylinder(10.0, 20.0)
        result = apply_edge_features(shape, edge_feature_prob=0.0,
                                     chamfer_range=(0.3, 0.8), fillet_range=(0.3, 0.8))
        assert result is not None
