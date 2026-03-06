# -*- coding: utf-8 -*-
"""
core/turning/features.py 유닛 테스트

- create_step_cut / create_groove_cut 형상 유효성
- create_stock 형상 유효성
- apply_step_cut / apply_groove_cut 실행 정확성
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

from core.turning.params import TurningParams
from core.turning.features import (
    create_stock,
    create_step_cut,
    create_groove_cut,
    apply_step_cut,
    apply_groove_cut,
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
        assert p.stock_height_range == (10.0, 15.0)
        assert p.stock_radius_range == (6.0, 12.0)
        assert p.step_depth_range == (0.8, 1.5)
        assert p.step_margin == 0.5
        assert p.groove_depth_range == (0.6, 1.2)
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
# create_stock 테스트
# ============================================================================

class TestCreateStock:
    def test_basic(self):
        shape = create_stock(height=20.0, radius=10.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_face_count(self):
        shape = create_stock(height=20.0, radius=10.0)
        n_faces = count_faces(shape)
        assert n_faces == 3  # 원통: 측면 + 상면 + 하면


# ============================================================================
# create_step_cut 테스트
# ============================================================================

class TestCreateStepCut:
    def test_top_direction(self):
        shape = create_step_cut(zpos=10.0, direction='top', outer_r=10.0, inner_r=8.0, stock_height=20.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_bottom_direction(self):
        shape = create_step_cut(zpos=10.0, direction='bottom', outer_r=10.0, inner_r=8.0, stock_height=20.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            create_step_cut(zpos=10.0, direction='left', outer_r=10.0, inner_r=8.0, stock_height=20.0)

    def test_top_zpos_at_stock_top_raises(self):
        with pytest.raises(ValueError):
            create_step_cut(zpos=20.0, direction='top', outer_r=10.0, inner_r=8.0, stock_height=20.0)

    def test_bottom_zpos_at_zero_raises(self):
        with pytest.raises(ValueError):
            create_step_cut(zpos=0.0, direction='bottom', outer_r=10.0, inner_r=8.0, stock_height=20.0)


# ============================================================================
# create_groove_cut 테스트
# ============================================================================

class TestCreateGrooveCut:
    def test_basic_shape(self):
        shape = create_groove_cut(zpos=5.0, width=2.0, outer_r=10.0, inner_r=9.0)
        assert shape is not None
        assert not shape.IsNull()

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            create_groove_cut(zpos=5.0, width=0.0, outer_r=10.0, inner_r=9.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            create_groove_cut(zpos=5.0, width=-1.0, outer_r=10.0, inner_r=9.0)


# ============================================================================
# apply_step_cut / apply_groove_cut 테스트
# ============================================================================

class TestApplyStepCut:
    def test_top_adds_faces(self):
        stock = make_cylinder(10.0, 20.0)
        # direction=top, zpos=10, stock_height=20 → z=[10, 20]
        result = apply_step_cut(stock, z_cut_min=10.0, z_cut_max=20.0, outer_radius=10.0, inner_radius=7.0)
        assert result is not None
        assert not result.IsNull()
        assert count_faces(result) > count_faces(stock)

    def test_bottom_adds_faces(self):
        stock = make_cylinder(10.0, 20.0)
        # direction=bottom, zpos=10 → z=[0, 10]
        result = apply_step_cut(stock, z_cut_min=0.0, z_cut_max=10.0, outer_radius=10.0, inner_radius=7.0)
        assert result is not None
        assert not result.IsNull()
        assert count_faces(result) > count_faces(stock)

    def test_zero_height_raises(self):
        stock = make_cylinder(10.0, 20.0)
        with pytest.raises(ValueError):
            # z_cut_min == z_cut_max → height=0
            apply_step_cut(stock, z_cut_min=20.0, z_cut_max=20.0, outer_radius=10.0, inner_radius=7.0)

    def test_generates_exactly_2_new_faces(self):
        """Step Boolean Cut은 정상 시 inner 원통면 + 링 평면 = 2개의 새 face를 생성해야 함."""
        from core.design_operation import DesignOperation
        from core.turning.features import create_step_cut
        stock = make_cylinder(10.0, 20.0)
        cut = create_step_cut(z_cut_min=10.0, z_cut_max=20.0, outer_r=10.0, inner_r=7.0)
        op = DesignOperation(stock)
        op.cut(cut)
        assert len(op.generated_faces) == 2, f"Step generated_faces={len(op.generated_faces)}, expected=2"


class TestApplyGrooveCut:
    def test_adds_faces(self):
        stock = make_cylinder(10.0, 20.0)
        result = apply_groove_cut(stock, zpos=8.0, width=2.0, outer_radius=10.0, inner_radius=8.5)
        assert result is not None
        assert not result.IsNull()
        assert count_faces(result) > count_faces(stock)

    def test_zero_width_raises(self):
        stock = make_cylinder(10.0, 20.0)
        with pytest.raises(ValueError):
            apply_groove_cut(stock, zpos=8.0, width=0.0, outer_radius=10.0, inner_radius=8.5)

    def test_generates_exactly_3_new_faces(self):
        """Groove Boolean Cut은 정상 시 inner 원통면 + 상단 링 + 하단 링 = 3개의 새 face를 생성해야 함."""
        from core.design_operation import DesignOperation
        from core.turning.features import create_groove_cut
        stock = make_cylinder(10.0, 20.0)
        cut = create_groove_cut(zpos=8.0, width=2.0, outer_r=10.0, inner_r=8.5)
        op = DesignOperation(stock)
        op.cut(cut)
        assert len(op.generated_faces) == 3, f"Groove generated_faces={len(op.generated_faces)}, expected=3"


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
        shape = apply_step_cut(stock, z_cut_min=10.0, z_cut_max=20.0, outer_radius=10.0, inner_radius=7.0)
        edges = collect_circular_edges(shape)
        assert len(edges) > 0


# ============================================================================
# apply_edge_features 테스트
# ============================================================================

class TestApplyEdgeFeatures:
    def test_returns_valid_shape(self):
        stock = make_cylinder(10.0, 20.0)
        shape = apply_step_cut(stock, z_cut_min=10.0, z_cut_max=20.0, outer_radius=10.0, inner_radius=7.0)

        result = apply_edge_features(shape, edge_feature_prob=1.0,
                                     chamfer_range=(0.3, 0.8), fillet_range=(0.3, 0.8))
        assert result is not None
        assert not result.IsNull()

    def test_no_crash_when_prob_zero(self):
        shape = make_cylinder(10.0, 20.0)
        result = apply_edge_features(shape, edge_feature_prob=0.0,
                                     chamfer_range=(0.3, 0.8), fillet_range=(0.3, 0.8))
        assert result is not None
