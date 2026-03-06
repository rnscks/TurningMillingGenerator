# -*- coding: utf-8 -*-
"""
core/turning/planner.py 유닛 테스트

- TurningPlanner.plan_and_apply() → 올바른 형상 생성
- TurningTreePlanner.plan() → Top-Down BFS로 node.region 확정
- TurningShapeBuilder.build() → 형상 생성
- z 범위 충돌 검사

테스트 실행:
    pytest tests/turning/test_planner.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tree.node import load_tree, Region
from core.turning.planner import (
    TurningPlanner, TurningTreePlanner,
)
from core.turning.builder import TurningShapeBuilder
from core.turning.params import TurningParams
from OCC.Extend.TopologyUtils import TopologyExplorer


# ============================================================================
# 테스트용 트리 데이터
# ============================================================================

TREE_BASE_ONLY = {
    "N": 1,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [], "depth": 0}
    ]
}

TREE_WITH_STEP = {
    "N": 2,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
    ]
}

TREE_WITH_GROOVE = {
    "N": 2,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1], "depth": 0},
        {"id": 1, "label": "g", "parent": 0, "children": [], "depth": 1},
    ]
}

TREE_STEP_AND_GROOVE = {
    "N": 3,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1, 2], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
        {"id": 2, "label": "g", "parent": 0, "children": [], "depth": 1},
    ]
}

TREE_TWO_STEPS = {
    "N": 3,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1, 2], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
        {"id": 2, "label": "s", "parent": 0, "children": [], "depth": 1},
    ]
}

TREE_NESTED = {
    "N": 4,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [2, 3], "depth": 1},
        {"id": 2, "label": "s", "parent": 1, "children": [], "depth": 2},
        {"id": 3, "label": "g", "parent": 1, "children": [], "depth": 2},
    ]
}

TREE_SIBLING_GROOVES = {
    "N": 4,
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1, 2, 3], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
        {"id": 2, "label": "g", "parent": 0, "children": [], "depth": 1},
        {"id": 3, "label": "g", "parent": 0, "children": [], "depth": 1},
    ]
}


def count_faces(shape):
    return len(list(TopologyExplorer(shape).faces()))


# ============================================================================
# TurningPlanner 기본 테스트 (plan_and_apply 통합)
# ============================================================================

class TestTurningPlannerBasic:
    def test_base_only_returns_valid_shape(self):
        planner = TurningPlanner()
        root = load_tree(TREE_BASE_ONLY)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert stock_h > 0
        assert stock_r > 0
        assert shape is not None
        assert not shape.IsNull()

    def test_single_step_adds_faces(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert shape is not None
        assert not shape.IsNull()
        assert count_faces(shape) > 3

    def test_single_groove_adds_faces(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_GROOVE)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert shape is not None
        assert not shape.IsNull()
        assert count_faces(shape) > 3

    def test_step_and_groove(self):
        planner = TurningPlanner()
        root = load_tree(TREE_STEP_AND_GROOVE)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert shape is not None
        assert not shape.IsNull()

    def test_two_steps(self):
        planner = TurningPlanner()
        root = load_tree(TREE_TWO_STEPS)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert shape is not None
        assert not shape.IsNull()


# ============================================================================
# Stock 크기 검증
# ============================================================================

class TestStockSize:
    def test_stock_within_range(self):
        params = TurningParams(
            stock_height_range=(10.0, 10.0),
            stock_radius_range=(5.0, 5.0),
        )
        planner = TurningPlanner(params)
        root = load_tree(TREE_WITH_STEP)
        stock_h, stock_r, _ = planner.plan_and_apply(root)

        assert stock_h == pytest.approx(10.0, abs=1e-9)
        assert stock_r == pytest.approx(5.0, abs=1e-9)

    def test_stock_radius_greater_than_min_remaining(self):
        planner = TurningPlanner()
        root = load_tree(TREE_NESTED)
        stock_h, stock_r, _ = planner.plan_and_apply(root)
        assert stock_r > planner.params.min_remaining_radius


# ============================================================================
# 복잡한 트리 검증
# ============================================================================

class TestComplexTrees:
    def test_nested_tree(self):
        planner = TurningPlanner()
        root = load_tree(TREE_NESTED)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert stock_h > 0
        assert stock_r > 0
        assert shape is not None

    def test_sibling_grooves(self):
        planner = TurningPlanner()
        root = load_tree(TREE_SIBLING_GROOVES)
        stock_h, stock_r, shape = planner.plan_and_apply(root)

        assert shape is not None
        assert not shape.IsNull()


# ============================================================================
# z 범위 충돌 검사 검증
# ============================================================================



# ============================================================================
# TurningTreePlanner: Top-Down BFS region 확정 테스트
# ============================================================================

class TestTurningTreePlannerPlan:
    def test_plan_returns_positive_stock_size(self):
        tp = TurningTreePlanner()
        root = load_tree(TREE_WITH_STEP)
        stock_h, stock_r = tp.plan(root)

        assert stock_h > 0
        assert stock_r > 0

    def test_plan_within_stock_range(self):
        params = TurningParams(
            stock_height_range=(10.0, 20.0),
            stock_radius_range=(5.0, 10.0),
        )
        tp = TurningTreePlanner(params)
        root = load_tree(TREE_WITH_STEP)
        stock_h, stock_r = tp.plan(root)

        assert 10.0 <= stock_h <= 20.0
        assert 5.0 <= stock_r <= 10.0

    def test_plan_fills_root_region(self):
        tp = TurningTreePlanner()
        root = load_tree(TREE_WITH_STEP)
        stock_h, stock_r = tp.plan(root)

        assert root.region is not None
        assert root.region.z_min == pytest.approx(0.0)
        assert root.region.z_max == pytest.approx(stock_h)
        assert root.region.radius == pytest.approx(stock_r)

    def test_plan_fills_step_region(self):
        tp = TurningTreePlanner()
        root = load_tree(TREE_WITH_STEP)
        tp.plan(root)

        step = root.children[0]
        assert step.region is not None
        assert isinstance(step.region, Region)

    def test_plan_fills_groove_region(self):
        tp = TurningTreePlanner()
        root = load_tree(TREE_WITH_GROOVE)
        tp.plan(root)

        groove = root.children[0]
        assert groove.region is not None
        assert isinstance(groove.region, Region)

    def test_step_region_within_stock(self):
        """Step region이 stock 범위를 벗어나지 않아야 함."""
        params = TurningParams(
            stock_height_range=(20.0, 20.0),
            stock_radius_range=(8.0, 8.0),
        )
        tp = TurningTreePlanner(params)
        root = load_tree(TREE_WITH_STEP)
        stock_h, stock_r = tp.plan(root)

        step = root.children[0]
        if step.region is not None:
            assert step.region.z_min >= 0.0
            assert step.region.z_max <= stock_h
            assert step.region.radius < stock_r

    def test_groove_region_within_parent(self):
        """Groove region이 부모(stock) 범위 안에 있어야 함."""
        tp = TurningTreePlanner()
        root = load_tree(TREE_WITH_GROOVE)
        stock_h, stock_r = tp.plan(root)

        groove = root.children[0]
        if groove.region is not None:
            assert groove.region.z_min >= 0.0
            assert groove.region.z_max <= stock_h
            assert groove.region.radius < stock_r

    def test_step_direction_set(self):
        """Step region의 direction이 설정되어 있어야 함."""
        tp = TurningTreePlanner()
        root = load_tree(TREE_WITH_STEP)
        tp.plan(root)

        step = root.children[0]
        if step.region is not None:
            assert step.region.direction in ('top', 'bottom')

    def test_sibling_grooves_in_different_slots(self):
        """형제 groove들이 서로 다른 위치에 배치되어야 함."""
        tp = TurningTreePlanner()
        root = load_tree(TREE_SIBLING_GROOVES)
        tp.plan(root)

        grooves = [c for c in root.children if c.label == 'g']
        regions = [g.region for g in grooves if g.region is not None]
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                overlap = regions[i].z_min < regions[j].z_max and regions[i].z_max > regions[j].z_min
                assert not overlap, \
                    f"형제 groove region 겹침: {regions[i]} vs {regions[j]}"


# ============================================================================
# TurningShapeBuilder.build() 단위 테스트
# ============================================================================

class TestTurningShapeBuilder:
    def _prepare_builder(self, tree_data, params=None):
        params = params or TurningParams()
        tp = TurningTreePlanner(params)
        root = load_tree(tree_data)
        stock_h, stock_r = tp.plan(root)
        builder = TurningShapeBuilder(stock_h, stock_r, params)
        return builder, root

    def test_build_returns_valid_shape(self):
        builder, root = self._prepare_builder(TREE_BASE_ONLY)
        shape = builder.build(root)
        assert shape is not None
        assert not shape.IsNull()

    def test_build_step_adds_faces(self):
        builder, root = self._prepare_builder(TREE_WITH_STEP)
        shape = builder.build(root)
        assert count_faces(shape) > 3

    def test_build_groove_adds_faces(self):
        builder, root = self._prepare_builder(TREE_WITH_GROOVE)
        shape = builder.build(root)
        assert count_faces(shape) > 3

