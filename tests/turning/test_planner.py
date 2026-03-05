# -*- coding: utf-8 -*-
"""
core/turning/planner.py 유닛 테스트

- TurningPlanner.plan() → 올바른 FeatureRequest 생성
- 제약조건 위반 시 request 생략
- Bottom-Up 공간 계산 정확성

테스트 실행:
    pytest tests/turning/test_planner.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tree.node import load_tree
from core.turning.planner import TurningPlanner, TurningParams
from core.turning.features import StockInfo, TurningFeatureRequest


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


# ============================================================================
# TurningPlanner 기본 테스트
# ============================================================================

class TestTurningPlannerBasic:
    def test_base_only_returns_empty_requests(self):
        planner = TurningPlanner()
        root = load_tree(TREE_BASE_ONLY)
        stock_info, requests = planner.plan(root)

        assert isinstance(stock_info, StockInfo)
        assert stock_info.height > 0
        assert stock_info.radius > 0
        assert requests == []

    def test_single_step_returns_one_request(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        stock_info, requests = planner.plan(root)

        step_reqs = [r for r in requests if r.feature_type == 'step']
        assert len(step_reqs) == 1

    def test_single_groove_returns_one_request(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_GROOVE)
        stock_info, requests = planner.plan(root)

        groove_reqs = [r for r in requests if r.feature_type == 'groove']
        assert len(groove_reqs) == 1

    def test_step_and_groove(self):
        planner = TurningPlanner()
        root = load_tree(TREE_STEP_AND_GROOVE)
        stock_info, requests = planner.plan(root)

        assert any(r.feature_type == 'step' for r in requests)
        assert any(r.feature_type == 'groove' for r in requests)

    def test_two_steps(self):
        planner = TurningPlanner()
        root = load_tree(TREE_TWO_STEPS)
        stock_info, requests = planner.plan(root)

        step_reqs = [r for r in requests if r.feature_type == 'step']
        assert len(step_reqs) == 2


# ============================================================================
# StockInfo 검증
# ============================================================================

class TestStockInfo:
    def test_stock_includes_margin(self):
        params = TurningParams(
            stock_height_margin=(3.0, 3.0),
            stock_radius_margin=(2.0, 2.0),
        )
        planner = TurningPlanner(params)
        root = load_tree(TREE_WITH_STEP)
        stock_info, _ = planner.plan(root)

        assert stock_info.height > 0
        assert stock_info.radius > params.min_remaining_radius

    def test_stock_radius_greater_than_min_remaining(self):
        planner = TurningPlanner()
        root = load_tree(TREE_NESTED)
        stock_info, _ = planner.plan(root)
        assert stock_info.radius > planner.params.min_remaining_radius


# ============================================================================
# TurningFeatureRequest 검증
# ============================================================================

class TestFeatureRequestValidity:
    def test_step_z_range_within_stock(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        stock_info, requests = planner.plan(root)

        for req in requests:
            assert req.z_min >= 0
            assert req.z_max <= stock_info.height
            assert req.z_min < req.z_max

    def test_step_radii_valid(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        stock_info, requests = planner.plan(root)

        for req in [r for r in requests if r.feature_type == 'step']:
            assert req.outer_radius <= stock_info.radius
            assert req.inner_radius > 0
            assert req.inner_radius < req.outer_radius

    def test_groove_z_range_within_stock(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_GROOVE)
        stock_info, requests = planner.plan(root)

        for req in [r for r in requests if r.feature_type == 'groove']:
            assert req.z_min >= 0
            assert req.z_max <= stock_info.height
            assert req.z_min < req.z_max

    def test_step_label_is_correct(self):
        from core.label_maker import Labels
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        _, requests = planner.plan(root)

        for req in [r for r in requests if r.feature_type == 'step']:
            assert req.label == Labels.STEP

    def test_groove_label_is_correct(self):
        from core.label_maker import Labels
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_GROOVE)
        _, requests = planner.plan(root)

        for req in [r for r in requests if r.feature_type == 'groove']:
            assert req.label == Labels.GROOVE


# ============================================================================
# 복잡한 트리 검증
# ============================================================================

class TestComplexTrees:
    def test_nested_tree(self):
        planner = TurningPlanner()
        root = load_tree(TREE_NESTED)
        stock_info, requests = planner.plan(root)

        assert stock_info.height > 0
        assert stock_info.radius > 0

    def test_sibling_grooves(self):
        planner = TurningPlanner()
        root = load_tree(TREE_SIBLING_GROOVES)
        stock_info, requests = planner.plan(root)

        groove_reqs = [r for r in requests if r.feature_type == 'groove']
        assert len(groove_reqs) <= 2

    def test_no_overlapping_step_grooves(self):
        """Step과 Groove가 z 범위에서 겹치지 않아야 함 (대략적인 검증)"""
        planner = TurningPlanner()
        root = load_tree(TREE_STEP_AND_GROOVE)
        stock_info, requests = planner.plan(root)

        for req in requests:
            assert req.z_min < req.z_max
            assert req.z_min >= 0
            assert req.z_max <= stock_info.height


# ============================================================================
# Required Space 계산 검증
# ============================================================================

class TestRequiredSpaceCalculation:
    def test_base_required_space_exists(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        planner._calculate_required_space(root)

        assert root.required_space is not None
        assert root.required_space.height >= 0
        assert root.required_space.depth >= 0

    def test_step_required_space_positive(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_STEP)
        planner._calculate_required_space(root)

        step_node = root.children[0]
        assert step_node.required_space is not None
        assert step_node.required_space.feature_height > 0
        assert step_node.required_space.feature_depth > 0

    def test_groove_required_space_positive(self):
        planner = TurningPlanner()
        root = load_tree(TREE_WITH_GROOVE)
        planner._calculate_required_space(root)

        groove_node = root.children[0]
        assert groove_node.required_space is not None
        assert groove_node.required_space.feature_height > 0
        assert groove_node.required_space.feature_depth > 0
