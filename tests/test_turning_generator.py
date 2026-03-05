# -*- coding: utf-8 -*-
"""
turning 모듈 테스트 (구 turning_generator.py → 신 planner + features)

새 API:
- TurningPlanner (core.turning.planner)
- apply_turning_requests, create_stock (core.turning.features)
- TreeNode, Region, RequiredSpace, load_tree (core.tree.node)

테스트 실행:
    pytest tests/test_turning_generator.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tree.generator import generate_trees
from core.turning.features import create_stock, apply_turning_requests
from core.tree.node import Region, TreeNode, load_tree
from core.turning.planner import TurningPlanner
from core.turning.params import TurningParams


# ============================================================================
# TurningParams 테스트
# ============================================================================

class TestTurningParams:
    def test_default_params(self):
        params = TurningParams()
        assert params.stock_height_margin == (3.0, 8.0)
        assert params.stock_radius_margin == (2.0, 5.0)
        assert params.step_depth_range == (0.8, 1.5)
        assert params.groove_depth_range == (0.4, 0.8)
        assert params.groove_margin == 0.5
        assert params.min_remaining_radius == 2.0
        assert params.step_margin == 0.5

    def test_custom_params(self):
        params = TurningParams(
            step_depth_range=(1.0, 2.0),
            groove_width_range=(2.0, 4.0),
            groove_margin=0.2
        )
        assert params.step_depth_range == (1.0, 2.0)
        assert params.groove_width_range == (2.0, 4.0)
        assert params.groove_margin == 0.2

    def test_edge_feature_params(self):
        params = TurningParams()
        assert params.chamfer_range == (0.3, 0.8)
        assert params.fillet_range == (0.3, 0.8)
        assert params.edge_feature_prob == 0.3


# ============================================================================
# Region 테스트
# ============================================================================

class TestRegion:
    def test_region_height(self):
        region = Region(z_min=5.0, z_max=15.0, radius=10.0)
        assert region.height == 10.0

    def test_region_repr(self):
        region = Region(z_min=0.0, z_max=10.0, radius=5.0, direction='top')
        repr_str = repr(region)
        assert "z=[0.00, 10.00]" in repr_str
        assert "r=5.00" in repr_str

    def test_region_default_direction(self):
        region = Region(z_min=0.0, z_max=10.0, radius=5.0)
        assert region.direction is None



# ============================================================================
# TurningPlanner 기본 테스트
# ============================================================================

def _generate_and_build(tree_data, params=None):
    """헬퍼: plan() 결과로 stock + requests를 반환하고, 형상도 생성"""
    planner = TurningPlanner(params or TurningParams())
    root = load_tree(tree_data)
    stock_info, requests = planner.plan(root)
    shape = create_stock(stock_info)
    shape = apply_turning_requests(shape, requests)
    return shape, stock_info, requests


class TestTurningPlannerBasic:
    def test_planner_creation(self):
        params = TurningParams()
        planner = TurningPlanner(params)
        assert planner.params == params

    def test_load_tree(self):
        trees = generate_trees(n_nodes=3, max_depth=2)
        root = load_tree(trees[0])
        assert root is not None
        assert root.label == 'b'
        assert root.depth == 0

    def test_generate_from_tree(self):
        trees = generate_trees(n_nodes=4, max_depth=3)
        shape, stock_info, requests = _generate_and_build(trees[0])
        assert shape is not None
        assert not shape.IsNull()


# ============================================================================
# Bottom-Up 계산 테스트
# ============================================================================

class TestBottomUpCalculation:
    def test_required_space_computed(self):
        trees = generate_trees(n_nodes=5, max_depth=4)
        planner = TurningPlanner(TurningParams())
        root = load_tree(trees[0])
        planner._calculate_required_space(root)

        def check_all_nodes(node):
            assert node.required_space is not None
            for child in node.children:
                check_all_nodes(child)

        check_all_nodes(root)

    def test_margin_stored_in_required_space(self):
        trees = generate_trees(n_nodes=6, max_depth=8)

        for tree in trees[:50]:
            params = TurningParams()
            planner = TurningPlanner(params)
            root = load_tree(tree)
            planner._calculate_required_space(root)

            def check_margin(node):
                rs = node.required_space
                if node.label == 'b':
                    assert rs.margin == 0.0
                elif node.label == 's':
                    assert rs.margin == params.step_margin
                elif node.label == 'g':
                    assert abs(rs.margin - params.groove_margin) < 1e-6
                for child in node.children:
                    check_margin(child)

            check_margin(root)

    def test_groove_width_sufficient_for_children(self):
        trees = generate_trees(n_nodes=6, max_depth=8)

        for tree in trees[:50]:
            params = TurningParams()
            planner = TurningPlanner(params)
            root = load_tree(tree)
            planner._calculate_required_space(root)

            def check_groove_width(node):
                if node.label == 'g':
                    groove_children = [c for c in node.children if c.label == 'g']
                    if groove_children:
                        children_total = sum(c.required_space.height for c in groove_children)
                        rs = node.required_space
                        usable = rs.feature_height - 2 * rs.margin
                        assert usable >= children_total - 1e-6
                for child in node.children:
                    check_groove_width(child)

            check_groove_width(root)

    def test_base_height_covers_groove_children(self):
        trees = generate_trees(n_nodes=6, max_depth=8)

        for tree in trees[:50]:
            params = TurningParams()
            planner = TurningPlanner(params)
            root = load_tree(tree)
            planner._calculate_required_space(root)

            groove_children = [c for c in root.children if c.label == 'g']
            if groove_children:
                groove_total = sum(c.required_space.height for c in groove_children)
                assert root.required_space.height >= groove_total - 1e-6


# ============================================================================
# Groove 분산 배치 테스트
# ============================================================================

class TestGrooveDistribution:
    def _find_tree_with_sibling_grooves(self):
        trees = generate_trees(n_nodes=6, max_depth=8)
        for tree in trees[:100]:
            nodes = tree['nodes']
            for n in nodes:
                if n['label'] in ['b', 's']:
                    groove_kids = [
                        nodes[cid] for cid in n.get('children', [])
                        if nodes[cid]['label'] == 'g'
                    ]
                    if len(groove_kids) >= 2:
                        return tree
        return None

    def test_sibling_grooves_shape_generation(self):
        tree = self._find_tree_with_sibling_grooves()
        if tree is None:
            pytest.skip("형제 Groove가 있는 트리를 찾지 못함")

        shape, stock_info, requests = _generate_and_build(tree)
        assert shape is not None


# ============================================================================
# Step 방향 테스트
# ============================================================================

class TestStepDirection:
    def _find_tree_with_two_step_children_of_base(self):
        trees = generate_trees(n_nodes=6, max_depth=8)
        for tree in trees[:200]:
            nodes = tree['nodes']
            base_node = [n for n in nodes if n['label'] == 'b'][0]
            step_children = [
                nodes[cid] for cid in base_node.get('children', [])
                if nodes[cid]['label'] == 's'
            ]
            if len(step_children) >= 2:
                return tree
        return None

    def test_bidirectional_step_shape_generation(self):
        tree = self._find_tree_with_two_step_children_of_base()
        if tree is None:
            pytest.skip("Base에 Step 자식이 2개인 트리를 찾지 못함")

        shape, stock_info, requests = _generate_and_build(tree)
        assert shape is not None


# ============================================================================
# 중첩 Groove 테스트
# ============================================================================

class TestNestedGrooves:
    def _find_tree_with_nested_grooves(self):
        trees = generate_trees(n_nodes=6, max_depth=8)
        for tree in trees[:100]:
            nodes = tree['nodes']
            for n in nodes:
                if n['label'] == 'g':
                    groove_kids = [
                        nodes[cid] for cid in n.get('children', [])
                        if nodes[cid]['label'] == 'g'
                    ]
                    if len(groove_kids) >= 1:
                        return tree
        return None

    def test_nested_grooves_shape_generation(self):
        tree = self._find_tree_with_nested_grooves()
        if tree is None:
            pytest.skip("중첩 Groove가 있는 트리를 찾지 못함")

        shape, stock_info, requests = _generate_and_build(tree)
        assert shape is not None


# ============================================================================
# 형상 생성 통합 테스트
# ============================================================================

class TestShapeGeneration:
    def test_generate_multiple_shapes(self):
        trees = generate_trees(n_nodes=6, max_depth=4)[:10]
        params = TurningParams()

        success_count = 0
        for tree in trees:
            try:
                shape, _, _ = _generate_and_build(tree, params)
                if shape is not None and not shape.IsNull():
                    success_count += 1
            except Exception as e:
                print(f"Error: {e}")

        assert success_count >= len(trees) * 0.8

    def test_stock_dimensions_reasonable(self):
        trees = generate_trees(n_nodes=5, max_depth=3)[:5]
        params = TurningParams()

        for tree in trees:
            shape, stock_info, _ = _generate_and_build(tree, params)
            if shape is not None:
                assert stock_info.height > 5.0
                assert stock_info.height < 100.0
                assert stock_info.radius > 3.0
                assert stock_info.radius < 50.0


# ============================================================================
# Groove 검증 테스트
# ============================================================================

class TestGrooveValidation:
    def test_groove_retry_on_failure(self):
        trees = generate_trees(n_nodes=6, max_depth=8)

        for tree in trees[:100]:
            nodes = tree['nodes']
            for n in nodes:
                if n['label'] in ['b', 's']:
                    groove_kids = [
                        nodes[cid] for cid in n.get('children', [])
                        if nodes[cid]['label'] == 'g'
                    ]
                    if len(groove_kids) >= 2:
                        shape, _, _ = _generate_and_build(tree)
                        assert shape is not None
                        return

        pytest.skip("형제 Groove 트리를 찾지 못함")


# ============================================================================
# Margin 일관성 테스트
# ============================================================================

class TestMarginConsistency:
    def test_groove_margin_based(self):
        params = TurningParams(groove_margin=0.2)
        planner = TurningPlanner(params)

        trees = generate_trees(n_nodes=6, max_depth=8)
        for tree in trees[:30]:
            root = load_tree(tree)
            planner._calculate_required_space(root)

            def check_groove_margin(node):
                if node.label == 'g':
                    rs = node.required_space
                    assert abs(rs.margin - 0.2) < 1e-6
                for child in node.children:
                    check_groove_margin(child)

            check_groove_margin(root)

    def test_groove_zone_equals_height(self):
        params = TurningParams()
        planner = TurningPlanner(params)

        trees = generate_trees(n_nodes=6, max_depth=8)
        for tree in trees[:30]:
            root = load_tree(tree)
            planner._calculate_required_space(root)

            def check_zone(node):
                if node.label == 'g' and len(node.children) == 0:
                    rs = node.required_space
                    expected = rs.feature_height + 2 * rs.margin
                    assert abs(rs.height - expected) < 1e-6
                for child in node.children:
                    check_zone(child)

            check_zone(root)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
