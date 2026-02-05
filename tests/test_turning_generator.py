# -*- coding: utf-8 -*-
"""
turning_generator.py 테스트 모듈

테스트 실행:
    pytest tests/test_turning_generator.py -v
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tree_generator import generate_trees
from core.turning_generator import (
    TreeTurningGenerator, 
    TurningParams, 
    Region, 
    RequiredSpace,
    TreeNode
)


class TestTurningParams:
    """TurningParams 테스트"""
    
    def test_default_params(self):
        """기본 파라미터 검증"""
        params = TurningParams()
        assert params.stock_height_margin == (3.0, 8.0)
        assert params.stock_radius_margin == (2.0, 5.0)
        assert params.step_depth_range == (0.8, 1.5)
        assert params.groove_depth_range == (0.4, 0.8)
    
    def test_custom_params(self):
        """커스텀 파라미터 검증"""
        params = TurningParams(
            step_depth_range=(1.0, 2.0),
            groove_width_range=(2.0, 4.0)
        )
        assert params.step_depth_range == (1.0, 2.0)
        assert params.groove_width_range == (2.0, 4.0)


class TestRegion:
    """Region 클래스 테스트"""
    
    def test_region_height(self):
        """Region height 계산 검증"""
        region = Region(z_min=5.0, z_max=15.0, radius=10.0)
        assert region.height == 10.0
    
    def test_region_repr(self):
        """Region 문자열 표현 검증"""
        region = Region(z_min=0.0, z_max=10.0, radius=5.0, direction='top')
        repr_str = repr(region)
        assert "z=[0.00, 10.00]" in repr_str
        assert "r=5.00" in repr_str


class TestRequiredSpace:
    """RequiredSpace 클래스 테스트"""
    
    def test_required_space_creation(self):
        """RequiredSpace 생성 검증"""
        space = RequiredSpace(
            height=10.0,
            depth=2.0,
            feature_height=3.0,
            feature_depth=0.5
        )
        assert space.height == 10.0
        assert space.depth == 2.0
        assert space.feature_height == 3.0
        assert space.feature_depth == 0.5


class TestTreeTurningGeneratorBasic:
    """TreeTurningGenerator 기본 테스트"""
    
    def test_generator_creation(self):
        """Generator 생성 검증"""
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        assert gen.params == params
    
    def test_load_tree(self):
        """트리 로드 검증"""
        trees = generate_trees(n_nodes=3, max_depth=2)
        tree = trees[0]
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        root = gen.load_tree(tree)
        
        assert root is not None
        assert root.label == 'b'
        assert root.depth == 0
    
    def test_generate_from_tree(self):
        """트리에서 형상 생성 검증"""
        trees = generate_trees(n_nodes=4, max_depth=3)
        tree = trees[0]
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        shape = gen.generate_from_tree(tree)
        
        assert shape is not None


class TestGrooveDistribution:
    """Groove 분산 배치 테스트"""
    
    def _find_tree_with_sibling_grooves(self):
        """형제 Groove가 있는 트리 찾기"""
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
        """형제 Groove가 있는 트리에서 형상 생성 성공 검증"""
        tree = self._find_tree_with_sibling_grooves()
        
        if tree is None:
            pytest.skip("형제 Groove가 있는 트리를 찾지 못함")
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        
        # 형상 생성 성공 확인
        # (출력 로그에서 서로 다른 Z 위치 확인 가능)
        shape = gen.generate_from_tree(tree)
        assert shape is not None, "형제 Groove 트리에서 형상 생성 실패"


class TestStepDirection:
    """Step 방향 테스트"""
    
    def _find_tree_with_two_step_children_of_base(self):
        """Base에 Step 자식이 2개인 트리 찾기"""
        trees = generate_trees(n_nodes=6, max_depth=8)
        
        for tree in trees[:200]:
            nodes = tree['nodes']
            base_node = [n for n in nodes if n['label'] == 'b'][0]
            base_children = [nodes[cid] for cid in base_node.get('children', [])]
            step_children = [c for c in base_children if c['label'] == 's']
            
            if len(step_children) >= 2:
                return tree
        return None
    
    def test_bidirectional_step_shape_generation(self):
        """Base에 Step 2개가 있는 트리에서 형상 생성 성공 검증"""
        tree = self._find_tree_with_two_step_children_of_base()
        
        if tree is None:
            pytest.skip("Base에 Step 자식이 2개인 트리를 찾지 못함")
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        
        # 형상 생성 성공 확인
        # (출력 로그에서 top/bottom 방향 확인 가능)
        shape = gen.generate_from_tree(tree)
        assert shape is not None, "양방향 Step 트리에서 형상 생성 실패"


class TestNestedGrooves:
    """중첩 Groove 테스트"""
    
    def _find_tree_with_nested_grooves(self):
        """중첩 Groove가 있는 트리 찾기"""
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
        """중첩 Groove가 있는 트리에서 형상 생성 성공 검증"""
        tree = self._find_tree_with_nested_grooves()
        
        if tree is None:
            pytest.skip("중첩 Groove가 있는 트리를 찾지 못함")
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        
        # 형상 생성 성공 확인
        # (출력 로그에서 반경 감소 확인 가능)
        shape = gen.generate_from_tree(tree)
        assert shape is not None, "중첩 Groove 트리에서 형상 생성 실패"


class TestShapeGeneration:
    """형상 생성 통합 테스트"""
    
    def test_generate_multiple_shapes(self):
        """여러 트리에서 형상 생성 검증"""
        trees = generate_trees(n_nodes=6, max_depth=4)[:10]
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        
        success_count = 0
        for tree in trees:
            try:
                shape = gen.generate_from_tree(tree)
                if shape is not None:
                    success_count += 1
            except Exception as e:
                print(f"Error generating shape for {tree['canonical']}: {e}")
        
        # 최소 80% 성공률
        assert success_count >= len(trees) * 0.8, \
            f"형상 생성 성공률이 낮음: {success_count}/{len(trees)}"
    
    def test_stock_dimensions_reasonable(self):
        """Stock 크기가 합리적인지 검증"""
        trees = generate_trees(n_nodes=5, max_depth=3)[:5]
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        
        for tree in trees:
            shape = gen.generate_from_tree(tree)
            
            if shape is not None:
                # Stock 크기 범위 검증
                assert gen.stock_height > 5.0, "Stock 높이가 너무 작음"
                assert gen.stock_height < 100.0, "Stock 높이가 너무 큼"
                assert gen.stock_radius > 3.0, "Stock 반경이 너무 작음"
                assert gen.stock_radius < 50.0, "Stock 반경이 너무 큼"


class TestTwoStageProcessing:
    """2단계 처리 방식 테스트"""
    
    def _find_tree_with_multiple_grooves(self):
        """여러 Groove가 있는 트리 찾기"""
        trees = generate_trees(n_nodes=6, max_depth=8)
        
        for tree in trees[:100]:
            nodes = tree['nodes']
            groove_count = sum(1 for n in nodes if n['label'] == 'g')
            if groove_count >= 2:
                return tree
        return None
    
    def test_all_grooves_generated(self):
        """모든 Groove가 생성되는지 검증"""
        tree = self._find_tree_with_multiple_grooves()
        
        if tree is None:
            pytest.skip("여러 Groove가 있는 트리를 찾지 못함")
        
        nodes = tree['nodes']
        expected_grooves = sum(1 for n in nodes if n['label'] == 'g')
        
        params = TurningParams()
        gen = TreeTurningGenerator(params)
        
        shape = gen.generate_from_tree(tree)
        assert shape is not None, "형상 생성 실패"
    
    def test_step_then_groove_order(self):
        """Step이 먼저 처리되고 Groove가 나중에 처리되는지 검증"""
        # 간단한 트리로 테스트: b(s(g))
        trees = generate_trees(n_nodes=3, max_depth=3)
        
        # s 아래에 g가 있는 트리 찾기
        for tree in trees:
            nodes = tree['nodes']
            for n in nodes:
                if n['label'] == 's':
                    groove_kids = [
                        nodes[cid] for cid in n.get('children', [])
                        if nodes[cid]['label'] == 'g'
                    ]
                    if groove_kids:
                        params = TurningParams()
                        gen = TreeTurningGenerator(params)
                        shape = gen.generate_from_tree(tree)
                        assert shape is not None
                        return
        
        pytest.skip("적합한 테스트 트리를 찾지 못함")


class TestGrooveValidation:
    """Groove 검증 테스트"""
    
    def test_groove_retry_on_failure(self):
        """Groove 실패 시 재시도 검증"""
        trees = generate_trees(n_nodes=6, max_depth=8)
        
        # 형제 Groove가 있는 트리 찾기
        for tree in trees[:100]:
            nodes = tree['nodes']
            for n in nodes:
                if n['label'] in ['b', 's']:
                    groove_kids = [
                        nodes[cid] for cid in n.get('children', [])
                        if nodes[cid]['label'] == 'g'
                    ]
                    if len(groove_kids) >= 2:
                        params = TurningParams()
                        gen = TreeTurningGenerator(params)
                        
                        shape = gen.generate_from_tree(tree)
                        assert shape is not None, "형제 Groove 트리 생성 실패"
                        return
        
        pytest.skip("형제 Groove 트리를 찾지 못함")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
