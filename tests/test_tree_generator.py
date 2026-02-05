"""
tree_generator.py 테스트 모듈

테스트 실행:
    pytest tests/test_tree_generator.py -v
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tree_generator import TreeGenerator, TreeGeneratorParams, generate_trees


class TestTreeGeneratorParams:
    """TreeGeneratorParams 테스트"""
    
    def test_default_params(self):
        """기본 파라미터 검증"""
        params = TreeGeneratorParams()
        assert params.n_nodes == 6
        assert params.max_depth == 3
        assert params.labels == ['s', 'g']
        assert params.max_children_per_node == 4
    
    def test_custom_params(self):
        """커스텀 파라미터 검증"""
        params = TreeGeneratorParams(
            n_nodes=10,
            max_depth=5,
            labels=['s', 'g', 'x'],
            max_children_per_node=3
        )
        assert params.n_nodes == 10
        assert params.max_depth == 5
        assert params.labels == ['s', 'g', 'x']
        assert params.max_children_per_node == 3


class TestTreeGeneratorValidation:
    """입력 검증 테스트"""
    
    def test_max_depth_zero_raises_error(self):
        """max_depth=0일 때 ValueError 발생 검증"""
        params = TreeGeneratorParams(n_nodes=3, max_depth=0)
        generator = TreeGenerator(params)
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_all_trees()
        
        assert "max_depth must be at least 1" in str(exc_info.value)
    
    def test_max_depth_negative_raises_error(self):
        """max_depth가 음수일 때 ValueError 발생 검증"""
        params = TreeGeneratorParams(n_nodes=3, max_depth=-1)
        generator = TreeGenerator(params)
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_all_trees()
        
        assert "max_depth must be at least 1" in str(exc_info.value)
    
    def test_n_nodes_zero_returns_empty(self):
        """n_nodes=0일 때 빈 리스트 반환 검증"""
        params = TreeGeneratorParams(n_nodes=0, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        assert trees == []
    
    def test_n_nodes_negative_returns_empty(self):
        """n_nodes가 음수일 때 빈 리스트 반환 검증"""
        params = TreeGeneratorParams(n_nodes=-1, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        assert trees == []


class TestTreeGeneratorSingleNode:
    """단일 노드 트리 테스트"""
    
    def test_single_node_tree(self):
        """n_nodes=1일 때 루트만 있는 트리 생성 검증"""
        params = TreeGeneratorParams(n_nodes=1, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        assert len(trees) == 1
        tree = trees[0]
        
        assert tree["N"] == 1
        assert tree["canonical"] == "b"
        assert len(tree["nodes"]) == 1
        
        root = tree["nodes"][0]
        assert root["id"] == 0
        assert root["label"] == "b"
        assert root["parent"] is None
        assert root["children"] == []
        assert root["depth"] == 0


class TestTreeGeneratorStructure:
    """트리 구조 검증 테스트"""
    
    def test_tree_structure_validity(self):
        """생성된 트리 구조의 유효성 검증"""
        params = TreeGeneratorParams(n_nodes=4, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            nodes = tree["nodes"]
            
            # 노드 수 검증
            assert tree["N"] == len(nodes)
            assert len(nodes) == params.n_nodes
            
            # 각 노드 검증
            for node in nodes:
                # 필수 필드 존재 검증
                assert "id" in node
                assert "label" in node
                assert "parent" in node
                assert "children" in node
                assert "depth" in node
                
                # 깊이 제약 검증
                assert node["depth"] <= params.max_depth
                
                # 라벨 검증
                if node["depth"] == 0:
                    assert node["label"] == "b"
                else:
                    assert node["label"] in params.labels
    
    def test_parent_child_consistency(self):
        """부모-자식 관계 일관성 검증"""
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}
            
            for node in nodes:
                # 부모 관계 검증
                if node["parent"] is not None:
                    parent = node_map[node["parent"]]
                    assert node["id"] in parent["children"]
                
                # 자식 관계 검증
                for child_id in node["children"]:
                    child = node_map[child_id]
                    assert child["parent"] == node["id"]
    
    def test_depth_calculation(self):
        """깊이 계산 정확성 검증"""
        params = TreeGeneratorParams(n_nodes=5, max_depth=4)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}
            
            for node in nodes:
                if node["parent"] is None:
                    assert node["depth"] == 0
                else:
                    parent = node_map[node["parent"]]
                    assert node["depth"] == parent["depth"] + 1


class TestTreeGeneratorCanonical:
    """Canonical 문자열 테스트"""
    
    def test_canonical_uniqueness(self):
        """canonical 문자열 고유성 검증"""
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        canonicals = [tree["canonical"] for tree in trees]
        assert len(canonicals) == len(set(canonicals))
    
    def test_canonical_format(self):
        """canonical 문자열 형식 검증"""
        params = TreeGeneratorParams(n_nodes=3, max_depth=2)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            canonical = tree["canonical"]
            # 루트는 항상 'b'로 시작
            assert canonical.startswith("b")
            # 자식이 있으면 괄호로 감싸짐
            if tree["N"] > 1:
                assert "(" in canonical and ")" in canonical


class TestTreeGeneratorPartitions:
    """_partitions 메서드 테스트"""
    
    def test_partitions_basic(self):
        """기본 분할 검증"""
        generator = TreeGenerator()
        
        # partitions(3, 1) = [(3,)]
        result = generator._partitions(3, 1)
        assert result == [(3,)]
        
        # partitions(4, 2) = [(1,3), (2,2), (3,1)]
        result = generator._partitions(4, 2)
        assert set(result) == {(1, 3), (2, 2), (3, 1)}
    
    def test_partitions_sum(self):
        """분할 합계 검증"""
        generator = TreeGenerator()
        
        for n in range(2, 7):
            for k in range(1, n + 1):
                partitions = generator._partitions(n, k)
                for p in partitions:
                    assert sum(p) == n
                    assert len(p) == k
                    assert all(x >= 1 for x in p)


class TestTreeGeneratorBalancedSample:
    """균형 샘플링 테스트"""
    
    def test_balanced_sample_count(self):
        """샘플 개수 검증"""
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)
        
        sample = generator.generate_balanced_sample(total_count=10)
        all_trees = generator.generate_all_trees()
        
        expected_count = min(10, len(all_trees))
        assert len(sample) == expected_count
    
    def test_balanced_sample_when_fewer_trees(self):
        """트리 수가 요청보다 적을 때 검증"""
        params = TreeGeneratorParams(n_nodes=2, max_depth=2)
        generator = TreeGenerator(params)
        
        all_trees = generator.generate_all_trees()
        sample = generator.generate_balanced_sample(total_count=100)
        
        # 전체 트리 수보다 많이 요청하면 전체 반환
        assert len(sample) == len(all_trees)
    
    def test_balanced_sample_diversity(self):
        """샘플 다양성 검증"""
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)
        
        sample = generator.generate_balanced_sample(
            total_count=20,
            ensure_diversity=True
        )
        
        # 다양한 step/groove 비율이 포함되었는지 확인
        ratios = set()
        for tree in sample:
            nodes = tree["nodes"]
            s_count = sum(1 for n in nodes if n["label"] == "s")
            g_count = sum(1 for n in nodes if n["label"] == "g")
            ratios.add((s_count, g_count))
        
        # 최소 2개 이상의 다른 비율이 포함되어야 함
        assert len(ratios) >= 2


class TestConvenienceFunction:
    """편의 함수 테스트"""
    
    def test_generate_trees_default(self):
        """기본 파라미터로 generate_trees 호출 검증"""
        trees = generate_trees()
        
        assert len(trees) > 0
        for tree in trees:
            assert tree["N"] == 6
            assert tree["max_depth_constraint"] == 3
    
    def test_generate_trees_custom(self):
        """커스텀 파라미터로 generate_trees 호출 검증"""
        trees = generate_trees(n_nodes=4, max_depth=2, labels=['s', 'g'])
        
        for tree in trees:
            assert tree["N"] == 4
            assert tree["max_depth_constraint"] == 2


class TestTreeGeneratorEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_max_depth_equals_one(self):
        """max_depth=1일 때 검증 (루트 + 리프만 가능)"""
        params = TreeGeneratorParams(n_nodes=3, max_depth=1)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            for node in tree["nodes"]:
                assert node["depth"] <= 1
    
    def test_large_tree(self):
        """큰 트리 생성 검증"""
        params = TreeGeneratorParams(n_nodes=8, max_depth=4)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        assert len(trees) > 0
        for tree in trees:
            assert tree["N"] == 8
    
    def test_single_label(self):
        """단일 라벨만 사용할 때 검증"""
        params = TreeGeneratorParams(n_nodes=4, max_depth=3, labels=['s'])
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            for node in tree["nodes"]:
                if node["depth"] > 0:
                    assert node["label"] == 's'


class TestGeometricConstraints:
    """기하학적 제약 조건 테스트"""
    
    def _get_parent_child_pairs(self, tree: Dict) -> List[Tuple[str, str]]:
        """트리에서 모든 부모-자식 라벨 쌍 추출"""
        nodes = tree["nodes"]
        node_map = {n["id"]: n for n in nodes}
        pairs = []
        
        for node in nodes:
            for child_id in node["children"]:
                child = node_map[child_id]
                pairs.append((node["label"], child["label"]))
        
        return pairs
    
    def test_groove_cannot_have_step_child(self):
        """Groove(g)의 자식으로 Step(s)이 올 수 없음 검증"""
        params = TreeGeneratorParams(n_nodes=6, max_depth=4)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            pairs = self._get_parent_child_pairs(tree)
            for parent_label, child_label in pairs:
                if parent_label == 'g':
                    assert child_label != 's', \
                        f"Groove → Step 금지 위반: {tree['canonical']}"
    
    def test_base_max_two_step_children(self):
        """Base(b)의 자식으로 Step(s)은 최대 2개까지만 검증"""
        params = TreeGeneratorParams(n_nodes=6, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}
            
            # Base 노드 찾기
            base_node = next(n for n in nodes if n["label"] == "b")
            
            # Base의 Step 자식 수 카운트
            step_children_count = sum(
                1 for child_id in base_node["children"]
                if node_map[child_id]["label"] == 's'
            )
            
            assert step_children_count <= 2, \
                f"Base → Step 최대 2개 위반: {tree['canonical']}, count={step_children_count}"
    
    def test_step_max_one_step_child(self):
        """Step(s)의 자식으로 Step(s)은 최대 1개까지만 검증"""
        params = TreeGeneratorParams(n_nodes=6, max_depth=4)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}
            
            # 모든 Step 노드에 대해 검증
            for node in nodes:
                if node["label"] == 's':
                    step_children_count = sum(
                        1 for child_id in node["children"]
                        if node_map[child_id]["label"] == 's'
                    )
                    
                    assert step_children_count <= 1, \
                        f"Step → Step 최대 1개 위반: {tree['canonical']}, count={step_children_count}"
    
    def test_groove_can_have_groove_child(self):
        """Groove(g)의 자식으로 Groove(g)는 허용됨 검증"""
        # Groove만 있는 트리가 생성 가능한지 확인
        params = TreeGeneratorParams(n_nodes=4, max_depth=4, labels=['g'])
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        # 중첩 Groove 구조가 있어야 함
        assert len(trees) > 0
        
        for tree in trees:
            for node in tree["nodes"]:
                if node["depth"] > 0:
                    assert node["label"] == 'g'
    
    def test_step_can_have_groove_child(self):
        """Step(s)의 자식으로 Groove(g)는 허용됨 검증"""
        params = TreeGeneratorParams(n_nodes=3, max_depth=2)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        
        # Step → Groove 패턴이 존재하는지 확인
        has_step_groove = False
        for tree in trees:
            pairs = self._get_parent_child_pairs(tree)
            if ('s', 'g') in pairs:
                has_step_groove = True
                break
        
        assert has_step_groove, "Step → Groove 패턴이 생성되어야 함"


class TestAllowedChildLabels:
    """허용 자식 라벨 메서드 테스트"""
    
    def test_base_allowed_children(self):
        """Base의 허용 자식 라벨 검증"""
        generator = TreeGenerator()
        allowed = generator._get_allowed_child_labels('b')
        assert 's' in allowed
        assert 'g' in allowed
    
    def test_step_allowed_children(self):
        """Step의 허용 자식 라벨 검증"""
        generator = TreeGenerator()
        allowed = generator._get_allowed_child_labels('s')
        assert 's' in allowed
        assert 'g' in allowed
    
    def test_groove_allowed_children(self):
        """Groove의 허용 자식 라벨 검증 (Step 불가)"""
        generator = TreeGenerator()
        allowed = generator._get_allowed_child_labels('g')
        assert 's' not in allowed
        assert 'g' in allowed


class TestValidateChildrenCombination:
    """자식 조합 검증 메서드 테스트"""
    
    def test_base_with_three_steps_invalid(self):
        """Base에서 Step 3개는 무효"""
        generator = TreeGenerator()
        result = generator._validate_children_combination('b', ('s', 's', 's'))
        assert result is False
    
    def test_base_with_two_steps_valid(self):
        """Base에서 Step 2개는 유효"""
        generator = TreeGenerator()
        result = generator._validate_children_combination('b', ('s', 's'))
        assert result is True
    
    def test_step_with_two_steps_invalid(self):
        """Step에서 Step 2개는 무효"""
        generator = TreeGenerator()
        result = generator._validate_children_combination('s', ('s', 's'))
        assert result is False
    
    def test_step_with_one_step_valid(self):
        """Step에서 Step 1개는 유효"""
        generator = TreeGenerator()
        result = generator._validate_children_combination('s', ('s',))
        assert result is True
    
    def test_groove_with_step_invalid(self):
        """Groove에서 Step은 무효"""
        generator = TreeGenerator()
        result = generator._validate_children_combination('g', ('s',))
        assert result is False
    
    def test_groove_with_groove_valid(self):
        """Groove에서 Groove는 유효"""
        generator = TreeGenerator()
        result = generator._validate_children_combination('g', ('g',))
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
