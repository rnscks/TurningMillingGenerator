# -*- coding: utf-8 -*-
"""
core/tree/generator.py 테스트

테스트 실행:
    pytest tests/tree/test_generator.py -v
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tree.generator import TreeGenerator, TreeGeneratorParams, generate_trees


class TestTreeGeneratorParams:
    def test_default_params(self):
        params = TreeGeneratorParams()
        assert params.n_nodes == 6
        assert params.max_depth == 3
        assert params.labels == ['s', 'g']
        assert params.max_children_per_node == 4

    def test_custom_params(self):
        params = TreeGeneratorParams(
            n_nodes=10, max_depth=5,
            labels=['s', 'g', 'x'],
            max_children_per_node=3
        )
        assert params.n_nodes == 10
        assert params.max_depth == 5
        assert params.labels == ['s', 'g', 'x']
        assert params.max_children_per_node == 3


class TestTreeGeneratorValidation:
    def test_max_depth_zero_raises_error(self):
        params = TreeGeneratorParams(n_nodes=3, max_depth=0)
        generator = TreeGenerator(params)
        with pytest.raises(ValueError) as exc_info:
            generator.generate_all_trees()
        assert "max_depth must be at least 1" in str(exc_info.value)

    def test_max_depth_negative_raises_error(self):
        params = TreeGeneratorParams(n_nodes=3, max_depth=-1)
        generator = TreeGenerator(params)
        with pytest.raises(ValueError) as exc_info:
            generator.generate_all_trees()
        assert "max_depth must be at least 1" in str(exc_info.value)

    def test_n_nodes_zero_returns_empty(self):
        params = TreeGeneratorParams(n_nodes=0, max_depth=3)
        generator = TreeGenerator(params)
        assert generator.generate_all_trees() == []

    def test_n_nodes_negative_returns_empty(self):
        params = TreeGeneratorParams(n_nodes=-1, max_depth=3)
        generator = TreeGenerator(params)
        assert generator.generate_all_trees() == []


class TestTreeGeneratorSingleNode:
    def test_single_node_tree(self):
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
    def test_tree_structure_validity(self):
        params = TreeGeneratorParams(n_nodes=4, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        for tree in trees:
            nodes = tree["nodes"]
            assert tree["N"] == len(nodes)
            assert len(nodes) == params.n_nodes

            for node in nodes:
                assert "id" in node
                assert "label" in node
                assert "parent" in node
                assert "children" in node
                assert "depth" in node
                assert node["depth"] <= params.max_depth
                if node["depth"] == 0:
                    assert node["label"] == "b"
                else:
                    assert node["label"] in params.labels

    def test_parent_child_consistency(self):
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}

            for node in nodes:
                if node["parent"] is not None:
                    parent = node_map[node["parent"]]
                    assert node["id"] in parent["children"]

                for child_id in node["children"]:
                    child = node_map[child_id]
                    assert child["parent"] == node["id"]

    def test_depth_calculation(self):
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
    def test_canonical_uniqueness(self):
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        canonicals = [tree["canonical"] for tree in trees]
        assert len(canonicals) == len(set(canonicals))

    def test_canonical_format(self):
        params = TreeGeneratorParams(n_nodes=3, max_depth=2)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        for tree in trees:
            canonical = tree["canonical"]
            assert canonical.startswith("b")
            if tree["N"] > 1:
                assert "(" in canonical and ")" in canonical


class TestTreeGeneratorPartitions:
    def test_partitions_basic(self):
        generator = TreeGenerator()
        assert generator._partitions(3, 1) == [(3,)]
        result = generator._partitions(4, 2)
        assert set(result) == {(1, 3), (2, 2), (3, 1)}

    def test_partitions_sum(self):
        generator = TreeGenerator()
        for n in range(2, 7):
            for k in range(1, n + 1):
                for p in generator._partitions(n, k):
                    assert sum(p) == n
                    assert len(p) == k
                    assert all(x >= 1 for x in p)


class TestTreeGeneratorBalancedSample:
    def test_balanced_sample_count(self):
        params = TreeGeneratorParams(n_nodes=5, max_depth=3)
        generator = TreeGenerator(params)

        sample = generator.generate_balanced_sample(total_count=10)
        all_trees = generator.generate_all_trees()
        assert len(sample) == min(10, len(all_trees))

    def test_balanced_sample_when_fewer_trees(self):
        params = TreeGeneratorParams(n_nodes=2, max_depth=2)
        generator = TreeGenerator(params)
        all_trees = generator.generate_all_trees()
        sample = generator.generate_balanced_sample(total_count=100)
        assert len(sample) == len(all_trees)


class TestConvenienceFunction:
    def test_generate_trees_default(self):
        trees = generate_trees()
        assert len(trees) > 0
        for tree in trees:
            assert tree["N"] == 6
            assert tree["max_depth_constraint"] == 3

    def test_generate_trees_custom(self):
        trees = generate_trees(n_nodes=4, max_depth=2)
        for tree in trees:
            assert tree["N"] == 4


class TestGeometricConstraints:
    def _get_parent_child_pairs(self, tree: Dict) -> List[Tuple[str, str]]:
        nodes = tree["nodes"]
        node_map = {n["id"]: n for n in nodes}
        pairs = []
        for node in nodes:
            for child_id in node["children"]:
                child = node_map[child_id]
                pairs.append((node["label"], child["label"]))
        return pairs

    def test_groove_cannot_have_step_child(self):
        params = TreeGeneratorParams(n_nodes=6, max_depth=4)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        for tree in trees:
            for parent_label, child_label in self._get_parent_child_pairs(tree):
                if parent_label == 'g':
                    assert child_label != 's'

    def test_base_max_two_step_children(self):
        params = TreeGeneratorParams(n_nodes=6, max_depth=3)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}
            base_node = next(n for n in nodes if n["label"] == "b")
            step_count = sum(
                1 for cid in base_node["children"]
                if node_map[cid]["label"] == 's'
            )
            assert step_count <= 2

    def test_step_max_one_step_child(self):
        params = TreeGeneratorParams(n_nodes=6, max_depth=4)
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()

        for tree in trees:
            nodes = tree["nodes"]
            node_map = {n["id"]: n for n in nodes}
            for node in nodes:
                if node["label"] == 's':
                    step_count = sum(
                        1 for cid in node["children"]
                        if node_map[cid]["label"] == 's'
                    )
                    assert step_count <= 1

    def test_groove_can_have_groove_child(self):
        params = TreeGeneratorParams(n_nodes=4, max_depth=4, labels=['g'])
        generator = TreeGenerator(params)
        trees = generator.generate_all_trees()
        assert len(trees) > 0


class TestAllowedChildLabels:
    def test_base_allowed_children(self):
        generator = TreeGenerator()
        allowed = generator._get_allowed_child_labels('b')
        assert 's' in allowed
        assert 'g' in allowed

    def test_groove_allowed_children(self):
        generator = TreeGenerator()
        allowed = generator._get_allowed_child_labels('g')
        assert 's' not in allowed
        assert 'g' in allowed


class TestValidateChildrenCombination:
    def test_base_with_three_steps_invalid(self):
        generator = TreeGenerator()
        assert generator._validate_children_combination('b', ('s', 's', 's')) is False

    def test_base_with_two_steps_valid(self):
        generator = TreeGenerator()
        assert generator._validate_children_combination('b', ('s', 's')) is True

    def test_step_with_two_steps_invalid(self):
        generator = TreeGenerator()
        assert generator._validate_children_combination('s', ('s', 's')) is False

    def test_groove_with_step_invalid(self):
        generator = TreeGenerator()
        assert generator._validate_children_combination('g', ('s',)) is False

    def test_groove_with_groove_valid(self):
        generator = TreeGenerator()
        assert generator._validate_children_combination('g', ('g',)) is True
