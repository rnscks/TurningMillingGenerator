# -*- coding: utf-8 -*-
"""
core/tree/io.py 테스트

테스트 실행:
    pytest tests/tree/test_io.py -v
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tree.io import (
    load_trees, save_trees,
    classify_trees_by_step_count, classify_trees_by_groove_count,
    get_tree_stats, filter_trees,
    find_bidirectional_step_trees, find_sibling_groove_trees,
)


SAMPLE_TREES = [
    {
        "N": 3, "max_depth_constraint": 2, "canonical": "b(s,g)",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1, 2], "depth": 0},
            {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
            {"id": 2, "label": "g", "parent": 0, "children": [], "depth": 1},
        ]
    },
    {
        "N": 4, "max_depth_constraint": 3, "canonical": "b(s(s),g)",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1, 3], "depth": 0},
            {"id": 1, "label": "s", "parent": 0, "children": [2], "depth": 1},
            {"id": 2, "label": "s", "parent": 1, "children": [], "depth": 2},
            {"id": 3, "label": "g", "parent": 0, "children": [], "depth": 1},
        ]
    },
    {
        "N": 5, "max_depth_constraint": 3, "canonical": "b(s,s,g,g)",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1, 2, 3, 4], "depth": 0},
            {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
            {"id": 2, "label": "s", "parent": 0, "children": [], "depth": 1},
            {"id": 3, "label": "g", "parent": 0, "children": [], "depth": 1},
            {"id": 4, "label": "g", "parent": 0, "children": [], "depth": 1},
        ]
    },
    {
        "N": 3, "max_depth_constraint": 2, "canonical": "b(g(g))",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1], "depth": 0},
            {"id": 1, "label": "g", "parent": 0, "children": [2], "depth": 1},
            {"id": 2, "label": "g", "parent": 1, "children": [], "depth": 2},
        ]
    },
]


class TestLoadSaveTrees:
    def test_save_and_load_roundtrip(self, tmp_path):
        filepath = tmp_path / "test_trees.json"
        save_trees(SAMPLE_TREES, str(filepath))

        loaded = load_trees(str(filepath))
        assert len(loaded) == len(SAMPLE_TREES)
        for orig, loaded_t in zip(SAMPLE_TREES, loaded):
            assert orig["N"] == loaded_t["N"]
            assert orig["canonical"] == loaded_t["canonical"]

    def test_save_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "sub" / "dir" / "trees.json"
        save_trees(SAMPLE_TREES, str(filepath))
        assert filepath.exists()

    def test_load_dict_with_trees_key(self, tmp_path):
        filepath = tmp_path / "trees.json"
        data = {"trees": SAMPLE_TREES}
        with open(filepath, 'w') as f:
            json.dump(data, f)
        loaded = load_trees(str(filepath))
        assert len(loaded) == len(SAMPLE_TREES)

    def test_load_bare_list(self, tmp_path):
        filepath = tmp_path / "trees.json"
        with open(filepath, 'w') as f:
            json.dump(SAMPLE_TREES, f)
        loaded = load_trees(str(filepath))
        assert len(loaded) == len(SAMPLE_TREES)

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_trees("nonexistent_path_12345.json")

    def test_load_invalid_format_raises(self, tmp_path):
        filepath = tmp_path / "bad.json"
        with open(filepath, 'w') as f:
            json.dump({"not_trees": "data"}, f)
        with pytest.raises(ValueError):
            load_trees(str(filepath))


class TestClassifyTrees:
    def test_classify_by_step_count(self):
        result = classify_trees_by_step_count(SAMPLE_TREES)
        assert 1 in result
        assert 2 in result
        assert 0 in result[1]
        assert 1 in result[2]

    def test_classify_by_groove_count(self):
        result = classify_trees_by_groove_count(SAMPLE_TREES)
        assert 1 in result
        assert 2 in result

    def test_classify_empty_list(self):
        assert classify_trees_by_step_count([]) == {}

    def test_all_trees_classified(self):
        result = classify_trees_by_step_count(SAMPLE_TREES)
        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(SAMPLE_TREES)))


class TestGetTreeStats:
    def test_basic_stats(self):
        stats = get_tree_stats(SAMPLE_TREES[0])
        assert stats['n_nodes'] == 3
        assert stats['s_count'] == 1
        assert stats['g_count'] == 1
        assert stats['b_count'] == 1
        assert stats['canonical'] == "b(s,g)"

    def test_bidirectional_step_detection(self):
        stats = get_tree_stats(SAMPLE_TREES[2])
        assert stats['base_step_children'] == 2

    def test_sibling_grooves_detection(self):
        stats = get_tree_stats(SAMPLE_TREES[2])
        assert stats['has_sibling_grooves'] is True

    def test_no_sibling_grooves(self):
        stats = get_tree_stats(SAMPLE_TREES[0])
        assert stats['has_sibling_grooves'] is False

    def test_nested_grooves_not_siblings(self):
        stats = get_tree_stats(SAMPLE_TREES[3])
        assert stats['has_sibling_grooves'] is False


class TestFindFunctions:
    def test_find_bidirectional_step(self):
        result = find_bidirectional_step_trees(SAMPLE_TREES)
        assert 2 in result
        assert 0 not in result

    def test_find_sibling_grooves(self):
        result = find_sibling_groove_trees(SAMPLE_TREES)
        assert 2 in result
        assert 3 not in result

    def test_find_empty_list(self):
        assert find_bidirectional_step_trees([]) == []
        assert find_sibling_groove_trees([]) == []


class TestFilterTrees:
    def test_filter_min_steps(self):
        result = filter_trees(SAMPLE_TREES, min_steps=2)
        for idx in result:
            assert get_tree_stats(SAMPLE_TREES[idx])['s_count'] >= 2

    def test_filter_max_steps(self):
        result = filter_trees(SAMPLE_TREES, max_steps=1)
        for idx in result:
            assert get_tree_stats(SAMPLE_TREES[idx])['s_count'] <= 1

    def test_filter_no_constraints_returns_all(self):
        result = filter_trees(SAMPLE_TREES)
        assert len(result) == len(SAMPLE_TREES)

    def test_filter_impossible_returns_empty(self):
        result = filter_trees(SAMPLE_TREES, min_steps=100)
        assert result == []


class TestTreeNodeIO:
    """core/tree/node.py의 load_tree 함수 테스트"""

    def test_load_tree_basic(self):
        from core.tree.node import load_tree
        tree_data = SAMPLE_TREES[0]
        root = load_tree(tree_data)
        assert root is not None
        assert root.label == 'b'
        assert len(root.children) == 2

    def test_load_tree_parent_refs(self):
        from core.tree.node import load_tree
        tree_data = SAMPLE_TREES[1]
        root = load_tree(tree_data)
        # Step 자식이 있어야 함
        step_child = next(c for c in root.children if c.label == 's')
        assert step_child.parent_node is root
        # 중첩 Step
        nested_step = next(c for c in step_child.children if c.label == 's')
        assert nested_step.parent_node is step_child

    def test_load_tree_region_initially_none(self):
        from core.tree.node import load_tree
        root = load_tree(SAMPLE_TREES[0])
        assert root.region is None
        assert root.required_space is None
