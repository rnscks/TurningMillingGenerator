# -*- coding: utf-8 -*-
"""
core/tree/node.py 테스트

테스트 실행:
    pytest tests/tree/test_node.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tree.node import Region, TreeNode, load_tree


SIMPLE_TREE = {
    "N": 3, "max_depth_constraint": 2, "canonical": "b(s,g)",
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1, 2], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
        {"id": 2, "label": "g", "parent": 0, "children": [], "depth": 1},
    ]
}

NESTED_TREE = {
    "N": 4, "max_depth_constraint": 3, "canonical": "b(s(s),g)",
    "nodes": [
        {"id": 0, "label": "b", "parent": None, "children": [1, 3], "depth": 0},
        {"id": 1, "label": "s", "parent": 0, "children": [2], "depth": 1},
        {"id": 2, "label": "s", "parent": 1, "children": [], "depth": 2},
        {"id": 3, "label": "g", "parent": 0, "children": [], "depth": 1},
    ]
}


class TestRegion:
    def test_height_property(self):
        r = Region(z_min=0.0, z_max=5.0, radius=10.0)
        assert r.height == pytest.approx(5.0)

    def test_height_zero(self):
        r = Region(z_min=3.0, z_max=3.0, radius=5.0)
        assert r.height == pytest.approx(0.0)

    def test_direction_default_none(self):
        r = Region(z_min=0.0, z_max=1.0, radius=2.0)
        assert r.direction is None

    def test_direction_set(self):
        r = Region(z_min=0.0, z_max=1.0, radius=2.0, direction='top')
        assert r.direction == 'top'

    def test_repr_contains_values(self):
        r = Region(z_min=1.0, z_max=4.0, radius=3.0, direction='bottom')
        s = repr(r)
        assert '1.00' in s
        assert '4.00' in s
        assert '3.00' in s
        assert 'bottom' in s



class TestTreeNode:
    def test_init_fields(self):
        node = TreeNode(node_id=5, label='s', parent_id=2, depth=1)
        assert node.id == 5
        assert node.label == 's'
        assert node.parent_id == 2
        assert node.depth == 1

    def test_init_defaults(self):
        node = TreeNode(node_id=0, label='b', parent_id=None, depth=0)
        assert node.children == []
        assert node.region is None
        assert node.parent_node is None

    def test_repr(self):
        node = TreeNode(node_id=3, label='g', parent_id=1, depth=2)
        s = repr(node)
        assert '3' in s
        assert 'g' in s
        assert '2' in s

    def test_children_list_is_independent(self):
        n1 = TreeNode(node_id=0, label='b', parent_id=None, depth=0)
        n2 = TreeNode(node_id=1, label='s', parent_id=0, depth=1)
        n1.children.append(n2)
        n3 = TreeNode(node_id=2, label='g', parent_id=None, depth=0)
        assert len(n3.children) == 0


class TestLoadTree:
    def test_root_label(self):
        root = load_tree(SIMPLE_TREE)
        assert root.label == 'b'

    def test_root_has_no_parent(self):
        root = load_tree(SIMPLE_TREE)
        assert root.parent_id is None
        assert root.parent_node is None

    def test_children_count(self):
        root = load_tree(SIMPLE_TREE)
        assert len(root.children) == 2

    def test_children_labels(self):
        root = load_tree(SIMPLE_TREE)
        labels = {c.label for c in root.children}
        assert labels == {'s', 'g'}

    def test_parent_refs_set(self):
        root = load_tree(SIMPLE_TREE)
        for child in root.children:
            assert child.parent_node is root

    def test_nested_parent_refs(self):
        root = load_tree(NESTED_TREE)
        step = next(c for c in root.children if c.label == 's')
        nested = next(c for c in step.children if c.label == 's')
        assert nested.parent_node is step

    def test_region_initially_none(self):
        root = load_tree(SIMPLE_TREE)
        assert root.region is None
        for child in root.children:
            assert child.region is None

    def test_required_space_initially_none(self):
        root = load_tree(SIMPLE_TREE)
        assert root.region is None

    def test_depth_values(self):
        root = load_tree(NESTED_TREE)
        assert root.depth == 0
        for child in root.children:
            assert child.depth == 1
            for grandchild in child.children:
                assert grandchild.depth == 2

    def test_node_count(self):
        root = load_tree(NESTED_TREE)

        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.children)

        assert count_nodes(root) == 4
