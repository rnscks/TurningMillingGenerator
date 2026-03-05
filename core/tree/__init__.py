"""
트리 서브패키지

- node: TreeNode, Region, RequiredSpace, load_tree
- generator: TreeGenerator, TreeGeneratorParams, generate_trees
- io: load_trees, save_trees, classify, filter, stats
"""

from core.tree.node import TreeNode, Region, RequiredSpace, load_tree
from core.tree.generator import TreeGenerator, TreeGeneratorParams, generate_trees
from core.tree.io import (
    load_trees,
    save_trees,
    classify_trees_by_step_count,
    classify_trees_by_groove_count,
    get_tree_stats,
    filter_trees,
    find_bidirectional_step_trees,
    find_sibling_groove_trees,
)

__all__ = [
    'TreeNode', 'Region', 'RequiredSpace', 'load_tree',
    'TreeGenerator', 'TreeGeneratorParams', 'generate_trees',
    'load_trees', 'save_trees',
    'classify_trees_by_step_count', 'classify_trees_by_groove_count',
    'get_tree_stats', 'filter_trees',
    'find_bidirectional_step_trees', 'find_sibling_groove_trees',
]
