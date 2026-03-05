"""
터닝-밀링 형상 생성 핵심 모듈

서브패키지:
- tree: TreeNode, Region, RequiredSpace, TreeGenerator, TreeGeneratorParams, load/save io
- turning: TurningParams, StockInfo, TurningFeatureRequest, TurningPlanner, 형상 함수들
- milling: MillingParams, MillingFeatureRequest, MillingAnalyzer, FaceAnalyzer, 형상 함수들

독립 모듈:
- design_operation: DesignOperation
- label_maker: LabelMaker, Labels
- step_io: load_step, save_step, save_labeled_step
"""

from core.tree import (
    TreeNode, Region, RequiredSpace, load_tree,
    TreeGenerator, TreeGeneratorParams, generate_trees,
    load_trees, save_trees,
    classify_trees_by_step_count, classify_trees_by_groove_count,
    get_tree_stats, filter_trees,
    find_bidirectional_step_trees, find_sibling_groove_trees,
)
from core.turning import (
    TurningParams, StockInfo, TurningFeatureRequest,
    TurningPlanner,
    create_step_cut, create_groove_cut, create_stock,
    apply_turning_requests, apply_edge_features,
)
from core.milling import (
    MillingParams, MillingFeatureRequest,
    MillingAnalyzer,
    FaceAnalyzer, FaceDimensionResult,
    compute_hole_scale_range,
    create_blind_hole, create_through_hole,
    create_rectangular_pocket, create_rectangular_passage,
    apply_milling_requests,
)
from core.design_operation import DesignOperation
from core.label_maker import LabelMaker, Labels
from core.step_io import load_step, save_step, save_labeled_step

__all__ = [
    # tree
    'TreeNode', 'Region', 'RequiredSpace', 'load_tree',
    'TreeGenerator', 'TreeGeneratorParams', 'generate_trees',
    'load_trees', 'save_trees',
    'classify_trees_by_step_count', 'classify_trees_by_groove_count',
    'get_tree_stats', 'filter_trees',
    'find_bidirectional_step_trees', 'find_sibling_groove_trees',
    # turning
    'TurningParams', 'StockInfo', 'TurningFeatureRequest',
    'TurningPlanner',
    'create_step_cut', 'create_groove_cut', 'create_stock',
    'apply_turning_requests', 'apply_edge_features',
    # milling
    'MillingParams', 'MillingFeatureRequest',
    'MillingAnalyzer',
    'FaceAnalyzer', 'FaceDimensionResult',
    'compute_hole_scale_range',
    'create_blind_hole', 'create_through_hole',
    'create_rectangular_pocket', 'create_rectangular_passage',
    'apply_milling_requests',
    # 공통
    'DesignOperation', 'LabelMaker', 'Labels',
    'load_step', 'save_step', 'save_labeled_step',
]
