"""
터닝-밀링 형상 생성 핵심 모듈

- tree_generator: 트리 구조 생성
- turning_generator: 트리 기반 터닝 형상 생성
- face_analyzer: 면 치수 분석
- milling_adder: 밀링 특징형상 추가
"""

from core.tree_generator import TreeGenerator, TreeGeneratorParams, generate_trees
from core.turning_generator import TreeTurningGenerator, TurningParams, TreeNode, Region
from core.face_analyzer import FaceAnalyzer, FaceDimensionResult
from core.milling_adder import (
    MillingFeatureAdder, 
    FeatureParams, HoleParams,  # HoleParams는 FeatureParams의 별칭
    ValidFaceInfo, 
    FeaturePlacement, HolePlacement,  # HolePlacement는 FeaturePlacement의 별칭
    FeatureType,
    compute_hole_scale_range,
)

__all__ = [
    'TreeGenerator', 'TreeGeneratorParams', 'generate_trees',
    'TreeTurningGenerator', 'TurningParams', 'TreeNode', 'Region',
    'FaceAnalyzer', 'FaceDimensionResult',
    'MillingFeatureAdder', 'FeatureParams', 'HoleParams',
    'ValidFaceInfo', 'FeaturePlacement', 'HolePlacement',
    'FeatureType', 'compute_hole_scale_range',
]
