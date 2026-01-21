"""
터닝-밀링 형상 생성 핵심 모듈

- turning_generator: 트리 기반 터닝 형상 생성
- face_analyzer: 면 치수 분석
- milling_adder: 밀링 특징형상 추가
"""

from core.turning_generator import TreeTurningGenerator, TurningParams, TreeNode, Region
from core.face_analyzer import FaceAnalyzer, FaceDimensionResult
from core.milling_adder import MillingFeatureAdder, HoleParams, ValidFaceInfo, HolePlacement

__all__ = [
    'TreeTurningGenerator', 'TurningParams', 'TreeNode', 'Region',
    'FaceAnalyzer', 'FaceDimensionResult',
    'MillingFeatureAdder', 'HoleParams', 'ValidFaceInfo', 'HolePlacement',
]
