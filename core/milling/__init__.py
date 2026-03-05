"""
밀링 특징형상 서브패키지

- features: 파라미터 + 형상 생성 함수 + request 적용 함수
- analyzer: MillingAnalyzer (면 분석 + 배치 계획)
- face_analyzer: FaceAnalyzer (면 치수 분석 유틸리티)
"""

from core.milling.face_analyzer import FaceAnalyzer, FaceDimensionResult
from core.milling.analyzer import MillingAnalyzer
from core.milling.features import (
    MillingParams,
    MillingFeatureRequest,
    compute_hole_scale_range,
    compute_feature_center_cylinder,
    compute_feature_center_cone,
    create_blind_hole,
    create_through_hole,
    create_rectangular_pocket,
    create_rectangular_passage,
    apply_milling_requests,
)

__all__ = [
    'FaceAnalyzer', 'FaceDimensionResult',
    'MillingAnalyzer',
    'MillingParams',
    'MillingFeatureRequest',
    'compute_hole_scale_range',
    'compute_feature_center_cylinder',
    'compute_feature_center_cone',
    'create_blind_hole',
    'create_through_hole',
    'create_rectangular_pocket',
    'create_rectangular_passage',
    'apply_milling_requests',
]
