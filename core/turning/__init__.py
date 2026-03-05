"""
터닝 특징형상 서브패키지

- planner: TurningParams + TurningPlanner (트리 기반 배치 계획 + 형상 적용)
- features: 형상 생성 함수 + request 적용 함수 (내부 구현)
"""

from core.turning.planner import TurningPlanner, TurningParams
from core.turning.features import (
    StockInfo,
    TurningFeatureRequest,
    create_step_cut,
    create_groove_cut,
    create_stock,
    apply_chamfer,
    apply_fillet,
    apply_edge_features,
    apply_turning_requests,
)

__all__ = [
    'TurningPlanner',
    'TurningParams',
    'StockInfo',
    'TurningFeatureRequest',
    'create_step_cut',
    'create_groove_cut',
    'create_stock',
    'apply_chamfer',
    'apply_fillet',
    'apply_edge_features',
    'apply_turning_requests',
]
