"""
터닝 플래너 제약조건 파라미터
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TurningParams:
    """터닝 플래너 제약조건 파라미터"""
    stock_height_range: Tuple[float, float] = (15.0, 30.0)
    stock_radius_range: Tuple[float, float] = (6.0, 12.0)

    step_depth_range: Tuple[float, float] = (0.8, 1.5)
    step_margin: float = 0.5        # step zpos 경계면 양쪽 간격

    groove_depth_range: Tuple[float, float] = (0.4, 0.8)
    groove_width_range: Tuple[float, float] = (1.5, 3.0)
    groove_margin: float = 0.5      # groove 상단/하단 양쪽 간격

    min_remaining_radius: float = 2.0

    chamfer_range: Tuple[float, float] = (0.3, 0.8)
    fillet_range: Tuple[float, float] = (0.3, 0.8)
    edge_feature_prob: float = 0.3
