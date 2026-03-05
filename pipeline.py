"""
터닝·밀링 복합 형상 합성 파이프라인  (core 모듈)

TurningMillingGenerator 클래스가 트리 구조를 입력받아
아래 순서로 형상을 합성합니다:

  1. 터닝 계획 (TurningPlanner) : 트리 노드 → Step·Groove·Stock 배치
  2. 터닝 형상 생성 (Boolean Cut)
  3. 엣지 특징 (Chamfer·Fillet) 적용
  4. 밀링 분석 (MillingAnalyzer) : 유효 면 선택 → 홀·포켓 배치 결정
  5. 밀링 형상 생성 (Boolean Cut)
  6. 선택적 라벨링 (LabelMaker) : face 단위 특징형상 라벨 부여

이 모듈은 파이프라인 로직만 담으며, 직접 실행하지 않습니다.
데이터셋 생성은 generate_dataset.py, 형상 파일 저장은 run_pipeline.py 를 사용하세요.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Shape

from core.tree.node import load_tree
from core.turning.planner import TurningPlanner, TurningParams
from core.turning.features import StockInfo, apply_edge_features
from core.milling.features import MillingParams, MillingFeatureRequest, apply_milling_requests
from core.milling.analyzer import MillingAnalyzer
from core.label_maker import LabelMaker
from core.step_io import save_step


# ============================================================================
# 파이프라인 파라미터
# ============================================================================

@dataclass
class TurningMillingParams:
    """전체 파이프라인 파라미터"""
    turning: TurningParams = None
    milling: MillingParams = None

    enable_milling: bool = True
    target_face_types: List[str] = None
    max_features: int = 5
    features_per_face: int = 2

    enable_labeling: bool = False

    def __post_init__(self):
        if self.turning is None:
            self.turning = TurningParams()
        if self.milling is None:
            self.milling = MillingParams()
        if self.target_face_types is None:
            self.target_face_types = ["Cylinder"]


# ============================================================================
# 메인 파이프라인
# ============================================================================

class TurningMillingGenerator:
    """터닝-밀링 형상 생성 파이프라인"""

    def __init__(self, params: TurningMillingParams = None):
        self.params = params or TurningMillingParams()
        self.planner = TurningPlanner(self.params.turning)
        self.analyzer = MillingAnalyzer(self.params.milling)

        self.shape: Optional[TopoDS_Shape] = None
        self.milling_requests: List[MillingFeatureRequest] = []
        self.label_maker: Optional[LabelMaker] = None
        self._stock_info: Optional[StockInfo] = None

    def generate_from_tree(
        self,
        tree: Dict,
        apply_edge_feats: bool = True,
    ) -> Tuple[TopoDS_Shape, List[MillingFeatureRequest]]:
        """트리 구조에서 터닝+밀링 형상 생성.

        Args:
            tree: 트리 구조 딕셔너리
            apply_edge_feats: 챔퍼/필렛 적용 여부

        Returns:
            (최종 형상, 밀링 피처 요청 리스트)
        """
        if self.params.enable_labeling:
            self.label_maker = LabelMaker()
        else:
            self.label_maker = None

        # 1. 터닝 계획 → 실행
        root = load_tree(tree)
        stock_info, shape = self.planner.plan_and_apply(root, self.label_maker)
        self._stock_info = stock_info

        # 2. 엣지 피처 후처리
        if apply_edge_feats:
            tp = self.params.turning
            shape = apply_edge_features(
                shape,
                tp.edge_feature_prob,
                tp.chamfer_range,
                tp.fillet_range,
                self.label_maker,
            )

        if not self.params.enable_milling:
            self.shape = shape
            self.milling_requests = []
            return self.shape, self.milling_requests

        # 3. 밀링 분석 → 실행
        self.milling_requests = self.analyzer.analyze(
            shape,
            target_face_types=self.params.target_face_types,
            max_features=self.params.max_features,
            features_per_face=self.params.features_per_face,
        )
        shape = apply_milling_requests(shape, self.milling_requests, self.label_maker)

        self.shape = shape
        return self.shape, self.milling_requests

    def save(self, filepath: str) -> bool:
        """STEP 파일로 저장"""
        if self.shape is None:
            return False
        return save_step(self.shape, filepath)

    def get_generation_info(self) -> Dict:
        """생성 정보 반환"""
        stock_height = self._stock_info.height if self._stock_info else 0.0
        stock_radius = self._stock_info.radius if self._stock_info else 0.0
        return {
            "stock_height": stock_height,
            "stock_radius": stock_radius,
            "n_milling_features": len(self.milling_requests),
            "milling_features": [
                {
                    "face_id": r.face_id,
                    "face_type": r.face_type,
                    "feature_type": r.feature_type,
                    "diameter": r.diameter,
                    "width": r.width,
                    "length": r.length,
                    "depth": r.depth,
                    "center": [r.center.X(), r.center.Y(), r.center.Z()],
                    "direction": [r.direction.X(), r.direction.Y(), r.direction.Z()],
                }
                for r in self.milling_requests
            ]
        }
