"""
터닝-밀링 형상 생성 파이프라인

트리 기반 터닝 형상 생성 → 밀링 특징형상 추가 → 최종 형상 출력
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Shape

from core import (
    TreeTurningGenerator, TurningParams,
    MillingFeatureAdder, HoleParams, HolePlacement
)
from core.label_maker import LabelMaker, Labels
from utils import save_step


# ============================================================================
# 파이프라인 파라미터
# ============================================================================

@dataclass
class TurningMillingParams:
    """전체 파이프라인 파라미터"""
    turning: TurningParams = None
    hole: HoleParams = None
    
    # 밀링 설정
    enable_milling: bool = True
    target_face_types: List[str] = None
    max_holes: int = 5
    holes_per_face: int = 2
    hole_probability: float = 0.8
    
    # 라벨링 설정
    enable_labeling: bool = False
    
    def __post_init__(self):
        if self.turning is None:
            self.turning = TurningParams()
        if self.hole is None:
            self.hole = HoleParams()
        if self.target_face_types is None:
            self.target_face_types = ["Plane", "Cylinder"]


# ============================================================================
# 메인 파이프라인
# ============================================================================

class TurningMillingGenerator:
    """터닝-밀링 형상 생성 파이프라인"""
    
    def __init__(self, params: TurningMillingParams = None):
        self.params = params or TurningMillingParams()
        self.turning_gen = TreeTurningGenerator(self.params.turning)
        self.milling_adder = MillingFeatureAdder(self.params.hole)
        
        self.shape: Optional[TopoDS_Shape] = None
        self.placements: List[HolePlacement] = []
        self.label_maker: Optional[LabelMaker] = None
    
    def generate_from_tree(
        self,
        tree: Dict,
        apply_edge_features: bool = True
    ) -> Tuple[TopoDS_Shape, List[HolePlacement]]:
        """
        트리 구조에서 터닝+밀링 형상 생성.
        
        Args:
            tree: 트리 구조 딕셔너리
            apply_edge_features: 챔퍼/필렛 적용 여부
            
        Returns:
            (최종 형상, 홀 배치 정보 리스트)
        """
        # 라벨링 초기화
        if self.params.enable_labeling:
            self.label_maker = LabelMaker()
        else:
            self.label_maker = None
        
        # 1. 터닝 형상 생성
        turning_shape = self.turning_gen.generate_from_tree(
            tree, 
            apply_edge_features=apply_edge_features,
            label_maker=self.label_maker
        )
        
        if not self.params.enable_milling:
            self.shape = turning_shape
            self.placements = []
            return self.shape, self.placements
        
        # 2. 밀링 특징형상 추가
        self.shape, self.placements = self.milling_adder.add_milling_features(
            turning_shape,
            target_face_types=self.params.target_face_types,
            max_total_holes=self.params.max_holes,
            holes_per_face=self.params.holes_per_face,
            label_maker=self.label_maker
        )
        
        return self.shape, self.placements
    
    def save(self, filepath: str) -> bool:
        """STEP 파일로 저장"""
        if self.shape is None:
            return False
        return save_step(self.shape, filepath)
    
    def get_generation_info(self) -> Dict:
        """생성 정보 반환"""
        return {
            "stock_height": self.turning_gen.stock_height,
            "stock_radius": self.turning_gen.stock_radius,
            "n_holes": len(self.placements),
            "holes": [
                {
                    "face_id": p.face_id,
                    "face_type": p.face_type,
                    "diameter": p.diameter,
                    "depth": p.depth,
                    "center": [p.center_3d.X(), p.center_3d.Y(), p.center_3d.Z()],
                    "direction": [p.direction.X(), p.direction.Y(), p.direction.Z()]
                }
                for p in self.placements
            ]
        }


# ============================================================================
# 트리 데이터 유틸리티 (utils.tree_io에서 re-export)
# ============================================================================

from utils.tree_io import load_trees, classify_trees_by_step_count
