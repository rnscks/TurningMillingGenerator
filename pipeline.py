"""
터닝-밀링 복합 형상 생성 파이프라인

생성 흐름:
1) Stock 원기둥 생성
2) Step/Groove 터닝 가공
3) 챔퍼/라운드 엣지 처리
4) 면 분석 (폭/너비 계산)
5) 밀링 특징형상 추가 (홀)
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from OCC.Core.TopoDS import TopoDS_Shape

from core import (
    TreeTurningGenerator, TurningParams,
    MillingFeatureAdder, HoleParams, HolePlacement,
)
from utils import save_step


# ============================================================================
# Pipeline Parameters
# ============================================================================

@dataclass
class TurningMillingParams:
    """터닝-밀링 복합 형상 파라미터"""
    # 터닝 파라미터
    turning: TurningParams = field(default_factory=TurningParams)
    
    # 밀링 파라미터
    hole: HoleParams = field(default_factory=HoleParams)
    
    # 밀링 추가 옵션
    enable_milling: bool = True
    target_face_types: List[str] = field(default_factory=lambda: ["Plane", "Cylinder"])
    max_holes: int = 4
    holes_per_face: int = 1
    hole_probability: float = 0.7  # 밀링 특징 추가 확률


# ============================================================================
# TurningMillingGenerator Class
# ============================================================================

class TurningMillingGenerator:
    """
    터닝-밀링 복합 형상 생성기 (통합 파이프라인).
    
    사용법:
        params = TurningMillingParams(
            turning=TurningParams(stock_radius_range=(6.5, 8.5)),
            hole=HoleParams(max_holes_per_face=2),
        )
        
        generator = TurningMillingGenerator(params)
        shape, placements = generator.generate_from_tree(tree_data)
        generator.save("output.step")
    """
    
    def __init__(self, params: TurningMillingParams = None):
        self.params = params or TurningMillingParams()
        self.turning_gen = TreeTurningGenerator(self.params.turning)
        self.milling_adder = MillingFeatureAdder(self.params.hole)
        
        self.shape: Optional[TopoDS_Shape] = None
        self.placements: List[HolePlacement] = []
        
    def generate_from_tree(
        self,
        tree_data: Dict,
        apply_edge_features: bool = True,
        apply_milling: bool = None
    ) -> Tuple[TopoDS_Shape, List[HolePlacement]]:
        """
        트리 데이터로부터 터닝-밀링 복합 형상 생성.
        
        Args:
            tree_data: 트리 JSON 데이터
            apply_edge_features: 챔퍼/라운드 적용 여부
            apply_milling: 밀링 특징 추가 여부 (None이면 params 따름)
            
        Returns:
            (생성된 형상, 밀링 배치 정보 리스트)
        """
        # 1. 터닝 형상 생성
        print("  [1] 터닝 형상 생성...")
        self.shape = self.turning_gen.generate_from_tree(tree_data, apply_edge_features)
        
        if self.shape is None or self.shape.IsNull():
            raise RuntimeError("터닝 형상 생성 실패")
        
        # 2. 밀링 특징 추가 여부 결정
        if apply_milling is None:
            apply_milling = self.params.enable_milling
        
        if apply_milling and random.random() < self.params.hole_probability:
            print("  [2] 밀링 특징형상 추가...")
            self.shape, self.placements = self.milling_adder.add_milling_features(
                self.shape,
                target_face_types=self.params.target_face_types,
                max_total_holes=self.params.max_holes,
                holes_per_face=self.params.holes_per_face
            )
        else:
            print("  [2] 밀링 특징형상 스킵")
            self.placements = []
        
        return self.shape, self.placements
    
    def save(self, filepath: str) -> bool:
        """생성된 형상을 STEP 파일로 저장."""
        if self.shape is None or self.shape.IsNull():
            print("저장할 형상이 없습니다.")
            return False
        return save_step(self.shape, filepath)
    
    def get_generation_info(self) -> Dict:
        """생성 정보 반환."""
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
# Tree Data Loader
# ============================================================================

def load_trees(json_path: str) -> List[Dict]:
    """트리 JSON 파일 로드."""
    with open(json_path, 'r') as f:
        return json.load(f)


def classify_trees_by_step_count(trees: List[Dict]) -> Dict[int, List[int]]:
    """트리를 step 개수별로 분류."""
    step_count_map: Dict[int, List[int]] = {}
    for i, tree in enumerate(trees):
        step_count = sum(1 for n in tree['nodes'] if n['label'] == 's')
        if step_count not in step_count_map:
            step_count_map[step_count] = []
        step_count_map[step_count].append(i)
    return step_count_map


# ============================================================================
# Batch Generation
# ============================================================================

def generate_batch(
    trees: List[Dict],
    params: TurningMillingParams,
    output_dir: str,
    selected_indices: List[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    배치로 터닝-밀링 형상 생성.
    
    Args:
        trees: 트리 데이터 리스트
        params: 생성 파라미터
        output_dir: 출력 디렉토리
        selected_indices: 선택된 트리 인덱스 (None이면 전체)
        seed: 랜덤 시드
        
    Returns:
        생성 정보 리스트
    """
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if selected_indices is None:
        selected_indices = list(range(len(trees)))
    
    generation_info = []
    
    for idx in selected_indices:
        tree = trees[idx]
        n_nodes = tree['N']
        max_depth = tree['max_depth_constraint']
        canonical = tree['canonical']
        
        s_count = sum(1 for n in tree['nodes'] if n['label'] == 's')
        g_count = sum(1 for n in tree['nodes'] if n['label'] == 'g')
        
        print(f"\n{'=' * 60}")
        print(f"=== ID:{idx}, S:{s_count}, G:{g_count}, {canonical} ===")
        print(f"{'=' * 60}")
        
        generator = TurningMillingGenerator(params)
        
        try:
            shape, placements = generator.generate_from_tree(
                tree, 
                apply_edge_features=True,
                apply_milling=True
            )
            
            if shape and not shape.IsNull():
                n_holes = len(placements)
                filename = f"tm_N{n_nodes}_H{max_depth}_S{s_count}_G{g_count}_M{n_holes}_{idx:03d}.step"
                output_file = output_path / filename
                generator.save(str(output_file))
                
                info = generator.get_generation_info()
                info["tree_id"] = idx
                info["filename"] = filename
                info["s_count"] = s_count
                info["g_count"] = g_count
                generation_info.append(info)
            else:
                print(f"  모델 생성 실패")
                
        except Exception as e:
            print(f"  오류 발생: {e}")
    
    # 생성 정보 JSON 저장
    info_path = output_path / "generation_info.json"
    with open(info_path, 'w') as f:
        json.dump(generation_info, f, indent=2)
    print(f"\n생성 정보 저장: {info_path}")
    
    return generation_info


# ============================================================================
# Main
# ============================================================================

def main():
    """메인: 터닝-밀링 복합 형상 배치 생성"""
    # 트리 데이터 로드
    json_path = Path("trees_N6_H3.json")
    if not json_path.exists():
        print(f"Error: {json_path} 파일을 찾을 수 없습니다.")
        return
    
    trees = load_trees(str(json_path))
    print(f"총 {len(trees)}개의 트리 로드됨")
    
    # 파라미터 설정
    params = TurningMillingParams(
        turning=TurningParams(
            stock_height_range=(15.0, 20.0),
            stock_radius_range=(6.5, 8.5),
            step_depth_range=(0.8, 1.5),
            step_height_range=(2.5, 6.5),
            groove_depth_range=(0.5, 1.0),
            groove_width_range=(2.5, 6.5),
            chamfer_range=(0.3, 0.8),
            fillet_range=(0.3, 0.8),
            edge_feature_prob=0.3,
        ),
        hole=HoleParams(
            max_holes_per_face=2,
        ),
        enable_milling=True,
        target_face_types=["Plane", "Cylinder"],
        max_holes=4,
        holes_per_face=1,
        hole_probability=0.8,
    )
    
    # Step 개수별로 분류
    step_count_map = classify_trees_by_step_count(trees)
    
    print("\nStep 개수별 트리 분포:")
    for count, indices in sorted(step_count_map.items()):
        print(f"  {count}개 step: {len(indices)}개 트리")
    
    # 다양한 구조 선택
    selected_indices = []
    for step_count in range(6):
        if step_count in step_count_map and step_count_map[step_count]:
            candidates = step_count_map[step_count]
            selected = candidates[:min(2, len(candidates))]
            selected_indices.extend(selected)
    
    print(f"\n선택된 트리: {selected_indices}")
    
    # 배치 생성
    generation_info = generate_batch(
        trees=trees,
        params=params,
        output_dir="generated_models",
        selected_indices=selected_indices,
        seed=42
    )
    
    print(f"\n완료! 총 {len(generation_info)}개 모델 생성")
    
    # 요약
    total_holes = sum(info["n_holes"] for info in generation_info)
    models_with_holes = sum(1 for info in generation_info if info["n_holes"] > 0)
    print(f"\n요약:")
    print(f"  - 밀링 특징 있는 모델: {models_with_holes}/{len(generation_info)}")
    print(f"  - 총 홀 수: {total_holes}")


if __name__ == "__main__":
    main()
