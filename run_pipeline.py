"""
터닝-밀링 파이프라인 실행 스크립트

전체 워크플로우:
1. 트리 구조 생성 (또는 로드)
2. 트리 시각화
3. 터닝+밀링 형상 생성
4. 결과 저장 (STEP, 시각화 이미지, 생성 정보 JSON)

모든 결과를 results/ 폴더에 저장:
- results/trees/          : 트리 구조 JSON 및 시각화
- results/step/           : 생성된 STEP 파일
- results/visualization/  : 밀링 시각화 이미지
- results/generation_info.json : 생성 정보
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from pipeline import TurningMillingGenerator, TurningMillingParams
from core import TurningParams, FeatureParams, generate_trees
from utils import (
    save_step, load_trees, save_trees,
    classify_trees_by_step_count, get_tree_stats,
    find_bidirectional_step_trees, find_sibling_groove_trees,
)
from viz import visualize_milling_process, visualize_trees


# ============================================================================
# 트리 생성 또는 로드
# ============================================================================

def load_or_generate_trees(
    json_path: str = None,
    n_nodes: int = 6,
    max_depth: int = 3,
    output_dir: str = "results"
) -> List[Dict]:
    """
    트리 데이터 로드 또는 생성.
    
    Args:
        json_path: 기존 트리 JSON 파일 경로 (None이면 새로 생성)
        n_nodes: 생성 시 노드 수
        max_depth: 생성 시 최대 깊이
        output_dir: 결과 저장 디렉토리
        
    Returns:
        트리 딕셔너리 리스트
    """
    results_path = Path(output_dir)
    trees_dir = results_path / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    
    if json_path and Path(json_path).exists():
        # 기존 파일에서 로드
        print(f"\n트리 로드: {json_path}")
        trees = load_trees(json_path)
        print(f"  {len(trees)}개 트리 로드됨")
    else:
        # 새로 생성
        print(f"\n트리 생성: N={n_nodes}, H={max_depth}")
        trees = generate_trees(n_nodes, max_depth)
        print(f"  {len(trees)}개 트리 생성됨")
        
        # 저장
        save_path = trees_dir / f"trees_N{n_nodes}_H{max_depth}.json"
        save_trees(trees, str(save_path))
        print(f"  저장: {save_path}")
    
    # 트리 시각화 저장
    print("\n트리 시각화 생성...")
    visualize_trees(trees, output_dir=str(trees_dir), prefix="trees", max_display=20)
    
    return trees


# ============================================================================
# 형상 생성 파이프라인
# ============================================================================

def run_generation_pipeline(
    trees: List[Dict],
    selected_indices: List[int],
    params: TurningMillingParams,
    results_dir: str = "results",
    seed: int = 42
) -> List[Dict]:
    """
    형상 생성 파이프라인 실행.
    
    Args:
        trees: 트리 딕셔너리 리스트
        selected_indices: 생성할 트리 인덱스 리스트
        params: 생성 파라미터
        results_dir: 결과 저장 디렉토리
        seed: 랜덤 시드
        
    Returns:
        생성 정보 리스트
    """
    random.seed(seed)
    
    results_path = Path(results_dir)
    step_dir = results_path / "step"
    viz_dir = results_path / "visualization"
    
    step_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    generation_info = []
    
    for idx in selected_indices:
        if idx >= len(trees):
            print(f"  [Skip] 인덱스 {idx}가 범위를 벗어남")
            continue
            
        tree = trees[idx]
        stats = get_tree_stats(tree)
        
        model_name = f"model_N{stats['n_nodes']}_S{stats['s_count']}_G{stats['g_count']}_{idx:03d}"
        
        print(f"\n{'=' * 60}")
        print(f"모델: {model_name}")
        print(f"트리: {stats['canonical']}")
        print(f"{'=' * 60}")
        
        generator = TurningMillingGenerator(params)
        
        try:
            final_shape, placements = generator.generate_from_tree(
                tree, apply_edge_features=True
            )
            
            if final_shape and not final_shape.IsNull():
                n_holes = len(placements)
                step_filename = f"{model_name}_H{n_holes}.step"
                step_filepath = step_dir / step_filename
                save_step(final_shape, str(step_filepath))
                
                visualize_milling_process(
                    generator.turning_gen.shape,
                    final_shape,
                    placements,
                    params.feature,
                    viz_dir,
                    model_name
                )
                
                info = generator.get_generation_info()
                info["tree_id"] = idx
                info["model_name"] = model_name
                info["step_file"] = step_filename
                info["s_count"] = stats['s_count']
                info["g_count"] = stats['g_count']
                info["canonical"] = stats['canonical']
                generation_info.append(info)
                
                print(f"\n  완료: {n_holes}개 피처 추가됨")
            else:
                print(f"  모델 생성 실패")
                
        except Exception as e:
            print(f"  오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 생성 정보 JSON 저장
    info_path = results_path / "generation_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(generation_info, f, indent=2, ensure_ascii=False)
    print(f"\n생성 정보 저장: {info_path}")
    
    return generation_info


# ============================================================================
# 전체 파이프라인
# ============================================================================

def run_full_pipeline(
    json_path: str = None,
    n_nodes: int = 8,
    max_depth: int = 4,
    selected_indices: List[int] = None,
    params: TurningMillingParams = None,
    results_dir: str = "results",
    seed: int = 42
) -> List[Dict]:
    """
    전체 파이프라인 실행: 트리 생성 → 형상 생성 → 저장.
    
    Args:
        json_path: 기존 트리 JSON 파일 경로 (None이면 새로 생성)
        n_nodes: 트리 생성 시 노드 수
        max_depth: 트리 생성 시 최대 깊이
        selected_indices: 생성할 트리 인덱스 (None이면 자동 선택)
        params: 생성 파라미터
        results_dir: 결과 저장 디렉토리
        seed: 랜덤 시드
        
    Returns:
        생성 정보 리스트
    """
    # 기본 파라미터
    if params is None:
        params = TurningMillingParams()
    
    # 1. 트리 로드 또는 생성
    trees = load_or_generate_trees(json_path, n_nodes, max_depth, results_dir)
    
    # 2. 트리 선택 (양방향 Step, 형제 Groove 우선 포함)
    if selected_indices is None:
        selected_indices = []
        selected_set = set()
        
        # 2-1. 양방향 Step 트리 최소 2개 포함
        bidirectional_indices = find_bidirectional_step_trees(trees)
        print(f"\n양방향 Step 트리: {len(bidirectional_indices)}개 발견")
        for idx in bidirectional_indices[:2]:
            if idx not in selected_set:
                selected_indices.append(idx)
                selected_set.add(idx)
                stats = get_tree_stats(trees[idx])
                print(f"  [양방향] #{idx}: {stats['canonical']}")
        
        # 2-2. 형제 Groove 트리 최소 2개 포함
        sibling_groove_indices = find_sibling_groove_trees(trees)
        print(f"형제 Groove 트리: {len(sibling_groove_indices)}개 발견")
        for idx in sibling_groove_indices[:2]:
            if idx not in selected_set:
                selected_indices.append(idx)
                selected_set.add(idx)
                stats = get_tree_stats(trees[idx])
                print(f"  [형제G] #{idx}: {stats['canonical']}")
        
        # 2-3. 나머지는 step 개수별로 다양하게 선택
        step_count_map = classify_trees_by_step_count(trees)
        for s_count in sorted(step_count_map.keys()):
            indices = step_count_map[s_count]
            for idx in indices:
                if idx not in selected_set and len(selected_indices) < 10:
                    selected_indices.append(idx)
                    selected_set.add(idx)
                    break
        
        # 최대 10개까지만
        selected_indices = selected_indices[:10]
    
    print(f"\n선택된 트리 ({len(selected_indices)}개): {selected_indices}")
    
    # 3. 트리 통계 출력
    step_count_map = classify_trees_by_step_count(trees)
    print("\nStep 개수별 트리 분포:")
    for count, indices in sorted(step_count_map.items()):
        print(f"  {count}개 step: {len(indices)}개 트리")
    
    # 4. 형상 생성
    generation_info = run_generation_pipeline(
        trees=trees,
        selected_indices=selected_indices,
        params=params,
        results_dir=results_dir,
        seed=seed
    )
    
    # 5. 결과 요약
    print(f"\n{'=' * 60}")
    print("파이프라인 완료!")
    print(f"{'=' * 60}")
    print(f"  총 트리: {len(trees)}개")
    print(f"  생성된 모델: {len(generation_info)}개")
    if generation_info:
        total_holes = sum(info["n_holes"] for info in generation_info)
        print(f"  총 홀 수: {total_holes}")
    print(f"\n  결과 저장 위치:")
    print(f"    - 트리 데이터/시각화: {results_dir}/trees/")
    print(f"    - STEP 파일: {results_dir}/step/")
    print(f"    - 밀링 시각화: {results_dir}/visualization/")
    print(f"    - 생성 정보: {results_dir}/generation_info.json")
    
    return generation_info


# ============================================================================
# Main
# ============================================================================

def main():
    """메인 실행"""
    # 파라미터 설정 (Bottom-Up 방식)
    params = TurningMillingParams(
        turning=TurningParams(
            # Stock margin (필요 크기에 추가되는 여유)
            stock_height_margin=(3.0, 8.0),
            stock_radius_margin=(2.0, 5.0),
            # Step 파라미터
            step_depth_range=(0.8, 1.5),
            step_height_range=(2.0, 4.0),
            step_margin=0.5,
            # Groove 파라미터
            groove_depth_range=(0.4, 0.8),
            groove_width_range=(1.5, 3.0),
            groove_margin_ratio=0.15,
            # 챔퍼/라운드
            chamfer_range=(0.3, 0.8),
            fillet_range=(0.3, 0.8),
            edge_feature_prob=0.3,
        ),
        feature=FeatureParams(
            diameter_min=1.0,
            diameter_max_ratio=0.85,
            clearance=0.15,
            depth_ratio=2.0,
            min_spacing=1.0,
            max_features_per_face=3,
            rect_aspect_min=0.4,
            rect_aspect_max=2.5,
        ),
        enable_milling=True,
        target_face_types=["Cylinder", "Cone"],
        max_holes=8,
        holes_per_face=2,
    )
    
    # 옵션 1: 기존 트리 파일 사용
    existing_trees_path = Path("trees_N6_H3.json")
    
    if existing_trees_path.exists():
        # 기존 파일에서 로드하여 실행 - 자동 선택 (양방향 Step, 형제 Groove 우선)
        run_full_pipeline(
            json_path=str(existing_trees_path),
            selected_indices=None,  # 자동 선택 (양방향 Step, 형제 Groove 우선 포함)
            params=params,
            results_dir="results",
            seed=12345
        )
    else:
        # 새로 생성하여 실행 - 자동 선택
        run_full_pipeline(
            n_nodes=6,
            max_depth=4,
            selected_indices=None,  # 자동 선택 (양방향 Step, 형제 Groove 우선 포함)
            params=params,
            results_dir="results",
            seed=12345
        )


if __name__ == "__main__":
    main()
