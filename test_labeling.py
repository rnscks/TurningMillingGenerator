"""
Face 라벨링 테스트 스크립트

여러 트리 모델을 생성하고, OCC Viewer 메뉴에서 모델을 전환하며 확인합니다.

사용법:
    python test_labeling.py
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display

from pipeline import TurningMillingGenerator, TurningMillingParams
from core import TurningParams, HoleParams
from core.label_maker import LabelMaker
from utils import save_labeled_step
from utils.tree_io import load_trees, get_tree_stats, generate_trees
from viz.label_viz import load_label_props, get_face_center


# ============================================================================
# 전역 상태 (OCC 콜백에서 접근)
# ============================================================================

PROPS_PATH = "config/LABEL_PROPS.json"
display = None
models: List[Dict] = []  # [{name, labeled_faces, stats}, ...]


# ============================================================================
# 모델 생성
# ============================================================================

def build_params() -> TurningMillingParams:
    return TurningMillingParams(
        turning=TurningParams(
            stock_height_margin=(3.0, 8.0),
            stock_radius_margin=(2.0, 5.0),
            step_depth_range=(0.8, 1.5),
            step_height_range=(2.0, 4.0),
            step_margin=0.5,
            groove_depth_range=(0.4, 0.8),
            groove_width_range=(1.5, 3.0),
            groove_margin_ratio=0.15,
            chamfer_range=(0.3, 0.8),
            fillet_range=(0.3, 0.8),
            edge_feature_prob=0.3,
        ),
        hole=HoleParams(
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
        max_holes=4,
        holes_per_face=1,
        hole_probability=1.0,
        enable_labeling=True,
    )


def select_diverse_trees(trees: List[Dict], count: int = 5) -> List[Tuple[int, Dict]]:
    """Step/Groove 개수 다양하게 트리 선택."""
    selected = []
    seen_patterns = set()
    
    for i, tree in enumerate(trees):
        stats = get_tree_stats(tree)
        pattern = (stats['s_count'], stats['g_count'])
        if pattern not in seen_patterns:
            selected.append((i, tree))
            seen_patterns.add(pattern)
        if len(selected) >= count:
            break
    
    # 부족하면 앞에서부터 채움
    for i, tree in enumerate(trees):
        if len(selected) >= count:
            break
        if not any(s[0] == i for s in selected):
            selected.append((i, tree))
    
    return selected[:count]


def generate_models(trees: List[Dict], selected: List[Tuple[int, Dict]]) -> List[Dict]:
    """선택된 트리들로 라벨링된 모델 생성."""
    params = build_params()
    result = []
    
    for idx, tree in selected:
        stats = get_tree_stats(tree)
        name = f"#{idx} S{stats['s_count']}G{stats['g_count']} {stats['canonical']}"
        
        random.seed(42 + idx)
        
        print(f"\n--- {name} ---")
        generator = TurningMillingGenerator(params)
        
        try:
            shape, placements = generator.generate_from_tree(tree, apply_edge_features=True)
            
            if generator.label_maker and generator.label_maker.get_total_faces() > 0:
                lm = generator.label_maker
                print(f"  Faces: {lm.get_total_faces()}, Labels: {lm.get_label_counts()}")
                
                result.append({
                    "name": name,
                    "labeled_faces": dict(lm.labeled_faces),
                    "counts": lm.get_label_counts(),
                    "total": lm.get_total_faces(),
                })
            else:
                print(f"  라벨링 실패, 스킵")
        except Exception as e:
            print(f"  오류: {e}")
    
    return result


# ============================================================================
# OCC Viewer 렌더링
# ============================================================================

def render_model(model: Dict, with_label_name: bool = False):
    """모델을 OCC Viewer에 렌더링."""
    global display
    display.EraseAll()
    
    label_names, color_table = load_label_props(PROPS_PATH)
    labeled_faces = model["labeled_faces"]
    
    for idx, face in enumerate(labeled_faces.keys()):
        label_id = labeled_faces[face]
        label_name = label_names[label_id] if label_id < len(label_names) else f"unknown_{label_id}"
        color = color_table.get(label_name, [128, 128, 128])
        
        occ_color = Quantity_Color(
            color[0] / 255.0, color[1] / 255.0, color[2] / 255.0,
            Quantity_TOC_RGB
        )
        
        if with_label_name:
            display.DisplayShape(face, color=occ_color, transparency=0.3)
            display.DisplayMessage(
                get_face_center(face),
                f"{idx}: {label_name}",
                height=20, message_color=(0, 0, 0)
            )
        else:
            display.DisplayShape(face, color=occ_color, transparency=0.0)
    
    display.FitAll()
    display.Repaint()
    
    # 콘솔 출력
    print(f"\n=== {model['name']} ===")
    print(f"  Total: {model['total']} faces")
    for name, count in sorted(model["counts"].items()):
        print(f"  {name}: {count}")


# ============================================================================
# 메뉴 콜백 팩토리
# ============================================================================

def make_color_callback(model_idx: int, name: str):
    """색상만 표시 콜백."""
    def callback(*args):
        render_model(models[model_idx], with_label_name=False)
    callback.__name__ = name
    return callback


def make_label_callback(model_idx: int, name: str):
    """색상 + 라벨명 표시 콜백."""
    def callback(*args):
        render_model(models[model_idx], with_label_name=False)
    callback.__name__ = name
    return callback


# ============================================================================
# Main
# ============================================================================

def main():
    global display, models
    
    # 1. 트리 로드
    trees_path = Path("trees_N6_H3.json")
    results_tree_path = Path("results/trees/trees_N6_H3.json")
    
    if trees_path.exists():
        trees = load_trees(str(trees_path))
    elif results_tree_path.exists():
        trees = load_trees(str(results_tree_path))
    else:
        trees = generate_trees(6, 3)
    
    print(f"트리 {len(trees)}개 로드됨")
    
    # 2. 다양한 트리 선택 & 모델 생성
    selected = select_diverse_trees(trees, count=5)
    
    print(f"\n{'=' * 60}")
    print(f"모델 {len(selected)}개 생성 시작")
    print(f"{'=' * 60}")
    
    models = generate_models(trees, selected)
    
    if not models:
        print("생성된 모델이 없습니다.")
        return
    
    # 3. 라벨 STEP 저장
    output_dir = Path("results/labeled_step")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. OCC Viewer + 메뉴
    print(f"\n{'=' * 60}")
    print("OCC Viewer 실행")
    print(f"  모델 {len(models)}개 등록됨 (메뉴에서 선택)")
    print(f"{'=' * 60}")
    
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    
    # 모델별 메뉴 등록
    add_menu("Models")
    for i, model in enumerate(models):
        add_function_to_menu("Models", make_color_callback(i, model["name"]))
    
    add_menu("Models + Labels")
    for i, model in enumerate(models):
        add_function_to_menu("Models + Labels", make_label_callback(i, model["name"]))
    
    # 첫 번째 모델 자동 표시
    render_model(models[0], with_label_name=False)
    
    start_display()


if __name__ == "__main__":
    main()
