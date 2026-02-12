"""
라벨링 결과 OCC 3D Viewer 시각화

Face 라벨에 따라 색상을 입혀 3D Viewer에 표시합니다.
LABEL_PROPS.json의 색상 설정을 사용합니다.
"""

import json
from typing import Dict, Tuple, List

from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt


def load_label_props(file_path: str) -> Tuple[List[str], Dict[str, List[int]]]:
    """라벨 속성 JSON 파일 로드."""
    with open(file_path, 'r', encoding='utf-8') as f:
        props = json.load(f)
    label_names: List[str] = props['LABEL_NAMES']
    color_table: Dict[str, List[int]] = props['LABEL_COLORS']
    return label_names, color_table


def get_face_center(face: TopoDS_Face) -> gp_Pnt:
    """Face의 무게 중심점 계산."""
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    center = props.CentreOfMass()
    return gp_Pnt(center.X(), center.Y(), center.Z())


def display_labeled_faces(
    labeled_faces: Dict[TopoDS_Face, int],
    props_file_path: str = "config/LABEL_PROPS.json",
    with_label_name: bool = False
) -> None:
    """
    라벨링된 Face를 OCC 3D Viewer에 색상별로 표시.
    
    Args:
        labeled_faces: {Face: label_id} 매핑
        props_file_path: LABEL_PROPS.json 경로
        with_label_name: True면 Face 중심에 라벨명 텍스트 표시
    """
    display, start_display, _, _ = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    label_names, color_table = load_label_props(props_file_path)
    
    for idx, face in enumerate(labeled_faces.keys()):
        label_id = labeled_faces[face]
        if label_id < len(label_names):
            label_name = label_names[label_id]
        else:
            label_name = f"unknown_{label_id}"
        
        if label_name in color_table:
            color = color_table[label_name]
        else:
            color = [128, 128, 128]
        
        occ_color = Quantity_Color(
            color[0] / 255.0, color[1] / 255.0, color[2] / 255.0,
            Quantity_TOC_RGB
        )
        
        if with_label_name:
            display.DisplayShape(
                face, update=True,
                color=occ_color, transparency=0.3
            )
            display.DisplayMessage(
                get_face_center(face),
                f"{idx}: {label_name}",
                height=25,
                message_color=(0, 0, 0)
            )
        else:
            display.DisplayShape(
                face, color=occ_color, transparency=0.0
            )
    
    display.FitAll()
    
    # 범례 출력
    label_counts: Dict[str, int] = {}
    for label_id in labeled_faces.values():
        name = label_names[label_id] if label_id < len(label_names) else f"unknown_{label_id}"
        label_counts[name] = label_counts.get(name, 0) + 1
    
    print("\n=== Face Label Summary ===")
    for name, count in sorted(label_counts.items()):
        if name in color_table:
            c = color_table[name]
            print(f"  {name}: {count} faces  (RGB: {c[0]}, {c[1]}, {c[2]})")
        else:
            print(f"  {name}: {count} faces")
    print(f"  Total: {len(labeled_faces)} faces")
    
    start_display()
