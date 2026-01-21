"""
밀링 프로세스 시각화 모듈

터닝 형상에 밀링 특징형상을 추가하는 과정을 단계별로 시각화.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

import pyvista as pv

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.milling_adder import HolePlacement


# ============================================================================
# PyVista Mesh Conversion
# ============================================================================

def shape_to_pyvista(shape: TopoDS_Shape, mesh_quality: float = 0.1) -> Optional[pv.PolyData]:
    """
    OCC Shape를 PyVista PolyData로 변환.
    
    Args:
        shape: TopoDS_Shape
        mesh_quality: 메쉬 품질 (낮을수록 정밀)
        
    Returns:
        pv.PolyData 또는 None
    """
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, mesh_quality, False, 0.5, True)
    mesh_tool.Perform()
    
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    location = TopLoc_Location()
    
    for face in faces:
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            continue
        
        # 정점 추출
        nb_nodes = triangulation.NbNodes()
        for i in range(1, nb_nodes + 1):
            node = triangulation.Node(i)
            if not location.IsIdentity():
                node = node.Transformed(location.Transformation())
            all_vertices.append([node.X(), node.Y(), node.Z()])
        
        # 삼각형 추출
        nb_triangles = triangulation.NbTriangles()
        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            all_faces.append([3, vertex_offset + n1 - 1, 
                             vertex_offset + n2 - 1, 
                             vertex_offset + n3 - 1])
        
        vertex_offset += nb_nodes
    
    if not all_vertices or not all_faces:
        return None
    
    vertices = np.array(all_vertices)
    faces = np.hstack(all_faces)
    
    return pv.PolyData(vertices, faces)


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_milling_process(
    original_shape: TopoDS_Shape,
    final_shape: TopoDS_Shape,
    placements: List[HolePlacement],
    output_dir: str = "milling_viz_results",
    model_name: str = "model"
):
    """
    밀링 프로세스를 단계별로 시각화.
    
    Args:
        original_shape: 밀링 전 터닝 형상
        final_shape: 밀링 후 최종 형상
        placements: 홀 배치 정보 리스트
        output_dir: 출력 디렉토리
        model_name: 모델 이름
    """
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n밀링 프로세스 시각화: {output_path}")
    
    # 1. 원본 터닝 형상
    _save_shape_visualization(
        original_shape,
        str(output_path / "step1_original_turning.png"),
        title="Step 1: Original Turning Shape"
    )
    
    # 2. 최종 형상 (밀링 후)
    _save_shape_visualization(
        final_shape,
        str(output_path / "step2_final_with_milling.png"),
        title="Step 2: Final Shape with Milling Features"
    )
    
    # 3. 각 홀 위치 표시
    if placements:
        _save_holes_visualization(
            final_shape,
            placements,
            str(output_path / "step3_hole_positions.png"),
            title="Step 3: Hole Positions"
        )
    
    print(f"  시각화 완료: {len(placements)}개 홀")


def _save_shape_visualization(
    shape: TopoDS_Shape,
    filepath: str,
    title: str = "Shape"
):
    """형상을 PNG 파일로 저장."""
    mesh = shape_to_pyvista(shape)
    if mesh is None:
        print(f"    메쉬 변환 실패: {filepath}")
        return
    
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.add_mesh(mesh, color='steelblue', show_edges=True, 
                     edge_color='gray', opacity=1.0)
    plotter.add_axes()
    plotter.add_title(title, font_size=14)
    
    # 카메라 설정
    plotter.camera_position = 'iso'
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 30
    
    plotter.screenshot(filepath)
    plotter.close()
    print(f"    저장: {Path(filepath).name}")


def _save_holes_visualization(
    shape: TopoDS_Shape,
    placements: List[HolePlacement],
    filepath: str,
    title: str = "Holes"
):
    """형상과 홀 위치를 함께 시각화."""
    mesh = shape_to_pyvista(shape)
    if mesh is None:
        print(f"    메쉬 변환 실패: {filepath}")
        return
    
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.add_mesh(mesh, color='steelblue', show_edges=True, 
                     edge_color='gray', opacity=0.8)
    
    # 홀 위치 표시
    for i, p in enumerate(placements):
        center = [p.center_3d.X(), p.center_3d.Y(), p.center_3d.Z()]
        
        # 점으로 표시
        point = pv.PolyData(np.array([center]))
        plotter.add_mesh(point, color='red', point_size=15, 
                        render_points_as_spheres=True)
        
        # 방향 화살표
        direction = [p.direction.X(), p.direction.Y(), p.direction.Z()]
        arrow = pv.Arrow(start=center, direction=direction, 
                        scale=p.depth * 0.5)
        plotter.add_mesh(arrow, color='yellow')
        
        # 라벨
        plotter.add_point_labels(
            [center], [f"H{i+1}\nD={p.diameter:.1f}"],
            font_size=10, point_color='red', text_color='white',
            shape_opacity=0.7
        )
    
    plotter.add_axes()
    plotter.add_title(title, font_size=14)
    
    plotter.camera_position = 'iso'
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 30
    
    plotter.screenshot(filepath)
    plotter.close()
    print(f"    저장: {Path(filepath).name}")


def visualize_comparison(
    before_shape: TopoDS_Shape,
    after_shape: TopoDS_Shape,
    filepath: str,
    title: str = "Before vs After"
):
    """
    밀링 전후 비교 시각화.
    
    Args:
        before_shape: 밀링 전 형상
        after_shape: 밀링 후 형상
        filepath: 저장 경로
        title: 제목
    """
    before_mesh = shape_to_pyvista(before_shape)
    after_mesh = shape_to_pyvista(after_shape)
    
    if before_mesh is None or after_mesh is None:
        print(f"    메쉬 변환 실패")
        return
    
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1600, 800])
    
    # 왼쪽: Before
    plotter.subplot(0, 0)
    plotter.add_mesh(before_mesh, color='lightgray', show_edges=True, 
                     edge_color='gray', opacity=1.0)
    plotter.add_axes()
    plotter.add_title("Before (Turning Only)", font_size=12)
    plotter.camera_position = 'iso'
    
    # 오른쪽: After
    plotter.subplot(0, 1)
    plotter.add_mesh(after_mesh, color='steelblue', show_edges=True, 
                     edge_color='gray', opacity=1.0)
    plotter.add_axes()
    plotter.add_title("After (Turning + Milling)", font_size=12)
    plotter.camera_position = 'iso'
    
    plotter.screenshot(filepath)
    plotter.close()
    print(f"  비교 이미지 저장: {Path(filepath).name}")
