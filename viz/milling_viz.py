"""
밀링 프로세스 시각화 모듈

터닝+밀링 형상 생성 결과를 시각화:
- 홀별 유효 면 + 배치 시각화
- 최종 형상 시각화
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer

from core import (
    FaceAnalyzer, FeaturePlacement, HolePlacement, 
    FaceDimensionResult, FeatureParams, HoleParams, FeatureType
)
from core.milling_adder import compute_hole_scale_range


# ============================================================================
# 피처 메시 생성 헬퍼
# ============================================================================

def create_feature_mesh(placement: FeaturePlacement) -> Optional[pv.PolyData]:
    """
    피처 배치 정보로부터 PyVista 메시 생성.
    
    - 홀: Cylinder
    - 사각 피처: Box
    """
    center = np.array([
        placement.center_3d.X(), 
        placement.center_3d.Y(), 
        placement.center_3d.Z()
    ])
    direction = np.array([
        placement.direction.X(), 
        placement.direction.Y(), 
        placement.direction.Z()
    ])
    
    feature_type = placement.feature_type
    
    if feature_type in [FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE]:
        # 홀: 원기둥
        return pv.Cylinder(
            center=center + direction * placement.depth / 2,
            direction=direction.tolist(),
            radius=placement.diameter / 2,
            height=placement.depth
        )
    elif feature_type in [FeatureType.RECTANGULAR_POCKET, FeatureType.RECTANGULAR_PASSAGE]:
        # 사각 피처: 박스
        # 실제 모델 생성 코드(create_rectangular_pocket)와 동일한 좌표계 계산
        
        # local_x, local_y 계산 (core/milling_adder.py와 동일)
        z_axis = np.array([0, 0, 1])
        if abs(direction[2]) > 0.9:
            # 방향이 거의 Z축인 경우
            local_x = np.array([1, 0, 0])
        else:
            local_x = np.cross(direction, z_axis)
            local_x = local_x / np.linalg.norm(local_x)
        
        local_y = np.cross(direction, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        
        # 박스 꼭짓점 계산 (중심에서 시작)
        half_w = placement.width / 2
        half_l = placement.length / 2
        
        # 시작점 (중심에서 -half_w, -half_l 오프셋)
        start = center - half_w * local_x - half_l * local_y
        
        # 8개 꼭짓점 계산
        vertices = []
        for dz in [0, placement.depth]:
            for dy in [0, placement.length]:
                for dx in [0, placement.width]:
                    pt = start + dx * local_x + dy * local_y + dz * direction
                    vertices.append(pt)
        
        # 면 정의 (PyVista 형식)
        faces = np.array([
            [4, 0, 1, 3, 2],  # bottom
            [4, 4, 5, 7, 6],  # top
            [4, 0, 1, 5, 4],  # front
            [4, 2, 3, 7, 6],  # back
            [4, 0, 2, 6, 4],  # left
            [4, 1, 3, 7, 5],  # right
        ])
        
        box = pv.PolyData(np.array(vertices), faces)
        return box
    
    return None


def get_feature_label(placement: FeaturePlacement, idx: int) -> str:
    """피처 라벨 생성."""
    feature_type = placement.feature_type
    
    if feature_type in [FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE]:
        type_name = "H" if feature_type == FeatureType.BLIND_HOLE else "TH"
        return f"{type_name}{idx+1}\nD={placement.diameter:.2f}mm\nd={placement.depth:.2f}mm"
    else:
        type_name = "P" if feature_type == FeatureType.RECTANGULAR_POCKET else "PS"
        return f"{type_name}{idx+1}\n{placement.width:.1f}x{placement.length:.1f}mm\nd={placement.depth:.2f}mm"


def get_feature_size(placement: FeaturePlacement) -> float:
    """피처의 대표 크기 반환."""
    if placement.diameter > 0:
        return placement.diameter
    return max(placement.width, placement.length)


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class FaceAnalysisData:
    """시각화를 위한 면 분석 데이터"""
    face_id: int
    face: TopoDS_Face
    face_type: str
    width: float
    height: float
    hole_d_min: float
    hole_d_max: float
    is_valid_for_milling: bool
    dimension: FaceDimensionResult
    r_outer: float = 0.0
    r_inner: float = 0.0
    z_position: float = 0.0


# ============================================================================
# OCC → PyVista 변환
# ============================================================================

def face_to_pyvista(face: TopoDS_Face, mesh_quality: float = 0.1) -> Optional[pv.PolyData]:
    """TopoDS_Face를 PyVista PolyData로 변환"""
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)
    
    if triangulation is None:
        return None
    
    vertices = []
    nb_nodes = triangulation.NbNodes()
    for i in range(1, nb_nodes + 1):
        node = triangulation.Node(i)
        if not location.IsIdentity():
            node = node.Transformed(location.Transformation())
        vertices.append([node.X(), node.Y(), node.Z()])
    
    faces = []
    nb_triangles = triangulation.NbTriangles()
    for i in range(1, nb_triangles + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        faces.append([3, n1 - 1, n2 - 1, n3 - 1])
    
    if not vertices or not faces:
        return None
    
    return pv.PolyData(np.array(vertices), np.hstack(faces))


def shape_to_pyvista(shape: TopoDS_Shape, mesh_quality: float = 0.1) -> Optional[pv.PolyData]:
    """OCC Shape를 PyVista PolyData로 변환."""
    mesh_tool = BRepMesh_IncrementalMesh(shape, mesh_quality, False, 0.5, True)
    mesh_tool.Perform()
    
    topo = TopologyExplorer(shape)
    meshes = []
    
    for face in topo.faces():
        mesh = face_to_pyvista(face, mesh_quality)
        if mesh is not None and mesh.n_points > 0:
            meshes.append(mesh)
    
    if not meshes:
        return None
    
    return pv.merge(meshes)


# ============================================================================
# 면 분석 (시각화용)
# ============================================================================

def analyze_faces_for_visualization(shape: TopoDS_Shape, params: HoleParams) -> List[FaceAnalysisData]:
    """형상의 모든 면을 분석하여 시각화용 데이터 생성."""
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    analyzer = FaceAnalyzer()
    dim_results = analyzer.analyze_shape(shape)
    
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    analysis_results = []
    
    for dim in dim_results:
        face = faces[dim.face_id] if dim.face_id < len(faces) else None
        if face is None:
            continue
        
        width = dim.width if dim.width is not None else 0.0
        height = dim.height if dim.height is not None else width
        
        d_min, d_max = compute_hole_scale_range(dim, params)
        is_valid = d_max >= d_min and d_max > 0
        
        analysis_results.append(FaceAnalysisData(
            face_id=dim.face_id,
            face=face,
            face_type=dim.surface_type,
            width=width,
            height=height,
            hole_d_min=d_min,
            hole_d_max=d_max,
            is_valid_for_milling=is_valid,
            dimension=dim,
            r_outer=dim.r_outer if dim.r_outer > 0 else dim.r_max,
            r_inner=dim.r_inner if dim.r_inner > 0 else dim.r_min,
            z_position=(dim.z_max + dim.z_min) / 2,
        ))
    
    return analysis_results


# ============================================================================
# 홀별 유효 면 + 배치 시각화
# ============================================================================

def visualize_hole_valid_faces_and_placement(
    shape: TopoDS_Shape,
    analysis_results: List[FaceAnalysisData],
    placement: HolePlacement,
    hole_idx: int,
    prev_placements: List[HolePlacement],
    output_path: Path
):
    """
    각 홀에 대해:
    1. 해당 홀 직경 기준 유효 면 표시
       - 선택된 면: 밝은 초록
       - 다른 후보 면: 노란색
       - 비유효 면: 회색
    2. 홀 위치 및 방향 표시
    
    Side View 단일 뷰, 터닝 형상 투명하게
    """
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    
    current_size = get_feature_size(placement)
    
    # 후보 valid face 개수 카운트
    n_candidates = 0
    
    # 면들 렌더링
    for result in analysis_results:
        mesh = face_to_pyvista(result.face)
        if mesh is None or mesh.n_points == 0:
            continue
        
        # 해당 피처 크기 기준 유효성 판단
        is_valid_for_this_hole = (result.hole_d_min <= current_size <= result.hole_d_max)
        
        if result.face_id == placement.face_id:
            # 선택된 면 (홀이 배치된 면): 밝은 초록
            plotter.add_mesh(mesh, color='limegreen', opacity=0.7, 
                           show_edges=True, edge_color='darkgreen', line_width=2)
            n_candidates += 1
        elif is_valid_for_this_hole:
            # 다른 후보 valid face: 노란색으로 하이라이트
            plotter.add_mesh(mesh, color='gold', opacity=0.5, 
                           show_edges=True, edge_color='darkorange', line_width=1.5)
            n_candidates += 1
        else:
            # 비유효 면: 회색, 매우 투명
            plotter.add_mesh(mesh, color='lightgray', opacity=0.15, 
                           show_edges=True, edge_color='gray', line_width=0.5)
    
    # 이전 피처들 표시 (회색)
    for prev_p in prev_placements:
        prev_mesh = create_feature_mesh(prev_p)
        if prev_mesh:
            plotter.add_mesh(prev_mesh, color='dimgray', opacity=0.7)
    
    # 현재 피처 표시
    curr_center = np.array([placement.center_3d.X(), placement.center_3d.Y(), placement.center_3d.Z()])
    curr_dir = np.array([placement.direction.X(), placement.direction.Y(), placement.direction.Z()])
    
    # 피처 메시 (빨간색)
    curr_mesh = create_feature_mesh(placement)
    if curr_mesh:
        plotter.add_mesh(curr_mesh, color='red', opacity=0.9)
    
    # 피처 중심점
    plotter.add_points(curr_center.reshape(1, -1), color='red', point_size=15, 
                      render_points_as_spheres=True)
    
    # 방향 화살표 (파란색) - 크기 축소
    arrow_length = placement.depth * 0.5
    arrow = pv.Arrow(start=curr_center.tolist(), 
                    direction=curr_dir.tolist(), 
                    scale=arrow_length,
                    tip_radius=0.15,
                    shaft_radius=0.05)
    plotter.add_mesh(arrow, color='blue')
    
    # 피처 라벨 표시
    plotter.add_point_labels(
        [curr_center.tolist()], 
        [get_feature_label(placement, hole_idx)],
        font_size=11, text_color='white', shape_color='darkred', 
        shape_opacity=0.9, point_size=0
    )
    
    plotter.add_axes()
    
    # 타이틀 (피처 타입 표시)
    feature_name = placement.feature_type.value if hasattr(placement, 'feature_type') else "hole"
    plotter.add_title(
        f'Feature {hole_idx+1} ({feature_name}): Valid Faces ({n_candidates} candidates)\n'
        f'Selected: Face {placement.face_id} ({placement.face_type})',
        font_size=12
    )
    
    # Side View (XZ 평면에서 보기)
    plotter.camera_position = 'xz'
    plotter.camera.azimuth = 0
    plotter.camera.elevation = 0
    
    plotter.screenshot(str(output_path))
    plotter.close()


def visualize_final_shape_with_holes(
    final_shape: TopoDS_Shape,
    placements: List[FeaturePlacement],
    output_path: Path
):
    """
    최종 형상 시각화 (Side View, 단일 뷰)
    터닝 형상 투명, 피처 위치/방향 강조
    """
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    
    # 전체 형상 (투명하게)
    mesh = shape_to_pyvista(final_shape)
    if mesh is not None:
        plotter.add_mesh(mesh, color='steelblue', opacity=0.25, 
                        show_edges=True, edge_color='gray', line_width=0.5)
    
    # 각 피처 표시
    for i, p in enumerate(placements):
        center = np.array([p.center_3d.X(), p.center_3d.Y(), p.center_3d.Z()])
        direction = np.array([p.direction.X(), p.direction.Y(), p.direction.Z()])
        
        # 피처 메시
        feature_mesh = create_feature_mesh(p)
        if feature_mesh:
            plotter.add_mesh(feature_mesh, color='red', opacity=0.8)
        
        # 중심점
        plotter.add_points(center.reshape(1, -1), color='red', point_size=12, 
                          render_points_as_spheres=True)
        
        # 방향 화살표 - 크기 축소
        arrow = pv.Arrow(start=center.tolist(), direction=direction.tolist(), 
                        scale=p.depth * 0.4,
                        tip_radius=0.15,
                        shaft_radius=0.05)
        plotter.add_mesh(arrow, color='blue')
        
        # 피처 라벨
        plotter.add_point_labels(
            [center.tolist()], 
            [get_feature_label(p, i)],
            font_size=10, text_color='white', shape_color='darkred', 
            shape_opacity=0.85, point_size=0
        )
    
    plotter.add_axes()
    plotter.add_title(f'Final Shape with {len(placements)} Features', font_size=14)
    
    # Side View
    plotter.camera_position = 'xz'
    plotter.camera.azimuth = 0
    plotter.camera.elevation = 0
    
    plotter.screenshot(str(output_path))
    plotter.close()


# ============================================================================
# 전체 시각화 프로세스
# ============================================================================

def visualize_milling_process(
    turning_shape: TopoDS_Shape,
    final_shape: TopoDS_Shape,
    placements: List[FeaturePlacement],
    params: FeatureParams,
    output_dir: Path,
    model_name: str
):
    """
    밀링 프로세스 시각화.
    
    저장되는 이미지:
    - feature_01_valid_faces.png: 피처 1 기준 유효 면 + 배치
    - feature_02_valid_faces.png: 피처 2 기준 유효 면 + 배치
    - ...
    - final_shape.png: 최종 형상
    """
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  시각화 저장: {model_dir}/")
    
    # 면 분석
    analysis_results = analyze_faces_for_visualization(turning_shape, params)
    
    # 각 피처별 유효 면 + 배치 시각화
    if placements:
        print("  [Feature] 피처별 유효 면 및 배치 시각화...")
        for i, placement in enumerate(placements):
            prev_placements = placements[:i]
            output_path = model_dir / f"feature_{i+1:02d}_valid_faces.png"
            
            visualize_hole_valid_faces_and_placement(
                turning_shape,
                analysis_results,
                placement,
                i,
                prev_placements,
                output_path
            )
            print(f"    - feature_{i+1:02d}_valid_faces.png")
    
    # 최종 형상
    print("  [Final] 최종 형상 시각화...")
    visualize_final_shape_with_holes(
        final_shape,
        placements,
        model_dir / "final_shape.png"
    )
    print(f"    - final_shape.png")
    
    n_images = len(placements) + 1
    print(f"\n  시각화 완료: {n_images}개 이미지 생성")
