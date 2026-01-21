"""
터닝-밀링 모델 생성 과정 단계별 시각화

1. 각 면의 폭/너비 → 홀 스케일 범위 계산
2. 유효한 면 인식 및 하이라이트
3. 유효 UV 범위 시각화
4. 홀 배치 및 적용 과정 시각화
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Annulus, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer

from milling_feature_adder import (
    MillingFeatureAdder, HoleParams, HolePlacement,
    ValidFaceInfo, analyze_face_for_milling, compute_hole_scale_range
)
from projection_face_dimension import (
    compute_face_dimension, FaceDimensionResult,
    get_face_wires, sample_wire_points, points_to_rz, get_surface_type
)


# ============================================================================
# OCC → PyVista 변환
# ============================================================================

def face_to_pyvista(face: TopoDS_Face, linear_deflection: float = 0.1) -> Optional[pv.PolyData]:
    """TopoDS_Face를 PyVista PolyData로 변환"""
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)
    
    if triangulation is None:
        return None
    
    # 정점 추출
    vertices = []
    nb_nodes = triangulation.NbNodes()
    for i in range(1, nb_nodes + 1):
        node = triangulation.Node(i)
        if not location.IsIdentity():
            node = node.Transformed(location.Transformation())
        vertices.append([node.X(), node.Y(), node.Z()])
    
    # 삼각형 추출
    faces = []
    nb_triangles = triangulation.NbTriangles()
    for i in range(1, nb_triangles + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        faces.append([3, n1 - 1, n2 - 1, n3 - 1])
    
    if not vertices or not faces:
        return None
    
    vertices = np.array(vertices)
    faces = np.array(faces).flatten()
    
    return pv.PolyData(vertices, faces)


def shape_to_pyvista(shape: TopoDS_Shape, linear_deflection: float = 0.1) -> pv.PolyData:
    """TopoDS_Shape를 PyVista PolyData로 변환"""
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, linear_deflection, False, 0.5, True)
    mesh_tool.Perform()
    
    topo = TopologyExplorer(shape)
    meshes = []
    
    for face in topo.faces():
        mesh = face_to_pyvista(face, linear_deflection)
        if mesh is not None:
            meshes.append(mesh)
    
    if not meshes:
        return pv.PolyData()
    
    return pv.merge(meshes)


def load_step_file(filepath: str) -> Optional[TopoDS_Shape]:
    """STEP 파일 로드"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    
    if status == IFSelect_RetDone:
        reader.TransferRoots()
        return reader.OneShape()
    return None


# ============================================================================
# 단계별 분석 데이터
# ============================================================================

@dataclass
class FaceAnalysisData:
    """면 분석 데이터 (projection_face_dimension.py 기반)"""
    face_id: int
    face: TopoDS_Face
    face_type: str
    
    # 치수 (projection_face_dimension.py에서 계산)
    width: float
    height: float
    
    # 홀 스케일 범위
    hole_d_min: float
    hole_d_max: float
    
    # 유효성
    is_valid_for_milling: bool
    
    # 상세 치수
    dimension: FaceDimensionResult = None
    
    # 추가 정보
    r_outer: float = 0.0
    r_inner: float = 0.0
    z_position: float = 0.0


def analyze_faces_for_visualization(
    shape: TopoDS_Shape,
    params: HoleParams
) -> List[FaceAnalysisData]:
    """
    형상의 모든 면을 분석하여 시각화용 데이터 생성.
    
    projection_face_dimension.py의 치수 계산 함수 사용.
    """
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    analysis_results = []
    
    for i, face in enumerate(faces):
        # projection_face_dimension.py의 치수 계산 사용
        dim = compute_face_dimension(face, i, samples_per_edge=30)
        
        # 폭/너비 추출
        width = dim.width if dim.width is not None else 0.0
        height = dim.height if dim.height is not None else width
        
        # 홀 스케일 범위 계산
        d_min, d_max = compute_hole_scale_range(dim, params)
        
        # 유효성 판단: 최소 직경 이상의 홀이 가능해야 함
        is_valid = d_max >= d_min and d_max > 0
        
        analysis_results.append(FaceAnalysisData(
            face_id=i,
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
# Step 1: 폭/너비 및 홀 스케일 범위 시각화
# ============================================================================

def visualize_step1_dimensions_and_scale(
    analysis_results: List[FaceAnalysisData],
    output_path: str = "step1_dimensions_scale.png"
):
    """
    Step 1: 각 면의 폭/너비와 홀 스케일 범위 시각화
    """
    # 유효한 면만 필터링
    valid_faces = [r for r in analysis_results if r.width > 0.1]
    
    if not valid_faces:
        print("유효한 면이 없습니다.")
        return
    
    n_faces = len(valid_faces)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 색상 맵
    face_types = list(set(r.face_type for r in valid_faces))
    colors = plt.cm.Set2(np.linspace(0, 1, len(face_types)))
    type_color_map = dict(zip(face_types, colors))
    
    # ===== 서브플롯 1: 폭/너비 =====
    ax1 = axes[0]
    face_ids = [r.face_id for r in valid_faces]
    widths = [r.width for r in valid_faces]
    heights = [r.height if r.height else 0 for r in valid_faces]
    bar_colors = [type_color_map[r.face_type] for r in valid_faces]
    
    x = np.arange(len(face_ids))
    width_bar = 0.35
    
    ax1.bar(x - width_bar/2, widths, width_bar, color=bar_colors, 
            edgecolor='black', linewidth=0.5, label='Width')
    ax1.bar(x + width_bar/2, heights, width_bar, color=bar_colors,
            edgecolor='black', linewidth=0.5, alpha=0.6, label='Height')
    
    ax1.set_xlabel('Face ID', fontsize=11)
    ax1.set_ylabel('Dimension (mm)', fontsize=11)
    ax1.set_title('Step 1-A: Face Width & Height', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"F{fid}" for fid in face_ids], rotation=45, fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # ===== 서브플롯 2: 홀 스케일 범위 =====
    ax2 = axes[1]
    
    milling_faces = [r for r in valid_faces if r.face_type in ["Plane (Ring)", "Plane (Disk)", "Cylinder"]]
    
    if milling_faces:
        face_ids_m = [r.face_id for r in milling_faces]
        d_mins = [r.hole_d_min for r in milling_faces]
        d_maxs = [r.hole_d_max for r in milling_faces]
        bar_colors_m = [type_color_map[r.face_type] for r in milling_faces]
        
        x_m = np.arange(len(face_ids_m))
        
        # 범위 표시 (min-max)
        for i, (r, d_min, d_max) in enumerate(zip(milling_faces, d_mins, d_maxs)):
            color = type_color_map[r.face_type]
            ax2.bar(i, d_max - d_min, bottom=d_min, color=color, 
                   edgecolor='black', linewidth=0.5, alpha=0.7)
            # 점으로 min, max 표시
            ax2.plot(i, d_min, 'v', color='red', markersize=8)
            ax2.plot(i, d_max, '^', color='green', markersize=8)
            # 수치 표시
            ax2.text(i, d_max + 0.1, f'{d_max:.2f}', ha='center', fontsize=8, color='green')
            ax2.text(i, d_min - 0.15, f'{d_min:.2f}', ha='center', fontsize=8, color='red')
        
        ax2.set_xlabel('Face ID', fontsize=11)
        ax2.set_ylabel('Hole Diameter Range (mm)', fontsize=11)
        ax2.set_title('Step 1-B: Hole Scale Range\n(based on Width/Height)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_m)
        ax2.set_xticklabels([f"F{fid}" for fid in face_ids_m], rotation=45, fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # 범례
        ax2.plot([], [], 'v', color='red', markersize=8, label='D_min')
        ax2.plot([], [], '^', color='green', markersize=8, label='D_max')
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No milling-compatible faces', ha='center', va='center', 
                fontsize=14, transform=ax2.transAxes)
    
    # ===== 서브플롯 3: 면 타입별 분포 =====
    ax3 = axes[2]
    
    type_counts = {}
    for r in valid_faces:
        type_counts[r.face_type] = type_counts.get(r.face_type, 0) + 1
    
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors_pie = [type_color_map[t] for t in labels]
    
    explode = [0.05 if 'Plane' in t else 0 for t in labels]  # 평면 강조
    
    wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, 
                                        colors=colors_pie, autopct='%1.0f%%',
                                        shadow=True, startangle=90)
    ax3.set_title('Step 1-C: Face Type Distribution', fontsize=12, fontweight='bold')
    
    # 범례 (밀링 가능 여부)
    milling_types = ["Plane (Ring)", "Plane (Disk)", "Cylinder"]
    ax3.text(0, -1.4, f"Milling-compatible types: {', '.join(milling_types)}", 
            ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Step 1: Face Dimension Analysis & Hole Scale Range',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  저장: {output_path}")


# ============================================================================
# Step 2: 유효한 면 인식 시각화
# ============================================================================

def visualize_step2_valid_faces(
    shape: TopoDS_Shape,
    analysis_results: List[FaceAnalysisData],
    output_path: str = "step2_valid_faces.png"
):
    """
    Step 2: 유효한 밀링 대상 면 하이라이트 시각화 (PyVista)
    """
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 700), off_screen=True)
    
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    # ===== 왼쪽: 전체 형상 + 면 타입별 색상 =====
    pl.subplot(0, 0)
    pl.add_text("Step 2-A: All Faces by Type", font_size=12, position='upper_edge')
    
    type_colors = {
        "Plane (Ring)": (0.2, 0.8, 0.2),    # 녹색
        "Plane (Disk)": (0.3, 0.9, 0.3),    # 연녹색
        "Cylinder": (0.2, 0.4, 0.9),        # 파란색
        "Cone": (0.9, 0.6, 0.2),            # 주황색
        "Torus": (0.8, 0.2, 0.8),           # 보라색
        "Other": (0.5, 0.5, 0.5),           # 회색
    }
    
    for result in analysis_results:
        mesh = face_to_pyvista(result.face)
        if mesh is not None and mesh.n_points > 0:
            color = type_colors.get(result.face_type, (0.5, 0.5, 0.5))
            pl.add_mesh(mesh, color=color, opacity=0.8, show_edges=True, edge_color='gray')
            
            # 면 중심에 라벨
            center = mesh.center
            pl.add_point_labels([center], [f"F{result.face_id}"], font_size=10, 
                              text_color='black', shape_color='white', shape_opacity=0.7)
    
    pl.add_axes()
    pl.camera_position = 'iso'
    
    # ===== 오른쪽: 밀링 유효 면만 하이라이트 =====
    pl.subplot(0, 1)
    pl.add_text("Step 2-B: Valid Faces for Milling (Highlighted)", font_size=12, position='upper_edge')
    
    # 단순화된 색상: 유효성 기준
    VALID_COLOR = (0.2, 0.9, 0.2)      # 초록색 - 유효한 면
    INVALID_COLOR = (0.3, 0.5, 0.9)    # 파란색 - 유효하지 않은 면
    
    for result in analysis_results:
        mesh = face_to_pyvista(result.face)
        if mesh is not None and mesh.n_points > 0:
            if result.is_valid_for_milling:
                # 유효한 면: 초록색, 불투명
                pl.add_mesh(mesh, color=VALID_COLOR, opacity=1.0, show_edges=True, 
                           edge_color='black', line_width=2)
                
                # 면 정보 라벨
                center = mesh.center
                label = f"F{result.face_id}\nD:[{result.hole_d_min:.1f}-{result.hole_d_max:.1f}]"
                pl.add_point_labels([center], [label], font_size=9, text_color='white',
                                  shape_color='darkgreen', shape_opacity=0.9)
            else:
                # 비유효: 파란색, 반투명
                pl.add_mesh(mesh, color=INVALID_COLOR, opacity=0.5, show_edges=True, 
                           edge_color='gray', line_width=0.5)
    
    pl.add_axes()
    pl.camera_position = 'iso'
    
    pl.screenshot(output_path)
    pl.close()
    
    print(f"  저장: {output_path}")


# ============================================================================
# Step 3: 유효 UV 범위 시각화
# ============================================================================

def visualize_step3_valid_uv_regions(
    analysis_results: List[FaceAnalysisData],
    output_path: str = "step3_valid_uv.png"
):
    """
    Step 3: 유효 UV 범위 시각화 (면 타입별)
    
    projection_face_dimension.py의 치수 기준 사용:
    - Ring/Disk: X-Y 뷰 (반경 기반)
    - Cylinder: R-Z 뷰 (축방향 + 반경)
    """
    # 밀링 대상 면만 필터링 (유효하고 평면/원통 타입)
    milling_faces = [r for r in analysis_results 
                     if r.is_valid_for_milling and ("Plane" in r.face_type or "Cylinder" in r.face_type)]
    
    if not milling_faces:
        print("밀링 가능한 면이 없습니다.")
        return
    
    n_faces = min(6, len(milling_faces))
    n_cols = min(3, n_faces)
    n_rows = (n_faces + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_faces == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(milling_faces[:n_faces]):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        dim = result.dimension
        clearance = 0.4
        
        if "Ring" in result.face_type:
            # 링: X-Y 뷰로 환형 영역 표시
            theta = np.linspace(0, 2 * np.pi, 100)
            
            r_outer = result.r_outer
            r_inner = result.r_inner
            
            ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta),
                   'b-', linewidth=2, label=f'Outer R={r_outer:.2f}')
            ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta),
                   'r-', linewidth=2, label=f'Inner R={r_inner:.2f}')
            
            # 유효 영역
            margin = result.hole_d_max / 2 + clearance
            r_valid_min = r_inner + margin
            r_valid_max = r_outer - margin
            
            if r_valid_max > r_valid_min:
                for r in [r_valid_min, r_valid_max]:
                    ax.plot(r * np.cos(theta), r * np.sin(theta),
                           'g--', linewidth=1.5, alpha=0.7)
                ax.fill_between(r_valid_max * np.cos(theta), 
                               r_valid_min * np.sin(theta),
                               r_valid_max * np.sin(theta),
                               alpha=0.2, color='green', label='Valid Region')
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_aspect('equal')
            ax.set_title(f'Face {result.face_id}: {result.face_type}\n'
                        f'W={result.width:.2f}mm (thickness), Z={result.z_position:.2f}mm',
                        fontsize=10, fontweight='bold')
            
        elif "Disk" in result.face_type:
            # 디스크: X-Y 뷰
            theta = np.linspace(0, 2 * np.pi, 100)
            r_outer = result.r_outer
            
            ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta),
                   'b-', linewidth=2, label=f'R={r_outer:.2f}')
            
            margin = result.hole_d_max / 2 + clearance
            r_valid = r_outer - margin
            
            if r_valid > 0:
                ax.fill(r_valid * np.cos(theta), r_valid * np.sin(theta),
                       alpha=0.2, color='green', label='Valid Region')
                ax.plot(r_valid * np.cos(theta), r_valid * np.sin(theta),
                       'g--', linewidth=1.5)
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_aspect('equal')
            ax.set_title(f'Face {result.face_id}: {result.face_type}\n'
                        f'W=H={result.width:.2f}mm (diameter), Z={result.z_position:.2f}mm',
                        fontsize=10, fontweight='bold')
            
        elif "Cylinder" in result.face_type:
            # 원통: R-Z 뷰 (축방향 단면)
            # Width=직경(2R), Height=Δz
            radius = dim.r_max  # 반경
            z_min, z_max = dim.z_min, dim.z_max
            
            # 원통 단면 (R-Z)
            ax.plot([z_min, z_max], [radius, radius], 'b-', linewidth=2, label=f'R={radius:.2f}')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            
            # 유효 영역 (z방향 margin)
            margin = result.hole_d_max / 2 + clearance
            z_valid_min = z_min + margin
            z_valid_max = z_max - margin
            
            if z_valid_max > z_valid_min:
                ax.axvspan(z_valid_min, z_valid_max, alpha=0.2, color='green', label='Valid Z Range')
                ax.axvline(z_valid_min, color='green', linestyle='--', linewidth=1.5)
                ax.axvline(z_valid_max, color='green', linestyle='--', linewidth=1.5)
            
            ax.set_xlabel('Z (axial, mm)')
            ax.set_ylabel('R (radial, mm)')
            ax.set_title(f'Face {result.face_id}: {result.face_type}\n'
                        f'W={result.width:.2f}mm (dia), H={result.height:.2f}mm (Δz)',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        ax.legend(loc='upper right', fontsize=8)
    
    # 빈 서브플롯 숨기기
    for idx in range(n_faces, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Step 3: Valid Regions for Hole Placement\n(Based on projection_face_dimension.py)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  저장: {output_path}")


# ============================================================================
# Step 4: 홀 배치 시각화
# ============================================================================

def visualize_step4_hole_placement(
    shape: TopoDS_Shape,
    analysis_results: List[FaceAnalysisData],
    placements: List[HolePlacement],
    output_path: str = "step4_hole_placement.png"
):
    """
    Step 4: 홀 배치 결과 시각화
    """
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 700), off_screen=True)
    
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    # ===== 왼쪽: 홀 위치 및 방향 =====
    pl.subplot(0, 0)
    pl.add_text("Step 4-A: Hole Centers & Directions", font_size=12, position='upper_edge')
    
    # 전체 형상 (반투명)
    full_mesh = shape_to_pyvista(shape)
    if full_mesh.n_points > 0:
        pl.add_mesh(full_mesh, color='lightgray', opacity=0.5, show_edges=True)
    
    # 각 홀 위치와 방향 표시
    for i, placement in enumerate(placements):
        center = np.array([placement.center_3d.X(), 
                          placement.center_3d.Y(), 
                          placement.center_3d.Z()])
        direction = np.array([placement.direction.X(), 
                             placement.direction.Y(), 
                             placement.direction.Z()])
        
        # 홀 중심점
        pl.add_points(center.reshape(1, -1), color='red', point_size=15, 
                     render_points_as_spheres=True)
        
        # 방향 화살표
        arrow_length = placement.depth * 0.8
        arrow = pv.Arrow(start=center, direction=direction, scale=arrow_length)
        pl.add_mesh(arrow, color='blue')
        
        # 라벨
        label = f"H{i+1}\nD={placement.diameter:.2f}\ndepth={placement.depth:.2f}"
        pl.add_point_labels([center], [label], font_size=9, text_color='white',
                          shape_color='darkblue', shape_opacity=0.9)
    
    pl.add_axes()
    pl.camera_position = 'iso'
    
    # ===== 오른쪽: 대상 면과 홀 위치 상세 =====
    pl.subplot(0, 1)
    pl.add_text("Step 4-B: Hole Placement on Target Faces", font_size=12, position='upper_edge')
    
    # 대상 면들
    placed_face_ids = set(p.face_id for p in placements)
    
    for result in analysis_results:
        mesh = face_to_pyvista(result.face)
        if mesh is not None and mesh.n_points > 0:
            if result.face_id in placed_face_ids:
                # 홀이 배치된 면: 밝은 색
                pl.add_mesh(mesh, color='lightgreen', opacity=0.9, 
                           show_edges=True, edge_color='darkgreen', line_width=2)
            elif result.is_valid_for_milling:
                # 밀링 가능하지만 홀 없는 면
                pl.add_mesh(mesh, color='lightyellow', opacity=0.7, 
                           show_edges=True, edge_color='gray')
            else:
                # 기타 면
                pl.add_mesh(mesh, color='lightgray', opacity=0.3, 
                           show_edges=True, edge_color='gray', line_width=0.5)
    
    # 홀 원기둥 표시
    for i, placement in enumerate(placements):
        center = np.array([placement.center_3d.X(), 
                          placement.center_3d.Y(), 
                          placement.center_3d.Z()])
        direction = np.array([placement.direction.X(), 
                             placement.direction.Y(), 
                             placement.direction.Z()])
        
        # 홀 실린더 생성
        radius = placement.diameter / 2
        hole_cyl = pv.Cylinder(center=center + direction * placement.depth / 2,
                               direction=direction,
                               radius=radius,
                               height=placement.depth)
        pl.add_mesh(hole_cyl, color='red', opacity=0.8)
    
    pl.add_axes()
    pl.camera_position = 'iso'
    
    pl.screenshot(output_path)
    pl.close()
    
    print(f"  저장: {output_path}")


# ============================================================================
# 전체 프로세스 시각화
# ============================================================================

def visualize_step4_uv_sampling(
    analysis_results: List[FaceAnalysisData],
    placements: List[HolePlacement],
    output_path: str = "step4_uv_sampling.png"
):
    """
    Step 4 추가: 홀 배치 과정 시각화 (X-Y 뷰 또는 R-Z 뷰)
    """
    # 홀이 배치된 면만 필터링
    placed_face_ids = set(p.face_id for p in placements)
    placed_faces = [r for r in analysis_results if r.face_id in placed_face_ids]
    
    if not placed_faces:
        print("배치된 홀이 없습니다.")
        return
    
    n_faces = len(placed_faces)
    
    fig, axes = plt.subplots(1, n_faces, figsize=(6 * n_faces, 5))
    if n_faces == 1:
        axes = [axes]
    
    clearance = 0.4
    
    for idx, result in enumerate(placed_faces):
        ax = axes[idx]
        dim = result.dimension
        
        # 해당 면의 홀들
        face_placements = [p for p in placements if p.face_id == result.face_id]
        
        if "Plane" in result.face_type:
            # X-Y 뷰
            theta = np.linspace(0, 2 * np.pi, 100)
            
            r_outer = result.r_outer
            r_inner = result.r_inner
            
            ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta),
                   'b-', linewidth=2, label='Boundary')
            
            if "Ring" in result.face_type:
                ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta),
                       'b-', linewidth=2)
            
            # 유효 영역
            margin = result.hole_d_max / 2 + clearance
            r_valid_min = r_inner + margin if "Ring" in result.face_type else 0
            r_valid_max = r_outer - margin
            
            if r_valid_max > r_valid_min:
                ax.fill_between(r_valid_max * np.cos(theta),
                               r_valid_min * np.sin(theta) if r_valid_min > 0 else np.zeros(100),
                               r_valid_max * np.sin(theta),
                               alpha=0.2, color='green', label='Valid Region')
            
            # 홀 배치 표시
            for p in face_placements:
                cx, cy = p.center_3d.X(), p.center_3d.Y()
                hole_circle = plt.Circle((cx, cy), p.diameter / 2, 
                                         fill=True, facecolor='red', 
                                         edgecolor='darkred', linewidth=2, alpha=0.7)
                ax.add_patch(hole_circle)
                
                # 화살표 (방향)
                dx, dy = p.direction.X() * 2, p.direction.Y() * 2
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    ax.arrow(cx, cy, dx, dy, head_width=0.3, head_length=0.2,
                            fc='blue', ec='darkblue')
                
                ax.plot(cx, cy, 'k*', markersize=10)
                ax.text(cx + 0.5, cy + 0.5, f'D={p.diameter:.1f}', fontsize=8)
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_aspect('equal')
            
        elif "Cylinder" in result.face_type:
            # θ-Z 전개도 (원통 펼침)
            radius = dim.r_max
            z_min, z_max = dim.z_min, dim.z_max
            
            # 도메인 박스
            rect = plt.Rectangle((0, z_min), 2*np.pi, z_max - z_min,
                                 fill=True, facecolor='lightblue', edgecolor='blue',
                                 linewidth=2, alpha=0.5, label='Cylinder Domain')
            ax.add_patch(rect)
            
            # 유효 영역
            margin = result.hole_d_max / 2 + clearance
            z_valid_min = z_min + margin
            z_valid_max = z_max - margin
            
            if z_valid_max > z_valid_min:
                valid_rect = plt.Rectangle((0, z_valid_min), 2*np.pi, z_valid_max - z_valid_min,
                                          fill=True, facecolor='lightgreen',
                                          edgecolor='green', linewidth=2, alpha=0.5,
                                          label='Valid Region')
                ax.add_patch(valid_rect)
            
            # 홀 위치
            for p in face_placements:
                cx, cy, cz = p.center_3d.X(), p.center_3d.Y(), p.center_3d.Z()
                theta_pos = math.atan2(cy, cx)
                if theta_pos < 0:
                    theta_pos += 2 * math.pi
                
                # 홀 표시 (θ-Z 좌표)
                hole_circle = plt.Circle((theta_pos, cz), p.diameter / 2 / radius,
                                         fill=True, facecolor='red',
                                         edgecolor='darkred', linewidth=2, alpha=0.7)
                ax.add_patch(hole_circle)
                ax.plot(theta_pos, cz, 'k*', markersize=10)
                ax.text(theta_pos + 0.2, cz + 0.5, f'D={p.diameter:.1f}', fontsize=8)
            
            ax.set_xlabel('θ (rad)')
            ax.set_ylabel('Z (mm)')
            ax.set_xlim(-0.3, 2*np.pi + 0.3)
            ax.set_ylim(z_min - 1, z_max + 1)
        
        ax.set_title(f'Face {result.face_id}: {result.face_type}\n'
                    f'W={result.width:.1f}mm, H={result.height:.1f}mm',
                    fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Step 4: Hole Placement on Valid Regions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  저장: {output_path}")


def visualize_step4_sequential_holes(
    analysis_results: List[FaceAnalysisData],
    placements: List[HolePlacement],
    params: HoleParams,
    output_dir: str = ".",
    shape: TopoDS_Shape = None
):
    """
    Step 4 확장: 홀이 순차적으로 추가될 때 유효 영역 변화 시각화
    
    각 홀이 추가될 때마다:
    - 해당 홀 직경 기준으로 유효한 면 3D 표시
    - 이전에 배치된 홀들과의 간격 제약
    - 남은 유효 영역 표시
    """
    if not placements:
        print("  순차적 홀 시각화: 배치된 홀 없음")
        return
    
    n_holes = len(placements)
    
    # 홀이 배치된 면들의 정보 수집
    placed_face_ids = set(p.face_id for p in placements)
    placed_faces_dict = {r.face_id: r for r in analysis_results if r.face_id in placed_face_ids}
    
    # 각 홀에 대해 시각화
    for hole_idx, current_placement in enumerate(placements):
        # ===== Part A: 해당 홀 직경 기준 유효 면 3D 시각화 (PyVista) =====
        if shape is not None:
            pl = pv.Plotter(shape=(1, 2), window_size=(1400, 600), off_screen=True)
            
            current_diameter = current_placement.diameter
            
            # 왼쪽: 해당 홀 직경 기준 유효 면
            pl.subplot(0, 0)
            pl.add_text(f"Hole #{hole_idx+1} (D={current_diameter:.1f}mm): Valid Faces", 
                       font_size=11, position='upper_edge')
            
            for result in analysis_results:
                mesh = face_to_pyvista(result.face)
                if mesh is not None and mesh.n_points > 0:
                    # 해당 홀 직경 기준 유효성 판단
                    is_valid_for_this_hole = (
                        result.hole_d_min <= current_diameter <= result.hole_d_max
                    )
                    
                    if is_valid_for_this_hole:
                        pl.add_mesh(mesh, color=(0.2, 0.9, 0.2), opacity=1.0, 
                                   show_edges=True, edge_color='black', line_width=2)
                        center = mesh.center
                        pl.add_point_labels([center], [f"F{result.face_id}"], 
                                          font_size=10, text_color='white',
                                          shape_color='darkgreen', shape_opacity=0.9)
                    else:
                        pl.add_mesh(mesh, color=(0.3, 0.5, 0.9), opacity=0.4, 
                                   show_edges=True, edge_color='gray', line_width=0.5)
            
            pl.add_axes()
            pl.camera_position = 'iso'
            
            # 오른쪽: 홀 배치 상황
            pl.subplot(0, 1)
            pl.add_text(f"Hole #{hole_idx+1} Placement (Face {current_placement.face_id})", 
                       font_size=11, position='upper_edge')
            
            # 전체 형상 반투명
            for result in analysis_results:
                mesh = face_to_pyvista(result.face)
                if mesh is not None and mesh.n_points > 0:
                    if result.face_id == current_placement.face_id:
                        pl.add_mesh(mesh, color=(0.2, 0.9, 0.2), opacity=0.7, 
                                   show_edges=True, edge_color='black')
                    else:
                        pl.add_mesh(mesh, color='lightgray', opacity=0.3, 
                                   show_edges=True, edge_color='gray')
            
            # 이전 홀들 (회색 구)
            for prev_idx, prev_p in enumerate(placements[:hole_idx]):
                sphere = pv.Sphere(radius=prev_p.diameter/2, 
                                  center=[prev_p.center_3d.X(), prev_p.center_3d.Y(), prev_p.center_3d.Z()])
                pl.add_mesh(sphere, color='gray', opacity=0.6)
            
            # 현재 홀 (빨간 구)
            curr_sphere = pv.Sphere(radius=current_placement.diameter/2,
                                   center=[current_placement.center_3d.X(), 
                                          current_placement.center_3d.Y(), 
                                          current_placement.center_3d.Z()])
            pl.add_mesh(curr_sphere, color='red', opacity=0.9)
            
            # 방향 화살표
            arrow = pv.Arrow(start=[current_placement.center_3d.X(), 
                                   current_placement.center_3d.Y(), 
                                   current_placement.center_3d.Z()],
                           direction=[current_placement.direction.X() * 3,
                                    current_placement.direction.Y() * 3,
                                    current_placement.direction.Z() * 3],
                           scale='auto')
            pl.add_mesh(arrow, color='blue')
            
            pl.add_axes()
            pl.camera_position = 'iso'
            
            # 저장
            pv_output = f"{output_dir}/step4_hole_{hole_idx+1:02d}_3d.png"
            pl.screenshot(pv_output)
            pl.close()
            print(f"  저장: {pv_output}")
        
        # ===== Part B: 2D 상세 시각화 (Matplotlib) =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 이전 홀들 (0 ~ hole_idx-1)
        prev_placements = placements[:hole_idx]
        # 현재 홀
        current_hole = current_placement
        
        face_result = placed_faces_dict.get(current_hole.face_id)
        if face_result is None:
            continue
        
        dim = face_result.dimension
        clearance = params.clearance
        
        # ===== 왼쪽: 전체 형상에서 현재까지의 홀 배치 =====
        ax1 = axes[0]
        ax1.set_title(f'Step 4-{hole_idx+1}A: Hole #{hole_idx+1} Placement\n'
                     f'Face {current_hole.face_id} ({current_hole.face_type})',
                     fontsize=11, fontweight='bold')
        
        # 모든 유효 면 표시 (간략화된 R-Z 뷰)
        for face_id, result in placed_faces_dict.items():
            r = result.r_outer
            z_min, z_max = result.dimension.z_min, result.dimension.z_max
            
            # 면 영역 표시
            if result.face_type == "Plane (Disk)" or result.face_type == "Plane (Ring)":
                # 평면: 수평선으로 표시
                ax1.plot([0, r], [result.z_position, result.z_position], 
                        'b-', linewidth=3, alpha=0.7)
                ax1.fill_between([0, r], result.z_position - 0.3, result.z_position + 0.3,
                               alpha=0.3, color='green' if face_id == current_hole.face_id else 'lightgray')
            else:
                # 원통/Cone: 수직선으로 표시
                ax1.plot([r, r], [z_min, z_max], 'b-', linewidth=3, alpha=0.7)
                ax1.fill_betweenx([z_min, z_max], r - 0.5, r + 0.5,
                                alpha=0.3, color='blue' if face_id == current_hole.face_id else 'lightgray')
        
        # 이전 홀들 표시 (회색)
        for prev_p in prev_placements:
            prev_r = math.sqrt(prev_p.center_3d.X()**2 + prev_p.center_3d.Y()**2)
            prev_z = prev_p.center_3d.Z()
            circle = plt.Circle((prev_r, prev_z), prev_p.diameter/2, 
                               fill=True, facecolor='gray', edgecolor='darkgray',
                               linewidth=1.5, alpha=0.6)
            ax1.add_patch(circle)
            ax1.text(prev_r + 0.5, prev_z, f'H{placements.index(prev_p)+1}', fontsize=8, color='gray')
        
        # 현재 홀 표시 (빨간색)
        curr_r = math.sqrt(current_hole.center_3d.X()**2 + current_hole.center_3d.Y()**2)
        curr_z = current_hole.center_3d.Z()
        circle = plt.Circle((curr_r, curr_z), current_hole.diameter/2, 
                           fill=True, facecolor='red', edgecolor='darkred',
                           linewidth=2, alpha=0.8)
        ax1.add_patch(circle)
        ax1.text(curr_r + 0.5, curr_z + 0.5, f'H{hole_idx+1}\nD={current_hole.diameter:.1f}', 
                fontsize=9, color='red', fontweight='bold')
        
        ax1.set_xlabel('R (radial, mm)')
        ax1.set_ylabel('Z (axial, mm)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # ===== 오른쪽: 해당 면의 유효 영역 상세 =====
        ax2 = axes[1]
        ax2.set_title(f'Step 4-{hole_idx+1}B: Valid Region on Face {current_hole.face_id}\n'
                     f'After {hole_idx} hole(s), placing hole #{hole_idx+1}',
                     fontsize=11, fontweight='bold')
        
        if "Plane" in face_result.face_type:
            # X-Y 뷰
            theta = np.linspace(0, 2 * np.pi, 100)
            r_outer = face_result.r_outer
            r_inner = face_result.r_inner
            
            # 경계
            ax2.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), 'b-', linewidth=2)
            if "Ring" in face_result.face_type:
                ax2.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), 'b-', linewidth=2)
            
            # 기본 유효 영역
            margin = face_result.hole_d_max / 2 + clearance
            r_valid_min = r_inner + margin if "Ring" in face_result.face_type else 0
            r_valid_max = r_outer - margin
            
            if r_valid_max > r_valid_min:
                ax2.fill_between(r_valid_max * np.cos(theta),
                               r_valid_min * np.sin(theta) if r_valid_min > 0 else np.zeros(100),
                               r_valid_max * np.sin(theta),
                               alpha=0.2, color='green', label='Initial Valid')
            
            # 이전 홀들로 인한 제외 영역
            for prev_p in prev_placements:
                if prev_p.face_id == current_hole.face_id:
                    px, py = prev_p.center_3d.X(), prev_p.center_3d.Y()
                    exclusion_r = (prev_p.diameter + current_hole.diameter) / 2 + params.min_spacing
                    excl_circle = plt.Circle((px, py), exclusion_r,
                                           fill=True, facecolor='orange', alpha=0.3,
                                           edgecolor='darkorange', linewidth=1.5, linestyle='--')
                    ax2.add_patch(excl_circle)
                    
                    # 이전 홀
                    hole_circle = plt.Circle((px, py), prev_p.diameter / 2,
                                           fill=True, facecolor='gray', edgecolor='darkgray',
                                           linewidth=2, alpha=0.7)
                    ax2.add_patch(hole_circle)
            
            # 현재 홀
            cx, cy = current_hole.center_3d.X(), current_hole.center_3d.Y()
            hole_circle = plt.Circle((cx, cy), current_hole.diameter / 2,
                                    fill=True, facecolor='red', edgecolor='darkred',
                                    linewidth=2, alpha=0.8)
            ax2.add_patch(hole_circle)
            ax2.plot(cx, cy, 'k*', markersize=12)
            ax2.text(cx + 1, cy + 1, f'H{hole_idx+1}\nD={current_hole.diameter:.1f}mm',
                    fontsize=10, color='red', fontweight='bold')
            
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
            ax2.set_aspect('equal')
            
        elif "Cylinder" in face_result.face_type:
            # θ-Z 전개도
            radius = dim.r_max
            z_min, z_max = dim.z_min, dim.z_max
            
            # 도메인
            rect = plt.Rectangle((0, z_min), 2*np.pi, z_max - z_min,
                                fill=True, facecolor='lightblue', edgecolor='blue',
                                linewidth=2, alpha=0.5)
            ax2.add_patch(rect)
            
            # 기본 유효 영역
            margin = face_result.hole_d_max / 2 + clearance
            z_valid_min = z_min + margin
            z_valid_max = z_max - margin
            
            if z_valid_max > z_valid_min:
                valid_rect = plt.Rectangle((0, z_valid_min), 2*np.pi, z_valid_max - z_valid_min,
                                          fill=True, facecolor='lightgreen',
                                          edgecolor='green', linewidth=2, alpha=0.4)
                ax2.add_patch(valid_rect)
            
            # 이전 홀들로 인한 제외 영역
            for prev_p in prev_placements:
                if prev_p.face_id == current_hole.face_id:
                    px, py, pz = prev_p.center_3d.X(), prev_p.center_3d.Y(), prev_p.center_3d.Z()
                    p_theta = math.atan2(py, px)
                    if p_theta < 0:
                        p_theta += 2 * math.pi
                    
                    exclusion_r = (prev_p.diameter + current_hole.diameter) / 2 + params.min_spacing
                    excl_theta = exclusion_r / radius
                    
                    excl_rect = plt.Rectangle((p_theta - excl_theta, pz - exclusion_r),
                                             2 * excl_theta, 2 * exclusion_r,
                                             fill=True, facecolor='orange', alpha=0.3,
                                             edgecolor='darkorange', linewidth=1.5, linestyle='--')
                    ax2.add_patch(excl_rect)
                    
                    # 이전 홀
                    hole_circle = plt.Circle((p_theta, pz), prev_p.diameter / 2 / radius,
                                           fill=True, facecolor='gray', edgecolor='darkgray',
                                           linewidth=2, alpha=0.7)
                    ax2.add_patch(hole_circle)
            
            # 현재 홀
            cx, cy, cz = current_hole.center_3d.X(), current_hole.center_3d.Y(), current_hole.center_3d.Z()
            c_theta = math.atan2(cy, cx)
            if c_theta < 0:
                c_theta += 2 * math.pi
            
            hole_circle = plt.Circle((c_theta, cz), current_hole.diameter / 2 / radius,
                                    fill=True, facecolor='red', edgecolor='darkred',
                                    linewidth=2, alpha=0.8)
            ax2.add_patch(hole_circle)
            ax2.plot(c_theta, cz, 'k*', markersize=12)
            ax2.text(c_theta + 0.3, cz + 0.5, f'H{hole_idx+1}\nD={current_hole.diameter:.1f}mm',
                    fontsize=10, color='red', fontweight='bold')
            
            ax2.set_xlabel('θ (rad)')
            ax2.set_ylabel('Z (mm)')
            ax2.set_xlim(-0.5, 2*np.pi + 0.5)
            ax2.set_ylim(z_min - 1, z_max + 1)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'Sequential Hole Placement: Hole {hole_idx+1}/{n_holes}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = f"{output_dir}/step4_hole_{hole_idx+1:02d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  저장: {output_path}")


def visualize_full_process(
    step_file: str,
    step_output_dir: str = "generated_turning_milling_STEP",
    viz_output_dir: str = "milling_visualization",
    params: HoleParams = None
):
    """
    전체 밀링 특징형상 추가 프로세스 시각화
    
    Args:
        step_file: 입력 STEP 파일 경로
        step_output_dir: STEP 결과물 저장 폴더
        viz_output_dir: 시각화 이미지 저장 폴더
        params: 홀 파라미터
    """
    if params is None:
        params = HoleParams()  # 기본값 사용
    
    # 출력 폴더 생성
    step_output_path = Path(step_output_dir)
    step_output_path.mkdir(exist_ok=True)
    
    viz_output_path = Path(viz_output_dir)
    viz_output_path.mkdir(exist_ok=True)
    
    # 모델별 시각화 하위 폴더
    model_name = Path(step_file).stem
    model_viz_path = viz_output_path / model_name
    model_viz_path.mkdir(exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"터닝-밀링 생성 과정 시각화")
    print(f"입력: {step_file}")
    print(f"STEP 출력: {step_output_dir}/")
    print(f"시각화 출력: {viz_output_dir}/{model_name}/")
    print(f"{'=' * 60}")
    
    # 형상 로드
    shape = load_step_file(step_file)
    if shape is None:
        print("형상 로드 실패")
        return None, []
    
    # Step 1: 면 분석
    print("\n[Step 1] 면 치수 분석 및 홀 스케일 범위 계산...")
    analysis_results = analyze_faces_for_visualization(shape, params)
    visualize_step1_dimensions_and_scale(
        analysis_results, 
        str(model_viz_path / "step1_dimensions_scale.png")
    )
    
    # Step 2: 유효 면 인식
    print("\n[Step 2] 밀링 유효 면 인식...")
    visualize_step2_valid_faces(
        shape, 
        analysis_results,
        str(model_viz_path / "step2_valid_faces.png")
    )
    
    # Step 3: UV 영역
    print("\n[Step 3] 유효 UV 범위 계산...")
    visualize_step3_valid_uv_regions(
        analysis_results,
        str(model_viz_path / "step3_valid_uv.png")
    )
    
    # Step 4: 홀 배치
    print("\n[Step 4] 홀 배치...")
    random.seed(42)  # 재현성
    
    adder = MillingFeatureAdder(params)
    new_shape, placements = adder.add_milling_features(
        shape,
        target_face_types=["Plane", "Cylinder"],  # 부분 매칭으로 Ring/Disk 모두 포함
        max_total_holes=3,
        holes_per_face=1
    )
    
    if placements:
        visualize_step4_hole_placement(
            shape,  # 원본 형상 사용 (홀 추가 전)
            analysis_results,
            placements,
            str(model_viz_path / "step4_hole_placement.png")
        )
        
        visualize_step4_uv_sampling(
            analysis_results,
            placements,
            str(model_viz_path / "step4_uv_sampling.png")
        )
        
        # 순차적 홀 배치 시각화 (각 홀 직경별 유효 면 포함)
        visualize_step4_sequential_holes(
            analysis_results,
            placements,
            params,
            str(model_viz_path),
            shape  # 3D 시각화용
        )
        
        # STEP 파일 저장
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone
        
        n_holes = len(placements)
        output_step_name = f"{model_name}_milling_H{n_holes}.step"
        output_step_file = step_output_path / output_step_name
        
        writer = STEPControl_Writer()
        writer.Transfer(new_shape, STEPControl_AsIs)
        status = writer.Write(str(output_step_file))
        
        if status == IFSelect_RetDone:
            print(f"  STEP 저장: {output_step_file}")
        else:
            print(f"  STEP 저장 실패")
    else:
        print("  홀이 배치되지 않았습니다.")
        new_shape = shape
    
    print(f"\n{'=' * 60}")
    print(f"완료!")
    print(f"  시각화: {model_viz_path}/")
    if placements:
        print(f"  STEP: {step_output_path}/")
    print(f"{'=' * 60}")
    
    # 요약 출력
    print("\n[요약]")
    print(f"  총 면 수: {len(analysis_results)}")
    milling_faces = [r for r in analysis_results if r.is_valid_for_milling]
    print(f"  밀링 가능 면: {len(milling_faces)}")
    print(f"  배치된 홀 수: {len(placements)}")
    
    for p in placements:
        print(f"    - Face {p.face_id} ({p.face_type}): D={p.diameter:.2f}mm, depth={p.depth:.2f}mm")
    
    return new_shape, placements


# ============================================================================
# Main
# ============================================================================

def process_multiple_models(
    input_dir: str = "generated_turning_models",
    step_output_dir: str = "generated_turning_milling_STEP",
    viz_output_dir: str = "milling_visualization",
    max_models: int = 5
):
    """
    여러 터닝 모델에 밀링 특징형상 추가 및 시각화
    """
    input_path = Path(input_dir)
    step_files = list(input_path.glob("*.step"))
    
    if not step_files:
        print(f"STEP 파일을 찾을 수 없습니다: {input_dir}")
        return
    
    print(f"\n총 {len(step_files)}개 STEP 파일 발견")
    print(f"처리할 모델 수: {min(max_models, len(step_files))}")
    
    params = HoleParams()  # 기본값 사용
    
    results = []
    
    for i, step_file in enumerate(step_files[:max_models]):
        print(f"\n{'#' * 60}")
        print(f"# 모델 {i+1}/{min(max_models, len(step_files))}: {step_file.name}")
        print(f"{'#' * 60}")
        
        try:
            new_shape, placements = visualize_full_process(
                str(step_file),
                step_output_dir=step_output_dir,
                viz_output_dir=viz_output_dir,
                params=params
            )
            
            results.append({
                "file": step_file.name,
                "n_holes": len(placements) if placements else 0,
                "success": True
            })
        except Exception as e:
            print(f"오류 발생: {e}")
            results.append({
                "file": step_file.name,
                "n_holes": 0,
                "success": False,
                "error": str(e)
            })
    
    # 최종 요약
    print(f"\n{'=' * 60}")
    print("전체 처리 완료!")
    print(f"{'=' * 60}")
    print(f"  성공: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"  총 홀 수: {sum(r['n_holes'] for r in results)}")
    print(f"\n  STEP 출력: {step_output_dir}/")
    print(f"  시각화 출력: {viz_output_dir}/")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 단일 파일 처리
        step_file = sys.argv[1]
        visualize_full_process(
            step_file,
            step_output_dir="generated_turning_milling_STEP",
            viz_output_dir="milling_visualization"
        )
    else:
        # 여러 모델 일괄 처리
        process_multiple_models(
            input_dir="generated_turning_models",
            step_output_dir="generated_turning_milling_STEP",
            viz_output_dir="milling_visualization",
            max_models=5
        )
