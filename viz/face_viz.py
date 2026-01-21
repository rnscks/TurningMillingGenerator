"""
Face 치수 시각화 모듈

각 Face의 원본 3D 형상, 투영 결과, 치수 측정을 시각화.
"""

import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.face_analyzer import (
    FaceAnalyzer, FaceDimensionResult,
    get_face_wires, sample_wire_points, points_to_rz, points_to_xy
)
from utils.step_io import load_step


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_face_dimensions(
    shape: TopoDS_Shape,
    output_dir: str = "face_viz_results",
    min_dimension: float = 0.1
):
    """
    형상의 모든 유효 Face를 개별 이미지로 저장.
    
    Args:
        shape: 분석할 형상
        output_dir: 출력 디렉토리
        min_dimension: 최소 치수 (이하는 제외)
    """
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    analyzer = FaceAnalyzer()
    results = analyzer.analyze_shape(shape)
    
    # 전체 범위 계산
    global_bounds = _compute_global_bounds(faces)
    
    # 출력 폴더 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFace 시각화 저장 중: {output_path}")
    
    saved_files = []
    for i, (face, result) in enumerate(zip(faces, results)):
        if result.width is None or result.width < min_dimension:
            continue
        
        type_short = result.surface_type.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"Face_{i:02d}_{type_short}.png"
        
        filepath = visualize_single_face(
            face, i, result, str(output_path), filename,
            all_faces=faces, global_bounds=global_bounds
        )
        saved_files.append(filepath)
        print(f"  저장: {filename}")
    
    print(f"\n총 {len(saved_files)}개 Face 이미지 저장됨")
    return saved_files


def visualize_single_face(
    face: TopoDS_Face,
    face_id: int,
    result: FaceDimensionResult,
    output_dir: str,
    filename: str,
    all_faces: List[TopoDS_Face] = None,
    global_bounds: Dict = None
) -> str:
    """
    단일 Face의 원본 3D + 투영 + 치수 측정 시각화.
    
    Args:
        face: 시각화할 Face
        face_id: Face ID
        result: 치수 분석 결과
        output_dir: 출력 디렉토리
        filename: 파일명
        all_faces: 전체 형상의 모든 face (전체 맥락 표시용)
        global_bounds: 전체 형상의 범위 (공통 스케일용)
        
    Returns:
        저장된 파일 경로
    """
    is_plane_face = "Plane" in result.surface_type
    is_rotational_face = result.surface_type in ["Cylinder", "Cone", "Torus"]
    
    fig = plt.figure(figsize=(12, 5))
    n_cols = 2
    
    # ========== 1열: 전체 형상에서 해당 Face 위치 ==========
    ax3d = fig.add_subplot(1, n_cols, 1, projection='3d')
    location = TopLoc_Location()
    
    # 전체 형상을 회색으로 먼저 그리기
    if all_faces is not None:
        for other_face in all_faces:
            other_tri = BRep_Tool.Triangulation(other_face, location)
            if other_tri is not None:
                verts, tris = _extract_triangulation(other_tri, location)
                if len(verts) > 0 and len(tris) > 0:
                    ax3d.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                      triangles=tris, color='lightgray',
                                      edgecolor='none', linewidth=0, alpha=0.3)
    
    # 현재 Face 강조
    triangulation = BRep_Tool.Triangulation(face, location)
    if triangulation is not None:
        vertices, triangles = _extract_triangulation(triangulation, location)
        if len(vertices) > 0 and len(triangles) > 0:
            face_color = _get_face_color(result)
            ax3d.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                              triangles=triangles, color=face_color,
                              edgecolor='darkgray', linewidth=0.3, alpha=0.9)
    
    # Wire 경계 표시
    outer_wire, inner_wires = get_face_wires(face)
    outer_pts = []
    if outer_wire:
        outer_pts = sample_wire_points(outer_wire, 100)
        xs = [p.X() for p in outer_pts]
        ys = [p.Y() for p in outer_pts]
        zs = [p.Z() for p in outer_pts]
        ax3d.plot(xs, ys, zs, 'b-', linewidth=2.5, label='Outer')
    
    inner_pts_all = []
    for iw in inner_wires:
        inner_pts = sample_wire_points(iw, 100)
        inner_pts_all.extend(inner_pts)
        xs = [p.X() for p in inner_pts]
        ys = [p.Y() for p in inner_pts]
        zs = [p.Z() for p in inner_pts]
        ax3d.plot(xs, ys, zs, 'r-', linewidth=2.5, label='Inner')
    
    if global_bounds:
        ax3d.set_xlim(global_bounds['x_min'], global_bounds['x_max'])
        ax3d.set_ylim(global_bounds['y_min'], global_bounds['y_max'])
        ax3d.set_zlim(global_bounds['z_min'], global_bounds['z_max'])
    
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ring_flag = " [RING]" if result.is_ring else ""
    ax3d.set_title(f'3D Context\nF{face_id}: {result.surface_type}{ring_flag}', fontsize=10)
    if outer_pts or inner_pts_all:
        ax3d.legend(fontsize=8, loc='upper right')
    
    # ========== 2열: 투영 + 치수 ==========
    if is_rotational_face:
        _draw_rz_projection(fig, n_cols, 2, face, result, outer_pts, inner_pts_all, 
                           triangulation, location, all_faces, global_bounds)
    else:
        _draw_xy_projection(fig, n_cols, 2, result, outer_pts, inner_pts_all,
                           inner_wires, all_faces, global_bounds)
    
    # 전체 타이틀
    w_str = f"{result.width:.2f}" if result.width else "N/A"
    h_str = f"{result.height:.2f}" if result.height else "N/A"
    plt.suptitle(f'Face {face_id}: {result.surface_type}{ring_flag}\n'
                 f'Width = {w_str}mm, Height = {h_str}mm', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_global_bounds(faces: List[TopoDS_Face]) -> Optional[Dict]:
    """전체 형상의 범위 계산."""
    all_pts = []
    for face in faces:
        outer_wire, _ = get_face_wires(face)
        if outer_wire:
            pts = sample_wire_points(outer_wire, 50)
            all_pts.extend(pts)
    
    if not all_pts:
        return None
    
    xs = [p.X() for p in all_pts]
    ys = [p.Y() for p in all_pts]
    zs = [p.Z() for p in all_pts]
    rs = [np.sqrt(p.X()**2 + p.Y()**2) for p in all_pts]
    
    margin = 0.5
    return {
        'x_min': min(xs) - margin,
        'x_max': max(xs) + margin,
        'y_min': min(ys) - margin,
        'y_max': max(ys) + margin,
        'z_min': min(zs) - margin,
        'z_max': max(zs) + margin,
        'r_max': max(rs) + margin
    }


def _extract_triangulation(triangulation, location):
    """삼각화에서 정점과 삼각형 추출."""
    vertices = []
    for i in range(1, triangulation.NbNodes() + 1):
        node = triangulation.Node(i)
        if not location.IsIdentity():
            node = node.Transformed(location.Transformation())
        vertices.append([node.X(), node.Y(), node.Z()])
    
    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        triangles.append([n1 - 1, n2 - 1, n3 - 1])
    
    return np.array(vertices), np.array(triangles) if triangles else np.array([])


def _get_face_color(result: FaceDimensionResult) -> str:
    """면 타입별 색상 반환."""
    if result.is_ring:
        return 'orange'
    elif result.surface_type == "Cylinder":
        return 'dodgerblue'
    elif result.surface_type == "Cone":
        return 'limegreen'
    elif result.surface_type == "Torus":
        return 'coral'
    else:
        return 'gold'


def _draw_rz_projection(fig, n_cols, col_idx, face, result, outer_pts, inner_pts_all,
                        triangulation, location, all_faces, global_bounds):
    """R-Z 투영 그리기 (회전면용)."""
    ax_rz = fig.add_subplot(1, n_cols, col_idx)
    
    # 전체 형상의 R-Z 단면
    if all_faces is not None:
        for other_face in all_faces:
            other_tri = BRep_Tool.Triangulation(other_face, location)
            if other_tri is not None:
                for i in range(1, other_tri.NbTriangles() + 1):
                    tri = other_tri.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    pts_3d = [other_tri.Node(n1), other_tri.Node(n2), other_tri.Node(n3)]
                    rs = [np.sqrt(p.X()**2 + p.Y()**2) for p in pts_3d]
                    zs = [p.Z() for p in pts_3d]
                    ax_rz.fill(zs, rs, color='lightgray', alpha=0.15, edgecolor='none')
    
    # 현재 면의 R-Z 투영
    face_color = _get_face_color(result)
    if triangulation is not None:
        for i in range(1, triangulation.NbTriangles() + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            pts_3d = [triangulation.Node(n1), triangulation.Node(n2), triangulation.Node(n3)]
            if not location.IsIdentity():
                pts_3d = [p.Transformed(location.Transformation()) for p in pts_3d]
            rs = [np.sqrt(p.X()**2 + p.Y()**2) for p in pts_3d]
            zs = [p.Z() for p in pts_3d]
            ax_rz.fill(zs, rs, color=face_color, alpha=0.6, edgecolor='darkgray', linewidth=0.3)
    
    # 외경/내경 경계선
    if outer_pts:
        outer_r, outer_z = points_to_rz(outer_pts)
        ax_rz.plot(outer_z, outer_r, 'b-', linewidth=2.5, label='Outer')
    
    if inner_pts_all:
        inner_r, inner_z = points_to_rz(inner_pts_all)
        ax_rz.plot(inner_z, inner_r, 'r-', linewidth=2.5, label='Inner')
    
    ax_rz.set_xlabel('Z (axial)')
    ax_rz.set_ylabel('R (radial)')
    ax_rz.set_title('R-Z Projection (Side View)', fontsize=10)
    ax_rz.grid(True, alpha=0.3)
    ax_rz.legend(fontsize=8)
    
    if global_bounds:
        ax_rz.set_xlim(global_bounds['z_min'] - 0.5, global_bounds['z_max'] + 0.5)
        ax_rz.set_ylim(0, global_bounds['r_max'] * 1.1)
    
    # 치수 표시
    if outer_pts:
        outer_r, outer_z = points_to_rz(outer_pts)
        _draw_dimension_arrows_rz(ax_rz, result, outer_r, outer_z)


def _draw_xy_projection(fig, n_cols, col_idx, result, outer_pts, inner_pts_all,
                        inner_wires, all_faces, global_bounds):
    """X-Y 투영 그리기 (평면용)."""
    ax_proj = fig.add_subplot(1, n_cols, col_idx)
    
    # 전체 형상 회색으로
    if all_faces is not None:
        for other_face in all_faces:
            ow, _ = get_face_wires(other_face)
            if ow:
                opts = sample_wire_points(ow, 50)
                ox, oy = points_to_xy(opts)
                ax_proj.scatter(ox, oy, c='lightgray', s=3, alpha=0.3)
    
    if outer_pts:
        outer_x, outer_y = points_to_xy(outer_pts)
        ax_proj.scatter(outer_x, outer_y, c='blue', s=10, alpha=0.9, label='Outer')
        ax_proj.plot(np.append(outer_x, outer_x[0]), np.append(outer_y, outer_y[0]), 
                    'b-', linewidth=1.5, alpha=0.7)
    
    if inner_pts_all:
        inner_x, inner_y = points_to_xy(inner_pts_all)
        ax_proj.scatter(inner_x, inner_y, c='red', s=10, alpha=0.9, label='Inner')
        for iw in inner_wires:
            ipts = sample_wire_points(iw, 100)
            ix, iy = points_to_xy(ipts)
            ax_proj.plot(np.append(ix, ix[0]), np.append(iy, iy[0]), 
                        'r-', linewidth=1.5, alpha=0.7)
    
    ax_proj.set_xlabel('X')
    ax_proj.set_ylabel('Y')
    ax_proj.set_title('X-Y Projection (Top View)', fontsize=10)
    ax_proj.set_aspect('equal')
    ax_proj.grid(True, alpha=0.3)
    ax_proj.legend(fontsize=8)
    ax_proj.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax_proj.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    if global_bounds:
        max_r = max(abs(global_bounds['x_min']), abs(global_bounds['x_max']),
                   abs(global_bounds['y_min']), abs(global_bounds['y_max']))
        ax_proj.set_xlim(-max_r * 1.1, max_r * 1.1)
        ax_proj.set_ylim(-max_r * 1.1, max_r * 1.1)
    
    # 치수 표시
    _draw_dimension_arrows_xy(ax_proj, result, outer_pts, inner_pts_all)


def _draw_dimension_arrows_rz(ax, result, outer_r, outer_z):
    """R-Z 투영에 치수 화살표 그리기."""
    r_min_val = np.min(outer_r)
    r_max_val = np.max(outer_r)
    z_min_val = np.min(outer_z)
    z_max_val = np.max(outer_z)
    z_mid = (z_min_val + z_max_val) / 2
    
    # Height (빨간색 가로 화살표)
    if result.delta_z > 0.01:
        ax.annotate('', xy=(z_max_val, r_max_val + 0.3), 
                   xytext=(z_min_val, r_max_val + 0.3),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(z_mid, r_max_val + 0.5,
               f'H={result.height:.2f}', fontsize=10, color='red',
               ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
    
    # Width (녹색 세로 화살표)
    if result.surface_type == "Cylinder":
        ax.annotate('', xy=(z_min_val - 0.3, r_max_val), 
                   xytext=(z_min_val - 0.3, 0),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(z_min_val - 0.5, r_max_val / 2,
               f'W={result.width:.2f}', fontsize=10, color='green',
               ha='right', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    else:
        ax.annotate('', xy=(z_min_val - 0.3, r_max_val), 
                   xytext=(z_min_val - 0.3, r_min_val),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(z_min_val - 0.5, (r_max_val + r_min_val) / 2,
               f'W={result.width:.2f}', fontsize=10, color='green',
               ha='right', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))


def _draw_dimension_arrows_xy(ax, result, outer_pts, inner_pts_all):
    """X-Y 투영에 치수 화살표 그리기."""
    if result.is_ring and outer_pts and inner_pts_all:
        outer_x, outer_y = points_to_xy(outer_pts)
        inner_x, inner_y = points_to_xy(inner_pts_all)
        r_out = np.max(np.sqrt(outer_x**2 + outer_y**2))
        r_in = np.max(np.sqrt(inner_x**2 + inner_y**2))
        
        ax.annotate('', xy=(r_out, 0), xytext=(r_in, 0),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
        ax.text((r_out + r_in) / 2, 0.5, 
               f'W={result.ring_thickness:.2f}',
               fontsize=10, color='green', ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    elif outer_pts:
        outer_x, outer_y = points_to_xy(outer_pts)
        r_out = np.max(np.sqrt(outer_x**2 + outer_y**2))
        
        ax.annotate('', xy=(r_out, 0), xytext=(-r_out, 0),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
        ax.text(0, r_out * 0.15, f'W=H={result.width:.2f}',
               fontsize=10, color='green', ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
