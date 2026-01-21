"""
터닝 모델 UV 기반 폭/너비 시각화 (pyvista + matplotlib)
- UV 그리드 명시적 시각화 (inside/outside 포인트 표시)
- 대표 라인(중앙) 기준 계산
- 어떤 coarse 샘플이 계산에 사용되었는지 표시
"""
import numpy as np
import pyvista as pv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from uv_face_dimension import (
    load_step_file, extract_faces, compute_face_dimension_simple,
    FaceDimension, CoarseSamplePoint
)


def face_to_pyvista_mesh(face: TopoDS_Face, linear_deflection: float = 0.1) -> pv.PolyData:
    """단일 Face를 PyVista PolyData로 변환"""
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)
    
    if triangulation is None:
        return pv.PolyData()
    
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
        faces.extend([3, n1 - 1, n2 - 1, n3 - 1])
    
    if not vertices:
        return pv.PolyData()
    
    return pv.PolyData(np.array(vertices), np.array(faces))


def visualize_uv_grid_2d(dim: FaceDimension, ax_width, ax_height, face_id: int):
    """
    Face의 UV 그리드를 2D로 시각화.
    - inside: 파란색 점
    - outside: 빨간색 점
    - 경계점: 녹색 삼각형
    - inside 구간: 밝은 배경
    """
    # Width (U 방향) 시각화
    if dim.width_trim_result and dim.width_trim_result.coarse_points:
        coarse_pts = dim.width_trim_result.coarse_points
        v_fixed = dim.width_trim_result.fixed_param
        
        inside_u = [p.param for p in coarse_pts if p.is_inside]
        outside_u = [p.param for p in coarse_pts if not p.is_inside]
        
        ax_width.scatter(inside_u, [1]*len(inside_u), c='blue', s=20, label='Inside', alpha=0.7, marker='o')
        ax_width.scatter(outside_u, [1]*len(outside_u), c='red', s=20, label='Outside', alpha=0.7, marker='x')
        
        # 경계점 표시
        if dim.width_trim_result.boundary_params:
            bp = dim.width_trim_result.boundary_params
            ax_width.scatter(bp, [1]*len(bp), c='green', s=80, marker='^', label='Boundary', zorder=5)
        
        # Inside 구간 배경
        for iv in dim.width_trim_result.intervals:
            ax_width.axvspan(iv.start, iv.end, alpha=0.2, color='blue')
            ax_width.annotate(f'{iv.length_3d:.2f}mm', 
                             xy=((iv.start + iv.end)/2, 1.15), 
                             ha='center', fontsize=8, color='blue')
        
        ax_width.set_xlim(dim.u_min - 0.1*(dim.u_max-dim.u_min), 
                          dim.u_max + 0.1*(dim.u_max-dim.u_min))
        ax_width.set_ylim(0.5, 1.5)
        ax_width.set_xlabel(f'U (v_fixed={v_fixed:.2f})')
        ax_width.set_ylabel('')
        ax_width.set_yticks([])
        ax_width.set_title(f'Width: {dim.width:.2f}mm ({len(coarse_pts)} samples)')
        ax_width.legend(loc='upper right', fontsize=7)
        ax_width.grid(axis='x', alpha=0.3)
    
    # Height (V 방향) 시각화
    if dim.height_trim_result and dim.height_trim_result.coarse_points:
        coarse_pts = dim.height_trim_result.coarse_points
        u_fixed = dim.height_trim_result.fixed_param
        
        inside_v = [p.param for p in coarse_pts if p.is_inside]
        outside_v = [p.param for p in coarse_pts if not p.is_inside]
        
        ax_height.scatter([1]*len(inside_v), inside_v, c='blue', s=20, alpha=0.7, marker='o')
        ax_height.scatter([1]*len(outside_v), outside_v, c='red', s=20, alpha=0.7, marker='x')
        
        # 경계점 표시
        if dim.height_trim_result.boundary_params:
            bp = dim.height_trim_result.boundary_params
            ax_height.scatter([1]*len(bp), bp, c='green', s=80, marker='^', zorder=5)
        
        # Inside 구간 배경
        for iv in dim.height_trim_result.intervals:
            ax_height.axhspan(iv.start, iv.end, alpha=0.2, color='blue')
            ax_height.annotate(f'{iv.length_3d:.2f}mm', 
                              xy=(1.15, (iv.start + iv.end)/2), 
                              va='center', fontsize=8, color='blue')
        
        ax_height.set_ylim(dim.v_min - 0.1*(dim.v_max-dim.v_min), 
                           dim.v_max + 0.1*(dim.v_max-dim.v_min))
        ax_height.set_xlim(0.5, 1.5)
        ax_height.set_ylabel(f'V (u_fixed={u_fixed:.2f})')
        ax_height.set_xlabel('')
        ax_height.set_xticks([])
        ax_height.set_title(f'Height: {dim.height:.2f}mm ({len(coarse_pts)} samples)')
        ax_height.grid(axis='y', alpha=0.3)


def visualize_uv_grid_2d_combined(dim: FaceDimension, ax, face_id: int):
    """
    Face의 UV 그리드를 2D 평면에 combined로 시각화.
    - 중앙 수평선(U 방향 폭 측정)과 중앙 수직선(V 방향 너비 측정)을 그림
    - inside: 파란색, outside: 빨간색
    """
    u_min, u_max = dim.u_min, dim.u_max
    v_min, v_max = dim.v_min, dim.v_max
    
    # UV 박스 그리기
    ax.add_patch(Rectangle((u_min, v_min), u_max-u_min, v_max-v_min, 
                            fill=False, edgecolor='gray', linewidth=2))
    
    # Width 라인 (U 방향, v=v_mid)
    if dim.width_trim_result and dim.width_trim_result.coarse_points:
        v_fixed = dim.width_trim_result.fixed_param
        coarse_pts = dim.width_trim_result.coarse_points
        
        for p in coarse_pts:
            color = 'blue' if p.is_inside else 'red'
            marker = 'o' if p.is_inside else 'x'
            ax.scatter(p.param, v_fixed, c=color, s=15, marker=marker, alpha=0.7)
        
        # 경계점
        if dim.width_trim_result.boundary_params:
            for bp in dim.width_trim_result.boundary_params:
                ax.scatter(bp, v_fixed, c='green', s=60, marker='^', zorder=5)
        
        # Inside 구간 하이라이트
        for iv in dim.width_trim_result.intervals:
            ax.plot([iv.start, iv.end], [v_fixed, v_fixed], 'b-', linewidth=3, alpha=0.7)
    
    # Height 라인 (V 방향, u=u_mid)
    if dim.height_trim_result and dim.height_trim_result.coarse_points:
        u_fixed = dim.height_trim_result.fixed_param
        coarse_pts = dim.height_trim_result.coarse_points
        
        for p in coarse_pts:
            color = 'blue' if p.is_inside else 'red'
            marker = 'o' if p.is_inside else 'x'
            ax.scatter(u_fixed, p.param, c=color, s=15, marker=marker, alpha=0.7)
        
        if dim.height_trim_result.boundary_params:
            for bp in dim.height_trim_result.boundary_params:
                ax.scatter(u_fixed, bp, c='green', s=60, marker='^', zorder=5)
        
        for iv in dim.height_trim_result.intervals:
            ax.plot([u_fixed, u_fixed], [iv.start, iv.end], 'b-', linewidth=3, alpha=0.7)
    
    # 설정
    margin = 0.1 * max(u_max - u_min, v_max - v_min)
    ax.set_xlim(u_min - margin, u_max + margin)
    ax.set_ylim(v_min - margin, v_max + margin)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    inner_flag = " [RING]" if dim.has_inner_trim else ""
    n_w = dim.width_trim_result.coarse_samples if dim.width_trim_result else 0
    n_h = dim.height_trim_result.coarse_samples if dim.height_trim_result else 0
    ax.set_title(f'Face {face_id}: {dim.surface_type}{inner_flag}\n'
                 f'W={dim.width:.2f}mm, H={dim.height:.2f}mm\n'
                 f'Coarse: {n_w}/{n_h} samples', fontsize=9)


def visualize_face_with_grid_3d(face: TopoDS_Face, dim: FaceDimension, 
                                 plotter: pv.Plotter, show_inside_only: bool = True):
    """
    Face의 3D 메쉬와 coarse 그리드 포인트를 시각화.
    """
    mesh = face_to_pyvista_mesh(face)
    if mesh.n_points > 0:
        face_color = 'lightyellow' if dim.has_inner_trim else 'lightblue'
        plotter.add_mesh(mesh, color=face_color, show_edges=True,
                        edge_color='gray', line_width=0.5, opacity=0.7)
    
    # Width 라인의 coarse 포인트 (3D)
    if dim.width_trim_result and dim.width_trim_result.coarse_points:
        inside_pts = np.array([p.pt_3d for p in dim.width_trim_result.coarse_points if p.is_inside])
        outside_pts = np.array([p.pt_3d for p in dim.width_trim_result.coarse_points if not p.is_inside])
        
        if len(inside_pts) > 0:
            plotter.add_points(inside_pts, color='blue', point_size=8, 
                              render_points_as_spheres=True, label='Inside (U)')
        if len(outside_pts) > 0 and not show_inside_only:
            plotter.add_points(outside_pts, color='red', point_size=6,
                              render_points_as_spheres=True, label='Outside (U)')
        
        # Inside 구간 라인
        adaptor = BRepAdaptor_Surface(face, True)
        v_fixed = dim.width_trim_result.fixed_param
        for iv in dim.width_trim_result.intervals:
            line_pts = []
            for j in range(50):
                u = iv.start + (iv.end - iv.start) * j / 49
                pt = adaptor.Value(u, v_fixed)
                line_pts.append([pt.X(), pt.Y(), pt.Z()])
            if len(line_pts) > 1:
                line = pv.lines_from_points(np.array(line_pts))
                plotter.add_mesh(line, color='red', line_width=4)
    
    # Height 라인의 coarse 포인트 (3D)
    if dim.height_trim_result and dim.height_trim_result.coarse_points:
        inside_pts = np.array([p.pt_3d for p in dim.height_trim_result.coarse_points if p.is_inside])
        outside_pts = np.array([p.pt_3d for p in dim.height_trim_result.coarse_points if not p.is_inside])
        
        if len(inside_pts) > 0:
            plotter.add_points(inside_pts, color='cyan', point_size=8,
                              render_points_as_spheres=True, label='Inside (V)')
        if len(outside_pts) > 0 and not show_inside_only:
            plotter.add_points(outside_pts, color='orange', point_size=6,
                              render_points_as_spheres=True, label='Outside (V)')
        
        # Inside 구간 라인
        adaptor = BRepAdaptor_Surface(face, True)
        u_fixed = dim.height_trim_result.fixed_param
        for iv in dim.height_trim_result.intervals:
            line_pts = []
            for j in range(50):
                v = iv.start + (iv.end - iv.start) * j / 49
                pt = adaptor.Value(u_fixed, v)
                line_pts.append([pt.X(), pt.Y(), pt.Z()])
            if len(line_pts) > 1:
                line = pv.lines_from_points(np.array(line_pts))
                plotter.add_mesh(line, color='green', line_width=4)


def visualize_single_face_detailed(filepath: str, face_idx: int = 0,
                                     save_path: Optional[str] = None):
    """
    단일 Face의 UV 그리드 상세 시각화.
    """
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    faces = extract_faces(shape)
    if face_idx >= len(faces):
        print(f"Face index {face_idx} out of range (total: {len(faces)})")
        return
    
    face = faces[face_idx]
    dim = compute_face_dimension_simple(face, face_idx)
    
    # Figure 생성 (2행 2열)
    fig = plt.figure(figsize=(14, 10))
    
    # 상단 왼쪽: UV 평면 combined 뷰
    ax1 = fig.add_subplot(2, 2, 1)
    visualize_uv_grid_2d_combined(dim, ax1, face_idx)
    
    # 상단 오른쪽: Width (U 방향) 1D 뷰
    ax2 = fig.add_subplot(2, 2, 2)
    
    # 하단 왼쪽: Height (V 방향) 1D 뷰
    ax3 = fig.add_subplot(2, 2, 3)
    
    visualize_uv_grid_2d(dim, ax2, ax3, face_idx)
    
    # 하단 오른쪽: 정보 테이블
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # 정보 텍스트 생성
    v_fixed_str = f"{dim.width_trim_result.fixed_param:.4f}" if dim.width_trim_result else "N/A"
    u_fixed_str = f"{dim.height_trim_result.fixed_param:.4f}" if dim.height_trim_result else "N/A"
    coarse_w = dim.width_trim_result.coarse_samples if dim.width_trim_result else 0
    coarse_h = dim.height_trim_result.coarse_samples if dim.height_trim_result else 0
    ivl_w = dim.width_trim_result.num_intervals if dim.width_trim_result else 0
    ivl_h = dim.height_trim_result.num_intervals if dim.height_trim_result else 0
    ring_str = "Yes" if dim.has_inner_trim else "No"
    
    info_text = f"""Face {face_idx}: {dim.surface_type}
========================================

UV 범위:
  U: [{dim.u_min:.4f}, {dim.u_max:.4f}]
  V: [{dim.v_min:.4f}, {dim.v_max:.4f}]

폭 (Width) 계산:
  대표 라인: v = {v_fixed_str}
  Coarse 샘플 수: {coarse_w}
  Inside 구간 수: {ivl_w}
  Max Segment: {dim.width:.4f} mm

너비 (Height) 계산:
  대표 라인: u = {u_fixed_str}
  Coarse 샘플 수: {coarse_h}
  Inside 구간 수: {ivl_h}
  Max Segment: {dim.height:.4f} mm

내부 구멍 (RING): {ring_str}
종횡비: {dim.aspect_ratio:.2f}
"""
    ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'UV Grid Visualization: {Path(filepath).name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiple_faces_uv_grid(filepath: str, max_faces: int = 6,
                                       save_path: Optional[str] = None):
    """
    여러 Face의 UV 그리드 시각화 (matplotlib 2D).
    """
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    faces = extract_faces(shape)
    
    # 유효한 Face 선택 (RING 우선)
    face_data = []
    for i, face in enumerate(faces):
        dim = compute_face_dimension_simple(face, i)
        if not dim.is_sliver:
            face_data.append((i, face, dim))
    
    # RING 면 우선, 그 다음 다양한 타입
    ring_faces = [(i, f, d) for i, f, d in face_data if d.has_inner_trim]
    other_faces = [(i, f, d) for i, f, d in face_data if not d.has_inner_trim]
    
    selected = ring_faces[:2] + other_faces[:max_faces - len(ring_faces[:2])]
    selected = selected[:max_faces]
    
    n_faces = len(selected)
    if n_faces == 0:
        print("표시할 Face가 없습니다.")
        return
    
    cols = min(3, n_faces)
    rows = (n_faces + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_faces == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    
    for idx, (face_id, face, dim) in enumerate(selected):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[0][col]
        
        visualize_uv_grid_2d_combined(dim, ax, face_id)
    
    # 빈 subplot 숨기기
    for idx in range(n_faces, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[0][col]
        ax.axis('off')
    
    plt.suptitle(f'UV Grid Visualization (Inside/Outside)\n{Path(filepath).name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_with_grid(filepath: str, max_faces: int = 4,
                            save_path: Optional[str] = None):
    """
    3D 모델과 UV 그리드 포인트 시각화 (pyvista).
    """
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    faces = extract_faces(shape)
    
    # 유효한 Face 선택
    face_data = []
    for i, face in enumerate(faces):
        dim = compute_face_dimension_simple(face, i)
        if not dim.is_sliver:
            face_data.append((i, face, dim))
    
    # RING 면 + 다양한 타입
    ring_faces = [(i, f, d) for i, f, d in face_data if d.has_inner_trim]
    other_faces = [(i, f, d) for i, f, d in face_data if not d.has_inner_trim]
    
    selected = ring_faces[:2] + other_faces[:max_faces - len(ring_faces[:2])]
    selected = selected[:max_faces]
    
    n_faces = len(selected)
    if n_faces == 0:
        print("표시할 Face가 없습니다.")
        return
    
    cols = min(2, n_faces)
    rows = (n_faces + cols - 1) // cols
    
    plotter = pv.Plotter(shape=(rows, cols), off_screen=(save_path is not None))
    plotter.set_background('white')
    
    for idx, (face_id, face, dim) in enumerate(selected):
        row = idx // cols
        col = idx % cols
        plotter.subplot(row, col)
        
        visualize_face_with_grid_3d(face, dim, plotter, show_inside_only=False)
        
        inner_flag = " [RING]" if dim.has_inner_trim else ""
        n_w = dim.width_trim_result.coarse_samples if dim.width_trim_result else 0
        plotter.add_title(f"F{face_id}: {dim.surface_type}{inner_flag}\n"
                          f"W={dim.width:.2f}mm, H={dim.height:.2f}mm\n"
                          f"Coarse: {n_w} samples", font_size=9)
        plotter.view_isometric()
    
    if save_path:
        plotter.screenshot(save_path)
        print(f"저장: {save_path}")
    else:
        plotter.show()
    
    plotter.close()


def print_dimension_summary(filepath: str):
    """치수 분석 요약 출력."""
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    faces = extract_faces(shape)
    
    print(f"\n{'='*100}")
    print(f"파일: {filepath}")
    print(f"총 {len(faces)}개 Face")
    print(f"{'='*100}")
    
    print(f"\n{'Face':>5} | {'Type':^12} | {'RING':^4} | {'Width(mm)':>10} | {'Height(mm)':>10} | "
          f"{'Coarse W':>9} | {'Coarse H':>9} | {'Ivl W':>5} | {'Ivl H':>5}")
    print("-" * 100)
    
    for i, face in enumerate(faces):
        dim = compute_face_dimension_simple(face, i)
        if dim.is_sliver:
            continue
        
        ring = "Yes" if dim.has_inner_trim else ""
        n_w = dim.width_trim_result.coarse_samples if dim.width_trim_result else 0
        n_h = dim.height_trim_result.coarse_samples if dim.height_trim_result else 0
        ivl_w = dim.width_trim_result.num_intervals if dim.width_trim_result else 0
        ivl_h = dim.height_trim_result.num_intervals if dim.height_trim_result else 0
        
        print(f"{i:>5} | {dim.surface_type:^12} | {ring:^4} | {dim.width:>10.3f} | {dim.height:>10.3f} | "
              f"{n_w:>9} | {n_h:>9} | {ivl_w:>5} | {ivl_h:>5}")
    
    print("-" * 100)


def main():
    """메인 함수"""
    model_dir = Path("generated_turning_models")
    step_files = sorted(model_dir.glob("*.step"))
    
    if not step_files:
        print("생성된 STEP 파일이 없습니다.")
        return
    
    print(f"총 {len(step_files)}개 모델 발견")
    
    sample_file = step_files[2] if len(step_files) > 2 else step_files[0]
    
    # 1. 치수 요약 출력
    print("\n" + "="*60)
    print("1. 치수 분석 요약")
    print("="*60)
    print_dimension_summary(str(sample_file))
    
    # 2. 여러 Face UV 그리드 시각화 (2D)
    print("\n" + "="*60)
    print("2. UV 그리드 시각화 (2D)")
    print("="*60)
    visualize_multiple_faces_uv_grid(
        str(sample_file),
        max_faces=6,
        save_path='uv_dimension_faces.png'
    )
    
    # 3. 단일 Face 상세 시각화
    print("\n" + "="*60)
    print("3. 단일 Face 상세 시각화")
    print("="*60)
    
    # RING 면이 있으면 해당 Face, 없으면 첫 번째 유효 Face
    shape = load_step_file(str(sample_file))
    if shape:
        faces = extract_faces(shape)
        ring_idx = None
        first_valid_idx = None
        
        for i, face in enumerate(faces):
            dim = compute_face_dimension_simple(face, i)
            if not dim.is_sliver:
                if first_valid_idx is None:
                    first_valid_idx = i
                if dim.has_inner_trim and ring_idx is None:
                    ring_idx = i
        
        target_idx = ring_idx if ring_idx is not None else (first_valid_idx if first_valid_idx is not None else 0)
        visualize_single_face_detailed(
            str(sample_file),
            face_idx=target_idx,
            save_path='uv_dimension_single.png'
        )
    
    # 4. 3D 모델과 그리드 시각화
    print("\n" + "="*60)
    print("4. 3D 모델 + UV 그리드 시각화")
    print("="*60)
    visualize_3d_with_grid(
        str(sample_file),
        max_faces=4,
        save_path='uv_dimension_summary.png'
    )
    
    print("\n" + "="*60)
    print("시각화 완료!")
    print("  - uv_dimension_faces.png: 여러 Face UV 그리드 (2D)")
    print("  - uv_dimension_single.png: 단일 Face 상세 분석")
    print("  - uv_dimension_summary.png: 3D 모델 + 그리드 포인트")
    print("="*60)


if __name__ == "__main__":
    main()
