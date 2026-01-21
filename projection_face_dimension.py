"""
투영 기반 터닝 Face 폭/너비 계산 모듈

터닝 형상 (회전축 = Z)의 특성을 활용하여 간단한 투영으로 유효 치수 계산:
- R-Z 단면 투영: R = sqrt(x^2 + y^2), Z = z
- XY 평면 투영: 평면의 반경 계산용

면 타입별 폭/너비 정의:
- Cylinder: width = 2R (직경), height = Δz (축방향)
- Cone: width = Δr, height = Δz
- Torus/Round: width = Δr, height = Δz
- Circular Plane (디스크): width = height = D_out
- Ring Plane: width = t (두께 = R_out - R_in), height = None
"""
import math
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Extend.TopologyUtils import TopologyExplorer


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FaceDimensionResult:
    """Face 치수 계산 결과"""
    face_id: int
    surface_type: str
    
    # 핵심 치수 (mm)
    width: Optional[float] = None      # 유효 폭
    height: Optional[float] = None     # 유효 너비 (축방향)
    
    # 상세 치수
    r_max: float = 0.0                 # 최대 반경
    r_min: float = 0.0                 # 최소 반경
    z_max: float = 0.0                 # 최대 Z
    z_min: float = 0.0                 # 최소 Z
    delta_r: float = 0.0               # 반경 변화량
    delta_z: float = 0.0               # Z 변화량
    
    # 링 평면 전용
    r_outer: float = 0.0               # 외반경 (outer wire)
    r_inner: float = 0.0               # 내반경 (inner wire)
    ring_thickness: float = 0.0        # 링 두께
    
    # 메타 정보
    is_ring: bool = False              # 링 평면 여부
    is_z_plane: bool = False           # Z 평면 여부 (평면이 XY와 평행)
    n_inner_wires: int = 0             # 내부 wire 수 (구멍 수)
    sample_count: int = 0              # 샘플링된 점 수
    
    @property
    def diameter(self) -> float:
        """직경 (2 * r_max)"""
        return 2 * self.r_max


# ============================================================================
# Wire/Edge Sampling
# ============================================================================

def sample_edge_points(edge: TopoDS_Edge, n_samples: int = 30) -> List[gp_Pnt]:
    """
    Edge에서 균등하게 점 샘플링.
    
    Args:
        edge: TopoDS_Edge
        n_samples: 샘플 수
        
    Returns:
        gp_Pnt 리스트
    """
    adaptor = BRepAdaptor_Curve(edge)
    t_min = adaptor.FirstParameter()
    t_max = adaptor.LastParameter()
    
    points = []
    for i in range(n_samples):
        t = t_min + (t_max - t_min) * i / (n_samples - 1)
        pt = adaptor.Value(t)
        points.append(pt)
    
    return points


def sample_wire_points(wire: TopoDS_Wire, samples_per_edge: int = 30) -> List[gp_Pnt]:
    """
    Wire의 모든 Edge에서 점 샘플링.
    """
    points = []
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        edge_pts = sample_edge_points(edge, samples_per_edge)
        points.extend(edge_pts)
        explorer.Next()
    
    return points


def get_face_wires(face: TopoDS_Face) -> Tuple[Optional[TopoDS_Wire], List[TopoDS_Wire]]:
    """
    Face에서 outer wire와 inner wire들을 분리.
    가장 큰 면적의 wire를 outer로 판단.
    
    Returns:
        (outer_wire, [inner_wires])
    """
    wires = []
    explorer = TopExp_Explorer(face, TopAbs_WIRE)
    
    while explorer.More():
        wire = topods.Wire(explorer.Current())
        wires.append(wire)
        explorer.Next()
    
    if not wires:
        return None, []
    
    if len(wires) == 1:
        return wires[0], []
    
    # Wire 면적으로 outer 판단 (가장 큰 것)
    # 간단히 bounding 크기로 대체
    def wire_extent(w):
        pts = sample_wire_points(w, samples_per_edge=10)
        if not pts:
            return 0
        xs = [p.X() for p in pts]
        ys = [p.Y() for p in pts]
        zs = [p.Z() for p in pts]
        return (max(xs) - min(xs)) * (max(ys) - min(ys)) + (max(zs) - min(zs))
    
    wires_with_extent = [(w, wire_extent(w)) for w in wires]
    wires_with_extent.sort(key=lambda x: x[1], reverse=True)
    
    outer = wires_with_extent[0][0]
    inners = [w for w, _ in wires_with_extent[1:]]
    
    return outer, inners


# ============================================================================
# Point Cloud Projection & Analysis
# ============================================================================

def points_to_rz(points: List[gp_Pnt]) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D 점들을 R-Z 좌표로 변환.
    R = sqrt(x^2 + y^2), Z = z
    
    Returns:
        (R_array, Z_array)
    """
    r_vals = []
    z_vals = []
    
    for pt in points:
        r = math.sqrt(pt.X()**2 + pt.Y()**2)
        z = pt.Z()
        r_vals.append(r)
        z_vals.append(z)
    
    return np.array(r_vals), np.array(z_vals)


def analyze_rz_points(r_vals: np.ndarray, z_vals: np.ndarray) -> Dict:
    """
    R-Z 점들에서 치수 분석.
    """
    if len(r_vals) == 0:
        return {'r_max': 0, 'r_min': 0, 'z_max': 0, 'z_min': 0, 'delta_r': 0, 'delta_z': 0}
    
    return {
        'r_max': float(np.max(r_vals)),
        'r_min': float(np.min(r_vals)),
        'z_max': float(np.max(z_vals)),
        'z_min': float(np.min(z_vals)),
        'delta_r': float(np.max(r_vals) - np.min(r_vals)),
        'delta_z': float(np.max(z_vals) - np.min(z_vals)),
        'r_mean': float(np.mean(r_vals)),
        'z_mean': float(np.mean(z_vals)),
    }


# ============================================================================
# Surface Type Classification
# ============================================================================

def get_surface_type(face: TopoDS_Face) -> Tuple[str, int]:
    """
    Face의 서피스 타입 분류.
    
    Returns:
        (type_name, type_enum)
    """
    adaptor = BRepAdaptor_Surface(face, True)
    surf_type = adaptor.GetType()
    
    type_map = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BSplineSurface: "BSpline",
    }
    
    return type_map.get(surf_type, "Other"), surf_type


def is_z_aligned_plane(face: TopoDS_Face, angle_tol_deg: float = 2.0) -> bool:
    """
    Plane이 Z축에 수직인지 (XY 평면과 평행) 확인.
    """
    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_Plane:
        return False
    
    plane = adaptor.Plane()
    normal = plane.Axis().Direction()
    
    # Z 방향과의 각도
    dot = abs(normal.Z())  # |cos(angle)|
    angle_rad = math.acos(min(1.0, dot))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg < angle_tol_deg


# ============================================================================
# Face Dimension Calculation by Type
# ============================================================================

def compute_cylinder_dimension(face: TopoDS_Face, face_id: int,
                                 samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    원통면 치수 계산.
    - width = 2R (직경)
    - height = Δz (축방향 높이)
    """
    outer_wire, inner_wires = get_face_wires(face)
    
    all_points = []
    if outer_wire:
        all_points.extend(sample_wire_points(outer_wire, samples_per_edge))
    for iw in inner_wires:
        all_points.extend(sample_wire_points(iw, samples_per_edge))
    
    r_vals, z_vals = points_to_rz(all_points)
    stats = analyze_rz_points(r_vals, z_vals)
    
    # 원통: R은 거의 상수, 평균 사용
    radius = stats['r_mean']
    
    return FaceDimensionResult(
        face_id=face_id,
        surface_type="Cylinder",
        width=2 * radius,           # 직경
        height=stats['delta_z'],    # 축방향 높이
        r_max=stats['r_max'],
        r_min=stats['r_min'],
        z_max=stats['z_max'],
        z_min=stats['z_min'],
        delta_r=stats['delta_r'],
        delta_z=stats['delta_z'],
        n_inner_wires=len(inner_wires),
        sample_count=len(all_points),
    )


def compute_cone_dimension(face: TopoDS_Face, face_id: int,
                            samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    원추면 (챔퍼) 치수 계산.
    - width = Δr (반경 변화량)
    - height = Δz (축방향)
    """
    outer_wire, inner_wires = get_face_wires(face)
    
    all_points = []
    if outer_wire:
        all_points.extend(sample_wire_points(outer_wire, samples_per_edge))
    for iw in inner_wires:
        all_points.extend(sample_wire_points(iw, samples_per_edge))
    
    r_vals, z_vals = points_to_rz(all_points)
    stats = analyze_rz_points(r_vals, z_vals)
    
    return FaceDimensionResult(
        face_id=face_id,
        surface_type="Cone",
        width=stats['delta_r'],     # 반경 변화량
        height=stats['delta_z'],    # 축방향
        r_max=stats['r_max'],
        r_min=stats['r_min'],
        z_max=stats['z_max'],
        z_min=stats['z_min'],
        delta_r=stats['delta_r'],
        delta_z=stats['delta_z'],
        n_inner_wires=len(inner_wires),
        sample_count=len(all_points),
    )


def compute_torus_dimension(face: TopoDS_Face, face_id: int,
                             samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    토러스 (라운드/필렛) 치수 계산.
    - width = Δr (반경 방향 두께)
    - height = Δz (축방향 두께)
    """
    outer_wire, inner_wires = get_face_wires(face)
    
    all_points = []
    if outer_wire:
        all_points.extend(sample_wire_points(outer_wire, samples_per_edge))
    for iw in inner_wires:
        all_points.extend(sample_wire_points(iw, samples_per_edge))
    
    r_vals, z_vals = points_to_rz(all_points)
    stats = analyze_rz_points(r_vals, z_vals)
    
    return FaceDimensionResult(
        face_id=face_id,
        surface_type="Torus",
        width=stats['delta_r'],     # 반경 방향 두께
        height=stats['delta_z'],    # 축방향 두께
        r_max=stats['r_max'],
        r_min=stats['r_min'],
        z_max=stats['z_max'],
        z_min=stats['z_min'],
        delta_r=stats['delta_r'],
        delta_z=stats['delta_z'],
        n_inner_wires=len(inner_wires),
        sample_count=len(all_points),
    )


def compute_plane_dimension(face: TopoDS_Face, face_id: int,
                             samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    평면 치수 계산 (디스크 또는 링).
    
    - 디스크 (inner wire 없음): width = height = D_out
    - 링 (inner wire 있음): width = t (두께), height = None
    """
    outer_wire, inner_wires = get_face_wires(face)
    is_ring = len(inner_wires) > 0
    is_z_plane = is_z_aligned_plane(face)
    
    # Outer wire 점 샘플링
    outer_points = []
    if outer_wire:
        outer_points = sample_wire_points(outer_wire, samples_per_edge)
    
    # Inner wire 점 샘플링 (링인 경우)
    inner_points = []
    for iw in inner_wires:
        inner_points.extend(sample_wire_points(iw, samples_per_edge))
    
    # R-Z 변환
    outer_r, outer_z = points_to_rz(outer_points)
    inner_r, inner_z = points_to_rz(inner_points)
    
    # Outer 분석
    outer_stats = analyze_rz_points(outer_r, outer_z)
    r_outer = outer_stats['r_max']
    
    # Inner 분석 (링인 경우)
    r_inner = 0.0
    if is_ring and len(inner_r) > 0:
        # Inner wire는 구멍이므로 r 최대값이 구멍 반경
        r_inner = float(np.max(inner_r))
    
    # 전체 점으로 통합 분석
    all_r = np.concatenate([outer_r, inner_r]) if len(inner_r) > 0 else outer_r
    all_z = np.concatenate([outer_z, inner_z]) if len(inner_z) > 0 else outer_z
    all_stats = analyze_rz_points(all_r, all_z)
    
    # 치수 계산
    if is_ring:
        # 링 평면: 두께만 유효
        ring_thickness = r_outer - r_inner
        return FaceDimensionResult(
            face_id=face_id,
            surface_type="Plane (Ring)",
            width=ring_thickness,       # 링 두께
            height=None,                # 축방향 의미 없음
            r_max=all_stats['r_max'],
            r_min=all_stats['r_min'],
            z_max=all_stats['z_max'],
            z_min=all_stats['z_min'],
            delta_r=all_stats['delta_r'],
            delta_z=all_stats['delta_z'],
            r_outer=r_outer,
            r_inner=r_inner,
            ring_thickness=ring_thickness,
            is_ring=True,
            is_z_plane=is_z_plane,
            n_inner_wires=len(inner_wires),
            sample_count=len(outer_points) + len(inner_points),
        )
    else:
        # 디스크: 직경
        diameter = 2 * r_outer
        return FaceDimensionResult(
            face_id=face_id,
            surface_type="Plane (Disk)",
            width=diameter,             # 직경
            height=diameter,            # 원형이므로 동일
            r_max=all_stats['r_max'],
            r_min=all_stats['r_min'],
            z_max=all_stats['z_max'],
            z_min=all_stats['z_min'],
            delta_r=all_stats['delta_r'],
            delta_z=all_stats['delta_z'],
            r_outer=r_outer,
            is_ring=False,
            is_z_plane=is_z_plane,
            n_inner_wires=0,
            sample_count=len(outer_points),
        )


def compute_bspline_dimension(face: TopoDS_Face, face_id: int,
                               samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    BSpline 서피스 치수 계산 (블렌드/라운드 등).
    Torus와 유사하게 Δr, Δz 사용.
    """
    outer_wire, inner_wires = get_face_wires(face)
    
    all_points = []
    if outer_wire:
        all_points.extend(sample_wire_points(outer_wire, samples_per_edge))
    for iw in inner_wires:
        all_points.extend(sample_wire_points(iw, samples_per_edge))
    
    r_vals, z_vals = points_to_rz(all_points)
    stats = analyze_rz_points(r_vals, z_vals)
    
    return FaceDimensionResult(
        face_id=face_id,
        surface_type="BSpline",
        width=stats['delta_r'],
        height=stats['delta_z'],
        r_max=stats['r_max'],
        r_min=stats['r_min'],
        z_max=stats['z_max'],
        z_min=stats['z_min'],
        delta_r=stats['delta_r'],
        delta_z=stats['delta_z'],
        n_inner_wires=len(inner_wires),
        sample_count=len(all_points),
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def compute_face_dimension(face: TopoDS_Face, face_id: int,
                            samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    Face의 투영 기반 치수 계산 (타입별 분기).
    
    Args:
        face: TopoDS_Face
        face_id: Face 식별자
        samples_per_edge: Edge당 샘플 수
        
    Returns:
        FaceDimensionResult
    """
    surf_type, type_enum = get_surface_type(face)
    
    if type_enum == GeomAbs_Cylinder:
        return compute_cylinder_dimension(face, face_id, samples_per_edge)
    elif type_enum == GeomAbs_Cone:
        return compute_cone_dimension(face, face_id, samples_per_edge)
    elif type_enum == GeomAbs_Torus:
        return compute_torus_dimension(face, face_id, samples_per_edge)
    elif type_enum == GeomAbs_Plane:
        return compute_plane_dimension(face, face_id, samples_per_edge)
    elif type_enum == GeomAbs_BSplineSurface:
        return compute_bspline_dimension(face, face_id, samples_per_edge)
    else:
        # 기타: 일반적인 R-Z 분석
        outer_wire, inner_wires = get_face_wires(face)
        all_points = []
        if outer_wire:
            all_points.extend(sample_wire_points(outer_wire, samples_per_edge))
        for iw in inner_wires:
            all_points.extend(sample_wire_points(iw, samples_per_edge))
        
        r_vals, z_vals = points_to_rz(all_points)
        stats = analyze_rz_points(r_vals, z_vals)
        
        return FaceDimensionResult(
            face_id=face_id,
            surface_type=surf_type,
            width=stats['delta_r'],
            height=stats['delta_z'],
            r_max=stats['r_max'],
            r_min=stats['r_min'],
            z_max=stats['z_max'],
            z_min=stats['z_min'],
            delta_r=stats['delta_r'],
            delta_z=stats['delta_z'],
            n_inner_wires=len(inner_wires),
            sample_count=len(all_points),
        )


# ============================================================================
# File I/O
# ============================================================================

def load_step_file(filepath: str) -> Optional[TopoDS_Shape]:
    """STEP 파일 로드"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    
    if status == IFSelect_RetDone:
        reader.TransferRoots()
        return reader.OneShape()
    else:
        print(f"STEP 파일 로드 실패: {filepath}")
        return None


def extract_faces(shape: TopoDS_Shape) -> List[TopoDS_Face]:
    """Shape에서 모든 Face 추출"""
    topo = TopologyExplorer(shape)
    return list(topo.faces())


# ============================================================================
# Analysis & Output
# ============================================================================

def analyze_turning_model(filepath: str, 
                           samples_per_edge: int = 30,
                           min_dimension: float = 0.1) -> List[FaceDimensionResult]:
    """
    터닝 모델의 모든 Face 치수 분석.
    """
    shape = load_step_file(filepath)
    if shape is None:
        return []
    
    faces = extract_faces(shape)
    print(f"총 {len(faces)}개 Face 발견")
    
    results = []
    for i, face in enumerate(faces):
        result = compute_face_dimension(face, i, samples_per_edge)
        results.append(result)
        
        # 유효 치수만 출력
        w_str = f"{result.width:.2f}" if result.width is not None else "N/A"
        h_str = f"{result.height:.2f}" if result.height is not None else "N/A"
        ring_str = " [RING]" if result.is_ring else ""
        
        if result.width is not None and result.width > min_dimension:
            print(f"  Face {i}: {result.surface_type}{ring_str}, "
                  f"width={w_str}mm, height={h_str}mm")
    
    return results


def print_summary_table(results: List[FaceDimensionResult]):
    """결과 요약 테이블 출력"""
    print("\n" + "=" * 120)
    print(f"{'Face':>5} | {'Type':^15} | {'Width(mm)':>10} | {'Height(mm)':>10} | "
          f"{'R_max':>8} | {'R_min':>8} | {'ΔR':>8} | {'ΔZ':>8} | {'Ring':>4}")
    print("-" * 120)
    
    for r in results:
        # 작은 face 스킵
        if r.width is not None and r.width < 0.1:
            continue
            
        w_str = f"{r.width:.3f}" if r.width is not None else "N/A"
        h_str = f"{r.height:.3f}" if r.height is not None else "N/A"
        ring_str = "Yes" if r.is_ring else ""
        
        print(f"{r.face_id:>5} | {r.surface_type:^15} | {w_str:>10} | {h_str:>10} | "
              f"{r.r_max:>8.2f} | {r.r_min:>8.2f} | {r.delta_r:>8.2f} | {r.delta_z:>8.2f} | {ring_str:>4}")
    
    print("=" * 120)


def print_detailed_results(results: List[FaceDimensionResult]):
    """상세 결과 출력"""
    print("\n" + "=" * 80)
    print("투영 기반 Face 치수 분석 결과 (회전축 = Z)")
    print("=" * 80)
    
    for r in results:
        if r.width is not None and r.width < 0.1:
            continue
            
        print(f"\n{'─' * 60}")
        print(f"Face {r.face_id}: {r.surface_type}")
        print(f"{'─' * 60}")
        
        w_str = f"{r.width:.4f} mm" if r.width is not None else "N/A"
        h_str = f"{r.height:.4f} mm" if r.height is not None else "N/A"
        
        print(f"  유효 폭 (Width):  {w_str}")
        print(f"  유효 너비 (Height): {h_str}")
        
        print(f"\n  R-Z 투영 분석:")
        print(f"    R 범위: [{r.r_min:.4f}, {r.r_max:.4f}] (ΔR = {r.delta_r:.4f})")
        print(f"    Z 범위: [{r.z_min:.4f}, {r.z_max:.4f}] (ΔZ = {r.delta_z:.4f})")
        print(f"    직경: {r.diameter:.4f}")
        
        if r.is_ring:
            print(f"\n  링 정보:")
            print(f"    외반경 (R_outer): {r.r_outer:.4f}")
            print(f"    내반경 (R_inner): {r.r_inner:.4f}")
            print(f"    링 두께: {r.ring_thickness:.4f}")
        
        print(f"\n  메타: {r.sample_count}개 샘플, {r.n_inner_wires}개 내부 wire")
    
    print("\n" + "=" * 80)


# ============================================================================
# Visualization
# ============================================================================

def visualize_rz_projection(filepath: str, save_path: Optional[str] = None):
    """
    R-Z 투영 시각화 (matplotlib).
    각 Face의 경계점을 R-Z 평면에 표시.
    """
    import matplotlib.pyplot as plt
    
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    faces = extract_faces(shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: 전체 R-Z 투영
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(faces)))
    
    for i, face in enumerate(faces):
        result = compute_face_dimension(face, i)
        if result.width is not None and result.width < 0.1:
            continue
        
        outer_wire, inner_wires = get_face_wires(face)
        
        # Outer wire
        if outer_wire:
            pts = sample_wire_points(outer_wire, 50)
            r_vals, z_vals = points_to_rz(pts)
            label = f"F{i}: {result.surface_type[:8]}"
            ax1.scatter(z_vals, r_vals, c=[colors[i % 10]], s=5, alpha=0.7, label=label)
        
        # Inner wires
        for iw in inner_wires:
            pts = sample_wire_points(iw, 50)
            r_vals, z_vals = points_to_rz(pts)
            ax1.scatter(z_vals, r_vals, c=[colors[i % 10]], s=5, alpha=0.5, marker='x')
    
    ax1.set_xlabel('Z (axial)')
    ax1.set_ylabel('R (radial)')
    ax1.set_title('R-Z Cross-Section (All Faces)')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 오른쪽: Face별 치수 바 차트
    ax2 = axes[1]
    
    valid_results = [r for r in [compute_face_dimension(f, i) for i, f in enumerate(faces)]
                     if r.width is not None and r.width > 0.1]
    
    if valid_results:
        face_ids = [r.face_id for r in valid_results]
        widths = [r.width if r.width else 0 for r in valid_results]
        heights = [r.height if r.height else 0 for r in valid_results]
        
        x = np.arange(len(face_ids))
        width_bar = 0.35
        
        bar_colors_w = ['orange' if r.is_ring else 'steelblue' for r in valid_results]
        bar_colors_h = ['gold' if r.is_ring else 'coral' for r in valid_results]
        
        ax2.bar(x - width_bar/2, widths, width_bar, color=bar_colors_w, label='Width', edgecolor='black', linewidth=0.5)
        ax2.bar(x + width_bar/2, heights, width_bar, color=bar_colors_h, label='Height', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Face ID')
        ax2.set_ylabel('Dimension (mm)')
        ax2.set_title('Face Dimensions (Orange=Ring)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"F{fid}" for fid in face_ids], rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Projection-Based Face Dimension Analysis\n{Path(filepath).name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_face_dimension_detail(filepath: str, max_faces: int = 6,
                                      save_path: Optional[str] = None):
    """
    각 Face별로 원본 3D 면, R-Z 투영, 치수 계산 과정을 함께 시각화.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool
    
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    faces = extract_faces(shape)
    
    # 유효한 Face만 선택 (다양한 타입 우선)
    face_data = []
    for i, face in enumerate(faces):
        result = compute_face_dimension(face, i)
        if result.width is not None and result.width > 0.1:
            face_data.append((i, face, result))
    
    # 다양한 타입 선택
    type_priority = ["Plane (Ring)", "Cone", "Cylinder", "Torus", "Plane (Disk)"]
    selected = []
    type_seen = set()
    
    for target_type in type_priority:
        for i, f, r in face_data:
            if r.surface_type == target_type and target_type not in type_seen:
                selected.append((i, f, r))
                type_seen.add(target_type)
                break
    
    # 부족하면 추가
    for i, f, r in face_data:
        if len(selected) >= max_faces:
            break
        if (i, f, r) not in selected:
            selected.append((i, f, r))
    
    selected = selected[:max_faces]
    n_faces = len(selected)
    
    if n_faces == 0:
        print("No valid faces to display.")
        return
    
    # 레이아웃: 각 Face당 2개 subplot (3D + R-Z)
    fig = plt.figure(figsize=(14, 4 * n_faces))
    
    for idx, (face_id, face, result) in enumerate(selected):
        # 왼쪽: 3D 원본 면
        ax3d = fig.add_subplot(n_faces, 3, idx * 3 + 1, projection='3d')
        
        # Face 메쉬 추출
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is not None:
            vertices = []
            nb_nodes = triangulation.NbNodes()
            for i in range(1, nb_nodes + 1):
                node = triangulation.Node(i)
                if not location.IsIdentity():
                    node = node.Transformed(location.Transformation())
                vertices.append([node.X(), node.Y(), node.Z()])
            
            triangles = []
            nb_triangles = triangulation.NbTriangles()
            for i in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                triangles.append([n1 - 1, n2 - 1, n3 - 1])
            
            vertices = np.array(vertices)
            triangles = np.array(triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                # Face 색상
                face_color = 'lightyellow' if result.is_ring else 'lightblue'
                ax3d.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                  triangles=triangles, color=face_color,
                                  edgecolor='gray', linewidth=0.2, alpha=0.8)
        
        # Wire 경계 표시
        outer_wire, inner_wires = get_face_wires(face)
        if outer_wire:
            pts = sample_wire_points(outer_wire, 100)
            xs = [p.X() for p in pts]
            ys = [p.Y() for p in pts]
            zs = [p.Z() for p in pts]
            ax3d.plot(xs, ys, zs, 'b-', linewidth=2, label='Outer')
        
        for iw in inner_wires:
            pts = sample_wire_points(iw, 100)
            xs = [p.X() for p in pts]
            ys = [p.Y() for p in pts]
            zs = [p.Z() for p in pts]
            ax3d.plot(xs, ys, zs, 'r-', linewidth=2, label='Inner')
        
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ring_flag = " [RING]" if result.is_ring else ""
        ax3d.set_title(f'F{face_id}: {result.surface_type}{ring_flag}\n(Original 3D)', fontsize=9)
        ax3d.legend(fontsize=7, loc='upper right')
        
        # 중앙: R-Z 투영
        ax_rz = fig.add_subplot(n_faces, 3, idx * 3 + 2)
        
        outer_r, outer_z = [], []
        if outer_wire:
            pts = sample_wire_points(outer_wire, 100)
            outer_r, outer_z = points_to_rz(pts)
        
        inner_r, inner_z = [], []
        for iw in inner_wires:
            pts = sample_wire_points(iw, 100)
            r, z = points_to_rz(pts)
            inner_r.extend(r)
            inner_z.extend(z)
        inner_r, inner_z = np.array(inner_r), np.array(inner_z)
        
        if len(outer_r) > 0:
            ax_rz.scatter(outer_z, outer_r, c='blue', s=15, alpha=0.8, label='Outer')
        if len(inner_r) > 0:
            ax_rz.scatter(inner_z, inner_r, c='red', s=15, alpha=0.8, label='Inner')
        
        ax_rz.set_xlabel('Z (axial)')
        ax_rz.set_ylabel('R (radial)')
        ax_rz.set_title(f'R-Z Projection\nR=sqrt(x²+y²)', fontsize=9)
        ax_rz.grid(True, alpha=0.3)
        ax_rz.legend(fontsize=7)
        ax_rz.set_aspect('equal')
        
        # 오른쪽: 치수 계산 시각화
        ax_dim = fig.add_subplot(n_faces, 3, idx * 3 + 3)
        
        if len(outer_r) > 0:
            ax_dim.scatter(outer_z, outer_r, c='blue', s=8, alpha=0.5)
        if len(inner_r) > 0:
            ax_dim.scatter(inner_z, inner_r, c='red', s=8, alpha=0.5)
        
        all_r = np.concatenate([outer_r, inner_r]) if len(inner_r) > 0 else np.array(outer_r)
        all_z = np.concatenate([outer_z, inner_z]) if len(inner_z) > 0 else np.array(outer_z)
        
        if len(all_r) > 0:
            r_min, r_max = np.min(all_r), np.max(all_r)
            z_min, z_max = np.min(all_z), np.max(all_z)
            r_margin = max(0.5, (r_max - r_min) * 0.4) if r_max > r_min else 1.0
            z_margin = max(0.5, (z_max - z_min) * 0.4) if z_max > z_min else 1.0
            
            # 치수 표시
            _draw_dimension_annotations(ax_dim, result, r_min, r_max, z_min, z_max, r_margin, z_margin)
            
            ax_dim.set_xlim(z_min - z_margin, z_max + z_margin)
            y_bottom = max(0, r_min - r_margin * 0.3)
            ax_dim.set_ylim(y_bottom, r_max + r_margin * 0.8)
        
        ax_dim.set_xlabel('Z (axial)')
        ax_dim.set_ylabel('R (radial)')
        w_str = f"{result.width:.2f}" if result.width else "N/A"
        h_str = f"{result.height:.2f}" if result.height else "N/A"
        ax_dim.set_title(f'Dimension Calculation\nW={w_str}mm, H={h_str}mm', fontsize=9, fontweight='bold')
        ax_dim.grid(True, alpha=0.3)
    
    plt.suptitle(f'Face Dimension Analysis: Original 3D → R-Z Projection → Dimension\n{Path(filepath).name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def _draw_dimension_annotations(ax, result, r_min, r_max, z_min, z_max, r_margin, z_margin):
    """치수 화살표와 라벨 그리기 - 간결 버전"""
    
    if result.surface_type == "Cylinder":
        r_mean = result.r_max
        # Width (녹색 텍스트만 - 직경 = 2R)
        ax.text(z_min - z_margin * 0.1, r_mean - r_margin * 0.1,
               f'W={result.width:.2f}', fontsize=10, color='green',
               ha='right', va='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
        
        # Height (빨간색 가로 화살표)
        if result.delta_z > 0.01:
            ax.annotate('', xy=(z_max, r_mean + r_margin * 0.15), 
                       xytext=(z_min, r_mean + r_margin * 0.15),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text((z_min + z_max) / 2, r_mean + r_margin * 0.35,
                   f'H={result.height:.2f}', fontsize=10, color='red', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
    
    elif result.surface_type == "Cone":
        # Width (녹색 세로 화살표)
        ax.annotate('', xy=(z_min - z_margin * 0.15, r_max), 
                   xytext=(z_min - z_margin * 0.15, r_min),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(z_min - z_margin * 0.25, (r_min + r_max) / 2,
               f'W={result.width:.2f}', fontsize=10, color='green',
               ha='right', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
        
        # Height (빨간색 가로 화살표)
        if result.delta_z > 0.01:
            ax.annotate('', xy=(z_max, r_max + r_margin * 0.2), 
                       xytext=(z_min, r_max + r_margin * 0.2),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text((z_min + z_max) / 2, r_max + r_margin * 0.4,
                   f'H={result.height:.2f}', fontsize=10, color='red',
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
    
    elif result.surface_type == "Torus":
        # Width (녹색 세로 화살표)
        ax.annotate('', xy=(z_min - z_margin * 0.15, r_max), 
                   xytext=(z_min - z_margin * 0.15, r_min),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(z_min - z_margin * 0.25, (r_min + r_max) / 2,
               f'W={result.width:.2f}', fontsize=10, color='green',
               ha='right', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
        
        # Height (빨간색 가로 화살표)
        if result.delta_z > 0.01:
            ax.annotate('', xy=(z_max, r_max + r_margin * 0.2), 
                       xytext=(z_min, r_max + r_margin * 0.2),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text((z_min + z_max) / 2, r_max + r_margin * 0.4,
                   f'H={result.height:.2f}', fontsize=10, color='red',
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
    
    elif "Ring" in result.surface_type:
        r_outer = result.r_outer
        r_inner = result.r_inner
        z_pos = result.z_max
        
        # Width only (녹색 세로 화살표) - 링 두께
        ax.annotate('', xy=(z_pos + z_margin * 0.15, r_outer), 
                   xytext=(z_pos + z_margin * 0.15, r_inner),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
        ax.text(z_pos + z_margin * 0.3, (r_outer + r_inner) / 2,
               f'W={result.ring_thickness:.2f}', 
               fontsize=10, color='green', ha='left', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    
    elif "Disk" in result.surface_type:
        r_out = result.r_outer
        z_pos = result.z_max
        
        # Width = Height (녹색 세로 화살표)
        ax.annotate('', xy=(z_pos - z_margin * 0.15, r_out), 
                   xytext=(z_pos - z_margin * 0.15, 0),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(z_pos - z_margin * 0.25, r_out * 0.5,
               f'W=H={result.width:.2f}', fontsize=10, color='green',
               ha='right', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))


# ============================================================================
# Main
# ============================================================================

def points_to_xy(points: List[gp_Pnt]) -> Tuple[np.ndarray, np.ndarray]:
    """3D 점들을 X-Y 좌표로 변환 (Top view)"""
    x_vals = [pt.X() for pt in points]
    y_vals = [pt.Y() for pt in points]
    return np.array(x_vals), np.array(y_vals)


def visualize_single_face(face: TopoDS_Face, face_id: int, result: FaceDimensionResult,
                           output_dir: str, filename: str,
                           all_faces: List[TopoDS_Face] = None,
                           global_bounds: dict = None):
    """
    단일 Face의 원본 3D + 투영 + 치수 측정 시각화.
    - 평면(Plane): X-Y 투영 (Top view)
    - Cylinder/Cone/Torus: R-Z + X-Y 둘 다 (Side + Top view)
    
    all_faces: 전체 형상의 모든 face (전체 맥락 표시용)
    global_bounds: 전체 형상의 범위 (공통 스케일용)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool
    
    # 평면인지 확인
    is_plane_face = "Plane" in result.surface_type
    # 회전면(Cylinder, Cone, Torus)인지 확인
    is_rotational_face = result.surface_type in ["Cylinder", "Cone", "Torus"]
    
    # 레이아웃 결정: 모두 2패널
    # Plane: 3D + X-Y
    # 회전면: 3D + R-Z
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
                verts = []
                for i in range(1, other_tri.NbNodes() + 1):
                    node = other_tri.Node(i)
                    if not location.IsIdentity():
                        node = node.Transformed(location.Transformation())
                    verts.append([node.X(), node.Y(), node.Z()])
                
                tris = []
                for i in range(1, other_tri.NbTriangles() + 1):
                    tri = other_tri.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    tris.append([n1 - 1, n2 - 1, n3 - 1])
                
                verts = np.array(verts)
                tris = np.array(tris)
                
                if len(verts) > 0 and len(tris) > 0:
                    ax3d.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                      triangles=tris, color='lightgray',
                                      edgecolor='none', linewidth=0, alpha=0.3)
    
    # 현재 Face 강조 (색상으로)
    triangulation = BRep_Tool.Triangulation(face, location)
    
    if triangulation is not None:
        vertices = []
        nb_nodes = triangulation.NbNodes()
        for i in range(1, nb_nodes + 1):
            node = triangulation.Node(i)
            if not location.IsIdentity():
                node = node.Transformed(location.Transformation())
            vertices.append([node.X(), node.Y(), node.Z()])
        
        triangles = []
        nb_triangles = triangulation.NbTriangles()
        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            triangles.append([n1 - 1, n2 - 1, n3 - 1])
        
        vertices = np.array(vertices)
        triangles = np.array(triangles)
        
        if len(vertices) > 0 and len(triangles) > 0:
            # 면 타입별 색상
            if result.is_ring:
                face_color = 'orange'
            elif result.surface_type == "Cylinder":
                face_color = 'dodgerblue'
            elif result.surface_type == "Cone":
                face_color = 'limegreen'
            elif result.surface_type == "Torus":
                face_color = 'coral'
            else:
                face_color = 'gold'
            
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
    
    # 전체 범위로 축 설정
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
    
    if is_rotational_face:
        # ========== 회전면: R-Z 투영 (Side view) - 면 채움 + Width/Height 측정 ==========
        ax_rz = fig.add_subplot(1, n_cols, 2)
        
        # 전체 형상의 R-Z 단면을 회색 면으로 그리기
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
        
        # 현재 면의 R-Z 투영 (면 채움)
        if triangulation is not None:
            # 면 타입별 색상
            if result.surface_type == "Cylinder":
                face_color = 'dodgerblue'
            elif result.surface_type == "Cone":
                face_color = 'limegreen'
            elif result.surface_type == "Torus":
                face_color = 'coral'
            else:
                face_color = 'gold'
            
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                pts_3d = [triangulation.Node(n1), triangulation.Node(n2), triangulation.Node(n3)]
                if not location.IsIdentity():
                    pts_3d = [p.Transformed(location.Transformation()) for p in pts_3d]
                rs = [np.sqrt(p.X()**2 + p.Y()**2) for p in pts_3d]
                zs = [p.Z() for p in pts_3d]
                ax_rz.fill(zs, rs, color=face_color, alpha=0.6, edgecolor='darkgray', linewidth=0.3)
        
        # 외경 경계선 (파란색)
        if outer_pts:
            outer_r, outer_z = points_to_rz(outer_pts)
            ax_rz.plot(outer_z, outer_r, 'b-', linewidth=2.5, label='Outer')
        
        # 내경 경계선 (빨간색) - Cone/Torus의 구멍 반영
        if inner_pts_all:
            inner_r, inner_z = points_to_rz(inner_pts_all)
            ax_rz.plot(inner_z, inner_r, 'r-', linewidth=2.5, label='Inner')
        
        ax_rz.set_xlabel('Z (axial)')
        ax_rz.set_ylabel('R (radial)')
        ax_rz.set_title('R-Z Projection (Side View)', fontsize=10)
        ax_rz.grid(True, alpha=0.3)
        ax_rz.legend(fontsize=8)
        
        # 전체 범위 설정
        if global_bounds:
            ax_rz.set_xlim(global_bounds['z_min'] - 0.5, global_bounds['z_max'] + 0.5)
            ax_rz.set_ylim(0, global_bounds['r_max'] * 1.1)
        
        # R-Z에서 Width와 Height 표시
        if outer_pts:
            r_min_val = np.min(outer_r)
            r_max_val = np.max(outer_r)
            z_min_val = np.min(outer_z)
            z_max_val = np.max(outer_z)
            z_mid = (z_min_val + z_max_val) / 2
            
            # Height (빨간색 가로 화살표) - ΔZ
            if result.delta_z > 0.01:
                ax_rz.annotate('', xy=(z_max_val, r_max_val + 0.3), 
                              xytext=(z_min_val, r_max_val + 0.3),
                              arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax_rz.text(z_mid, r_max_val + 0.5,
                          f'H={result.height:.2f}', fontsize=10, color='red',
                          ha='center', va='bottom', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
            
            # Width (녹색 세로 화살표)
            if result.surface_type == "Cylinder":
                # Cylinder: Width = 2R (직경)
                ax_rz.annotate('', xy=(z_min_val - 0.3, r_max_val), 
                              xytext=(z_min_val - 0.3, 0),
                              arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                ax_rz.text(z_min_val - 0.5, r_max_val / 2,
                          f'W={result.width:.2f}', fontsize=10, color='green',
                          ha='right', va='center', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
            else:
                # Cone/Torus: Width = ΔR
                ax_rz.annotate('', xy=(z_min_val - 0.3, r_max_val), 
                              xytext=(z_min_val - 0.3, r_min_val),
                              arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                ax_rz.text(z_min_val - 0.5, (r_max_val + r_min_val) / 2,
                          f'W={result.width:.2f}', fontsize=10, color='green',
                          ha='right', va='center', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    else:
        # ========== 평면: X-Y 투영 결과 + 치수 측정 ==========
        ax_proj = fig.add_subplot(1, n_cols, 2)
        
        # 전체 형상을 회색으로 먼저 그리기
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
        
        # 전체 범위 설정
        if global_bounds:
            max_r = max(abs(global_bounds['x_min']), abs(global_bounds['x_max']),
                       abs(global_bounds['y_min']), abs(global_bounds['y_max']))
            ax_proj.set_xlim(-max_r * 1.1, max_r * 1.1)
            ax_proj.set_ylim(-max_r * 1.1, max_r * 1.1)
        
        # X-Y 투영에서 치수 표시
        if result.is_ring and outer_pts and inner_pts_all:
            outer_x, outer_y = points_to_xy(outer_pts)
            inner_x, inner_y = points_to_xy(inner_pts_all)
            r_out = np.max(np.sqrt(outer_x**2 + outer_y**2))
            r_in = np.max(np.sqrt(inner_x**2 + inner_y**2))
            
            # 두께 화살표 (녹색)
            ax_proj.annotate('', xy=(r_out, 0), xytext=(r_in, 0),
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
            ax_proj.text((r_out + r_in) / 2, 0.5, 
                       f'W={result.ring_thickness:.2f}',
                       fontsize=10, color='green', ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
        elif outer_pts:
            # 디스크: 직경 표시
            outer_x, outer_y = points_to_xy(outer_pts)
            r_out = np.max(np.sqrt(outer_x**2 + outer_y**2))
            
            ax_proj.annotate('', xy=(r_out, 0), xytext=(-r_out, 0),
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
            ax_proj.text(0, r_out * 0.15, f'W=H={result.width:.2f}',
                       fontsize=10, color='green', ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    
    # 전체 타이틀
    w_str = f"{result.width:.2f}" if result.width else "N/A"
    h_str = f"{result.height:.2f}" if result.height else "N/A"
    plt.suptitle(f'Face {face_id}: {result.surface_type}{ring_flag}\n'
                 f'Width = {w_str}mm, Height = {h_str}mm', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 저장
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def visualize_all_faces_separate(filepath: str, output_dir: str = "projection_results"):
    """
    모든 유효 Face를 개별 이미지로 저장.
    전체 형상의 맥락과 공통 스케일로 표시.
    """
    import matplotlib.pyplot as plt
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    
    shape = load_step_file(filepath)
    if shape is None:
        return
    
    # 메쉬 생성
    mesh_tool = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh_tool.Perform()
    
    faces = extract_faces(shape)
    
    # 전체 형상의 global bounds 계산
    all_pts = []
    for face in faces:
        outer_wire, _ = get_face_wires(face)
        if outer_wire:
            pts = sample_wire_points(outer_wire, 50)
            all_pts.extend(pts)
    
    if all_pts:
        xs = [p.X() for p in all_pts]
        ys = [p.Y() for p in all_pts]
        zs = [p.Z() for p in all_pts]
        rs = [np.sqrt(p.X()**2 + p.Y()**2) for p in all_pts]
        
        margin = 0.5
        global_bounds = {
            'x_min': min(xs) - margin,
            'x_max': max(xs) + margin,
            'y_min': min(ys) - margin,
            'y_max': max(ys) + margin,
            'z_min': min(zs) - margin,
            'z_max': max(zs) + margin,
            'r_max': max(rs) + margin
        }
    else:
        global_bounds = None
    
    # 출력 폴더 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 모델명으로 하위 폴더 생성
    model_name = Path(filepath).stem
    model_dir = output_path / model_name
    model_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving individual face visualizations to: {model_dir}")
    
    saved_files = []
    for i, face in enumerate(faces):
        result = compute_face_dimension(face, i)
        
        # 작은 면은 스킵
        if result.width is None or result.width < 0.1:
            continue
        
        # 파일명 생성
        type_short = result.surface_type.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"Face_{i:02d}_{type_short}.png"
        
        saved_path = visualize_single_face(
            face, i, result, str(model_dir), filename,
            all_faces=faces, global_bounds=global_bounds
        )
        saved_files.append(saved_path)
        print(f"  Saved: {filename}")
    
    print(f"\nTotal {len(saved_files)} face images saved.")
    return saved_files


if __name__ == "__main__":
    import sys
    
    # 테스트 파일 (토러스가 있는 파일 선택)
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
    else:
        step_file = "generated_turning_models/turning_N6_H3_S0_G5_000.step"
    
    print(f"\n{'=' * 60}")
    print(f"Projection-Based Face Dimension Analysis")
    print(f"File: {step_file}")
    print(f"{'=' * 60}")
    
    # 분석
    results = analyze_turning_model(step_file, samples_per_edge=50)
    
    # 요약 테이블
    print_summary_table(results)
    
    # 각 Face별 개별 이미지 저장
    print("\n" + "=" * 60)
    print("Generating individual face visualizations...")
    print("=" * 60)
    visualize_all_faces_separate(step_file, output_dir="projection_results")
    
    print("\nDone!")
    print("  - Individual face images saved in: projection_results/")
