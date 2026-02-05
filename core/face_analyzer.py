"""
Face 치수 분석 모듈

터닝 형상 (회전축 = Z)의 특성을 활용하여 면의 유효 치수를 계산:
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
from dataclasses import dataclass

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.gp import gp_Pnt
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
)
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
# Wire/Edge Sampling Utilities
# ============================================================================

def sample_edge_points(edge: TopoDS_Edge, n_samples: int = 30) -> List[gp_Pnt]:
    """Edge에서 균등하게 점 샘플링."""
    # n_samples가 2 미만이면 최소 2개로 설정 (ZeroDivisionError 방지)
    n_samples = max(2, n_samples)
    
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
    """Wire의 모든 Edge에서 점 샘플링."""
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
    """
    r_vals = []
    z_vals = []
    
    for pt in points:
        r = math.sqrt(pt.X()**2 + pt.Y()**2)
        z = pt.Z()
        r_vals.append(r)
        z_vals.append(z)
    
    return np.array(r_vals), np.array(z_vals)


def points_to_xy(points: List[gp_Pnt]) -> Tuple[np.ndarray, np.ndarray]:
    """3D 점들을 X-Y 좌표로 변환 (Top view)"""
    x_vals = [pt.X() for pt in points]
    y_vals = [pt.Y() for pt in points]
    return np.array(x_vals), np.array(y_vals)


def analyze_rz_points(r_vals: np.ndarray, z_vals: np.ndarray) -> Dict:
    """R-Z 점들에서 치수 분석."""
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
    """Plane이 Z축에 수직인지 (XY 평면과 평행) 확인."""
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

def _compute_cylinder_dimension(face: TopoDS_Face, face_id: int,
                                 samples_per_edge: int = 30) -> FaceDimensionResult:
    """원통면 치수 계산: width = 2R (직경), height = Δz (축방향)"""
    outer_wire, inner_wires = get_face_wires(face)
    
    all_points = []
    if outer_wire:
        all_points.extend(sample_wire_points(outer_wire, samples_per_edge))
    for iw in inner_wires:
        all_points.extend(sample_wire_points(iw, samples_per_edge))
    
    r_vals, z_vals = points_to_rz(all_points)
    stats = analyze_rz_points(r_vals, z_vals)
    
    radius = stats['r_mean']
    
    return FaceDimensionResult(
        face_id=face_id,
        surface_type="Cylinder",
        width=2 * radius,
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


def _compute_cone_dimension(face: TopoDS_Face, face_id: int,
                            samples_per_edge: int = 30) -> FaceDimensionResult:
    """원추면 (챔퍼) 치수 계산: width = Δr, height = Δz"""
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


def _compute_torus_dimension(face: TopoDS_Face, face_id: int,
                             samples_per_edge: int = 30) -> FaceDimensionResult:
    """토러스 (라운드/필렛) 치수 계산: width = Δr, height = Δz"""
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


def _compute_plane_dimension(face: TopoDS_Face, face_id: int,
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
        r_inner = float(np.max(inner_r))
    
    # 전체 점으로 통합 분석
    all_r = np.concatenate([outer_r, inner_r]) if len(inner_r) > 0 else outer_r
    all_z = np.concatenate([outer_z, inner_z]) if len(inner_z) > 0 else outer_z
    all_stats = analyze_rz_points(all_r, all_z)
    
    # 치수 계산
    if is_ring:
        ring_thickness = r_outer - r_inner
        return FaceDimensionResult(
            face_id=face_id,
            surface_type="Plane (Ring)",
            width=ring_thickness,
            height=None,
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
        diameter = 2 * r_outer
        return FaceDimensionResult(
            face_id=face_id,
            surface_type="Plane (Disk)",
            width=diameter,
            height=diameter,
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


def _compute_bspline_dimension(face: TopoDS_Face, face_id: int,
                               samples_per_edge: int = 30) -> FaceDimensionResult:
    """BSpline 서피스 치수 계산 (블렌드/라운드 등)."""
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
# FaceAnalyzer Class
# ============================================================================

class FaceAnalyzer:
    """
    터닝 형상의 면 치수 분석기.
    
    사용법:
        analyzer = FaceAnalyzer(samples_per_edge=30)
        results = analyzer.analyze_shape(shape)
        
        for result in results:
            print(f"Face {result.face_id}: {result.surface_type}")
            print(f"  Width: {result.width}, Height: {result.height}")
    """
    
    def __init__(self, samples_per_edge: int = 30):
        """
        Args:
            samples_per_edge: 각 Edge당 샘플 점 수
        """
        self.samples_per_edge = samples_per_edge
    
    def analyze_face(self, face: TopoDS_Face, face_id: int) -> FaceDimensionResult:
        """
        단일 Face 치수 분석.
        
        Args:
            face: 분석할 Face
            face_id: Face 식별자
            
        Returns:
            FaceDimensionResult
        """
        surf_type, type_enum = get_surface_type(face)
        
        if type_enum == GeomAbs_Cylinder:
            return _compute_cylinder_dimension(face, face_id, self.samples_per_edge)
        elif type_enum == GeomAbs_Cone:
            return _compute_cone_dimension(face, face_id, self.samples_per_edge)
        elif type_enum == GeomAbs_Torus:
            return _compute_torus_dimension(face, face_id, self.samples_per_edge)
        elif type_enum == GeomAbs_Plane:
            return _compute_plane_dimension(face, face_id, self.samples_per_edge)
        elif type_enum == GeomAbs_BSplineSurface:
            return _compute_bspline_dimension(face, face_id, self.samples_per_edge)
        else:
            # 기타: 일반적인 R-Z 분석
            outer_wire, inner_wires = get_face_wires(face)
            all_points = []
            if outer_wire:
                all_points.extend(sample_wire_points(outer_wire, self.samples_per_edge))
            for iw in inner_wires:
                all_points.extend(sample_wire_points(iw, self.samples_per_edge))
            
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
    
    def analyze_shape(self, shape: TopoDS_Shape) -> List[FaceDimensionResult]:
        """
        Shape의 모든 Face 치수 분석.
        
        Args:
            shape: 분석할 Shape
            
        Returns:
            FaceDimensionResult 리스트
        """
        topo = TopologyExplorer(shape)
        faces = list(topo.faces())
        
        results = []
        for i, face in enumerate(faces):
            result = self.analyze_face(face, i)
            results.append(result)
        
        return results
    
    def get_valid_faces(
        self,
        shape: TopoDS_Shape,
        min_dimension: float = 0.1,
        target_types: List[str] = None
    ) -> List[FaceDimensionResult]:
        """
        유효한 면만 필터링하여 반환.
        
        Args:
            shape: 분석할 Shape
            min_dimension: 최소 치수 (이하는 제외)
            target_types: 대상 면 타입 (None이면 전체)
            
        Returns:
            필터링된 FaceDimensionResult 리스트
        """
        if target_types is None:
            target_types = ["Plane", "Cylinder", "Cone", "Torus"]
        
        results = self.analyze_shape(shape)
        
        valid = []
        for r in results:
            # 치수 유효성 검사
            if r.width is None or r.width < min_dimension:
                continue
            
            # 타입 검사
            if any(t in r.surface_type for t in target_types):
                valid.append(r)
        
        return valid
    
    def print_summary(self, results: List[FaceDimensionResult]):
        """분석 결과 요약 출력."""
        print("\n" + "=" * 100)
        print(f"{'Face':>5} | {'Type':^15} | {'Width':>10} | {'Height':>10} | "
              f"{'R_max':>8} | {'R_min':>8} | {'Ring':>4}")
        print("-" * 100)
        
        for r in results:
            if r.width is not None and r.width < 0.1:
                continue
            
            w_str = f"{r.width:.3f}" if r.width is not None else "N/A"
            h_str = f"{r.height:.3f}" if r.height is not None else "N/A"
            ring_str = "Yes" if r.is_ring else ""
            
            print(f"{r.face_id:>5} | {r.surface_type:^15} | {w_str:>10} | {h_str:>10} | "
                  f"{r.r_max:>8.2f} | {r.r_min:>8.2f} | {ring_str:>4}")
        
        print("=" * 100)


# ============================================================================
# Backward Compatibility Function
# ============================================================================

def compute_face_dimension(face: TopoDS_Face, face_id: int,
                           samples_per_edge: int = 30) -> FaceDimensionResult:
    """
    면 치수 계산 (기존 API 호환용).
    
    새 코드에서는 FaceAnalyzer 클래스 사용 권장.
    """
    analyzer = FaceAnalyzer(samples_per_edge)
    return analyzer.analyze_face(face, face_id)
