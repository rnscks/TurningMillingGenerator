"""
UV 기반 터닝 Face 폭/너비 계산 모듈 (A안: 트림 경계 반영)

- 폭(width): v = v_mid에서 u_min → u_max로 가는 iso-curve의 3D 길이 (inside 구간만)
- 너비(height): u = u_mid에서 v_min → v_max로 가는 iso-curve의 3D 길이 (inside 구간만)
- FClass2d로 inside/outside 판별
- Bisection으로 경계점 정밀화
- Chord error 기반 적응형 분할로 정확한 길이 계산
- max segment를 대표값으로 사용
"""
import math
import statistics
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion
)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SegmentInfo:
    """분할된 세그먼트 정보"""
    param_start: float
    param_end: float
    chord_length: float
    chord_error: float
    depth: int


@dataclass
class IsoLineResult:
    """Iso-line 계산 결과"""
    total_length: float
    num_segments: int
    max_depth: int
    segments: List[SegmentInfo]
    max_chord_error: float
    avg_chord_error: float


@dataclass
class InsideInterval:
    """Face 내부의 유효 파라미터 구간"""
    start: float
    end: float
    length_3d: float = 0.0  # 3D 길이


@dataclass
class CoarseSamplePoint:
    """Coarse 샘플 포인트 정보"""
    param: float           # u 또는 v 파라미터
    is_inside: bool        # inside 여부
    pt_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D 좌표


@dataclass
class TrimmedIsoResult:
    """트림 반영된 Iso-line 계산 결과"""
    intervals: List[InsideInterval]
    total_length: float
    max_segment_length: float
    num_intervals: int
    coarse_samples: int
    boundary_refinements: int
    # 그리드 시각화용
    coarse_points: List[CoarseSamplePoint] = field(default_factory=list)
    boundary_params: List[float] = field(default_factory=list)  # bisection으로 찾은 경계점들
    fixed_param: float = 0.0  # 고정된 v 또는 u 값


@dataclass
class FaceDimension:
    """Face의 UV 기반 치수 정보"""
    face_id: int
    surface_type: str
    
    # UV 범위 (트림 반영된 bbox)
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    
    # 계산된 치수 (mm) - max segment 기준
    width: float          # v=v_mid에서의 u방향 최대 연속 구간 길이
    height: float         # u=u_mid에서의 v방향 최대 연속 구간 길이
    
    # 안정화된 치수 (여러 라인 중앙값)
    width_stable: float
    height_stable: float
    
    # 모든 샘플 값 (max segment)
    width_samples: List[float] = field(default_factory=list)
    height_samples: List[float] = field(default_factory=list)
    
    # 트림 분석 결과
    width_trim_result: Optional[TrimmedIsoResult] = None
    height_trim_result: Optional[TrimmedIsoResult] = None
    
    # 추가 정보
    is_periodic_u: bool = False
    is_periodic_v: bool = False
    is_sliver: bool = False
    has_inner_trim: bool = False  # 내부 구멍 존재 여부
    
    @property
    def aspect_ratio(self) -> float:
        """종횡비"""
        if self.height_stable > 0:
            return self.width_stable / self.height_stable
        return 0.0


# ============================================================================
# Basic Utility Functions
# ============================================================================

def pnt_mid(p0: gp_Pnt, p1: gp_Pnt) -> gp_Pnt:
    """두 점의 중간점"""
    return gp_Pnt(
        0.5 * (p0.X() + p1.X()),
        0.5 * (p0.Y() + p1.Y()),
        0.5 * (p0.Z() + p1.Z()),
    )


def dist(p0: gp_Pnt, p1: gp_Pnt) -> float:
    """두 점 사이 거리"""
    return p0.Distance(p1)


def get_surface_type_name(adaptor: BRepAdaptor_Surface) -> str:
    """서피스 타입 문자열 반환"""
    type_map = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BSplineSurface: "BSpline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion: "Extrusion",
    }
    surf_type = adaptor.GetType()
    return type_map.get(surf_type, f"Other({surf_type})")


# ============================================================================
# Inside/Outside Classification (FClass2d)
# ============================================================================

def inside_uv(fclass2d: BRepTopAdaptor_FClass2d, u: float, v: float) -> bool:
    """
    (u, v) 점이 Face 내부인지 판별.
    TopAbs_IN 또는 TopAbs_ON이면 True.
    
    Note: Perform(pnt2d, RecadreOnPeriodic=True) - 두 번째 인자는 주기면 보정 여부
    """
    state = fclass2d.Perform(gp_Pnt2d(u, v), False)
    return (state == TopAbs_IN) or (state == TopAbs_ON)


def bisect_boundary_u(fclass2d: BRepTopAdaptor_FClass2d, 
                       ua: float, ub: float, v_fixed: float,
                       inside_a: bool, tol_param: float = 1e-9, max_iter: int = 40) -> float:
    """
    [ua, ub] 사이에서 inside가 바뀌는 지점 u*를 이분탐색으로 찾음.
    
    Args:
        fclass2d: Face classifier
        ua, ub: 탐색 구간
        v_fixed: 고정된 v 값
        inside_a: ua에서의 inside 여부
        tol_param: 파라미터 종료 허용 오차
        max_iter: 최대 반복 횟수
        
    Returns:
        경계점 u*
    """
    a, b = ua, ub
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        inside_m = inside_uv(fclass2d, m, v_fixed)
        if inside_m == inside_a:
            a = m
        else:
            b = m
        if abs(b - a) <= tol_param:
            break
    return 0.5 * (a + b)


def bisect_boundary_v(fclass2d: BRepTopAdaptor_FClass2d,
                       va: float, vb: float, u_fixed: float,
                       inside_a: bool, tol_param: float = 1e-9, max_iter: int = 40) -> float:
    """[va, vb] 사이에서 inside가 바뀌는 지점 v*를 찾음."""
    a, b = va, vb
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        inside_m = inside_uv(fclass2d, u_fixed, m)
        if inside_m == inside_a:
            a = m
        else:
            b = m
        if abs(b - a) <= tol_param:
            break
    return 0.5 * (a + b)


# ============================================================================
# Coarse Sampling + Boundary Refinement
# ============================================================================

def estimate_iso_length_rough(surf, u0: float, u1: float, v_fixed: float, 
                               n_samples: int = 32) -> float:
    """
    대략적인 iso-curve 길이 추정 (coarse chord 합).
    n_coarse를 자동 결정하기 위해 사용.
    """
    total = 0.0
    prev_pt = surf.Value(u0, v_fixed)
    for i in range(1, n_samples + 1):
        u = u0 + (u1 - u0) * i / n_samples
        pt = surf.Value(u, v_fixed)
        total += dist(prev_pt, pt)
        prev_pt = pt
    return total


def compute_n_coarse(length_estimate: float, delta_mm: float = 1.0,
                      n_min: int = 64, n_max: int = 512) -> int:
    """
    대략적인 길이 추정값을 기반으로 coarse 샘플 수 결정.
    
    Args:
        length_estimate: 대략적인 3D 길이 (mm)
        delta_mm: 목표 coarse 샘플 간격 (mm)
        n_min, n_max: 샘플 수 범위
        
    Returns:
        coarse 샘플 수
    """
    if length_estimate <= 0:
        return n_min
    n = int(math.ceil(length_estimate / delta_mm))
    return max(n_min, min(n_max, n))


def find_inside_intervals_u_detailed(fclass2d: BRepTopAdaptor_FClass2d, surf,
                                       u_min: float, u_max: float, v_fixed: float,
                                       n_coarse: int = 128, tol_param: float = 1e-9) -> Tuple[List[InsideInterval], List[CoarseSamplePoint], List[float]]:
    """
    v = v_fixed에서 u 방향으로 inside인 구간들을 찾고, 경계를 bisection으로 정밀화.
    
    Returns:
        (intervals, coarse_points, boundary_params)
        - intervals: InsideInterval 리스트
        - coarse_points: CoarseSamplePoint 리스트 (시각화용)
        - boundary_params: bisection으로 찾은 경계점 파라미터들
    """
    # Coarse 샘플링
    us = [u_min + (u_max - u_min) * i / (n_coarse - 1) for i in range(n_coarse)]
    ins = [inside_uv(fclass2d, u, v_fixed) for u in us]
    
    # Coarse 샘플 정보 저장 (3D 좌표 포함)
    coarse_points = []
    for u, is_in in zip(us, ins):
        pt = surf.Value(u, v_fixed)
        coarse_points.append(CoarseSamplePoint(
            param=u,
            is_inside=is_in,
            pt_3d=(pt.X(), pt.Y(), pt.Z())
        ))
    
    intervals = []
    boundary_params = []
    start_u = None
    
    for i in range(n_coarse - 1):
        a, b = us[i], us[i + 1]
        ina, inb = ins[i], ins[i + 1]
        
        # Inside 구간 시작
        if start_u is None and ina:
            start_u = a
        
        # Inside → Outside 전환: 구간 종료
        if ina and (not inb) and start_u is not None:
            u_star = bisect_boundary_u(fclass2d, a, b, v_fixed, inside_a=True, tol_param=tol_param)
            intervals.append(InsideInterval(start=start_u, end=u_star))
            boundary_params.append(u_star)
            start_u = None
        
        # Outside → Inside 전환: 새 구간 시작
        if (not ina) and inb:
            u_star = bisect_boundary_u(fclass2d, a, b, v_fixed, inside_a=False, tol_param=tol_param)
            boundary_params.append(u_star)
            start_u = u_star
    
    # 마지막 구간 처리
    if start_u is not None:
        if ins[-1]:
            intervals.append(InsideInterval(start=start_u, end=u_max))
    
    # 유효하지 않은 구간 제거
    intervals = [iv for iv in intervals if abs(iv.end - iv.start) > 1e-10]
    
    return intervals, coarse_points, boundary_params


def find_inside_intervals_u(fclass2d: BRepTopAdaptor_FClass2d, surf,
                             u_min: float, u_max: float, v_fixed: float,
                             n_coarse: int = 128, tol_param: float = 1e-9) -> List[InsideInterval]:
    """간단 버전 - 구간만 반환"""
    intervals, _, _ = find_inside_intervals_u_detailed(fclass2d, surf, u_min, u_max, v_fixed, n_coarse, tol_param)
    return intervals


def find_inside_intervals_v_detailed(fclass2d: BRepTopAdaptor_FClass2d, surf,
                                       v_min: float, v_max: float, u_fixed: float,
                                       n_coarse: int = 128, tol_param: float = 1e-9) -> Tuple[List[InsideInterval], List[CoarseSamplePoint], List[float]]:
    """
    u = u_fixed에서 v 방향으로 inside인 구간들을 찾음.
    
    Returns:
        (intervals, coarse_points, boundary_params)
    """
    vs = [v_min + (v_max - v_min) * i / (n_coarse - 1) for i in range(n_coarse)]
    ins = [inside_uv(fclass2d, u_fixed, v) for v in vs]
    
    # Coarse 샘플 정보 저장
    coarse_points = []
    for v, is_in in zip(vs, ins):
        pt = surf.Value(u_fixed, v)
        coarse_points.append(CoarseSamplePoint(
            param=v,
            is_inside=is_in,
            pt_3d=(pt.X(), pt.Y(), pt.Z())
        ))
    
    intervals = []
    boundary_params = []
    start_v = None
    
    for i in range(n_coarse - 1):
        a, b = vs[i], vs[i + 1]
        ina, inb = ins[i], ins[i + 1]
        
        if start_v is None and ina:
            start_v = a
        
        if ina and (not inb) and start_v is not None:
            v_star = bisect_boundary_v(fclass2d, a, b, u_fixed, inside_a=True, tol_param=tol_param)
            intervals.append(InsideInterval(start=start_v, end=v_star))
            boundary_params.append(v_star)
            start_v = None
        
        if (not ina) and inb:
            v_star = bisect_boundary_v(fclass2d, a, b, u_fixed, inside_a=False, tol_param=tol_param)
            boundary_params.append(v_star)
            start_v = v_star
    
    if start_v is not None:
        if ins[-1]:
            intervals.append(InsideInterval(start=start_v, end=v_max))
    
    intervals = [iv for iv in intervals if abs(iv.end - iv.start) > 1e-10]
    
    return intervals, coarse_points, boundary_params


def find_inside_intervals_v(fclass2d: BRepTopAdaptor_FClass2d, surf,
                             v_min: float, v_max: float, u_fixed: float,
                             n_coarse: int = 128, tol_param: float = 1e-9) -> List[InsideInterval]:
    """간단 버전 - 구간만 반환"""
    intervals, _, _ = find_inside_intervals_v_detailed(fclass2d, surf, v_min, v_max, u_fixed, n_coarse, tol_param)
    return intervals


# ============================================================================
# Adaptive Length Integration
# ============================================================================

def adaptive_iso_length_u_on_interval(surf, u0: float, u1: float, v_fixed: float,
                                        eps_mm: float = 0.1, max_depth: int = 14,
                                        min_param_step: float = 1e-7) -> float:
    """
    유효 구간 [u0, u1]에서 chord-error 기반 적응형 적분으로 길이 계산.
    """
    P0 = surf.Value(u0, v_fixed)
    P1 = surf.Value(u1, v_fixed)
    
    stack = [(u0, u1, 0, P0, P1)]
    total = 0.0
    
    while stack:
        a, b, depth, Pa, Pb = stack.pop()
        
        if depth >= max_depth or abs(b - a) <= min_param_step:
            total += dist(Pa, Pb)
            continue
        
        m = 0.5 * (a + b)
        Pm = surf.Value(m, v_fixed)
        
        M = pnt_mid(Pa, Pb)
        e = dist(Pm, M)
        
        if e <= eps_mm:
            total += dist(Pa, Pb)
        else:
            stack.append((m, b, depth + 1, Pm, Pb))
            stack.append((a, m, depth + 1, Pa, Pm))
    
    return total


def adaptive_iso_length_v_on_interval(surf, v0: float, v1: float, u_fixed: float,
                                        eps_mm: float = 0.1, max_depth: int = 14,
                                        min_param_step: float = 1e-7) -> float:
    """유효 구간 [v0, v1]에서 v 방향 적응형 적분."""
    P0 = surf.Value(u_fixed, v0)
    P1 = surf.Value(u_fixed, v1)
    
    stack = [(v0, v1, 0, P0, P1)]
    total = 0.0
    
    while stack:
        a, b, depth, Pa, Pb = stack.pop()
        
        if depth >= max_depth or abs(b - a) <= min_param_step:
            total += dist(Pa, Pb)
            continue
        
        m = 0.5 * (a + b)
        Pm = surf.Value(u_fixed, m)
        
        M = pnt_mid(Pa, Pb)
        e = dist(Pm, M)
        
        if e <= eps_mm:
            total += dist(Pa, Pb)
        else:
            stack.append((m, b, depth + 1, Pm, Pb))
            stack.append((a, m, depth + 1, Pa, Pm))
    
    return total


# ============================================================================
# Main Trimmed Iso-curve Length Calculation
# ============================================================================

def compute_trimmed_iso_length_u(face: TopoDS_Face, surf, fclass2d,
                                   u_min: float, u_max: float, v_fixed: float,
                                   delta_mm: float = 1.0, eps_mm: float = 0.1,
                                   n_min: int = 64, n_max: int = 512,
                                   max_depth: int = 14) -> TrimmedIsoResult:
    """
    트림 경계를 반영한 U 방향 iso-curve 길이 계산.
    
    Returns:
        TrimmedIsoResult with intervals, lengths, and coarse sample info
    """
    # 1) 대략적인 길이 추정
    length_est = estimate_iso_length_rough(surf, u_min, u_max, v_fixed, n_samples=32)
    
    # 2) n_coarse 결정
    n_coarse = compute_n_coarse(length_est, delta_mm, n_min, n_max)
    
    # 3) Inside intervals 찾기 (bisection refinement 포함) + coarse 샘플 정보
    intervals, coarse_points, boundary_params = find_inside_intervals_u_detailed(
        fclass2d, surf, u_min, u_max, v_fixed, n_coarse)
    
    # 4) 각 interval에서 적응형 적분
    for iv in intervals:
        iv.length_3d = adaptive_iso_length_u_on_interval(surf, iv.start, iv.end, v_fixed,
                                                          eps_mm, max_depth)
    
    # 5) 결과 집계
    total_length = sum(iv.length_3d for iv in intervals)
    max_segment = max((iv.length_3d for iv in intervals), default=0.0)
    
    return TrimmedIsoResult(
        intervals=intervals,
        total_length=total_length,
        max_segment_length=max_segment,
        num_intervals=len(intervals),
        coarse_samples=n_coarse,
        boundary_refinements=len(boundary_params),
        coarse_points=coarse_points,
        boundary_params=boundary_params,
        fixed_param=v_fixed
    )


def compute_trimmed_iso_length_v(face: TopoDS_Face, surf, fclass2d,
                                   v_min: float, v_max: float, u_fixed: float,
                                   delta_mm: float = 1.0, eps_mm: float = 0.1,
                                   n_min: int = 64, n_max: int = 512,
                                   max_depth: int = 14) -> TrimmedIsoResult:
    """트림 경계를 반영한 V 방향 iso-curve 길이 계산."""
    # V 방향 rough estimate
    total = 0.0
    prev_pt = surf.Value(u_fixed, v_min)
    for i in range(1, 33):
        v = v_min + (v_max - v_min) * i / 32
        pt = surf.Value(u_fixed, v)
        total += dist(prev_pt, pt)
        prev_pt = pt
    length_est = total
    
    n_coarse = compute_n_coarse(length_est, delta_mm, n_min, n_max)
    
    # Inside intervals 찾기 + coarse 샘플 정보
    intervals, coarse_points, boundary_params = find_inside_intervals_v_detailed(
        fclass2d, surf, v_min, v_max, u_fixed, n_coarse)
    
    for iv in intervals:
        iv.length_3d = adaptive_iso_length_v_on_interval(surf, iv.start, iv.end, u_fixed,
                                                          eps_mm, max_depth)
    
    total_length = sum(iv.length_3d for iv in intervals)
    max_segment = max((iv.length_3d for iv in intervals), default=0.0)
    
    return TrimmedIsoResult(
        intervals=intervals,
        total_length=total_length,
        max_segment_length=max_segment,
        num_intervals=len(intervals),
        coarse_samples=n_coarse,
        boundary_refinements=len(boundary_params),
        coarse_points=coarse_points,
        boundary_params=boundary_params,
        fixed_param=u_fixed
    )


# ============================================================================
# Face Dimension Computation (Main Entry Point)
# ============================================================================

def compute_face_dimension_simple(face: TopoDS_Face, face_id: int,
                                    eps_mm: float = 0.1, max_depth: int = 14,
                                    delta_mm: float = 1.0,
                                    n_min: int = 64, n_max: int = 512,
                                    min_dimension: float = 0.1) -> FaceDimension:
    """
    Face의 UV 기반 폭/너비 계산 - 대표 라인(중앙) 하나만 사용하는 단순화 버전.
    
    Args:
        face: TopoDS_Face
        face_id: Face 식별자
        eps_mm: 허용 chord error (mm)
        max_depth: 최대 분할 깊이
        delta_mm: coarse 샘플링 목표 간격 (mm)
        n_min, n_max: coarse 샘플 수 범위
        min_dimension: 슬리버 판정 임계값 (mm)
        
    Returns:
        FaceDimension 객체 (중앙 라인 기준, coarse 그리드 정보 포함)
    """
    # Surface adaptor 생성
    adaptor = BRepAdaptor_Surface(face, True)
    
    # UV 범위 (트림 반영된 bbox)
    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    
    # 주기성 체크
    is_periodic_u = adaptor.IsUPeriodic()
    is_periodic_v = adaptor.IsVPeriodic()
    
    # 서피스 타입
    surf_type = get_surface_type_name(adaptor)
    
    # FClass2d 생성 (inside/outside 판별용)
    fclass2d = BRepTopAdaptor_FClass2d(face, 1e-6)
    
    # 중앙 라인 위치
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)
    
    # 폭 계산: v = v_mid에서 u 방향 (대표 라인 하나)
    width_trim_result = None
    width = 0.0
    has_inner_trim = False
    
    try:
        width_trim_result = compute_trimmed_iso_length_u(
            face, adaptor, fclass2d,
            u_min, u_max, v_mid,
            delta_mm, eps_mm, n_min, n_max, max_depth
        )
        width = width_trim_result.max_segment_length
        if width_trim_result.num_intervals > 1:
            has_inner_trim = True
    except Exception as e:
        print(f"  Warning: width 계산 실패: {e}")
    
    # 너비 계산: u = u_mid에서 v 방향 (대표 라인 하나)
    height_trim_result = None
    height = 0.0
    
    try:
        height_trim_result = compute_trimmed_iso_length_v(
            face, adaptor, fclass2d,
            v_min, v_max, u_mid,
            delta_mm, eps_mm, n_min, n_max, max_depth
        )
        height = height_trim_result.max_segment_length
        if height_trim_result.num_intervals > 1:
            has_inner_trim = True
    except Exception as e:
        print(f"  Warning: height 계산 실패: {e}")
    
    # 슬리버 판정
    is_sliver = (width < min_dimension or height < min_dimension)
    
    return FaceDimension(
        face_id=face_id,
        surface_type=surf_type,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        width=width,
        height=height,
        width_stable=width,  # 단일 라인이므로 동일
        height_stable=height,
        width_samples=[width],
        height_samples=[height],
        width_trim_result=width_trim_result,
        height_trim_result=height_trim_result,
        is_periodic_u=is_periodic_u,
        is_periodic_v=is_periodic_v,
        is_sliver=is_sliver,
        has_inner_trim=has_inner_trim,
    )


def compute_face_dimension(face: TopoDS_Face, face_id: int,
                            eps_mm: float = 0.1, max_depth: int = 14,
                            num_samples: int = 5,
                            delta_mm: float = 1.0,
                            n_min: int = 64, n_max: int = 512,
                            min_dimension: float = 0.1,
                            detailed: bool = False) -> FaceDimension:
    """
    Face의 UV 기반 폭/너비 계산 (트림 경계 반영, A안).
    여러 샘플 라인에서 중앙값으로 안정화.
    
    Args:
        face: TopoDS_Face
        face_id: Face 식별자
        eps_mm: 허용 chord error (mm) - 정밀 적분용
        max_depth: 최대 분할 깊이
        num_samples: 안정화를 위한 샘플 라인 수
        delta_mm: coarse 샘플링 목표 간격 (mm)
        n_min, n_max: coarse 샘플 수 범위
        min_dimension: 슬리버 판정 임계값 (mm)
        detailed: 상세 결과 저장 여부 (True면 중앙 라인의 coarse 정보 저장)
        
    Returns:
        FaceDimension 객체 (max segment 기준)
    """
    # Surface adaptor 생성
    adaptor = BRepAdaptor_Surface(face, True)
    
    # UV 범위 (트림 반영된 bbox)
    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    
    # 주기성 체크
    is_periodic_u = adaptor.IsUPeriodic()
    is_periodic_v = adaptor.IsVPeriodic()
    
    # 서피스 타입
    surf_type = get_surface_type_name(adaptor)
    
    # FClass2d 생성 (inside/outside 판별용)
    fclass2d = BRepTopAdaptor_FClass2d(face, 1e-6)
    
    # 샘플 위치 생성
    if num_samples == 1:
        u_samples = [0.5 * (u_min + u_max)]
        v_samples = [0.5 * (v_min + v_max)]
    else:
        # 약간 안쪽으로 샘플링 (경계에서 약간 떨어진 위치)
        margin_u = (u_max - u_min) * 0.02
        margin_v = (v_max - v_min) * 0.02
        u_samples = [u_min + margin_u + (u_max - u_min - 2*margin_u) * i / (num_samples - 1) 
                     for i in range(num_samples)]
        v_samples = [v_min + margin_v + (v_max - v_min - 2*margin_v) * i / (num_samples - 1) 
                     for i in range(num_samples)]
    
    # 폭 계산: 각 v 위치에서 u 방향 max segment
    width_samples = []
    width_trim_result = None
    has_inner_trim = False
    
    for i, v_pos in enumerate(v_samples):
        try:
            result = compute_trimmed_iso_length_u(
                face, adaptor, fclass2d,
                u_min, u_max, v_pos,
                delta_mm, eps_mm, n_min, n_max, max_depth
            )
            width_samples.append(result.max_segment_length)
            
            # 내부 구멍 감지 (구간이 2개 이상이면)
            if result.num_intervals > 1:
                has_inner_trim = True
            
            # 중앙 라인 결과 저장 (항상 저장하도록 변경)
            if i == num_samples // 2:
                width_trim_result = result
                
        except Exception as e:
            print(f"  Warning: width 계산 실패 at v={v_pos}: {e}")
    
    # 너비 계산: 각 u 위치에서 v 방향 max segment
    height_samples = []
    height_trim_result = None
    
    for i, u_pos in enumerate(u_samples):
        try:
            result = compute_trimmed_iso_length_v(
                face, adaptor, fclass2d,
                v_min, v_max, u_pos,
                delta_mm, eps_mm, n_min, n_max, max_depth
            )
            height_samples.append(result.max_segment_length)
            
            if result.num_intervals > 1:
                has_inner_trim = True
            
            # 중앙 라인 결과 저장 (항상 저장하도록 변경)
            if i == num_samples // 2:
                height_trim_result = result
                
        except Exception as e:
            print(f"  Warning: height 계산 실패 at u={u_pos}: {e}")
    
    # 안정화: 중앙값 사용
    if width_samples:
        width_stable = statistics.median(width_samples)
        width_mid = width_samples[len(width_samples) // 2]
    else:
        width_stable = 0.0
        width_mid = 0.0
    
    if height_samples:
        height_stable = statistics.median(height_samples)
        height_mid = height_samples[len(height_samples) // 2]
    else:
        height_stable = 0.0
        height_mid = 0.0
    
    # 슬리버 판정
    is_sliver = (width_stable < min_dimension or height_stable < min_dimension)
    
    return FaceDimension(
        face_id=face_id,
        surface_type=surf_type,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        width=width_mid,
        height=height_mid,
        width_stable=width_stable,
        height_stable=height_stable,
        width_samples=width_samples,
        height_samples=height_samples,
        width_trim_result=width_trim_result,
        height_trim_result=height_trim_result,
        is_periodic_u=is_periodic_u,
        is_periodic_v=is_periodic_v,
        is_sliver=is_sliver,
        has_inner_trim=has_inner_trim,
    )


# ============================================================================
# File I/O and Analysis
# ============================================================================

def extract_faces(shape: TopoDS_Shape) -> List[TopoDS_Face]:
    """Shape에서 모든 Face 추출"""
    topo = TopologyExplorer(shape)
    return list(topo.faces())


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


def analyze_turning_model(filepath: str, eps_mm: float = 0.1,
                           num_samples: int = 5, detailed: bool = False) -> List[FaceDimension]:
    """터닝 모델의 모든 face에 대해 폭/너비 분석."""
    shape = load_step_file(filepath)
    if shape is None:
        return []
    
    faces = extract_faces(shape)
    print(f"총 {len(faces)}개 Face 발견")
    
    dimensions = []
    for i, face in enumerate(faces):
        dim = compute_face_dimension(face, i, eps_mm=eps_mm, num_samples=num_samples, detailed=detailed)
        dimensions.append(dim)
        
        if not dim.is_sliver:
            inner_flag = " [RING]" if dim.has_inner_trim else ""
            print(f"  Face {i}: {dim.surface_type}{inner_flag}, "
                  f"width={dim.width_stable:.2f}mm, height={dim.height_stable:.2f}mm")
    
    return dimensions


# ============================================================================
# Output Functions
# ============================================================================

def print_detailed_analysis(dimensions: List[FaceDimension], verbose: bool = True):
    """상세 분석 결과 출력"""
    print("\n" + "=" * 100)
    print("UV 기반 Face 폭/너비 분석 결과 (트림 반영, max segment 기준)")
    print("=" * 100)
    
    for dim in dimensions:
        if dim.is_sliver:
            continue
        
        print(f"\n{'─' * 80}")
        inner_flag = " [내부 구멍 있음]" if dim.has_inner_trim else ""
        print(f"Face {dim.face_id}: {dim.surface_type}{inner_flag}")
        print(f"{'─' * 80}")
        
        # UV 범위 (트림 반영)
        print(f"  UV 범위 (트림 반영):")
        print(f"    U: [{dim.u_min:.4f}, {dim.u_max:.4f}] (범위: {dim.u_max - dim.u_min:.4f})")
        print(f"    V: [{dim.v_min:.4f}, {dim.v_max:.4f}] (범위: {dim.v_max - dim.v_min:.4f})")
        
        # 주기성
        if dim.is_periodic_u or dim.is_periodic_v:
            periodic_str = []
            if dim.is_periodic_u:
                periodic_str.append("U (주기면)")
            if dim.is_periodic_v:
                periodic_str.append("V (주기면)")
            print(f"  주기성: {', '.join(periodic_str)}")
        
        # 폭 계산 결과
        print(f"\n  폭 (Width) - U 방향 max segment:")
        print(f"    중앙값 ({len(dim.width_samples)}개 샘플): {dim.width_stable:.4f} mm")
        if dim.width_samples:
            print(f"    샘플 값: {[f'{w:.2f}' for w in dim.width_samples]}")
            if len(dim.width_samples) > 1:
                std_dev = statistics.stdev(dim.width_samples)
                print(f"    표준편차: {std_dev:.4f} mm")
        
        if dim.width_trim_result and verbose:
            result = dim.width_trim_result
            print(f"    트림 분석:")
            print(f"      Inside 구간 수: {result.num_intervals}")
            print(f"      Coarse 샘플 수: {result.coarse_samples}")
            print(f"      총 길이: {result.total_length:.4f} mm")
            print(f"      최대 구간: {result.max_segment_length:.4f} mm")
            if result.intervals:
                print(f"    Inside 구간들:")
                for j, iv in enumerate(result.intervals):
                    print(f"      [{iv.start:.4f}, {iv.end:.4f}] → 길이={iv.length_3d:.4f}mm")
        
        # 너비 계산 결과
        print(f"\n  너비 (Height) - V 방향 max segment:")
        print(f"    중앙값 ({len(dim.height_samples)}개 샘플): {dim.height_stable:.4f} mm")
        if dim.height_samples:
            print(f"    샘플 값: {[f'{h:.2f}' for h in dim.height_samples]}")
            if len(dim.height_samples) > 1:
                std_dev = statistics.stdev(dim.height_samples)
                print(f"    표준편차: {std_dev:.4f} mm")
        
        if dim.height_trim_result and verbose:
            result = dim.height_trim_result
            print(f"    트림 분석:")
            print(f"      Inside 구간 수: {result.num_intervals}")
            print(f"      Coarse 샘플 수: {result.coarse_samples}")
            print(f"      총 길이: {result.total_length:.4f} mm")
            print(f"      최대 구간: {result.max_segment_length:.4f} mm")
            if result.intervals:
                print(f"    Inside 구간들:")
                for j, iv in enumerate(result.intervals):
                    print(f"      [{iv.start:.4f}, {iv.end:.4f}] → 길이={iv.length_3d:.4f}mm")
        
        # 종횡비
        print(f"\n  종횡비 (Width/Height): {dim.aspect_ratio:.4f}")
    
    print("\n" + "=" * 100)


def print_summary_table(dimensions: List[FaceDimension]):
    """요약 테이블 출력"""
    valid_dims = [d for d in dimensions if not d.is_sliver]
    
    print("\n" + "=" * 120)
    print(f"{'Face':>5} | {'Type':^12} | {'Inner':^5} | {'U범위':^14} | {'V범위':^14} | "
          f"{'Width(mm)':>10} | {'Height(mm)':>10} | {'Aspect':>8} | {'Ivls':>5}")
    print("-" * 120)
    
    for d in valid_dims:
        u_range = f"[{d.u_min:.2f},{d.u_max:.2f}]"
        v_range = f"[{d.v_min:.2f},{d.v_max:.2f}]"
        inner = "RING" if d.has_inner_trim else ""
        
        n_ivl_w = d.width_trim_result.num_intervals if d.width_trim_result else "-"
        n_ivl_h = d.height_trim_result.num_intervals if d.height_trim_result else "-"
        ivls = f"{n_ivl_w}/{n_ivl_h}"
        
        print(f"{d.face_id:>5} | {d.surface_type:^12} | {inner:^5} | {u_range:^14} | {v_range:^14} | "
              f"{d.width_stable:>10.3f} | {d.height_stable:>10.3f} | {d.aspect_ratio:>8.3f} | {ivls:>5}")
    
    print("-" * 120)
    
    if valid_dims:
        widths = [d.width_stable for d in valid_dims]
        heights = [d.height_stable for d in valid_dims]
        print(f"{'통계':>5} | {'':^12} | {'':^5} | {'':^14} | {'':^14} | "
              f"{'min=' + f'{min(widths):.2f}':>10} | {'min=' + f'{min(heights):.2f}':>10} |")
        print(f"{'':>5} | {'':^12} | {'':^5} | {'':^14} | {'':^14} | "
              f"{'max=' + f'{max(widths):.2f}':>10} | {'max=' + f'{max(heights):.2f}':>10} |")
    
    print("=" * 120)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
    else:
        step_file = "generated_turning_models/turning_N6_H3_S2_G3_002.step"
    
    print(f"\n{'=' * 60}")
    print(f"분석 파일: {step_file}")
    print(f"A안: 트림 경계 반영 + max segment 기준")
    print(f"{'=' * 60}")
    
    shape = load_step_file(step_file)
    if shape:
        faces = extract_faces(shape)
        print(f"총 {len(faces)}개 Face 발견\n")
        
        dimensions = []
        for i, face in enumerate(faces):
            dim = compute_face_dimension(face, i, eps_mm=0.1, num_samples=5, detailed=True)
            dimensions.append(dim)
        
        # 요약 테이블
        print_summary_table(dimensions)
        
        # 상세 분석
        print("\n상세 분석 (유효 Face만):")
        print_detailed_analysis(dimensions, verbose=True)
