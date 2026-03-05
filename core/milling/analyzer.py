"""
밀링 특징형상 배치 분석 및 계획 (MillingAnalyzer)

FaceAnalyzer를 사용하여 형상의 면을 분석하고
각 면에 배치 가능한 MillingFeatureRequest 목록을 생성한다.

클래스인 이유: params 상태 보유 + 배치된 피처 간 간격 관리 필요
"""

import random
from typing import List, Optional, Tuple

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.milling.face_analyzer import FaceAnalyzer, FaceDimensionResult
from core.milling.features import (
    MillingParams,
    MillingFeatureRequest,
    compute_hole_scale_range,
    compute_feature_center_cylinder,
    compute_feature_center_cone,
)
from core.label_maker import Labels


class MillingAnalyzer:
    """밀링 특징형상 배치 분석 및 계획.

    사용법:
        analyzer = MillingAnalyzer(MillingParams())
        requests = analyzer.analyze(shape, max_features=5, features_per_face=1)
    """

    def __init__(self, params: MillingParams = None):
        self.params = params or MillingParams()
        self._face_analyzer = FaceAnalyzer()
        self._placed_requests: List[MillingFeatureRequest] = []

    def analyze(
        self,
        shape: TopoDS_Shape,
        target_face_types: Optional[List[str]] = None,
        max_features: int = 5,
        features_per_face: int = 1,
    ) -> List[MillingFeatureRequest]:
        """면 분석 → 유효면 필터링 → 배치 계획 생성.

        Args:
            shape: 분석할 터닝 형상
            target_face_types: 대상 면 타입 (기본: ["Cylinder"])
            max_features: 총 최대 피처 수
            features_per_face: 면당 피처 수

        Returns:
            MillingFeatureRequest 목록
        """
        if target_face_types is None:
            target_face_types = ["Cylinder"]

        self._placed_requests = []

        topo = TopologyExplorer(shape)
        faces = list(topo.faces())
        dim_results = self._face_analyzer.analyze_shape(shape)

        print(f"  밀링 대상 면: {len(faces)}개")

        valid_face_infos = self._filter_valid_faces(
            faces, dim_results, target_face_types
        )

        if not valid_face_infos:
            print("  밀링 가능한 면 없음")
            return []

        print(f"  필터링된 대상 면: {len(valid_face_infos)}개")

        for info in valid_face_infos:
            dim = info['dim']
            w = f"{dim.width:.2f}" if dim.width else "N/A"
            h = f"{dim.height:.2f}" if dim.height else "N/A"
            d_range = f"[{info['d_min']:.2f}-{info['d_max']:.2f}]"
            print(f"    Face {info['face_id']} ({dim.surface_type}): W={w}, H={h}, D={d_range}")

        requests = []
        face_usage: dict = {}

        for info in valid_face_infos:
            if len(requests) >= max_features:
                break

            face_id = info['face_id']
            if face_usage.get(face_id, 0) >= self.params.max_features_per_face:
                continue

            for _ in range(features_per_face):
                if len(requests) >= max_features:
                    break
                if face_usage.get(face_id, 0) >= self.params.max_features_per_face:
                    break

                req = self._plan_feature_for_face(info)
                if req is not None:
                    requests.append(req)
                    self._placed_requests.append(req)
                    face_usage[face_id] = face_usage.get(face_id, 0) + 1

                    size_str = (
                        f"D={req.diameter:.2f}mm" if req.diameter > 0
                        else f"W={req.width:.2f}mm, L={req.length:.2f}mm"
                    )
                    through_str = "(관통)" if req.is_through else "(블라인드)"
                    print(
                        f"    피처 추가: Face {face_id}, {req.feature_type} {through_str}, "
                        f"{size_str}, depth={req.depth:.2f}mm"
                    )

        return requests

    # =========================================================================
    # 내부 헬퍼
    # =========================================================================

    def _filter_valid_faces(
        self,
        faces: List[TopoDS_Face],
        dim_results: List[FaceDimensionResult],
        target_types: List[str],
    ) -> List[dict]:
        valid = []
        for i, (face, dim) in enumerate(zip(faces, dim_results)):
            if "Plane" in dim.surface_type:
                continue
            if not any(t in dim.surface_type for t in target_types):
                continue

            d_min, d_max = compute_hole_scale_range(dim, self.params)
            if d_max < d_min or d_max <= 0:
                continue

            adaptor = BRepAdaptor_Surface(face, True)
            surf_type = adaptor.GetType()

            radius = 0.0
            if surf_type == GeomAbs_Cylinder:
                radius = adaptor.Cylinder().Radius()
            elif surf_type == GeomAbs_Cone:
                radius = (dim.r_max + dim.r_min) / 2

            valid.append({
                'face_id': i,
                'face': face,
                'dim': dim,
                'd_min': d_min,
                'd_max': d_max,
                'radius': radius,
                'r_outer': dim.r_outer if dim.r_outer > 0 else dim.r_max,
                'r_inner': dim.r_inner if dim.r_inner > 0 else dim.r_min,
            })

        return valid

    def _plan_feature_for_face(self, info: dict) -> Optional[MillingFeatureRequest]:
        """단일 면에 대한 피처 배치 계획 생성."""
        d_min = info['d_min']
        d_max = info['d_max']
        dim = info['dim']

        feature_types = ['blind_hole', 'through_hole', 'rect_pocket', 'rect_passage']
        feature_type = random.choice(feature_types)

        diameter = random.uniform(d_min, d_max)

        result = None
        if "Cylinder" in dim.surface_type:
            result = compute_feature_center_cylinder(info, diameter, self.params)
        elif "Cone" in dim.surface_type:
            result = compute_feature_center_cone(info, diameter, self.params)

        if result is None:
            return None

        center, direction, available_depth = result

        for existing in self._placed_requests:
            dist = center.Distance(existing.center)
            existing_size = (
                existing.diameter if existing.diameter > 0
                else max(existing.width, existing.length)
            )
            min_dist = (diameter + existing_size) / 2 + self.params.min_spacing
            if dist < min_dist:
                return None

        if feature_type == 'blind_hole':
            depth = diameter * self.params.depth_ratio
            return MillingFeatureRequest(
                feature_type='blind_hole',
                center=center,
                direction=direction,
                diameter=diameter,
                depth=depth,
                is_through=False,
                label=Labels.BLIND_HOLE,
                face_id=info['face_id'],
                face_type=dim.surface_type,
            )

        elif feature_type == 'through_hole':
            depth = available_depth + self.params.through_extra
            return MillingFeatureRequest(
                feature_type='through_hole',
                center=center,
                direction=direction,
                diameter=diameter,
                depth=depth,
                is_through=True,
                label=Labels.THROUGH_HOLE,
                face_id=info['face_id'],
                face_type=dim.surface_type,
            )

        elif feature_type == 'rect_pocket':
            aspect = random.uniform(self.params.rect_aspect_min, self.params.rect_aspect_max)
            width = diameter
            length = diameter * aspect
            depth = diameter * self.params.depth_ratio
            return MillingFeatureRequest(
                feature_type='rect_pocket',
                center=center,
                direction=direction,
                width=width,
                length=length,
                depth=depth,
                is_through=False,
                label=Labels.RECTANGULAR_POCKET,
                face_id=info['face_id'],
                face_type=dim.surface_type,
            )

        elif feature_type == 'rect_passage':
            aspect = random.uniform(self.params.rect_aspect_min, self.params.rect_aspect_max)
            width = diameter
            length = diameter * aspect
            depth = available_depth + self.params.through_extra
            return MillingFeatureRequest(
                feature_type='rect_passage',
                center=center,
                direction=direction,
                width=width,
                length=length,
                depth=depth,
                is_through=True,
                label=Labels.RECTANGULAR_PASSAGE,
                face_id=info['face_id'],
                face_type=dim.surface_type,
            )

        return None
