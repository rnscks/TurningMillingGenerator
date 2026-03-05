# -*- coding: utf-8 -*-
"""
core/milling/analyzer.py 유닛 테스트

- MillingAnalyzer.analyze() → 올바른 FeatureRequest 생성
- 간격/크기 제약조건 검증
- 면 타입 필터링 동작

테스트 실행:
    pytest tests/milling/test_analyzer.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

from core.milling.analyzer import MillingAnalyzer
from core.milling.features import MillingParams, MillingFeatureRequest


# ============================================================================
# 헬퍼
# ============================================================================

def make_cylinder(radius=10.0, height=20.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def make_turning_shape(stock_radius=10.0, height=20.0, step_radius=7.0, step_height=5.0):
    """Step이 있는 터닝 형상"""
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    stock = BRepPrimAPI_MakeCylinder(axis, stock_radius, height).Shape()

    step_axis = gp_Ax2(gp_Pnt(0, 0, height - step_height), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(step_axis, stock_radius, step_height).Shape()
    inner = BRepPrimAPI_MakeCylinder(step_axis, step_radius, step_height).Shape()
    ring = BRepAlgoAPI_Cut(outer, inner).Shape()

    return BRepAlgoAPI_Cut(stock, ring).Shape()


# ============================================================================
# MillingAnalyzer 기본 테스트
# ============================================================================

class TestMillingAnalyzerBasic:
    def test_analyze_returns_list(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape)

        assert isinstance(requests, list)

    def test_analyze_max_features_respected(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()

        for max_f in [1, 2, 3]:
            requests = analyzer.analyze(shape, max_features=max_f)
            assert len(requests) <= max_f

    def test_analyze_returns_valid_requests(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=3)

        for req in requests:
            assert isinstance(req, MillingFeatureRequest)
            assert req.feature_type in (
                'blind_hole', 'through_hole', 'rect_pocket', 'rect_passage'
            )
            assert req.depth > 0

    def test_analyze_simple_cylinder(self):
        analyzer = MillingAnalyzer()
        shape = make_cylinder(10.0, 20.0)
        requests = analyzer.analyze(shape, max_features=3)

        assert isinstance(requests, list)


# ============================================================================
# 피처 크기 제약조건 테스트
# ============================================================================

class TestMillingAnalyzerConstraints:
    def test_blind_hole_depth_positive(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=5)

        for req in [r for r in requests if r.feature_type == 'blind_hole']:
            assert req.depth > 0
            assert req.diameter > 0

    def test_through_hole_depth_positive(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=5)

        for req in [r for r in requests if r.feature_type == 'through_hole']:
            assert req.depth > 0
            assert req.diameter > 0
            assert req.is_through is True

    def test_rect_pocket_dimensions_positive(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=5)

        for req in [r for r in requests if r.feature_type in ('rect_pocket', 'rect_passage')]:
            assert req.width > 0
            assert req.length > 0
            assert req.depth > 0

    def test_feature_spacing(self):
        """배치된 피처들 간 간격 검증 (analyze 후 내부 상태)"""
        params = MillingParams(min_spacing=2.0)
        analyzer = MillingAnalyzer(params)
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=5)

        for i, req_a in enumerate(requests):
            for j, req_b in enumerate(requests):
                if i >= j:
                    continue
                dist = req_a.center.Distance(req_b.center)
                size_a = req_a.diameter if req_a.diameter > 0 else max(req_a.width, req_a.length)
                size_b = req_b.diameter if req_b.diameter > 0 else max(req_b.width, req_b.length)
                min_dist = (size_a + size_b) / 2 + params.min_spacing
                assert dist >= min_dist - 1e-6


# ============================================================================
# face_type 필터 테스트
# ============================================================================

class TestMillingAnalyzerFaceFilter:
    def test_default_uses_cylinder_only(self):
        """기본 타겟은 Cylinder 측면만"""
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape)

        for req in requests:
            assert "Plane" not in req.face_type

    def test_no_valid_faces_returns_empty(self):
        """유효한 면이 없으면 빈 리스트"""
        params = MillingParams(diameter_min=1000.0)
        analyzer = MillingAnalyzer(params)
        shape = make_turning_shape()
        requests = analyzer.analyze(shape)

        assert requests == []


# ============================================================================
# face_analyzer.py 사용 검증 (face_type 할당)
# ============================================================================

class TestFaceAnalyzerIntegration:
    def test_requests_have_face_id(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=3)

        for req in requests:
            assert req.face_id >= 0

    def test_requests_have_label(self):
        analyzer = MillingAnalyzer()
        shape = make_turning_shape()
        requests = analyzer.analyze(shape, max_features=3)

        for req in requests:
            assert req.label > 0
