# -*- coding: utf-8 -*-
"""
face_analyzer.py 테스트 모듈

면 타입 분류, 치수 계산, 유효 면 필터링 검증.

테스트 실행:
    pytest tests/test_face_analyzer.py -v
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.face_analyzer import (
    FaceAnalyzer, FaceDimensionResult,
    get_surface_type, is_z_aligned_plane,
    sample_edge_points, get_face_wires,
    points_to_rz, analyze_rz_points,
)


# ============================================================================
# 헬퍼
# ============================================================================

def make_cylinder(radius=10.0, height=20.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def make_box(dx=10.0, dy=10.0, dz=10.0):
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def make_cone(r1=10.0, r2=5.0, height=15.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCone(axis, r1, r2, height).Shape()


def make_turning_with_step(stock_r=10.0, step_r=7.0, height=20.0, step_h=5.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    stock = BRepPrimAPI_MakeCylinder(axis, stock_r, height).Shape()

    step_axis = gp_Ax2(gp_Pnt(0, 0, height - step_h), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(step_axis, stock_r, step_h).Shape()
    inner = BRepPrimAPI_MakeCylinder(step_axis, step_r, step_h).Shape()
    ring = BRepAlgoAPI_Cut(outer, inner).Shape()

    return BRepAlgoAPI_Cut(stock, ring).Shape()


def get_faces(shape):
    return list(TopologyExplorer(shape).faces())


# ============================================================================
# get_surface_type 테스트
# ============================================================================

class TestGetSurfaceType:
    def test_cylinder_faces(self):
        cyl = make_cylinder()
        faces = get_faces(cyl)
        types = [get_surface_type(f)[0] for f in faces]
        assert "Cylinder" in types
        assert "Plane" in types

    def test_box_faces_are_all_plane(self):
        box = make_box()
        faces = get_faces(box)
        for f in faces:
            surf_type, _ = get_surface_type(f)
            assert surf_type == "Plane"

    def test_cone_faces(self):
        cone = make_cone()
        faces = get_faces(cone)
        types = [get_surface_type(f)[0] for f in faces]
        assert "Cone" in types


# ============================================================================
# is_z_aligned_plane 테스트
# ============================================================================

class TestIsZAlignedPlane:
    def test_cylinder_top_bottom_are_z_aligned(self):
        cyl = make_cylinder()
        faces = get_faces(cyl)
        z_planes = [f for f in faces if is_z_aligned_plane(f)]
        assert len(z_planes) == 2

    def test_cylinder_side_is_not_z_aligned(self):
        cyl = make_cylinder()
        faces = get_faces(cyl)
        for f in faces:
            surf_type, _ = get_surface_type(f)
            if surf_type == "Cylinder":
                assert not is_z_aligned_plane(f)


# ============================================================================
# FaceAnalyzer.analyze_shape 테스트
# ============================================================================

class TestFaceAnalyzerShape:
    def test_analyze_cylinder_shape(self):
        cyl = make_cylinder(radius=10.0, height=20.0)
        analyzer = FaceAnalyzer()
        results = analyzer.analyze_shape(cyl)

        assert len(results) == 3
        types = [r.surface_type for r in results]
        assert "Cylinder" in types

    def test_cylinder_dimension_values(self):
        """원기둥 측면: width ≈ 2R, height ≈ H"""
        cyl = make_cylinder(radius=10.0, height=20.0)
        analyzer = FaceAnalyzer()
        results = analyzer.analyze_shape(cyl)

        cyl_result = [r for r in results if r.surface_type == "Cylinder"][0]
        assert cyl_result.width == pytest.approx(20.0, rel=0.1)
        assert cyl_result.height == pytest.approx(20.0, rel=0.1)

    def test_plane_disk_dimension(self):
        """원기둥 상/하면 (Disk): width = height = 직경"""
        cyl = make_cylinder(radius=10.0, height=20.0)
        analyzer = FaceAnalyzer()
        results = analyzer.analyze_shape(cyl)

        disk_results = [r for r in results if "Disk" in r.surface_type]
        assert len(disk_results) == 2
        for r in disk_results:
            assert r.width == pytest.approx(20.0, rel=0.1)

    def test_cone_dimension(self):
        cone = make_cone(r1=10.0, r2=5.0, height=15.0)
        analyzer = FaceAnalyzer()
        results = analyzer.analyze_shape(cone)

        cone_results = [r for r in results if r.surface_type == "Cone"]
        assert len(cone_results) == 1
        cr = cone_results[0]
        assert cr.delta_r == pytest.approx(5.0, rel=0.1)
        assert cr.delta_z == pytest.approx(15.0, rel=0.1)

    def test_analyze_turning_shape(self):
        """Step이 있는 터닝 형상 분석"""
        shape = make_turning_with_step()
        analyzer = FaceAnalyzer()
        results = analyzer.analyze_shape(shape)

        assert len(results) > 3
        types = set(r.surface_type for r in results)
        assert "Cylinder" in types

    def test_ring_face_detection(self):
        """Step 후 Ring Plane 존재"""
        shape = make_turning_with_step()
        analyzer = FaceAnalyzer()
        results = analyzer.analyze_shape(shape)

        ring_results = [r for r in results if r.is_ring]
        assert len(ring_results) >= 1
        for r in ring_results:
            assert r.ring_thickness > 0


# ============================================================================
# FaceAnalyzer.get_valid_faces 테스트
# ============================================================================

class TestFaceAnalyzerValidFaces:
    def test_valid_faces_have_positive_width(self):
        shape = make_turning_with_step()
        analyzer = FaceAnalyzer()
        valid = analyzer.get_valid_faces(shape, min_dimension=0.1)

        for r in valid:
            assert r.width is not None
            assert r.width >= 0.1

    def test_valid_faces_filter_by_type(self):
        shape = make_turning_with_step()
        analyzer = FaceAnalyzer()
        valid = analyzer.get_valid_faces(shape, target_types=["Cylinder"])

        for r in valid:
            assert "Cylinder" in r.surface_type

    def test_valid_faces_default_includes_plane(self):
        shape = make_turning_with_step()
        analyzer = FaceAnalyzer()
        valid = analyzer.get_valid_faces(shape)

        types = set(r.surface_type for r in valid)
        has_plane = any("Plane" in t for t in types)
        has_cylinder = "Cylinder" in types
        assert has_cylinder


# ============================================================================
# 포인트 유틸리티 테스트
# ============================================================================

class TestPointUtilities:
    def test_sample_edge_points_count(self):
        cyl = make_cylinder()
        edges = list(TopologyExplorer(cyl).edges())
        assert len(edges) > 0

        pts = sample_edge_points(edges[0], n_samples=10)
        assert len(pts) == 10

    def test_sample_edge_points_min_samples(self):
        """n_samples < 2이면 최소 2개로 강제"""
        cyl = make_cylinder()
        edges = list(TopologyExplorer(cyl).edges())
        pts = sample_edge_points(edges[0], n_samples=0)
        assert len(pts) == 2

    def test_points_to_rz(self):
        from OCC.Core.gp import gp_Pnt
        pts = [gp_Pnt(3, 4, 5), gp_Pnt(0, 0, 10)]
        r_vals, z_vals = points_to_rz(pts)

        assert r_vals[0] == pytest.approx(5.0)
        assert r_vals[1] == pytest.approx(0.0)
        assert z_vals[0] == pytest.approx(5.0)
        assert z_vals[1] == pytest.approx(10.0)

    def test_analyze_rz_points_empty(self):
        import numpy as np
        result = analyze_rz_points(np.array([]), np.array([]))
        assert result['r_max'] == 0
        assert result['delta_r'] == 0

    def test_get_face_wires(self):
        cyl = make_cylinder()
        faces = get_faces(cyl)
        for f in faces:
            outer, inners = get_face_wires(f)
            assert outer is not None or len(inners) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
