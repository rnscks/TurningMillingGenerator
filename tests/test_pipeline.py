# -*- coding: utf-8 -*-
"""
pipeline.py 통합 테스트 모듈

전체 파이프라인 (트리 → 터닝 → 밀링 → 라벨링) E2E 검증.

테스트 실행:
    pytest tests/test_pipeline.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tree_generator import generate_trees
from core.label_maker import Labels
from pipeline import TurningMillingGenerator, TurningMillingParams
from core import TurningParams, FeatureParams


# ============================================================================
# 헬퍼
# ============================================================================

def get_sample_trees(n_nodes=4, max_depth=3, count=3):
    trees = generate_trees(n_nodes=n_nodes, max_depth=max_depth)
    return trees[:count]


# ============================================================================
# TurningMillingParams 테스트
# ============================================================================

class TestTurningMillingParams:
    def test_default_params(self):
        params = TurningMillingParams()
        assert isinstance(params.turning, TurningParams)
        assert isinstance(params.feature, FeatureParams)
        assert params.enable_milling is True
        assert params.enable_labeling is False
        assert params.target_face_types == ["Plane", "Cylinder"]

    def test_custom_params(self):
        params = TurningMillingParams(
            enable_milling=False,
            enable_labeling=True,
            max_holes=10,
        )
        assert params.enable_milling is False
        assert params.enable_labeling is True
        assert params.max_holes == 10


# ============================================================================
# 파이프라인 기본 동작 테스트
# ============================================================================

class TestPipelineBasic:
    def test_generate_turning_only(self):
        """밀링 비활성 → 터닝 형상만 생성"""
        trees = get_sample_trees()
        params = TurningMillingParams(enable_milling=False)
        gen = TurningMillingGenerator(params)

        shape, placements = gen.generate_from_tree(trees[0], apply_edge_features=False)

        assert shape is not None
        assert not shape.IsNull()
        assert placements == []

    def test_generate_with_milling(self):
        """밀링 활성 → 터닝 + 밀링 피처"""
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=True,
            max_holes=2,
            holes_per_face=1,
            target_face_types=["Cylinder"],
        )
        gen = TurningMillingGenerator(params)

        shape, placements = gen.generate_from_tree(trees[0], apply_edge_features=False)

        assert shape is not None
        assert not shape.IsNull()
        assert isinstance(placements, list)

    def test_multiple_trees_generate_successfully(self):
        """여러 트리에서 형상 생성 성공률"""
        trees = get_sample_trees(n_nodes=5, max_depth=3, count=5)
        params = TurningMillingParams(enable_milling=False)

        success = 0
        for tree in trees:
            gen = TurningMillingGenerator(params)
            try:
                shape, _ = gen.generate_from_tree(tree, apply_edge_features=False)
                if shape and not shape.IsNull():
                    success += 1
            except Exception:
                pass

        assert success >= len(trees) * 0.8


# ============================================================================
# 라벨링 통합 테스트
# ============================================================================

class TestPipelineLabeling:
    def test_labeling_enabled(self):
        """enable_labeling=True → label_maker 생성"""
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=False,
            enable_labeling=True,
        )
        gen = TurningMillingGenerator(params)

        gen.generate_from_tree(trees[0], apply_edge_features=False)

        assert gen.label_maker is not None
        assert gen.label_maker.get_total_faces() > 0

    def test_labeling_disabled(self):
        """enable_labeling=False → label_maker는 None"""
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=False,
            enable_labeling=False,
        )
        gen = TurningMillingGenerator(params)

        gen.generate_from_tree(trees[0], apply_edge_features=False)

        assert gen.label_maker is None

    def test_labeling_has_stock_label(self):
        """라벨링 시 stock 라벨 존재"""
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=False,
            enable_labeling=True,
        )
        gen = TurningMillingGenerator(params)

        gen.generate_from_tree(trees[0], apply_edge_features=False)

        counts = gen.label_maker.get_label_counts()
        assert "stock" in counts

    def test_labeling_with_milling_adds_feature_labels(self):
        """밀링 + 라벨링 → 피처 라벨 추가됨"""
        trees = get_sample_trees(n_nodes=4)
        params = TurningMillingParams(
            enable_milling=True,
            enable_labeling=True,
            max_holes=3,
            holes_per_face=1,
            target_face_types=["Cylinder"],
        )
        gen = TurningMillingGenerator(params)

        shape, placements = gen.generate_from_tree(trees[0], apply_edge_features=False)

        if placements:
            counts = gen.label_maker.get_label_counts()
            label_names = set(counts.keys())
            feature_labels = {"blind_hole", "through_hole", "rectangular_pocket", "rectangular_passage"}
            has_feature = bool(label_names & feature_labels)
            assert has_feature, f"피처 라벨이 없음: {counts}"


# ============================================================================
# get_generation_info 테스트
# ============================================================================

class TestGetGenerationInfo:
    def test_info_fields_exist(self):
        trees = get_sample_trees()
        params = TurningMillingParams(enable_milling=False)
        gen = TurningMillingGenerator(params)

        gen.generate_from_tree(trees[0], apply_edge_features=False)
        info = gen.get_generation_info()

        assert "stock_height" in info
        assert "stock_radius" in info
        assert "n_holes" in info
        assert "holes" in info
        assert info["stock_height"] > 0
        assert info["stock_radius"] > 0
        assert info["n_holes"] == 0

    def test_info_with_holes(self):
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=True,
            max_holes=2,
            target_face_types=["Cylinder"],
        )
        gen = TurningMillingGenerator(params)

        shape, placements = gen.generate_from_tree(trees[0], apply_edge_features=False)
        info = gen.get_generation_info()

        assert info["n_holes"] == len(placements)
        assert len(info["holes"]) == len(placements)

        for h in info["holes"]:
            assert "face_id" in h
            assert "diameter" in h
            assert "depth" in h
            assert "center" in h
            assert "direction" in h
            assert len(h["center"]) == 3
            assert len(h["direction"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
