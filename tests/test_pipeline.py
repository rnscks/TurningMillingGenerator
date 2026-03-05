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

from core.tree.generator import generate_trees
from core.label_maker import Labels
from pipeline import TurningMillingGenerator, TurningMillingParams
from core import TurningParams, MillingParams


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
        assert isinstance(params.milling, MillingParams)
        assert params.enable_milling is True
        assert params.enable_labeling is False
        assert params.target_face_types == ["Cylinder"]

    def test_custom_params(self):
        params = TurningMillingParams(
            enable_milling=False,
            enable_labeling=True,
            max_features=10,
        )
        assert params.enable_milling is False
        assert params.enable_labeling is True
        assert params.max_features == 10


# ============================================================================
# 파이프라인 기본 동작 테스트
# ============================================================================

class TestPipelineBasic:
    def test_generate_turning_only(self):
        """밀링 비활성 → 터닝 형상만 생성"""
        trees = get_sample_trees()
        params = TurningMillingParams(enable_milling=False)
        gen = TurningMillingGenerator(params)

        shape, requests = gen.generate_from_tree(trees[0], apply_edge_feats=False)

        assert shape is not None
        assert not shape.IsNull()
        assert requests == []

    def test_generate_with_milling(self):
        """밀링 활성 → 터닝 + 밀링 피처"""
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=True,
            max_features=2,
            features_per_face=1,
            target_face_types=["Cylinder"],
        )
        gen = TurningMillingGenerator(params)

        shape, requests = gen.generate_from_tree(trees[0], apply_edge_feats=False)

        assert shape is not None
        assert not shape.IsNull()
        assert isinstance(requests, list)

    def test_multiple_trees_generate_successfully(self):
        """여러 트리에서 형상 생성 성공률"""
        trees = get_sample_trees(n_nodes=5, max_depth=3, count=5)
        params = TurningMillingParams(enable_milling=False)

        success = 0
        for tree in trees:
            gen = TurningMillingGenerator(params)
            try:
                shape, _ = gen.generate_from_tree(tree, apply_edge_feats=False)
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

        gen.generate_from_tree(trees[0], apply_edge_feats=False)

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

        gen.generate_from_tree(trees[0], apply_edge_feats=False)

        assert gen.label_maker is None

    def test_labeling_has_stock_label(self):
        """라벨링 시 stock 라벨 존재"""
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=False,
            enable_labeling=True,
        )
        gen = TurningMillingGenerator(params)

        gen.generate_from_tree(trees[0], apply_edge_feats=False)

        counts = gen.label_maker.get_label_counts()
        assert "stock" in counts

    def test_labeling_with_milling_adds_feature_labels(self):
        """밀링 + 라벨링 → 피처 라벨 추가됨"""
        trees = get_sample_trees(n_nodes=4)
        params = TurningMillingParams(
            enable_milling=True,
            enable_labeling=True,
            max_features=3,
            features_per_face=1,
            target_face_types=["Cylinder"],
        )
        gen = TurningMillingGenerator(params)

        shape, requests = gen.generate_from_tree(trees[0], apply_edge_feats=False)

        if requests:
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

        gen.generate_from_tree(trees[0], apply_edge_feats=False)
        info = gen.get_generation_info()

        assert "stock_height" in info
        assert "stock_radius" in info
        assert "n_milling_features" in info
        assert "milling_features" in info
        assert info["stock_height"] > 0
        assert info["stock_radius"] > 0
        assert info["n_milling_features"] == 0

    def test_info_with_milling_features(self):
        trees = get_sample_trees()
        params = TurningMillingParams(
            enable_milling=True,
            max_features=2,
            target_face_types=["Cylinder"],
        )
        gen = TurningMillingGenerator(params)

        shape, requests = gen.generate_from_tree(trees[0], apply_edge_feats=False)
        info = gen.get_generation_info()

        assert info["n_milling_features"] == len(requests)
        assert len(info["milling_features"]) == len(requests)

        for feat in info["milling_features"]:
            assert "face_id" in feat
            assert "feature_type" in feat
            assert "depth" in feat
            assert "center" in feat
            assert "direction" in feat
            assert len(feat["center"]) == 3
            assert len(feat["direction"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
