# -*- coding: utf-8 -*-
"""
tree_io.py 테스트 모듈

트리 I/O, 분류, 필터링, 통계 함수 검증.
OCC 의존성 없음 (순수 Python 테스트).

테스트 실행:
    pytest tests/test_tree_io.py -v
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tree_io import (
    load_trees, save_trees,
    classify_trees_by_step_count, classify_trees_by_groove_count,
    get_tree_stats, filter_trees,
    find_bidirectional_step_trees, find_sibling_groove_trees,
)


# ============================================================================
# 테스트 데이터
# ============================================================================

SAMPLE_TREES = [
    {
        "N": 3, "max_depth_constraint": 2, "canonical": "b(s,g)",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1, 2], "depth": 0},
            {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
            {"id": 2, "label": "g", "parent": 0, "children": [], "depth": 1},
        ]
    },
    {
        "N": 4, "max_depth_constraint": 3, "canonical": "b(s(s),g)",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1, 3], "depth": 0},
            {"id": 1, "label": "s", "parent": 0, "children": [2], "depth": 1},
            {"id": 2, "label": "s", "parent": 1, "children": [], "depth": 2},
            {"id": 3, "label": "g", "parent": 0, "children": [], "depth": 1},
        ]
    },
    {
        "N": 5, "max_depth_constraint": 3, "canonical": "b(s,s,g,g)",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1, 2, 3, 4], "depth": 0},
            {"id": 1, "label": "s", "parent": 0, "children": [], "depth": 1},
            {"id": 2, "label": "s", "parent": 0, "children": [], "depth": 1},
            {"id": 3, "label": "g", "parent": 0, "children": [], "depth": 1},
            {"id": 4, "label": "g", "parent": 0, "children": [], "depth": 1},
        ]
    },
    {
        "N": 3, "max_depth_constraint": 2, "canonical": "b(g(g))",
        "nodes": [
            {"id": 0, "label": "b", "parent": None, "children": [1], "depth": 0},
            {"id": 1, "label": "g", "parent": 0, "children": [2], "depth": 1},
            {"id": 2, "label": "g", "parent": 1, "children": [], "depth": 2},
        ]
    },
]


# ============================================================================
# load_trees / save_trees 테스트
# ============================================================================

class TestLoadSaveTrees:
    def test_save_and_load_roundtrip(self, tmp_path):
        """저장 후 로드하면 동일한 데이터"""
        filepath = tmp_path / "test_trees.json"
        save_trees(SAMPLE_TREES, str(filepath))

        loaded = load_trees(str(filepath))
        assert len(loaded) == len(SAMPLE_TREES)
        for orig, loaded_t in zip(SAMPLE_TREES, loaded):
            assert orig["N"] == loaded_t["N"]
            assert orig["canonical"] == loaded_t["canonical"]

    def test_save_creates_parent_dirs(self, tmp_path):
        """부모 디렉토리 자동 생성"""
        filepath = tmp_path / "sub" / "dir" / "trees.json"
        save_trees(SAMPLE_TREES, str(filepath))
        assert filepath.exists()

    def test_load_dict_with_trees_key(self, tmp_path):
        """{'trees': [...]} 형식 로드"""
        filepath = tmp_path / "trees.json"
        data = {"trees": SAMPLE_TREES}
        with open(filepath, 'w') as f:
            json.dump(data, f)

        loaded = load_trees(str(filepath))
        assert len(loaded) == len(SAMPLE_TREES)

    def test_load_bare_list(self, tmp_path):
        """[...] 형식 (리스트 직접) 로드"""
        filepath = tmp_path / "trees.json"
        with open(filepath, 'w') as f:
            json.dump(SAMPLE_TREES, f)

        loaded = load_trees(str(filepath))
        assert len(loaded) == len(SAMPLE_TREES)

    def test_load_nonexistent_file_raises(self):
        """존재하지 않는 파일 → FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_trees("nonexistent_path_12345.json")

    def test_load_invalid_format_raises(self, tmp_path):
        """잘못된 형식 → ValueError"""
        filepath = tmp_path / "bad.json"
        with open(filepath, 'w') as f:
            json.dump({"not_trees": "data"}, f)

        with pytest.raises(ValueError):
            load_trees(str(filepath))


# ============================================================================
# classify_trees 테스트
# ============================================================================

class TestClassifyTrees:
    def test_classify_by_step_count(self):
        result = classify_trees_by_step_count(SAMPLE_TREES)
        assert 1 in result
        assert 2 in result
        assert 0 in result
        assert 0 in result[1]
        assert 1 in result[2]

    def test_classify_by_groove_count(self):
        result = classify_trees_by_groove_count(SAMPLE_TREES)
        assert 1 in result
        assert 2 in result

    def test_classify_empty_list(self):
        result = classify_trees_by_step_count([])
        assert result == {}

    def test_classify_indices_are_valid(self):
        """반환된 인덱스가 원래 리스트 범위 내"""
        result = classify_trees_by_step_count(SAMPLE_TREES)
        for indices in result.values():
            for idx in indices:
                assert 0 <= idx < len(SAMPLE_TREES)

    def test_all_trees_classified(self):
        """모든 트리가 정확히 한 번씩 분류됨"""
        result = classify_trees_by_step_count(SAMPLE_TREES)
        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(SAMPLE_TREES)))


# ============================================================================
# get_tree_stats 테스트
# ============================================================================

class TestGetTreeStats:
    def test_basic_stats(self):
        stats = get_tree_stats(SAMPLE_TREES[0])
        assert stats['n_nodes'] == 3
        assert stats['s_count'] == 1
        assert stats['g_count'] == 1
        assert stats['b_count'] == 1
        assert stats['canonical'] == "b(s,g)"

    def test_bidirectional_step_detection(self):
        """Base에 Step 2개 → base_step_children == 2"""
        stats = get_tree_stats(SAMPLE_TREES[2])
        assert stats['base_step_children'] == 2

    def test_single_step_detection(self):
        stats = get_tree_stats(SAMPLE_TREES[0])
        assert stats['base_step_children'] == 1

    def test_sibling_grooves_detection(self):
        """Base에 Groove 2개 → has_sibling_grooves == True"""
        stats = get_tree_stats(SAMPLE_TREES[2])
        assert stats['has_sibling_grooves'] is True

    def test_no_sibling_grooves(self):
        stats = get_tree_stats(SAMPLE_TREES[0])
        assert stats['has_sibling_grooves'] is False

    def test_nested_grooves_not_siblings(self):
        """중첩 Groove는 형제가 아님"""
        stats = get_tree_stats(SAMPLE_TREES[3])
        assert stats['has_sibling_grooves'] is False


# ============================================================================
# find 함수 테스트
# ============================================================================

class TestFindFunctions:
    def test_find_bidirectional_step(self):
        result = find_bidirectional_step_trees(SAMPLE_TREES)
        assert 2 in result
        assert 0 not in result

    def test_find_sibling_grooves(self):
        result = find_sibling_groove_trees(SAMPLE_TREES)
        assert 2 in result
        assert 3 not in result

    def test_find_empty_list(self):
        assert find_bidirectional_step_trees([]) == []
        assert find_sibling_groove_trees([]) == []


# ============================================================================
# filter_trees 테스트
# ============================================================================

class TestFilterTrees:
    def test_filter_min_steps(self):
        result = filter_trees(SAMPLE_TREES, min_steps=2)
        for idx in result:
            stats = get_tree_stats(SAMPLE_TREES[idx])
            assert stats['s_count'] >= 2

    def test_filter_max_steps(self):
        result = filter_trees(SAMPLE_TREES, max_steps=1)
        for idx in result:
            stats = get_tree_stats(SAMPLE_TREES[idx])
            assert stats['s_count'] <= 1

    def test_filter_min_grooves(self):
        result = filter_trees(SAMPLE_TREES, min_grooves=2)
        for idx in result:
            stats = get_tree_stats(SAMPLE_TREES[idx])
            assert stats['g_count'] >= 2

    def test_filter_combined(self):
        result = filter_trees(SAMPLE_TREES, min_steps=1, max_grooves=1)
        for idx in result:
            stats = get_tree_stats(SAMPLE_TREES[idx])
            assert stats['s_count'] >= 1
            assert stats['g_count'] <= 1

    def test_filter_no_constraints_returns_all(self):
        result = filter_trees(SAMPLE_TREES)
        assert len(result) == len(SAMPLE_TREES)

    def test_filter_impossible_returns_empty(self):
        result = filter_trees(SAMPLE_TREES, min_steps=100)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
