# -*- coding: utf-8 -*-
"""
label_maker.py 테스트 모듈

라벨 상수 정합성 및 LabelMaker의 라벨 전파 규칙 검증.

테스트 실행:
    pytest tests/test_label_maker.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

from core.design_operation import DesignOperation, collect_faces
from core.label_maker import LabelMaker, Labels


# ============================================================================
# 헬퍼
# ============================================================================

def make_stock(radius=10.0, height=20.0):
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def make_cut_tool(radius=3.0, z_offset=10.0):
    """옆에서 파는 홀 도구"""
    axis = gp_Ax2(gp_Pnt(10, 0, z_offset), gp_Dir(-1, 0, 0))
    return BRepPrimAPI_MakeCylinder(axis, radius, 15.0).Shape()


def make_step_tool(stock_radius=10.0, new_radius=7.0, z_min=0.0, height=5.0):
    """Step 커팅 도구 (링 형태)"""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    axis = gp_Ax2(gp_Pnt(0, 0, z_min), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(axis, stock_radius, height).Shape()
    inner = BRepPrimAPI_MakeCylinder(axis, new_radius, height).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


# ============================================================================
# Labels 상수 테스트
# ============================================================================

class TestLabels:
    def test_label_ids_are_sequential(self):
        """라벨 ID가 0부터 순차적"""
        assert Labels.STOCK == 0
        assert Labels.STEP == 1
        assert Labels.GROOVE == 2
        assert Labels.CHAMFER == 3
        assert Labels.FILLET == 4
        assert Labels.BLIND_HOLE == 5
        assert Labels.THROUGH_HOLE == 6
        assert Labels.RECTANGULAR_POCKET == 7
        assert Labels.RECTANGULAR_PASSAGE == 8

    def test_names_count_matches_max_id(self):
        """NAMES 리스트 길이 = 최대 ID + 1"""
        max_id = Labels.RECTANGULAR_PASSAGE
        assert len(Labels.NAMES) == max_id + 1

    def test_names_index_matches_constant(self):
        """NAMES[id]가 해당 라벨 이름과 일치"""
        assert Labels.NAMES[Labels.STOCK] == "stock"
        assert Labels.NAMES[Labels.STEP] == "step"
        assert Labels.NAMES[Labels.GROOVE] == "groove"
        assert Labels.NAMES[Labels.CHAMFER] == "chamfer"
        assert Labels.NAMES[Labels.FILLET] == "fillet"
        assert Labels.NAMES[Labels.BLIND_HOLE] == "blind_hole"
        assert Labels.NAMES[Labels.THROUGH_HOLE] == "through_hole"
        assert Labels.NAMES[Labels.RECTANGULAR_POCKET] == "rectangular_pocket"
        assert Labels.NAMES[Labels.RECTANGULAR_PASSAGE] == "rectangular_passage"


# ============================================================================
# LabelMaker 초기화 테스트
# ============================================================================

class TestLabelMakerInitialize:
    def test_initialize_labels_all_faces(self):
        """initialize 후 모든 face에 base_label 부여"""
        stock = make_stock()
        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)

        faces = collect_faces(stock)
        assert lm.get_total_faces() == len(faces)

    def test_initialize_default_label_is_zero(self):
        """기본 base_label은 0 (STOCK)"""
        stock = make_stock()
        lm = LabelMaker()
        lm.initialize(stock)

        for label in lm.labeled_faces.values():
            assert label == 0

    def test_initialize_custom_label(self):
        """커스텀 base_label 지정"""
        stock = make_stock()
        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STEP)

        for label in lm.labeled_faces.values():
            assert label == Labels.STEP

    def test_get_label_counts_after_initialize(self):
        """초기화 후 라벨 카운트 = {stock: 면 수}"""
        stock = make_stock()
        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)

        counts = lm.get_label_counts()
        assert "stock" in counts
        assert counts["stock"] == 3


# ============================================================================
# LabelMaker 라벨 전파 테스트
# ============================================================================

class TestLabelMakerUpdateLabel:
    def test_update_creates_generated_labels(self):
        """Cut 후 generated faces에 새 라벨 부여"""
        stock = make_stock()
        tool = make_cut_tool()

        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)

        op = DesignOperation(stock)
        result = op.cut(tool)
        assert result is not None

        lm.update_label(op, Labels.BLIND_HOLE)

        counts = lm.get_label_counts()
        assert "blind_hole" in counts
        assert counts["blind_hole"] > 0

    def test_update_preserves_modified_labels(self):
        """Cut 후 modified faces는 원본 라벨 유지"""
        stock = make_stock()
        tool = make_cut_tool()

        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)

        op = DesignOperation(stock)
        result = op.cut(tool)
        assert result is not None

        lm.update_label(op, Labels.BLIND_HOLE)

        counts = lm.get_label_counts()
        assert "stock" in counts, "Modified faces should retain STOCK label"

    def test_total_faces_after_update(self):
        """update 후 total_faces = modified + generated"""
        stock = make_stock()
        tool = make_cut_tool()

        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)
        initial_total = lm.get_total_faces()

        op = DesignOperation(stock)
        result = op.cut(tool)
        assert result is not None

        lm.update_label(op, Labels.BLIND_HOLE)

        result_faces = collect_faces(result)
        assert lm.get_total_faces() == len(result_faces)

    def test_sequential_updates_accumulate_labels(self):
        """여러 번 연산 → 여러 라벨이 누적"""
        stock = make_stock(radius=10.0, height=20.0)

        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)

        # 1차: Step 커팅
        step_tool = make_step_tool(stock_radius=10.0, new_radius=7.0, z_min=15.0, height=5.0)
        op1 = DesignOperation(stock)
        shape_after_step = op1.cut(step_tool)
        assert shape_after_step is not None
        lm.update_label(op1, Labels.STEP)

        counts_after_step = lm.get_label_counts()
        assert "stock" in counts_after_step
        assert "step" in counts_after_step

        # 2차: Hole 커팅
        hole_tool = make_cut_tool(radius=2.0, z_offset=8.0)
        op2 = DesignOperation(shape_after_step)
        shape_after_hole = op2.cut(hole_tool)
        assert shape_after_hole is not None
        lm.update_label(op2, Labels.BLIND_HOLE)

        counts_final = lm.get_label_counts()
        assert "stock" in counts_final
        assert "step" in counts_final
        assert "blind_hole" in counts_final

    def test_step_label_persists_after_hole(self):
        """Step 라벨 → Hole 추가 후에도 Step 라벨이 유지"""
        stock = make_stock(radius=10.0, height=20.0)

        lm = LabelMaker()
        lm.initialize(stock, base_label=Labels.STOCK)

        step_tool = make_step_tool(stock_radius=10.0, new_radius=7.0, z_min=15.0, height=5.0)
        op1 = DesignOperation(stock)
        shape1 = op1.cut(step_tool)
        assert shape1 is not None
        lm.update_label(op1, Labels.STEP)

        step_count_before = lm.get_label_counts().get("step", 0)

        hole_tool = make_cut_tool(radius=1.5, z_offset=17.0)
        op2 = DesignOperation(shape1)
        shape2 = op2.cut(hole_tool)
        assert shape2 is not None
        lm.update_label(op2, Labels.BLIND_HOLE)

        step_count_after = lm.get_label_counts().get("step", 0)
        assert step_count_after >= step_count_before, \
            f"Step 라벨이 감소함: {step_count_before} → {step_count_after}"


# ============================================================================
# LabelMaker 엣지 케이스
# ============================================================================

class TestLabelMakerEdgeCases:
    def test_empty_label_maker(self):
        """초기화 전 상태"""
        lm = LabelMaker()
        assert lm.get_total_faces() == 0
        assert lm.get_label_counts() == {}

    def test_reinitialize_clears_previous(self):
        """재초기화하면 이전 라벨 삭제"""
        stock1 = make_stock(radius=10.0)
        stock2 = make_stock(radius=5.0)

        lm = LabelMaker()
        lm.initialize(stock1, base_label=Labels.STOCK)
        count1 = lm.get_total_faces()

        lm.initialize(stock2, base_label=Labels.STEP)
        count2 = lm.get_total_faces()

        assert count2 == 3
        counts = lm.get_label_counts()
        assert "stock" not in counts
        assert "step" in counts

    def test_get_label_counts_unknown_label(self):
        """정의되지 않은 라벨 ID 처리"""
        stock = make_stock()
        lm = LabelMaker()
        lm.initialize(stock, base_label=99)

        counts = lm.get_label_counts()
        assert "unknown_99" in counts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
