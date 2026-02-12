"""
Face 라벨링 관리 모듈

형상 생성 시 각 Face에 특징형상 라벨을 자동으로 부여합니다.

라벨 전파 규칙:
- Modified faces → 원본 라벨 유지 (형태만 변형)
- Generated faces → 새 특징형상 라벨 (Boolean Cut으로 노출된 면)
- Deleted faces → 매핑에서 자연 제거
"""

from typing import Dict, Optional

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face

from core.design_operation import DesignOperation, collect_faces


class Labels:
    """라벨 상수 (LABEL_PROPS.json의 LABEL_NAMES 인덱스와 일치)"""
    STOCK = 0
    STEP = 1
    GROOVE = 2
    CHAMFER = 3
    FILLET = 4
    BLIND_HOLE = 5
    THROUGH_HOLE = 6
    RECTANGULAR_POCKET = 7
    
    NAMES = [
        "stock", "step", "groove", "chamfer", "fillet",
        "blind_hole", "through_hole", "rectangular_pocket"
    ]


class LabelMaker:
    """
    Face-라벨 매핑 관리.
    
    매 DesignOperation 후 update_label()을 호출하면
    Modified/Generated/Deleted 정보를 기반으로 라벨을 전파합니다.
    
    사용법:
        label_maker = LabelMaker()
        label_maker.initialize(stock_shape, base_label=Labels.STOCK)
        
        # 각 DesignOperation 수행 후:
        label_maker.update_label(operation, Labels.STEP)
    """
    
    def __init__(self):
        self.labeled_faces: Dict[TopoDS_Face, int] = {}
    
    def initialize(self, stock_shape: TopoDS_Shape, base_label: int = 0) -> None:
        """Stock 형상의 모든 Face에 기본 라벨 부여."""
        faces = collect_faces(stock_shape)
        self.labeled_faces = {face: base_label for face in faces}
    
    def update_label(self, operation: DesignOperation, label: int) -> None:
        """
        DesignOperation 후 라벨 갱신.
        
        1. Modified faces → 원본 라벨 유지
        2. Generated faces → 새 라벨 부여
        3. Deleted faces → 자연 제거 (modified에 미포함)
        
        Args:
            operation: 완료된 DesignOperation (History 정보 보유)
            label: Generated faces에 부여할 라벨
        """
        new_labeled_faces: Dict[TopoDS_Face, int] = {}
        
        # 1. Modified faces: 원본 라벨 유지
        for original_face, original_label in self.labeled_faces.items():
            modified_faces = operation.get_modified_faces(original_face)
            for modified_face in modified_faces:
                if modified_face not in new_labeled_faces:
                    new_labeled_faces[modified_face] = original_label
        
        # 2. Generated faces: 새 라벨 부여
        for new_face in operation.get_generated_faces():
            if new_face not in new_labeled_faces:
                new_labeled_faces[new_face] = label
        
        self.labeled_faces = new_labeled_faces
    
    def get_label_counts(self) -> Dict[str, int]:
        """라벨별 Face 개수 반환."""
        counts: Dict[str, int] = {}
        for label in self.labeled_faces.values():
            name = Labels.NAMES[label] if label < len(Labels.NAMES) else f"unknown_{label}"
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def get_total_faces(self) -> int:
        """총 Face 개수."""
        return len(self.labeled_faces)
