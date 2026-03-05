"""
STEP 파일 입출력 공통 모듈
"""

from typing import Optional, Dict
from pathlib import Path

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.STEPConstruct import stepconstruct
from OCC.Core.TopLoc import TopLoc_Location


def load_step(filepath: str) -> Optional[TopoDS_Shape]:
    """
    STEP 파일 로드.
    
    Args:
        filepath: STEP 파일 경로
        
    Returns:
        TopoDS_Shape 또는 실패시 None
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(filepath))
    
    if status == IFSelect_RetDone:
        reader.TransferRoots()
        return reader.OneShape()
    else:
        print(f"STEP 파일 로드 실패: {filepath}")
        return None


def save_step(shape: TopoDS_Shape, filepath: str) -> bool:
    """
    형상을 STEP 파일로 저장.
    
    Args:
        shape: 저장할 형상
        filepath: 저장 경로
        
    Returns:
        성공 여부
    """
    # 디렉토리 생성
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    
    status = writer.Write(str(filepath))
    
    if status == IFSelect_RetDone:
        print(f"저장 완료: {filepath}")
        return True
    else:
        print(f"저장 실패: {filepath}")
        return False


def save_labeled_step(
    shape: TopoDS_Shape,
    face_labels: Dict[TopoDS_Face, int],
    filepath: str
) -> bool:
    """
    라벨이 포함된 STEP 파일 저장.
    
    각 Face의 라벨을 STEP 엔티티의 Name 속성에 기록합니다.
    
    Args:
        shape: 저장할 형상
        face_labels: {Face: label_id} 매핑
        filepath: 저장 경로
        
    Returns:
        성공 여부
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    
    finderp = writer.WS().TransferWriter().FinderProcess()
    loc = TopLoc_Location()
    
    labeled_count = 0
    for face, label in face_labels.items():
        item = stepconstruct.FindEntity(finderp, face, loc)
        if item is not None:
            item.SetName(TCollection_HAsciiString(str(label)))
            labeled_count += 1
    
    status = writer.Write(str(filepath))
    
    if status == IFSelect_RetDone:
        print(f"라벨 STEP 저장 완료: {filepath} ({labeled_count}/{len(face_labels)} faces labeled)")
        return True
    else:
        print(f"라벨 STEP 저장 실패: {filepath}")
        return False
