"""
OCC Boolean/Fillet 연산 래퍼 - Face History 추적

OCC의 Boolean/Fillet API가 제공하는 History를 통해
원본 Face → 변형 Face 매핑을 추적합니다.

- Modified: 연산으로 형태가 변형된 Face (원본과 1:N 매핑)
- Generated: 연산으로 새로 생성된 Face (원본에 없던 면)
- Deleted: 연산으로 제거된 Face (IsDeleted)
"""

from typing import Dict, List, Optional

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, topods
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCC.Extend.TopologyUtils import TopologyExplorer


def collect_faces(shape: TopoDS_Shape) -> List[TopoDS_Face]:
    """형상의 모든 Face 수집."""
    return list(TopologyExplorer(shape).faces())


def search_same_face(
    target: TopoDS_Face, 
    faces: List[TopoDS_Face]
) -> Optional[TopoDS_Face]:
    """
    Face 목록에서 IsSame()으로 동일한 Face 검색.
    
    IsSame(): 동일한 TShape 참조 (위상적 동일성, Orientation 무관)
    IsEqual()과 달리 Orientation 차이를 허용하므로,
    Boolean 연산 후 face 매칭에 적합합니다.
    """
    for face in faces:
        if face.IsSame(target):
            return face
    return None


class DesignOperation:
    """
    OCC Boolean/Fillet 연산 래퍼.
    
    연산 실행 후 History API (Modified, IsDeleted)를 통해 
    Face 변형 정보를 추적합니다.
    
    Attributes:
        origin_faces: 연산 전 Face 목록
        modified_faces: {원본Face: [변형Face]} 매핑
        generated_faces: 새로 생성된 Face 목록
        processed_faces: 연산 후 전체 Face 목록
    """
    
    def __init__(self, shape: TopoDS_Shape):
        self.origin_shape = shape
        self.origin_faces: List[TopoDS_Face] = collect_faces(shape)
        self.modified_faces: Dict[TopoDS_Face, List[TopoDS_Face]] = {}
        self.generated_faces: List[TopoDS_Face] = []
        self.processed_faces: List[TopoDS_Face] = []
        self._builder = None
    
    def _update_face_props(self, processed_shape: TopoDS_Shape) -> None:
        """
        연산 후 Face 변형 정보 수집.
        
        1. processed_faces에서 시작 (모든 결과 face를 generated로 간주)
        2. origin_faces를 순회하며:
           - IsDeleted → 건너뜀
           - Modified 있음 → modified_faces에 추가, generated에서 제거
           - Modified 없음 → 원본 그대로 유지, generated에서 제거
        3. 남은 generated_faces = 완전히 새로 생성된 face
        """
        self.processed_faces = collect_faces(processed_shape)
        self.generated_faces = list(self.processed_faces)
        self.modified_faces = {}
        
        for face in self.origin_faces:
            if self._builder.IsDeleted(face):
                continue
            
            self.modified_faces[face] = []
            modified_list: TopTools_ListOfShape = self._builder.Modified(face)
            
            if modified_list.Size() == 0:
                # 변형 없음 - 원본 그대로 유지
                processed_face = search_same_face(face, self.processed_faces)
                if processed_face is not None:
                    self.modified_faces[face].append(processed_face)
                    target = search_same_face(processed_face, self.generated_faces)
                    if target is not None and target in self.generated_faces:
                        self.generated_faces.remove(target)
            else:
                # 변형됨 - Modified 목록 순회
                iterator = TopTools_ListIteratorOfListOfShape(modified_list)
                while iterator.More():
                    modified_face = topods.Face(iterator.Value())
                    processed_face = search_same_face(modified_face, self.processed_faces)
                    if processed_face is not None:
                        self.modified_faces[face].append(processed_face)
                        target = search_same_face(processed_face, self.generated_faces)
                        if target is not None and target in self.generated_faces:
                            self.generated_faces.remove(target)
                    iterator.Next()
    
    def get_modified_faces(self, face: TopoDS_Face) -> List[TopoDS_Face]:
        """원본 Face에 대한 변형된 Face 목록 반환."""
        matched = search_same_face(face, self.origin_faces)
        if matched is not None:
            return self.modified_faces.get(matched, [])
        return []
    
    def get_generated_faces(self) -> List[TopoDS_Face]:
        """새로 생성된 Face 목록 반환."""
        return self.generated_faces
    
    def cut(self, tool_shape: TopoDS_Shape) -> Optional[TopoDS_Shape]:
        """Boolean Cut 수행 + History 추적."""
        self._builder = BRepAlgoAPI_Cut(self.origin_shape, tool_shape)
        self._builder.Build()
        
        if not self._builder.IsDone():
            return None
        
        result = self._builder.Shape()
        if result is None or result.IsNull():
            return None
        
        self._update_face_props(result)
        return result
    
    def chamfer(self, edge: TopoDS_Edge, distance: float) -> Optional[TopoDS_Shape]:
        """Chamfer 수행 + History 추적."""
        self._builder = BRepFilletAPI_MakeChamfer(self.origin_shape)
        self._builder.Add(distance, edge)
        self._builder.Build()
        
        if not self._builder.IsDone():
            return None
        
        result = self._builder.Shape()
        if result is None or result.IsNull():
            return None
        
        self._update_face_props(result)
        return result
    
    def fillet(self, edge: TopoDS_Edge, radius: float) -> Optional[TopoDS_Shape]:
        """Fillet 수행 + History 추적."""
        self._builder = BRepFilletAPI_MakeFillet(self.origin_shape)
        self._builder.Add(radius, edge)
        self._builder.Build()
        
        if not self._builder.IsDone():
            return None
        
        result = self._builder.Shape()
        if result is None or result.IsNull():
            return None
        
        self._update_face_props(result)
        return result
