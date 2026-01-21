"""테스트 터닝 케이스의 면 타입 분석"""
from pathlib import Path
from projection_face_dimension import (
    load_step_file, extract_faces, get_surface_type, compute_face_dimension
)

# 테스트 파일 목록
model_dir = Path("generated_turning_models")
step_files = sorted(model_dir.glob("*.step"))

print(f"Total {len(step_files)} test files found\n")

# 각 파일의 면 타입 분석
for step_file in step_files[:3]:  # 처음 3개만
    print(f"{'='*60}")
    print(f"File: {step_file.name}")
    print(f"{'='*60}")
    
    shape = load_step_file(str(step_file))
    if shape is None:
        continue
    
    faces = extract_faces(shape)
    print(f"Total faces: {len(faces)}")
    
    # 타입별 카운트
    types = {}
    for i, face in enumerate(faces):
        surf_type, _ = get_surface_type(face)
        result = compute_face_dimension(face, i)
        
        # 링 여부 반영
        if result.is_ring:
            surf_type = f"{surf_type} (Ring)"
        
        types[surf_type] = types.get(surf_type, 0) + 1
    
    print(f"\nFace types:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")
    
    # 토러스가 있는지 체크
    has_torus = any('Torus' in t for t in types.keys())
    print(f"\nHas Torus (round/fillet): {has_torus}")
    print()
