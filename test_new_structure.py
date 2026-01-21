"""새 구조 테스트 스크립트"""

import json
import random

from pipeline import TurningMillingGenerator, TurningMillingParams
from core import TurningParams, HoleParams

def main():
    random.seed(42)
    
    # 트리 데이터 로드
    with open('trees_N6_H3.json', 'r') as f:
        trees = json.load(f)
    
    print(f"총 {len(trees)}개 트리 로드됨")
    
    # 파라미터 설정
    params = TurningMillingParams(
        turning=TurningParams(),
        hole=HoleParams(),
        max_holes=2,
    )
    
    # 테스트: 첫 번째 트리로 형상 생성
    generator = TurningMillingGenerator(params)
    tree = trees[0]
    canonical = tree['canonical']
    print(f"Tree: {canonical}")
    
    shape, placements = generator.generate_from_tree(tree)
    print(f"Shape valid: {shape is not None and not shape.IsNull()}")
    print(f"Holes: {len(placements)}")
    
    # 저장 테스트
    generator.save('generated_models/test_new_structure.step')
    print("테스트 완료!")

if __name__ == "__main__":
    main()
