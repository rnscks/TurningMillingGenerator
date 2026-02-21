"""
트리 데이터 입출력 및 유틸리티 모듈

트리 구조:
- b (base): 루트 노드 - Stock 원기둥
- s (step): 단차 가공
- g (groove): 홈 가공

기능:
- 트리 데이터 로드/저장
- 트리 분류 및 필터링
"""

import json
from typing import List, Dict, Optional
from pathlib import Path


# ============================================================================
# 트리 데이터 로드/저장
# ============================================================================

def load_trees(json_path: str) -> List[Dict]:
    """
    JSON 파일에서 트리 데이터 로드.
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        트리 딕셔너리 리스트
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"트리 파일을 찾을 수 없습니다: {json_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # "trees" 키가 있으면 해당 값 반환, 아니면 리스트로 가정
    if isinstance(data, dict) and "trees" in data:
        return data["trees"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"올바르지 않은 트리 데이터 형식: {json_path}")


def save_trees(trees: List[Dict], json_path: str):
    """
    트리 데이터를 JSON 파일로 저장.
    
    Args:
        trees: 트리 딕셔너리 리스트
        json_path: 저장할 JSON 파일 경로
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"trees": trees}, f, indent=2, ensure_ascii=False)


def classify_trees_by_step_count(trees: List[Dict]) -> Dict[int, List[int]]:
    """
    트리를 step(s) 노드 개수별로 분류.
    
    Args:
        trees: 트리 딕셔너리 리스트
        
    Returns:
        {step_count: [tree_indices]} 형태의 딕셔너리
    """
    result = {}
    for i, tree in enumerate(trees):
        s_count = sum(1 for n in tree.get('nodes', []) if n.get('label') == 's')
        if s_count not in result:
            result[s_count] = []
        result[s_count].append(i)
    return result


def classify_trees_by_groove_count(trees: List[Dict]) -> Dict[int, List[int]]:
    """
    트리를 groove(g) 노드 개수별로 분류.
    
    Args:
        trees: 트리 딕셔너리 리스트
        
    Returns:
        {groove_count: [tree_indices]} 형태의 딕셔너리
    """
    result = {}
    for i, tree in enumerate(trees):
        g_count = sum(1 for n in tree.get('nodes', []) if n.get('label') == 'g')
        if g_count not in result:
            result[g_count] = []
        result[g_count].append(i)
    return result


def get_tree_stats(tree: Dict) -> Dict:
    """
    단일 트리의 통계 정보 반환.
    
    Args:
        tree: 트리 딕셔너리
        
    Returns:
        통계 정보 딕셔너리
    """
    nodes = tree.get('nodes', [])
    labels = [n.get('label', '') for n in nodes]
    
    # Base의 Step 자식 개수 확인 (양방향 Step 여부)
    base_step_children = 0
    for n in nodes:
        if n.get('label') == 'b':
            children_ids = n.get('children', [])
            base_step_children = sum(
                1 for cid in children_ids 
                if nodes[cid].get('label') == 's'
            )
            break
    
    # 형제 Groove 존재 여부 확인
    has_sibling_grooves = False
    for n in nodes:
        if n.get('label') in ['b', 's']:
            children_ids = n.get('children', [])
            groove_children = sum(
                1 for cid in children_ids 
                if nodes[cid].get('label') == 'g'
            )
            if groove_children >= 2:
                has_sibling_grooves = True
                break
    
    return {
        'n_nodes': tree.get('N', len(nodes)),
        'max_depth': tree.get('max_depth_constraint', 0),
        'canonical': tree.get('canonical', ''),
        's_count': labels.count('s'),
        'g_count': labels.count('g'),
        'b_count': labels.count('b'),
        'base_step_children': base_step_children,  # 양방향 Step 여부 (2 이상이면 양방향)
        'has_sibling_grooves': has_sibling_grooves,  # 형제 Groove 존재 여부
    }


def find_bidirectional_step_trees(trees: List[Dict]) -> List[int]:
    """
    양방향 Step 트리 인덱스 반환 (Base에 Step 자식이 2개 이상).
    
    Args:
        trees: 트리 딕셔너리 리스트
        
    Returns:
        양방향 Step 트리 인덱스 리스트
    """
    result = []
    for i, tree in enumerate(trees):
        stats = get_tree_stats(tree)
        if stats['base_step_children'] >= 2:
            result.append(i)
    return result


def find_sibling_groove_trees(trees: List[Dict]) -> List[int]:
    """
    형제 Groove가 있는 트리 인덱스 반환.
    
    Args:
        trees: 트리 딕셔너리 리스트
        
    Returns:
        형제 Groove 트리 인덱스 리스트
    """
    result = []
    for i, tree in enumerate(trees):
        stats = get_tree_stats(tree)
        if stats['has_sibling_grooves']:
            result.append(i)
    return result


def filter_trees(
    trees: List[Dict],
    min_steps: int = None,
    max_steps: int = None,
    min_grooves: int = None,
    max_grooves: int = None,
) -> List[int]:
    """
    조건에 맞는 트리 인덱스 필터링.
    
    Args:
        trees: 트리 딕셔너리 리스트
        min_steps: 최소 step 개수
        max_steps: 최대 step 개수
        min_grooves: 최소 groove 개수
        max_grooves: 최대 groove 개수
        
    Returns:
        조건에 맞는 트리 인덱스 리스트
    """
    result = []
    for i, tree in enumerate(trees):
        stats = get_tree_stats(tree)
        
        if min_steps is not None and stats['s_count'] < min_steps:
            continue
        if max_steps is not None and stats['s_count'] > max_steps:
            continue
        if min_grooves is not None and stats['g_count'] < min_grooves:
            continue
        if max_grooves is not None and stats['g_count'] > max_grooves:
            continue
        
        result.append(i)
    
    return result

