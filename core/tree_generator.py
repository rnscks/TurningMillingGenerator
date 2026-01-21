"""
터닝 형상 트리 구조 생성기

트리 구조 규칙:
- b (base): 루트 노드 (항상 1개)
- s (step): 단차 가공 노드
- g (groove): 홈 가공 노드
- 최대 깊이 H 제약
- 총 노드 수 N 제약
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from itertools import product
import copy


@dataclass
class TreeGeneratorParams:
    """트리 생성 파라미터"""
    n_nodes: int = 6                    # 총 노드 수
    max_depth: int = 3                  # 최대 깊이 (루트=0 기준)
    labels: List[str] = None           # 사용 가능한 라벨 (b 제외)
    max_children_per_node: int = 4     # 노드당 최대 자식 수
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = ['s', 'g']  # step, groove


class TreeGenerator:
    """
    터닝 형상 트리 구조 생성기.
    
    N개의 노드, 최대 깊이 H를 갖는 모든 유효한 트리 구조를 열거.
    각 트리는 canonical 문자열로 표현되어 중복 제거됨.
    
    사용법:
        generator = TreeGenerator(TreeGeneratorParams(n_nodes=6, max_depth=3))
        trees = generator.generate_all_trees()
        generator.save_trees(trees, "output.json")
    """
    
    def __init__(self, params: TreeGeneratorParams = None):
        self.params = params or TreeGeneratorParams()
    
    def generate_all_trees(self) -> List[Dict]:
        """
        모든 유효한 트리 구조 생성.
        
        Returns:
            트리 딕셔너리 리스트 (중복 제거됨)
        """
        n = self.params.n_nodes
        h = self.params.max_depth
        labels = self.params.labels
        
        if n < 1:
            return []
        
        # 루트만 있는 경우
        if n == 1:
            return [self._create_single_node_tree()]
        
        # 모든 트리 구조 열거 (재귀적 생성)
        all_trees = []
        seen_canonicals: Set[str] = set()
        
        # 루트(b) 아래에 N-1개 노드를 배치하는 모든 방법 열거
        self._enumerate_trees(
            remaining_nodes=n - 1,
            max_depth_remaining=h,
            all_trees=all_trees,
            seen_canonicals=seen_canonicals
        )
        
        return all_trees
    
    def _create_single_node_tree(self) -> Dict:
        """단일 노드(루트만) 트리 생성"""
        return {
            "N": 1,
            "max_depth_constraint": self.params.max_depth,
            "canonical": "b",
            "nodes": [
                {"id": 0, "label": "b", "parent": None, "children": [], "depth": 0}
            ]
        }
    
    def _enumerate_trees(
        self,
        remaining_nodes: int,
        max_depth_remaining: int,
        all_trees: List[Dict],
        seen_canonicals: Set[str]
    ):
        """
        트리 구조 열거 (DFS 방식).
        루트 아래에 배치 가능한 모든 서브트리 조합을 탐색.
        """
        if remaining_nodes == 0:
            # 노드 수 제약 불만족
            return
        
        # 루트 아래 자식들의 서브트리 분배 열거
        # 각 자식은 특정 라벨(s 또는 g)을 가지고, 해당 서브트리에 일정 노드 수를 할당
        
        # 분배: 1~max_children 개의 자식에 remaining_nodes 개의 노드를 분배
        for n_children in range(1, min(remaining_nodes + 1, self.params.max_children_per_node + 1)):
            # 노드 수 분배: 각 자식 서브트리에 최소 1개씩
            for partition in self._partitions(remaining_nodes, n_children):
                # 각 자식의 라벨 조합
                for labels_combo in product(self.params.labels, repeat=n_children):
                    # 각 자식 서브트리에 대해 재귀적으로 생성
                    subtrees_options = []
                    valid = True
                    
                    for child_size, child_label in zip(partition, labels_combo):
                        if max_depth_remaining < 1:
                            valid = False
                            break
                        
                        child_subtrees = self._generate_subtrees(
                            node_count=child_size,
                            label=child_label,
                            depth_remaining=max_depth_remaining - 1
                        )
                        
                        if not child_subtrees:
                            valid = False
                            break
                        
                        subtrees_options.append(child_subtrees)
                    
                    if not valid:
                        continue
                    
                    # 모든 조합
                    for subtree_combo in product(*subtrees_options):
                        tree = self._build_tree_from_subtrees(list(subtree_combo))
                        canonical = tree["canonical"]
                        
                        if canonical not in seen_canonicals:
                            seen_canonicals.add(canonical)
                            all_trees.append(tree)
    
    def _generate_subtrees(
        self,
        node_count: int,
        label: str,
        depth_remaining: int
    ) -> List[Dict]:
        """
        특정 라벨과 노드 수를 갖는 서브트리 생성.
        
        Args:
            node_count: 서브트리 총 노드 수 (루트 포함)
            label: 서브트리 루트 라벨
            depth_remaining: 남은 허용 깊이
            
        Returns:
            가능한 서브트리 리스트
        """
        if node_count < 1 or depth_remaining < 0:
            return []
        
        if node_count == 1:
            # 리프 노드
            return [{
                "label": label,
                "children": [],
                "canonical": label
            }]
        
        if depth_remaining == 0:
            # 더 이상 자식 추가 불가
            return []
        
        # 자식 서브트리 생성
        subtrees = []
        child_nodes = node_count - 1  # 루트 제외
        
        for n_children in range(1, min(child_nodes + 1, self.params.max_children_per_node + 1)):
            for partition in self._partitions(child_nodes, n_children):
                for labels_combo in product(self.params.labels, repeat=n_children):
                    child_subtrees_options = []
                    valid = True
                    
                    for child_size, child_label in zip(partition, labels_combo):
                        child_result = self._generate_subtrees(
                            node_count=child_size,
                            label=child_label,
                            depth_remaining=depth_remaining - 1
                        )
                        
                        if not child_result:
                            valid = False
                            break
                        
                        child_subtrees_options.append(child_result)
                    
                    if not valid:
                        continue
                    
                    for children in product(*child_subtrees_options):
                        # canonical 문자열 생성 (정렬하여 중복 제거)
                        sorted_children = sorted(children, key=lambda x: x["canonical"])
                        children_canonical = ",".join(c["canonical"] for c in sorted_children)
                        canonical = f"{label}({children_canonical})"
                        
                        subtrees.append({
                            "label": label,
                            "children": list(sorted_children),
                            "canonical": canonical
                        })
        
        # 중복 제거
        seen = set()
        unique_subtrees = []
        for st in subtrees:
            if st["canonical"] not in seen:
                seen.add(st["canonical"])
                unique_subtrees.append(st)
        
        return unique_subtrees
    
    def _build_tree_from_subtrees(self, subtrees: List[Dict]) -> Dict:
        """
        서브트리 리스트로부터 완전한 트리 딕셔너리 생성.
        """
        # 서브트리 정렬 (canonical 기준)
        sorted_subtrees = sorted(subtrees, key=lambda x: x["canonical"])
        
        # canonical 문자열
        if not sorted_subtrees:
            canonical = "b"
        else:
            children_canonical = ",".join(st["canonical"] for st in sorted_subtrees)
            canonical = f"b({children_canonical})"
        
        # 노드 리스트 생성
        nodes = []
        node_id_counter = [0]  # mutable for nested function
        
        def add_node(subtree: Dict, parent_id: Optional[int], depth: int) -> int:
            """재귀적으로 노드 추가"""
            current_id = node_id_counter[0]
            node_id_counter[0] += 1
            
            children_ids = []
            for child_st in subtree.get("children", []):
                child_id = add_node(child_st, current_id, depth + 1)
                children_ids.append(child_id)
            
            nodes.append({
                "id": current_id,
                "label": subtree["label"],
                "parent": parent_id,
                "children": children_ids,
                "depth": depth
            })
            
            return current_id
        
        # 루트 노드
        root_subtree = {
            "label": "b",
            "children": sorted_subtrees,
            "canonical": canonical
        }
        add_node(root_subtree, None, 0)
        
        # ID 순으로 정렬
        nodes.sort(key=lambda x: x["id"])
        
        return {
            "N": len(nodes),
            "max_depth_constraint": self.params.max_depth,
            "canonical": canonical,
            "nodes": nodes
        }
    
    def _partitions(self, n: int, k: int) -> List[Tuple[int, ...]]:
        """
        정수 n을 k개의 양의 정수로 분할하는 모든 방법 (순서 고려).
        
        예: partitions(4, 2) = [(1,3), (2,2), (3,1)]
        """
        if k == 1:
            return [(n,)]
        
        result = []
        for i in range(1, n - k + 2):  # 최소 1씩 남겨야 함
            for rest in self._partitions(n - i, k - 1):
                result.append((i,) + rest)
        
        return result
    
    def generate_balanced_sample(
        self,
        total_count: int = 50,
        ensure_diversity: bool = True
    ) -> List[Dict]:
        """
        균형잡힌 샘플 트리 생성.
        
        step/groove 비율을 다양하게 포함하도록 샘플링.
        
        Args:
            total_count: 생성할 총 트리 수
            ensure_diversity: step/groove 비율 다양성 보장 여부
            
        Returns:
            샘플링된 트리 리스트
        """
        all_trees = self.generate_all_trees()
        
        if len(all_trees) <= total_count:
            return all_trees
        
        if not ensure_diversity:
            import random
            return random.sample(all_trees, total_count)
        
        # step/groove 비율별로 분류
        by_ratio: Dict[Tuple[int, int], List[Dict]] = {}
        for tree in all_trees:
            nodes = tree["nodes"]
            s_count = sum(1 for n in nodes if n["label"] == "s")
            g_count = sum(1 for n in nodes if n["label"] == "g")
            key = (s_count, g_count)
            
            if key not in by_ratio:
                by_ratio[key] = []
            by_ratio[key].append(tree)
        
        # 각 비율에서 균등하게 샘플링
        result = []
        import random
        
        ratios = list(by_ratio.keys())
        per_ratio = max(1, total_count // len(ratios))
        
        for ratio in ratios:
            trees = by_ratio[ratio]
            sample_count = min(len(trees), per_ratio)
            result.extend(random.sample(trees, sample_count))
        
        # 부족하면 추가 샘플링
        while len(result) < total_count and len(result) < len(all_trees):
            remaining = [t for t in all_trees if t not in result]
            if not remaining:
                break
            result.append(random.choice(remaining))
        
        return result[:total_count]


def generate_trees(
    n_nodes: int = 6,
    max_depth: int = 3,
    labels: List[str] = None
) -> List[Dict]:
    """
    트리 구조 생성 편의 함수.
    
    Args:
        n_nodes: 총 노드 수
        max_depth: 최대 깊이
        labels: 사용할 라벨 리스트 (기본: ['s', 'g'])
        
    Returns:
        생성된 트리 딕셔너리 리스트
    """
    params = TreeGeneratorParams(
        n_nodes=n_nodes,
        max_depth=max_depth,
        labels=labels or ['s', 'g']
    )
    generator = TreeGenerator(params)
    return generator.generate_all_trees()
