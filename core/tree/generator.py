"""
터닝 형상 트리 구조 생성기

트리 구조 규칙:
- b (base): 루트 노드 (항상 1개)
- s (step): 단차 가공 노드
- g (groove): 홈 가공 노드
- 최대 깊이 H 제약
- 총 노드 수 N 제약

기하학적 제약 조건:
- Groove → Step: 불가능 (좁은 홈 안에서 계단 불가)
- Groove → Groove: 가능 (중첩 홈)
- Step → Step: 가능하지만 자식으로 Step은 1개만
- Step → Groove: 가능
- Base → Step: 2방향으로 최대 2개까지
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from itertools import product
import random


@dataclass
class TreeGeneratorParams:
    """트리 생성 파라미터"""
    n_nodes: int = 6
    max_depth: int = 3
    labels: List[str] = None
    max_children_per_node: int = 4
    max_step_children_from_base: int = 2
    max_step_children_from_step: int = 1

    def __post_init__(self):
        if self.labels is None:
            self.labels = ['s', 'g']


class TreeGenerator:
    """
    터닝 형상 트리 구조 생성기.

    N개의 노드, 최대 깊이 H를 갖는 모든 유효한 트리 구조를 열거.
    각 트리는 canonical 문자열로 표현되어 중복 제거됨.

    사용법:
        generator = TreeGenerator(TreeGeneratorParams(n_nodes=6, max_depth=3))
        trees = generator.generate_all_trees()
    """

    def __init__(self, params: TreeGeneratorParams = None):
        self.params = params or TreeGeneratorParams()

    def _get_allowed_child_labels(self, parent_label: str) -> List[str]:
        if parent_label == 'g':
            return ['g'] if 'g' in self.params.labels else []
        else:
            return self.params.labels

    def _get_max_children_count(
        self,
        parent_label: str,
        child_label: str,
        current_counts: Dict[str, int] = None
    ) -> int:
        if current_counts is None:
            current_counts = {}

        if parent_label == 'b' and child_label == 's':
            return self.params.max_step_children_from_base
        elif parent_label == 's' and child_label == 's':
            return self.params.max_step_children_from_step
        else:
            return self.params.max_children_per_node

    def _validate_children_combination(
        self,
        parent_label: str,
        children_labels: Tuple[str, ...]
    ) -> bool:
        allowed = self._get_allowed_child_labels(parent_label)
        for child_label in children_labels:
            if child_label not in allowed:
                return False

        label_counts = {}
        for label in children_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            max_count = self._get_max_children_count(parent_label, label)
            if count > max_count:
                return False

        return True

    def generate_all_trees(self) -> List[Dict]:
        n = self.params.n_nodes
        h = self.params.max_depth
        labels = self.params.labels

        if n < 1:
            return []

        if h < 1:
            raise ValueError(f"max_depth must be at least 1, got {h}")

        if n == 1:
            return [self._create_single_node_tree()]

        all_trees = []
        seen_canonicals: Set[str] = set()

        self._enumerate_trees(
            remaining_nodes=n - 1,
            max_depth_remaining=h,
            all_trees=all_trees,
            seen_canonicals=seen_canonicals
        )

        return all_trees

    def _create_single_node_tree(self) -> Dict:
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
        if remaining_nodes == 0:
            return

        for n_children in range(1, min(remaining_nodes + 1, self.params.max_children_per_node + 1)):
            for partition in self._partitions(remaining_nodes, n_children):
                for labels_combo in product(self.params.labels, repeat=n_children):
                    if not self._validate_children_combination('b', labels_combo):
                        continue

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
        if node_count < 1 or depth_remaining < 0:
            return []

        if node_count == 1:
            return [{
                "label": label,
                "children": [],
                "canonical": label
            }]

        if depth_remaining == 0:
            return []

        allowed_child_labels = self._get_allowed_child_labels(label)
        if not allowed_child_labels:
            return []

        subtrees = []
        child_nodes = node_count - 1

        for n_children in range(1, min(child_nodes + 1, self.params.max_children_per_node + 1)):
            for partition in self._partitions(child_nodes, n_children):
                for labels_combo in product(allowed_child_labels, repeat=n_children):
                    if not self._validate_children_combination(label, labels_combo):
                        continue

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
                        sorted_children = sorted(children, key=lambda x: x["canonical"])
                        children_canonical = ",".join(c["canonical"] for c in sorted_children)
                        canonical = f"{label}({children_canonical})"

                        subtrees.append({
                            "label": label,
                            "children": list(sorted_children),
                            "canonical": canonical
                        })

        seen = set()
        unique_subtrees = []
        for st in subtrees:
            if st["canonical"] not in seen:
                seen.add(st["canonical"])
                unique_subtrees.append(st)

        return unique_subtrees

    def _build_tree_from_subtrees(self, subtrees: List[Dict]) -> Dict:
        sorted_subtrees = sorted(subtrees, key=lambda x: x["canonical"])

        if not sorted_subtrees:
            canonical = "b"
        else:
            children_canonical = ",".join(st["canonical"] for st in sorted_subtrees)
            canonical = f"b({children_canonical})"

        nodes = []
        node_id_counter = [0]

        def add_node(subtree: Dict, parent_id: Optional[int], depth: int) -> int:
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

        root_subtree = {
            "label": "b",
            "children": sorted_subtrees,
            "canonical": canonical
        }
        add_node(root_subtree, None, 0)

        nodes.sort(key=lambda x: x["id"])

        return {
            "N": len(nodes),
            "max_depth_constraint": self.params.max_depth,
            "canonical": canonical,
            "nodes": nodes
        }

    def _partitions(self, n: int, k: int) -> List[Tuple[int, ...]]:
        if k == 1:
            return [(n,)]

        result = []
        for i in range(1, n - k + 2):
            for rest in self._partitions(n - i, k - 1):
                result.append((i,) + rest)

        return result

    def generate_balanced_sample(
        self,
        total_count: int = 50,
        ensure_diversity: bool = True
    ) -> List[Dict]:
        all_trees = self.generate_all_trees()

        if len(all_trees) <= total_count:
            return all_trees

        if not ensure_diversity:
            return random.sample(all_trees, total_count)

        by_ratio: Dict[Tuple[int, int], List[Dict]] = {}
        for tree in all_trees:
            nodes = tree["nodes"]
            s_count = sum(1 for n in nodes if n["label"] == "s")
            g_count = sum(1 for n in nodes if n["label"] == "g")
            key = (s_count, g_count)

            if key not in by_ratio:
                by_ratio[key] = []
            by_ratio[key].append(tree)

        result = []
        ratios = list(by_ratio.keys())
        per_ratio = max(1, total_count // len(ratios))

        for ratio in ratios:
            trees = by_ratio[ratio]
            sample_count = min(len(trees), per_ratio)
            result.extend(random.sample(trees, sample_count))

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
    """트리 구조 생성 편의 함수."""
    params = TreeGeneratorParams(
        n_nodes=n_nodes,
        max_depth=max_depth,
        labels=labels or ['s', 'g']
    )
    generator = TreeGenerator(params)
    return generator.generate_all_trees()
