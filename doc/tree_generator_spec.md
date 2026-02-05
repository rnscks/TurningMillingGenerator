# TreeGenerator 로직 요약 및 테스트 가이드

## 개요

`tree_generator.py`는 터닝 형상의 트리 구조를 생성하는 모듈입니다. 트리의 각 노드는 가공 피처를 나타내며, 주어진 제약 조건(노드 수, 최대 깊이, 기하학적 제약) 내에서 가능한 모든 유효한 트리 구조를 열거합니다.

---

## 핵심 개념

### 노드 라벨
| 라벨 | 의미 | 설명 |
|------|------|------|
| `b` | base | 루트 노드 (항상 1개, depth=0) |
| `s` | step | 단차 가공 노드 |
| `g` | groove | 홈 가공 노드 |

### 트리 제약 조건
- **N**: 총 노드 수 (루트 포함)
- **H (max_depth)**: 최대 깊이 (루트=0 기준)
- **max_children_per_node**: 노드당 최대 자식 수 (기본값: 4)
- **max_step_children_from_base**: Base에서 Step 자식 최대 수 (기본값: 2)
- **max_step_children_from_step**: Step에서 Step 자식 최대 수 (기본값: 1)

### 기하학적 제약 조건

터닝 가공의 물리적 특성에 따른 제약:

| 부모 → 자식 | 기하학적 의미 | 허용 여부 |
|-------------|---------------|-----------|
| Step → Step | 계단 위에 더 작은 계단 | ✅ 가능 (최대 1개) |
| Step → Groove | 계단 면에 홈 파기 | ✅ 가능 |
| Groove → Groove | 홈 안에 더 깊은 홈 | ✅ 가능 (중첩 홈) |
| Groove → Step | 좁은 홈 안에서 계단 | ❌ **불가능** |
| Base → Step | 베이스에서 단차 | ✅ 가능 (최대 2개) |
| Base → Groove | 베이스에서 홈 | ✅ 가능 |

**Groove → Step이 불가능한 이유:**
Groove는 좁은 폭(width, 2~3mm)의 홈이며, 그 영역 안에서 Step의 높이(height, 2.5~6.5mm)를 확보하는 것은 기하학적으로 성립하지 않습니다.

---

## 주요 알고리즘

### 1. 트리 열거 방식
```
1. 루트(b) 노드 생성
2. 나머지 N-1개 노드를 자식으로 분배
3. 각 분배 방식에 대해:
   - 자식 수 결정 (1 ~ max_children)
   - 노드 수 분할 (_partitions)
   - 라벨 조합 생성 (cartesian product)
   - 재귀적으로 서브트리 생성
4. canonical 문자열로 중복 제거
```

### 2. Canonical 문자열
중복 트리를 제거하기 위한 정규화된 문자열 표현입니다.

**형식**: `label(child1,child2,...)` (자식은 canonical 기준 정렬)

**예시**:
- `b` - 루트만 있는 트리
- `b(s)` - 루트 아래 step 노드 1개
- `b(g,s)` - 루트 아래 groove, step 노드 (정렬됨)
- `b(s(g),s)` - 중첩 구조

### 3. 정수 분할 (_partitions)
n을 k개의 양의 정수로 분할하는 모든 **순서 있는** 조합을 생성합니다.

```
partitions(4, 2) = [(1,3), (2,2), (3,1)]
```

이는 수학적으로 **composition** (순서 있는 분할)에 해당합니다.

---

## 클래스 구조

### TreeGeneratorParams
```python
@dataclass
class TreeGeneratorParams:
    n_nodes: int = 6                      # 총 노드 수
    max_depth: int = 3                    # 최대 깊이
    labels: List[str] = ['s', 'g']        # 사용 가능한 라벨
    max_children_per_node: int = 4        # 노드당 최대 자식 수
    max_step_children_from_base: int = 2  # Base에서 Step 자식 최대 수
    max_step_children_from_step: int = 1  # Step에서 Step 자식 최대 수
```

### TreeGenerator 주요 메서드
| 메서드 | 설명 |
|--------|------|
| `generate_all_trees()` | 모든 유효한 트리 생성 |
| `generate_balanced_sample()` | step/groove 비율 다양성을 고려한 샘플링 |
| `_get_allowed_child_labels()` | 부모 라벨에 따른 허용 자식 라벨 반환 |
| `_get_max_children_count()` | 부모-자식 조합별 최대 자식 수 반환 |
| `_validate_children_combination()` | 자식 조합 유효성 검증 |
| `_generate_subtrees()` | 재귀적 서브트리 생성 |
| `_partitions()` | 정수 분할 |
| `_build_tree_from_subtrees()` | 서브트리로부터 완전한 트리 구성 |

---

## 출력 형식

### 트리 딕셔너리 구조
```python
{
    "N": 4,                           # 노드 수
    "max_depth_constraint": 3,        # 최대 깊이 제약
    "canonical": "b(g,s(s))",         # 정규화된 문자열
    "nodes": [                        # 노드 리스트
        {
            "id": 0,
            "label": "b",
            "parent": None,
            "children": [1, 2],
            "depth": 0
        },
        # ...
    ]
}
```

---

## 테스트 주의사항

### 1. 입력 검증 테스트
- `max_depth < 1`: `ValueError` 발생 확인 필수
- `n_nodes <= 0`: 빈 리스트 반환 확인

### 2. 구조 검증 테스트
트리 생성 후 반드시 검증해야 할 항목:

| 검증 항목 | 설명 |
|-----------|------|
| 노드 수 | `tree["N"] == len(tree["nodes"])` |
| 깊이 제약 | 모든 노드의 depth ≤ max_depth |
| 부모-자식 일관성 | 양방향 참조 일치 |
| 루트 조건 | depth=0인 노드는 label="b", parent=None |
| 라벨 유효성 | depth>0인 노드는 지정된 labels 중 하나 |

### 3. 기하학적 제약 테스트 (중요!)
**반드시 검증해야 할 기하학적 제약:**

| 테스트 항목 | 설명 |
|-------------|------|
| Groove → Step 금지 | Groove의 자식으로 Step이 없어야 함 |
| Base → Step 최대 2개 | Base의 Step 자식 수 ≤ 2 |
| Step → Step 최대 1개 | Step의 Step 자식 수 ≤ 1 |
| Groove → Groove 허용 | 중첩 Groove 구조 가능 확인 |
| Step → Groove 허용 | Step 아래 Groove 가능 확인 |

```python
# 부모-자식 쌍 추출 헬퍼 함수
def get_parent_child_pairs(tree):
    nodes = tree["nodes"]
    node_map = {n["id"]: n for n in nodes}
    pairs = []
    for node in nodes:
        for child_id in node["children"]:
            child = node_map[child_id]
            pairs.append((node["label"], child["label"]))
    return pairs

# 검증 예시
for parent, child in get_parent_child_pairs(tree):
    if parent == 'g':
        assert child != 's', "Groove → Step 금지 위반!"
```

### 4. Canonical 문자열 테스트
- **고유성**: 모든 canonical 문자열은 서로 달라야 함
- **형식**: 루트는 항상 "b"로 시작
- **정렬**: 자식은 canonical 기준 사전순 정렬

### 5. 분할 함수 테스트
`_partitions(n, k)` 검증:
- 각 분할의 합 = n
- 각 분할의 길이 = k
- 모든 요소 ≥ 1

### 6. 샘플링 테스트
`generate_balanced_sample()` 검증:
- 반환 개수 ≤ min(요청 수, 전체 트리 수)
- `ensure_diversity=True`일 때 다양한 step/groove 비율 포함

### 7. 엣지 케이스
| 케이스 | 예상 결과 |
|--------|-----------|
| n_nodes=1 | 루트만 있는 트리 1개 |
| max_depth=1 | 루트+리프만 가능 |
| labels=['s'] | step 노드만 생성 |
| labels=['g'] | groove 노드만 생성 (중첩 가능) |
| 매우 큰 N, H | 성능 테스트 (트리 수 폭발적 증가 주의) |

### 8. 성능 고려사항
- N과 H가 커지면 트리 수가 기하급수적으로 증가
- N=8, H=4 정도까지는 합리적인 시간 내 처리 가능
- 대규모 테스트 시 timeout 설정 권장
- 기하학적 제약으로 인해 제약 없는 경우보다 트리 수가 적음

---

## 테스트 실행 방법

```bash
# 전체 테스트
pytest tests/test_tree_generator.py -v

# 특정 클래스만
pytest tests/test_tree_generator.py::TestTreeGeneratorValidation -v

# 특정 테스트만
pytest tests/test_tree_generator.py::TestTreeGeneratorValidation::test_max_depth_zero_raises_error -v

# 커버리지 포함
pytest tests/test_tree_generator.py --cov=core.tree_generator --cov-report=term-missing
```

---

## 알려진 제한사항

1. **메모이제이션 미적용**: `_generate_subtrees()`가 동일 인자로 여러 번 호출될 수 있음
2. **대규모 입력**: N > 10, H > 5 조합은 매우 많은 트리를 생성하여 메모리/시간 문제 발생 가능
3. **순서 있는 분할**: `_partitions`는 composition을 생성하므로 (1,3)과 (3,1)이 별개로 처리됨 (의도된 동작)

---

## 터닝 형상 생성기 연동 (turning_generator.py)

### Bottom-Up 파라미터 결정 방식

트리 구조에서 실제 터닝 형상을 생성할 때, **Bottom-Up 방식**으로 파라미터를 결정합니다.

#### 왜 Bottom-Up인가?

| 방식 | 문제점 |
|------|--------|
| **Top-Down** | 부모가 먼저 크기를 결정 → 자식에게 남은 공간 부족 → 스킵 발생 |
| **Bottom-Up** | 자식이 먼저 필요 크기 계산 → 부모가 감싸는 형태 → 모든 피처 생성 보장 |

#### Bottom-Up 처리 흐름

```
1. [리프 → 루트] 필요 공간 계산
   - 리프 노드: 자신의 피처 크기만 필요
   - 비-리프 노드: 자식들 높이 합 + 자신의 피처 + margin
   
2. [루트] Stock 크기 결정
   - 필요 높이 + 여유 margin
   - 필요 깊이 + min_remaining_radius + margin
   
3. [루트 → 리프] 형상 생성
   - 공간이 이미 보장되어 있으므로 스킵 없이 모든 피처 생성
```

#### RequiredSpace 데이터 구조

```python
@dataclass
class RequiredSpace:
    height: float           # 필요한 z 높이
    depth: float            # 필요한 반경 깊이 (누적)
    feature_height: float   # 자신의 피처 높이
    feature_depth: float    # 자신의 피처 깊이
```

#### TurningParams (Bottom-Up 전용)

```python
@dataclass
class TurningParams:
    # Stock margin (필요 크기에 추가)
    stock_height_margin: (3.0, 8.0)
    stock_radius_margin: (2.0, 5.0)
    
    # Step 파라미터
    step_depth_range: (0.8, 1.5)    # 반경 방향
    step_height_range: (2.0, 4.0)   # z 방향
    step_margin: 0.5
    
    # Groove 파라미터
    groove_depth_range: (0.4, 0.8)
    groove_width_range: (1.5, 3.0)
    groove_margin: 0.3
```

#### 예시: `b(s(g(g,g)))`

```
1. 리프 Groove들: 각각 height=2.0mm (width + margin)
2. 부모 Groove: height = max(2.0, 2.0+2.0) + margin = 4.6mm
3. Step: height = 4.6 + step_height + margin = 8.0mm
4. Base: total_height = 8.0mm → Stock height = 8.0 + 5.0 = 13.0mm
```

모든 Groove와 Step이 스킵 없이 생성됩니다.

### Groove 분산 배치

같은 부모 아래의 **형제 Groove**들은 Z축으로 분산 배치됩니다.

#### 문제점 (수정 전)
```python
# 모든 Groove가 부모 영역 중앙에 배치됨
center_z = (parent_region.z_min + parent_region.z_max) / 2
zpos = center_z - groove_width / 2
```

형제 Groove들이 같은 위치에 배치되어 겹침 발생.

#### 해결 방법 (수정 후)
```python
def _apply_groove_bottomup(
    self, node, parent_region, 
    groove_index: int = 0,      # 형제 중 인덱스
    total_grooves: int = 1      # 형제 총 개수
):
    if total_grooves == 1:
        # 단일 groove: 중앙 배치
        zpos = center_z - groove_width / 2
    else:
        # 여러 groove: Z축으로 분산 배치
        spacing = available_height / total_grooves
        zpos = parent_region.z_min + margin + groove_index * spacing
```

#### 배치 타입 비교

| 구조 | 설명 | 배치 방식 |
|------|------|-----------|
| `s(g,g)` | 형제 Groove | Z축 분산 (서로 다른 위치) |
| `g(g)` | 중첩 Groove | 같은 Z, 더 작은 반경 (안쪽으로 파고 들어감) |

### Step 방향 처리 (top/bottom)

Base 노드의 자식 Step들은 **교대로 top/bottom 방향**이 할당됩니다.

#### 방향 할당 규칙

| 부모 | 자식 Step 순서 | 방향 |
|------|----------------|------|
| Base | 첫 번째 Step | `top` (위에서 깎음) |
| Base | 두 번째 Step | `bottom` (아래에서 깎음) |
| Step | Step 자식 | 부모와 **같은 방향** 유지 |

#### 예시: `b(s,s(s(s)))`

```
Base
├── Step (top): z=[22.48, 25.52] - 위쪽에서 시작
└── Step (bottom): z=[0.00, 16.76] - 아래쪽에서 시작
    └── Step (bottom): z=[0.00, 12.14] - 방향 유지
        └── Step (bottom): z=[0.00, 8.45] - 방향 유지
```

**참고**: Base의 Step 자식이 1개만 있는 경우(트리의 76%) 기본값 `top`만 사용됩니다.

---

## 버전 히스토리

### v1.5 (현재)
- **트리 선택 로직 개선** (`run_pipeline.py`, `utils/tree_io.py`):
  - `find_bidirectional_step_trees()`: 양방향 Step 트리 (Base에 Step 자식 2개) 탐색
  - `find_sibling_groove_trees()`: 형제 Groove 트리 탐색
  - 양방향 Step 트리 최소 2개, 형제 Groove 트리 최소 2개 우선 선택
  - `get_tree_stats()`에 `base_step_children`, `has_sibling_grooves` 필드 추가

### v1.4
- **2단계 처리 방식 도입**:
  - Step을 먼저 모두 처리한 후 Groove 처리 (Boolean Cut 충돌 방지)
  - `_process_steps_only()`: Step만 처리하여 region 확정
  - `_process_grooves_only()`: Step 완료 후 Groove 처리
- **Groove 검증 및 재시도 로직**:
  - `_apply_groove_with_validation()`: Groove 적용 실패 시 최대 3회 재시도
  - Boolean Cut 결과 검증 (IsNull 체크)
  - Groove 생성 개수 추적 및 경고 출력
- **트리의 Groove 개수 검증**:
  - `_count_grooves_in_tree()`: 예상 Groove 개수와 실제 생성 개수 비교

### v1.3
- **Groove 분산 배치 로직 추가**:
  - `_apply_groove_bottomup()`에 `groove_index`, `total_grooves` 파라미터 추가
  - 형제 Groove들이 Z축으로 분산 배치되어 겹침 방지
  - 중첩 Groove(`g(g)`)는 같은 Z, 더 작은 반경으로 처리 (기존 동작 유지)
- **Step 방향 처리 명확화**:
  - Base의 자식 Step들에 `top`/`bottom` 교대 할당 명시
  - Step 체인에서 방향 유지 로직 문서화

### v1.2
- **Bottom-Up 파라미터 결정 방식 도입**:
  - 리프 노드부터 필요 공간 계산
  - Stock 크기를 요구사항에 맞춰 자동 조정
  - 모든 피처가 스킵 없이 생성됨
- `RequiredSpace` 데이터 클래스 추가
- `_calculate_required_space()`, `_create_stock_from_requirements()` 메서드 추가
- `_apply_step_bottomup()`, `_apply_groove_bottomup()` 메서드 추가

### v1.1
- **기하학적 제약 추가**:
  - Groove → Step 금지 규칙 구현
  - Base → Step 최대 2개 제한
  - Step → Step 최대 1개 제한
- 새로운 메서드 추가: `_get_allowed_child_labels()`, `_get_max_children_count()`, `_validate_children_combination()`
- 새로운 파라미터: `max_step_children_from_base`, `max_step_children_from_step`

### v1.0
- 초기 구현 (기하학적 제약 없음, Top-Down 방식)
