# TurningGenerator 로직 상세 명세

## 개요

`turning_generator.py`는 트리 구조를 기반으로 3D 터닝 형상을 생성하는 모듈입니다. OCC (Open CASCADE Technology)를 사용하여 실린더 스톡에서 Step(단차)과 Groove(홈) 피처를 Boolean Cut으로 생성합니다.

---

## 핵심 데이터 구조

### Region (영역)

각 피처가 차지하는 3D 공간을 정의합니다.

```python
@dataclass
class Region:
    z_min: float      # Z축 시작점
    z_max: float      # Z축 끝점
    radius: float     # 반경 (외측 경계)
    direction: str    # 'top' 또는 'bottom' (Step 방향)
```

#### 영역 의미

| 속성 | 의미 | 비고 |
|------|------|------|
| z_min, z_max | Z축 범위 | 피처가 존재하는 높이 범위 |
| radius | 외측 반경 | 피처의 반경 방향 외측 경계 |
| direction | 방향 | Step이 깎이는 방향 |

### RequiredSpace (필요 공간)

Bottom-Up 계산 시 노드가 필요로 하는 공간입니다.

```python
@dataclass
class RequiredSpace:
    height: float         # 필요한 z 높이 (자식 포함)
    depth: float          # 필요한 반경 깊이 (누적)
    feature_height: float # 자신의 피처 높이만
    feature_depth: float  # 자신의 피처 깊이만
```

### TurningParams (생성 파라미터)

형상 생성에 사용되는 파라미터 범위입니다.

```python
@dataclass
class TurningParams:
    # Stock margin
    stock_height_margin: tuple = (3.0, 8.0)
    stock_radius_margin: tuple = (2.0, 5.0)
    min_remaining_radius: float = 5.0
    
    # Step 파라미터
    step_depth_range: tuple = (0.8, 1.5)     # 반경 방향 깊이
    step_height_range: tuple = (2.0, 4.0)    # Z 방향 높이
    step_margin: float = 0.5                  # Step 영역 margin
    
    # Groove 파라미터
    groove_depth_range: tuple = (0.4, 0.8)   # 반경 방향 깊이
    groove_width_range: tuple = (1.5, 3.0)   # Z 방향 폭
    groove_margin: float = 0.3                # Groove margin
```

---

## 핵심 알고리즘: Bottom-Up 파라미터 결정

### 기존 Top-Down 방식의 문제점

| 문제 | 결과 |
|------|------|
| 부모가 먼저 크기를 결정 | 자식에게 남은 공간 부족 |
| 중첩 피처 스킵 발생 | Groove가 누락됨 |
| 예측 불가능한 결과 | 트리의 피처 수와 실제 생성 수 불일치 |

### Bottom-Up 해결 방식

```
[Phase 1: 리프 → 루트] 필요 공간 계산
       ↓
[Phase 2: 루트] Stock 크기 결정  
       ↓
[Phase 3: 루트 → 리프] 형상 생성 (스킵 없음)
```

### Phase 1: 필요 공간 계산 (`_calculate_required_space`)

리프 노드부터 시작하여 각 노드가 필요로 하는 공간을 상향 전파합니다.

```python
def _calculate_required_space(self, node: TreeNode) -> RequiredSpace:
    # 1. 모든 자식의 필요 공간을 먼저 계산 (재귀)
    for child in node.children:
        self._calculate_required_space(child)
    
    # 2. Step 자식과 Groove 자식 분리
    # (핵심: Groove 자식은 부모 영역 내부에 배치되므로 추가 높이 불필요)
    step_children = [c for c in node.children if c.label == 's']
    groove_children = [c for c in node.children if c.label == 'g']
    
    step_children_height = sum(c.required_space.height for c in step_children)
    
    # 3. 노드 타입별 필요 공간 결정
    if node.label == 's':
        # Step: Step 자식들 높이만 합산 (Groove 자식은 제외)
        total_height = step_children_height + feature_height + margin
```

#### Step과 Groove 자식의 높이 처리 차이

| 자식 타입 | 높이 처리 | 이유 |
|-----------|-----------|------|
| **Step 자식** | 부모 높이에 합산 | Step은 부모 영역 끝에서 확장됨 |
| **Groove 자식** | 합산 안 함 | Groove는 부모 영역 **내부**에 배치됨 |

```
Step(s) 영역 예시:
┌─────────────────┐ z_max
│   Groove (g)    │  ← Groove는 Step 영역 내부에 배치
│   ────────      │     (추가 공간 불필요)
│                 │
│   Step Child    │  ← Step 자식은 하단에서 확장
│   ┌─────────┐   │     (공간 필요)
│   │         │   │
│   └─────────┘   │
└─────────────────┘ z_min
```

### Phase 2: Stock 크기 결정 (`_create_stock_from_requirements`)

```python
def _create_stock_from_requirements(self, root: TreeNode):
    required = root.required_space
    
    # 높이: 필요 높이 + margin
    height_margin = random.uniform(*self.params.stock_height_margin)
    self.stock_height = required.height + height_margin
    
    # 반경: 필요 깊이 + 최소 잔여 + margin
    radius_margin = random.uniform(*self.params.stock_radius_margin)
    self.stock_radius = required.depth + self.params.min_remaining_radius + radius_margin
```

### Phase 3: 2단계 형상 생성

Boolean Cut 충돌을 방지하기 위해 Step과 Groove를 분리하여 처리합니다.

```
Stage 1: Step 처리 (_process_steps_only)
         └── 모든 Step 피처 생성 및 Region 확정

Stage 2: Groove 처리 (_process_grooves_only)
         └── 확정된 Region에 Groove 배치
```

---

## 피처 영역 규칙

### Step 영역 규칙

Step은 부모 영역의 **끝에서 확장**됩니다.

| 방향 | 시작점 | 확장 방향 |
|------|--------|-----------|
| `top` | 부모 z_max | 아래로 (z_min 방향) |
| `bottom` | 부모 z_min | 위로 (z_max 방향) |

```python
if direction == 'top':
    cut_z_max = parent_region.z_max
    cut_z_min = cut_z_max - (step_children_height + step_height + margin)
else:  # bottom
    cut_z_min = parent_region.z_min
    cut_z_max = cut_z_min + (step_children_height + step_height + margin)
```

#### Step 영역 시각화

```
[Top Direction]              [Bottom Direction]
┌───────────────┐ z_max      ┌───────────────┐ z_max
│ ┌───────┐     │            │               │
│ │ Step  │     │            │               │
│ │(깎임) │     │            │               │
│ └───────┘     │            │ ┌───────┐     │
│               │            │ │ Step  │     │
│               │            │ │(깎임) │     │
└───────────────┘ z_min      └─┴───────┴─────┘ z_min
```

### Groove 영역 규칙

Groove는 부모 영역 **내부에 배치**됩니다.

| 배치 조건 | 위치 결정 방식 |
|-----------|----------------|
| 단일 Groove | 부모 영역 중앙 |
| 형제 Groove (N개) | Z축으로 균등 분산 |

```python
def _apply_groove_bottomup(self, node, parent_region, groove_index=0, total_grooves=1):
    if total_grooves == 1:
        # 단일: 중앙 배치
        center_z = (parent_region.z_min + parent_region.z_max) / 2
        zpos = center_z - groove_width / 2
    else:
        # 복수: 분산 배치
        available = parent_region.z_max - parent_region.z_min - 2 * margin
        spacing = available / total_grooves
        zpos = parent_region.z_min + margin + groove_index * spacing
```

#### Groove 배치 시각화

```
[단일 Groove]                [형제 Groove 2개]
┌───────────────┐            ┌───────────────┐
│               │            │   [Groove 1]  │  ← index=0
│   [Groove]    │  ← 중앙    │───────────────│
│               │            │   [Groove 2]  │  ← index=1
└───────────────┘            └───────────────┘
```

### Step과 Groove 형제 관계

Step과 Groove가 같은 부모의 자식일 때:

```
Parent Step 영역
┌─────────────────────────────┐ z_max
│                             │
│   ┌─────────┐               │  Groove: 부모 영역 내부
│   │ Groove  │               │  (Step 영역과 무관)
│   └─────────┘               │
│                             │
│   ┌─────────────────────┐   │  Step Child: 부모 영역 끝에서 확장
│   │    Child Step       │   │  (Groove 영역과 무관)
│   │                     │   │
│   └─────────────────────┘   │
└─────────────────────────────┘ z_min

핵심: Step 자식과 Groove 자식은 서로의 영역을 침범하지 않음
- Step은 Z축 방향으로 **확장** (끝에서 새 영역 생성)
- Groove는 기존 영역 **내부**에 배치 (반경 방향으로만 파고듦)
```

---

## 방향 처리 로직

### Base 노드의 자식 방향 할당

```python
# Base의 자식 Step들에 교대로 방향 할당
directions = ['top', 'bottom']
step_index = 0
for child in node.children:
    if child.label == 's':
        child_direction = directions[step_index % 2]
        step_index += 1
```

| 자식 순서 | 방향 | 설명 |
|-----------|------|------|
| 1번째 Step | `top` | 위쪽에서 깎임 |
| 2번째 Step | `bottom` | 아래쪽에서 깎임 |

### Step 체인에서 방향 유지

```
b(s, s(s(s)))

Base
├── Step #1 (top)       ← 첫 번째 자식
└── Step #2 (bottom)    ← 두 번째 자식
    └── Step (bottom)   ← 부모 방향 유지
        └── Step (bottom)
```

---

## Groove 검증 및 재시도

Boolean Cut이 실패할 수 있으므로 검증 로직이 포함됩니다.

```python
def _apply_groove_with_validation(self, node, parent_region, 
                                   groove_index=0, total_grooves=1, max_retries=3):
    for attempt in range(max_retries):
        result = self._apply_groove_bottomup(node, parent_region, 
                                              groove_index, total_grooves)
        
        # Boolean Cut 결과 검증
        if self.shape is not None and not self.shape.IsNull():
            return True
        
        # 실패 시 위치 조정 후 재시도
        if attempt < max_retries - 1:
            groove_index += random.uniform(-0.1, 0.1)  # 미세 조정
    
    return False
```

---

## API 레퍼런스

### TreeTurningGenerator

```python
class TreeTurningGenerator:
    def __init__(self, params: TurningParams = None):
        """생성기 초기화"""
    
    def generate_from_tree(self, root: TreeNode, apply_edge_features: bool = False) -> TopoDS_Shape:
        """
        트리 구조에서 3D 형상 생성
        
        Args:
            root: 트리 루트 노드
            apply_edge_features: 챔퍼/라운드 적용 여부
        
        Returns:
            생성된 OCC TopoDS_Shape
        """
```

### 내부 메서드

| 메서드 | 설명 |
|--------|------|
| `_calculate_required_space(node)` | Bottom-Up 필요 공간 계산 |
| `_create_stock_from_requirements(root)` | Stock 크기 결정 |
| `_process_steps_only(node, parent_region, direction)` | Stage 1: Step만 처리 |
| `_process_grooves_only(node, parent_region)` | Stage 2: Groove만 처리 |
| `_apply_step_bottomup(node, parent_region, direction)` | Step Boolean Cut |
| `_apply_groove_bottomup(node, parent_region, index, total)` | Groove Boolean Cut |
| `_apply_groove_with_validation(...)` | 검증 포함 Groove 적용 |
| `_count_grooves_in_tree(node)` | 트리 내 Groove 개수 |

---

## 예시: `b(s(g,s(g)))` 트리 처리

### 트리 구조
```
b (Base)
└── s (Step, top)
    ├── g (Groove)
    └── s (Step, top)
        └── g (Groove)
```

### Phase 1: 필요 공간 계산 (Bottom-Up)

```
1. 리프 노드들:
   - Groove (깊이 1): height=2.5, depth=0.6
   - Groove (깊이 3): height=2.5, depth=0.6

2. Step (깊이 2):
   - Step 자식 없음, Groove 자식 1개
   - step_children_height = 0  (Groove는 제외!)
   - total_height = 0 + 3.0 + 1.0 = 4.0

3. Step (깊이 1):
   - Step 자식 1개 (height=4.0), Groove 자식 1개
   - step_children_height = 4.0  (Step만 포함!)
   - total_height = 4.0 + 3.0 + 1.0 = 8.0

4. Base:
   - step_children_height = 8.0
   - total_height = 8.0
```

### Phase 2: Stock 생성

```
Stock height = 8.0 + 5.0(margin) = 13.0mm
Stock radius = depth_sum + 5.0 + 3.0 = ~10.0mm
```

### Phase 3: 형상 생성 (Top-Down)

```
Stage 1: Step 처리
1. Step (top): z=[5.0, 13.0], r=8.5
2. Step (top): z=[5.0, 9.0], r=7.0

Stage 2: Groove 처리
1. Groove 1: Step 1 영역 내, 중앙 배치
2. Groove 2: Step 2 영역 내, 중앙 배치
```

---

## 버전 히스토리

### v1.6 (현재)
- **Step-Groove 영역 분리 로직 개선**:
  - `_calculate_required_space()`: Step 자식과 Groove 자식 분리 계산
  - `_apply_step_bottomup()`: Groove 자식 높이를 제외한 Step 자식 높이만 사용
  - Step은 Step 자식 공간만 확보, Groove는 부모 영역 내부 배치 보장

### v1.5
- 2단계 처리 방식 도입 (Step 먼저, Groove 나중)
- Groove 검증 및 재시도 로직 추가
- Groove 분산 배치 (형제 Groove Z축 분산)

### v1.4
- Bottom-Up 파라미터 결정 방식 도입
- RequiredSpace 데이터 클래스 추가
- 모든 피처 스킵 없이 생성 보장

### v1.3
- Step 방향 처리 (top/bottom)
- Base 자식 Step 교대 방향 할당

### v1.0
- 초기 구현 (Top-Down 방식)
