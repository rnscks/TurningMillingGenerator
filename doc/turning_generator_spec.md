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

### RequiredSpace (필요 공간)

Bottom-Up 계산 시 노드가 필요로 하는 공간입니다. **margin을 포함하여 한 곳에서만 계산**됩니다.

```python
@dataclass
class RequiredSpace:
    height: float           # 필요한 z 높이 (margin 포함 총 크기)
    depth: float            # 필요한 반경 깊이 (누적)
    feature_height: float   # 자신의 피처 높이 (step height 또는 groove 커팅 폭)
    feature_depth: float    # 자신의 피처 깊이
    margin: float = 0.0     # 자신의 z방향 margin (양쪽 각각)
```

#### margin 단일 소스 원칙

margin은 `_calculate_required_space()`에서 **한 번만 계산**되어 `RequiredSpace.margin`에 저장됩니다.
배치 시에는 저장된 값을 그대로 사용하며, **재계산하지 않습니다**.

| 노드 타입 | margin 계산 | 저장 위치 |
|-----------|------------|-----------|
| Base (b) | `0.0` (Stock 자체가 최외곽) | `RequiredSpace.margin` |
| Step (s) | `step_margin` (고정값, 기본 0.5) | `RequiredSpace.margin` |
| Groove (g) | `feature_height * groove_margin_ratio` (비율 기반) | `RequiredSpace.margin` |

### TurningParams (생성 파라미터)

```python
@dataclass
class TurningParams:
    # Stock margin
    stock_height_margin: tuple = (3.0, 8.0)
    stock_radius_margin: tuple = (2.0, 5.0)
    min_remaining_radius: float = 2.0
    
    # Step 파라미터
    step_depth_range: tuple = (0.8, 1.5)         # 반경 방향 깊이
    step_height_range: tuple = (2.0, 4.0)        # Z 방향 높이
    step_margin: float = 0.5                      # Step 고정 margin
    
    # Groove 파라미터
    groove_depth_range: tuple = (0.4, 0.8)       # 반경 방향 깊이
    groove_width_range: tuple = (1.5, 3.0)       # Z 방향 폭
    groove_margin_ratio: float = 0.15            # Groove 비율 기반 margin (feature_height 대비)
    
    # 챔퍼/라운드
    chamfer_range: tuple = (0.3, 0.8)
    fillet_range: tuple = (0.3, 0.8)
    edge_feature_prob: float = 0.3
```

---

## 핵심 알고리즘: Bottom-Up 파라미터 결정

### 처리 흐름

```
[Phase 1: 리프 → 루트] 필요 공간 + margin 계산 (단일 소스)
       ↓
[Phase 2: 루트] Stock 크기 결정  
       ↓
[Phase 3: 루트 → 리프] 형상 생성 (저장된 margin 사용, 재계산 없음)
```

### Phase 1: 필요 공간 계산 (`_calculate_required_space`)

리프 노드부터 시작하여 각 노드의 필요 공간과 margin을 상향 전파합니다.

```python
def _calculate_required_space(self, node: TreeNode) -> RequiredSpace:
    # 1. 모든 자식의 필요 공간을 먼저 계산 (재귀)
    for child in node.children:
        self._calculate_required_space(child)
    
    # 2. Step 자식과 Groove 자식 분리
    step_children_height = sum(c.required_space.height for c in step_children)
    groove_children_height = sum(c.required_space.height for c in groove_children)
    
    # 3. 노드 타입별 크기 + margin 결정
    if node.label == 'b':
        node_margin = 0.0
        total_height = max(step_children_height, groove_children_height)
        
    elif node.label == 's':
        node_margin = step_margin  # 고정값
        step_based = step_children_height + feature_height + 2 * node_margin
        total_height = max(step_based, groove_children_height)
        
    elif node.label == 'g':
        # 자식 groove를 수용할 수 있도록 커팅 폭 결정
        usable_ratio = 1.0 - 2.0 * groove_margin_ratio
        min_width = groove_children_height / usable_ratio
        feature_height = max(random_width, min_width)
        node_margin = feature_height * groove_margin_ratio
        total_height = feature_height + 2 * node_margin
    
    # margin은 여기서 한 번만 저장 → 이후 재계산 없음
    node.required_space = RequiredSpace(
        height=total_height, depth=...,
        feature_height=feature_height, feature_depth=...,
        margin=node_margin
    )
```

#### 높이 계산 규칙

| 노드 | total_height | 의미 |
|------|-------------|------|
| Base | `max(step_children, groove_children)` | Step과 Groove 모두 수용 |
| Step | `max(step_based, groove_children)` | Step 자식 + Groove 자식 모두 수용 |
| Groove | `feature_height + 2 * margin` | 커팅 폭 + 양쪽 margin |

#### Groove 커팅 폭 결정 규칙

자식 groove가 있으면, 자식들의 총 필요 높이를 내부에 수용할 수 있는 폭 보장:

```
feature_height >= groove_children_height / (1 - 2 * ratio)

예시 (ratio=0.15):
  자식 총 필요 = 5.0mm
  → feature_height >= 5.0 / 0.7 = 7.14mm
  → margin = 7.14 * 0.15 = 1.07mm (양쪽)
  → 내부 사용 가능 = 7.14 - 2.14 = 5.0mm ✓
```

### Phase 2: Stock 크기 결정 (`_create_stock_from_requirements`)

```python
stock_height = root.required_space.height + random(stock_height_margin)
stock_radius = root.required_space.depth + min_remaining_radius + random(stock_radius_margin)
```

### Phase 3: 형상 생성 (2단계)

```
Stage 1: Step 처리 (_process_steps_only)
         └── 모든 Step Boolean Cut + Region 확정
         └── Step 커팅 크기 = required.height (groove 자식 포함)

Stage 2: Groove 처리 (_process_grooves_only)
         └── 확정된 Region에 Groove 배치
         └── margin = required.margin (저장된 값 사용)
```

---

## 피처 영역 규칙

### Step 영역 규칙

Step은 부모 영역의 **끝에서 확장**됩니다. 커팅 크기는 `required.height`를 직접 사용합니다.

```python
# required.height = max(step_based, groove_children) → 이미 groove 자식 공간 포함
cut_size = required.height

if direction == 'top':
    cut_z_max = parent_region.z_max
    cut_z_min = cut_z_max - cut_size
else:
    cut_z_min = parent_region.z_min
    cut_z_max = cut_z_min + cut_size
```

### Groove 영역 규칙

Groove는 부모 영역 **내부에 배치**됩니다. margin은 `required.margin`에서 읽습니다.

#### 단일 Groove: 중앙 배치

```python
groove_margin = required.margin  # 저장된 값 (재계산 없음)
center_z = (parent.z_min + parent.z_max) / 2
zpos = center_z - groove_width / 2
zpos = clamp(zpos, parent.z_min + groove_margin, parent.z_max - groove_width - groove_margin)
```

#### 형제 Groove: zone 기반 순차 배치

각 groove의 **zone = margin + width + margin** (= `required.height`)을 순차 배치합니다.

```python
# 모든 값은 bottom-up에서 이미 계산됨 → 재계산 없음
sibling_zones = [sib.required_space.height for sib in siblings]
sibling_margins = [sib.required_space.margin for sib in siblings]

# 여유 공간을 균등 gap으로 분배
remaining = parent_height - sum(sibling_zones)
gap = remaining / (N + 1)

# 순차 배치: gap + [margin + width + margin] + gap + ...
zpos = parent.z_min + gap
for i in range(groove_index):
    zpos += sibling_zones[i] + gap
zpos += sibling_margins[groove_index]  # leading margin
```

#### Zone 기반 배치 시각화

```
부모 영역 (Step 또는 Groove)
┌────────────────────────────────────────┐
│ gap │ m │ Groove 1 │ m │ gap │ m │ Groove 2 │ m │ gap │
└────────────────────────────────────────┘

m = 각 groove의 required.margin (groove_width * ratio)
gap = (부모 높이 - 전체 zone 합) / (N+1)

→ 단차쌍(벽) = groove margin × 2 (인접한 두 groove의 margin 합)
→ 겹침 불가 (zone 단위로 순차 배치)
```

---

## 방향 처리 로직

### Base 노드의 자식 방향 할당

```python
# Base의 자식 Step들에 교대로 방향 할당
for i, child in enumerate(step_children):
    child_dir = 'top' if i % 2 == 0 else 'bottom'
```

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

## API 레퍼런스

### TreeTurningGenerator

```python
class TreeTurningGenerator:
    def __init__(self, params: TurningParams = None):
        """생성기 초기화"""
    
    def generate_from_tree(self, root: TreeNode, apply_edge_features: bool = False) -> TopoDS_Shape:
        """트리 구조에서 3D 형상 생성"""
```

### 내부 메서드

| 메서드 | 설명 |
|--------|------|
| `_calculate_required_space(node)` | Bottom-Up 필요 공간 + margin 계산 (단일 소스) |
| `_create_stock_from_requirements(root)` | Stock 크기 결정 |
| `_process_steps_only(node, parent_region, direction)` | Stage 1: Step만 처리 |
| `_process_grooves_only(node, parent_region)` | Stage 2: Groove만 처리 |
| `_apply_step_bottomup(node, parent_region, direction)` | Step Boolean Cut (required.height 사용) |
| `_apply_groove_with_validation(...)` | Groove Boolean Cut (required.margin 사용, zone 기반 배치) |
| `_count_grooves_in_tree(node)` | 트리 내 Groove 개수 |

---

## 예시: `b(s(s(s(g,g))))` 트리 처리

### 트리 구조
```
b (Base)
└── s (Step, top)
    └── s (Step, top)
        └── s (Step, top)
            ├── g (Groove)
            └── g (Groove)
```

### Phase 1: 필요 공간 + margin 계산 (Bottom-Up)

```
1. 리프 Groove들 (ratio=0.15):
   - g1: feature_height=2.0, margin=0.3, total_height=2.6
   - g2: feature_height=1.8, margin=0.27, total_height=2.34

2. innermost Step:
   - step_based = 0 + 3.0 + 1.0 = 4.0
   - groove_children_height = 2.6 + 2.34 = 4.94
   - total_height = max(4.0, 4.94) = 4.94, margin=0.5

3. 중간 Step, 상위 Step, Base → 상향 전파
```

### Phase 3: 형상 생성 (저장된 margin 사용)

```
Stage 1: Step 처리
  innermost Step 커팅: cut_size = required.height = 4.94  ← groove 공간 포함

Stage 2: Groove 처리 (zone 기반 순차 배치)
  g1 zone = 2.6 (margin=0.3 + width=2.0 + margin=0.3)
  g2 zone = 2.34 (margin=0.27 + width=1.8 + margin=0.27)
  total = 4.94, parent = 4.94, gap = 0

  g1: zpos = z_min + 0 + 0.3 → [z_min+0.3, z_min+2.3]
  g2: zpos = z_min + 2.6 + 0 + 0.27 → [z_min+2.87, z_min+4.67]

  벽 두께:
    시작 벽 = 0.3mm (g1.margin)
    g1↔g2 벽 = 0.57mm (g1.margin + g2.margin)  
    끝 벽 = 0.27mm (g2.margin)
  → 모든 단차쌍 보장
```

---

## 버전 히스토리

### v1.9 (현재)
- **margin 단일 소스 리팩터링**:
  - `RequiredSpace`에 `margin` 필드 추가
  - `_calculate_required_space()`에서 margin을 한 번만 계산하여 저장
  - 모든 배치 코드에서 `required.margin` 사용 (재계산 없음)
  - `_apply_step_bottomup()`: `required.height` 직접 사용 (groove 자식 공간 포함)
  - `_apply_groove_with_validation()`: `required.margin` 사용, zone 기반 순차 배치
  - margin 계산 분산으로 인한 불일치 문제 구조적으로 해결

### v1.8
- **Groove 겹침 방지 및 영역 보장**:
  - Groove `feature_height`가 자식 groove들의 필요 높이를 수용하도록 보장
  - 형제 groove 순차 배치 (각 groove 실제 폭 기반)
  - groove_width가 부모 영역 초과 시 비례 축소

### v1.7
- **Base/Step의 Groove 자식 높이 반영**:
  - Base와 Step에서 Groove 자식 높이를 `total_height`에 반영
  - Groove-only 트리에서 Stock 높이 부족 문제 해결

### v1.6
- **Step-Groove 영역 분리 로직 개선**

### v1.5
- 2단계 처리 방식 도입 (Step 먼저, Groove 나중)
- Groove 검증 및 재시도 로직 추가

### v1.4
- Bottom-Up 파라미터 결정 방식 도입
- RequiredSpace 데이터 클래스 추가

### v1.3
- Step 방향 처리 (top/bottom)

### v1.0
- 초기 구현 (Top-Down 방식)
