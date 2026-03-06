# 터닝 특징형상 생성 검증 및 연산 중단 문제 해결

날짜: 2026-03-05

---

## 1. generated_faces 수 검증 도입

### 문제

Step과 Groove Boolean Cut이 적용된 후 형상이 기하학적으로 비정상인 경우를 감지할 수단이 없었음. OCC Boolean 연산이 성공(IsDone=True)을 반환해도 결과 형상이 잘못 생성되는 경우가 존재함.

### 해결

`DesignOperation.generated_faces` (Boolean Cut으로 새로 생긴 face 목록)를 활용해 적용 직후 face 수를 검증하도록 `_apply_cut` 내부에 추가.

- **Step**: inner 원통면 1개 + 링 평면(단차) 1개 → `expected_new_faces = 2`
- **Groove**: inner 원통면 1개 + 상단 링 1개 + 하단 링 1개 → `expected_new_faces = 3`

불일치 시 Warning을 출력하여 비정상 형상을 조기에 인지할 수 있게 함.

### 관련 파일

- `core/turning/features.py` — `_apply_cut`, `apply_step_cut`, `apply_groove_cut`

---

## 2. face 수 불일치 시 연산 중단 문제

### 문제

`_apply_cut`에서 `generated_faces` 수가 기댓값과 다를 때 `return None`을 반환하도록 구현함. 이로 인해 다음 연쇄 문제가 발생:

1. features.py에서 None 반환 → Planner에서 `_register_z_range` 미호출 → z 범위가 미점유로 남음
2. 하지만 `node.region`은 이미 축소된 채로 남아 있어 자식/형제 노드들이 부정확한 region 기반으로 배치 시도
3. 기하학적으로 겹치거나 잘못된 위치에 groove가 배치되면서 이후 모든 Boolean Cut이 연쇄적으로 실패
4. 결과적으로 터닝 생성 중간에 모든 후속 특징형상이 누락된 불완전한 형상이 생성됨

### 원인 분석

face 수 불일치는 "비정상적인 형상이 생겼을 수 있음"을 나타내는 진단 정보이지, 연산 자체가 실패한 것이 아님. OCC Boolean Cut이 IsDone=True로 성공하고 유효한 형상을 반환한 경우라면 계속 진행하는 것이 맞음.

Planner의 z 범위 충돌 검사는 상위 레벨에서 기하학적 겹침을 방지하는 역할이고, features.py의 face 수 검증은 하위 레벨의 진단 도구 역할. 두 레벨의 책임을 혼용하면 Planner가 알지 못하는 상태에서 shape가 되돌아가는 불일치가 발생.

### 해결

`_apply_cut`에서 face 수 불일치 시 `return None` 유지 (올바른 감지 동작).

대신 **Planner에서 연산 실패(`result is None`) 시 두 가지를 명확히 처리**하도록 수정:
1. `node.region`을 `parent_region`으로 복구 → 잘못된 축소된 region을 자식에게 물려주지 않음
2. 해당 노드의 자식 순회 건너뜀 (`step_applied` / `groove_applied` 플래그) → 잘못된 region 기반의 연쇄 연산 차단

```python
# 수정 전: 실패해도 node.region이 축소된 채로 자식 순회 → 연쇄 실패
result = apply_step_cut(...)
if result is not None:
    shape = result
else:
    print("[Warning] ...")
# 실패와 관계없이 자식 순회 계속됨

# 수정 후: 실패 시 region 복구 + 자식 스킵
result = apply_step_cut(...)
if result is not None:
    shape = result
    step_applied = True
else:
    node.region = parent_region   # region 복구
    print("[Warning] ... 자식 노드 스킵")

if step_applied:
    # 자식 순회 (성공했을 때만)
    for child in step_children: ...
```

### 관련 파일

- `core/turning/features.py` — `_apply_cut` (`return None` 복구)
- `core/turning/planner.py` — `_apply_step_cuts`, `_apply_groove_cuts` (실패 시 region 복구 + 자식 스킵)

---

## 3. 터닝 전용 시각화 모드 추가

### 문제

`view_labeling.py`가 항상 밀링까지 포함한 전체 파이프라인을 실행하여, 터닝 결과만 단독으로 확인하기 어려웠음.

### 해결

`main()` 상단에 `TURNING_ONLY` 플래그 추가. `True`로 설정하면 `enable_milling=False`로 파이프라인을 실행하여 Step / Groove / Chamfer / Fillet 라벨만 확인 가능.

```python
# view_labeling.py main() 상단
TURNING_ONLY = False   # True: 터닝만, False: 터닝 + 밀링
```

### 관련 파일

- `view_labeling.py` — `TURNING_ONLY` 플래그, `build_params`, `generate_models`
