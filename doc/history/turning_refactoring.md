# Turning 모듈 리팩토링 이력

## 완료된 작업 (2026-02-27)

### Q2 - groove_margin 절대값 통일
- **변경 전**: `TurningParams.groove_margin_ratio = 0.15` (groove 폭 대비 비율)
- **변경 후**: `TurningParams.groove_margin = 0.5` (step_margin과 동일한 절대값)
- **변경 파일**: `core/turning/planner.py` (TurningParams 정의 + _calculate_required_space 로직)
- **로직 변경**:
  ```
  # 변경 전: min_width = groove_children_height / usable_ratio (역산 방식)
  # 변경 후: min_width = groove_children_height + 2 * node_margin (step과 동일)
  ```

### Q3 - TurningParams 위치 이동
- **변경 전**: `core/turning/features.py`에 정의
- **변경 후**: `core/turning/planner.py`로 이동 (플래너 설정 객체가 플래너 모듈에 위치)
- `apply_edge_features` 시그니처 변경:
  ```
  # 변경 전: apply_edge_features(shape, params: TurningParams, label_maker)
  # 변경 후: apply_edge_features(shape, edge_feature_prob, chamfer_range, fillet_range, label_maker)
  ```
- **영향 파일**: `pipeline.py`, `core/turning/__init__.py`, `tests/turning/test_features.py`, `tests/turning/test_planner.py`

### Q4 - TurningFeatureRequest 공개 API 제거
- `planner.plan_and_apply(root, label_maker)` 메서드 추가
- `pipeline.py`에서 plan → create_stock → initialize → apply_turning_requests 4단계를 `plan_and_apply` 1호출로 단순화
- `TurningFeatureRequest`는 `features.py` 내부 구현으로만 사용 (외부 직접 의존 불필요)
- **영향 파일**: `core/turning/planner.py`, `pipeline.py`

---

## 잔여 작업 (우선순위 순)

### [P1] step_margin 단방향 수정
- **파일**: `core/turning/planner.py` - `_calculate_required_space`
- **문제**: `step_based_height = step_children_height + feature_height + 2 * node_margin`
  - 스텝은 항상 `parent_region.z_max`(stock 끝)에서 시작하므로 끝 방향 마진은 무의미
  - `2 * margin` → `1 * margin`으로 줄이는 것이 의미상 정확
- **수정 방향**:
  ```python
  # 변경 전
  step_based_height = step_children_height + feature_height + 2 * node_margin
  # 변경 후
  step_based_height = step_children_height + feature_height + node_margin
  ```

### [P2] 다중 스텝 자식 겹침 문제
- **파일**: `core/turning/planner.py` - `_collect_step_requests`
- **문제**: 스텝 노드가 여러 스텝 자식을 가지면 모두 동일한 `direction`과 `z_max`를 공유하여
  여러 자식이 겹치는 절삭을 생성함. Boolean Cut 결과 형상은 만들어지지만
  "각 스텝이 독립된 계단을 만든다"는 트리의 의미가 깨짐.
- **수정 방향**: 스텝 자식이 여러 명일 경우 루트처럼 교번 방향 부여(top/bottom)하거나,
  또는 자식 스텝은 방향 반전 후 서브 영역에서 배치

### [P3] min_remaining_radius 중복 적용 검증
- **파일**: `core/turning/planner.py` - `_make_step_request`, `_make_groove_request`
- **문제**: Step이 반지름을 `r_step`으로 줄인 후 Groove가 같은 z 구간에 추가 절삭 시,
  Groove의 `parent_region.radius`가 Step 적용 전 원본 반지름 기준인지 Step 후 반지름 기준인지 검증 필요.
  실제 잔여 반지름이 `min_remaining_radius` 이하로 떨어질 수 있는 경로 존재 가능.
- **검증 방법**: Step + Groove가 동일 z 구간에 겹치는 트리를 생성하여
  최종 `inner_radius` 추적

### [P4] Stock 크기 상한 제약
- **파일**: `core/turning/planner.py` - `TurningParams`, `plan()`
- **문제**: 트리 깊이/노드 수에 따라 Stock 크기가 무한히 커져 데이터 품질 편차 발생
- **수정 방향**: `TurningParams`에 `stock_height_max`, `stock_radius_max` 추가 후
  `plan()` 내에서 clamp 적용:
  ```python
  self._stock_height = min(required.height + height_margin, self.params.stock_height_max)
  self._stock_radius = min(required.depth + ..., self.params.stock_radius_max)
  ```
  또는 트리 생성 단계에서 노드 수/깊이 상한을 제한하는 방식으로 처리
