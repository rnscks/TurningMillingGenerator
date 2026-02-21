# TurningMillingGenerator 리팩토링 기록

---

## 2025-02-21 — 1차 리팩토링

### 배경

코드 리뷰에서 다음 문제가 확인됨:
- Legacy dead code에 존재하지 않는 필드 참조 (런타임 에러 가능)
- Boolean Cut 결과 null 체크 누락
- 미사용 필드/별칭이 코드베이스에 산재
- 라벨 체계에 `RECTANGULAR_PASSAGE` 누락 (pocket과 구분 불가)
- 재귀 함수에 깊이 제한 없음
- 밀링 피처 동일 면 과다 배치 가능
- `test_labeling.py` 파일명이 pytest 테스트와 혼동
- 레이어 간 의존 방향 역전 (utils→core, core→utils, 파이프라인 우회)

---

### 항목 1. Dead Code / Legacy 메서드 제거

**왜:** `_create_stock()`, `_apply_step()`, `_apply_groove()`, `_process_node_bottomup()`,
`_apply_groove_bottomup()` 등 legacy 메서드가 존재하지 않는 필드(`stock_height_range`,
`stock_radius_range`, `min_base_height`)를 참조. 호출 시 `AttributeError` 발생.
`save()` 메서드는 core→utils 역참조 유발.

**무엇을:**
- `core/turning_generator.py`에서 6개 메서드 제거:
  - `_create_stock()` → `_create_stock_from_requirements()`로 대체됨
  - `_apply_step()` → `_apply_step_bottomup()`로 대체됨
  - `_apply_groove()` → `_apply_groove_with_validation()`로 대체됨
  - `_apply_groove_bottomup()` → `_apply_groove_with_validation()`가 동일 역할
  - `_process_node_bottomup()` → `_process_steps_only()` + `_process_grooves_only()`로 대체됨
  - `save()` → `pipeline.py`에서 처리
- `from utils.step_io import save_step` 임포트 제거

**어떻게:** 호출부 없음 확인 후 메서드와 관련 임포트 삭제.

**결과:** `turning_generator.py` 1045줄 → 785줄 (260줄 감소). dead code 완전 제거.

---

### 항목 2. 버그 수정 — `_cut_with_tracking()` null 체크

**왜:** `label_maker`가 None인 경우의 else 분기에서 `BRepAlgoAPI_Cut` 결과를 검증하지 않고
바로 `self.shape`에 할당. null shape가 전파되면 이후 모든 연산 실패.

**무엇을:** `core/turning_generator.py`의 `_cut_with_tracking()` else 분기

**어떻게:**
```python
# Before
self.shape = BRepAlgoAPI_Cut(self.shape, tool_shape).Shape()

# After
result = BRepAlgoAPI_Cut(self.shape, tool_shape).Shape()
if result is None or result.IsNull():
    raise RuntimeError("Boolean Cut 실패: 결과 형상이 null입니다")
self.shape = result
```

**결과:** null shape 전파 방지. 실패 시 명확한 에러 메시지 출력.

---

### 항목 3. 미사용 필드·로직 정리

**왜:** `hole_probability` 필드가 선언만 되고 어디서도 참조되지 않음.
`HoleParams`, `HolePlacement`는 리네이밍 후 남은 별칭.
`create_hole()`은 호출부 없는 legacy 함수.

**무엇을:**

| 대상 | 위치 | 조치 |
|------|------|------|
| `hole_probability` 필드 | `pipeline.py` | 제거 |
| `HoleParams` 별칭 | `milling_adder.py` | 제거, 전체 `FeatureParams`로 통일 |
| `HolePlacement` 별칭 | `milling_adder.py` | 제거, 전체 `FeaturePlacement`로 통일 |
| `create_hole()` 함수 | `milling_adder.py` | 제거 |
| `hole` 파라미터명 | `pipeline.py`, `run_pipeline.py`, `view_labeling.py` | `feature`로 변경 |

**어떻게:**
- `pipeline.py`: `hole: HoleParams` → `feature: FeatureParams`, `hole_probability` 삭제
- `core/milling_adder.py`: 별칭 2개 + legacy 함수 1개 삭제
- `core/__init__.py`: re-export 목록에서 별칭 제거
- `run_pipeline.py`, `view_labeling.py`: `HoleParams` → `FeatureParams`, `hole=` → `feature=`, `hole_probability` 삭제

**결과:** 코드베이스에서 deprecated 별칭 완전 제거. 네이밍 통일.

---

### 항목 4. 라벨 추가 — RECTANGULAR_PASSAGE

**왜:** `FeatureType.RECTANGULAR_PASSAGE`에 대응하는 라벨이 없어서
`Labels.RECTANGULAR_POCKET`(7)을 재사용. pocket과 passage가 라벨 상 구분 불가.

**무엇을:**
- `core/label_maker.py`: `RECTANGULAR_PASSAGE = 8` 추가, `NAMES` 리스트에 추가
- `config/LABEL_PROPS.json`: `"rectangular_passage": [30, 144, 255]` 추가
- `core/milling_adder.py`: passage 생성 시 `Labels.RECTANGULAR_PASSAGE` 사용

**어떻게:** 라벨 상수, 이름 리스트, 색상 설정 3곳에 일관되게 추가.

**결과:** 최종 라벨 체계 (9종):

| ID | 이름 | 색상 RGB |
|----|------|----------|
| 0 | stock | (105, 105, 105) |
| 1 | step | (0, 206, 209) |
| 2 | groove | (255, 69, 0) |
| 3 | chamfer | (255, 0, 255) |
| 4 | fillet | (57, 255, 20) |
| 5 | blind_hole | (255, 20, 147) |
| 6 | through_hole | (255, 0, 0) |
| 7 | rectangular_pocket | (0, 0, 139) |
| 8 | rectangular_passage | (30, 144, 255) |

---

### 항목 5. 재귀 안전장치

**왜:** `turning_generator.py`의 재귀 함수들에 깊이 제한이 없음.
파라미터 설정에 따라 `RecursionError` 가능.

**무엇을:** 5개 재귀 함수에 `MAX_RECURSION_DEPTH = 50` 제한 추가:
- `_calculate_required_space()`
- `_process_steps_only()`
- `_process_grooves_only()`
- `_assign_groove_regions()`
- `_count_grooves_in_tree()`

**어떻게:** 각 함수에 `_depth` 파라미터 추가, 초과 시 `RecursionError` 발생.
재귀 호출부에서 `_depth + 1` 전달.

**결과:** 무한 재귀 방지. 초과 시 노드 정보를 포함한 에러 메시지 출력.

---

### 항목 6. 밀링 피처 중복 배치 방지

**왜:** `add_milling_features()`에서 면 분석을 1회만 수행.
Boolean Cut 후에도 동일 `ValidFaceInfo`를 재사용하여
같은 면에 `max_features_per_face` 이상 배치 가능.

**무엇을:** `core/milling_adder.py`의 `add_milling_features()` 내부

**어떻게:** `face_usage_count: dict` 추가.
피처 배치 성공마다 `face_id` 카운트 증가.
루프 시작과 내부에서 `max_features_per_face` 초과 시 해당 면 건너뜀.

**결과:** 기존 `min_spacing` 거리 체크 + face_id 기반 사용 횟수 추적 이중 보호.

---

### 항목 7. 파일명 정리

**왜:** `test_labeling.py`는 pytest 테스트가 아닌 OCC Viewer 시각화 스크립트.
`test_` 접두사가 혼동 유발.

**무엇을:** `test_labeling.py` → `view_labeling.py`로 이름 변경

**어떻게:** `git mv` 후 docstring 수정 (제목, 사용법).

**결과:** 파일 용도가 이름에서 명확히 드러남.

---

### 항목 8. 아키텍처 정리 — 레이어 역전 해소

**왜:** 3가지 레이어 역전이 존재:
1. `utils/tree_io.py` → `core/tree_generator.py` (인프라가 도메인에 의존)
2. `core/turning_generator.py` → `utils/step_io.py` (도메인이 인프라에 의존)
3. `run_pipeline.py`가 `TurningMillingGenerator.generate_from_tree()` 대신
   내부 컴포넌트를 직접 호출 (파이프라인 우회, `enable_labeling` 등 설정 무시)

추가로 `pipeline.py`의 불필요한 re-export도 존재.

**무엇을:**

| 역전 | 조치 |
|------|------|
| `utils/tree_io.py` → `core` | `generate_trees()`, `generate_and_save_trees()` 를 `utils`에서 제거. 호출부에서 `core.generate_trees` 직접 임포트 |
| `core/turning_generator.py` → `utils` | 항목 1에서 `save()` 메서드 + 임포트 제거로 해소 완료 |
| `run_pipeline.py` 파이프라인 우회 | `generator.generate_from_tree(tree)` 단일 호출로 변경 |
| `pipeline.py` 불필요 re-export | `from utils.tree_io import ...` 제거 |

**어떻게:**
- `utils/tree_io.py`: 하단 `generate_trees()`, `generate_and_save_trees()` 삭제 (70줄)
- `utils/__init__.py`: re-export 목록에서 해당 함수 제거
- `run_pipeline.py`: `from core import generate_trees` 임포트, `run_generation_pipeline()` 내부를 `generator.generate_from_tree()` 호출로 단순화
- `view_labeling.py`: 동일하게 `from core import generate_trees`로 변경

**결과:** 의존 관계가 올바른 방향으로 정리됨:
```
Application (run_pipeline.py, pipeline.py)
     ↓
Domain (core/)          ← 외부 의존 없음
     ↓
Infrastructure (utils/) ← core에서 임포트하지 않음
```

---

### 변경 파일 요약

| 파일 | 변경 내용 |
|------|----------|
| `core/turning_generator.py` | legacy 메서드 6개 제거, null 체크 추가, 재귀 안전장치 5개 함수 |
| `core/milling_adder.py` | 별칭 2개 + legacy 함수 제거, passage 라벨 수정, 중복 배치 방지 |
| `core/label_maker.py` | `RECTANGULAR_PASSAGE = 8` 추가 |
| `core/__init__.py` | 별칭 re-export 제거 |
| `config/LABEL_PROPS.json` | `rectangular_passage` 색상 추가 |
| `pipeline.py` | `hole` → `feature`, `hole_probability` 제거, 불필요 re-export 제거 |
| `run_pipeline.py` | `FeatureParams` 통일, 파이프라인 우회 제거, `core`에서 직접 임포트 |
| `view_labeling.py` | 파일 리네임 + `FeatureParams` 통일 + 임포트 경로 변경 |
| `utils/tree_io.py` | `generate_trees`, `generate_and_save_trees` 제거 |
| `utils/__init__.py` | re-export 목록 정리 |

---

### 후속 과제 (다음 리팩토링)

| 우선순위 | 항목 | 설명 |
|---------|------|------|
| 1 | `design_operation.py` 테스트 작성 | Face History 추적 검증 |
| 1 | `label_maker.py` 테스트 작성 | 라벨 전파 규칙 검증 |
| 2 | `milling_adder.py` 테스트 작성 | 피처 스케일 계산, 배치 로직 검증 |
| 2 | `face_analyzer.py` 테스트 작성 | 면 치수 분석 검증 |
| 3 | `pipeline.py` 통합 테스트 | 전체 파이프라인 E2E 검증 |
| 3 | `utils/tree_io.py` 테스트 작성 | I/O, 분류/필터 검증 |
