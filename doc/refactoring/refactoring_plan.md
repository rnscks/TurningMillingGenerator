# TurningMillingGenerator 리팩토링 기록

---

## 2025-02-21 — 1차: Dead Code 제거, 버그 수정, 아키텍처 정리

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
| ~~1~~ | ~~`design_operation.py` 테스트 작성~~ | → 2025-02-21 2차 리팩토링에서 완료 |
| ~~1~~ | ~~`label_maker.py` 테스트 작성~~ | → 2025-02-21 2차 리팩토링에서 완료 |
| ~~2~~ | ~~`milling_adder.py` 테스트 작성~~ | → 2025-02-21 2차 리팩토링에서 완료 |
| ~~2~~ | ~~`face_analyzer.py` 테스트 작성~~ | → 2025-02-21 2차 리팩토링에서 완료 |
| ~~3~~ | ~~`pipeline.py` 통합 테스트~~ | → 2025-02-21 2차 리팩토링에서 완료 |
| ~~3~~ | ~~`utils/tree_io.py` 테스트 작성~~ | → 2025-02-21 2차 리팩토링에서 완료 |

---

## 2025-02-21 — 2차: 미테스트 모듈 6개 테스트 작성 (98 케이스)

### 배경

1차 리팩토링 후 핵심 모듈 6개에 테스트가 없는 상태.
버그 재발 방지 및 향후 변경 시 회귀 검증을 위해 테스트 작성.

---

### 항목 1. `design_operation.py` 테스트 (1순위)

**왜:** Face History 추적 (Modified/Generated/Deleted)은 라벨링 시스템의 기반.
이 로직에 버그가 있으면 모든 라벨이 틀어짐.

**무엇을:** `tests/test_design_operation.py` 신규 작성

**어떻게:**
- `collect_faces`: 원기둥 3면, 박스 6면 검증
- `search_same_face`: 동일 shape 내 검색, 다른 shape 미검색
- `DesignOperation.cut()`: 결과 유효성, origin_faces 기록, processed/generated/modified 추적 완전성
- `DesignOperation.chamfer()/fillet()`: 결과 유효성, 새 면 생성, 추적 완전성
- 실패 케이스: 교차하지 않는 도구로 Cut

**결과:** 16개 테스트 케이스

---

### 항목 2. `label_maker.py` 테스트 (1순위)

**왜:** 라벨 전파 규칙(Modified→원본 유지, Generated→신규 부여)이 핵심 도메인 로직.
상수 정합성(Labels ↔ LABEL_PROPS.json)도 검증 필요.

**무엇을:** `tests/test_label_maker.py` 신규 작성

**어떻게:**
- `Labels`: ID 순차성, NAMES 인덱스 일치 (9개 라벨 전수 검증)
- `LabelMaker.initialize()`: 전체 face 라벨링, 기본값, 커스텀 라벨
- `LabelMaker.update_label()`: Generated 면 새 라벨, Modified 면 원본 유지, 총 face 수 일치
- 연속 연산: Step → Hole 순차 적용 시 라벨 누적 및 Step 라벨 보존
- 엣지 케이스: 빈 상태, 재초기화, 범위 외 라벨 ID

**결과:** 15개 테스트 케이스

---

### 항목 3. `milling_adder.py` 테스트 (2순위)

**왜:** 피처 스케일 계산, 유효 면 판단, 4종 피처 생성이 모두 검증 필요.

**무엇을:** `tests/test_milling_adder.py` 신규 작성

**어떻게:**
- `compute_hole_scale_range()`: 유효 범위, None/0 입력, clearance 초과, min > max 케이스
- 피처 생성 함수 4종: `create_blind_hole`, `create_through_hole`, `create_rectangular_pocket`, `create_rectangular_passage` 각각 결과 유효성
- `MillingFeatureAdder`: 면 분석, Plane 필터링, 피처 추가, max_total_holes 준수, feature_types 필터, 유효 면 없는 경우

**결과:** 17개 테스트 케이스

---

### 항목 4. `face_analyzer.py` 테스트 (2순위)

**왜:** 면 치수 계산이 밀링 피처 배치의 입력. 잘못되면 피처 크기/위치 전체에 영향.

**무엇을:** `tests/test_face_analyzer.py` 신규 작성

**어떻게:**
- `get_surface_type()`: Cylinder/Plane/Cone 타입 분류
- `is_z_aligned_plane()`: 상하면 검출, 측면 비검출
- `FaceAnalyzer.analyze_shape()`: 원기둥(3면), 치수값(width≈2R, height≈H), 원추, Step 형상, Ring 감지
- `FaceAnalyzer.get_valid_faces()`: 최소 치수 필터, 타입 필터
- 포인트 유틸리티: `sample_edge_points` 개수, `points_to_rz` 변환, `analyze_rz_points` 빈 입력

**결과:** 17개 테스트 케이스

---

### 항목 5. `utils/tree_io.py` 테스트 (3순위)

**왜:** 트리 I/O와 분류/필터 함수는 파이프라인 입력 단계. OCC 의존 없는 순수 Python 테스트.

**무엇을:** `tests/test_tree_io.py` 신규 작성

**어떻게:**
- `save_trees`/`load_trees`: 라운드트립, 부모 디렉토리 자동 생성, dict/list 형식 로드, 미존재 파일, 잘못된 형식
- `classify_trees_by_step/groove_count`: 분류 결과, 빈 리스트, 인덱스 유효성, 전수 분류
- `get_tree_stats`: 기본 통계, 양방향 Step 감지, 형제 Groove 감지, 중첩 Groove 비형제 판별
- `find_bidirectional_step_trees`, `find_sibling_groove_trees`: 탐색 결과
- `filter_trees`: 단일 조건, 복합 조건, 무조건 전체 반환, 불가능 조건

**결과:** 22개 테스트 케이스

---

### 항목 6. `pipeline.py` 통합 테스트 (3순위)

**왜:** 전체 파이프라인(트리→터닝→밀링→라벨링) 흐름이 정상 동작하는지 E2E 검증.

**무엇을:** `tests/test_pipeline.py` 신규 작성

**어떻게:**
- `TurningMillingParams`: 기본값, 커스텀 설정
- 기본 동작: 터닝 전용, 밀링 포함, 여러 트리 성공률
- 라벨링: 활성/비활성 시 label_maker 상태, stock 라벨 존재, 밀링 피처 라벨 추가
- `get_generation_info()`: 필드 존재, 피처 정보 구조

**결과:** 11개 테스트 케이스

---

### 변경 파일 요약

| 파일 | 신규 | 테스트 수 |
|------|------|----------|
| `tests/test_design_operation.py` | O | 16 |
| `tests/test_label_maker.py` | O | 15 |
| `tests/test_milling_adder.py` | O | 17 |
| `tests/test_face_analyzer.py` | O | 17 |
| `tests/test_tree_io.py` | O | 22 |
| `tests/test_pipeline.py` | O | 11 |
| **합계** | | **98** |

기존 테스트 2개 (`test_tree_generator.py`, `test_turning_generator.py`) 포함 총 8개 테스트 파일.

### 검증 결과

```
pytest tests/ -v
======= 170 passed, 285 warnings in 4.40s =======
```

- 170개 전체 PASSED
- warnings는 pythonocc-core SWIG 바인딩의 `DeprecationWarning` (우리 코드 무관)

---

## 리팩토링 현황 요약

### 완료된 항목

| 날짜 | 차수 | 내용 | 상태 |
|------|------|------|------|
| 2025-02-21 | 1차 | Dead Code 제거, 버그 수정, 미사용 정리, 라벨 추가, 재귀 안전장치, 중복 방지, 파일명 정리, 아키텍처 정리 (8개 항목) | 완료 |
| 2025-02-21 | 2차 | 미테스트 모듈 6개 테스트 작성 (98 케이스) | 완료 (170 passed) |

### 남은 항목

| 우선순위 | 항목 | 설명 |
|---------|------|------|
| 낮음 | `pytest.ini` 추가 | SWIG DeprecationWarning 필터링, 테스트 기본 옵션 설정 |
| 낮음 | 의존성 관리 파일 추가 | `requirements.txt` 또는 `pyproject.toml` (pythonocc-core, numpy, matplotlib, pyvista 등) |
