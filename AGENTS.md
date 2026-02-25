# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

CNC 터닝-밀링 파트 합성 데이터 생성기. 순수 Python 프로젝트이며 외부 서비스(DB, 웹서버 등) 불필요. 자세한 내용은 README.md 참조.

### Environment

- **Python 3.11** + **conda** (Miniforge) 사용. conda 환경 이름: `occ`
- `pythonocc-core`는 C++ 바인딩(OpenCASCADE)이므로 반드시 conda-forge 채널을 통해 설치해야 함. pip 단독으로는 어려울 수 있음.
- conda 활성화: `export PATH="$HOME/miniforge3/bin:$PATH"` 후 `conda run -n occ <command>` 또는 `conda activate occ`
- matplotlib 시각화 시 headless 환경에서는 `MPLBACKEND=Agg` 환경변수 설정 필요

### Running commands

- **테스트**: `conda run -n occ python -m pytest tests/ -v`
- **파이프라인 실행**: `MPLBACKEND=Agg conda run -n occ python run_pipeline.py`
- **설치 확인**: `conda run -n occ python -c "from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder; print('OK')"`

### Gotchas

- SWIG `DeprecationWarning` (SwigPyPacked/SwigPyObject)은 pythonocc-core의 알려진 경고이며 무시해도 됨
- `results/` 디렉토리는 파이프라인 실행 시 자동 생성됨
- `pyvista`는 3D 인터랙티브 뷰어용이며 핵심 파이프라인에는 불필요 (optional)
