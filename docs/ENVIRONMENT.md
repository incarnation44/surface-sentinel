# 실행 환경 및 도구 안내 — Surface Sentinel

## 기본 요구 사항

| 항목 | 기준 |
|---|---|
| OS | Windows 10/11 (x64) · Ubuntu 24.04 (Linux CI 검증) |
| Python | 3.11 (`.venv` 가상환경, `uv` 관리) |
| 패키지 관리자 | `uv` 0.12 이상 (`uv.lock` 36 packages 고정) |
| 핵심 라이브러리 | OpenCV, NumPy, FastAPI, Jinja2, SQLite, pytest, Ruff |
| 선택 extra (`--extra yolo`) | `onnxruntime` (격리 비교 및 교차 증거 스모크용) |
| 브라우저 테스트 (`-m browser`) | 시스템 Chrome 우선, 미설치 시 Playwright Chromium 자동 폴백 |

## 초기 설치 및 점검

```bash
uv sync --frozen                 # 기본 의존성 동기화
uv sync --frozen --extra yolo    # ONNX 비교 트랙 포함 시
uv run surface-sentinel --version
uv run surface-sentinel doctor
```

## 새 PC에서 복원

```bash
git clone https://github.com/incarnation44/surface-sentinel.git
cd surface-sentinel
uv sync --frozen --extra yolo
uv run surface-sentinel gate
uv run pytest -m "not data and not browser" -q
```

Git 제외 자산(아래 표)은 필요할 때만 같은 경로에 직접 복사한다.

* 원본 데이터셋: `data/raw/kolektorsdd2/` (`train/`, `test/`)
* 비교용 YOLO 모델: `data/derived/yolo/models/{det,seg}/weights/best.{pt,onnx}`
* 기존 실행 기록: `runs/` (`records.db` 및 `<run_id>/`)

## 주요 실행 명령

```bash
# 1. 정적 검사 · 테스트 · 게이트
uv run ruff check .
uv run pytest -m "not data and not browser"   # CI 기준 989 passed (데이터셋 없이 실행 가능)
uv run pytest                                 # 로컬 전체 1004건 (데이터셋·브라우저 포함)
uv run surface-sentinel gate                  # 저장소 하드 게이트 6/6 PASS
uv run surface-sentinel records verify --db runs/records.db

# 2. 검사 및 배치
uv run surface-sentinel inspect --image tests/fixtures/synthetic/dark_defect.png --mask tests/fixtures/synthetic/dark_defect_mask.png --label defect --split tune --config configs/base.toml --output runs/inspect
uv run surface-sentinel batch --split tune --limit 20
uv run surface-sentinel eval --split tune

# 3. 오프라인 웹 콘솔 & API
uv run surface-sentinel serve --with-api
```

## Git 제외 경로 (로컬 배치 필요)

아래 경로는 용량과 데이터셋 라이선스(CC BY-NC-SA 4.0) 정책상 `.gitignore` 대상이다.

| 경로 | 내용 | 없을 때 동작 |
|---|---|---|
| `data/raw/kolektorsdd2/` | 원본 KolektorSDD2 이미지·마스크 | 비데이터 테스트(989건)·합성 `inspect`·웹 콘솔 정상 동작 |
| `data/derived/` | YOLO 모델(`best.pt`, `best.onnx`)·캐시·오버레이 | OpenCV 메인 파이프라인은 정상 동작 |
| `runs/` | 배치 실행 산출물 및 `records.db` | 새로 `batch`·`inspect`·`serve` 실행 시 생성 |

## 불변 계약

* `configs/base.toml`(r20 `a762ba63`)과 `configs/agent.toml`은 동결 상태다.
* `data/splits/test.lock.json`(`access_count=2`)의 봉인 `test` 분할은 재개봉하지 않는다(`G_SEAL_LOCK`이 `access_count > 2`를 거부).
* 웹 콘솔 실행과 사용법은 [WEB_GUI_GUIDE.md](WEB_GUI_GUIDE.md)를 따른다.
