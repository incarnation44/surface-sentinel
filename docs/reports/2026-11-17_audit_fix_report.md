# 감사 피드백 적용 보고서 — 2026-11-17 안정화 사이클

- 날짜: 2026-11-17
- 입력: 2026-11-17 외부 종합 점검 피드백(P0 출시 차단 / P1 정합성 / P2 통계·운영 / P3 유지보수)
- 원칙: **알고리즘·설정 불변.** `configs/base.toml`(r20, `a762ba63`)·`configs/agent.toml`·
  review band 동결 유지. 봉인 split 재접근 없음(원장 불변: test access=2, official_test access=1).
  기존 봉인/채택 보고서·증거 JSON은 소급 수정하지 않음.

## 1. 적용 항목 (감사 번호순)

### 5.4 dirty fingerprint 충돌 + dedup 응답 정합성 — 적용

- `evaluation/recalibration.py` `git_code_identity()`: dirty 트리에서 코드 내용
  digest(status + `git diff HEAD` + `src/`·`configs/` untracked 파일 SHA256)를
  `code_sha`에 합성(`<HEAD>-dirty-<digest16>`). 같은 HEAD라도 미커밋 내용이
  다르면 fingerprint가 달라져 서로 다른 코드 결과가 한 검사로 dedup되지 않는다.
  clean 트리는 기존과 완전 동일(하위 호환, subprocess 2회 유지).
- `api/app.py`: dedup 적중 시 응답의 저장 필드(run_id·sample_id·status·score·
  reasons·overlay_url)를 **DB canonical 행**에서 구성. components는 저장된
  엔진 관찰 우선, 없으면(단일 엔진) 이번 결과(같은 fingerprint = 결정론).
- 수용 증거: `tests/unit/test_code_identity.py` 11건(결정론·내용 민감·clean 호환·
  실패 폴백), `tests/unit/test_api_boundary.py` dedup canonical 테스트.

### 5.5 API label/split 경계 검증 — 적용

- `api/app.py` `_validate_label_split()`: 허용 집합의 단일 정본
  (`pipeline.stages.VALID_LABELS/VALID_SPLITS`)을 import해 **파일 읽기·검사·DB
  기록 전에** 422로 거부. 잘못된 값이 200+ERROR와 검사 행을 만들지 않는다.
- 봉인 split(test/official_test)은 허용 값이므로 기존 파이프라인 거부(ERROR)
  동작 유지 — 우회 없음.
- 수용 증거: `test_api_boundary.py` — 422 + 행 미생성 + 정상 값 회귀 없음.

### 5.7 ONNX 모델 SHA lifecycle 캐시 — 적용

- `yolo/runtime.py`: 경로별 `(stat 시그니처, SHA)` 캐시. 시그니처
  (size·mtime_ns·ctime_ns·ino) 동일 → 파일 읽기 없이 캐시 SHA, 변경 → 전체
  재해시. 세션 신원은 계속 **바이트 SHA**(stat은 재해시 트리거일 뿐).
  `invalidate(model_path)` 명시적 무효화 API 추가. 잔여 위험(stat 보존 악의적
  변경) 독스트링에 명시.
- 수용 증거: `test_yolo_runtime.py` 10건 — N회 추론 해시 1회, 같은 크기 재작성
  무효화, 기존 세션 재사용 계약 전부 유지.

### 5.8 예상 못한 예외 진단 로깅 — 적용

- `pipeline/inspect.py`: `except Exception` → `logger.exception`, 스키마 폴백 →
  `logger.error(exc_info=True)`. stage·예외 타입·run_id·sample_id·image SHA·
  config_hash를 내부 로그에 남긴다. 결과 스키마·판정 불변, 응답에 stack trace·
  내부 경로 미노출. `InspectResponse`에 `error_code`/`error_stage` 진단 필드 추가.
- 수용 증거: `tests/unit/test_pipeline_error_logging.py` 3건(로그 컨텍스트·
  비노출·정상 경로 무로그).

### 5.9 업로드 bounded read — 적용

- `api/app.py` `_read_upload_bounded()`: 64KiB 청크로 `MAX_UPLOAD_BYTES`까지만
  읽고 초과 즉시 413. 읽기 총량 ≤ 상한 + 청크 1개(메모리 상한 증명 테스트).
- 수용 증거: `test_api_boundary.py` — 읽기 총량 계측, 413, 상한 정확값 accepted.

### 5.10 신뢰구간 통계 해석 — 적용 (하위 호환 방식)

- 기존 Wilson 값·정의 문자열·`Ratio` 스키마 **불변**(동결 증거·게이트 호환).
  `metrics.py` 독스트링에 근사 한계 명시(balanced accuracy 평균 Wilson,
  pixel 독립 시행 가정).
- `evaluation/bootstrap.py`에 결정론적 이미지/그룹 단위 percentile CI 추가:
  `balanced_accuracy_bootstrap_ci`, `pixel_ratio_bootstrap_ci`(seed 고정,
  퇴화 규칙 None, ERROR 분모 제외).
- tune 캐시 2,001장 실측 비교(seed=20261117, B=2000): pixel precision Wilson 폭
  0.0028 → bootstrap 0.0638, pixel recall 0.0017 → 0.0959(약 30~50배 과소평가
  확인). balanced accuracy 0.0853 → 0.0658. 점추정 3개 모두 기존 값과 일치.
  r20 채택 게이트는 balanced CI를 쓰지 않으므로 채택 결론 불변.
- 상세: `docs/reports/2026-11-17_ci_bootstrap_note.md`. 봉인 split 미접근.

### 5.11 문서·주석 정정 — 적용

- `batch/runner.py` 독스트링: "봉인 읽기 경로 없음" → 명시적 `allow_sealed` +
  잠금 원장 경로의 유일한 예외로 정정.
- `io/atomic.py` 독스트링: 원자적 게시 보장과 crash durability 미보장(특히
  Windows) 범위 구분(감사 5.12 겸용).
- `ENVIRONMENT.md`·`PROJECT_STATUS.md`·`master-design.md`의 현재 상태 서술
  정정, 파괴적 `git reset --hard` 권고 제거.
- 역사 보고서(T05 채택 당시 "sealed 미실행" 등)는 작성 시점 증거이므로 보존.

### 5.13 DB append-only 스키마 강제 — 적용

- `records/schema.sql`: 4개 테이블 × UPDATE/DELETE 금지 트리거 8개
  (`RAISE(ABORT)`, 한국어 메시지, 멱등). 마커 구간이 DDL 정본.
- `records/db.py`: `init_db`가 마이그레이션 후 트리거 재보장(DROP/RENAME으로
  유실되는 경로 방지). `verify_records_integrity()` 추가.
- `cli.py`: `records verify` 서브커맨드(읽기 전용, PASS=0 / 위반=5).
- 마이그레이션 예외 절차·PG 이식 노트: `docs/design/append-only-enforcement.md`.
- 기존 "변조 시뮬레이션" 테스트 3건은 트리거 일시 해제 후 복원 방식으로 갱신
  (설계 문서 §7 절차). 스키마 열 추가는 하지 않음(error_code join 계약 문서화).

### 5.14 검토 저장소 권한 정리 — 문서화 적용

- 파일 기반 `ReviewStore`(run 디렉터리, UI)와 SQLite `review_decisions`
  (API·워커·UI 조회)의 역할 분담을 문서화. 대규모 모듈 분할 리팩터링은
  출시 차단 요인이 아니므로 이번 사이클에서 제외(감사 권고 순서 준수).

## 2. 검증 증거 (Windows, PowerShell)

| 검사 | 명령 | 결과 |
|---|---|---|
| 전체 스위트 | `uv run --no-sync pytest -o addopts="" -q` | **907 passed** (56.39s), exit 0 |
| CI 선택식 | `pytest -m "not data and not browser" --collect-only` | **892/907** (15 deselected), exit 0 |
| nightly 선택식 | `pytest -m "not data" --collect-only` | **900/907** (7 deselected), exit 0 |
| data / browser | 마커 분류 | data 7(로컬 전용) · browser 8 |
| lint | `uv run --no-sync ruff check .` | All checks passed!, exit 0 |
| lock | `uv lock --check` | Resolved 36 packages, exit 0 |
| 하드 게이트 | `uv run --no-sync surface-sentinel gate` | **PASS 6/6**, exit 0 |
| 재현성 | `uv run --no-sync surface-sentinel repro check --trace runs/agent-comparison/agent/20261113T035419Z-a762ba63-tune/agent_trace.jsonl` | 이벤트 9건, 종료 DONE/OK, exit 0 |
| 기록 DB 무결성 | `uv run --no-sync surface-sentinel records verify --db data/derived/records.db` | integrity ok · 외래키 0 · 트리거 8/8 **PASS**, exit 0 |
| 봉인 원장 | gate G_SEAL_LOCK | test access=2, official_test access=1 (**불변**) |

이전 기준선(7e10ab8): 844 passed · CI 829 · nightly 837. 신규 테스트 +63.
CI/nightly 수치는 원격 GitHub 실행이 아니라 현재 환경에서 워크플로 선택식을
수집한 결과다(이전 점검과 같은 방식).

## 3. 이번 사이클에서 하지 않은 것 (의도적 제외)

| 항목 | 이유 |
|---|---|
| r20 임계값·review band·후보 설정 변경 | 감사 권고: 안정화와 알고리즘 개선 분리. 봉인 test 반복 열람 금지 |
| 봉인 test/official_test 재실행 | 접근 원장 보존. 새 독립 holdout/현장 데이터 확보 후 별도 사이클 |
| escape 성능 게이트 통과 처리 | Wilson 상한 21.55% FAIL은 그대로 유지 — 기준 완화·포장 없음 |
| YOLO/에이전트 승격 | paired 이득 0 확인 완료. 검토 우선순위 보조로 유지 |
| PPTX/PDF/영상 재생성 | 동결 산출물. 에라타 문서로 보충(위) |
| 대형 모듈 분할 리팩터링 | 출시 차단 요인 아님(P3). 정합성 수정 우선 |
| 생산 파일럿 검증 | 실제 라인 데이터·하드웨어 필요 — 별도 단계 |

## 4. 남은 과제 (다음 사이클)

1. **P0 유지**: 새 독립 검증 데이터(현장 또는 신규 holdout)에서 사전 등록
   escape 기준 통과. 봉인 test FAIL·낮은 IoU 0.50 recall·REVIEW 34.71%는
   문서·UI 개선으로 해결 불가.
2. 알고리즘 개선: 저대비·미세 결함 후보 recall 분석 → 별도 config → paired
   bootstrap 게이트 → 새 독립 holdout 최종 평가.
3. 생산 파일럿: 유병률·검토 인력·p95/p99·장시간 안정성·fail-safe·운영 규칙.
4. bootstrap CI를 신규 평가 보고서에 기본 병기(기존 동결 보고서는 불변).
5. `records verify`를 CI/nightly에 편입 검토(현재는 수동 명령).
