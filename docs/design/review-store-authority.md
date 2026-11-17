# 검토 저장소 권한 정리 — 파일 `ReviewStore` vs SQLite `review_decisions`

- 범위: **문서화만.** 코드·스키마·설정·알고리즘 변경 0. 감사 5.14(검토 저장소 권한 정리) 대응.
- 목적: 사람 검토 결정이 두 곳(파일 `review_decisions.jsonl`, SQLite `review_decisions`)에
  존재하는 이유와 **각각 무엇의 정본인지**를 실제 코드 기준으로 고정한다.
- 근거 파일: `src/surface_sentinel/review/store.py`, `src/surface_sentinel/records/schema.sql`,
  `src/surface_sentinel/records/store.py`, `src/surface_sentinel/api/app.py`,
  `src/surface_sentinel/ui/app.py`, `src/surface_sentinel/ui/data.py`,
  `src/surface_sentinel/agent/audit.py`, `src/surface_sentinel/agent/calibration.py`.

## 0. 요약

| 저장소 | 물리 위치 | 기록 단위 | 정본인 질문 | 쓰기 주체 |
|---|---|---|---|---|
| `ReviewStore` | `<run_dir>/review_decisions.jsonl` | **run 디렉터리 × 샘플** | "이 run의 이 샘플에 대한 사람 판정은?" | 로컬 UI(`POST …/review`)만 |
| SQLite `review_decisions` | `runs/records.db` (테이블) | **검사 기록(`inspections.id`)** | "검사 1건에 달린 사람 판정 이력은?" | 검사 API(`POST /api/inspections/{id}/review`)만 |

두 저장소는 **키 축이 다르다**(`run_id`+`sample_id` vs `inspection_id`). 어느 한쪽이 다른 쪽의
부분집합이 아니며, 현재 코드에는 **한쪽 쓰기가 다른 쪽에 자동 반영되는 경로가 없다**.
따라서 "하나의 정본"으로 합칠 수 없고, 질문별 정본을 구분해 조회해야 한다.

## 1. 파일 기반 `ReviewStore`

- 정의: `review/store.py:137` — `"""run 디렉터리 하나의 검토 결정 로그. 쓰기는 append뿐이다."""`
- 경로: `ReviewStore(run_dir).path == run_dir / "review_decisions.jsonl"`
  (`review/store.py:144`, 상수 `REVIEW_DECISIONS_JSONL`).
- 레코드 필드(`ReviewDecision`): `decision_id`, `run_id`(= `run_dir.name`), `sample_id`,
  `machine_status`, `final_status`, `reason_code`, `note`, `reviewer`, `decided_at`,
  `config_hash`, `overlay_sha256`. 모든 값은 문자열이며 스키마 검증을 통과해야 한다.
- 의미: 기계 판정(`machine_status`)은 바꾸지 않고, 사람이 내린 **최종 상태와 사유 코드**를
  run 문맥의 증거(설정 해시·오버레이 해시)와 함께 남긴다.
- 쓰기(`append`, `review/store.py:177`): 검증 → 한 줄 JSON append → `flush` + `os.fsync`.
  - 검증 실패는 `ReviewError`, 쓰기 실패는 `ReviewStorageError`로 드러낸다(감사 append 실패 은폐 금지).
  - 파일 락은 없다. 한 프로세스(로컬 UI)가 단독으로 쓰는 전제다.
- 읽기: `read_all()`(기록 순 전체), `latest()`(샘플별 마지막 결정 = 최종 상태, 설계 §5.3).
- 쓰기 조건(UI, `ui/app.py`의 리뷰 저장 핸들러):
  1. run 결과 레코드의 기계 판정이 `REVIEW`일 때만 허용(OK/NG 불변).
  2. `overlay_path`가 있고 실제 파일이 존재해야 한다.
  3. 오버레이 파일 SHA256을 계산해 `overlay_sha256`으로 함께 기록한다.
- 읽는 곳(전부 UI): run 상세, 샘플 상세, `/mask` 뷰어, `/review`, `/metrics`, `/watchdog`.
  `/watchdog`는 `read_all()` 전체를 Phase 4 피드백 분석(`analyze_human_feedback`)에 넘긴다.
- run 디렉터리와 함께 이동·삭제되는 **작업 공간 단위 기록**이다. Git 무시 대상 산출물과 같은
  수명 주기를 가진다.

## 2. SQLite `review_decisions`

- 스키마: `records/schema.sql:43` — `id`, `inspection_id`(FK `inspections.id`), `decision`(OK/NG),
  `reviewer`, `reason`, `created_at`. 파일 store의 `final_status`·`reason_code`·`overlay_sha256`·
  `config_hash`에 해당하는 열이 없다.
- 기록 단위: `inspections` 행 1건. `inspections`는 `UNIQUE (image_sha256, inspection_fingerprint)`로
  중복 제거되며, 같은 이미지+같은 신원이면 새 행을 만들지 않고 **기존 id를 돌려준다**
  (`records/store.py:insert_inspection`).
- 쓰기(`records/store.py:178 append_review`): `inspections` 존재 확인 → `INSERT`만 수행.
  기계 판정(`inspections.status`)은 절대 UPDATE 하지 않는다.
- 강제: 스키마 트리거 2개(`records/schema.sql:134`, `:140`)가 `review_decisions`의 `UPDATE`/`DELETE`를
  `RAISE(ABORT)`로 막는다. 즉 DB 수준에서 append-only가 강제된다(감사 5.13, `append-only-enforcement.md`).
- API 경로(`api/app.py`의 `review` 엔드포인트):
  - 검사 기록 없음 → 404
  - 기계 판정이 `REVIEW`가 아님 → 400 (OK/NG는 사람이 덮어쓰지 않는다)
  - 그 외 → `append_review` 후 해당 검사의 판정 이력을 append 순으로 재조회해 응답에 담는다.
- 읽는 곳:
  - API 응답의 `reviews`(검사별 이력, `ORDER BY id ASC`)
  - UI `/inspections` 목록: 건수 카운트(`ui/data.py:366`)
  - UI `/inspections/{id}` 상세: 판정 이력(`ui/data.py:449`)
  - 감사(`agent/audit.py`): OK/NG/ERROR 검사에 달린 **불법 사람 판정 0건** 검사
- 워커(`agent/worker.py`)는 `inspections`·`engine_observations`·`agent_assessments`만 쓰고
  `review_decisions`는 쓰지 않는다(모듈 내 `review` 심볼 없음).

## 3. 정본 판정 (질문별)

| 질문 | 정본 | 이유 |
|---|---|---|
| 이 run·이 샘플의 사람 최종 판정과 사유 코드 | 파일 `review_decisions.jsonl` | `final_status`·`reason_code`·`overlay_sha256`·`config_hash`를 함께 보존하는 유일한 기록 |
| 검사 기록 1건(`inspection_id`)의 사람 판정 이력 | SQLite `review_decisions` | API가 `inspection_id` 축으로 append하고 DB 트리거가 변조를 차단 |
| 사람 판정의 불법 여부(OK/NG에 판정이 달렸는가) | SQLite | `audit.py`가 DB 기준으로 0건 검사 |
| run 단위 사람-기계 불일치·캘리브레이션 피드백 | 파일 `review_decisions.jsonl` | Phase 4 `analyze_human_feedback` 입력이 파일 `read_all()` 결과 |

즉 **파일 store = run 작업 공간의 검토 기록**, **SQLite = 시스템 검사 이력의 판정 기록**이다.
둘 다 append-only이며, 둘 다 "기계 판정은 불변"이라는 같은 계약을 지킨다.

## 4. 통합 조회 방법 (코드 변경 없이)

두 저장소를 잇는 키는 **`(run_id, sample_id)` 또는 이미지 SHA**다.

- run 쪽 레코드(`result-v1`, `io/artifacts.py:build_result_record`): `run_id`, `sample_id`,
  `input_sha256`(= 이미지 SHA256), `status`, `config_hash`.
- DB `inspections`: `run_id`, `sample_id`, `image_sha256`, `inspection_fingerprint`.
- 따라서 외부 스크립트/노트북에서
  1. `runs/<run_id>/results.jsonl`을 읽고
  2. `runs/<run_id>/review_decisions.jsonl`을 `sample_id`로 붙이고
  3. `records.db`의 `inspections`/`review_decisions`를 `(run_id, sample_id)` 또는
     `image_sha256`으로 붙이면
  한 샘플의 **기계 판정 · run 검토 결정 · 시스템 판정 이력**을 한 표로 볼 수 있다.
- 현재 코드/CLI에는 이 3중 조인을 수행하는 명령이 없다. `records verify`는 무결성 검사일 뿐
  조회 도구가 아니다.

## 5. 한계와 위험 (코드에서 확인되는 사실)

- **원자성 없음**: 파일 append(fsync)와 SQLite 트랜잭션은 서로 묶여 있지 않다. 한쪽만 성공하는
  상태가 가능하다.
- **수명 주기 차이**: 파일 store는 run 디렉터리와 함께 사라질 수 있고, DB 행은 남는다.
  반대로 DB가 없는 환경(파일만 있는 run)에서는 파일 store만 존재한다.
- **필드 비대칭**: DB에는 `final_status`·`reason_code`·`overlay_sha256`·`config_hash`가 없고,
  파일 store에는 `inspection_id`가 없다. 서로를 대체할 수 없다.
- **동시 쓰기**: 파일 store에 락이 없으므로 UI는 단일 프로세스 운용을 전제로 한다.
  DB 쪽은 SQLite 트랜잭션으로 직렬화된다.
- **자동 동기화 없음**: UI에서 남긴 판정은 파일에만, API에서 남긴 판정은 DB에만 쌓인다.
  같은 판정을 양쪽에 자동 복제하는 코드는 없다.

## 6. 후속(이번 사이클 제외)

- 대규모 모듈 분할·스키마 통합 리팩터링은 **출시 차단 요인이 아니므로** 이번 안정화 사이클에서
  제외했다(감사 권고 순서: 정합성 → 통계 → 유지보수).
- 통합이 필요해지면 상위 작업으로 승격해 (1) 두 저장소 중 하나를 정본으로 정하고
  (2) 마이그레이션·이중 기록 금지 규칙을 먼저 문서화한 뒤 코드를 바꾼다.
  **이 문서는 현행 동작을 설명할 뿐, 동작을 바꾸지 않는다.**