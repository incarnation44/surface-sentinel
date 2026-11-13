# 에이전트 확장 + 무결성 기반 — 구체 구현 설계

- **상태:** Phase 0(§1.1~1.4) 확정·구현·검증 완료 (ADR 0008,
  `docs/decisions/0008-r-design-02-phase0-signoff.md`). Phase 1~4(§2 교차 증거·트리아지·Watchdog·
  반사실·캘리브레이션, observe-only) 구현·실측 완료.
  구현 모듈: `agent/evidence.py`(Phase 1), `agent/review.py`·`agent/stability.py`(Phase 2),
  `agent/audit.py`·`agent/drift.py`(Phase 3), `agent/counterfactual.py`·`agent/calibration.py`(Phase 4).
  `engine_observations`·`agent_assessments` 모두 가동 중이며 `OK/NG` 기계 판정은 불변이다.
- **범위:** 판정 로드맵 2단계(P0-2 fingerprint · P0-3 서빙 동일성)와 동일 기반. 6단계(RKNN·Orange Pi)
  하드웨어는 제외한다.
- **상위 설계:** `docs/design/master-design.md` (SS-DESIGN-1.1).

> 핵심 결정: **새 기능을 먼저 붙이지 않는다.** 무결성 기반(Phase 0 = P0-2 + P0-3)을 먼저 닫고,
> 그 위에 **observe-only 에이전트**(판정 불변·별도 증거)를 얹는다. 네트워크·자동 설정
> 변경·자동 재학습·자동 채택은 포함하지 않는다. 6단계 하드웨어는 구현하지 않는다.

---

## 0. 현재 코드 사실 (대조 근거, file:line)

| # | 위치 | 현재 동작 | 문제 |
|---|---|---|---|
| F1 | `records/schema.sql:21` | `image_sha256 TEXT NOT NULL UNIQUE` | 같은 이미지를 다른 엔진·모델·설정으로 검사해도 첫 행을 재사용 → 엔진별 관찰이 덮이거나 잘못 재사용됨 |
| F2 | `records/store.py:100-105,180-190` | dedup·조회가 `image_sha256` 단독 | F1과 동일. fingerprint 없음 |
| F3 | `api/app.py:347` | `find_inspection_id(connection, sha256)` | API dedup도 이미지 단독 |
| F4 | `api/app.py:180-187` | `_model_version`이 YOLO면 엔진 이름 문자열만 | 실제 ONNX 모델 SHA 미기록 → provenance 불완전 |
| F5 | `yolo/inspect.py:49-68` | `engine_task("combined")` → 환경변수 없으면 **det 기본** | 게이트는 seg로 평가했는데 서빙은 숨은 det 기본 → 평가/서빙 모델 불일치 가능 |
| F6 | `yolo/inspect.py:241` | 이미지마다 `YoloOnnxEngine(model_path, task)` 새로 생성 | 세션 재사용 없음 → cold init 비용 매 이미지 |
| F7 | `yolo/inspect.py:164-171` | `_merge_combined`가 `score=max(opencv,yolo)`, 성분·타이밍은 YOLO 것만 | 비교 불가능한 점수 단순 max, OpenCV 성분/근거 소실 |
| F8 | `yolo/evaluate.py:540-555` | 평가 결합: `decide_from_detections` + `inspect_image(split="tune")` + `combine()` | 상태는 서빙과 같은 `combine()`이라 일치. 단 score는 YOLO 점수(서빙 max와 다름), 성분 보존 경로 다름 |
| F9 | `yolo/combine.py:12-21` | `NG > REVIEW > ERROR > OK` 고정 | 결합 규칙 자체는 평가·서빙 공용(유지 대상) |
| F10 | `yolo/adapter.py:27-28` | `ng_conf=0.50, review_lo=0.25` | 정책 해시로 fingerprint에 묶어야 함 |
| F11 | `config.py:126-128` | `AppConfig.config_hash` = canonical TOML SHA256 | 재사용 가능(동결 `a762ba63…`) |
| F12 | `evaluation/recalibration.py:64-76` | `git_head()` = `git rev-parse HEAD` **subprocess** | 워커 핫패스는 subprocess 금지 → code_sha는 진입점에서 1회 계산 후 주입 |
| F13 | `agent/worker.py` | sha dedup, `split="tune"`, subprocess 금지 | API와 같은 fingerprint 규칙 필요 |
| F14 | `agent/engine.py`, `agent/config.py` | T09 `OBSERVE→PLAN→ACT→VERIFY`, 예산·trace·거부 감사 | 에이전트 프레임워크 재사용 기반(판정 소유권은 파이프라인) |

**이미 일치하는 것:** 결합 상태 규칙(F9)은 평가·서빙 공용. `config_hash`(F11) 재사용 가능.
**핵심 격차:** dedup 키(F1~F3), 모델 SHA(F4), 숨은 det 기본(F5), 세션 재사용(F6), 점수·성분 병합(F7/F8).

---

## 1. Phase 0 — 무결성 기반 (= P0-2 + P0-3, 판정 로드맵 수용 범위)

### 1.1 EngineSpec + inspection_fingerprint (P0-2 핵심)

신규 `yolo/spec.py`(또는 `records/identity.py`)에 동결 dataclass `EngineSpec`:

```
engine, task, onnx_path, model_sha256, config_hash,
yolo_policy_hash, combine_policy_hash, preprocessing_version,
code_sha, code_dirty
```

fingerprint = `SHA256(canonical JSON)` (sort_keys, compact, ensure_ascii). **image_sha256은 포함하지 않는다**
(별도 컬럼으로 UNIQUE 쌍을 이루므로 중복).

- opencv: `task=""`, `model_sha256=""`, `yolo_policy_hash=""`, `combine_policy_hash=""`.
- `model_sha256`: ONNX 파일 바이트 SHA256. EngineSpec 생성 시 1회 계산(세션 레지스트리 키).
- `yolo_policy_hash`: `{ng_conf, review_lo}` canonical 해시(F10).
- `combine_policy_hash`: `combine()` 규칙 버전 상수 해시(규칙 고정, F9).
- `preprocessing_version`: letterbox·NMS·입력 변환 버전 상수(`inference.py` 전처리 변경 시 bump).
- `code_sha`/`code_dirty`: **진입점(API lifespan·워커 init·CLI)에서 `git_head()` 1회 계산 후 주입**(F12).
  워커 핫패스에서 subprocess 호출 금지. 릴리스 측정은 `code_dirty=false` 요구.

### 1.2 스키마 + 저장소 dedup 변경

`records/schema.sql`:
- 신규 컬럼: `inspection_fingerprint TEXT NOT NULL DEFAULT ''`, `task TEXT NOT NULL DEFAULT ''`,
  `model_sha256 TEXT NOT NULL DEFAULT ''`, `provenance_complete INTEGER NOT NULL DEFAULT 0`.
- UNIQUE: `image_sha256` 단독 → **`UNIQUE(image_sha256, inspection_fingerprint)`**.
  (SQLite는 UNIQUE 제약 ALTER 불가 → 신규 DB는 schema.sql에 직접, 기존 DB는 §1.3 재빌드.)

`records/store.py`:
- `insert_inspection(..., fingerprint, task, model_sha256, provenance_complete)`: dedup을
  `(image_sha256, inspection_fingerprint)`로(F2). IntegrityError 재조회도 같은 키.
- `find_inspection_id(connection, image_sha256, fingerprint)`: 같은 키(F3).
- API(`app.py`)·워커(`worker.py`)가 공유 EngineSpec으로 fingerprint 계산 후 전달.

### 1.3 기존 DB v2 migration (리뷰 보존)

기존 DB를 삭제·덮어쓰지 않는다.
1. API·워커 중지.
2. 기존 DB를 read-only로 열기.
3. 새 스키마로 **별도 v2 파일** 생성.
4. `inspections`·`review_decisions`를 **id 그대로** 복사(외래키 유지).
5. 기존 행 backfill: 저장된 engine·config_hash·model_version으로 legacy fingerprint 부여,
   모델 SHA 미상이므로 `provenance_complete=0`. **과거 모델 SHA를 추측해 채우지 않는다.**
6. 검증: inspection·review 건수 일치, 외래키 연결, `PRAGMA integrity_check`, 리뷰 매핑.
7. 전부 일치 시 파일 교체. 원본은 롤백용으로 보관(코드 수정 전 `.bak` 금지 규칙과는 별개의
   데이터 마이그레이션 안전 절차).

### 1.4 서빙 동일성 (P0-3)

- **숨은 det 기본 제거**(F5): `combined`는 task를 명시 설정으로 요구. 없으면 설정 오류.
  채택 전까지 기본 엔진은 opencv 유지. combined-seg 채택은 val 게이트 통과 + 사용자 승인 시에만.
- **단일 EngineSpec/병합 경로**(F8): 평가(`evaluate.py`)·API(`app.py`)·워커(`worker.py`)가
  같은 EngineSpec과 같은 어댑터·결합을 사용. 상태는 이미 `combine()` 공용(F9) — 유지.
- **점수·성분 보존**(F7): `max()` 단순 병합 폐기. 엔진별 score·status·reasons·components·timing·overlay를
  `engine_observations`에 보존. combined score가 필요하면 "보정 안 된 순위용"으로 명시하고 임계값 판정에 미사용.
- **ONNX 세션 재사용**(F6): `YoloEngineRegistry` 키 `(model_sha256, task, onnxruntime_version, providers)`.
  전역 싱글턴 대신 API 앱/lifespan·워커 실행 범위 주입. 평가는 `run_yolo_evaluation`의 실행당 단일 엔진 재사용 또는 스크립트별 `YoloEngineRegistry` 주입으로 충족. cold/warm 분리 기록.
  검증 전까지 세션을 여러 스레드에 무제한 공유하지 않음(세션별 lock 또는 풀).
- **provenance 기록**(F4): 응답·DB에 task·model_sha256·policy hash·code_sha 기록.

### 1.5 신규 테이블(에이전트 전용 — §6 승인 대상)

- `engine_observations`: inspection_id, engine, task, status, score, reasons_json, components_json,
  model_sha256, overlay_path, error_code, timing_json.
- `agent_assessments`(append-only): inspection_id, assessment_hash, policy_hash, flags_json,
  review_priority, stability, recommended_action, diagnostics_json, trace_path, created_at.

### 1.6 Phase 0 완료 기준

- 같은 이미지+같은 fingerprint → 1행. 같은 이미지라도 다른 엔진·모델·설정 → 별도 행.
- 기존 리뷰 id·연결 그대로 유지. `integrity_check` ok.
- `combined`에 task 없으면 명확한 설정 오류.
- 한 세션으로 여러 이미지 검사(재사용).
- 평가·API가 같은 모델 SHA·정책 사용(짝 일치 테스트).
- combined-seg가 사전등록 게이트 미통과면 기본 채택 안 함(opencv 유지). **현재 채택 보류 상태 유지.**

---

## 2. Phases 1–4 — observe-only 에이전트 (신규 범위, §6 승인 필요)

상위 통합안은 대화에 제시됨. 여기서는 코드 기반 요약만.

| Phase | 모듈(신규) | 내용 | 판정 영향 |
|---|---|---|---|
| 1 교차 증거 ✅ | `agent/evidence.py` | 엔진 status·localization(IoU/containment/중심거리/면적비/ROI) 비교, 결정론 greedy matching, 전용 플래그, dual overlay — **구현·검증 완료(observe-only)** | 없음(별도 AgentAssessment) |
| 2 리뷰+민감도 ✅ | `agent/review.py`, `agent/stability.py` | P0~P3 등급·종횡비 RMS 보정(391→57), Recall@Top-10% 40.91%(Lift 4.09x), 48×48 패치 중재 56건 — **완료** | 없음. 원본 NG를 OK로 바꾸지 않음 |
| 3 감사+drift ✅ | `agent/audit.py`, `agent/drift.py` | 6종 저장소 하드 게이트(`gate`), `repro check`, 읽기 전용 DB 감사, PSI 드리프트 감시 — **완료** | 없음. 설정·모델 자동 변경 금지 |
| 4 반사실+캘리브레이션 ✅ | `agent/counterfactual.py`, `agent/calibration.py` | 닫힌형 반사실 병목 분해(95.12% 면적 병목), 2D 등위선 위상도, 6×14 핫스팟, 피드백 캘리브레이션 — **완료** | 없음. sealed 미접근·자동 채택 금지 |

공통: 기존 `agent/engine.py` 상태 머신·`trace.py` replay 재사용(F14). 계산은 각 모듈 순수 함수,
상태 전이만 엔진. 에이전트 출력은 `SampleResult.status`와 분리. 모든 도구는 예산·결정론·중복 거부·결과 해시·replay 준수.

---

## 3. 테스트 계획

- `test_records_fingerprint.py`: 같은 이미지+같은 fp 1행 / 다른 엔진·모델·설정 별도 행 / 리뷰 보존.
- v1→v2 migration: 건수·외래키·integrity·리뷰 매핑·`provenance_complete=0`.
- `test_yolo_runtime.py`: EngineSpec, combined task 누락 오류, 세션 재사용.
- parity: 같은 이미지에서 평가 vs API 모델 SHA·정책·status 일치.
- API combined/provenance 통합, 실제 ONNX combined 검사, trace v1/v2 replay 호환.
- 각 단계 완료 시 실제 실행: 전체 pytest / 비데이터 pytest / 실제 ONNX data / Ruff / `uv lock --check` /
  API·워커 cold·warm p50·p95·p99 / git clean. **테스트 개수는 예단하지 않고 실행 결과 그대로 보고.**

---

## 4. 운영 적용 초기값

| 기능 | 초기 상태 |
|---|---|
| provenance 감사 | 항상 활성 |
| 교차 증거 기록 | 활성 |
| 리뷰 우선순위 | 활성 |
| 민감도 검사 | 비활성(검증 후 제한 활성) |
| drift | 경고 전용 |
| 능동 학습·실험 | 오프라인만 |
| 에이전트의 기계 판정 변경 | **영구 금지** |
| 자동 모델·설정 채택 | **영구 금지** |

---

## 5. 동결·봉인 제약 (침범 금지)

- 동결 config `a762ba63…`(r20)·`combine()`·`YoloDecisionPolicy` 기본값·`AppConfig` 해시 변경 없음.
- sealed `test`/`official_test` 미접근(`test.lock.json` access_count=2 유지). 재측정은 사전등록 + 동결
  clean 코드/모델 SHA/config/정책으로만, "재사용 확인"으로 표기(최초 독립 평가 아님).
- 기본 엔진 opencv 유지. combined 채택은 게이트 통과 + 사용자 승인 시에만.
- weights/ONNX/data 커밋 금지. 임의 삭제 금지. T10 착수는 별도 사용자 결정.

---

## 6. 승인 경계

설계 계약상 범위·알고리즘·데이터 분할·평가 기준·UI 구조는 설계 개정(ADR) 없이 바꾸지 않는다.

- **판정 로드맵 범위:** P0-2 fingerprint dedup, P0-3 서빙 동일성(§1.1~1.4). 스키마·점수 병합 변경을
  수반하므로 R-DESIGN-02 개정안으로 확정했다(ADR 0008).
- **확장 범위:** §1.5 에이전트 전용 테이블, §2 Phases 1–4, review-queue API/UI, dual overlay는 단계별
  완료 기준을 통과한 뒤에만 진행했다.
- **순서:** §1(Phase 0) 확정 → 구현·검증 → 완료 기준 통과 시에만 §2 단계별 진행.

---

## 7. 확정 사항

- R-DESIGN-02(fingerprint 입력 집합·code_sha 주입 위치·migration 롤백 정책) 확정 — ADR 0008.
  진입점 1회 주입·단일 트랜잭션 롤백·preprocess-v1/combine-v1 상수 확정.
- ONNX 세션 재사용 동시성 모델: **lock 채택**(ADR 0008 D4) — 세션별 lock + 생성 전역 lock,
  주입식 레지스트리. 풀 방식은 동시성 부하 실측 후 별도 개정 대상.
- `preprocessing_version`·`combine_policy_hash` 상수(ADR 0008 D5): `preprocess-v1` · `combine-v1`.
  규칙 변경 시 수동으로 올린다.
- Phase 1~4(§2) 구현·실측 완료: observe-only, `gate` 6/6 PASS, `repro check` exit 0, 판정 불변.
  자동 채택·네트워크 없음.
