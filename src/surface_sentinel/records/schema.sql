-- 검사 기록 스키마 (E5, ADR 0007)
--
-- SQLite를 정본으로 쓰고 PostgreSQL로 이식 가능한 형태만 쓴다.
-- 타입은 INTEGER / TEXT / REAL 셋만 쓴다.
--   PostgreSQL 매핑: INTEGER → INTEGER, TEXT → TEXT, REAL → DOUBLE PRECISION
--   AUTOINCREMENT(SQLite 전용) → GENERATED ALWAYS AS IDENTITY
--   TEXT 안의 시각·JSON은 애플리케이션이 ISO8601(UTC) / JSON 문자열로 직렬화한다.
--
-- 불변 규칙:
--   * inspections.status 는 기계 판정이며 UPDATE 하지 않는다.
--   * 사람 판정은 review_decisions 에 append-only 로만 쌓는다.
--   * 중복제거 키는 (image_sha256, inspection_fingerprint) 쌍이다(P0-2).
--     같은 이미지라도 엔진·task·모델·설정·정책·코드 버전이 다르면 다른 검사로 기록한다.
--   * inspection_fingerprint 는 이미지 SHA 를 제외한 검사 신원의 canonical SHA256
--     (records/fingerprint.py). provenance_complete=0 은 신원이 불완전한 과거 행.
--
-- PostgreSQL 이식 시 추가: CREATE EXTENSION IF NOT EXISTS pgcrypto;
--   그리고 REAL → DOUBLE PRECISION, AUTOINCREMENT → GENERATED ALWAYS AS IDENTITY.

CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    inspection_fingerprint TEXT NOT NULL DEFAULT '',  -- 중복제거 키(이미지와 쌍)
    label TEXT NOT NULL,
    split TEXT NOT NULL,
    status TEXT NOT NULL,                -- OK/NG/REVIEW/ERROR (기계 판정, 불변)
    score REAL NOT NULL,
    reasons TEXT NOT NULL,               -- JSON 배열
    engine TEXT NOT NULL,                -- opencv/yolo-det/yolo-seg/combined
    task TEXT NOT NULL DEFAULT '',       -- det/seg (opencv 는 '')
    config_hash TEXT NOT NULL,
    model_sha256 TEXT NOT NULL DEFAULT '',  -- 실제 ONNX 파일 SHA(opencv 는 '')
    model_version TEXT NOT NULL DEFAULT '',
    provenance_complete INTEGER NOT NULL DEFAULT 0,  -- 1=신원 완전, 0=과거/불완전
    overlay_path TEXT NOT NULL DEFAULT '',
    timing_ms TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,            -- ISO8601 UTC
    UNIQUE (image_sha256, inspection_fingerprint)
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    decision TEXT NOT NULL,              -- 사람 판정 OK/NG
    reviewer TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id)
);

-- 결합(combined) 검사의 엔진별 원본 관찰 (P0-3). 기계 판정은 불변이며,
-- 결합 결과가 점수·성분을 덮어써도 엔진별 증거가 소실되지 않도록 보존한다.
-- 한 검사당 엔진 하나씩 1행(UNIQUE). 단일 엔진 검사는 기록하지 않는다.
CREATE TABLE IF NOT EXISTS engine_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    engine TEXT NOT NULL,                -- opencv/yolo-det/yolo-seg (실제 엔진)
    task TEXT NOT NULL DEFAULT '',       -- det/seg (opencv 는 '')
    status TEXT NOT NULL,                -- 엔진별 기계 판정(불변)
    score REAL NOT NULL,
    reasons TEXT NOT NULL,               -- JSON 배열
    components TEXT NOT NULL DEFAULT '[]',  -- JSON 배열(ComponentFeature)
    model_sha256 TEXT NOT NULL DEFAULT '',
    overlay_path TEXT NOT NULL DEFAULT '',
    error_code TEXT NOT NULL DEFAULT '',
    timing_ms TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id),
    UNIQUE (inspection_id, engine)
);

-- 에이전트 교차 증거 평가 (Phase 1, 에이전트 확장 제안 §1.5). append-only.
-- 기계 판정(inspections.status)은 건드리지 않는 별도 증거 기록이다.
-- 같은 (inspection_id, assessment_hash) 는 한 번만 기록한다(중복제거).
-- review_priority·stability 는 Phase 2 예약 컬럼 — Phase 1 에서는 NULL.
CREATE TABLE IF NOT EXISTS agent_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    assessment_hash TEXT NOT NULL,       -- 입력+정책+플래그+지표 canonical SHA256
    policy_hash TEXT NOT NULL,           -- 교차 증거 임계값 정책 해시(evidence-v1)
    flags_json TEXT NOT NULL DEFAULT '[]',  -- JSON 배열(EvidenceFlag)
    review_priority TEXT,                -- Phase 2 예약(P0~P3), 지금은 NULL
    stability TEXT,                      -- Phase 2 예약(민감도), 지금은 NULL
    recommended_action TEXT NOT NULL,    -- NONE/INSPECT_EVIDENCE/HUMAN_REVIEW
    diagnostics_json TEXT NOT NULL DEFAULT '{}',  -- JSON(매칭·지표·ROI 전체)
    trace_path TEXT NOT NULL DEFAULT '', -- 에이전트 에피소드 트레이스(직접 기록은 '')
    created_at TEXT NOT NULL,            -- ISO8601 UTC
    FOREIGN KEY (inspection_id) REFERENCES inspections(id),
    UNIQUE (inspection_id, assessment_hash)
);

CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(status);
CREATE INDEX IF NOT EXISTS idx_review_inspection ON review_decisions(inspection_id);
CREATE INDEX IF NOT EXISTS idx_observations_inspection ON engine_observations(inspection_id);
CREATE INDEX IF NOT EXISTS idx_assessments_inspection ON agent_assessments(inspection_id);

-- APPEND-ONLY TRIGGERS (SQLite 전용) BEGIN
--
-- append-only 는 애플리케이션 규칙만이 아니라 스키마 수준에서도 강제한다(감사 5.13).
--   * inspections            : 기계 판정 행 — UPDATE·DELETE 금지(완전 불변)
--   * review_decisions       : 사람 판정 — append-only
--   * engine_observations    : 엔진별 원본 관찰 — append-only
--   * agent_assessments      : 에이전트 교차 증거 — append-only
-- 위반은 sqlite3.IntegrityError(ABORT)로 거부된다. 트리거 이름은
-- records/db.py 의 APPEND_ONLY_TRIGGER_NAMES 와 1:1로 맞춰야 한다(records verify 가 검사).
--
-- 마이그레이션 예외: 스키마를 바꾸는 마이그레이션은 이 트리거를 잠시 내려야 한다.
--   같은 트랜잭션 안에서 DROP TRIGGER → 스키마 변경 → db.init_db() 로 재보장.
--   자세한 절차는 docs/design/append-only-enforcement.md 참고.
--
-- PostgreSQL 이식 시(SQLite 트리거는 이식되지 않는다) 동등 장치가 필요하다:
--   CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS trigger AS $$
--   BEGIN RAISE EXCEPTION '% 는 append-only 입니다', TG_TABLE_NAME; END;
--   $$ LANGUAGE plpgsql;
--   CREATE TRIGGER trg_inspections_no_update BEFORE UPDATE OR DELETE ON inspections
--     FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
--   (테이블 4개 × UPDATE/DELETE 조합 모두 동일하게 건다.)
--   권한으로 막는 방법도 병행 가능: REVOKE UPDATE, DELETE ON inspections FROM app_role;

CREATE TRIGGER IF NOT EXISTS trg_inspections_no_update
BEFORE UPDATE ON inspections
BEGIN
    SELECT RAISE(ABORT, 'inspections 는 append-only 입니다: 기계 판정 행은 수정할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_inspections_no_delete
BEFORE DELETE ON inspections
BEGIN
    SELECT RAISE(ABORT, 'inspections 는 append-only 입니다: 기계 판정 행은 삭제할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_review_decisions_no_update
BEFORE UPDATE ON review_decisions
BEGIN
    SELECT RAISE(ABORT, 'review_decisions 는 append-only 입니다: 사람 판정 행은 수정할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_review_decisions_no_delete
BEFORE DELETE ON review_decisions
BEGIN
    SELECT RAISE(ABORT, 'review_decisions 는 append-only 입니다: 사람 판정 행은 삭제할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_engine_observations_no_update
BEFORE UPDATE ON engine_observations
BEGIN
    SELECT RAISE(ABORT, 'engine_observations 는 append-only 입니다: 관찰 행은 수정할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_engine_observations_no_delete
BEFORE DELETE ON engine_observations
BEGIN
    SELECT RAISE(ABORT, 'engine_observations 는 append-only 입니다: 관찰 행은 삭제할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_agent_assessments_no_update
BEFORE UPDATE ON agent_assessments
BEGIN
    SELECT RAISE(ABORT, 'agent_assessments 는 append-only 입니다: 평가 행은 수정할 수 없습니다.');
END;

CREATE TRIGGER IF NOT EXISTS trg_agent_assessments_no_delete
BEFORE DELETE ON agent_assessments
BEGIN
    SELECT RAISE(ABORT, 'agent_assessments 는 append-only 입니다: 평가 행은 삭제할 수 없습니다.');
END;
-- END APPEND-ONLY TRIGGERS
