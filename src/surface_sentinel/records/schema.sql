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
