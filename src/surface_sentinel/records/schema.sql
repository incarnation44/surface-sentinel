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
--   * image_sha256 UNIQUE 가 중복제거 키다(같은 이미지는 한 행만 갖는다).
--
-- PostgreSQL 이식 시 추가: CREATE EXTENSION IF NOT EXISTS pgcrypto;
--   그리고 REAL → DOUBLE PRECISION, AUTOINCREMENT → GENERATED ALWAYS AS IDENTITY.

CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    image_sha256 TEXT NOT NULL UNIQUE,   -- 중복제거 키
    label TEXT NOT NULL,
    split TEXT NOT NULL,
    status TEXT NOT NULL,                -- OK/NG/REVIEW/ERROR (기계 판정, 불변)
    score REAL NOT NULL,
    reasons TEXT NOT NULL,               -- JSON 배열
    engine TEXT NOT NULL,                -- opencv/yolo-det/yolo-seg/combined
    config_hash TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT '',
    overlay_path TEXT NOT NULL DEFAULT '',
    timing_ms TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL             -- ISO8601 UTC
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

CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(status);
CREATE INDEX IF NOT EXISTS idx_review_inspection ON review_decisions(inspection_id);
