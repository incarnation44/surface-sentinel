"""SQLite append-only 트리거 강제 단위 테스트 — 8개 트리거의 UPDATE/DELETE 차단과 무결성 검사 검증.

스키마(schema.sql)의 트리거 8개가 UPDATE/DELETE 를 SQLite 오류로 거부하는지,
v1→v2 마이그레이션 뒤에도 트리거가 남는지, ``records verify`` 가 정상/위반을
구분하는지 검증한다. 모든 DB 는 ``tmp_path`` 아래에만 만든다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from surface_sentinel.cli import main
from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.records.db import (
    APPEND_ONLY_TRIGGER_NAMES,
    connect,
    ensure_append_only_triggers,
    init_db,
    migrate_to_fingerprint,
    verify_records_integrity,
)
from surface_sentinel.records.store import (
    append_agent_assessment,
    append_review,
    insert_engine_observation,
    insert_inspection,
)

SHA_A = "a" * 64
CFG = "d" * 64

# v1 스키마(v2 이전) 재현 — 마이그레이션 공존 검증용. 정본 DDL 은 schema.sql 이다.
V1_SCHEMA = """
CREATE TABLE inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    image_sha256 TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    split TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    reasons TEXT NOT NULL,
    engine TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT '',
    overlay_path TEXT NOT NULL DEFAULT '',
    timing_ms TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id)
);
CREATE INDEX idx_inspections_status ON inspections(status);
CREATE INDEX idx_review_inspection ON review_decisions(inspection_id);
"""


def _result(**overrides: object) -> SampleResult:
    payload: dict[str, object] = {
        "run_id": "run-append-only",
        "sample_id": "70001",
        "original_id": "70001",
        "split": "tune",
        "label": "defect",
        "status": Status.NG,
        "score": 0.75,
        "reasons": (ReasonCode.DEFECT_EVIDENCE_STRONG,),
        "timing_ms": {"total": 12.5},
        "input_sha256": SHA_A,
        "overlay_path": "",
        "config_hash": CFG,
    }
    payload.update(overrides)
    return SampleResult(**payload)  # type: ignore[arg-type]


@pytest.fixture()
def conn(tmp_path: Path):
    connection = connect(tmp_path / "records.db")
    init_db(connection)
    try:
        yield connection
    finally:
        connection.close()


def _seed_all_tables(connection: sqlite3.Connection) -> None:
    """트리거 대상 4개 테이블에 각각 1행씩 넣는다(store 함수만 사용)."""

    inspection_id = insert_inspection(
        connection, _result(), engine="combined", image_sha256=SHA_A
    )
    append_review(connection, inspection_id, "NG", reviewer="qa", reason="실결함")
    insert_engine_observation(
        connection,
        inspection_id,
        _result(status=Status.NG, score=0.8),
        engine="yolo-seg",
        task="seg",
        model_sha256="e" * 64,
    )
    append_agent_assessment(
        connection,
        inspection_id,
        assessment_hash="f" * 64,
        policy_hash="1" * 64,
        flags=["STATUS_MISMATCH"],
        recommended_action="HUMAN_REVIEW",
        diagnostics={"engine_a": "opencv", "engine_b": "yolo-seg"},
    )


def _trigger_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
    }


# --- 1. 트리거가 UPDATE/DELETE 를 거부한다 -----------------------------------


def test_init_db_creates_eight_append_only_triggers(conn: sqlite3.Connection) -> None:
    names = _trigger_names(conn)
    assert len(APPEND_ONLY_TRIGGER_NAMES) == 8
    assert set(APPEND_ONLY_TRIGGER_NAMES) <= names


@pytest.mark.parametrize(
    ("table", "update_sql"),
    [
        ("inspections", "UPDATE inspections SET status = 'OK' WHERE id = ?"),
        ("review_decisions", "UPDATE review_decisions SET decision = 'OK' WHERE id = ?"),
        ("engine_observations", "UPDATE engine_observations SET score = 0.0 WHERE id = ?"),
        ("agent_assessments", "UPDATE agent_assessments SET policy_hash = 'x' WHERE id = ?"),
    ],
)
def test_update_is_rejected(
    conn: sqlite3.Connection, table: str, update_sql: str
) -> None:
    _seed_all_tables(conn)
    row_id = int(conn.execute(f"SELECT MIN(id) FROM {table}").fetchone()[0])
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(update_sql, (row_id,))
    conn.rollback()
    # 값이 실제로 바뀌지 않았다.
    assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1


@pytest.mark.parametrize(
    "table",
    ["inspections", "review_decisions", "engine_observations", "agent_assessments"],
)
def test_delete_is_rejected(conn: sqlite3.Connection, table: str) -> None:
    _seed_all_tables(conn)
    row_id = int(conn.execute(f"SELECT MIN(id) FROM {table}").fetchone()[0])
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
    conn.rollback()
    assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1


def test_insert_still_allowed(conn: sqlite3.Connection) -> None:
    # 트리거는 UPDATE/DELETE 만 막는다 — append 는 그대로 동작해야 한다.
    _seed_all_tables(conn)
    assert conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0] == 1


# --- 2. init_db 멱등 + 마이그레이션 공존 -------------------------------------


def test_init_db_is_idempotent_with_triggers(tmp_path: Path) -> None:
    connection = connect(tmp_path / "records.db")
    try:
        init_db(connection)
        init_db(connection)
        init_db(connection)
        assert set(APPEND_ONLY_TRIGGER_NAMES) <= _trigger_names(connection)
    finally:
        connection.close()


def test_triggers_restored_after_dropped_table(tmp_path: Path) -> None:
    connection = connect(tmp_path / "records.db")
    try:
        init_db(connection)
        connection.execute("DROP TABLE agent_assessments")  # 이 테이블 트리거도 함께 소멸
        connection.commit()
        init_db(connection)
        assert set(APPEND_ONLY_TRIGGER_NAMES) <= _trigger_names(connection)
    finally:
        connection.close()


def test_v1_migration_keeps_triggers(tmp_path: Path) -> None:
    connection = connect(tmp_path / "v1.db")
    try:
        connection.executescript(V1_SCHEMA)
        connection.execute(
            "INSERT INTO inspections (run_id, sample_id, image_sha256, label, split, status,"
            " score, reasons, engine, config_hash, model_version, created_at)"
            " VALUES ('r1','s1',?,'defect','tune','NG',0.7,'[]','opencv',?,'opencv-4.10',"
            "'2026-11-13T00:00:00Z')",
            (SHA_A, CFG),
        )
        connection.commit()

        init_db(connection)

        columns = {row[1] for row in connection.execute("PRAGMA table_info(inspections)")}
        assert "inspection_fingerprint" in columns
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1
        # DROP TABLE/RENAME 이 트리거를 지웠으므로 init_db 가 다시 건다.
        assert set(APPEND_ONLY_TRIGGER_NAMES) <= _trigger_names(connection)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE inspections SET status = 'OK'")
        connection.rollback()
    finally:
        connection.close()


def test_migration_with_existing_triggers_is_noop(tmp_path: Path) -> None:
    # 이미 v2(트리거 포함)인 DB 에서 migrate 는 no-op 이고 트리거도 그대로다.
    connection = connect(tmp_path / "v2.db")
    try:
        init_db(connection)
        assert migrate_to_fingerprint(connection) is False
        assert set(APPEND_ONLY_TRIGGER_NAMES) <= _trigger_names(connection)
    finally:
        connection.close()


def test_ensure_triggers_is_idempotent(conn: sqlite3.Connection) -> None:
    ensure_append_only_triggers(conn)
    ensure_append_only_triggers(conn)
    assert set(APPEND_ONLY_TRIGGER_NAMES) <= _trigger_names(conn)


# --- 3. 무결성 검사 함수 ------------------------------------------------------


def test_verify_integrity_pass(conn: sqlite3.Connection) -> None:
    report = verify_records_integrity(conn)
    assert report.ok is True
    assert report.integrity_check == "ok"
    assert report.foreign_key_violations == 0
    assert report.missing_triggers == ()


def test_verify_integrity_detects_missing_trigger(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER trg_review_decisions_no_update")
    report = verify_records_integrity(conn)
    assert report.ok is False
    assert report.missing_triggers == ("trg_review_decisions_no_update",)


def test_verify_integrity_detects_foreign_key_violation(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "INSERT INTO review_decisions (inspection_id, decision, reviewer, reason, created_at)"
        " VALUES (9999,'NG','qa','고아 행','2026-11-13T00:00:00Z')"
    )
    report = verify_records_integrity(conn)
    assert report.ok is False
    assert report.foreign_key_violations == 1


# --- 4. records verify CLI ---------------------------------------------------


def test_cli_records_verify_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "derived" / "records.db"
    assert main(["records", "init", "--db", str(db_path)]) == 0
    capsys.readouterr()

    assert main(["records", "verify", "--db", str(db_path)]) == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "8/8" in captured.out
    assert "FAIL" not in captured.err


def test_cli_records_verify_fails_on_missing_trigger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "derived" / "records.db"
    assert main(["records", "init", "--db", str(db_path)]) == 0
    connection = connect(db_path)
    try:
        connection.execute("DROP TRIGGER trg_inspections_no_delete")
    finally:
        connection.close()
    capsys.readouterr()

    assert main(["records", "verify", "--db", str(db_path)]) == 5
    captured = capsys.readouterr()
    assert "FAIL" in captured.err
    assert "trg_inspections_no_delete" in captured.out


def test_cli_records_verify_missing_db(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["records", "verify", "--db", str(tmp_path / "missing.db")]) == 3
    assert "기록 DB가 없습니다" in capsys.readouterr().err


def test_cli_records_verify_is_readonly_and_leaves_db_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """records verify 는 읽기 전용 연결(?mode=ro)만 사용해 DB·WAL 을 수정/생성하지 않는다."""

    db_path = tmp_path / "readonly_check.db"
    connection = connect(db_path)
    try:
        init_db(connection)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.commit()
    finally:
        connection.close()

    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    assert not wal_path.exists()
    before_stat = db_path.stat()
    capsys.readouterr()

    assert main(["records", "verify", "--db", str(db_path)]) == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert not wal_path.exists()
    assert not shm_path.exists()
    after_stat = db_path.stat()
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_size == before_stat.st_size