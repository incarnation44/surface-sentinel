"""agent_assessments 저장소 단위 테스트 (Phase 1, 제안 §1.5).

append-only 계약: 같은 (inspection_id, assessment_hash) 는 기존 행 id 를
돌려주고(중복 기록 금지), 다른 해시는 새 행으로 쌓는다. UPDATE/DELETE 는
없으며 기계 판정(``inspections.status``)은 건드리지 않는다.
``review_priority``·``stability`` 는 Phase 2 예약 컬럼(NULL).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.records.db import connect, init_db, load_schema_sql
from surface_sentinel.records.store import (
    RECOMMENDED_ACTIONS,
    RecordsError,
    append_agent_assessment,
    get_inspection,
    insert_inspection,
    list_agent_assessments,
)

SHA_A = "a" * 64
HASH_1 = "1" * 64
HASH_2 = "2" * 64
POLICY_HASH = "3" * 64


def _result() -> SampleResult:
    return SampleResult(
        run_id="run-test",
        sample_id="70001",
        original_id="70001",
        split="tune",
        label="defect",
        status=Status.NG,
        score=0.75,
        reasons=(ReasonCode.DEFECT_EVIDENCE_STRONG,),
        input_sha256=SHA_A,
        config_hash="d" * 64,
    )


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = connect(tmp_path / "records.db")
    init_db(connection)
    yield connection
    connection.close()


@pytest.fixture()
def inspection_id(conn: sqlite3.Connection) -> int:
    return insert_inspection(conn, _result(), engine="combined", image_sha256=SHA_A)


def _append(conn: sqlite3.Connection, inspection_id: int, **overrides: object) -> int:
    payload: dict[str, object] = {
        "assessment_hash": HASH_1,
        "policy_hash": POLICY_HASH,
        "flags": ["STATUS_MISMATCH"],
        "recommended_action": "HUMAN_REVIEW",
        "diagnostics": {"engine_a": "opencv", "engine_b": "yolo-det"},
    }
    payload.update(overrides)
    return append_agent_assessment(conn, inspection_id, **payload)  # type: ignore[arg-type]


def test_schema_defines_agent_assessments() -> None:
    text = load_schema_sql()
    assert "CREATE TABLE IF NOT EXISTS agent_assessments" in text
    assert "UNIQUE (inspection_id, assessment_hash)" in text


def test_init_db_creates_table_and_index(conn: sqlite3.Connection) -> None:
    names = {
        row["name"]
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }
    assert "agent_assessments" in names
    assert "idx_assessments_inspection" in names


def test_init_db_adds_table_to_existing_v2_db(tmp_path: Path) -> None:
    # 이미 v2(fingerprint)인 기존 DB 에도 다음 init_db 에서 테이블이 생긴다.
    connection = connect(tmp_path / "legacy.db")
    try:
        init_db(connection)
        connection.execute("DROP TABLE agent_assessments")  # 인덱스도 함께 제거됨
        connection.commit()
        init_db(connection)
        names = {
            row["name"] for row in connection.execute("SELECT name FROM sqlite_master")
        }
        assert "agent_assessments" in names
    finally:
        connection.close()


def test_append_returns_id_and_stores_fields(
    conn: sqlite3.Connection, inspection_id: int
) -> None:
    row_id = _append(conn, inspection_id, trace_path="runs/x/agent_trace.jsonl")
    assert row_id > 0
    rows = list_agent_assessments(conn, inspection_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == row_id
    assert row["inspection_id"] == inspection_id
    assert row["assessment_hash"] == HASH_1
    assert row["policy_hash"] == POLICY_HASH
    assert json.loads(row["flags_json"]) == ["STATUS_MISMATCH"]
    assert row["recommended_action"] == "HUMAN_REVIEW"
    assert json.loads(row["diagnostics_json"])["engine_a"] == "opencv"
    assert row["trace_path"] == "runs/x/agent_trace.jsonl"
    assert row["created_at"]
    # Phase 2 예약 컬럼은 비어 있다.
    assert row["review_priority"] is None
    assert row["stability"] is None


def test_same_hash_is_idempotent(conn: sqlite3.Connection, inspection_id: int) -> None:
    first = _append(conn, inspection_id)
    second = _append(conn, inspection_id)
    assert first == second
    assert len(list_agent_assessments(conn, inspection_id)) == 1


def test_different_hash_appends_new_row(
    conn: sqlite3.Connection, inspection_id: int
) -> None:
    first = _append(conn, inspection_id)
    second = _append(conn, inspection_id, assessment_hash=HASH_2)
    assert second != first
    rows = list_agent_assessments(conn, inspection_id)
    assert [row["id"] for row in rows] == [first, second]  # id 오름차순(결정론)


def test_append_never_updates_machine_verdict(
    conn: sqlite3.Connection, inspection_id: int
) -> None:
    _append(conn, inspection_id)
    _append(conn, inspection_id, assessment_hash=HASH_2, recommended_action="NONE")
    row = get_inspection(conn, inspection_id)
    assert row is not None
    assert row["status"] == "NG"  # 기계 판정 불변


def test_missing_inspection_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(RecordsError):
        _append(conn, 999_999)


def test_invalid_hash_rejected(conn: sqlite3.Connection, inspection_id: int) -> None:
    with pytest.raises(RecordsError):
        _append(conn, inspection_id, assessment_hash="not-a-sha")
    with pytest.raises(RecordsError):
        _append(conn, inspection_id, policy_hash="xyz")


def test_invalid_recommended_action_rejected(
    conn: sqlite3.Connection, inspection_id: int
) -> None:
    with pytest.raises(RecordsError):
        _append(conn, inspection_id, recommended_action="AUTO_ADOPT")
    assert set(RECOMMENDED_ACTIONS) == {"NONE", "INSPECT_EVIDENCE", "HUMAN_REVIEW"}


def test_list_empty_for_unknown_inspection(conn: sqlite3.Connection) -> None:
    assert list_agent_assessments(conn, 42) == []


def test_assessments_isolated_per_inspection(conn: sqlite3.Connection) -> None:
    other = insert_inspection(
        conn, _result(), engine="combined", image_sha256="b" * 64
    )
    first = insert_inspection(conn, _result(), engine="combined", image_sha256=SHA_A)
    _append(conn, first)
    assert list_agent_assessments(conn, other) == []
    assert len(list_agent_assessments(conn, first)) == 1


def test_phase2_review_priority_and_stability_stored_and_validated(
    conn: sqlite3.Connection, inspection_id: int
) -> None:
    from surface_sentinel.records.store import REVIEW_PRIORITIES, STABILITY_LEVELS

    assert REVIEW_PRIORITIES == ("P0", "P1", "P2", "P3")
    assert STABILITY_LEVELS == ("STABLE", "SENSITIVE", "UNSTABLE")

    row_id = _append(
        conn,
        inspection_id,
        review_priority="P0",
        stability="STABLE",
    )
    rows = list_agent_assessments(conn, inspection_id)
    assert rows[0]["id"] == row_id
    assert rows[0]["review_priority"] == "P0"
    assert rows[0]["stability"] == "STABLE"

    with pytest.raises(RecordsError):
        _append(conn, inspection_id, assessment_hash=HASH_2, review_priority="P9")
    with pytest.raises(RecordsError):
        _append(conn, inspection_id, assessment_hash=HASH_2, stability="BROKEN")
