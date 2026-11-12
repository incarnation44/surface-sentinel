"""기록 DB(SQLite) 단위 테스트 (E5, ADR 0007).

합성 :class:`SampleResult`만 사용한다. DB 파일과 스키마는 모두 ``tmp_path``
아래에 만들고 저장소의 ``data/derived``는 건드리지 않는다.
기계 판정은 불변이고 사람 판정은 ``review_decisions``에 append-only로만 쌓인다.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from surface_sentinel.cli import main
from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.records.db import connect, init_db, load_schema_sql
from surface_sentinel.records.store import (
    RecordsError,
    append_review,
    get_inspection,
    insert_inspection,
    list_inspections,
    stats,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _result(**overrides: object) -> SampleResult:
    """테스트용 ``SampleResult``. 기본값은 NG 한 건이다."""

    payload: dict[str, object] = {
        "run_id": "run-20261112T000000Z",
        "sample_id": "70001",
        "original_id": "70001",
        "split": "tune",
        "label": "defect",
        "status": Status.NG,
        "score": 0.75,
        "reasons": (ReasonCode.DEFECT_EVIDENCE_STRONG,),
        "timing_ms": {"total": 12.5},
        "input_sha256": SHA_A,
        "overlay_path": "runs/worker/70001_overlay.png",
        "config_hash": "d" * 64,
    }
    payload.update(overrides)
    return SampleResult(**payload)  # type: ignore[arg-type]


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = connect(tmp_path / "records.db")
    init_db(connection)
    yield connection
    connection.close()


def test_connect_uses_wal_and_foreign_keys(tmp_path: Path) -> None:
    connection = connect(tmp_path / "sub" / "records.db")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_init_db_creates_tables_and_indexes(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert {"inspections", "review_decisions"} <= names
    assert {
        "idx_inspections_status",
        "idx_review_inspection",
    } <= names


def test_init_db_is_idempotent(conn: sqlite3.Connection) -> None:
    init_db(conn)
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 0


def test_schema_sql_is_packaged_source_of_truth() -> None:
    text = load_schema_sql()
    assert "CREATE TABLE IF NOT EXISTS inspections" in text
    assert "CREATE TABLE IF NOT EXISTS review_decisions" in text
    # PostgreSQL 이식 지침이 주석으로 남아 있어야 한다.
    assert "GENERATED ALWAYS AS IDENTITY" in text


def test_insert_inspection_returns_new_id(conn: sqlite3.Connection) -> None:
    inspection_id = insert_inspection(
        conn,
        _result(),
        engine="opencv",
        image_sha256=SHA_A,
        model_version="opencv-4.10",
    )
    assert inspection_id == 1
    row = get_inspection(conn, inspection_id)
    assert row is not None
    assert row["status"] == "NG"
    assert row["engine"] == "opencv"
    assert row["model_version"] == "opencv-4.10"
    assert row["split"] == "tune"
    assert json.loads(row["reasons"]) == ["DEFECT_EVIDENCE_STRONG"]
    assert json.loads(row["timing_ms"]) == {"total": 12.5}
    assert row["created_at"].endswith("Z")


def test_duplicate_sha256_returns_existing_id(conn: sqlite3.Connection) -> None:
    first = insert_inspection(conn, _result(), engine="opencv", image_sha256=SHA_A)
    second = insert_inspection(
        conn,
        _result(sample_id="70002", status=Status.OK, score=0.1),
        engine="opencv",
        image_sha256=SHA_A,
    )
    assert second == first
    assert conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1
    # 중복제거는 첫 기록을 덮어쓰지 않는다.
    row = get_inspection(conn, first)
    assert row is not None
    assert row["sample_id"] == "70001"
    assert row["status"] == "NG"


@pytest.mark.parametrize("sha", ["", "xyz", "a" * 63, "G" * 64])
def test_insert_rejects_invalid_sha256(conn: sqlite3.Connection, sha: str) -> None:
    with pytest.raises(RecordsError):
        insert_inspection(conn, _result(), engine="opencv", image_sha256=sha)
    assert conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 0


def test_insert_rejects_unknown_engine(conn: sqlite3.Connection) -> None:
    with pytest.raises(RecordsError):
        insert_inspection(conn, _result(), engine="magic", image_sha256=SHA_A)


def test_insert_rejects_error_without_result_status(
    conn: sqlite3.Connection,
) -> None:
    # ERROR도 기록 대상이지만 status 문자열은 contracts 의 값만 허용한다.
    insert_inspection(conn, _result(status=Status.ERROR), engine="opencv", image_sha256=SHA_A)
    assert get_inspection(conn, 1)["status"] == "ERROR"  # type: ignore[index]


def test_append_review_does_not_change_machine_status(conn: sqlite3.Connection) -> None:
    inspection_id = insert_inspection(
        conn,
        _result(status=Status.REVIEW, reasons=(ReasonCode.EVIDENCE_IN_REVIEW_BAND,)),
        engine="opencv",
        image_sha256=SHA_A,
    )
    before = get_inspection(conn, inspection_id)
    review_id = append_review(conn, inspection_id, "NG", reviewer="ildoc", reason="실결함")
    assert review_id == 1
    after = get_inspection(conn, inspection_id)
    assert before is not None and after is not None
    assert after["status"] == "REVIEW"
    assert after["score"] == pytest.approx(before["score"])
    reviews = conn.execute("SELECT * FROM review_decisions").fetchall()
    assert len(reviews) == 1
    assert reviews[0]["decision"] == "NG"
    assert reviews[0]["reviewer"] == "ildoc"
    assert reviews[0]["reason"] == "실결함"
    assert reviews[0]["created_at"].endswith("Z")


def test_append_review_is_append_only(conn: sqlite3.Connection) -> None:
    inspection_id = insert_inspection(conn, _result(), engine="opencv", image_sha256=SHA_A)
    append_review(conn, inspection_id, "NG", reviewer="a", reason="첫 판정")
    append_review(conn, inspection_id, "OK", reviewer="b", reason="재검")
    rows = conn.execute(
        "SELECT decision, reviewer FROM review_decisions ORDER BY id ASC"
    ).fetchall()
    assert [(row["decision"], row["reviewer"]) for row in rows] == [
        ("NG", "a"),
        ("OK", "b"),
    ]


@pytest.mark.parametrize("decision", ["MAYBE", "ok", "", "REVIEW"])
def test_append_review_rejects_invalid_decision(
    conn: sqlite3.Connection, decision: str
) -> None:
    inspection_id = insert_inspection(conn, _result(), engine="opencv", image_sha256=SHA_A)
    with pytest.raises(RecordsError):
        append_review(conn, inspection_id, decision)
    assert conn.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0] == 0


def test_append_review_rejects_unknown_inspection(conn: sqlite3.Connection) -> None:
    with pytest.raises(RecordsError):
        append_review(conn, 999, "OK")
    assert conn.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0] == 0


def test_list_inspections_filters_and_pages(conn: sqlite3.Connection) -> None:
    insert_inspection(conn, _result(sample_id="a"), engine="opencv", image_sha256=SHA_A)
    insert_inspection(
        conn,
        _result(sample_id="b", status=Status.OK, reasons=(ReasonCode.NO_DEFECT_EVIDENCE,)),
        engine="opencv",
        image_sha256=SHA_B,
    )
    insert_inspection(
        conn, _result(sample_id="c", status=Status.REVIEW), engine="opencv", image_sha256=SHA_C
    )

    every = list_inspections(conn, limit=10, offset=0)
    assert [row["sample_id"] for row in every] == ["a", "b", "c"]

    only_ok = list_inspections(conn, status="OK", limit=10, offset=0)
    assert [row["sample_id"] for row in only_ok] == ["b"]

    page = list_inspections(conn, limit=1, offset=1)
    assert [row["sample_id"] for row in page] == ["b"]

    with pytest.raises(RecordsError):
        list_inspections(conn, status="MAYBE", limit=10, offset=0)
    with pytest.raises(RecordsError):
        list_inspections(conn, limit=-1, offset=0)


def test_stats_reports_denominators(conn: sqlite3.Connection) -> None:
    empty = stats(conn)
    assert empty["total"] == 0
    assert empty["by_status"] == {"OK": 0, "NG": 0, "REVIEW": 0, "ERROR": 0}
    assert empty["ratios"]["ng"].value is None
    assert empty["ratios"]["ng"].denominator == 0

    insert_inspection(conn, _result(sample_id="a"), engine="opencv", image_sha256=SHA_A)
    insert_inspection(
        conn,
        _result(sample_id="b", status=Status.OK, reasons=(ReasonCode.NO_DEFECT_EVIDENCE,)),
        engine="opencv",
        image_sha256=SHA_B,
    )
    insert_inspection(
        conn, _result(sample_id="c", status=Status.REVIEW), engine="opencv", image_sha256=SHA_C
    )
    payload = stats(conn)
    assert payload["total"] == 3
    assert payload["by_status"] == {"OK": 1, "NG": 1, "REVIEW": 1, "ERROR": 0}
    assert payload["ratios"]["ng"].numerator == 1
    assert payload["ratios"]["ng"].denominator == 3
    assert payload["ratios"]["ng"].value == pytest.approx(1 / 3)
    assert payload["ratios"]["reviewed"].denominator == 3
    assert payload["ratios"]["reviewed"].value == pytest.approx(0.0)

    append_review(conn, 3, "NG", reviewer="ildoc", reason="확인")
    payload = stats(conn)
    assert payload["ratios"]["reviewed"].numerator == 1
    assert payload["ratios"]["reviewed"].value == pytest.approx(1 / 3)
    assert payload["reviews"]["decisions"] == 1
    assert payload["reviews"]["reviewed_inspections"] == 1
    # 기계 판정은 사람 판정 뒤에도 그대로다.
    assert payload["by_status"] == {"OK": 1, "NG": 1, "REVIEW": 1, "ERROR": 0}


def test_get_inspection_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert get_inspection(conn, 42) is None


def test_cli_records_init_and_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "derived" / "records.db"
    assert main(["records", "init", "--db", str(db_path)]) == 0
    assert db_path.is_file()
    captured = capsys.readouterr()
    assert "기록 DB" in captured.out

    assert main(["records", "stats", "--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 0
    assert payload["by_status"] == {"OK": 0, "NG": 0, "REVIEW": 0, "ERROR": 0}


def test_cli_records_list_without_db_returns_exit_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["records", "list", "--db", str(tmp_path / "missing.db")]) == 3
    assert "기록 DB" in capsys.readouterr().err
