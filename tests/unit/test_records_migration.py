"""SQLite 스키마 v1→v2 마이그레이션 및 fingerprint 기반 중복제거 단위 테스트.

v1 스키마(이미지 단독 UNIQUE) DB 를 만들어 migration 이 id·리뷰를 보존하는지,
같은 이미지의 다른 엔진/신원이 별도 행으로 기록되는지 검증한다.
모든 DB 는 ``tmp_path`` 아래에만 만든다(저장소 data/derived 미접촉).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from surface_sentinel.contracts import ComponentFeature, ReasonCode, SampleResult, Status
from surface_sentinel.records.db import connect, init_db, migrate_to_fingerprint
from surface_sentinel.records.fingerprint import inspection_fingerprint
from surface_sentinel.records.store import (
    find_inspection_id,
    get_inspection,
    insert_engine_observation,
    insert_inspection,
    list_engine_observations,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
CFG = "d" * 64

# 테스트 픽스처용 v1 스키마(과거 스키마 재현). 프로덕션 DDL 정본은 schema.sql 이다.
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
        "run_id": "run-mig",
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


def _v1_connection(tmp_path: Path) -> sqlite3.Connection:
    connection = connect(tmp_path / "v1.db")
    connection.executescript(V1_SCHEMA)
    connection.execute(
        "INSERT INTO inspections (run_id, sample_id, image_sha256, label, split, status,"
        " score, reasons, engine, config_hash, model_version, created_at)"
        " VALUES ('r1','s1',?,'defect','tune','NG',0.7,'[]','opencv',?,'opencv-4.10',"
        "'2026-11-13T00:00:00Z')",
        (SHA_A, CFG),
    )
    connection.execute(
        "INSERT INTO inspections (run_id, sample_id, image_sha256, label, split, status,"
        " score, reasons, engine, config_hash, model_version, created_at)"
        " VALUES ('r2','s2',?,'good','tune','REVIEW',0.3,'[]','opencv',?,'opencv-4.10',"
        "'2026-11-13T00:00:01Z')",
        (SHA_B, CFG),
    )
    connection.execute(
        "INSERT INTO review_decisions (inspection_id, decision, reviewer, reason, created_at)"
        " VALUES (2,'NG','ildoc','실결함','2026-11-13T00:00:02Z')"
    )
    connection.commit()
    return connection


def test_migration_preserves_ids_reviews_and_marks_incomplete(tmp_path: Path) -> None:
    connection = _v1_connection(tmp_path)
    try:
        assert migrate_to_fingerprint(connection) is True
        rows = connection.execute("SELECT * FROM inspections ORDER BY id").fetchall()
        assert [row["id"] for row in rows] == [1, 2]
        assert [row["image_sha256"] for row in rows] == [SHA_A, SHA_B]
        assert all(row["provenance_complete"] == 0 for row in rows)
        assert all(row["inspection_fingerprint"] for row in rows)
        assert all(row["model_sha256"] == "" for row in rows)
        reviews = connection.execute("SELECT * FROM review_decisions").fetchall()
        assert len(reviews) == 1
        assert reviews[0]["inspection_id"] == 2
        assert reviews[0]["decision"] == "NG"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    connection = _v1_connection(tmp_path)
    try:
        assert migrate_to_fingerprint(connection) is True
        assert migrate_to_fingerprint(connection) is False
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0] == 1
    finally:
        connection.close()


def test_init_db_migrates_existing_v1(tmp_path: Path) -> None:
    connection = _v1_connection(tmp_path)
    try:
        init_db(connection)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(inspections)")}
        assert "inspection_fingerprint" in columns
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0] == 1
    finally:
        connection.close()


def test_same_image_different_engine_creates_separate_rows(tmp_path: Path) -> None:
    connection = connect(tmp_path / "v2.db")
    init_db(connection)
    try:
        first = insert_inspection(connection, _result(), engine="opencv", image_sha256=SHA_A)
        second = insert_inspection(
            connection,
            _result(status=Status.OK, score=0.1),
            engine="yolo-seg",
            image_sha256=SHA_A,
            task="seg",
            model_sha256="e" * 64,
        )
        assert first != second
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 2
        assert get_inspection(connection, first)["engine"] == "opencv"
        assert get_inspection(connection, second)["engine"] == "yolo-seg"
    finally:
        connection.close()


def test_same_image_same_fingerprint_deduplicates(tmp_path: Path) -> None:
    connection = connect(tmp_path / "v2.db")
    init_db(connection)
    try:
        fp = inspection_fingerprint(engine="opencv", config_hash=CFG)
        first = insert_inspection(
            connection, _result(), engine="opencv", image_sha256=SHA_A, fingerprint=fp
        )
        second = insert_inspection(
            connection,
            _result(sample_id="70002", status=Status.OK),
            engine="opencv",
            image_sha256=SHA_A,
            fingerprint=fp,
        )
        assert first == second
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1
        assert get_inspection(connection, first)["status"] == "NG"
    finally:
        connection.close()


def test_find_inspection_id_by_fingerprint(tmp_path: Path) -> None:
    connection = connect(tmp_path / "v2.db")
    init_db(connection)
    try:
        fp_opencv = inspection_fingerprint(engine="opencv", config_hash=CFG)
        fp_yolo = inspection_fingerprint(
            engine="yolo-seg", task="seg", model_sha256="e" * 64, config_hash=CFG
        )
        opencv_id = insert_inspection(
            connection, _result(), engine="opencv", image_sha256=SHA_A, fingerprint=fp_opencv
        )
        yolo_id = insert_inspection(
            connection,
            _result(),
            engine="yolo-seg",
            image_sha256=SHA_A,
            task="seg",
            model_sha256="e" * 64,
            fingerprint=fp_yolo,
        )
        assert find_inspection_id(connection, SHA_A, fp_opencv) == opencv_id
        assert find_inspection_id(connection, SHA_A, fp_yolo) == yolo_id
        assert find_inspection_id(connection, SHA_A, "f" * 64) is None
        assert find_inspection_id(connection, SHA_A) == opencv_id
    finally:
        connection.close()


@pytest.mark.parametrize("engine", ["opencv", "yolo-seg"])
def test_provenance_complete_flag(tmp_path: Path, engine: str) -> None:
    connection = connect(tmp_path / "v2.db")
    init_db(connection)
    try:
        inspection_id = insert_inspection(
            connection,
            _result(),
            engine=engine,
            image_sha256=SHA_A,
            model_sha256="e" * 64 if engine != "opencv" else "",
        )
        row = get_inspection(connection, inspection_id)
        assert row["provenance_complete"] == 1
    finally:
        connection.close()


def _component() -> ComponentFeature:
    return ComponentFeature(
        index=0,
        area_px=900,
        bbox=(100, 40, 130, 70),
        area_ratio=0.01,
        aspect_ratio=1.0,
        elongation=1.0,
        solidity=1.0,
        mean_contrast_gray=0.0,
        max_contrast_gray=0.0,
        integrated_contrast=0.0,
        border_distance_px=40.0,
        touches_roi_border=False,
        evidence=0.8,
    )


def test_engine_observation_preserves_and_dedups(tmp_path: Path) -> None:
    connection = connect(tmp_path / "v2.db")
    init_db(connection)
    try:
        inspection_id = insert_inspection(
            connection, _result(), engine="combined", image_sha256=SHA_A
        )
        yolo = _result(status=Status.NG, score=0.8, components=(_component(),))
        first = insert_engine_observation(
            connection, inspection_id, yolo, engine="yolo-seg", task="seg", model_sha256="e" * 64
        )
        # 같은 (inspection, engine) 재기록은 기존 id 반환(불변).
        again = insert_engine_observation(
            connection, inspection_id, _result(status=Status.OK, score=0.0), engine="yolo-seg"
        )
        assert first == again
        opencv_id = insert_engine_observation(
            connection, inspection_id, _result(status=Status.OK, score=0.1), engine="opencv"
        )
        assert opencv_id != first

        rows = list_engine_observations(connection, inspection_id)
        assert [row["engine"] for row in rows] == ["yolo-seg", "opencv"]
        seg_row = rows[0]
        assert seg_row["status"] == "NG"
        assert seg_row["score"] == pytest.approx(0.8)
        assert seg_row["task"] == "seg"
        assert seg_row["model_sha256"] == "e" * 64
        components = json.loads(seg_row["components"])
        assert components[0]["area_px"] == 900
        assert components[0]["evidence"] == pytest.approx(0.8)
    finally:
        connection.close()


def test_migration_creates_engine_observations_table(tmp_path: Path) -> None:
    connection = _v1_connection(tmp_path)
    try:
        assert migrate_to_fingerprint(connection) is True
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "engine_observations" in tables
        assert connection.execute("SELECT COUNT(*) FROM engine_observations").fetchone()[0] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()