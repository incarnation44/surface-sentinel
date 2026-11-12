"""결정론 워커 단위 테스트 (E5, ADR 0007 / D6).

``tmp_path`` 안에서만 합성 이미지를 만들고 검사한다. 워커는 네트워크/셸을
쓰지 않으며 판정을 바꾸지 않는다. 이 레인은 ``opencv`` 엔진 경로만 검증한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

import synthetic
from surface_sentinel.agent.worker import (
    WorkerError,
    audit_log_path,
    run_worker,
)
from surface_sentinel.config import AppConfig, load_config
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import list_inspections, stats

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "base.toml")
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


def _write_incoming(incoming: Path) -> dict[str, Path]:
    """정상 1장과 결함 1장을 결정론적으로 만든다."""

    good = synthetic.write_png(incoming / "good_a.png", synthetic.good_image())
    defect_image, _mask = synthetic.dark_defect_image()
    defect = synthetic.write_png(incoming / "defect_b.png", defect_image)
    return {"good": good, "defect": defect}


def _audit_events(db_path: Path) -> list[dict[str, object]]:
    path = audit_log_path(db_path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rows(db_path: Path) -> list[dict[str, object]]:
    connection = connect(db_path)
    try:
        init_db(connection)
        return list_inspections(connection, limit=100, offset=0)
    finally:
        connection.close()


def test_once_processes_two_images(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "records.db"

    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    rows = _rows(db_path)
    assert len(rows) == 2
    by_sample = {row["sample_id"]: row for row in rows}
    assert by_sample["good_a"]["status"] == "OK"
    assert by_sample["defect_b"]["status"] == "NG"
    assert all(row["engine"] == "opencv" for row in rows)
    assert all(row["split"] == "tune" for row in rows)
    assert all(len(str(row["image_sha256"])) == 64 for row in rows)
    assert all(row["config_hash"] == BASE_CONFIG.config_hash for row in rows)
    # run_id는 실행 단위로 하나이고 ISO 시각 + config 앞 8자를 쓴다.
    run_ids = {row["run_id"] for row in rows}
    assert len(run_ids) == 1
    assert RUN_ID_PATTERN.match(run_ids.pop()) is not None


def test_rerun_deduplicates_by_sha256(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "records.db"

    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    assert len(_rows(db_path)) == 2
    events = _audit_events(db_path)
    assert [event["event"] for event in events].count("skip") == 2
    assert [event["event"] for event in events].count("processed") == 2


def test_broken_image_is_retried_then_recorded_as_error(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "broken.png").write_bytes(b"not a png at all")
    db_path = tmp_path / "records.db"

    run_worker(
        incoming=incoming,
        db_path=db_path,
        config=BASE_CONFIG,
        max_retries=3,
        once=True,
    )

    rows = _rows(db_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "ERROR"
    attempts = [
        event["attempt"] for event in _audit_events(db_path) if event["event"] == "error"
    ]
    # max_retries=3 → 시도 3회를 모두 남기고 마지막에 ERROR로 기록한다.
    assert attempts == [1, 2, 3]


def test_zero_retries_still_records_error(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "broken.png").write_bytes(b"not a png at all")
    db_path = tmp_path / "records.db"

    run_worker(
        incoming=incoming,
        db_path=db_path,
        config=BASE_CONFIG,
        max_retries=0,
        once=True,
    )

    rows = _rows(db_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "ERROR"


def test_ng_overlay_kept_and_ok_overlay_skipped_when_rate_zero(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "records.db"

    run_worker(
        incoming=incoming,
        db_path=db_path,
        config=BASE_CONFIG,
        ok_sample_rate=0,
        once=True,
    )

    by_sample = {row["sample_id"]: row for row in _rows(db_path)}
    assert by_sample["defect_b"]["overlay_path"].endswith("defect_b_overlay.png")
    assert Path(str(by_sample["defect_b"]["overlay_path"])).is_file()
    # OK는 표본 밖이므로 오버레이를 남기지 않는다.
    assert by_sample["good_a"]["overlay_path"] == ""


def test_ok_overlay_kept_when_rate_one(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "records.db"

    run_worker(
        incoming=incoming,
        db_path=db_path,
        config=BASE_CONFIG,
        ok_sample_rate=1,
        once=True,
    )

    by_sample = {row["sample_id"]: row for row in _rows(db_path)}
    for sample_id in ("good_a", "defect_b"):
        overlay = Path(str(by_sample[sample_id]["overlay_path"]))
        assert overlay.is_file()
        assert overlay.name == f"{sample_id}_overlay.png"


def test_files_processed_in_sorted_order(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    # 이름 순서(z → a)와 파일 시스템 순서가 달라도 정렬해서 처리한다.
    synthetic.write_png(incoming / "z_last.png", synthetic.good_image())
    synthetic.write_png(
        incoming / "a_first.png",
        synthetic.to_bgr(synthetic.surface_gray(seed=99)),
    )
    db_path = tmp_path / "records.db"

    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    processed = [
        event["sample_id"]
        for event in _audit_events(db_path)
        if event["event"] == "processed"
    ]
    assert processed == ["a_first", "z_last"]


def test_audit_log_is_append_only_jsonl(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "records.db"

    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)
    first = audit_log_path(db_path).read_text(encoding="utf-8")
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)
    second = audit_log_path(db_path).read_text(encoding="utf-8")

    assert second.startswith(first)
    assert len(second.splitlines()) == len(first.splitlines()) * 2
    for line in second.splitlines():
        payload = json.loads(line)
        assert {"ts", "event", "run_id", "sample_id"} <= set(payload)


def test_invalid_arguments_raise(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    db_path = tmp_path / "records.db"

    with pytest.raises(WorkerError):
        run_worker(
            incoming=incoming,
            db_path=db_path,
            config=BASE_CONFIG,
            engine="magic",
            once=True,
        )
    with pytest.raises(WorkerError):
        run_worker(
            incoming=incoming,
            db_path=db_path,
            config=BASE_CONFIG,
            ok_sample_rate=-1,
            once=True,
        )


def test_missing_incoming_is_created(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming-new"
    db_path = tmp_path / "records.db"

    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    assert incoming.is_dir()
    assert _rows(db_path) == []


def test_empty_incoming_records_no_rows(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    db_path = tmp_path / "records.db"

    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    assert _rows(db_path) == []
    connection = connect(db_path)
    try:
        payload = stats(connection)
    finally:
        connection.close()
    assert payload["total"] == 0
    assert payload["ratios"]["ng"].denominator == 0


def test_resolve_yolo_entry_points_to_inspect_module() -> None:
    """워커는 ``surface_sentinel.yolo.inspect`` 진입점을 지연 import로 찾는다."""

    from surface_sentinel.agent.worker import _resolve_yolo_entry
    from surface_sentinel.yolo import inspect as yolo_inspect

    assert _resolve_yolo_entry("yolo-det") is yolo_inspect.inspect_with_yolo
    assert _resolve_yolo_entry("combined") is yolo_inspect.inspect_with_yolo


def test_worker_records_yolo_engine_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """engine="yolo-det" 경로가 stub 진입점으로 DB 기록까지 간다(모델 파일 불필요)."""

    from surface_sentinel.pipeline import InspectionOutcome
    from surface_sentinel.pipeline.evidence import sanitize_sample_id
    from surface_sentinel.yolo import inspect as yolo_inspect
    from surface_sentinel.yolo.adapter import to_sample_result
    from surface_sentinel.yolo.inference import Detection
    from surface_sentinel.yolo.overlay import save_overlay

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "records.db"
    calls: list[tuple[str, str, str]] = []

    def stub_entry(
        image_path: Path,
        config: AppConfig,
        *,
        engine: str,
        output_dir: Path | None = None,
        split: str = "tune",
        label: str = "unknown",
        **_kwargs: object,
    ) -> InspectionOutcome:
        calls.append((Path(image_path).name, engine, split))
        bgr = cv2.imread(str(image_path))
        assert bgr is not None
        detection = Detection(bbox_xyxy=(100.0, 40.0, 130.0, 70.0), score=0.8)
        sample_id = Path(image_path).stem
        result = to_sample_result(
            [detection],
            run_id="stub-run",
            sample_id=sample_id,
            image_size=(int(bgr.shape[1]), int(bgr.shape[0])),
            split=split,
            label=label,
            config_hash=config.config_hash,
        )
        overlay_path = ""
        if output_dir is not None:
            saved = save_overlay(
                bgr,
                [detection],
                Path(output_dir) / f"{sanitize_sample_id(sample_id)}_yolo.png",
            )
            overlay_path = str(saved)
        return InspectionOutcome(result=replace(result, overlay_path=overlay_path))

    monkeypatch.setattr(yolo_inspect, "inspect_with_yolo", stub_entry)

    run_worker(
        incoming=incoming,
        db_path=db_path,
        config=BASE_CONFIG,
        engine="yolo-det",
        ok_sample_rate=1,
        once=True,
    )

    rows = _rows(db_path)
    assert len(rows) == 2
    assert calls == [
        ("defect_b.png", "yolo-det", "tune"),
        ("good_a.png", "yolo-det", "tune"),
    ]
    for row in rows:
        assert row["engine"] == "yolo-det"
        assert row["model_version"] == "yolo-det"
        assert row["split"] == "tune"
        assert row["status"] == "NG"
        overlay = Path(str(row["overlay_path"]))
        assert overlay.is_file()
        assert overlay.name == f"{row['sample_id']}_yolo.png"


def test_cli_worker_run_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from surface_sentinel.cli import main

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _write_incoming(incoming)
    db_path = tmp_path / "derived" / "records.db"

    code = main(
        [
            "worker",
            "run",
            "--incoming",
            str(incoming),
            "--db",
            str(db_path),
            "--once",
            "--config",
            str(REPO_ROOT / "configs" / "base.toml"),
            "--ok-sample-rate",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "처리" in captured.out
    assert "기록" in captured.out
    assert len(_rows(db_path)) == 2
