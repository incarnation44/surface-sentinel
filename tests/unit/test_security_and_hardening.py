"""감사 1~3순위 반영 검증 테스트.

1순위: 디코드 전 픽셀 상한 가드, API 이벤트루프 분리, 동시성 세마포어, OpenCV 스레드 제한.
2순위: 폼/API CSRF Origin 검사, 메모·검토자 길이 제한, 동시 append 안전, 손상 JSONL 방어.
3순위: 워커 ERROR 재검사(append-only 원장 불변), YOLO 전치 출력 shape 가드.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import synthetic
import ui_run
from surface_sentinel.agent.worker import (
    WorkerError,
    audit_log_path,
    recheck_errors,
    run_worker,
)
from surface_sentinel.api import create_inspection_app
from surface_sentinel.cli import main as cli_main
from surface_sentinel.config import load_config
from surface_sentinel.contracts import ErrorCode, SampleResult, Stage, Status
from surface_sentinel.io.image_guard import (
    ImagePixelLimitError,
    check_image_pixel_budget,
    probe_image_dimensions,
)
from surface_sentinel.io.limits import (
    CsrfOriginError,
    configure_opencv_threads,
    pipeline_slot,
    validate_same_origin,
)
from surface_sentinel.pipeline import InspectionOutcome, inspect_image
from surface_sentinel.records.db import connect, init_db, verify_records_integrity
from surface_sentinel.records.fingerprint import inspection_fingerprint
from surface_sentinel.records.identity import opencv_spec
from surface_sentinel.records.store import (
    MAX_REASON_LENGTH,
    MAX_REVIEWER_LENGTH,
    RecordsError,
    append_review,
    insert_inspection,
    list_inspections,
)
from surface_sentinel.review.store import (
    MAX_NOTE_LENGTH,
    ReviewError,
    ReviewStore,
)
from surface_sentinel.ui import create_app
from surface_sentinel.ui.data import UiInputError, sample_result_from_record
from surface_sentinel.yolo.inference import decode_det, decode_seg
from surface_sentinel.yolo.inspect import inspect_with_yolo

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "base.toml")


def _png_header(width: int, height: int) -> bytes:
    """실제 압축 픽셀 없이 IHDR 치수만 담은 최소 PNG 바이트."""

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + b"\x00\x00\x00\x00"
    )


def _jpeg_sof0_header(width: int, height: int) -> bytes:
    """최소 JPEG SOI + SOF0 헤더."""

    body = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(body) + 2) + body + b"\xff\xd9"


def _bmp_header(width: int, height: int) -> bytes:
    """최소 54바이트 BMP 헤더."""

    file_hdr = b"BM" + struct.pack("<IHHI", 54, 0, 0, 54)
    dib_hdr = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0)
    return file_hdr + dib_hdr


def _tiff_header(width: int, height: int) -> bytes:
    """최소 리틀엔디안 TIFF 헤더(Tag 256=width, 257=height)."""

    entry_w = struct.pack("<HHII", 256, 4, 1, width)
    entry_h = struct.pack("<HHII", 257, 4, 1, height)
    return (
        b"II*\x00"
        + struct.pack("<I", 8)
        + struct.pack("<H", 2)
        + entry_w
        + entry_h
        + b"\x00" * 4
    )


def _valid_png_bytes(size: int = 96) -> bytes:
    image = np.tile(np.linspace(40, 200, size, dtype=np.uint8), (size, 1))
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


# --- 1순위: 디코드 전 픽셀 상한 가드 -----------------------------------------


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (_png_header, (20_000, 20_000)),
        (_jpeg_sof0_header, (10_000, 10_000)),
        (_bmp_header, (20_000, 20_000)),
        (_tiff_header, (20_000, 20_000)),
    ],
)
def test_probe_and_reject_oversized_headers_across_formats(
    builder: Any, expected: tuple[int, int]
) -> None:
    payload = builder(*expected)
    assert probe_image_dimensions(payload) == expected
    with pytest.raises(ImagePixelLimitError):
        check_image_pixel_budget(payload)


def test_pipeline_rejects_png_bomb_before_imdecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(_png_header(20_000, 20_000))

    def _forbid_decode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("디코드 전 가드가 먼저 차단해야 합니다")

    monkeypatch.setattr(cv2, "imdecode", _forbid_decode)
    outcome = inspect_image(bomb, BASE_CONFIG, split="tune")
    assert outcome.result.status is Status.ERROR
    assert outcome.result.error_code is ErrorCode.E_INPUT_SIZE


def test_yolo_inspect_rejects_png_bomb_before_imread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(_png_header(20_000, 20_000))
    dummy_onnx = tmp_path / "model.onnx"
    dummy_onnx.write_bytes(b"fake-onnx")

    def _forbid_imread(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("YOLO 경로도 디코드 전 가드가 먼저 차단해야 합니다")

    monkeypatch.setattr(cv2, "imread", _forbid_imread)
    outcome = inspect_with_yolo(
        bomb, BASE_CONFIG, engine="yolo-det", onnx_path=dummy_onnx, split="tune"
    )
    assert outcome.result.status is Status.ERROR
    assert outcome.result.error_code is ErrorCode.E_INPUT_SIZE


def test_api_rejects_png_bomb_with_413_before_disk_and_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("test-sha", False))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    client = TestClient(create_inspection_app(runs_dir=runs_dir))

    resp = client.post(
        "/api/inspect",
        files={"file": ("bomb.png", _png_header(20_000, 20_000), "image/png")},
        data={"label": "unknown", "split": "tune"},
    )
    assert resp.status_code == 413
    assert "해상도" in resp.json()["detail"]
    assert not (runs_dir / "api_uploads").exists()


def test_cinema_rejects_png_bomb_before_imread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    ui_run.make_run(runs_dir, raw_root)
    client = TestClient(create_app(runs_dir=runs_dir, raw_root=raw_root))

    def _forbid_imread(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("시네마 경로도 디코드 전 가드가 먼저 차단해야 합니다")

    monkeypatch.setattr(cv2, "imread", _forbid_imread)
    resp = client.post(
        "/cinema/inspect",
        files={"file": ("bomb.png", _png_header(20_000, 20_000), "image/png")},
    )
    assert resp.status_code == 400
    assert "해상도" in resp.json()["detail"]


# --- 1순위: 동시성 세마포어 및 OpenCV 스레드 상한 ----------------------------


def test_pipeline_slot_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURFACE_SENTINEL_MAX_CONCURRENCY", "2")
    active = 0
    peak = 0
    lock = threading.Lock()

    def _work() -> None:
        nonlocal active, peak
        with pipeline_slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda _: _work(), range(6)))

    assert peak <= 2


def test_configure_opencv_threads_applies_cap() -> None:
    applied = configure_opencv_threads(2)
    assert applied == 2
    assert cv2.getNumThreads() <= 2


# --- 2순위: CSRF Origin 검사, 길이 제한, 동시 append, 손상 JSONL 방어 ---------


def test_validate_same_origin_blocks_cross_site_and_allows_local() -> None:
    validate_same_origin({}, "127.0.0.1:8000")
    validate_same_origin(
        {"Origin": "http://127.0.0.1:8000", "Sec-Fetch-Site": "same-origin"},
        "127.0.0.1:8000",
    )
    with pytest.raises(CsrfOriginError):
        validate_same_origin({"Origin": "https://evil.example"}, "127.0.0.1:8000")
    with pytest.raises(CsrfOriginError):
        validate_same_origin({"Origin": "null"}, "127.0.0.1:8000")
    with pytest.raises(CsrfOriginError):
        validate_same_origin({"Referer": "https://evil.example/form"}, "127.0.0.1:8000")
    with pytest.raises(CsrfOriginError):
        validate_same_origin({"Sec-Fetch-Site": "cross-site"}, "127.0.0.1:8000")


def test_ui_and_api_post_routes_block_cross_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("test-sha", False))
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run_dir = ui_run.make_run(runs_dir, raw_root)
    client = TestClient(create_app(runs_dir=runs_dir, raw_root=raw_root, enable_api=True))

    evil = {"Origin": "https://evil.example"}
    r_review = client.post(
        f"/runs/{run_dir.name}/review/rev_01",
        data={
            "final_status": "OK",
            "reason_code": "FALSE_ALARM",
            "note": "ok",
            "reviewer": "local",
        },
        headers=evil,
    )
    assert r_review.status_code == 403

    r_cinema = client.post(
        "/cinema/inspect",
        files={"file": ("ok.png", _valid_png_bytes(), "image/png")},
        headers=evil,
    )
    assert r_cinema.status_code == 403

    r_api = client.post(
        "/api/inspect",
        files={"file": ("ok.png", _valid_png_bytes(), "image/png")},
        headers=evil,
    )
    assert r_api.status_code == 403


def test_review_length_limits_and_concurrent_append(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run_dir = ui_run.make_run(runs_dir, raw_root)
    store = ReviewStore(run_dir)
    hex64 = "a" * 64

    with pytest.raises(ReviewError):
        store.append(
            sample_id="rev_01",
            machine_status="REVIEW",
            final_status="OK",
            reason_code="FALSE_ALARM",
            reviewer="r" * (MAX_REVIEWER_LENGTH + 1),
            note="ok",
            config_hash=hex64,
            overlay_sha256=hex64,
        )
    with pytest.raises(ReviewError):
        store.append(
            sample_id="rev_01",
            machine_status="REVIEW",
            final_status="OK",
            reason_code="FALSE_ALARM",
            reviewer="local",
            note="n" * (MAX_NOTE_LENGTH + 1),
            config_hash=hex64,
            overlay_sha256=hex64,
        )

    def _write_one(idx: int) -> None:
        store.append(
            sample_id="rev_01",
            machine_status="REVIEW",
            final_status="OK",
            reason_code="FALSE_ALARM",
            reviewer=f"user-{idx}",
            note=f"note-{idx}-" + ("x" * 200),
            config_hash=hex64,
            overlay_sha256=hex64,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write_one, range(20)))

    assert len(store.read_all()) == 20


def test_records_append_review_enforces_length_limits(tmp_path: Path) -> None:
    db_path = tmp_path / "records.db"
    conn = connect(db_path)
    try:
        init_db(conn)
        sample = SampleResult(
            run_id="r1",
            sample_id="s1",
            original_id="s1",
            split="tune",
            label="unknown",
            status=Status.REVIEW,
            score=0.3,
            config_hash=BASE_CONFIG.config_hash,
        )
        insp_id = insert_inspection(conn, sample, engine="opencv", image_sha256="b" * 64)
        with pytest.raises(RecordsError):
            append_review(conn, insp_id, "OK", reviewer="x" * (MAX_REVIEWER_LENGTH + 1))
        with pytest.raises(RecordsError):
            append_review(conn, insp_id, "OK", reviewer="qa", reason="y" * (MAX_REASON_LENGTH + 1))
    finally:
        conn.close()


def test_corrupted_results_jsonl_raises_ui_input_error_not_500(tmp_path: Path) -> None:
    with pytest.raises(UiInputError):
        sample_result_from_record({"sample_id": "s1", "status": "NOT_A_STATUS"})
    with pytest.raises(UiInputError):
        sample_result_from_record(
            {"sample_id": "s1", "status": "OK", "components": [{"bbox": [1, 2]}]}
        )

    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run_dir = ui_run.make_run(runs_dir, raw_root)
    results_file = run_dir / "results.jsonl"
    lines = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
    lines[0]["status"] = "BROKEN_STATUS"
    results_file.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(runs_dir=runs_dir, raw_root=raw_root))
    resp = client.get(f"/runs/{run_dir.name}/samples/{lines[0]['sample_id']}")
    assert resp.status_code == 404


# --- 3순위: 워커 ERROR 재검사(append-only) & YOLO 전치 shape 가드 ------------


def test_fingerprint_backward_compatible_and_recheck_distinct() -> None:
    base = inspection_fingerprint(engine="opencv", config_hash=BASE_CONFIG.config_hash)
    empty_tag = inspection_fingerprint(
        engine="opencv", config_hash=BASE_CONFIG.config_hash, recheck_tag=""
    )
    recheck = inspection_fingerprint(
        engine="opencv", config_hash=BASE_CONFIG.config_hash, recheck_tag="recheck-1"
    )
    assert base == empty_tag
    assert base != recheck
    spec = opencv_spec(BASE_CONFIG.config_hash)
    assert spec.fingerprint == base
    assert spec.with_recheck_tag("recheck-1").fingerprint == recheck


def test_worker_recheck_recovers_error_preserving_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.agent import worker as worker_mod
    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("", False))

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    good_path = synthetic.write_png(incoming / "good_a.png", synthetic.good_image())
    db_path = tmp_path / "records.db"

    real_inspect = worker_mod._inspect

    def _transient_error(*args: Any, **kwargs: Any) -> InspectionOutcome:
        return InspectionOutcome(
            result=SampleResult(
                run_id="tmp",
                sample_id=good_path.stem,
                original_id=good_path.stem,
                split="tune",
                label="unknown",
                status=Status.ERROR,
                score=0.0,
                error_code=ErrorCode.E_PROCESS_INTERNAL,
                error_stage=Stage.PREPROCESS,
                config_hash=BASE_CONFIG.config_hash,
            )
        )

    # 1) 일시 오류로 ERROR 행이 기록된다.
    monkeypatch.setattr(worker_mod, "_inspect", _transient_error)
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, max_retries=1, once=True)

    # 2) 복구 후 일반 run_worker는 중복제거로 스킵한다.
    monkeypatch.setattr(worker_mod, "_inspect", real_inspect)
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    conn = connect(db_path)
    try:
        rows_before = list_inspections(conn)
        assert len(rows_before) == 1
        assert rows_before[0]["status"] == "ERROR"
    finally:
        conn.close()

    # 3) worker recheck 실행 → 기존 ERROR 행을 유지한 채 새 OK 행을 append 한다.
    counts = recheck_errors(incoming=incoming, db_path=db_path, config=BASE_CONFIG)
    assert counts == {"rechecked": 1, "resolved": 1, "still_error": 0, "skipped": 0, "error": 0}

    # 4) 다시 실행하면 이미 복구된 이미지로 판단해 건너뛴다(멱등).
    counts_second = recheck_errors(incoming=incoming, db_path=db_path, config=BASE_CONFIG)
    assert counts_second["rechecked"] == 0
    assert counts_second["skipped"] == 1

    conn = connect(db_path)
    try:
        rows_after = list_inspections(conn)
        assert len(rows_after) == 2
        assert rows_after[0]["status"] == "ERROR"  # 기존 행 불변
        assert rows_after[1]["status"] == "OK"  # 새 provenance 행
        assert rows_after[0]["inspection_fingerprint"] != rows_after[1]["inspection_fingerprint"]
        integrity = verify_records_integrity(conn)
        assert integrity.ok
    finally:
        conn.close()

    events = [
        json.loads(line)
        for line in audit_log_path(db_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recheck_events = [e for e in events if e["event"] == "recheck"]
    assert len(recheck_events) == 1
    assert recheck_events[0]["recheck_tag"] == "recheck-1"
    assert recheck_events[0]["previous_inspection_ids"] == [1]
    assert recheck_events[0]["status"] == "OK"


def test_worker_recheck_cli_and_invalid_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("", False))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    db_path = tmp_path / "records.db"

    with pytest.raises(WorkerError):
        recheck_errors(incoming=incoming, db_path=db_path, config=BASE_CONFIG, tag="bad tag!")

    code = cli_main(
        [
            "worker",
            "recheck",
            "--incoming",
            str(incoming),
            "--db",
            str(db_path),
            "--config",
            str(REPO_ROOT / "configs" / "base.toml"),
        ]
    )
    assert code == 0


def test_yolo_decode_rejects_transposed_exports() -> None:
    with pytest.raises(ValueError, match="전치"):
        decode_det(np.zeros((1, 8400, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="전치"):
        decode_det(np.zeros((1, 300, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="전치"):
        decode_seg(
            np.zeros((1, 3549, 37), dtype=np.float32),
            np.zeros((1, 32, 104, 104), dtype=np.float32),
            orig_w=200,
            orig_h=200,
        )
    # 정상 shape 는 그대로 통과한다.
    boxes, scores = decode_det(np.zeros((1, 5, 8400), dtype=np.float32))
    assert boxes.shape == (0, 4) and scores.shape == (0,)


# --- 보완 검증: fail-closed 헤더, 시네마 단일 디코드, 재검사 통계, DNS 리바인딩, 비동기 슬롯 ---


def test_unrecognized_header_and_disguised_ppm_blocked_before_imdecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.io.image_guard import ImageHeaderError

    # 1) 알 수 없는 헤더는 기본(fail-closed)으로 즉시 거부한다.
    with pytest.raises(ImageHeaderError):
        check_image_pixel_budget(b"UNKNOWN_MAGIC_HEADER_DATA")

    # 2) .png로 위장한 초대형 PPM(Netpbm)도 사전 가드가 디코드 전에 차단한다.
    ppm_bomb = b"P6\n20000 20000\n255\n" + b"\x00" * 16
    assert probe_image_dimensions(ppm_bomb) == (20_000, 20_000)
    with pytest.raises(ImagePixelLimitError):
        check_image_pixel_budget(ppm_bomb)

    # 3) 손상된 헤더 파일은 cv2.imdecode 호출 전에 E_INPUT_DECODE 로 끝난다.
    broken = tmp_path / "broken_header.png"
    broken.write_bytes(b"definitely-not-an-image")

    def _forbid_decode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("미인식 헤더는 cv2.imdecode 전에 차단되어야 합니다")

    monkeypatch.setattr(cv2, "imdecode", _forbid_decode)
    outcome = inspect_image(broken, BASE_CONFIG, split="tune")
    assert outcome.result.status is Status.ERROR
    assert outcome.result.error_code is ErrorCode.E_INPUT_DECODE


def test_cinema_trace_decodes_source_image_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.ui.cinema import run_cinema_trace

    img_path = synthetic.write_png(tmp_path / "single_decode.png", synthetic.good_image())
    real_imread = cv2.imread
    imread_paths: list[Path] = []

    def _spy_imread(path_str: str, *args: Any, **kwargs: Any) -> Any:
        imread_paths.append(Path(path_str).resolve())
        return real_imread(path_str, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", _spy_imread)
    payload = run_cinema_trace(img_path, sample_id="single_decode", config=BASE_CONFIG)
    assert payload["status"] == "OK"
    # 원본 이미지는 inspect_image 한 번만 디코드하며 run_cinema_trace가 중복 imread 하지 않는다.
    assert img_path.resolve() not in imread_paths


def test_stats_and_engine_counts_exclude_superseded_recheck_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.agent import worker as worker_mod
    from surface_sentinel.api.app import engine_status_counts
    from surface_sentinel.evaluation import recalibration
    from surface_sentinel.records.store import stats as records_stats

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("", False))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    good_path = synthetic.write_png(incoming / "img1.png", synthetic.good_image())
    db_path = tmp_path / "records.db"
    real_inspect = worker_mod._inspect

    def _transient_error(*args: Any, **kwargs: Any) -> InspectionOutcome:
        return InspectionOutcome(
            result=SampleResult(
                run_id="tmp",
                sample_id=good_path.stem,
                original_id=good_path.stem,
                split="tune",
                label="unknown",
                status=Status.ERROR,
                score=0.0,
                error_code=ErrorCode.E_PROCESS_INTERNAL,
                error_stage=Stage.PREPROCESS,
                config_hash=BASE_CONFIG.config_hash,
            )
        )

    monkeypatch.setattr(worker_mod, "_inspect", _transient_error)
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, max_retries=1, once=True)

    monkeypatch.setattr(worker_mod, "_inspect", real_inspect)
    recheck_errors(incoming=incoming, db_path=db_path, config=BASE_CONFIG)

    conn = connect(db_path)
    try:
        # 원장에는 2행(ERROR+OK)이 불변 보존되고 통계는 대체된 ERROR를 제외한다.
        assert len(list_inspections(conn)) == 2
        summary = records_stats(conn)
        assert summary["total"] == 1
        assert summary["by_status"] == {"OK": 1, "NG": 0, "REVIEW": 0, "ERROR": 0}
        assert engine_status_counts(conn) == {
            "opencv": {"OK": 1, "NG": 0, "REVIEW": 0, "ERROR": 0}
        }
    finally:
        conn.close()


def test_validate_same_origin_blocks_dns_rebinding_host() -> None:
    # 공격자 도메인이 127.0.0.1로 풀려 Host와 Origin이 일치해도 로컬 허용 목록 밖이면 차단한다.
    with pytest.raises(CsrfOriginError):
        validate_same_origin({"Origin": "http://evil.example:8000"}, "evil.example:8000")
    with pytest.raises(CsrfOriginError):
        validate_same_origin({}, "evil.example:8000")
    # 로컬 호스트는 정상 허용한다.
    validate_same_origin({"Origin": "http://localhost:8000"}, "localhost:8000")
    validate_same_origin({"Origin": "http://127.0.0.1:8000"}, "127.0.0.1:8000")


def test_async_pipeline_slot_caps_concurrency_before_threadpool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from surface_sentinel.io.limits import async_pipeline_slot

    monkeypatch.setenv("SURFACE_SENTINEL_MAX_CONCURRENCY", "2")

    async def _run() -> int:
        active = 0
        peak = 0

        async def _worker() -> None:
            nonlocal active, peak
            async with async_pipeline_slot():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.02)
                active -= 1

        await asyncio.gather(*(_worker() for _ in range(6)))
        return peak

    assert asyncio.run(_run()) <= 2


def test_image_guard_blocks_zero_dimensions_and_malformed_sof_fail_closed(
    tmp_path: Path,
) -> None:
    from surface_sentinel.io.image_guard import (
        ImageHeaderError,
        check_decoded_pixel_budget,
        probe_image_file_dimensions,
    )

    gif_zero = b"GIF89a" + struct.pack("<HH", 0, 16)
    webp_vp8_zero = (
        b"RIFF"
        + struct.pack("<I", 22)
        + b"WEBP"
        + b"VP8 "
        + struct.pack("<I", 10)
        + b"\x00\x00\x00"
        + b"\x9d\x01\x2a"
        + struct.pack("<HH", 0, 64)
    )
    jpeg_dnl_zero_h = _jpeg_sof0_header(128, 0)
    jpeg_short_sof = b"\xff\xd8\xff\xc0" + struct.pack(">H", 5) + b"\x08\x00\x10\x00\x10\xff\xd9"
    be_tiff_valid = (
        b"MM\x00*"
        + struct.pack(">I", 8)
        + struct.pack(">H", 2)
        + struct.pack(">HHII", 256, 4, 1, 320)
        + struct.pack(">HHII", 257, 4, 1, 240)
        + b"\x00" * 4
    )
    be_tiff_zero = (
        b"MM\x00*"
        + struct.pack(">I", 8)
        + struct.pack(">H", 2)
        + struct.pack(">HHII", 256, 4, 1, 0)
        + struct.pack(">HHII", 257, 4, 1, 240)
        + b"\x00" * 4
    )

    for payload in (
        _png_header(0, 100),
        _png_header(100, 0),
        gif_zero,
        _bmp_header(0, 100),
        _bmp_header(100, 0),
        _bmp_header(-64, 64),
        webp_vp8_zero,
        jpeg_dnl_zero_h,
        jpeg_short_sof,
        be_tiff_zero,
    ):
        assert probe_image_dimensions(payload) is None
        with pytest.raises(ImageHeaderError):
            check_image_pixel_budget(payload)

    # 유효한 top-down BMP(음수 height) 및 빅엔디안 TIFF는 정상 파싱된다.
    assert probe_image_dimensions(_bmp_header(64, -32)) == (64, 32)
    assert probe_image_dimensions(be_tiff_valid) == (320, 240)

    # 파일 끝 IFD 폴백 경로의 0-dim BE-TIFF도 None 반환(fail-closed).
    tail_ifd_offset = 262_144 + 64
    tail_ifd = (
        struct.pack(">H", 2)
        + struct.pack(">HHII", 256, 4, 1, 0)
        + struct.pack(">HHII", 257, 4, 1, 128)
        + b"\x00" * 4
    )
    tail_tiff = tmp_path / "tail_zero_be.tif"
    tail_tiff.write_bytes(
        b"MM\x00*"
        + struct.pack(">I", tail_ifd_offset)
        + b"\x00" * (tail_ifd_offset - 8)
        + tail_ifd
    )
    assert probe_image_file_dimensions(tail_tiff) is None
    with pytest.raises(ImageHeaderError):
        check_image_pixel_budget(tail_tiff)

    with pytest.raises(ImageHeaderError):
        check_decoded_pixel_budget(0, 100)
    with pytest.raises(ImageHeaderError):
        check_decoded_pixel_budget(100, -1)


def test_validate_same_origin_blocks_malformed_empty_userinfo_and_bad_scheme() -> None:
    # 정상 IPv6 로컬 호스트 허용
    validate_same_origin({"Origin": "http://[::1]:8000"}, "[::1]:8000")
    validate_same_origin({}, "[::1]:8000")

    bad_cases: list[tuple[dict[str, str], str]] = [
        ({}, "[invalid"),
        ({}, ""),
        ({}, "   "),
        ({}, "evil.com@127.0.0.1:8000"),
        ({}, "127.0.0.1/evil"),
        ({}, "127.0.0.1\\evil"),
        ({}, "127.0.0.1 8000"),
        ({}, "127.0.0.1:99999"),
        ({"Origin": "javascript://127.0.0.1:8000/%0a"}, "127.0.0.1:8000"),
        ({"Referer": "javascript://127.0.0.1:8000/form"}, "127.0.0.1:8000"),
        ({"Origin": "http://evil.com@127.0.0.1:8000"}, "127.0.0.1:8000"),
        ({"Referer": "http://evil.com@127.0.0.1:8000/form"}, "127.0.0.1:8000"),
        ({"Origin": "http://[invalid"}, "127.0.0.1:8000"),
        ({"Origin": "http://127.0.0.1:99999"}, "127.0.0.1:8000"),
    ]
    for headers, host in bad_cases:
        with pytest.raises(CsrfOriginError):
            validate_same_origin(headers, host)


def test_read_upload_bounded_rejects_bool_closes_on_invalid_args_and_sync_close() -> None:
    import asyncio

    from surface_sentinel.io.limits import read_upload_bounded

    class _SyncCloseUpload:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = list(chunks)
            self.closed = False

        async def read(self, size: int) -> bytes:
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

        def close(self) -> None:
            self.closed = True

    async def _exercise() -> None:
        u_bool_max = _SyncCloseUpload([b"a"])
        with pytest.raises(ValueError):
            await read_upload_bounded(u_bool_max, max_bytes=True, chunk_bytes=8)
        assert u_bool_max.closed is True

        u_bool_chunk = _SyncCloseUpload([b"a"])
        with pytest.raises(ValueError):
            await read_upload_bounded(u_bool_chunk, max_bytes=8, chunk_bytes=True)
        assert u_bool_chunk.closed is True

        u_neg = _SyncCloseUpload([b"a"])
        with pytest.raises(ValueError):
            await read_upload_bounded(u_neg, max_bytes=-1, chunk_bytes=8)
        assert u_neg.closed is True

        u_sync_ok = _SyncCloseUpload([b"ok"])
        assert await read_upload_bounded(u_sync_ok, max_bytes=4, chunk_bytes=2) == b"ok"
        assert u_sync_ok.closed is True

    asyncio.run(_exercise())


def test_repo_has_zero_pickle_load_calls() -> None:
    import ast

    offenders: list[str] = []
    for root_name in ("src", "scripts"):
        root_dir = REPO_ROOT / root_name
        for py_file in sorted(root_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in {"pickle", "_pickle"}:
                            offenders.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in {"pickle", "_pickle"}:
                        offenders.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"load", "loads"}
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"pickle", "_pickle"}
                ):
                    offenders.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"pickle 사용이 남아 있습니다: {offenders}"


def test_batch_and_cache_reject_manifest_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surface_sentinel.batch.runner import BatchInputError, _resolve_sample_paths
    from surface_sentinel.evaluation.cache import _read_mask, inspect_once
    from surface_sentinel.io.manifest import ManifestRow

    raw_root = tmp_path / "raw"
    (raw_root / "train").mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_valid_png_bytes(64))

    def _make_row(image_path: str, mask_path: str = "") -> ManifestRow:
        return ManifestRow(
            sample_id="s1",
            original_id="s1",
            group_id="g1",
            image_path=image_path,
            mask_path=mask_path,
            label="defect" if mask_path else "good",
            source_split="train",
            width=64,
            height=64,
            channels=3,
            format="PNG",
            image_sha256="0" * 64,
            mask_sha256="0" * 64,
            defect_pixels=10 if mask_path else 0,
        )

    # 절대 경로 표기는 플랫폼별로 다르다: POSIX에서 백슬래시 경로는 상대 파일명이라 탈출이 아니다.
    bad_paths = ["../outside.png", str(outside)]
    if os.name == "nt":
        bad_paths += ["/etc/passwd", "\\windows\\system32\\cmd.exe"]
    else:
        bad_paths += ["/etc/passwd"]
    for bad in bad_paths:
        with pytest.raises(BatchInputError):
            _resolve_sample_paths(raw_root, _make_row(bad))
        with pytest.raises(BatchInputError):
            _resolve_sample_paths(raw_root, _make_row("train/ok.png", bad))
        with pytest.raises(ValueError):
            inspect_once(
                _make_row(bad),
                BASE_CONFIG,
                raw_root=raw_root,
                split="tune",
                overlay_dir=None,
            )
        with pytest.raises(ValueError):
            inspect_once(
                _make_row("train/ok.png", bad),
                BASE_CONFIG,
                raw_root=raw_root,
                split="tune",
                overlay_dir=None,
            )

    # _read_mask 도 디코드 전 픽셀 상한 가드를 적용한다.
    bomb_mask = raw_root / "train" / "bomb_mask.png"
    bomb_mask.write_bytes(_png_header(20_000, 20_000))

    def _forbid_decode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("마스크 디코드 전 가드가 먼저 차단해야 합니다")

    monkeypatch.setattr(cv2, "imdecode", _forbid_decode)
    assert _read_mask(bomb_mask) is None


def test_yolo_decode_rejects_small_n_transposed_exports() -> None:
    """소형 N 전치 export([1, N, 4+nc])도 거부한다(rows>128 우회 차단)."""

    with pytest.raises(ValueError, match="전치"):
        decode_det(np.zeros((1, 40, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="전치"):
        decode_det(np.zeros((1, 64, 5), dtype=np.float32), num_classes=1)
    with pytest.raises(ValueError, match="전치"):
        decode_seg(
            np.zeros((1, 40, 13), dtype=np.float32),
            np.zeros((1, 8, 16, 16), dtype=np.float32),
            orig_w=64,
            orig_h=64,
        )
    # 정상 소형 합성 shape 는 그대로 통과한다.
    boxes, scores = decode_det(np.zeros((1, 5, 3), dtype=np.float32))
    assert boxes.shape == (0, 4) and scores.shape == (0,)


def test_unicode_path_read_and_atomic_overlay_write_windows_safe(tmp_path: Path) -> None:
    """한글 경로에서도 읽기/원자 PNG 쓰기가 동작하고 임시 파일이 남지 않는다."""

    from surface_sentinel.io.imageio import imread_unicode, imwrite_png_atomic
    from surface_sentinel.yolo.overlay import save_overlay

    korean_dir = tmp_path / "한글 경로" / "결함샘플"
    korean_dir.mkdir(parents=True)
    src = korean_dir / "입력_10021.png"
    bgr = np.zeros((16, 24, 3), np.uint8)
    bgr[4:10, 6:14] = (0, 0, 255)

    assert imwrite_png_atomic(src, bgr) is True
    loaded = imread_unicode(src)
    assert loaded is not None and np.array_equal(loaded, bgr)

    # 없는 경로·깨진 파일은 None(cv2.imread 계약 유지).
    assert imread_unicode(korean_dir / "없음.png") is None
    broken = korean_dir / "깨진.png"
    broken.write_bytes(b"not-an-image")
    assert imread_unicode(broken) is None

    # 오버레이 저장도 한글 경로에서 동작하고 .tmp 잔여물이 없다.
    target = korean_dir / "오버레이_yolo.png"
    saved = save_overlay(bgr, [], target)
    assert saved.is_file()
    assert not list(korean_dir.glob(".tmp-*"))

    # 대상이 디렉터리면 실패(False)하고 임시 파일을 정리한다.
    dir_target = korean_dir / "디렉터리대상"
    dir_target.mkdir()
    assert imwrite_png_atomic(dir_target, bgr) is False
    assert not list(korean_dir.glob(".tmp-*"))
