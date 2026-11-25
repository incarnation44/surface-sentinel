"""검사 API 입력 경계·업로드 용량 제한·중복제거 응답 일관성 단위 테스트.

* ``label``/``split`` 허용 집합 밖 값은 파일 읽기·검사·DB 기록 전에 422.
* 업로드는 ``MAX_UPLOAD_BYTES`` 까지만 청크로 읽고 초과분은 413.
* 중복제거 응답의 저장 필드는 DB의 canonical 행에서 온다.
* 봉인 split 의 기존 ERROR 동작(파이프라인 소유)은 그대로다.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from surface_sentinel.api import app as app_module
from surface_sentinel.api import create_inspection_app
from surface_sentinel.pipeline.stages import VALID_LABELS, VALID_SPLITS
from surface_sentinel.records.db import connect, init_db


def synthetic_png(seed: int = 0, size: int = 96) -> bytes:
    """검사 입력 검증(min_std)을 통과하는 결정론적 합성 PNG."""

    image = np.tile(np.linspace(40, 200, size, dtype=np.uint8), (size, 1))
    image = np.roll(image, seed * 7, axis=1)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


@pytest.fixture(autouse=True)
def _stable_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """실제 저장소 상태(dirty)에 의존하지 않도록 코드 신원을 고정한다."""

    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("test-sha", False))


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "runs"
    path.mkdir()
    return path


@pytest.fixture()
def client(runs_dir: Path) -> TestClient:
    return TestClient(create_inspection_app(runs_dir=runs_dir))


def upload(
    client: TestClient,
    payload: bytes,
    *,
    name: str = "sample.png",
    label: str = "unknown",
    split: str = "tune",
) -> object:
    return client.post(
        "/api/inspect",
        files={"file": (name, payload, "image/png")},
        data={"label": label, "split": split},
    )


def _count_rows(runs_dir: Path, table: str) -> int:
    """테이블 행 수(경계 거부 시 기록이 생기지 않았는지 확인용)."""

    assert table in {"inspections", "engine_observations"}
    connection = connect(runs_dir / "records.db")
    try:
        init_db(connection)
        row = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    finally:
        connection.close()
    return int(row["n"])


# --- 문제 1: label/split 경계 검증 -------------------------------------------


def test_invalid_label_rejected_before_recording(client: TestClient, runs_dir: Path) -> None:
    response = upload(client, synthetic_png(), label="bad")

    assert response.status_code == 422
    assert "label" in response.json()["detail"]
    # 파일 읽기·검사 실행·DB 기록 전에 끝났다.
    assert _count_rows(runs_dir, "inspections") == 0
    assert _count_rows(runs_dir, "engine_observations") == 0
    assert not (runs_dir / app_module.UPLOAD_DIRNAME).exists()


def test_invalid_split_rejected_before_recording(client: TestClient, runs_dir: Path) -> None:
    response = upload(client, synthetic_png(), split="bad")

    assert response.status_code == 422
    assert "split" in response.json()["detail"]
    assert _count_rows(runs_dir, "inspections") == 0
    assert not (runs_dir / app_module.UPLOAD_DIRNAME).exists()


@pytest.mark.parametrize("label", VALID_LABELS)
def test_valid_labels_still_accepted(client: TestClient, label: str) -> None:
    response = upload(client, synthetic_png(seed=1), label=label)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] != "ERROR"
    assert payload["error_code"] == ""
    assert payload["error_stage"] == ""


@pytest.mark.parametrize("split", ("test", "official_test"))
def test_sealed_split_keeps_pipeline_error(
    client: TestClient, runs_dir: Path, split: str
) -> None:
    """봉인 split 은 허용 값이지만 파이프라인이 ERROR 로 거부한다(우회하지 않는다)."""

    assert split in VALID_SPLITS
    response = upload(client, synthetic_png(seed=2), split=split)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ERROR"
    assert payload["error_code"]  # 진단 가능성 확보
    assert payload["error_stage"]


# --- 문제 2: bounded upload --------------------------------------------------


class _CountingUpload(UploadFile):
    """읽은 바이트 수를 기록하는 ``UploadFile``(경계 읽기 검증용)."""

    def __init__(self, payload: bytes, log: list[int]) -> None:
        super().__init__(file=io.BytesIO(payload), filename="big.png")
        self._log = log

    async def read(self, size: int = -1) -> bytes:
        data = await super().read(size)
        self._log.append(len(data))
        return data


def test_bounded_read_stops_at_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(app_module, "UPLOAD_CHUNK_BYTES", 256)
    log: list[int] = []
    upload_file = _CountingUpload(b"x" * 100_000, log)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(app_module._read_upload_bounded(upload_file))

    assert excinfo.value.status_code == 413
    # 전체 100 KiB 를 읽지 않았다: 총 읽기 <= 상한 + 청크 1개.
    assert 1000 < sum(log) <= 1000 + 256


def test_upload_over_limit_returns_413(
    client: TestClient, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)

    response = upload(client, b"x" * 4096)

    assert response.status_code == 413
    assert "업로드" in response.json()["detail"]
    assert _count_rows(runs_dir, "inspections") == 0


def test_upload_at_limit_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = synthetic_png(seed=4)
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", len(payload))

    response = upload(client, payload)

    assert response.status_code == 200
    assert response.json()["deduplicated"] is False


# --- 문제 3: dedup canonical 응답 --------------------------------------------


def test_dedup_response_uses_stored_canonical_row(
    client: TestClient, runs_dir: Path
) -> None:
    """dedup 응답이 새 계산값이 아니라 DB 저장 행을 쓰는지 본다.

    검사 행은 append-only(UPDATE 불가)이므로, 업로드 전에 같은
    (image_sha256, fingerprint) 키로 **다른 값의 행을 먼저 INSERT** 해
    canonical 행이 새 계산값과 갈라지게 만든다.
    """

    from surface_sentinel.config import default_config
    from surface_sentinel.records.identity import opencv_spec

    payload = synthetic_png(seed=5)
    sha = hashlib.sha256(payload).hexdigest()
    config_hash = default_config().config_hash
    # 앱은 픽스처로 고정한 코드 신원("test-sha", False)과 기본 설정을 쓴다.
    spec = opencv_spec(config_hash, code_sha="test-sha", code_dirty=False)

    connection = connect(runs_dir / "records.db")
    try:
        with connection:
            connection.execute(
                "INSERT INTO inspections ("
                "run_id, sample_id, image_sha256, inspection_fingerprint, label, split,"
                " status, score, reasons, engine, task, config_hash, model_sha256,"
                " model_version, provenance_complete, overlay_path, timing_ms, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "canon-run",
                    "canon-sample",
                    sha,
                    spec.fingerprint,
                    "unknown",
                    "tune",
                    "OK",
                    9.99,
                    json.dumps(["NO_DEFECT_EVIDENCE"]),
                    "opencv",
                    "",
                    config_hash,
                    "",
                    "opencv-test",
                    1,
                    "",
                    "{}",
                    "2026-01-01T00:00:00Z",
                ),
            )
    finally:
        connection.close()

    second = upload(client, payload).json()

    assert second["deduplicated"] is True
    assert second["run_id"] == "canon-run"
    assert second["sample_id"] == "canon-sample"
    assert second["status"] == "OK"
    assert second["score"] == pytest.approx(9.99)
    assert second["reasons"] == ["NO_DEFECT_EVIDENCE"]
    # 단일 엔진(opencv) 검사는 관찰 미기록 → components 는 이번 결과에서 온다.
    assert isinstance(second["components"], list)


def test_dedup_error_keeps_diagnostic_fields(client: TestClient) -> None:
    """단일 엔진 ERROR 행 dedup 시 error_code/error_stage 가 응답에 유지된다.

    봉인 split 거부는 파일을 읽기 전에 일어나는 ERROR 라 관찰 행이 없다.
    같은 fingerprint 면 파이프라인이 결정론적이라 이번 결과의 진단 값이
    원래 실행과 같다 — 빈 문자열로 지우지 않는다.
    """

    payload = synthetic_png(seed=7)
    first = upload(client, payload, split="test").json()
    assert first["status"] == "ERROR"
    assert first["error_code"]
    assert first["error_stage"]

    second = upload(client, payload, split="test").json()

    assert second["deduplicated"] is True
    assert second["inspection_id"] == first["inspection_id"]
    assert second["status"] == "ERROR"
    assert second["error_code"] == first["error_code"]
    assert second["error_stage"] == first["error_stage"]


def test_dedup_does_not_change_machine_status(client: TestClient) -> None:
    payload = synthetic_png(seed=6)
    first = upload(client, payload).json()
    second = upload(client, payload).json()

    assert second["status"] == first["status"]
    detail = client.get(f"/api/inspections/{first['inspection_id']}").json()
    assert detail["status"] == first["status"]


@pytest.mark.parametrize("engine", ("yolo-det", "yolo-seg", "combined"))
@pytest.mark.parametrize("split", ("test", "official_test"))
def test_api_yolo_sealed_split_denied_without_inference(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    split: str,
) -> None:
    """API /api/inspect 도 YOLO/combined 봉인 split 요청을 추론 전에 E_SEALED_DENIED 로 거부한다."""

    from surface_sentinel.yolo import inspect as yolo_inspect

    calls: list[object] = []

    class _FailEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            raise AssertionError("봉인 분할에서 YOLO 엔진이 생성되면 안 됩니다")

    monkeypatch.setattr(yolo_inspect, "YoloOnnxEngine", _FailEngine)
    response = client.post(
        "/api/inspect",
        params={"engine": engine},
        files={"file": ("sealed_probe.png", synthetic_png(seed=9), "image/png")},
        data={"label": "unknown", "split": split},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "E_SEALED_DENIED"
    assert payload["error_stage"] == "validate_input"
    assert calls == []


def test_dedup_canonical_non_error_clears_current_request_error_fields(
    client: TestClient, runs_dir: Path
) -> None:
    """단일 엔진 dedup 시 canonical 행이 OK 면 현재 요청의 ERROR 진단 필드가 혼입되지 않는다."""

    payload = synthetic_png(seed=11)
    first = upload(client, payload, split="tune").json()
    assert first["status"] != "ERROR"

    # 같은 (image_sha256, fingerprint)로 split="test" 요청 → 파이프라인은 E_SEALED_DENIED
    # ERROR 를 내지만, DB canonical 행은 비-ERROR 이므로 error_code/error_stage 를 비워야 한다.
    second = upload(client, payload, split="test").json()
    assert second["deduplicated"] is True
    assert second["inspection_id"] == first["inspection_id"]
    assert second["status"] == first["status"]
    assert second["error_code"] == ""
    assert second["error_stage"] == ""


def test_store_upload_writes_atomically_without_temp_leftovers(tmp_path: Path) -> None:
    payload = synthetic_png(seed=12)
    sha = hashlib.sha256(payload).hexdigest()
    upload_dir = tmp_path / "uploads"

    saved = app_module._store_upload(upload_dir, payload, sha, ".png")
    assert saved.is_file()
    assert saved.read_bytes() == payload
    # 임시 파일(.tmp-*) 잔여물 0건 확인
    assert [p.name for p in upload_dir.iterdir()] == [saved.name]


def test_inspection_spec_fallback_is_limited_to_error_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """모델 부재 시 일반 경로는 예외를 전파하고, ERROR 조기 거부 경로만 최소 spec 으로 폴백한다."""

    from surface_sentinel.config import default_config
    from surface_sentinel.yolo import inspect as yolo_inspect

    monkeypatch.setenv(yolo_inspect.ONNX_ROOT_ENV, str(tmp_path / "empty-models"))
    cfg = default_config()

    with pytest.raises(FileNotFoundError):
        app_module._inspection_spec(
            "yolo-det", cfg, code_sha="abc", code_dirty=False, allow_minimal_fallback=False
        )

    minimal = app_module._inspection_spec(
        "yolo-det", cfg, code_sha="abc", code_dirty=False, allow_minimal_fallback=True
    )
    assert minimal.engine == "yolo-det"
    assert minimal.model_sha256 == ""
    assert minimal.provenance_complete is False


def test_api_db_error_returns_503_json_and_handles_corrupt_json_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    import types

    from starlette.datastructures import UploadFile

    from surface_sentinel.config import default_config
    from surface_sentinel.records.identity import opencv_spec

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    app = create_inspection_app(runs_dir=runs_dir)
    client = TestClient(app, raise_server_exceptions=False)

    # 1) DB 오류는 내부 경로 없이 503 JSON으로 변환된다.
    orig_connect = app_module.connect

    def _boom_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open database: /secret/host/records.db")

    monkeypatch.setattr(app_module, "connect", _boom_connect)
    r_err = client.get("/api/inspections")
    assert r_err.status_code == 503
    assert r_err.json() == {"detail": "검사 기록 데이터베이스를 처리할 수 없습니다."}
    assert "/secret/host" not in r_err.text
    monkeypatch.setattr(app_module, "connect", orig_connect)

    # 2) 모델 FileNotFoundError 503 응답도 내부 경로를 노출하지 않는다.
    fake_yolo = types.ModuleType("surface_sentinel.yolo.inspect")

    def _missing(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("/secret/internal/best.onnx")

    fake_yolo.inspect_with_yolo = _missing  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "surface_sentinel.yolo.inspect", fake_yolo)
    r_model = client.post(
        "/api/inspect",
        params={"engine": "yolo-det"},
        files={"file": ("probe.png", synthetic_png(seed=31), "image/png")},
        data={"label": "unknown", "split": "tune"},
    )
    assert r_model.status_code == 503
    assert r_model.json()["detail"] == "yolo-det 엔진 모델 파일이 없습니다."
    assert "/secret/internal" not in r_model.text

    # 3) DB JSON 컬럼이 손상돼도 /api/inspections 및 상세 조회가 500 대신 안전하게 200을 반환한다.
    cfg_hash = default_config().config_hash
    spec = opencv_spec(cfg_hash, code_sha="test-sha", code_dirty=False)
    conn = connect(runs_dir / "records.db")
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO inspections ("
                "run_id, sample_id, image_sha256, inspection_fingerprint, label, split,"
                " status, score, reasons, engine, task, config_hash, model_sha256,"
                " model_version, provenance_complete, overlay_path, timing_ms, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "corrupt-run",
                    "corrupt-1",
                    "c" * 64,
                    spec.fingerprint,
                    "unknown",
                    "tune",
                    "OK",
                    0.1,
                    "{not-valid-json",
                    "opencv",
                    "",
                    cfg_hash,
                    "",
                    "opencv-test",
                    1,
                    str(runs_dir / "records.db"),
                    "{broken-timing",
                    "2026-01-01T00:00:00Z",
                ),
            )
            insp_id = int(cur.lastrowid or 0)
            conn.execute(
                "INSERT INTO engine_observations ("
                "inspection_id, engine, task, status, score, reasons, components,"
                " model_sha256, overlay_path, error_code, timing_ms, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    insp_id,
                    "opencv",
                    "",
                    "OK",
                    0.1,
                    "bad-json",
                    "bad-json",
                    "",
                    "",
                    "",
                    "bad-json",
                    "2026-01-01T00:00:00Z",
                ),
            )
            conn.execute(
                "INSERT INTO agent_assessments ("
                "inspection_id, assessment_hash, policy_hash, flags_json,"
                " review_priority, stability, recommended_action, diagnostics_json,"
                " trace_path, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    insp_id,
                    "d" * 64,
                    "e" * 64,
                    "bad-flags",
                    None,
                    None,
                    "NONE",
                    "bad-diag",
                    "",
                    "2026-01-01T00:00:00Z",
                ),
            )
    finally:
        conn.close()

    r_list = client.get("/api/inspections")
    assert r_list.status_code == 200
    assert r_list.json()["items"][0]["reasons"] == []
    assert r_list.json()["items"][0]["overlay_url"] is None

    r_detail = client.get(f"/api/inspections/{insp_id}")
    assert r_detail.status_code == 200
    detail = r_detail.json()
    assert detail["timing_ms"] == {}
    assert detail["observations"][0]["components"] == []
    assert detail["assessments"][0]["flags"] == []
    assert detail["assessments"][0]["diagnostics"] == {}

    # 4) _row_overlay_path는 비-PNG와 NUL 바이트를 거부한다.
    db_overlay = {"overlay_path": str(runs_dir / "records.db")}
    assert app_module._row_overlay_path(db_overlay, runs_dir) is None
    assert app_module._row_overlay_path({"overlay_path": "bad\x00.png"}, runs_dir) is None

    # 5) /api/inspect 조기 403/422 거부 경로에서도 UploadFile.close()가 호출된다.
    close_calls = 0
    orig_close = UploadFile.close

    async def tracked_close(self: UploadFile) -> None:
        nonlocal close_calls
        close_calls += 1
        await orig_close(self)

    monkeypatch.setattr(UploadFile, "close", tracked_close)
    r_422 = client.post(
        "/api/inspect",
        files={"file": ("ok.png", synthetic_png(seed=32), "image/png")},
        data={"label": "INVALID_LABEL", "split": "tune"},
    )
    assert r_422.status_code == 422
    assert close_calls >= 1
