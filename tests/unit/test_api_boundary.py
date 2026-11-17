"""API 입력 경계 검증·bounded upload·dedup canonical 응답 테스트 (감사 5.5/5.9/5.4).

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