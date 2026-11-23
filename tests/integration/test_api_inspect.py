"""E4 검사 API 통합 테스트 (TestClient).

합성 PNG만 업로드하고 ``tmp_path`` 밖은 건드리지 않는다. opencv 경로는 실제
``inspect_image``로, YOLO 경로는 ``sys.modules`` stub으로 검증한다(모델 파일
불필요). 기계 판정 불변·Status 직렬화·페이지네이션·분모 표시를 고정한다.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from surface_sentinel.api import create_inspection_app
from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.pipeline import InspectionOutcome
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import insert_inspection

VALID_STATUSES = {status.value for status in Status}


def synthetic_png(seed: int = 0, size: int = 96) -> bytes:
    """검사 입력 검증(min_std)을 통과하는 결정론적 합성 PNG."""

    image = np.tile(np.linspace(40, 200, size, dtype=np.uint8), (size, 1))
    image = np.roll(image, seed * 7, axis=1)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


@pytest.fixture(autouse=True)
def _stable_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """테스트가 실제 git 상태에 의존하지 않게 코드 신원을 빈 값으로 고정한다.

    실제 주입 경로(진입점 1회 계산)는 전용 테스트가 스텁으로 검증한다.
    """

    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("", False))


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
    engine: str | None = None,
    label: str = "unknown",
    split: str = "tune",
) -> object:
    params = {} if engine is None else {"engine": engine}
    return client.post(
        "/api/inspect",
        files={"file": (name, payload, "image/png")},
        data={"label": label, "split": split},
        params=params,
    )


def test_code_identity_computed_once_at_creation(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """code_sha 는 앱 생성 시 1회 계산 — 요청마다 재계산하지 않는다(P0-2 주입)."""

    from surface_sentinel.evaluation import recalibration

    calls: list[int] = []

    def fake_identity() -> tuple[str, bool]:
        calls.append(1)
        return ("d" * 40, False)

    monkeypatch.setattr(recalibration, "git_code_identity", fake_identity)
    client = TestClient(create_inspection_app(runs_dir=runs_dir))
    assert len(calls) == 1

    first = upload(client, synthetic_png(1), name="a.png")
    second = upload(client, synthetic_png(2), name="b.png")
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == 1  # 요청 2건이어도 생성 시 1회
    body = first.json()
    assert body["code_sha"] == "d" * 40
    assert body["code_dirty"] is False


def seed_row(
    runs_dir: Path,
    *,
    status: Status,
    sha_text: str,
    engine: str = "opencv",
    label: str = "good",
    score: float = 0.0,
) -> int:
    """DB에 기록 한 건을 직접 넣는다(분포·리뷰 검증용)."""

    sha = hashlib.sha256(sha_text.encode("utf-8")).hexdigest()
    connection = connect(runs_dir / "records.db")
    try:
        init_db(connection)
        result = SampleResult(
            run_id="seed-run",
            sample_id=f"seed_{sha[:8]}",
            original_id="seed",
            split="tune",
            label=label,
            status=status,
            score=score,
            reasons=(ReasonCode.NO_DEFECT_EVIDENCE,),
            config_hash="0" * 64,
        )
        return insert_inspection(connection, result, engine=engine, image_sha256=sha)
    finally:
        connection.close()


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "surface-sentinel-api"
    assert payload["inspections"] == 0
    assert set(payload["engines"]) == {"opencv", "yolo-det", "yolo-seg", "combined"}


def test_inspect_opencv_contract_json(client: TestClient, runs_dir: Path) -> None:
    response = upload(client, synthetic_png(), label="defect")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] in VALID_STATUSES
    # Status는 contracts와 같은 문자열로 직렬화된다(Enum repr 금지).
    assert payload["status"] == Status(payload["status"]).value
    for reason in payload["reasons"]:
        assert reason in {code.value for code in ReasonCode}
    assert isinstance(payload["score"], float)
    assert isinstance(payload["components"], list)
    assert payload["engine"] == "opencv"
    assert payload["deduplicated"] is False
    assert payload["inspection_id"] >= 1
    assert payload["overlay_url"] == f"/api/inspections/{payload['inspection_id']}/overlay.png"

    overlay = client.get(payload["overlay_url"])
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"
    assert overlay.content.startswith(b"\x89PNG")

    # 업로드 원본과 오버레이는 runs_dir 아래에만 만든다.
    stored = list(runs_dir.rglob("*.png"))
    assert stored
    for path in stored:
        assert path.resolve().is_relative_to(runs_dir.resolve())
    assert (runs_dir / "api_uploads").is_dir()


def test_inspect_unknown_engine_rejected(client: TestClient) -> None:
    response = upload(client, synthetic_png(), engine="magic")
    assert response.status_code == 422


def test_inspect_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/api/inspect",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "이미지" in response.json()["detail"]


def test_inspect_rejects_empty_file(client: TestClient) -> None:
    response = upload(client, b"")
    assert response.status_code == 400


def test_inspect_deduplicates_same_bytes(client: TestClient) -> None:
    payload = synthetic_png(seed=3)
    first = upload(client, payload).json()
    second = upload(client, payload).json()
    assert second["inspection_id"] == first["inspection_id"]
    assert second["deduplicated"] is True
    listing = client.get("/api/inspections").json()
    assert listing["total"] == 1
    assert len(listing["items"]) == 1


def test_inspections_list_and_detail(client: TestClient) -> None:
    created = upload(client, synthetic_png(seed=4), label="defect").json()

    listing = client.get("/api/inspections").json()
    assert listing["total"] == 1
    assert listing["limit"] == 50
    assert listing["offset"] == 0
    item = listing["items"][0]
    assert item["inspection_id"] == created["inspection_id"]
    assert item["engine"] == "opencv"

    detail = client.get(f"/api/inspections/{created['inspection_id']}").json()
    assert detail["status"] == created["status"]
    assert detail["label"] == "defect"
    assert detail["reviews"] == []
    assert isinstance(detail["timing_ms"], dict)
    assert detail["config_hash"]


def test_inspections_unknown_id_returns_404(client: TestClient) -> None:
    assert client.get("/api/inspections/999").status_code == 404
    assert client.post("/api/inspections/999/review", json={"decision": "NG"}).status_code == 404


def test_inspections_pagination_and_status_filter(runs_dir: Path, client: TestClient) -> None:
    seed_row(runs_dir, status=Status.OK, sha_text="page-1")
    seed_row(runs_dir, status=Status.NG, sha_text="page-2")
    seed_row(runs_dir, status=Status.REVIEW, sha_text="page-3", score=0.3)

    first = client.get("/api/inspections", params={"limit": 2, "offset": 0}).json()
    assert first["total"] == 3
    assert len(first["items"]) == 2
    assert [item["inspection_id"] for item in first["items"]] == [1, 2]

    second = client.get("/api/inspections", params={"limit": 2, "offset": 2}).json()
    assert [item["inspection_id"] for item in second["items"]] == [3]

    filtered = client.get("/api/inspections", params={"status": "NG"}).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["status"] == "NG"


def test_review_append_does_not_change_machine_status(
    runs_dir: Path, client: TestClient
) -> None:
    inspection_id = seed_row(
        runs_dir,
        status=Status.REVIEW,
        sha_text="review-1",
        score=0.3,
    )
    response = client.post(
        f"/api/inspections/{inspection_id}/review",
        json={"decision": "NG", "reviewer": "tester", "reason": "스크래치 확인"},
    )
    assert response.status_code == 200
    appended = response.json()
    assert appended["decision"] == "NG"
    assert appended["machine_status"] == "REVIEW"

    detail = client.get(f"/api/inspections/{inspection_id}").json()
    assert detail["status"] == "REVIEW"
    assert len(detail["reviews"]) == 1
    assert detail["reviews"][0]["decision"] == "NG"
    assert detail["reviews"][0]["reviewer"] == "tester"

    # DB의 기계 판정도 그대로다.
    connection = connect(runs_dir / "records.db")
    try:
        row = connection.execute(
            "SELECT status FROM inspections WHERE id = ?",
            (inspection_id,),
        ).fetchone()
    finally:
        connection.close()
    assert row["status"] == "REVIEW"


def test_review_rejects_ok_ng_machine_status(runs_dir: Path, client: TestClient) -> None:
    inspection_id = seed_row(runs_dir, status=Status.OK, sha_text="review-ok")
    response = client.post(
        f"/api/inspections/{inspection_id}/review",
        json={"decision": "NG"},
    )
    assert response.status_code == 400
    assert "불변" in response.json()["detail"]
    assert client.get(f"/api/inspections/{inspection_id}").json()["reviews"] == []


def test_review_rejects_invalid_decision(runs_dir: Path, client: TestClient) -> None:
    inspection_id = seed_row(runs_dir, status=Status.REVIEW, sha_text="review-bad")
    response = client.post(
        f"/api/inspections/{inspection_id}/review",
        json={"decision": "REVIEW"},
    )
    assert response.status_code == 422


def test_stats_reports_denominators(runs_dir: Path, client: TestClient) -> None:
    seed_row(runs_dir, status=Status.OK, sha_text="stats-1")
    seed_row(runs_dir, status=Status.NG, sha_text="stats-2", engine="yolo-det")
    seed_row(runs_dir, status=Status.REVIEW, sha_text="stats-3", score=0.3)

    payload = client.get("/api/stats").json()
    assert payload["total"] == 3
    assert payload["by_status"] == {"OK": 1, "NG": 1, "REVIEW": 1, "ERROR": 0}
    assert payload["ratios"]["ok"]["denominator"] == 3
    assert payload["ratios"]["ok"]["value"] == pytest.approx(1 / 3)
    assert payload["ratios"]["ok"]["definition"]
    assert payload["reviews"] == {"decisions": 0, "reviewed_inspections": 0}
    assert payload["by_engine"]["opencv"]["OK"] == 1
    assert payload["by_engine"]["yolo-det"]["NG"] == 1
    assert payload["by_engine"]["yolo-det"]["OK"] == 0


def test_stats_empty_db_has_zero_denominators(client: TestClient) -> None:
    payload = client.get("/api/stats").json()
    assert payload["total"] == 0
    assert payload["ratios"]["ok"]["value"] is None
    assert payload["ratios"]["ok"]["denominator"] == 0
    assert payload["by_engine"] == {}


def test_yolo_engine_missing_module_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 실제 모듈이 있어도 import 실패를 강제해 503 계약을 검증한다.
    monkeypatch.setitem(sys.modules, "surface_sentinel.yolo.inspect", None)
    response = upload(client, synthetic_png(seed=6), engine="yolo-det")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "yolo-det" in detail
    assert "ONNX" in detail


def test_yolo_engine_missing_model_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("surface_sentinel.yolo.inspect")

    def missing_model(*args: object, **kwargs: object) -> InspectionOutcome:
        raise FileNotFoundError("weights.onnx")

    fake.inspect_with_yolo = missing_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "surface_sentinel.yolo.inspect", fake)

    response = upload(client, synthetic_png(seed=7), engine="combined")
    assert response.status_code == 503
    assert "모델" in response.json()["detail"]


def test_yolo_engine_uses_stub_entry_point(
    runs_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    fake = types.ModuleType("surface_sentinel.yolo.inspect")

    def stub(
        image_path: str | Path,
        config: object,
        *,
        engine: str,
        output_dir: str | Path | None = None,
        split: str = "tune",
        label: str = "unknown",
        **kwargs: object,
    ) -> InspectionOutcome:
        calls.append(
            {
                "engine": engine,
                "output_dir": Path(output_dir) if output_dir is not None else None,
                "split": split,
                "label": label,
            }
        )
        result = SampleResult(
            run_id="yolo-run",
            sample_id=Path(image_path).stem,
            original_id="yolo",
            split=split,
            label=label,
            status=Status.NG,
            score=1.2,
            reasons=(ReasonCode.DEFECT_EVIDENCE_STRONG,),
            config_hash="1" * 64,
        )
        return InspectionOutcome(result=result)

    fake.inspect_with_yolo = stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "surface_sentinel.yolo.inspect", fake)

    response = upload(client, synthetic_png(seed=8), engine="yolo-seg", label="defect")
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "yolo-seg"
    assert payload["status"] == "NG"
    assert payload["reasons"] == ["DEFECT_EVIDENCE_STRONG"]

    assert len(calls) == 1
    assert calls[0]["engine"] == "yolo-seg"
    assert calls[0]["split"] == "tune"
    assert calls[0]["label"] == "defect"
    assert isinstance(calls[0]["output_dir"], Path)
    assert calls[0]["output_dir"].is_relative_to(runs_dir)

    detail = client.get(f"/api/inspections/{payload['inspection_id']}").json()
    assert detail["engine"] == "yolo-seg"
    assert detail["model_version"] == "yolo-seg"
    assert json.loads(json.dumps(detail["status"])) == "NG"


def test_concurrent_same_image_inspect_avoids_output_dir_race(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import concurrent.futures
    import threading
    import time

    from surface_sentinel.api import app as api_app_module

    orig_inspect = api_app_module._inspect
    active_lock = threading.Lock()
    active_dirs: set[Path] = set()
    overlap_detected = False

    def tracked_inspect(*args: object, **kwargs: object) -> InspectionOutcome:
        nonlocal overlap_detected
        out_dir = Path(kwargs["output_dir"])  # type: ignore[arg-type]
        with active_lock:
            if out_dir in active_dirs:
                overlap_detected = True
            active_dirs.add(out_dir)
        try:
            time.sleep(0.08)
            return orig_inspect(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            with active_lock:
                active_dirs.discard(out_dir)

    monkeypatch.setattr(api_app_module, "_inspect", tracked_inspect)
    app = create_inspection_app(runs_dir=runs_dir)
    payload = synthetic_png(seed=42)

    def send() -> dict[str, object]:
        with TestClient(app) as local_client:
            resp = upload(local_client, payload)
            assert resp.status_code == 200
            return resp.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(send)
        f2 = pool.submit(send)
        r1, r2 = f1.result(), f2.result()

    assert overlap_detected is False
    assert r1["inspection_id"] == r2["inspection_id"]
    assert sorted([bool(r1["deduplicated"]), bool(r2["deduplicated"])]) == [False, True]


def test_duplicate_key_waiters_do_not_starve_other_images(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """동일 이미지 대기 요청들이 async_pipeline_slot 을 고갈시키지 않아야 한다."""

    import asyncio
    import io
    import threading

    from starlette.datastructures import UploadFile
    from starlette.requests import Request

    from surface_sentinel.api import app as api_app_module

    monkeypatch.setenv("SURFACE_SENTINEL_MAX_CONCURRENCY", "2")
    orig_inspect = api_app_module._inspect
    first_entered = threading.Event()
    release_first = threading.Event()
    other_finished = threading.Event()
    same_payload = synthetic_png(seed=101)
    other_payload = synthetic_png(seed=202)
    same_prefix = hashlib.sha256(same_payload).hexdigest()[:16]

    def gated_inspect(*args: object, **kwargs: object) -> InspectionOutcome:
        out_dir = Path(kwargs["output_dir"])  # type: ignore[arg-type]
        if out_dir.name.startswith(same_prefix) and not first_entered.is_set():
            first_entered.set()
            release_first.wait(timeout=5.0)
        res = orig_inspect(*args, **kwargs)  # type: ignore[arg-type]
        if not out_dir.name.startswith(same_prefix):
            other_finished.set()
        return res

    monkeypatch.setattr(api_app_module, "_inspect", gated_inspect)
    app = create_inspection_app(runs_dir=runs_dir)
    route = next(r for r in app.router.routes if getattr(r, "path", "") == "/api/inspect")
    endpoint = route.endpoint

    def _make_req() -> Request:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/inspect",
            "headers": [(b"host", b"testserver")],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
        }
        return Request(scope)

    def _make_file(name: str, data: bytes) -> UploadFile:
        return UploadFile(file=io.BytesIO(data), filename=name)

    async def _run() -> list[int]:
        t1 = asyncio.create_task(
            endpoint(request=_make_req(), file=_make_file("a.png", same_payload))
        )
        for _ in range(100):
            if first_entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert first_entered.is_set()

        # 동일 이미지 중복 요청 2건 추가(슬롯=2인데 이 둘이 슬롯을 점유하면 안 됨)
        t2 = asyncio.create_task(
            endpoint(request=_make_req(), file=_make_file("a.png", same_payload))
        )
        t3 = asyncio.create_task(
            endpoint(request=_make_req(), file=_make_file("a.png", same_payload))
        )
        await asyncio.sleep(0.05)

        # 무관한 다른 이미지는 슬롯 2 중 남은 1개로 즉시 완료돼야 한다
        try:
            other_res = await asyncio.wait_for(
                endpoint(request=_make_req(), file=_make_file("b.png", other_payload)),
                timeout=1.5,
            )
        finally:
            release_first.set()

        assert other_finished.is_set()
        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        return [r1.inspection_id, r2.inspection_id, r3.inspection_id, other_res.inspection_id]

    ids = asyncio.run(_run())
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]
