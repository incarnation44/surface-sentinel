"""검사 API 단건 상세(`GET /inspections/{id}`) 및 UI 통합 마운트 테스트.

엔진 관찰·교차 증거 응답 포함 여부와 `--enable-api` 마운트 시 JSON 오류 반환을 검증한다.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.api import create_inspection_app
from surface_sentinel.contracts import Status
from surface_sentinel.pipeline import InspectionOutcome
from surface_sentinel.ui import create_app


def _png_bytes(seed: int = 1, size: int = 96) -> bytes:
    image = np.tile(np.linspace(30, 210, size, dtype=np.uint8), (size, 1))
    image = np.roll(image, seed * 5, axis=1)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def _stable_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("", False))


def _stub_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    obs_a = make_observation(
        "opencv",
        Status.REVIEW,
        components=[make_component(0, (12, 12, 24, 24), area_px=140, evidence=0.45)],
        score=0.45,
        reasons=["EVIDENCE_IN_REVIEW_BAND"],
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (14, 14, 22, 22), area_px=130, evidence=0.82)],
        score=0.82,
        task="det",
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    fake = types.ModuleType("surface_sentinel.yolo.inspect")

    def stub(image_path, config, **kwargs):  # noqa: ANN001
        return InspectionOutcome(
            result=replace(
                make_result(
                    Status.REVIEW,
                    score=0.45,
                    reasons=["EVIDENCE_IN_REVIEW_BAND"],
                ),
                components=(make_component(0, (12, 12, 24, 24), area_px=140, evidence=0.45),),
            ),
            engine_observations=(obs_a, obs_b),
        )

    fake.inspect_with_yolo = stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "surface_sentinel.yolo.inspect", fake)


def test_inspection_detail_returns_observations_and_assessments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_combined(monkeypatch)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    client = TestClient(create_inspection_app(runs_dir=runs_dir))

    created = client.post(
        "/api/inspect",
        files={"file": ("probe.png", _png_bytes(seed=3), "image/png")},
        data={"label": "defect", "split": "tune"},
        params={"engine": "combined"},
    )
    assert created.status_code == 200
    iid = created.json()["inspection_id"]

    detail = client.get(f"/api/inspections/{iid}")
    assert detail.status_code == 200
    payload = detail.json()

    assert len(payload["observations"]) == 2
    assert [o["engine"] for o in payload["observations"]] == ["opencv", "yolo-det"]
    assert payload["observations"][0]["status"] == "REVIEW"
    assert payload["observations"][1]["status"] == "NG"
    assert payload["observations"][1]["task"] == "det"

    assert len(payload["assessments"]) == 1
    assess = payload["assessments"][0]
    assert assess["review_priority"] in {"P0", "P1", "P2", "P3"}
    assert "STATUS_MISMATCH" in assess["flags"]
    assert assess["recommended_action"] == "HUMAN_REVIEW"
    assert "triage" in assess["diagnostics"]
    assert "arbitration" in assess["diagnostics"]
    assert assess["diagnostics"]["triage"]["argumentation"]["operator_guidance"]


def test_single_engine_detail_has_empty_observations_and_assessments(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    client = TestClient(create_inspection_app(runs_dir=runs_dir))
    created = client.post(
        "/api/inspect",
        files={"file": ("single.png", _png_bytes(seed=7), "image/png")},
        data={"label": "good", "split": "tune"},
    )
    assert created.status_code == 200
    detail = client.get(f"/api/inspections/{created.json()['inspection_id']}").json()
    assert detail["observations"] == []
    assert detail["assessments"] == []


def test_ui_with_enable_api_mounts_api_routes_and_returns_json_errors(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    raw_root.mkdir()
    app = create_app(runs_dir=runs_dir, raw_root=raw_root, enable_api=True)
    client = TestClient(app)

    ui_health = client.get("/healthz")
    assert ui_health.status_code == 200
    assert ui_health.json()["service"] == "surface-sentinel-ui"
    assert ui_health.json()["api_enabled"] is True

    api_health = client.get("/api/healthz")
    assert api_health.status_code == 200
    assert api_health.json()["service"] == "surface-sentinel-api"

    resp = client.post(
        "/api/inspect",
        files={"file": ("live.png", _png_bytes(seed=9), "image/png")},
        data={"label": "unknown", "split": "tune"},
    )
    assert resp.status_code == 200
    iid = resp.json()["inspection_id"]
    assert client.get(f"/api/inspections/{iid}").status_code == 200
    assert client.get("/api/stats").json()["total"] == 1

    # /api/* 404 오류는 HTML이 아닌 JSON으로 응답해야 한다.
    missing = client.get("/api/inspections/9999")
    assert missing.status_code == 404
    assert "detail" in missing.json()
    assert client.get("/docs").status_code == 404
