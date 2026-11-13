"""API 교차 증거 배선 통합 테스트 (Phase 1, observe-only).

결합(combined) 검사가 엔진별 관찰 2개를 보존하면 API 도 별도 증거
(agent_assessments)를 append 한다. YOLO 경로는 ``sys.modules`` stub 으로
검증한다(모델 파일 불필요). 기계 판정 불변·증거 실패 격리·중복 업로드
멱등을 고정한다.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.api import app as app_module
from surface_sentinel.api import create_inspection_app
from surface_sentinel.contracts import Status
from surface_sentinel.pipeline import InspectionOutcome
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import list_agent_assessments, list_inspections


def synthetic_png(seed: int = 0, size: int = 96) -> bytes:
    """검사 입력 검증(min_std)을 통과하는 결정론적 합성 PNG."""

    image = np.tile(np.linspace(40, 200, size, dtype=np.uint8), (size, 1))
    image = np.roll(image, seed * 7, axis=1)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


@pytest.fixture(autouse=True)
def _stable_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
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


def upload(client: TestClient, payload: bytes, *, engine: str | None = None) -> object:
    params = {} if engine is None else {"engine": engine}
    return client.post(
        "/api/inspect",
        files={"file": ("sample.png", payload, "image/png")},
        data={"label": "unknown", "split": "tune"},
        params=params,
    )


def _mismatched_observations() -> tuple:
    obs_a = make_observation(
        "opencv",
        Status.NG,
        components=[make_component(0, (10, 10, 20, 20))],
        score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    obs_b = make_observation("yolo-det", Status.OK, task="det")
    return (obs_a, obs_b)


def _stub_combined(
    monkeypatch: pytest.MonkeyPatch,
    *,
    combined_status: Status,
    observations: tuple,
) -> None:
    """결합 결과 + 엔진 관찰 2개를 돌려주는 yolo.inspect 스텁."""

    fake = types.ModuleType("surface_sentinel.yolo.inspect")

    def stub(image_path, config, **kwargs):  # noqa: ANN001 - 진입점 시그니처
        return InspectionOutcome(
            result=make_result(combined_status, score=0.9),
            engine_observations=observations,
        )

    fake.inspect_with_yolo = stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "surface_sentinel.yolo.inspect", fake)


def _assessments(runs_dir: Path) -> list[dict[str, object]]:
    connection = connect(runs_dir / "records.db")
    try:
        init_db(connection)
        rows = list_inspections(connection, limit=100, offset=0)
        result: list[dict[str, object]] = []
        for row in rows:
            result.extend(list_agent_assessments(connection, int(row["id"])))
        return result
    finally:
        connection.close()


def test_combined_upload_appends_assessment(
    runs_dir: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_combined(
        monkeypatch, combined_status=Status.NG, observations=_mismatched_observations()
    )
    response = upload(client, synthetic_png(seed=1), engine="combined")
    assert response.status_code == 200
    assert response.json()["status"] == "NG"  # 기계 판정 불변

    rows = _assessments(runs_dir)
    assert len(rows) == 1
    assert "STATUS_MISMATCH" in json.loads(str(rows[0]["flags_json"]))
    assert rows[0]["recommended_action"] == "HUMAN_REVIEW"


def test_opencv_upload_has_no_assessment(runs_dir: Path, client: TestClient) -> None:
    response = upload(client, synthetic_png(seed=2))
    assert response.status_code == 200
    assert _assessments(runs_dir) == []


def test_assessment_failure_does_not_break_response(
    runs_dir: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_combined(
        monkeypatch, combined_status=Status.NG, observations=_mismatched_observations()
    )

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("evidence boom")

    monkeypatch.setattr(app_module, "assess_cross_evidence", boom)
    response = upload(client, synthetic_png(seed=3), engine="combined")
    assert response.status_code == 200  # 증거 실패는 검사 응답을 막지 않는다
    assert response.json()["status"] == "NG"
    assert _assessments(runs_dir) == []


def test_duplicate_upload_keeps_single_assessment(
    runs_dir: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_combined(
        monkeypatch, combined_status=Status.NG, observations=_mismatched_observations()
    )
    payload = synthetic_png(seed=4)
    first = upload(client, payload, engine="combined")
    second = upload(client, payload, engine="combined")
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["deduplicated"] is True
    rows = _assessments(runs_dir)
    assert len(rows) == 1  # 같은 검사 + 같은 해시 → 중복 기록 없음
