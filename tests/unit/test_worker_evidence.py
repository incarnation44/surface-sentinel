"""워커 교차 증거 배선 단위 테스트 (Phase 1, observe-only).

결합 검사가 엔진별 관찰 2개를 보존하면 워커가 별도 증거(agent_assessments)를
append 한다. 증거 실패는 검사 기록을 막지 않는다(격리). 단일 엔진은 증거 없음.
기계 판정은 파이프라인 소유 — 증거가 판정을 바꾸지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import synthetic
from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.agent import worker
from surface_sentinel.agent.worker import audit_log_path, run_worker
from surface_sentinel.config import load_config
from surface_sentinel.contracts import Status
from surface_sentinel.pipeline import InspectionOutcome
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import list_agent_assessments, list_inspections

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "base.toml")


@pytest.fixture(autouse=True)
def _stable_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """테스트가 실제 git 상태에 의존하지 않게 코드 신원을 빈 값으로 고정한다."""

    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("", False))


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


def _patch_inspect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    combined_status: Status,
    observations: tuple,
) -> None:
    """실제 검사 대신 결합 결과 + 엔진 관찰을 돌려주는 가짜로 바꾼다."""

    def fake_inspect(path, config, **kwargs):  # noqa: ANN001 - 워커 _inspect 시그니처
        return InspectionOutcome(
            result=make_result(combined_status, score=0.9),
            engine_observations=observations,
        )

    monkeypatch.setattr(worker, "_inspect", fake_inspect)


def _incoming(tmp_path: Path, seed: int = 0) -> Path:
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    synthetic.write_png(incoming / "sample.png", synthetic.surface_gray(seed=seed))
    return incoming


def _assessments(db_path: Path) -> list[dict[str, object]]:
    connection = connect(db_path)
    try:
        init_db(connection)
        rows = list_inspections(connection, limit=100, offset=0)
        result: list[dict[str, object]] = []
        for row in rows:
            result.extend(list_agent_assessments(connection, int(row["id"])))
        return result
    finally:
        connection.close()


def _audit_events(db_path: Path) -> list[dict[str, object]]:
    path = audit_log_path(db_path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_two_observations_append_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inspect(
        monkeypatch, combined_status=Status.NG, observations=_mismatched_observations()
    )
    db_path = tmp_path / "records.db"
    run_worker(
        incoming=_incoming(tmp_path), db_path=db_path, config=BASE_CONFIG, once=True
    )

    rows = _assessments(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert "STATUS_MISMATCH" in json.loads(str(row["flags_json"]))
    assert row["recommended_action"] == "HUMAN_REVIEW"
    assert len(str(row["assessment_hash"])) == 64
    assert json.loads(str(row["diagnostics_json"]))["engine_a"] == "opencv"
    # 기계 판정은 결합 결과 그대로(증거가 바꾸지 않는다).
    inspection = _rows_status(db_path)
    assert inspection == "NG"
    events = [event["event"] for event in _audit_events(db_path)]
    assert "processed" in events
    assert "evidence_error" not in events


def test_aligned_observations_recommend_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = make_component(0, (10, 10, 20, 20))
    obs_a = make_observation(
        "opencv", Status.NG, components=[component], score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    obs_b = make_observation(
        "yolo-det", Status.NG, components=[component], score=0.9, task="det",
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    _patch_inspect(monkeypatch, combined_status=Status.NG, observations=(obs_a, obs_b))
    db_path = tmp_path / "records.db"
    run_worker(
        incoming=_incoming(tmp_path), db_path=db_path, config=BASE_CONFIG, once=True
    )

    rows = _assessments(db_path)
    assert len(rows) == 1
    assert json.loads(str(rows[0]["flags_json"])) == ["EVIDENCE_ALIGNED"]
    assert rows[0]["recommended_action"] == "NONE"


def test_verdict_unchanged_when_evidence_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 결합 판정 OK 인데 엔진 증거는 불일치 → 증거만 남고 판정은 OK 그대로.
    _patch_inspect(
        monkeypatch, combined_status=Status.OK, observations=_mismatched_observations()
    )
    db_path = tmp_path / "records.db"
    run_worker(
        incoming=_incoming(tmp_path), db_path=db_path, config=BASE_CONFIG, once=True
    )

    assert _rows_status(db_path) == "OK"
    rows = _assessments(db_path)
    assert len(rows) == 1
    assert rows[0]["recommended_action"] == "HUMAN_REVIEW"


def test_single_engine_has_no_assessment(tmp_path: Path) -> None:
    # 실제 opencv 단일 엔진 경로 — 관찰이 없으므로 증거 테이블은 비어 있다.
    db_path = tmp_path / "records.db"
    run_worker(
        incoming=_incoming(tmp_path), db_path=db_path, config=BASE_CONFIG, once=True
    )
    assert _assessments(db_path) == []


def test_assessment_failure_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inspect(
        monkeypatch, combined_status=Status.NG, observations=_mismatched_observations()
    )

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("evidence boom")

    monkeypatch.setattr(worker, "assess_cross_evidence", boom)
    db_path = tmp_path / "records.db"
    run_worker(
        incoming=_incoming(tmp_path), db_path=db_path, config=BASE_CONFIG, once=True
    )

    # 검사 기록은 살아 있고, 증거 실패는 감사 로그로만 남는다.
    assert _rows_status(db_path) == "NG"
    assert _assessments(db_path) == []
    errors = [
        event for event in _audit_events(db_path) if event["event"] == "evidence_error"
    ]
    assert len(errors) == 1
    assert "evidence boom" in str(errors[0]["message"])


def test_assessment_hash_stable_across_inspections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inspect(
        monkeypatch, combined_status=Status.NG, observations=_mismatched_observations()
    )
    db_path = tmp_path / "records.db"
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    synthetic.write_png(incoming / "a.png", synthetic.surface_gray(seed=1))
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)
    synthetic.write_png(incoming / "b.png", synthetic.surface_gray(seed=2))
    run_worker(incoming=incoming, db_path=db_path, config=BASE_CONFIG, once=True)

    rows = _assessments(db_path)
    assert len(rows) == 2  # 검사별 행은 쌓이되
    assert rows[0]["assessment_hash"] == rows[1]["assessment_hash"]  # 해시는 같다


def _rows_status(db_path: Path) -> str:
    connection = connect(db_path)
    try:
        init_db(connection)
        rows = list_inspections(connection, limit=100, offset=0)
        assert len(rows) == 1
        return str(rows[0]["status"])
    finally:
        connection.close()
