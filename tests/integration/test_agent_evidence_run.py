"""Phase 1 교차 증거 하네스 통합 테스트 (observe-only 에피소드).

결합 판정(캐시) + 엔진 관찰 2개를 주입하면 교차 증거 에피소드가 돈다:
dual overlay·결과 JSON·트레이스를 남기고, 판정은 결합 결과 그대로다.
봉인 분할 거부·관찰 부족 거부·replay 결정론을 고정한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.agent.config import load_agent_config
from surface_sentinel.agent.run import (
    AgentRunError,
    load_profiles,
    run_evidence_episode,
)
from surface_sentinel.agent.trace import audit_trace, load_trace, replay_trace
from surface_sentinel.config import load_config
from surface_sentinel.contracts import Status

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.toml"
AGENT_CONFIG_PATH = REPO_ROOT / "configs" / "agent.toml"
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"
SYNTHETIC_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"

BASE_CONFIG = load_config(BASE_CONFIG_PATH)
AGENT_CONFIG = load_agent_config(AGENT_CONFIG_PATH)
PROFILES = load_profiles(BASE_CONFIG_PATH, PROFILES_DIR)


def _observations() -> tuple:
    obs_a = make_observation(
        "opencv",
        Status.NG,
        components=[make_component(0, (10, 10, 20, 20))],
        score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    obs_b = make_observation("yolo-det", Status.OK, task="det")
    return (obs_a, obs_b)


def _run(tmp_path: Path, *, combined_status: Status = Status.OK, **kwargs):
    return run_evidence_episode(
        SYNTHETIC_DIR / "good.png",
        combined_result=make_result(combined_status, score=0.9),
        observations=_observations(),
        agent_config=AGENT_CONFIG,
        profiles=PROFILES,
        output_dir=tmp_path / "runs",
        **kwargs,
    )


def test_evidence_episode_writes_artifacts_and_keeps_verdict(tmp_path: Path) -> None:
    outcome = _run(tmp_path, combined_status=Status.OK)
    episode = outcome.episode
    # 판정 불변: 증거가 불일치해도 결합 판정 OK 그대로.
    assert episode.final_status is Status.OK
    assert episode.profiles_used == 1  # 재검사 없음(캐시 결합 결과)
    assert episode.actions_used == 2  # cross_evidence + run_profile
    assert outcome.assessment is not None
    assert "STATUS_MISMATCH" in [flag.value for flag in outcome.assessment.flags]
    assert outcome.overlay_path.is_file()
    assert outcome.overlay_path.stat().st_size > 0
    assert audit_trace(load_trace(outcome.trace_path)) == []

    payload = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "agent-evidence-run-v1"
    assert payload["final_status"] == "OK"
    assert payload["combined_status"] == "OK"
    assert payload["assessment"]["assessment_hash"] == outcome.assessment.assessment_hash
    assert payload["config_hash"] == BASE_CONFIG.config_hash


@pytest.mark.parametrize("status", [Status.OK, Status.NG, Status.REVIEW, Status.ERROR])
def test_verdict_mirrors_combined_result(tmp_path: Path, status: Status) -> None:
    outcome = _run(tmp_path, combined_status=status)
    assert outcome.episode.final_status is status


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_sealed_splits_are_never_opened(tmp_path: Path, split: str) -> None:
    with pytest.raises(AgentRunError) as excinfo:
        _run(tmp_path, split=split)
    assert excinfo.value.code == "E_SEALED_DENIED"


def test_missing_image_rejected(tmp_path: Path) -> None:
    with pytest.raises(AgentRunError) as excinfo:
        run_evidence_episode(
            tmp_path / "missing.png",
            combined_result=make_result(Status.OK),
            observations=_observations(),
            agent_config=AGENT_CONFIG,
            profiles=PROFILES,
            output_dir=tmp_path / "runs",
        )
    assert excinfo.value.code == "E_INPUT_MISSING"


def test_fewer_than_two_observations_rejected(tmp_path: Path) -> None:
    with pytest.raises(AgentRunError) as excinfo:
        run_evidence_episode(
            SYNTHETIC_DIR / "good.png",
            combined_result=make_result(Status.OK),
            observations=_observations()[:1],
            agent_config=AGENT_CONFIG,
            profiles=PROFILES,
            output_dir=tmp_path / "runs",
        )
    assert excinfo.value.code == "E_EVIDENCE_INPUT"


def test_missing_base_profile_rejected(tmp_path: Path) -> None:
    with pytest.raises(AgentRunError) as excinfo:
        run_evidence_episode(
            SYNTHETIC_DIR / "good.png",
            combined_result=make_result(Status.OK),
            observations=_observations(),
            agent_config=AGENT_CONFIG,
            profiles={"other": BASE_CONFIG},
            output_dir=tmp_path / "runs",
        )
    assert excinfo.value.code == "E_CONFIG_INVALID"


def test_replay_from_trace_file_is_deterministic(tmp_path: Path) -> None:
    outcome = _run(tmp_path, combined_status=Status.NG)
    report = replay_trace(load_trace(outcome.trace_path))
    assert report.audit_valid is True
    assert report.deterministic is True
    assert report.mismatches == ()
    assert report.final_status == "NG"


def test_run_id_collision_gets_suffix(tmp_path: Path) -> None:
    first = _run(tmp_path, run_id="fixed-run")
    second = _run(tmp_path, run_id="fixed-run")
    assert first.run_id == "fixed-run"
    assert second.run_id == "fixed-run-01"
    assert first.run_dir != second.run_dir
