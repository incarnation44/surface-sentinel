"""교차 증거 도구의 상태 머신 통합 단위 테스트 — observe-only 계약과 최종 판정 불변성 검증.

``cross_evidence`` 는 T09 상태 머신의 작업 도구로 동작한다: 예산 소모,
중복 거부, 결과 해시, replay 결정론. 판정은 항상 파이프라인(결합 결과) 소유이며
에이전트는 별도 증거만 남긴다. 2프로파일 재검사는 하지 않는다(민감도는 Phase 2).
"""

from __future__ import annotations

import pytest

from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.agent.config import AgentConfig, BudgetConfig, default_agent_config
from surface_sentinel.agent.engine import (
    WORK_TOOLS,
    AgentEngine,
    AgentGuardError,
    AgentState,
    choose_evidence_action,
)
from surface_sentinel.agent.evidence import EvidenceFlag, assess_cross_evidence
from surface_sentinel.agent.trace import TraceWriter, audit_trace, replay_trace
from surface_sentinel.config import default_config
from surface_sentinel.contracts import ErrorCode, Status


def _observations() -> tuple:
    obs_a = make_observation(
        "opencv",
        Status.NG,
        components=[make_component(0, (10, 10, 20, 20))],
        score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    obs_b = make_observation(
        "yolo-det",
        Status.OK,
        task="det",
    )
    return obs_a, obs_b


def _evidence_engine(
    combined_status: Status,
    *,
    reasons: tuple[str, ...] = (),
    agent_config: AgentConfig | None = None,
    observations: tuple | None = None,
    trace: TraceWriter | None = None,
) -> AgentEngine:
    """결합 결과를 이미 가진 관찰 전용 에피소드 엔진(재검사 없음)."""

    combined = make_result(combined_status, reasons=list(reasons), score=0.9)
    obs_a, obs_b = _observations() if observations is None else observations
    return AgentEngine(
        agent_config=agent_config or default_agent_config(),
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: combined,
        clock=lambda: 0.0,
        trace=trace,
        observations=(obs_a, obs_b),
    )


def test_cross_evidence_is_work_tool() -> None:
    assert "cross_evidence" in WORK_TOOLS


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        (Status.OK, AgentState.DONE),
        (Status.NG, AgentState.DONE),
        (Status.REVIEW, AgentState.REVIEW),
        (Status.ERROR, AgentState.ERROR),
    ],
)
def test_verdict_ownership_final_status_equals_combined(
    status: Status, terminal: AgentState
) -> None:
    engine = _evidence_engine(status, reasons=("EVIDENCE_IN_REVIEW_BAND",))
    episode = engine.run()
    assert episode.terminal_state is terminal
    assert episode.final_status is status  # 판정 불변 — 증거가 판정을 바꾸지 않는다
    assert engine.assessment is not None
    assert audit_trace(list(episode.events)) == []


def test_evidence_episode_runs_cross_evidence_then_base_only() -> None:
    engine = _evidence_engine(Status.OK)
    episode = engine.run()
    tools = [event["tool"] for event in episode.events if event.get("event") == "act"]
    assert tools == ["cross_evidence", "run_profile", "stop"]
    assert episode.actions_used == 2  # cross_evidence + run_profile (종료 도구 미소모)
    assert episode.profiles_used == 1


def test_review_with_quality_flag_does_not_run_second_profile() -> None:
    # observe-only: 민감도 재검사(2프로파일)는 Phase 2. 여기선 증거만 남기고 REVIEW.
    engine = _evidence_engine(Status.REVIEW, reasons=("INPUT_QUALITY_LOW",))
    episode = engine.run()
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.profiles_used == 1
    assert episode.reason == "INPUT_QUALITY_LOW"


def test_cross_evidence_result_is_hashed_in_trace() -> None:
    writer = TraceWriter(None)
    engine = _evidence_engine(Status.NG, trace=writer)
    episode = engine.run()
    acts = [event for event in episode.events if event.get("event") == "act"]
    evidence_act = acts[0]
    assert evidence_act["tool"] == "cross_evidence"
    assert evidence_act["result_hash"]
    assert evidence_act["result"]["assessment_hash"] == engine.assessment.assessment_hash
    assert "STATUS_MISMATCH" in evidence_act["result"]["flags"]


def test_episode_start_carries_observations_for_replay() -> None:
    engine = _evidence_engine(Status.OK)
    episode = engine.run()
    start = episode.events[0]
    assert start["event"] == "episode_start"
    assert len(start["engine_observations"]) == 2
    assert start["engine_observations"][0]["engine"] == "opencv"


def test_t09_episode_without_observations_is_unchanged() -> None:
    engine = AgentEngine(
        agent_config=default_agent_config(),
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: make_result(Status.OK),
        clock=lambda: 0.0,
    )
    episode = engine.run()
    assert "engine_observations" not in episode.events[0]
    assert episode.actions_used == 1
    assert engine.assessment is None


def test_duplicate_cross_evidence_rejected() -> None:
    engine = _evidence_engine(Status.OK)
    engine._state = AgentState.ACT
    engine.attempt("cross_evidence", {})
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("cross_evidence", {})
    assert excinfo.value.code == "DUPLICATE_ACTION"


def test_cross_evidence_without_observations_fails_to_review() -> None:
    engine = AgentEngine(
        agent_config=default_agent_config(),
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: make_result(Status.OK),
        clock=lambda: 0.0,
    )
    engine._state = AgentState.ACT
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("cross_evidence", {})
    assert excinfo.value.code == ErrorCode.E_TOOL_FAILED.value


def test_cross_evidence_consumes_action_budget() -> None:
    config = AgentConfig(budget=BudgetConfig(max_actions=1, max_profiles=1))
    engine = _evidence_engine(Status.OK, agent_config=config)
    episode = engine.run()
    # cross_evidence 가 예산 1회를 소모 → run_profile 거부 → 낙관적 OK 없이 REVIEW.
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.rejections[-1]["code"] == ErrorCode.E_BUDGET_EXCEEDED.value


@pytest.mark.parametrize("status", [Status.OK, Status.NG, Status.REVIEW, Status.ERROR])
def test_replay_is_deterministic(status: Status) -> None:
    writer = TraceWriter(None)
    engine = _evidence_engine(status, reasons=("EVIDENCE_IN_REVIEW_BAND",), trace=writer)
    engine.run()
    report = replay_trace(writer.events)
    assert report.audit_valid is True
    assert report.deterministic is True
    assert report.mismatches == ()
    assert report.final_status == status.value


def test_replay_detects_tampered_observations() -> None:
    writer = TraceWriter(None)
    engine = _evidence_engine(Status.NG, trace=writer)
    engine.run()
    events = [dict(event) for event in writer.events]
    start = events[0]
    observations = [dict(item) for item in start["engine_observations"]]
    result = dict(observations[1]["result"])
    result["score"] = 0.1  # 증거 변조 → 재계산 해시가 달라져야 한다
    observations[1]["result"] = result
    start["engine_observations"] = observations
    report = replay_trace(events)
    assert report.deterministic is False


def test_choose_evidence_action_table() -> None:
    policy = default_agent_config().policy
    assert choose_evidence_action([], policy, cross_evidence_done=False).tool == "cross_evidence"
    assert choose_evidence_action([], policy, cross_evidence_done=True).tool == "run_profile"


def test_assessment_matches_pure_function() -> None:
    engine = _evidence_engine(Status.NG)
    engine.run()
    obs_a, obs_b = _observations()
    expected = assess_cross_evidence(obs_a, obs_b)
    assert engine.assessment.assessment_hash == expected.assessment_hash
    assert EvidenceFlag.STATUS_MISMATCH in engine.assessment.flags


def test_phase2_tools_registered_in_work_tools() -> None:
    assert {"inspect_geometry", "arbitrate_patches"}.issubset(WORK_TOOLS)


def test_choose_evidence_action_arbitration_branch() -> None:
    policy = default_agent_config().policy
    assert (
        choose_evidence_action(
            [],
            policy,
            cross_evidence_done=True,
            assessment_flags=("LOCALIZATION_MISMATCH",),
            arbitration_available=True,
            arbitration_done=False,
            budget_remaining_actions=2,
        ).tool
        == "arbitrate_patches"
    )
    assert (
        choose_evidence_action(
            [],
            policy,
            cross_evidence_done=True,
            assessment_flags=("NG_SINGLE_ENGINE",),
            arbitration_available=True,
            arbitration_done=False,
            budget_remaining_actions=1,
        ).tool
        == "run_profile"
    )
    assert (
        choose_evidence_action(
            [],
            policy,
            cross_evidence_done=True,
            assessment_flags=("EVIDENCE_ALIGNED",),
            arbitration_available=True,
            arbitration_done=False,
            budget_remaining_actions=2,
        ).tool
        == "run_profile"
    )


@pytest.mark.parametrize("status", [Status.OK, Status.NG, Status.REVIEW])
def test_active_arbitration_within_max_actions_3_and_replay(status: Status) -> None:
    import numpy as np

    img = np.full((120, 120, 3), 120, dtype=np.uint8)
    img[30:50, 30:50, :] = 20
    obs_a = make_observation(
        "opencv",
        Status.NG,
        components=[make_component(0, (28, 28, 24, 24), evidence=0.9)],
        score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
    )
    obs_b = make_observation("yolo-seg", Status.OK, task="seg")
    combined = make_result(status, reasons=["EVIDENCE_IN_REVIEW_BAND"], score=0.9)
    writer = TraceWriter(None)
    engine = AgentEngine(
        agent_config=default_agent_config(),  # max_actions=3
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: combined,
        clock=lambda: 0.0,
        trace=writer,
        observations=(obs_a, obs_b),
        image_bgr=img,
    )
    episode = engine.run()
    tools = [event["tool"] for event in episode.events if event.get("event") == "act"]
    assert tools[:3] == ["cross_evidence", "arbitrate_patches", "run_profile"]
    assert episode.actions_used == 3
    assert episode.profiles_used == 1
    assert episode.final_status is status
    assert engine.arbitration is not None
    assert audit_trace(list(episode.events)) == []

    report = replay_trace(writer.events)
    assert report.audit_valid is True
    assert report.deterministic is True
    assert report.mismatches == ()
    assert report.final_status == status.value


def test_aligned_episode_with_image_skips_arbitration_and_replays() -> None:
    import numpy as np

    img = np.full((120, 120, 3), 120, dtype=np.uint8)
    obs_a = make_observation("opencv", Status.OK)
    obs_b = make_observation("yolo-seg", Status.OK, task="seg")
    writer = TraceWriter(None)
    engine = AgentEngine(
        agent_config=default_agent_config(),
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: make_result(Status.OK),
        clock=lambda: 0.0,
        trace=writer,
        observations=(obs_a, obs_b),
        image_bgr=img,
    )
    episode = engine.run()
    tools = [event["tool"] for event in episode.events if event.get("event") == "act"]
    assert tools == ["cross_evidence", "run_profile", "stop"]
    assert episode.actions_used == 2
    assert engine.arbitration is None
    report = replay_trace(writer.events)
    assert report.deterministic is True


def test_replay_detects_tampered_arbitration_hash() -> None:
    import numpy as np

    img = np.full((120, 120, 3), 120, dtype=np.uint8)
    img[30:50, 30:50, :] = 20
    obs_a = make_observation(
        "opencv", Status.NG, components=[make_component(0, (28, 28, 24, 24))], score=0.9
    )
    obs_b = make_observation("yolo-seg", Status.OK, task="seg")
    writer = TraceWriter(None)
    engine = AgentEngine(
        agent_config=default_agent_config(),
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: make_result(Status.NG, score=0.9),
        clock=lambda: 0.0,
        trace=writer,
        observations=(obs_a, obs_b),
        image_bgr=img,
    )
    engine.run()
    events = [dict(event) for event in writer.events]
    for event in events:
        if event.get("event") == "act" and event.get("tool") == "arbitrate_patches":
            res = dict(event["result"])
            res["verdict"] = "ALIGNED"
            event["result"] = res
    report = replay_trace(events)
    assert report.deterministic is False


def test_inspect_geometry_tool_execution() -> None:
    obs_a = make_observation(
        "opencv",
        Status.REVIEW,
        components=[
            make_component(0, (10, 20, 10, 5), evidence=0.35),
            make_component(1, (30, 22, 10, 5), evidence=0.34),
            make_component(2, (50, 24, 10, 5), evidence=0.33),
        ],
        score=0.35,
    )
    obs_b = make_observation("yolo-seg", Status.OK, task="seg")
    engine = AgentEngine(
        agent_config=default_agent_config(),
        profiles={"base": default_config()},
        inspector=lambda profile_id, config: make_result(Status.REVIEW),
        clock=lambda: 0.0,
        observations=(obs_a, obs_b),
    )
    engine._state = AgentState.ACT
    out = engine.attempt("inspect_geometry", {"engine": "opencv"})
    assert out["pattern"] == "COLLINEAR_SCRATCH_SUSPECT"
    assert engine.geometry is not None
