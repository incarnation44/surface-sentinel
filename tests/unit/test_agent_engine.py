"""T09 정책 엔진 단위 테스트.

설계 §6 계약: 잘못된 전이/예산/중복 거부, NG 이후 재시도 없음, 소진 시 REVIEW,
판정은 파이프라인 소유. 합성 결과(가짜 inspector)만 사용한다.
"""

from __future__ import annotations

import pytest

from agent_fakes import make_engine, make_result, observation
from surface_sentinel.agent.config import (
    AgentConfig,
    BudgetConfig,
    PolicyConfig,
    default_agent_config,
)
from surface_sentinel.agent.engine import (
    AgentGuardError,
    AgentState,
    Terminal,
    ToolCall,
    choose_action,
    compare_terminal,
)
from surface_sentinel.agent.trace import audit_trace
from surface_sentinel.contracts import ErrorCode, Status


def test_clean_ok_path_stops_with_pipeline_ok() -> None:
    engine, inspector = make_engine({"base": make_result(Status.OK)})
    episode = engine.run()
    assert episode.terminal_state is AgentState.DONE
    assert episode.final_status is Status.OK
    assert episode.actions_used == 1
    assert episode.profiles_used == 1
    assert inspector.calls == ["base"]
    assert episode.rejections == ()


def test_base_ng_is_terminal_without_retry() -> None:
    engine, inspector = make_engine(
        {"base": make_result(Status.NG, reasons=["DEFECT_EVIDENCE_STRONG"], score=0.9)}
    )
    episode = engine.run()
    assert episode.terminal_state is AgentState.DONE
    assert episode.final_status is Status.NG
    assert episode.actions_used == 1
    assert inspector.calls == ["base"]


def test_quality_low_runs_denoise_once_then_compare() -> None:
    engine, inspector = make_engine(
        {
            "base": make_result(
                Status.REVIEW,
                reasons=["INPUT_QUALITY_LOW"],
                quality_flags=["INPUT_QUALITY_LOW"],
            ),
            "denoise_strong": make_result(Status.OK),
        }
    )
    episode = engine.run()
    assert inspector.calls == ["base", "denoise_strong"]
    assert episode.actions_used == 3
    assert episode.profiles_used == 2
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.final_status is Status.REVIEW
    assert episode.reason == "PROFILE_MISMATCH"
    assert audit_trace(list(episode.events)) == []


def test_roi_uncertain_runs_roi_fixed_and_ng_wins() -> None:
    engine, inspector = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["ROI_UNCERTAIN"], roi_uncertain=True),
            "roi_fixed": make_result(Status.NG, reasons=["DEFECT_EVIDENCE_STRONG"], score=0.8),
        }
    )
    episode = engine.run()
    assert inspector.calls == ["base", "roi_fixed"]
    assert episode.terminal_state is AgentState.DONE
    assert episode.final_status is Status.NG
    assert episode.actions_used == 2


def test_review_band_escalates_without_second_profile() -> None:
    engine, inspector = make_engine(
        {"base": make_result(Status.REVIEW, reasons=["EVIDENCE_IN_REVIEW_BAND"])}
    )
    episode = engine.run()
    assert inspector.calls == ["base"]
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.final_status is Status.REVIEW
    assert episode.reason == "EVIDENCE_IN_REVIEW_BAND"


def test_both_profiles_review_stays_review() -> None:
    engine, _ = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
            "denoise_strong": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
        }
    )
    episode = engine.run()
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.reason == "PROFILES_REVIEW"


def test_unknown_profile_is_error() -> None:
    config = AgentConfig(
        policy=PolicyConfig(
            base_profile="base",
            quality_low_profile="ghost",
            roi_uncertain_profile="roi_fixed",
        )
    )
    engine, _ = make_engine(
        {"base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"])},
        agent_config=config,
    )
    episode = engine.run()
    assert episode.terminal_state is AgentState.ERROR
    assert episode.final_status is Status.ERROR
    assert episode.rejections[0]["code"] == ErrorCode.E_PROFILE_UNKNOWN.value


def test_action_budget_exhaustion_goes_to_review() -> None:
    config = AgentConfig(budget=BudgetConfig(max_actions=1, max_profiles=1))
    engine, _ = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
            "denoise_strong": make_result(Status.OK),
        },
        agent_config=config,
    )
    episode = engine.run()
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.final_status is Status.REVIEW
    assert episode.rejections[0]["code"] == ErrorCode.E_BUDGET_EXCEEDED.value


def test_profile_budget_exhaustion_goes_to_review() -> None:
    config = AgentConfig(budget=BudgetConfig(max_actions=2, max_profiles=1))
    engine, _ = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["ROI_UNCERTAIN"]),
            "roi_fixed": make_result(Status.OK),
        },
        agent_config=config,
    )
    episode = engine.run()
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.rejections[0]["code"] == ErrorCode.E_BUDGET_EXCEEDED.value


def test_latency_budget_exhaustion_goes_to_review() -> None:
    times = iter([0.0, 0.0, 0.0, 6.0])
    engine, _ = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
            "denoise_strong": make_result(Status.OK),
        },
        clock=lambda: next(times),
    )
    episode = engine.run()
    assert episode.terminal_state is AgentState.REVIEW
    assert episode.extra_latency_ms == 6000.0
    assert episode.rejections[-1]["code"] == ErrorCode.E_BUDGET_EXCEEDED.value


def test_duplicate_action_hash_rejected() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    engine._state = AgentState.ACT
    engine.attempt("run_profile", {"profile_id": "base"})
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("run_profile", {"profile_id": "base"})
    assert excinfo.value.code == "DUPLICATE_ACTION"


def test_invalid_transition_before_act_rejected() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("run_profile", {"profile_id": "base"})
    assert excinfo.value.code == "INVALID_TRANSITION"


def test_invalid_transition_after_terminal_rejected() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    engine.run()
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("stop", {})
    assert excinfo.value.code == "INVALID_TRANSITION"


def test_retry_after_ng_rejected() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    engine._state = AgentState.ACT
    engine._ng_seen = True
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("run_profile", {"profile_id": "base"})
    assert excinfo.value.code == "RETRY_AFTER_NG"


def test_unknown_tool_rejected() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    engine._state = AgentState.ACT
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("shell", {"cmd": "rm"})
    assert excinfo.value.code == "INVALID_TRANSITION"


def test_stop_cannot_confirm_review_as_ok() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    engine._state = AgentState.ACT
    engine._summaries = {"base": {"status": "REVIEW"}}
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("stop", {})
    assert excinfo.value.code == ErrorCode.E_TOOL_FAILED.value


def test_inspect_artifact_is_read_only() -> None:
    engine, inspector = make_engine({"base": make_result(Status.OK)})
    engine._state = AgentState.ACT
    engine._summaries = {"base": {"status": "OK", "score": 0.0}}
    result = engine.attempt("inspect_artifact", {"profile_id": "base"})
    assert result["status"] == "OK"
    assert inspector.calls == []
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("inspect_artifact", {"profile_id": "missing"})
    assert excinfo.value.code == ErrorCode.E_TOOL_FAILED.value


def test_compare_requires_cached_profiles() -> None:
    engine, _ = make_engine({"base": make_result(Status.OK)})
    engine._state = AgentState.ACT
    with pytest.raises(AgentGuardError) as excinfo:
        engine.attempt("compare", {"profile_a": "base", "profile_b": "other"})
    assert excinfo.value.code == ErrorCode.E_TOOL_FAILED.value


def test_verdict_ownership_ng_never_becomes_ok() -> None:
    engine, _ = make_engine(
        {"base": make_result(Status.NG, reasons=["DEFECT_EVIDENCE_STRONG"], score=1.0)}
    )
    episode = engine.run()
    assert episode.final_status is Status.NG
    for event in episode.events:
        if event["event"] == "episode_end":
            assert event["final_status"] == "NG"


def test_policy_table_rows() -> None:
    policy = default_agent_config().policy
    assert choose_action([], policy) == ToolCall("run_profile", {"profile_id": "base"})
    assert choose_action([observation(Status.OK)], policy) == ToolCall("stop", {})
    assert choose_action([observation(Status.NG)], policy) == Terminal(
        AgentState.DONE, Status.NG, "PIPELINE_NG"
    )
    assert choose_action([observation(Status.ERROR)], policy).state is AgentState.ERROR
    assert choose_action(
        [observation(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"])], policy
    ) == ToolCall("run_profile", {"profile_id": "denoise_strong"})
    assert choose_action(
        [observation(Status.REVIEW, reasons=["ROI_UNCERTAIN"])], policy
    ) == ToolCall("run_profile", {"profile_id": "roi_fixed"})
    assert choose_action(
        [observation(Status.REVIEW, reasons=["EVIDENCE_IN_REVIEW_BAND"])], policy
    ) == ToolCall("escalate_review", {"reason": "EVIDENCE_IN_REVIEW_BAND"})
    two = [
        observation(Status.OK),
        observation(Status.REVIEW, profile_id="denoise_strong"),
    ]
    assert choose_action(two, policy) == ToolCall(
        "compare", {"profile_a": "base", "profile_b": "denoise_strong"}
    )
    ng_two = [observation(Status.OK), observation(Status.NG, profile_id="denoise_strong")]
    assert choose_action(ng_two, policy) == Terminal(AgentState.DONE, Status.NG, "PIPELINE_NG")


def test_compare_terminal_table() -> None:
    assert compare_terminal(Status.OK, Status.OK) == Terminal(
        AgentState.DONE, Status.OK, "PROFILES_AGREE_OK"
    )
    assert compare_terminal(Status.NG, Status.OK).status is Status.NG
    assert compare_terminal(Status.OK, Status.NG).status is Status.NG
    assert compare_terminal(Status.OK, Status.REVIEW) == Terminal(
        AgentState.REVIEW, Status.REVIEW, "PROFILE_MISMATCH"
    )
    assert compare_terminal(Status.REVIEW, Status.OK).reason == "PROFILE_MISMATCH"
    assert compare_terminal(Status.REVIEW, Status.REVIEW).reason == "PROFILES_REVIEW"


def test_corrective_profile_runs_only_once() -> None:
    engine, inspector = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
            "denoise_strong": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
        }
    )
    episode = engine.run()
    assert inspector.calls == ["base", "denoise_strong"]
    assert episode.profiles_used == 2
    assert episode.terminal_state is AgentState.REVIEW
