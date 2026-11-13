"""결정론적 정책 엔진 (T09, 설계 §6).

상태 머신: ``OBSERVE → PLAN → ACT → VERIFY → PLAN | DONE | REVIEW | ERROR``.
제품 판정은 항상 파이프라인(``decide_status``)이 소유한다. 이 엔진은 제한된
도구를 선택하고 정책 표를 종료 상태에 대응시킬 뿐이며 판정을 만들거나 바꾸지
않는다. 네트워크/셸/파일 쓰기/설정 변경 도구는 존재하지 않는다.

예산 의미: 작업 도구(``run_profile``/``inspect_artifact``/``compare``)가 행동
예산을 소모하고, 종료 도구(``escalate_review``/``stop``)는 에피소드를 끝내는
전이로 기록되며 예산을 소모하지 않는다. 예산 소진은 낙관적 OK 가 아니라
``REVIEW`` 로 끝난다. NG 이후 재시도는 없다.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from ..config import AppConfig
from ..contracts import ErrorCode, SampleResult, Status
from .config import AgentConfig, PolicyConfig
from .trace import TraceWriter, hash_action, hash_args, hash_result, result_summary

WORK_TOOLS = frozenset({"run_profile", "inspect_artifact", "compare"})
TERMINAL_TOOLS = frozenset({"escalate_review", "stop"})
ALL_TOOLS = WORK_TOOLS | TERMINAL_TOOLS


class AgentState(StrEnum):
    OBSERVE = "OBSERVE"
    PLAN = "PLAN"
    ACT = "ACT"
    VERIFY = "VERIFY"
    DONE = "DONE"
    REVIEW = "REVIEW"
    ERROR = "ERROR"


TERMINAL_STATES = frozenset({AgentState.DONE, AgentState.REVIEW, AgentState.ERROR})


class AgentGuardError(RuntimeError):
    """가드(전이/예산/중복/NG 재시도/프로파일) 위반. 거부는 감사에 기록된다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ToolError(RuntimeError):
    """도구 실행 실패(캐시 없음 등). 판정을 만들지 않고 REVIEW 로 귀결된다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProfileObservation:
    """프로파일 하나의 파이프라인 판정 관찰값."""

    profile_id: str
    status: Status
    score: float
    reasons: tuple[str, ...]
    quality_flags: tuple[str, ...]
    roi_uncertain: bool
    error_code: str | None
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "status": self.status.value,
            "score": self.score,
            "reasons": list(self.reasons),
            "quality_flags": list(self.quality_flags),
            "roi_uncertain": self.roi_uncertain,
            "error_code": self.error_code,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Terminal:
    state: AgentState
    status: Status
    reason: str


Decision = ToolCall | Terminal


@dataclass(frozen=True)
class AgentEpisode:
    terminal_state: AgentState
    final_status: Status
    reason: str
    actions_used: int
    profiles_used: int
    base_latency_ms: float
    extra_latency_ms: float
    results: tuple[ProfileObservation, ...]
    rejections: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_state": self.terminal_state.value,
            "final_status": self.final_status.value,
            "reason": self.reason,
            "actions_used": self.actions_used,
            "profiles_used": self.profiles_used,
            "base_latency_ms": self.base_latency_ms,
            "extra_latency_ms": self.extra_latency_ms,
            "results": [observation.to_dict() for observation in self.results],
            "rejections": list(self.rejections),
        }


def choose_action(results: Sequence[ProfileObservation], policy: PolicyConfig) -> Decision:
    """설계 §6 정책 표. 관찰값만의 순수 함수라 결정론적이다."""

    used = {observation.profile_id for observation in results}
    if not results:
        return ToolCall("run_profile", {"profile_id": policy.base_profile})
    if len(results) == 1:
        first = results[0]
        if first.status is Status.OK:
            return ToolCall("stop", {})
        if first.status is Status.NG:
            return Terminal(AgentState.DONE, Status.NG, "PIPELINE_NG")
        if first.status is Status.ERROR:
            return Terminal(
                AgentState.ERROR,
                Status.ERROR,
                first.error_code or ErrorCode.E_PROCESS_INTERNAL.value,
            )
        if "INPUT_QUALITY_LOW" in first.reasons and policy.quality_low_profile not in used:
            return ToolCall("run_profile", {"profile_id": policy.quality_low_profile})
        if "ROI_UNCERTAIN" in first.reasons and policy.roi_uncertain_profile not in used:
            return ToolCall("run_profile", {"profile_id": policy.roi_uncertain_profile})
        reason = first.reasons[0] if first.reasons else "EVIDENCE_IN_REVIEW_BAND"
        return ToolCall("escalate_review", {"reason": reason})
    if len(results) == 2:
        if any(observation.status is Status.NG for observation in results):
            return Terminal(AgentState.DONE, Status.NG, "PIPELINE_NG")
        return ToolCall(
            "compare",
            {"profile_a": results[0].profile_id, "profile_b": results[1].profile_id},
        )
    return ToolCall("escalate_review", {"reason": "PROFILE_BUDGET"})


def compare_terminal(status_a: Status, status_b: Status) -> Terminal:
    """프로파일 쌍 → 종료 상태. 판정은 파이프라인 상태에서만 나온다."""

    if status_a is Status.NG or status_b is Status.NG:
        return Terminal(AgentState.DONE, Status.NG, "PROFILE_NG")
    if status_a is Status.OK and status_b is Status.OK:
        return Terminal(AgentState.DONE, Status.OK, "PROFILES_AGREE_OK")
    if status_a is status_b:
        return Terminal(AgentState.REVIEW, Status.REVIEW, "PROFILES_REVIEW")
    return Terminal(AgentState.REVIEW, Status.REVIEW, "PROFILE_MISMATCH")


class AgentEngine:
    """제한된 도구만 실행하는 결정론 상태 머신.

    ``inspector`` 는 하네스가 주입하는 파이프라인 호출이다. 도구는 프로파일
    레지스트리(읽기 전용)에서 선택만 하며 파일/셸/네트워크/설정 변경 능력은
    없다. 감사 기록은 ``TraceWriter`` 계층이 담당한다.
    """

    def __init__(
        self,
        *,
        agent_config: AgentConfig,
        profiles: Mapping[str, AppConfig],
        inspector: Callable[[str, AppConfig], SampleResult],
        clock: Callable[[], float] = time.perf_counter,
        trace: TraceWriter | None = None,
        profile_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self._config = agent_config
        self._profiles = dict(profiles)
        self._inspector = inspector
        self._clock = clock
        self._trace = trace if trace is not None else TraceWriter(None)
        self._profile_hashes = dict(profile_hashes) if profile_hashes is not None else None
        self._state = AgentState.OBSERVE
        self._results: dict[str, ProfileObservation] = {}
        self._summaries: dict[str, dict[str, Any]] = {}
        self._action_hashes: set[str] = set()
        self._actions_used = 0
        self._profiles_used = 0
        self._base_latency_ms = 0.0
        self._extra_latency_ms = 0.0
        self._base_seen = False
        self._ng_seen = False
        self._rejections: list[dict[str, Any]] = []
        self._episode: AgentEpisode | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    def run(
        self,
        *,
        image_path: str = "",
        label: str = "unknown",
        split: str = "tune",
    ) -> AgentEpisode:
        if self._episode is not None:
            raise AgentGuardError("INVALID_TRANSITION", "episode already finished")
        profiles_payload = (
            dict(self._profile_hashes)
            if self._profile_hashes is not None
            else {pid: cfg.config_hash for pid, cfg in sorted(self._profiles.items())}
        )
        self._emit(
            "episode_start",
            agent_config=self._config.to_dict(),
            image={"path": str(image_path), "label": label, "split": split},
            profiles=profiles_payload,
        )
        pending: ToolCall | None = None
        tool_result: dict[str, Any] | None = None
        while self._state not in TERMINAL_STATES:
            if self._state is AgentState.OBSERVE:
                self._emit("observe", results=[])
                self._state = AgentState.PLAN
            elif self._state is AgentState.PLAN:
                decision = choose_action(tuple(self._results.values()), self._config.policy)
                if isinstance(decision, Terminal):
                    self._emit(
                        "plan",
                        tool=None,
                        args={},
                        args_hash=hash_args({}),
                        terminal=self._terminal_payload(decision),
                    )
                    self._finish(decision)
                    break
                pending = decision
                self._emit(
                    "plan",
                    tool=decision.tool,
                    args=decision.args,
                    args_hash=hash_args(decision.args),
                    terminal=None,
                )
                self._state = AgentState.ACT
            elif self._state is AgentState.ACT:
                assert pending is not None
                try:
                    tool_result = self._perform(pending)
                except AgentGuardError as exc:
                    if exc.code == ErrorCode.E_PROFILE_UNKNOWN.value:
                        terminal = Terminal(AgentState.ERROR, Status.ERROR, exc.code)
                    else:
                        terminal = Terminal(AgentState.REVIEW, Status.REVIEW, exc.code)
                    self._finish(terminal)
                    break
                self._state = AgentState.VERIFY
            elif self._state is AgentState.VERIFY:
                assert pending is not None and tool_result is not None
                terminal = self._verify(pending, tool_result)
                if terminal is not None:
                    self._emit(
                        "verify",
                        tool=pending.tool,
                        next_state=terminal.state.value,
                        terminal=self._terminal_payload(terminal),
                    )
                    self._finish(terminal)
                    break
                self._emit(
                    "verify",
                    tool=pending.tool,
                    next_state=AgentState.PLAN.value,
                    terminal=None,
                )
                self._state = AgentState.PLAN
        assert self._episode is not None
        return self._episode

    def attempt(self, tool: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """외부 도구 진입점. ACT 상태가 아니면 잘못된 전이로 거부한다."""

        return self._perform(ToolCall(tool, dict(args or {})))

    def _perform(self, call: ToolCall) -> dict[str, Any]:
        args_hash = hash_args(call.args)
        action_hash = hash_action(call.tool, args_hash)
        try:
            self._guard(call)
        except AgentGuardError as exc:
            record = self._emit(
                "rejection",
                tool=call.tool,
                args=call.args,
                args_hash=args_hash,
                action_hash=action_hash,
                code=exc.code,
                message=str(exc),
            )
            self._rejections.append(record)
            raise
        started = self._clock()
        try:
            result = self._execute(call)
        except ToolError as exc:
            elapsed = round((self._clock() - started) * 1000.0, 6)
            self._account(call, elapsed)
            record = self._emit(
                "rejection",
                tool=call.tool,
                args=call.args,
                args_hash=args_hash,
                action_hash=action_hash,
                code=exc.code,
                message=str(exc),
                elapsed_ms=elapsed,
            )
            self._rejections.append(record)
            raise AgentGuardError(exc.code, str(exc)) from exc
        elapsed = round((self._clock() - started) * 1000.0, 6)
        self._account(call, elapsed)
        self._action_hashes.add(action_hash)
        self._emit(
            "act",
            tool=call.tool,
            args=call.args,
            args_hash=args_hash,
            action_hash=action_hash,
            result=result,
            result_hash=hash_result(result),
            elapsed_ms=elapsed,
        )
        if self._extra_latency_ms > self._config.budget.extra_latency_ms:
            record = self._emit(
                "rejection",
                tool=call.tool,
                args=call.args,
                args_hash=args_hash,
                action_hash=action_hash,
                code=ErrorCode.E_BUDGET_EXCEEDED.value,
                message="extra latency budget exceeded",
            )
            self._rejections.append(record)
            raise AgentGuardError(
                ErrorCode.E_BUDGET_EXCEEDED.value, "extra latency budget exceeded"
            )
        return result

    def _guard(self, call: ToolCall) -> None:
        if self._state is not AgentState.ACT:
            raise AgentGuardError(
                "INVALID_TRANSITION", f"tool call in state {self._state.value}"
            )
        if call.tool not in ALL_TOOLS:
            raise AgentGuardError("INVALID_TRANSITION", f"unknown tool {call.tool!r}")
        action_hash = hash_action(call.tool, hash_args(call.args))
        if action_hash in self._action_hashes:
            raise AgentGuardError(
                "DUPLICATE_ACTION", f"duplicate action_hash {action_hash[:12]}"
            )
        budget = self._config.budget
        if call.tool in WORK_TOOLS:
            if self._actions_used >= budget.max_actions:
                raise AgentGuardError(
                    ErrorCode.E_BUDGET_EXCEEDED.value, "action budget exhausted"
                )
            if self._extra_latency_ms >= budget.extra_latency_ms:
                raise AgentGuardError(
                    ErrorCode.E_BUDGET_EXCEEDED.value, "latency budget exhausted"
                )
        if call.tool == "run_profile":
            profile_id = call.args.get("profile_id")
            if not isinstance(profile_id, str) or profile_id not in self._profiles:
                raise AgentGuardError(
                    ErrorCode.E_PROFILE_UNKNOWN.value, f"unknown profile {profile_id!r}"
                )
            if self._profiles_used >= budget.max_profiles:
                raise AgentGuardError(
                    ErrorCode.E_BUDGET_EXCEEDED.value, "profile budget exhausted"
                )
            if self._ng_seen:
                raise AgentGuardError("RETRY_AFTER_NG", "no retry after NG")

    def _execute(self, call: ToolCall) -> dict[str, Any]:
        tool = call.tool
        if tool == "run_profile":
            profile_id = call.args["profile_id"]
            config = self._profiles[profile_id]
            result = self._inspector(profile_id, config)
            summary = result_summary(result)
            self._results[profile_id] = ProfileObservation(
                profile_id=profile_id,
                status=result.status,
                score=float(result.score),
                reasons=tuple(reason.value for reason in result.reasons),
                quality_flags=(
                    tuple(flag.value for flag in result.quality.flags)
                    if result.quality is not None
                    else ()
                ),
                roi_uncertain=bool(result.roi.uncertain) if result.roi is not None else False,
                error_code=(
                    result.error_code.value if result.error_code is not None else None
                ),
                elapsed_ms=0.0,
            )
            self._summaries[profile_id] = summary
            if result.status is Status.NG:
                self._ng_seen = True
            return {"profile_id": profile_id, **summary}
        if tool == "inspect_artifact":
            profile_id = call.args.get("profile_id")
            if profile_id not in self._summaries:
                raise ToolError(
                    ErrorCode.E_TOOL_FAILED.value, f"no cached artifact for {profile_id!r}"
                )
            return {"profile_id": profile_id, **self._summaries[profile_id]}
        if tool == "compare":
            profile_a = call.args.get("profile_a")
            profile_b = call.args.get("profile_b")
            if profile_a not in self._summaries or profile_b not in self._summaries:
                raise ToolError(
                    ErrorCode.E_TOOL_FAILED.value, "compare requires two cached profiles"
                )
            status_a = self._summaries[profile_a]["status"]
            status_b = self._summaries[profile_b]["status"]
            return {
                "profile_a": profile_a,
                "profile_b": profile_b,
                "status_a": status_a,
                "status_b": status_b,
                "agree": status_a == status_b,
            }
        if tool == "escalate_review":
            return {"terminal": "REVIEW", "reason": str(call.args.get("reason", ""))}
        if tool == "stop":
            if not self._summaries:
                raise ToolError(
                    ErrorCode.E_TOOL_FAILED.value, "stop requires a pipeline result"
                )
            statuses = {summary["status"] for summary in self._summaries.values()}
            if "NG" in statuses:
                final = "NG"
            elif statuses == {"OK"}:
                final = "OK"
            else:
                raise ToolError(
                    ErrorCode.E_TOOL_FAILED.value, "stop cannot confirm a verdict"
                )
            return {"terminal": "DONE", "status": final}
        raise ToolError(ErrorCode.E_TOOL_FAILED.value, f"unknown tool {tool!r}")

    def _account(self, call: ToolCall, elapsed_ms: float) -> None:
        if call.tool in WORK_TOOLS:
            self._actions_used += 1
        if call.tool == "run_profile":
            self._profiles_used += 1
            profile_id = call.args.get("profile_id")
            if profile_id in self._results:
                self._results[profile_id] = replace(
                    self._results[profile_id], elapsed_ms=elapsed_ms
                )
        if call.tool == "run_profile" and not self._base_seen:
            self._base_latency_ms = elapsed_ms
            self._base_seen = True
        else:
            self._extra_latency_ms = round(self._extra_latency_ms + elapsed_ms, 6)

    def _verify(self, call: ToolCall, result: Mapping[str, Any]) -> Terminal | None:
        if call.tool == "run_profile":
            status = Status(result["status"])
            if status is Status.NG:
                return Terminal(AgentState.DONE, Status.NG, "PIPELINE_NG")
            if status is Status.ERROR:
                return Terminal(
                    AgentState.ERROR,
                    Status.ERROR,
                    str(result.get("error_code") or ErrorCode.E_PROCESS_INTERNAL.value),
                )
            return None
        if call.tool == "compare":
            return compare_terminal(Status(result["status_a"]), Status(result["status_b"]))
        if call.tool == "escalate_review":
            return Terminal(
                AgentState.REVIEW, Status.REVIEW, str(result.get("reason") or "ESCALATED")
            )
        if call.tool == "stop":
            return Terminal(AgentState.DONE, Status(result["status"]), "STOP_CONFIRMED")
        return None

    def _finish(self, terminal: Terminal) -> None:
        self._state = terminal.state
        self._emit(
            "episode_end",
            terminal_state=terminal.state.value,
            final_status=terminal.status.value,
            reason=terminal.reason,
            actions_used=self._actions_used,
            profiles_used=self._profiles_used,
            base_latency_ms=self._base_latency_ms,
            extra_latency_ms=self._extra_latency_ms,
            rejections=len(self._rejections),
        )
        self._episode = AgentEpisode(
            terminal_state=terminal.state,
            final_status=terminal.status,
            reason=terminal.reason,
            actions_used=self._actions_used,
            profiles_used=self._profiles_used,
            base_latency_ms=self._base_latency_ms,
            extra_latency_ms=self._extra_latency_ms,
            results=tuple(self._results.values()),
            rejections=tuple(self._rejections),
            events=self._trace.events,
        )

    def _terminal_payload(self, terminal: Terminal) -> dict[str, str]:
        return {
            "state": terminal.state.value,
            "status": terminal.status.value,
            "reason": terminal.reason,
        }

    def _budget_snapshot(self) -> dict[str, Any]:
        budget = self._config.budget
        return {
            "actions_used": self._actions_used,
            "actions_max": budget.max_actions,
            "profiles_used": self._profiles_used,
            "profiles_max": budget.max_profiles,
            "extra_latency_ms": self._extra_latency_ms,
            "extra_latency_ms_max": budget.extra_latency_ms,
        }

    def _emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "state": self._state.value,
            "event": event,
            "budget": self._budget_snapshot(),
            **payload,
        }
        return self._trace.append(record)


__all__ = [
    "ALL_TOOLS",
    "TERMINAL_STATES",
    "TERMINAL_TOOLS",
    "WORK_TOOLS",
    "AgentEngine",
    "AgentEpisode",
    "AgentGuardError",
    "AgentState",
    "Decision",
    "ProfileObservation",
    "Terminal",
    "ToolCall",
    "ToolError",
    "choose_action",
    "compare_terminal",
]
