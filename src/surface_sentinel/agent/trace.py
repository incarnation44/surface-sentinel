"""에이전트 실행 단계를 해시 체인으로 기록하는 append-only 감사 로그 및 결정론적 리플레이 모듈.

감사 이벤트는 ``args_hash`` / ``action_hash`` / ``result_hash`` 로 무결성을
검증한다. replay 는 기록된 결과만으로 정책 엔진을 재실행해 동일한 행동 순서와
최종 판정을 검증한다. 도구는 파일을 쓰지 않으며 이 감사 계층만
``agent_trace.jsonl`` 을 append-only 로 기록한다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import (
    RESULT_SCHEMA_VERSION,
    ComponentFeature,
    EngineObservation,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Stage,
    Status,
)

TRACE_SCHEMA_VERSION = "agent-trace-v1"
TERMINAL_STATES = ("DONE", "REVIEW", "ERROR")


class TraceError(RuntimeError):
    """감사 로그를 읽거나 해석할 수 없을 때 발생한다."""


def canonical_json(value: Any) -> str:
    """키 정렬·구분자 고정 직렬화. 해시와 replay 비교의 기준이다."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_args(args: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(args)))


def hash_action(tool: str, args_hash: str) -> str:
    """중복 차단 키. 같은 도구+같은 인자는 같은 해시다."""

    return sha256_text(f"{tool}:{args_hash}")


def hash_result(result: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(result)))


def result_summary(result: SampleResult) -> dict[str, Any]:
    """정책이 사용하는 판정 요약. 파이프라인 결과를 그대로 옮긴다(판정 생성 없음)."""

    return {
        "status": result.status.value,
        "score": float(result.score),
        "reasons": [reason.value for reason in result.reasons],
        "quality_flags": (
            [flag.value for flag in result.quality.flags] if result.quality is not None else []
        ),
        "roi_uncertain": bool(result.roi.uncertain) if result.roi is not None else False,
        "error_code": result.error_code.value if result.error_code is not None else None,
        "config_hash": result.config_hash,
        "input_sha256": result.input_sha256,
    }


def rebuild_result(summary: Mapping[str, Any]) -> SampleResult:
    """기록된 요약에서 파이프라인 결과를 재구성한다(replay 전용)."""

    return SampleResult(
        run_id="replay",
        sample_id="replay",
        original_id="replay",
        split="tune",
        label="unknown",
        status=Status(summary["status"]),
        score=float(summary["score"]),
        reasons=tuple(ReasonCode(value) for value in summary["reasons"]),
        quality=QualityInfo(
            blur_variance=0.0,
            mean_gray=0.0,
            saturated_ratio=0.0,
            flags=tuple(ReasonCode(value) for value in summary["quality_flags"]),
        ),
        roi=RoiInfo(
            mode="replay",
            bbox=(0, 0, 0, 0),
            area_px=0,
            area_ratio=0.0,
            uncertain=bool(summary["roi_uncertain"]),
        ),
        error_code=ErrorCode(summary["error_code"]) if summary["error_code"] else None,
        config_hash=str(summary["config_hash"]),
        input_sha256=str(summary["input_sha256"]),
    )


def rebuild_observation(payload: Mapping[str, Any]) -> EngineObservation:
    """기록된 엔진 관찰 원본을 재구성한다(replay 전용, 컴포넌트 포함 전체 충실도).

    ``rebuild_result`` 가 판정 요약만 복원하는 것과 달리, 교차 증거 재계산에
    필요한 컴포넌트·ROI 까지 복원한다. 기록은 ``EngineObservation.to_dict()``
    왕복이므로 해시가 보존된다.
    """

    data = dict(payload["result"])
    quality = data.get("quality")
    roi = data.get("roi")
    result = SampleResult(
        run_id=str(data["run_id"]),
        sample_id=str(data["sample_id"]),
        original_id=str(data["original_id"]),
        split=str(data["split"]),
        label=str(data["label"]),
        status=Status(data["status"]),
        score=float(data["score"]),
        reasons=tuple(ReasonCode(value) for value in data.get("reasons", [])),
        components=tuple(
            ComponentFeature(**{**component, "bbox": tuple(component["bbox"])})
            for component in data.get("components", [])
        ),
        quality=(
            QualityInfo(
                blur_variance=float(quality["blur_variance"]),
                mean_gray=float(quality["mean_gray"]),
                saturated_ratio=float(quality["saturated_ratio"]),
                flags=tuple(ReasonCode(value) for value in quality.get("flags", [])),
            )
            if quality is not None
            else None
        ),
        roi=(
            RoiInfo(
                mode=str(roi["mode"]),
                bbox=tuple(roi["bbox"]),
                area_px=int(roi["area_px"]),
                area_ratio=float(roi["area_ratio"]),
                uncertain=bool(roi["uncertain"]),
            )
            if roi is not None
            else None
        ),
        timing_ms={str(key): float(value) for key, value in data.get("timing_ms", {}).items()},
        error_code=ErrorCode(data["error_code"]) if data.get("error_code") else None,
        error_stage=Stage(data["error_stage"]) if data.get("error_stage") else None,
        input_sha256=str(data.get("input_sha256", "")),
        mask_sha256=str(data.get("mask_sha256", "")),
        overlay_path=str(data.get("overlay_path", "")),
        config_hash=str(data.get("config_hash", "")),
        schema_version=str(data.get("schema_version", RESULT_SCHEMA_VERSION)),
    )
    return EngineObservation(
        engine=str(payload["engine"]), result=result, task=str(payload.get("task", ""))
    )


class TraceWriter:
    """append-only JSONL 기록기. seq 만 부여하며 과거 이벤트를 수정하지 않는다."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._events: list[dict[str, Any]] = []
        self._file = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8", newline="\n")

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        record = {"seq": len(self._events), **dict(event)}
        self._events.append(record)
        if self._file is not None:
            self._file.write(canonical_json(record) + "\n")
            self._file.flush()
        return record

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TraceError(f"cannot read trace: {trace_path}") from exc
    events: list[dict[str, Any]] = []
    for number, line in enumerate(lines):
        if not line.strip():
            raise TraceError(f"empty line at {number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceError(f"invalid JSON at line {number}: {exc}") from exc
        if not isinstance(event, dict):
            raise TraceError(f"line {number} is not an object")
        events.append(event)
    return events


def audit_trace(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """감사 완전성 검사. 오류 목록이 비면 완전하다."""

    errors: list[str] = []
    if not events:
        return ["empty trace"]
    if events[0].get("event") != "episode_start":
        errors.append("first event is not episode_start")
    if events[-1].get("event") != "episode_end":
        errors.append("last event is not episode_end")

    prev_actions = -1
    prev_profiles = -1
    for index, event in enumerate(events):
        if event.get("seq") != index:
            errors.append(f"seq gap at index {index}")
        budget = event.get("budget")
        if not isinstance(budget, dict):
            errors.append(f"budget missing at index {index}")
        else:
            try:
                actions = int(budget.get("actions_used", 0))
                profiles = int(budget.get("profiles_used", 0))
            except (TypeError, ValueError):
                errors.append(f"budget malformed at index {index}")
            else:
                if actions < prev_actions or profiles < prev_profiles:
                    errors.append(f"budget decreased at index {index}")
                prev_actions, prev_profiles = actions, profiles
        kind = event.get("event")
        if kind in {"act", "plan", "rejection"} and "args" in event:
            if not isinstance(event["args"], Mapping):
                errors.append(f"args not an object at index {index}")
            elif event.get("args_hash") != hash_args(event["args"]):
                errors.append(f"args_hash mismatch at index {index}")
        if kind == "act":
            expected_action = hash_action(str(event.get("tool")), str(event.get("args_hash")))
            if event.get("action_hash") != expected_action:
                errors.append(f"action_hash mismatch at index {index}")
            if "result" in event:
                if not isinstance(event["result"], Mapping):
                    errors.append(f"result not an object at index {index}")
                elif event.get("result_hash") != hash_result(event["result"]):
                    errors.append(f"result_hash mismatch at index {index}")

    end = events[-1]
    terminal = end.get("terminal_state")
    if terminal not in TERMINAL_STATES:
        errors.append(f"invalid terminal_state: {terminal!r}")
    else:
        allowed = {
            "DONE": ("OK", "NG"),
            "REVIEW": ("REVIEW",),
            "ERROR": ("ERROR",),
        }[terminal]
        if end.get("final_status") not in allowed:
            errors.append(f"final_status {end.get('final_status')!r} inconsistent with {terminal}")
    return errors


@dataclass(frozen=True)
class ReplayReport:
    deterministic: bool
    audit_valid: bool
    events_compared: int
    terminal_state: str
    final_status: str
    mismatches: tuple[str, ...]
    audit_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "audit_valid": self.audit_valid,
            "events_compared": self.events_compared,
            "terminal_state": self.terminal_state,
            "final_status": self.final_status,
            "mismatches": list(self.mismatches),
            "audit_errors": list(self.audit_errors),
        }


class _ReplayClock:
    """기록된 elapsed_ms 순서대로 시간을 재생하는 결정론 시계."""

    def __init__(self, elapsed_ms: Sequence[float]) -> None:
        self._times: list[float] = []
        cumulative = 0.0
        for elapsed in elapsed_ms:
            self._times.append(cumulative / 1000.0)
            cumulative += elapsed
            self._times.append(cumulative / 1000.0)
        self._index = 0

    def __call__(self) -> float:
        if self._index >= len(self._times):
            raise TraceError("replay clock exhausted (action count differs)")
        value = self._times[self._index]
        self._index += 1
        return value


def replay_trace(events: Sequence[Mapping[str, Any]]) -> ReplayReport:
    """기록된 결과만으로 엔진을 재실행해 결정론을 검증한다.

    비교에서 ``elapsed_ms`` 만 제외한다(타이밍은 실행 환경 값). 예산 검사에는
    기록된 elapsed 를 그대로 재생하므로 지연 예산 초과도 동일하게 재현된다.
    """

    from ..config import default_config
    from .config import parse_agent_config
    from .engine import AgentEngine

    audit_errors = audit_trace(events)
    if not events or events[0].get("event") != "episode_start":
        return ReplayReport(
            deterministic=False,
            audit_valid=False,
            events_compared=0,
            terminal_state="",
            final_status="",
            mismatches=("missing episode_start",),
            audit_errors=tuple(audit_errors),
        )

    start = events[0]
    try:
        agent_config = parse_agent_config(start["agent_config"])
    except (KeyError, TypeError, ValueError) as exc:
        return ReplayReport(
            deterministic=False,
            audit_valid=not audit_errors,
            events_compared=0,
            terminal_state="",
            final_status="",
            mismatches=(f"invalid episode_start agent_config: {exc}",),
            audit_errors=tuple(audit_errors),
        )
    try:
        image = start.get("image", {})
        if not isinstance(image, Mapping):
            raise ValueError("episode_start image must be an object")
        profile_hashes = dict(start.get("profiles", {}))
        observations = tuple(
            rebuild_observation(item) for item in start.get("engine_observations", [])
        )
        arbitration_available = bool(start.get("arbitration_available", False))
        recorded: dict[str, dict[str, Any]] = {}
        recorded_arbitration: dict[str, Any] | None = None
        arbitration_tampered = False
        elapsed: list[float] = []
        for event in events:
            if event.get("event") != "act":
                continue
            elapsed.append(float(event.get("elapsed_ms", 0.0)))
            if event.get("tool") == "run_profile":
                args = event.get("args")
                result = event.get("result")
                if not isinstance(args, Mapping) or not isinstance(result, Mapping):
                    raise ValueError("run_profile act requires object args/result")
                recorded[str(args["profile_id"])] = dict(result)
            elif event.get("tool") == "arbitrate_patches":
                result = event.get("result")
                if not isinstance(result, Mapping):
                    raise ValueError("arbitrate_patches act requires object result")
                arb_payload = dict(result)
                expected_arb = sha256_text(
                    canonical_json(
                        {k: v for k, v in arb_payload.items() if k != "arbitration_hash"}
                    )
                )
                if arb_payload.get("arbitration_hash") != expected_arb:
                    arbitration_tampered = True
                recorded_arbitration = arb_payload
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return ReplayReport(
            deterministic=False,
            audit_valid=not audit_errors,
            events_compared=0,
            terminal_state="",
            final_status="",
            mismatches=(f"malformed trace event: {exc}",),
            audit_errors=tuple(audit_errors),
        )

    def inspector(profile_id: str, _config: Any) -> SampleResult:
        if profile_id not in recorded:
            raise TraceError(f"no recorded result for profile {profile_id!r}")
        return rebuild_result(recorded[profile_id])

    registry = {profile_id: default_config() for profile_id in profile_hashes}
    writer = TraceWriter(None)
    engine = AgentEngine(
        agent_config=agent_config,
        profiles=registry,
        inspector=inspector,
        clock=_ReplayClock(elapsed),
        trace=writer,
        profile_hashes=profile_hashes,
        observations=observations,
        arbitration_available=arbitration_available,
        recorded_arbitration=recorded_arbitration,
    )
    try:
        engine.run(
            image_path=str(image.get("path", "")),
            label=str(image.get("label", "unknown")),
            split=str(image.get("split", "tune")),
        )
    except (TraceError, RuntimeError) as exc:
        return ReplayReport(
            deterministic=False,
            audit_valid=not audit_errors,
            events_compared=0,
            terminal_state="",
            final_status="",
            mismatches=(f"replay failed: {exc}",),
            audit_errors=tuple(audit_errors),
        )

    generated = writer.events
    mismatches: list[str] = []
    if arbitration_tampered:
        mismatches.append("arbitrate_patches: arbitration_hash mismatch")
    for index in range(max(len(generated), len(events))):
        if index >= len(generated):
            mismatches.append(f"event {index}: missing in replay")
            continue
        if index >= len(events):
            mismatches.append(f"event {index}: extra in replay")
            continue
        recorded_event = {k: v for k, v in events[index].items() if k != "elapsed_ms"}
        replayed_event = {k: v for k, v in generated[index].items() if k != "elapsed_ms"}
        if canonical_json(recorded_event) != canonical_json(replayed_event):
            mismatches.append(
                f"event {index} ({events[index].get('event')}): content mismatch"
            )

    end = generated[-1] if generated else {}
    return ReplayReport(
        deterministic=not mismatches and not audit_errors,
        audit_valid=not audit_errors,
        events_compared=min(len(generated), len(events)),
        terminal_state=str(end.get("terminal_state", "")),
        final_status=str(end.get("final_status", "")),
        mismatches=tuple(mismatches),
        audit_errors=tuple(audit_errors),
    )


__all__ = [
    "TERMINAL_STATES",
    "TRACE_SCHEMA_VERSION",
    "ReplayReport",
    "TraceError",
    "TraceWriter",
    "audit_trace",
    "canonical_json",
    "hash_action",
    "hash_args",
    "hash_result",
    "load_trace",
    "rebuild_observation",
    "rebuild_result",
    "replay_trace",
    "result_summary",
    "sha256_text",
]
