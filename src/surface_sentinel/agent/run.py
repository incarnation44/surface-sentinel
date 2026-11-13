"""에이전트 실행 하네스 (T09 + Phase 1 교차 증거).

프로파일은 시작 시 읽기 전용으로 로드되며 도구는 레지스트리에서 선택만 한다.
run 디렉터리와 감사 로그는 이 하네스가 열고, 도구 자체에는 파일 쓰기 능력이
없다. 파이프라인의 증거 저장(오버레이)은 파이프라인 소유이며 하네스가 지정한
run 디렉터리 아래로만 향한다. 봉인 분할(test/official_test)은 절대 열지 않는다.

``run_evidence_episode``(Phase 1)는 이미 계산된 결합 판정과 엔진 관찰 2개를
주입받아 교차 증거만 기록한다 — 재검사 없이 캐시 판정을 확인하고, dual
overlay·결과 JSON·트레이스를 남긴다. 판정은 항상 결합 결과 그대로다.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from ..config import AppConfig, load_config
from ..contracts import EngineObservation, SampleResult
from ..pipeline import inspect_image
from ..pipeline.stages import SEALED_SPLITS
from .config import AgentConfig
from .engine import AgentEngine, AgentEpisode
from .evidence import CrossEvidenceAssessment, save_dual_overlay
from .trace import TraceWriter

AGENT_RUN_SCHEMA_VERSION = "agent-run-v1"
AGENT_EVIDENCE_RUN_SCHEMA_VERSION = "agent-evidence-run-v1"
TRACE_FILENAME = "agent_trace.jsonl"
RESULT_FILENAME = "agent_result.json"
EVIDENCE_RESULT_FILENAME = "agent_evidence_result.json"
DUAL_OVERLAY_FILENAME = "dual_overlay.png"


class AgentRunError(RuntimeError):
    """실행 전제(봉인/입력/설정) 위반."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentRunOutcome:
    episode: AgentEpisode
    run_id: str
    run_dir: Path
    trace_path: Path
    result_path: Path


@dataclass(frozen=True)
class EvidenceRunOutcome:
    """교차 증거 에피소드 산출물(Phase 1). 판정은 결합 결과 그대로다."""

    episode: AgentEpisode
    assessment: CrossEvidenceAssessment
    run_id: str
    run_dir: Path
    trace_path: Path
    result_path: Path
    overlay_path: Path


def load_profiles(
    base_config_path: str | Path,
    profiles_dir: str | Path,
) -> dict[str, AppConfig]:
    """base + ``profiles/*.toml`` 을 이름순으로 로드한 읽기 전용 레지스트리."""

    registry: dict[str, AppConfig] = {"base": load_config(base_config_path)}
    directory = Path(profiles_dir)
    if directory.is_dir():
        for path in sorted(directory.glob("*.toml")):
            registry[path.stem] = load_config(path)
    return registry


def new_run_id(config_hash8: str, split: str, *, now: datetime | None = None) -> str:
    """설계 §5.2 형식: ``YYYYMMDDTHHMMSSZ-<config_hash8>-<split>``."""

    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash8}-{split}"


def run_agent_episode(
    image_path: str | Path,
    *,
    agent_config: AgentConfig,
    profiles: Mapping[str, AppConfig],
    output_dir: str | Path,
    label: str = "unknown",
    split: str = "tune",
    mask_path: str | Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
) -> AgentRunOutcome:
    """이미지 하나에 대해 에이전트 에피소드를 실행하고 run 디렉터리를 남긴다."""

    if split in SEALED_SPLITS:
        raise AgentRunError(
            "E_SEALED_DENIED", f"sealed split {split!r} is never opened by the agent"
        )
    image = Path(image_path)
    if not image.is_file():
        raise AgentRunError("E_INPUT_MISSING", f"image not found: {image}")
    base_config = profiles.get("base")
    if base_config is None:
        raise AgentRunError("E_CONFIG_INVALID", "profile registry must contain 'base'")

    base_run_id = run_id or new_run_id(base_config.config_hash[:8], split)
    candidate = base_run_id
    suffix = 1
    while (Path(output_dir) / candidate).exists():
        candidate = f"{base_run_id}-{suffix:02d}"
        suffix += 1
    run_dir = Path(output_dir) / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / TRACE_FILENAME
    trace = TraceWriter(trace_path)

    def inspector(profile_id: str, config: AppConfig) -> Any:
        outcome = inspect_image(
            image,
            config,
            mask_path=mask_path,
            label=label,
            split=split,
            output_dir=run_dir / "artifacts" / profile_id,
            save_stages=False,
        )
        return outcome.result

    engine = AgentEngine(
        agent_config=agent_config,
        profiles=profiles,
        inspector=inspector,
        clock=clock,
        trace=trace,
    )
    try:
        episode = engine.run(image_path=str(image), label=label, split=split)
    finally:
        trace.close()

    payload = {
        "schema_version": AGENT_RUN_SCHEMA_VERSION,
        "run_id": candidate,
        "image_path": str(image),
        "mask_path": str(mask_path) if mask_path is not None else "",
        "label": label,
        "split": split,
        "config_hash": base_config.config_hash,
        "agent_config": agent_config.to_dict(),
        **episode.to_dict(),
        "trace_path": str(trace_path),
    }
    result_path = run_dir / RESULT_FILENAME
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, result_path)
    return AgentRunOutcome(
        episode=episode,
        run_id=candidate,
        run_dir=run_dir,
        trace_path=trace_path,
        result_path=result_path,
    )


def run_evidence_episode(
    image_path: str | Path,
    *,
    combined_result: SampleResult,
    observations: Sequence[EngineObservation],
    agent_config: AgentConfig,
    profiles: Mapping[str, AppConfig],
    output_dir: str | Path,
    label: str = "unknown",
    split: str = "tune",
    clock: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
) -> EvidenceRunOutcome:
    """교차 증거 에피소드를 실행하고 run 디렉터리를 남긴다(Phase 1, observe-only).

    이미 계산된 결합 판정(``combined_result``)과 엔진 관찰 2개를 주입받는다.
    재검사는 하지 않는다 — ``run_profile`` 은 캐시된 결합 결과를 확인만 하고,
    종료 상태는 결합 판정을 그대로 따른다. dual overlay·결과 JSON·트레이스를
    남기며, 봉인 분할은 절대 열지 않는다.
    """

    if split in SEALED_SPLITS:
        raise AgentRunError(
            "E_SEALED_DENIED", f"sealed split {split!r} is never opened by the agent"
        )
    image = Path(image_path)
    if not image.is_file():
        raise AgentRunError("E_INPUT_MISSING", f"image not found: {image}")
    if len(observations) < 2:
        raise AgentRunError(
            "E_EVIDENCE_INPUT", "cross evidence requires two engine observations"
        )
    base_config = profiles.get("base")
    if base_config is None:
        raise AgentRunError("E_CONFIG_INVALID", "profile registry must contain 'base'")

    base_run_id = run_id or new_run_id(base_config.config_hash[:8], split)
    candidate = base_run_id
    suffix = 1
    while (Path(output_dir) / candidate).exists():
        candidate = f"{base_run_id}-{suffix:02d}"
        suffix += 1
    run_dir = Path(output_dir) / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / TRACE_FILENAME
    trace = TraceWriter(trace_path)

    def inspector(profile_id: str, config: AppConfig) -> Any:
        # 재검사 없음: 이미 계산된 결합 판정을 확인만 한다(판정 소유권 불변).
        return combined_result

    engine = AgentEngine(
        agent_config=agent_config,
        profiles=profiles,
        inspector=inspector,
        clock=clock,
        trace=trace,
        observations=tuple(observations),
    )
    try:
        episode = engine.run(image_path=str(image), label=label, split=split)
    finally:
        trace.close()
    assessment = engine.assessment
    assert assessment is not None  # 관찰 2개 주입 시 cross_evidence 가 반드시 돈다

    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise AgentRunError("E_INPUT_MISSING", f"image not readable: {image}")
    overlay_path = save_dual_overlay(
        bgr, observations[0], observations[1], assessment, run_dir / DUAL_OVERLAY_FILENAME
    )

    payload = {
        "schema_version": AGENT_EVIDENCE_RUN_SCHEMA_VERSION,
        "run_id": candidate,
        "image_path": str(image),
        "label": label,
        "split": split,
        "config_hash": base_config.config_hash,
        "agent_config": agent_config.to_dict(),
        "combined_status": combined_result.status.value,
        "assessment": assessment.to_dict(),
        **episode.to_dict(),
        "trace_path": str(trace_path),
        "overlay_path": str(overlay_path),
    }
    result_path = run_dir / EVIDENCE_RESULT_FILENAME
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, result_path)
    return EvidenceRunOutcome(
        episode=episode,
        assessment=assessment,
        run_id=candidate,
        run_dir=run_dir,
        trace_path=trace_path,
        result_path=result_path,
        overlay_path=overlay_path,
    )


__all__ = [
    "AGENT_EVIDENCE_RUN_SCHEMA_VERSION",
    "AGENT_RUN_SCHEMA_VERSION",
    "DUAL_OVERLAY_FILENAME",
    "EVIDENCE_RESULT_FILENAME",
    "RESULT_FILENAME",
    "TRACE_FILENAME",
    "AgentRunError",
    "AgentRunOutcome",
    "EvidenceRunOutcome",
    "load_profiles",
    "new_run_id",
    "run_agent_episode",
    "run_evidence_episode",
]
