"""에이전트 테스트 공용 가짜 검사기(`FakeInspector`) 및 합성 관찰·결과 빌더.

실제 KSDD2 이미지는 사용하지 않는다. pytest가 수집하지 않는 이름이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from surface_sentinel.agent.config import AgentConfig, default_agent_config
from surface_sentinel.agent.engine import AgentEngine, ProfileObservation
from surface_sentinel.config import AppConfig, default_config
from surface_sentinel.contracts import (
    ComponentFeature,
    EngineObservation,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Status,
)


def make_result(
    status: Status,
    *,
    reasons: Sequence[str] = (),
    quality_flags: Sequence[str] = (),
    roi_uncertain: bool = False,
    error_code: str | None = None,
    score: float = 0.0,
) -> SampleResult:
    return SampleResult(
        run_id="test",
        sample_id="test",
        original_id="test",
        split="tune",
        label="unknown",
        status=status,
        score=score,
        reasons=tuple(ReasonCode(value) for value in reasons),
        quality=QualityInfo(
            blur_variance=100.0,
            mean_gray=120.0,
            saturated_ratio=0.0,
            flags=tuple(ReasonCode(value) for value in quality_flags),
        ),
        roi=RoiInfo(
            mode="auto",
            bbox=(0, 0, 10, 10),
            area_px=100,
            area_ratio=0.5,
            uncertain=roi_uncertain,
        ),
        error_code=ErrorCode(error_code) if error_code else None,
        config_hash="test-config",
        input_sha256="test-sha",
    )


class FakeInspector:
    def __init__(self, by_profile: dict[str, SampleResult]) -> None:
        self.by_profile = by_profile
        self.calls: list[str] = []

    def __call__(self, profile_id: str, config: AppConfig) -> SampleResult:
        self.calls.append(profile_id)
        return self.by_profile[profile_id]


def make_engine(
    by_profile: dict[str, SampleResult],
    *,
    agent_config: AgentConfig | None = None,
    clock=None,
    trace=None,
) -> tuple[AgentEngine, FakeInspector]:
    inspector = FakeInspector(by_profile)
    profiles = {profile_id: default_config() for profile_id in by_profile}
    engine = AgentEngine(
        agent_config=agent_config or default_agent_config(),
        profiles=profiles,
        inspector=inspector,
        clock=clock or (lambda: 0.0),
        trace=trace,
    )
    return engine, inspector


def make_component(
    index: int,
    bbox: tuple[int, int, int, int],
    *,
    area_px: int | None = None,
    evidence: float = 1.0,
) -> ComponentFeature:
    """교차 증거 테스트용 컴포넌트. bbox 는 (x, y, w, h) 계약이다."""

    x, y, w, h = bbox
    area = w * h if area_px is None else area_px
    return ComponentFeature(
        index=index,
        area_px=area,
        bbox=bbox,
        area_ratio=0.01,
        aspect_ratio=(w / h) if h else 0.0,
        elongation=1.0,
        solidity=1.0,
        mean_contrast_gray=50.0,
        max_contrast_gray=80.0,
        integrated_contrast=area * 50.0,
        border_distance_px=10.0,
        touches_roi_border=False,
        evidence=evidence,
    )


def make_observation(
    engine: str,
    status: Status,
    *,
    components: Sequence[ComponentFeature] = (),
    score: float = 0.0,
    reasons: Sequence[str] = (),
    task: str = "",
    roi: RoiInfo | None = None,
    error_code: str | None = None,
) -> EngineObservation:
    """교차 증거 테스트용 엔진 관찰. 실제 검사 없이 계약 모양만 만든다."""

    result = make_result(status, reasons=reasons, score=score, error_code=error_code)
    result = replace(
        result,
        components=tuple(components),
        roi=roi if roi is not None else result.roi,
    )
    return EngineObservation(engine=engine, result=result, task=task)


def observation(
    status: Status,
    *,
    reasons: Sequence[str] = (),
    profile_id: str = "base",
) -> ProfileObservation:
    return ProfileObservation(
        profile_id=profile_id,
        status=status,
        score=0.0,
        reasons=tuple(reasons),
        quality_flags=(),
        roi_uncertain=False,
        error_code=None,
        elapsed_ms=0.0,
    )
