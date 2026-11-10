"""캐시 위에서 후보 config의 evidence·score·판정만 다시 계산해 지표를 낸다 (설계 §4.2·§4.3).

파이프라인 재실행은 없다. ``calculate_score``와 ``decide_status``는 순수
함수이므로, 캐시해 둔 컴포넌트 재료에 후보 config를 넣어 결과만 갈아끼운다.

판정 입력 중 캐시에서 오는 값은 ``quality.flags``, ``roi.uncertain``,
``refined.truncated``, ``component.touches_roi_border`` 네 가지다. 그래서
``decide_status``에 넘길 ``RoiResult``/``RefineResult``는 **값만 담은 대역**을
만든다(마스크는 쓰이지 않으므로 1x1 자리표시자다). 이 사실을 여기 적어 둔다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import AppConfig
from ..contracts import (
    ComponentFeature,
    QualityInfo,
    ReasonCode,
    Status,
)
from ..pipeline.stages import _component_evidence, calculate_score, decide_status
from ..pipeline.types import RefineResult, RoiResult
from .cache import CachedComponent, CachedSample
from .metrics import (
    Confusion,
    GtMatch,
    PixelCounts,
    Ratio,
    image_metrics,
    localization_metrics,
    metrics_to_dict,
)

_PLACEHOLDER_MASK = np.zeros((1, 1), dtype=np.uint8)


def _feature(index: int, component: CachedComponent, config: AppConfig) -> ComponentFeature:
    """캐시 값 + 후보 config로 ``ComponentFeature``를 다시 만든다.

    ``area_ratio``·``aspect_ratio``·``elongation``·``solidity``는 판정·점수에
    쓰이지 않으므로 채우지 않는다(0/1). 쓰이게 되면 캐시에 추가해야 한다.
    """

    return ComponentFeature(
        index=index,
        area_px=int(component.area_px),
        bbox=component.bbox,
        area_ratio=0.0,
        aspect_ratio=1.0,
        elongation=1.0,
        solidity=1.0,
        mean_contrast_gray=float(component.mean_contrast_gray),
        max_contrast_gray=float(component.max_contrast_gray),
        integrated_contrast=float(component.integrated_contrast),
        border_distance_px=float(component.border_distance_px),
        touches_roi_border=bool(component.touches_roi_border),
        evidence=_component_evidence(
            int(component.area_px),
            float(component.mean_contrast_gray),
            config,
        ),
    )


def _quality(sample: CachedSample) -> QualityInfo:
    return QualityInfo(
        blur_variance=sample.blur_variance,
        mean_gray=sample.mean_gray,
        saturated_ratio=sample.saturated_ratio,
        flags=tuple(ReasonCode(flag) for flag in sample.quality_flags),
    )


def _roi_proxy(sample: CachedSample) -> RoiResult:
    return RoiResult(
        mode="auto",
        mask=_PLACEHOLDER_MASK,
        bbox=(0, 0, 0, 0),
        area_px=0,
        area_ratio=1.0,
        uncertain=sample.roi_uncertain,
    )


def _refined_proxy(sample: CachedSample) -> RefineResult:
    return RefineResult(
        components=(),
        combined_mask=_PLACEHOLDER_MASK,
        truncated=sample.truncated,
        total_components=len(sample.components),
    )


@dataclass(frozen=True)
class SampleOutcome:
    """한 샘플의 후보 config 판정 결과."""

    sample_id: str
    label: str
    status: Status
    score: float
    reasons: tuple[str, ...]
    predicted: tuple[CachedComponent, ...]

    @property
    def reason_code(self) -> str:
        return self.reasons[0] if self.reasons else ""


def score_sample(sample: CachedSample, config: AppConfig) -> SampleOutcome:
    """캐시된 샘플 하나를 후보 config로 재채점한다."""

    if sample.error_code:
        return SampleOutcome(
            sample_id=sample.sample_id,
            label=sample.label,
            status=Status.ERROR,
            score=0.0,
            reasons=(),
            predicted=(),
        )
    features = tuple(
        _feature(index, component, config)
        for index, component in enumerate(sample.components)
    )
    scored = calculate_score(features, config)
    decision = decide_status(
        scored,
        _quality(sample),
        _roi_proxy(sample),
        _refined_proxy(sample),
        config,
    )
    valid_indices = {feature.index for feature in scored.valid_components}
    predicted = tuple(
        component
        for index, component in enumerate(sample.components)
        if index in valid_indices
    )
    return SampleOutcome(
        sample_id=sample.sample_id,
        label=sample.label,
        status=decision.status,
        score=float(scored.score),
        reasons=tuple(reason.value for reason in decision.reasons),
        predicted=predicted,
    )


def component_intersection(left: CachedComponent, right: CachedComponent) -> int:
    """bbox가 겹치는 구간에서 두 컴포넌트 마스크의 교집합 픽셀 수."""

    ax, ay, aw, ah = left.bbox
    bx, by, bw, bh = right.bbox
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0
    left_patch = left.mask[y0 - ay : y1 - ay, x0 - ax : x1 - ax] > 0
    right_patch = right.mask[y0 - by : y1 - by, x0 - bx : x1 - bx] > 0
    return int(np.count_nonzero(left_patch & right_patch))


def component_iou(left: CachedComponent, right: CachedComponent) -> float:
    intersection = component_intersection(left, right)
    if intersection <= 0:
        return 0.0
    union = left.area_px + right.area_px - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def _gt_matches(sample: CachedSample, outcome: SampleOutcome) -> list[GtMatch]:
    matches: list[GtMatch] = []
    for gt in sample.gt_components:
        best = 0.0
        for predicted in outcome.predicted:
            best = max(best, component_iou(gt, predicted))
        matches.append(GtMatch(iou=best, image_status=outcome.status.value))
    return matches


def _pixel_counts(sample: CachedSample, outcome: SampleOutcome) -> PixelCounts:
    gt_pixels = sample.gt_area_px
    pred_pixels = sum(component.area_px for component in outcome.predicted)
    true_positive = 0
    for predicted in outcome.predicted:
        for gt in sample.gt_components:
            true_positive += component_intersection(predicted, gt)
    return PixelCounts(
        gt_pixels=gt_pixels,
        pred_pixels=pred_pixels,
        tp_pixels=true_positive,
    )


@dataclass(frozen=True)
class VariantReport:
    """비교 집합 하나의 지표 묶음. 모든 비율에 분모가 붙는다."""

    name: str
    config_hash: str
    overrides: Mapping[str, Mapping[str, Any]]
    notes: tuple[str, ...]
    outcomes: tuple[SampleOutcome, ...]
    confusion: Confusion
    image: Mapping[str, Ratio]
    localization: Mapping[str, Ratio]
    counts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def evaluated(self) -> tuple[int, int]:
        return (
            self.confusion.evaluated("defect"),
            self.confusion.evaluated("good"),
        )

    def ratio(self, name: str) -> Ratio:
        if name in self.image:
            return self.image[name]
        return self.localization[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_hash": self.config_hash,
            "overrides": {key: dict(value) for key, value in self.overrides.items()},
            "notes": list(self.notes),
            "confusion": self.confusion.as_dict(),
            "counts": dict(self.counts),
            "image_metrics": metrics_to_dict(self.image),
            "localization_metrics": metrics_to_dict(self.localization),
        }


def summarise(
    name: str,
    samples: Sequence[CachedSample],
    outcomes: Sequence[SampleOutcome],
    *,
    config_hash: str,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    notes: Sequence[str] = (),
) -> VariantReport:
    """샘플 결과를 §4.2 이미지 지표와 §4.3 영역 지표로 집계한다."""

    if len(samples) != len(outcomes):
        raise ValueError(f"샘플 수와 결과 수가 다릅니다: {len(samples)} != {len(outcomes)}")
    by_id = {outcome.sample_id: outcome for outcome in outcomes}
    if len(by_id) != len(outcomes):
        raise ValueError("결과에 중복 sample_id가 있습니다.")

    pairs = [(outcome.label, outcome.status.value) for outcome in outcomes]
    confusion = Confusion.from_pairs(pairs)

    matches: list[GtMatch] = []
    pixels = PixelCounts()
    for sample in samples:
        if sample.label != "defect":
            continue
        outcome = by_id[sample.sample_id]
        matches.extend(_gt_matches(sample, outcome))
        pixels = pixels.merged(_pixel_counts(sample, outcome))

    image = image_metrics(confusion)
    localization = localization_metrics(matches, pixels)
    counts = {
        "samples": len(outcomes),
        "defect": confusion.total("defect"),
        "good": confusion.total("good"),
        "defect_evaluated": confusion.evaluated("defect"),
        "good_evaluated": confusion.evaluated("good"),
        "gt_components": len(matches),
        "predicted_components": sum(len(outcome.predicted) for outcome in outcomes),
        "status": {
            status: sum(1 for pair in pairs if pair[1] == status)
            for status in sorted({pair[1] for pair in pairs})
        },
    }
    return VariantReport(
        name=name,
        config_hash=config_hash,
        overrides=overrides or {},
        notes=tuple(notes),
        outcomes=tuple(outcomes),
        confusion=confusion,
        image=image,
        localization=localization,
        counts=counts,
    )


def evaluate_cached(
    samples: Sequence[CachedSample],
    config: AppConfig,
    *,
    name: str = "P",
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    notes: Sequence[str] = (),
) -> VariantReport:
    """캐시 전체를 후보 config로 재채점하고 지표를 낸다(파이프라인 재실행 없음)."""

    outcomes = [score_sample(sample, config) for sample in samples]
    return summarise(
        name,
        samples,
        outcomes,
        config_hash=config.config_hash,
        overrides=overrides,
        notes=notes,
    )


__all__ = [
    "SampleOutcome",
    "VariantReport",
    "component_intersection",
    "component_iou",
    "evaluate_cached",
    "score_sample",
    "summarise",
]
