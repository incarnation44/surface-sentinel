"""10단계 OpenCV 파이프라인 개별 단계 함수 단위 테스트.

자동/고정 ROI, 전처리 채널 토글, 배경 커널, 화질 플래그, 후보 정제 경계 조건을 검증한다.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

import synthetic as syn
from surface_sentinel.config import REVIEW_BAND_MAX, AppConfig, default_config
from surface_sentinel.contracts import (
    ComponentFeature,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    Stage,
    Status,
)
from surface_sentinel.pipeline import stages
from surface_sentinel.pipeline.errors import PipelineError
from surface_sentinel.pipeline.types import (
    RefinedComponent,
    RefineResult,
    RoiResult,
    ScoreResult,
)


def _surface_config(**roi_changes: object) -> AppConfig:
    base = default_config()
    return replace(base, roi=replace(base.roi, **roi_changes))


def _roi_result(
    mask: np.ndarray,
    *,
    area_px: int | None = None,
    uncertain: bool = False,
) -> RoiResult:
    count = int(np.count_nonzero(mask)) if area_px is None else area_px
    height, width = mask.shape[:2]
    return RoiResult(
        mode="fixed",
        mask=mask,
        bbox=(0, 0, width, height),
        area_px=count,
        area_ratio=count / float(height * width) if height * width else 0.0,
        uncertain=uncertain,
    )


def _component(
    index: int,
    area_px: int,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    mean_contrast: float,
    *,
    max_contrast: float | None = None,
    border_distance: float = 10.0,
    touches: bool = False,
) -> RefinedComponent:
    config = default_config()
    return RefinedComponent(
        index=index,
        area_px=area_px,
        bbox=bbox,
        mask=mask,
        mean_contrast_gray=mean_contrast,
        max_contrast_gray=mean_contrast if max_contrast is None else max_contrast,
        integrated_contrast=float(area_px) * mean_contrast,
        border_distance_px=border_distance,
        touches_roi_border=touches,
        evidence=stages._component_evidence(area_px, mean_contrast, config),
    )


def _feature(
    index: int = 0,
    *,
    area_px: int = 60,
    mean_contrast: float = 20.0,
    evidence: float | None = None,
    touches: bool = False,
) -> ComponentFeature:
    config = default_config()
    value = (
        stages._component_evidence(area_px, mean_contrast, config)
        if evidence is None
        else evidence
    )
    return ComponentFeature(
        index=index,
        area_px=area_px,
        bbox=(0, 0, 8, 8),
        area_ratio=0.01,
        aspect_ratio=1.0,
        elongation=1.0,
        solidity=1.0,
        mean_contrast_gray=mean_contrast,
        max_contrast_gray=mean_contrast,
        integrated_contrast=float(area_px) * mean_contrast,
        border_distance_px=5.0,
        touches_roi_border=touches,
        evidence=value,
    )


def _quality(flags: tuple[ReasonCode, ...] = ()) -> QualityInfo:
    return QualityInfo(
        blur_variance=100.0,
        mean_gray=120.0,
        saturated_ratio=0.0,
        flags=flags,
    )


def _refined(*, truncated: bool = False) -> RefineResult:
    return RefineResult(
        components=(),
        combined_mask=np.zeros((10, 10), dtype=np.uint8),
        truncated=truncated,
        total_components=0,
    )


def _decide(
    scored: ScoreResult,
    *,
    quality: QualityInfo | None = None,
    uncertain: bool = False,
    truncated: bool = False,
    config: AppConfig | None = None,
) -> tuple[Status, tuple[ReasonCode, ...]]:
    roi = _roi_result(np.full((10, 10), 255, dtype=np.uint8), uncertain=uncertain)
    result = stages.decide_status(
        scored,
        quality if quality is not None else _quality(),
        roi,
        _refined(truncated=truncated),
        config if config is not None else default_config(),
    )
    return result.status, result.reasons


# ---------------------------------------------------------------- ROI


def test_auto_roi_finds_surface_and_expands_bbox() -> None:
    gray = np.zeros((100, 200), dtype=np.uint8)
    gray[20:80, 30:170] = 200

    roi = stages.prepare_roi(gray, default_config())

    assert roi.mode == "auto"
    assert roi.uncertain is False
    assert roi.area_px > 0
    assert roi.bbox == (30, 20, 140, 60)
    assert roi.mask.shape == gray.shape
    assert roi.area_ratio > 0.2


def test_fixed_roi_uses_inset_and_min_area_ratio() -> None:
    gray = np.zeros((100, 200), dtype=np.uint8)
    config = _surface_config(mode="fixed", fixed_margin_px=5)

    roi = stages.prepare_roi(gray, config)

    assert roi.mode == "fixed"
    assert roi.bbox == (5, 5, 190, 90)
    assert roi.area_px == 190 * 90
    assert roi.uncertain is False
    assert int(np.count_nonzero(roi.mask)) == 190 * 90


def test_empty_roi_is_uncertain_but_not_an_exception() -> None:
    gray = np.zeros((100, 200), dtype=np.uint8)

    roi = stages.prepare_roi(gray, default_config())

    assert roi.bbox == (0, 0, 0, 0)
    assert roi.area_px == 0
    assert roi.area_ratio == 0.0
    assert roi.uncertain is True
    assert roi.mask.shape == gray.shape
    assert int(np.count_nonzero(roi.mask)) == 0


# ---------------------------------------------------------------- preprocess


def test_preprocess_response_quality_and_background_kernel() -> None:
    image = syn.to_bgr(syn.surface_gray(seed=201))
    image = syn.add_dark_rect(image, 60, 40, 12, 12, 20)
    image = syn.add_bright_rect(image, 200, 40, 12, 12, 230)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = default_config()
    roi = stages.prepare_roi(gray, config)

    result = stages.preprocess(gray, roi, config)

    assert result.bg_kernel == 21
    assert result.response.dtype == np.float32
    assert result.z_map.dtype == np.float32
    assert result.response[46, 66] > 12.0
    assert result.response[46, 206] > 12.0
    assert result.z_map[46, 66] >= 4.0
    assert np.all(result.response[roi.mask == 0] == 0.0)
    assert np.all(result.z_map[roi.mask == 0] == 0.0)
    assert np.all(np.isfinite(result.response))
    assert np.all(np.isfinite(result.z_map))
    assert result.quality.flags == ()


def test_preprocess_channel_toggles() -> None:
    dark_image = syn.to_bgr(syn.surface_gray(seed=202))
    dark_image = syn.add_dark_rect(dark_image, 60, 40, 12, 12, 20)
    bright_image = syn.to_bgr(syn.surface_gray(seed=203))
    bright_image = syn.add_bright_rect(bright_image, 60, 40, 12, 12, 230)

    base = default_config()
    dark_gray = cv2.cvtColor(dark_image, cv2.COLOR_BGR2GRAY)
    bright_gray = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
    dark_roi = stages.prepare_roi(dark_gray, base)
    bright_roi = stages.prepare_roi(bright_gray, base)

    dark_only = replace(base, preprocess=replace(base.preprocess, use_bright_channel=False))
    bright_only = replace(
        base, preprocess=replace(base.preprocess, use_dark_channel=False)
    )

    dark_result = stages.preprocess(dark_gray, dark_roi, dark_only)
    bright_result = stages.preprocess(bright_gray, bright_roi, bright_only)

    assert dark_result.response[46, 66] > 50.0
    assert bright_result.response[46, 66] > 50.0


def test_preprocess_rejects_disabled_channels() -> None:
    gray = syn.surface_gray(seed=204)
    config = replace(
        default_config(),
        preprocess=replace(
            default_config().preprocess,
            use_dark_channel=False,
            use_bright_channel=False,
        ),
    )
    roi = stages.prepare_roi(gray, default_config())

    with pytest.raises(PipelineError) as excinfo:
        stages.preprocess(gray, roi, config)

    assert excinfo.value.error_code is ErrorCode.E_CONFIG_INVALID
    assert excinfo.value.error_stage is Stage.PREPROCESS


def test_preprocess_quality_flag() -> None:
    gray = syn.surface_gray(seed=205)
    config = replace(
        default_config(),
        quality=replace(default_config().quality, dark_mean_min=200.0),
    )
    roi = stages.prepare_roi(gray, config)

    result = stages.preprocess(gray, roi, config)

    assert result.quality.flags == (ReasonCode.INPUT_QUALITY_LOW,)
    assert result.quality.mean_gray < 200.0


def test_preprocess_background_kernel_is_clamped_and_odd() -> None:
    config = default_config()
    assert stages._background_kernel(64, 1000, config) == 21
    assert stages._background_kernel(120, 320, config) == 21
    assert stages._background_kernel(400, 400, config) == 41
    assert stages._background_kernel(100, 100, config) == 21


def test_preprocess_empty_roi_returns_zero_response() -> None:
    gray = syn.surface_gray(seed=206)
    roi = stages.prepare_roi(np.zeros_like(gray), default_config())

    result = stages.preprocess(gray, roi, default_config())

    assert not result.response.any()
    assert not result.z_map.any()


def test_preprocess_non_finite_z_is_internal_error() -> None:
    gray = np.full((80, 80), 100, dtype=np.uint8)
    roi = _roi_result(np.full((80, 80), 255, dtype=np.uint8))
    config = replace(
        default_config(),
        preprocess=replace(default_config().preprocess, noise_floor_gray=0.0),
    )

    with pytest.raises(PipelineError) as excinfo:
        stages.preprocess(gray, roi, config)

    assert excinfo.value.error_code is ErrorCode.E_PROCESS_INTERNAL
    assert excinfo.value.error_stage is Stage.PREPROCESS


# ---------------------------------------------------------------- candidates


def test_generate_candidates_thresholds_and_roi() -> None:
    z_map = np.zeros((50, 50), dtype=np.float32)
    response = np.zeros((50, 50), dtype=np.float32)
    roi = np.zeros((50, 50), dtype=np.uint8)
    roi[5:45, 5:45] = 255
    z_map[20:26, 20:26] = 10.0
    response[20:26, 20:26] = 20.0

    result = stages.generate_candidates(z_map, response, roi, default_config())

    assert result.mask[22, 22] == 255
    assert result.mask[0, 0] == 0
    assert np.all(result.mask[roi == 0] == 0)


def test_generate_candidates_respects_both_thresholds() -> None:
    roi = np.full((40, 40), 255, dtype=np.uint8)
    config = default_config()

    low_z = np.zeros((40, 40), dtype=np.float32)
    low_z[10:16, 10:16] = 3.9
    high_response = np.full((40, 40), 20.0, dtype=np.float32)
    assert not stages.generate_candidates(
        low_z, high_response, roi, config
    ).mask.any()

    high_z = np.zeros((40, 40), dtype=np.float32)
    high_z[10:16, 10:16] = 10.0
    low_contrast = np.full((40, 40), 11.0, dtype=np.float32)
    assert not stages.generate_candidates(
        high_z, low_contrast, roi, config
    ).mask.any()


def test_generate_candidates_open_removes_isolated_pixel() -> None:
    roi = np.full((40, 40), 255, dtype=np.uint8)
    z_map = np.zeros((40, 40), dtype=np.float32)
    response = np.zeros((40, 40), dtype=np.float32)
    z_map[20, 20] = 20.0
    response[20, 20] = 30.0

    result = stages.generate_candidates(z_map, response, roi, default_config())

    assert not result.mask.any()


# ---------------------------------------------------------------- refine


def test_refine_filters_area_sorts_and_marks_border() -> None:
    roi_mask = np.zeros((80, 80), dtype=np.uint8)
    roi_mask[5:75, 5:75] = 255

    candidate = np.zeros((80, 80), dtype=np.uint8)
    candidate[5:11, 20:26] = 255
    candidate[40:46, 40:46] = 255
    candidate[5:10, 60:65] = 255

    response = np.zeros((80, 80), dtype=np.float32)
    response[5:11, 20:26] = 20.0
    response[40:46, 40:46] = 25.0
    response[5:10, 60:65] = 18.0
    z_map = np.zeros((80, 80), dtype=np.float32)
    z_map[candidate > 0] = 10.0

    config = replace(
        default_config(),
        candidates=replace(default_config().candidates, min_area_px=30),
    )
    result = stages.refine_mask(candidate, z_map, response, roi_mask, config)

    assert result.total_components == 2
    assert result.truncated is False
    assert len(result.components) == 2
    assert [component.index for component in result.components] == [0, 1]
    assert result.components[0].evidence >= result.components[1].evidence
    border_component = next(
        component
        for component in result.components
        if component.bbox[0] == 5 or component.bbox[1] == 5
    )
    assert border_component.touches_roi_border is True
    assert border_component.border_distance_px == pytest.approx(1.0)
    centered = next(
        component for component in result.components if not component.touches_roi_border
    )
    assert centered.border_distance_px > 1.0
    assert int(np.count_nonzero(result.combined_mask)) == sum(
        component.area_px for component in result.components
    )


def test_refine_truncates_to_max_candidates() -> None:
    roi_mask = np.full((80, 120), 255, dtype=np.uint8)
    candidate = np.zeros((80, 120), dtype=np.uint8)
    response = np.zeros((80, 120), dtype=np.float32)
    z_map = np.zeros((80, 120), dtype=np.float32)
    for index, (x, y) in enumerate(((10, 10), (40, 10), (70, 10))):
        candidate[y : y + 8, x : x + 8] = 255
        response[y : y + 8, x : x + 8] = 20.0 + index
        z_map[y : y + 8, x : x + 8] = 10.0

    config = replace(
        default_config(),
        candidates=replace(default_config().candidates, max_candidates=2),
    )
    result = stages.refine_mask(candidate, z_map, response, roi_mask, config)

    assert result.truncated is True
    assert result.total_components == 3
    assert len(result.components) == 2


def test_refine_area_filter_removes_small_component() -> None:
    roi_mask = np.full((60, 60), 255, dtype=np.uint8)
    candidate = np.zeros((60, 60), dtype=np.uint8)
    candidate[10:15, 10:15] = 255
    response = np.zeros((60, 60), dtype=np.float32)
    response[10:15, 10:15] = 30.0
    z_map = np.zeros((60, 60), dtype=np.float32)
    z_map[10:15, 10:15] = 10.0

    config = replace(
        default_config(),
        candidates=replace(default_config().candidates, min_area_px=30),
    )
    result = stages.refine_mask(candidate, z_map, response, roi_mask, config)

    assert result.components == ()
    assert result.total_components == 0
    assert result.truncated is False


# ---------------------------------------------------------------- features


def test_analyze_components_hand_calculated_features() -> None:
    component_mask = np.zeros((100, 100), dtype=np.uint8)
    component_mask[30:40, 20:30] = 255
    response = np.zeros((100, 100), dtype=np.float32)
    response[30:40, 20:30] = 24.0
    roi_mask = np.full((100, 100), 255, dtype=np.uint8)
    roi = _roi_result(roi_mask, area_px=10000)
    component = _component(0, 100, (20, 30, 10, 10), component_mask, 24.0)
    refined = RefineResult((component,), component_mask, False, 1)

    analyzed = stages.analyze_components(refined, roi, response, default_config())
    feature = analyzed.components[0]

    assert feature.index == 0
    assert feature.area_px == 100
    assert feature.bbox == (20, 30, 10, 10)
    assert feature.area_ratio == pytest.approx(0.01)
    assert feature.aspect_ratio == pytest.approx(1.0)
    assert feature.elongation == pytest.approx(1.0)
    assert feature.solidity == pytest.approx(1.0)
    assert feature.mean_contrast_gray == pytest.approx(24.0)
    assert feature.max_contrast_gray == pytest.approx(24.0)
    assert feature.integrated_contrast == pytest.approx(2400.0)
    assert feature.evidence == pytest.approx(math.sqrt((100 / 800) * (24 / 22)))


def test_analyze_components_aspect_and_elongation_for_wide_rectangle() -> None:
    component_mask = np.zeros((100, 100), dtype=np.uint8)
    component_mask[30:35, 20:40] = 255
    response = np.zeros((100, 100), dtype=np.float32)
    response[30:35, 20:40] = 30.0
    roi = _roi_result(np.full((100, 100), 255, dtype=np.uint8), area_px=10000)
    component = _component(0, 100, (20, 30, 20, 5), component_mask, 30.0)
    refined = RefineResult((component,), component_mask, False, 1)

    feature = stages.analyze_components(
        refined, roi, response, default_config()
    ).components[0]

    assert feature.aspect_ratio == pytest.approx(4.0)
    assert feature.elongation > 3.0


# ---------------------------------------------------------------- score


def test_calculate_score_uses_max_valid_evidence() -> None:
    low = _feature(0, area_px=60, mean_contrast=20.0)
    high = _feature(1, area_px=100, mean_contrast=30.0)

    result = stages.calculate_score((low, high), default_config())

    assert result.score == pytest.approx(max(low.evidence, high.evidence))
    assert len(result.valid_components) == 2


def test_calculate_score_excludes_invalid_components() -> None:
    small = _feature(0, area_px=10, mean_contrast=30.0)
    weak = _feature(1, area_px=60, mean_contrast=5.0)

    result = stages.calculate_score((small, weak), default_config())

    assert result.score == 0.0
    assert result.valid_components == ()


def test_component_evidence_caps_norms_at_1_5() -> None:
    config = default_config()

    assert stages._component_evidence(2000, 1000, config) == pytest.approx(1.5)
    assert stages._component_evidence(24, 12, config) == pytest.approx(
        math.sqrt((24 / 800) * (12 / 22))
    )


def test_calculate_score_rejects_non_finite_evidence() -> None:
    invalid = _feature(0, area_px=60, mean_contrast=20.0, evidence=float("inf"))

    with pytest.raises(PipelineError) as excinfo:
        stages.calculate_score((invalid,), default_config())

    assert excinfo.value.error_code is ErrorCode.E_PROCESS_INTERNAL
    assert excinfo.value.error_stage is Stage.CALCULATE_SCORE


# ---------------------------------------------------------------- decision


def test_decision_truth_table_rows() -> None:
    strong = _feature(0, evidence=1.2)
    status, reasons = _decide(ScoreResult(1.2, (strong,)))
    assert (status, reasons) == (Status.NG, (ReasonCode.DEFECT_EVIDENCE_STRONG,))

    status, reasons = _decide(ScoreResult(0.0, ()))
    assert (status, reasons) == (Status.OK, (ReasonCode.NO_DEFECT_EVIDENCE,))

    status, reasons = _decide(
        ScoreResult(0.0, ()),
        quality=_quality((ReasonCode.INPUT_QUALITY_LOW,)),
    )
    assert (status, reasons) == (Status.REVIEW, (ReasonCode.INPUT_QUALITY_LOW,))

    status, reasons = _decide(ScoreResult(0.0, ()), uncertain=True)
    assert (status, reasons) == (Status.REVIEW, (ReasonCode.ROI_UNCERTAIN,))

    border = _feature(0, evidence=0.45, touches=True)
    status, reasons = _decide(ScoreResult(0.45, (border,)))
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.BORDER_TOUCHING_CANDIDATE,),
    )

    status, reasons = _decide(ScoreResult(0.0, ()), truncated=True)
    assert (status, reasons) == (Status.REVIEW, (ReasonCode.CANDIDATE_TRUNCATED,))

    weak_a = _feature(0, evidence=0.45)
    weak_b = _feature(1, evidence=0.45)
    status, reasons = _decide(ScoreResult(0.45, (weak_a, weak_b)))
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.MULTIPLE_WEAK_CANDIDATES,),
    )

    status, reasons = _decide(ScoreResult(0.45, (weak_a,)))
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.EVIDENCE_IN_REVIEW_BAND,),
    )


def test_ok_branch_ignores_score_when_no_valid_components() -> None:
    status, reasons = _decide(ScoreResult(999.0, ()))
    assert (status, reasons) == (Status.OK, (ReasonCode.NO_DEFECT_EVIDENCE,))


def test_border_escalation_uses_frozen_review_band() -> None:
    border = _feature(0, evidence=REVIEW_BAND_MAX, touches=True)
    status, reasons = _decide(ScoreResult(REVIEW_BAND_MAX, (border,)))
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.BORDER_TOUCHING_CANDIDATE,),
    )

    below = _feature(0, evidence=REVIEW_BAND_MAX - 0.01, touches=True)
    status, reasons = _decide(ScoreResult(REVIEW_BAND_MAX - 0.01, (below,)))
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.EVIDENCE_IN_REVIEW_BAND,),
    )


def test_ng_priority_over_quality_and_roi_flags() -> None:
    strong = _feature(0, evidence=1.2, touches=True)
    status, reasons = _decide(
        ScoreResult(1.2, (strong,)),
        quality=_quality((ReasonCode.INPUT_QUALITY_LOW,)),
        uncertain=True,
        truncated=True,
    )

    assert status is Status.NG
    assert reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)


def test_decision_config_booleans_disable_escalation() -> None:
    base = default_config()
    border = _feature(0, evidence=0.45, touches=True)

    no_border = replace(
        base,
        decision=replace(base.decision, border_touch_escalates=False),
    )
    status, reasons = _decide(ScoreResult(0.45, (border,)), config=no_border)
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.EVIDENCE_IN_REVIEW_BAND,),
    )

    weak_a = _feature(0, evidence=0.45)
    weak_b = _feature(1, evidence=0.45)
    no_multi = replace(
        base,
        decision=replace(base.decision, multi_candidate_escalates=False),
    )
    status, reasons = _decide(
        ScoreResult(0.45, (weak_a, weak_b)),
        config=no_multi,
    )
    assert (status, reasons) == (
        Status.REVIEW,
        (ReasonCode.EVIDENCE_IN_REVIEW_BAND,),
    )

    no_truncation = replace(
        base,
        decision=replace(base.decision, truncation_escalates=False),
    )
    status, reasons = _decide(
        ScoreResult(0.0, ()),
        truncated=True,
        config=no_truncation,
    )
    assert (status, reasons) == (Status.OK, (ReasonCode.NO_DEFECT_EVIDENCE,))
