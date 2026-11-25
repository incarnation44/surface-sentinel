"""`SampleResult` 스키마 검증기 단위 테스트.

타이밍 딕셔너리 누락/null 차단, ERROR 상태 필수 필드 등 출력 계약 위반 탐지를 검증한다.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from surface_sentinel.contracts import (
    STAGE_ORDER,
    ComponentFeature,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Stage,
    Status,
)
from surface_sentinel.pipeline import SchemaError, validate_sample_result


def _timings() -> dict[str, float]:
    timings = {stage.value: 0.0 for stage in STAGE_ORDER}
    timings["total"] = 0.0
    return timings


def _result(**overrides: object) -> SampleResult:
    values: dict[str, object] = {
        "run_id": "inspect-12345678-sample",
        "sample_id": "sample",
        "original_id": "sample",
        "split": "tune",
        "label": "unknown",
        "status": Status.OK,
        "score": 0.0,
        "reasons": (ReasonCode.NO_DEFECT_EVIDENCE,),
        "timing_ms": _timings(),
        "input_sha256": "a" * 64,
        "config_hash": "b" * 64,
    }
    values.update(overrides)
    return SampleResult(**values)  # type: ignore[arg-type]


def test_valid_result_passes() -> None:
    validate_sample_result(_result())


def test_timing_ms_none_is_rejected() -> None:
    result = replace(_result(), timing_ms=None)

    with pytest.raises(SchemaError, match="timing_ms"):
        validate_sample_result(result)


def test_missing_stage_timing_is_rejected() -> None:
    timings = _timings()
    del timings[Stage.PREPROCESS.value]

    with pytest.raises(SchemaError, match="필수 키"):
        validate_sample_result(_result(timing_ms=timings))


def test_missing_total_timing_is_rejected() -> None:
    timings = _timings()
    del timings["total"]

    with pytest.raises(SchemaError, match="필수 키"):
        validate_sample_result(_result(timing_ms=timings))


def test_require_stage_timings_false_allows_partial_dict() -> None:
    validate_sample_result(
        _result(timing_ms={"total": 1.0}),
        require_stage_timings=False,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_timing_values_are_rejected(value: float) -> None:
    timings = _timings()
    timings["preprocess"] = value

    with pytest.raises(SchemaError, match="timing_ms"):
        validate_sample_result(_result(timing_ms=timings))


def test_non_string_timing_key_is_rejected() -> None:
    timings = _timings()
    timings[1] = 1.0  # type: ignore[index]

    with pytest.raises(SchemaError, match="문자열"):
        validate_sample_result(_result(timing_ms=timings))


def test_error_status_requires_code_and_stage() -> None:
    without_code = _result(status=Status.ERROR)
    with pytest.raises(SchemaError, match="error_code"):
        validate_sample_result(without_code)

    only_code = _result(status=Status.ERROR, error_code=ErrorCode.E_INPUT_MISSING)
    with pytest.raises(SchemaError, match="error_stage"):
        validate_sample_result(only_code)

    valid_error = _result(
        status=Status.ERROR,
        error_code=ErrorCode.E_INPUT_MISSING,
        error_stage=Stage.VALIDATE_INPUT,
    )
    validate_sample_result(valid_error)


def test_non_error_status_must_not_have_error_fields() -> None:
    with pytest.raises(SchemaError, match="error_code"):
        validate_sample_result(
            _result(
                error_code=ErrorCode.E_INPUT_MISSING,
                error_stage=Stage.VALIDATE_INPUT,
            )
        )


@pytest.mark.parametrize("score", [float("inf"), float("nan"), -0.5])
def test_invalid_score_is_rejected(score: float) -> None:
    with pytest.raises(SchemaError, match="score"):
        validate_sample_result(_result(score=score))


def test_config_hash_must_be_64_hex() -> None:
    with pytest.raises(SchemaError, match="config_hash"):
        validate_sample_result(_result(config_hash="short"))


def test_input_sha256_may_be_empty_or_64_hex() -> None:
    validate_sample_result(_result(input_sha256=""))
    with pytest.raises(SchemaError, match="input_sha256"):
        validate_sample_result(_result(input_sha256="not-a-hash"))


def test_reasons_must_be_reason_codes() -> None:
    with pytest.raises(SchemaError, match="reasons"):
        validate_sample_result(_result(reasons=("NO_DEFECT_EVIDENCE",)))


def test_components_must_be_component_features() -> None:
    with pytest.raises(SchemaError, match="components"):
        validate_sample_result(_result(components=({"index": 0},)))


def test_quality_and_roi_types_are_checked() -> None:
    validate_sample_result(
        _result(
            quality=QualityInfo(
                blur_variance=10.0,
                mean_gray=100.0,
                saturated_ratio=0.0,
                flags=(ReasonCode.INPUT_QUALITY_LOW,),
            ),
            roi=RoiInfo(
                mode="auto",
                bbox=(0, 0, 10, 10),
                area_px=100,
                area_ratio=1.0,
            ),
        )
    )
    with pytest.raises(SchemaError, match="quality"):
        validate_sample_result(_result(quality="bad"))


def test_schema_version_must_match_result_contract() -> None:
    with pytest.raises(SchemaError, match="schema_version"):
        validate_sample_result(_result(schema_version="result-v2"))


def test_timing_dict_default_is_not_shared() -> None:
    first = SampleResult(
        run_id="r",
        sample_id="s",
        original_id="s",
        split="tune",
        label="unknown",
        status=Status.OK,
        score=0.0,
        config_hash="c" * 64,
    )
    second = SampleResult(
        run_id="r",
        sample_id="s",
        original_id="s",
        split="tune",
        label="unknown",
        status=Status.OK,
        score=0.0,
        config_hash="c" * 64,
    )

    first.timing_ms["total"] = 1.0
    assert second.timing_ms == {}


def test_component_feature_round_trip_is_still_valid() -> None:
    component = ComponentFeature(
        index=0,
        area_px=60,
        bbox=(10, 10, 8, 8),
        area_ratio=0.01,
        aspect_ratio=1.0,
        elongation=1.0,
        solidity=1.0,
        mean_contrast_gray=20.0,
        max_contrast_gray=25.0,
        integrated_contrast=1200.0,
        border_distance_px=10.0,
        touches_roi_border=False,
        evidence=0.7,
    )
    result = _result(
        status=Status.NG,
        score=0.7,
        reasons=(ReasonCode.DEFECT_EVIDENCE_STRONG,),
        components=(component,),
        quality=QualityInfo(blur_variance=50.0, mean_gray=120.0, saturated_ratio=0.1),
        roi=RoiInfo(mode="auto", bbox=(0, 0, 20, 20), area_px=300, area_ratio=0.75),
    )
    validate_sample_result(replace(result))
