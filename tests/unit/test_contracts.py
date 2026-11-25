"""공용 계약 Enum(`Status`, `Stage`, `ErrorCode`) 및 `SampleResult` 직렬화 단위 테스트.

디스크 스키마와 연동되는 단계 순서·상태값·JSON 직렬화 규격이 깨지지 않도록 보장한다.
"""

from __future__ import annotations

import json

from surface_sentinel.contracts import (
    STAGE_ORDER,
    ComponentFeature,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    RunStatus,
    SampleResult,
    Stage,
    Status,
)


def test_stage_order_is_fixed() -> None:
    assert len(STAGE_ORDER) == 10
    assert STAGE_ORDER[0] is Stage.VALIDATE_INPUT
    assert STAGE_ORDER[-1] is Stage.SAVE_EVIDENCE
    assert [stage.value for stage in STAGE_ORDER] == [
        "validate_input",
        "load_image",
        "prepare_roi",
        "preprocess",
        "generate_candidates",
        "refine_mask",
        "analyze_components",
        "calculate_score",
        "decide_status",
        "save_evidence",
    ]


def test_status_values_include_error_but_run_status_does_not() -> None:
    assert {status.value for status in Status} == {"OK", "NG", "REVIEW", "ERROR"}
    assert {status.value for status in RunStatus} == {
        "COMPLETE",
        "COMPLETE_WITH_ERRORS",
        "FAILED",
    }


def test_error_codes_are_stable() -> None:
    assert ErrorCode.E_STORAGE_WRITE.value == "E_STORAGE_WRITE"
    assert ErrorCode.E_SEALED_DENIED.value == "E_SEALED_DENIED"


def test_sample_result_is_json_serializable() -> None:
    component = ComponentFeature(
        index=0,
        area_px=120,
        bbox=(10, 20, 40, 25),
        area_ratio=0.001,
        aspect_ratio=3.0,
        elongation=3.0,
        solidity=0.9,
        mean_contrast_gray=31.5,
        max_contrast_gray=55.0,
        integrated_contrast=3780.0,
        border_distance_px=12.0,
        touches_roi_border=False,
        evidence=1.2,
    )
    result = SampleResult(
        run_id="20261116T091500Z-a1b2c3d4-tune",
        sample_id="10023",
        original_id="10023",
        split="tune",
        label="defect",
        status=Status.NG,
        score=1.2,
        reasons=(ReasonCode.DEFECT_EVIDENCE_STRONG,),
        components=(component,),
        quality=QualityInfo(blur_variance=88.0, mean_gray=120.0, saturated_ratio=0.01),
        roi=RoiInfo(mode="auto", bbox=(0, 0, 630, 230), area_px=140000, area_ratio=0.97),
        timing_ms={"total": 12.5},
        input_sha256="a" * 64,
        config_hash="b" * 64,
    )

    payload = result.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == "result-v1"
    assert decoded["status"] == "NG"
    assert decoded["reasons"] == ["DEFECT_EVIDENCE_STRONG"]
    assert decoded["components"][0]["bbox"] == [10, 20, 40, 25]
    assert decoded["quality"]["flags"] == []
    assert decoded["roi"]["mode"] == "auto"
    assert decoded["timing_ms"] == {"total": 12.5}


def test_sample_result_allows_error_fields() -> None:
    result = SampleResult(
        run_id="run",
        sample_id="sample",
        original_id="sample",
        split="tune",
        label="good",
        status=Status.ERROR,
        score=0.0,
        error_code=ErrorCode.E_INPUT_DECODE,
        error_stage=Stage.LOAD_IMAGE,
    )
    payload = result.to_dict()
    assert payload["error_code"] == "E_INPUT_DECODE"
    assert payload["error_stage"] == "load_image"
