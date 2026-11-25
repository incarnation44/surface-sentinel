"""검사 결과(`SampleResult`) 반환 전 계약 스키마 검증기.

파이프라인은 결과를 반환하기 전에 이 검증기를 호출한다. 실패하면
``ERROR``/``E_PROCESS_INTERNAL``로 바꾸는 것은 오케스트레이터의 책임이다.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ..contracts import (
    RESULT_SCHEMA_VERSION,
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

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class SchemaError(ValueError):
    """결과가 result-v1 계약을 위반할 때 발생한다."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_sample_result(
    result: SampleResult,
    *,
    require_stage_timings: bool = True,
) -> None:
    """결과 스키마를 검증한다. 위반 시 ``SchemaError``를 낸다."""

    _require(isinstance(result, SampleResult), "결과가 SampleResult가 아닙니다.")
    _require(result.timing_ms is not None, "timing_ms가 None입니다.")
    _require(isinstance(result.timing_ms, dict), "timing_ms가 dict가 아닙니다.")

    if require_stage_timings:
        expected = [stage.value for stage in STAGE_ORDER] + ["total"]
        missing = [key for key in expected if key not in result.timing_ms]
        _require(not missing, f"timing_ms에 필수 키가 없습니다: {missing}")

    for key, value in result.timing_ms.items():
        _require(isinstance(key, str), "timing_ms 키가 문자열이 아닙니다.")
        _require(_is_number(value), f"timing_ms[{key!r}] 값이 숫자가 아닙니다.")
        _require(math.isfinite(float(value)), f"timing_ms[{key!r}] 값이 유한하지 않습니다.")
        _require(float(value) >= 0.0, f"timing_ms[{key!r}] 값이 음수입니다.")

    _require(isinstance(result.status, Status), "status가 Status가 아닙니다.")
    _require(_is_number(result.score), "score가 숫자가 아닙니다.")
    _require(math.isfinite(float(result.score)), "score가 유한하지 않습니다.")
    _require(float(result.score) >= 0.0, "score가 음수입니다.")

    _require(isinstance(result.reasons, tuple), "reasons가 tuple이 아닙니다.")
    for reason in result.reasons:
        _require(isinstance(reason, ReasonCode), f"reasons에 알 수 없는 값이 있습니다: {reason!r}")

    _require(isinstance(result.components, tuple), "components가 tuple이 아닙니다.")
    for component in result.components:
        _require(
            isinstance(component, ComponentFeature),
            f"components에 ComponentFeature가 아닌 값이 있습니다: {component!r}",
        )

    _require(
        result.quality is None or isinstance(result.quality, QualityInfo),
        "quality가 QualityInfo 또는 None이 아닙니다.",
    )
    if result.quality is not None:
        _require(isinstance(result.quality.flags, tuple), "quality.flags가 tuple이 아닙니다.")
        for flag in result.quality.flags:
            _require(
                isinstance(flag, ReasonCode),
                f"quality.flags에 알 수 없는 값이 있습니다: {flag!r}",
            )
    _require(
        result.roi is None or isinstance(result.roi, RoiInfo),
        "roi가 RoiInfo 또는 None이 아닙니다.",
    )

    if result.status is Status.ERROR:
        _require(result.error_code is not None, "ERROR인데 error_code가 없습니다.")
        _require(result.error_stage is not None, "ERROR인데 error_stage가 없습니다.")
        _require(isinstance(result.error_code, ErrorCode), "error_code가 ErrorCode가 아닙니다.")
        _require(isinstance(result.error_stage, Stage), "error_stage가 Stage가 아닙니다.")
    else:
        _require(result.error_code is None, "ERROR가 아닌데 error_code가 있습니다.")
        _require(result.error_stage is None, "ERROR가 아닌데 error_stage가 있습니다.")

    _require(
        result.input_sha256 == "" or bool(_HEX64.match(result.input_sha256)),
        "input_sha256이 빈 문자열도 64자 hex도 아닙니다.",
    )
    _require(
        bool(_HEX64.match(result.config_hash)),
        "config_hash가 64자 hex가 아닙니다.",
    )
    _require(
        result.schema_version == RESULT_SCHEMA_VERSION,
        f"schema_version이 {RESULT_SCHEMA_VERSION!r}이 아닙니다: {result.schema_version!r}",
    )


__all__ = ["SchemaError", "validate_sample_result"]
