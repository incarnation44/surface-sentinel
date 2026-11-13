"""Shared data contracts for inspection, evaluation, and persistence.

The enum values in this module are part of the on-disk schema. Renaming a
member requires a schema version bump and a migration plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from typing import Any

RESULT_SCHEMA_VERSION = "result-v1"
RUN_SCHEMA_VERSION = "run-v1"


class Status(StrEnum):
    """Product-level status. ``ERROR`` is not a product verdict."""

    OK = "OK"
    NG = "NG"
    REVIEW = "REVIEW"
    ERROR = "ERROR"


class RunStatus(StrEnum):
    COMPLETE = "COMPLETE"
    COMPLETE_WITH_ERRORS = "COMPLETE_WITH_ERRORS"
    FAILED = "FAILED"


class Stage(StrEnum):
    VALIDATE_INPUT = "validate_input"
    LOAD_IMAGE = "load_image"
    PREPARE_ROI = "prepare_roi"
    PREPROCESS = "preprocess"
    GENERATE_CANDIDATES = "generate_candidates"
    REFINE_MASK = "refine_mask"
    ANALYZE_COMPONENTS = "analyze_components"
    CALCULATE_SCORE = "calculate_score"
    DECIDE_STATUS = "decide_status"
    SAVE_EVIDENCE = "save_evidence"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.VALIDATE_INPUT,
    Stage.LOAD_IMAGE,
    Stage.PREPARE_ROI,
    Stage.PREPROCESS,
    Stage.GENERATE_CANDIDATES,
    Stage.REFINE_MASK,
    Stage.ANALYZE_COMPONENTS,
    Stage.CALCULATE_SCORE,
    Stage.DECIDE_STATUS,
    Stage.SAVE_EVIDENCE,
)


class ReasonCode(StrEnum):
    DEFECT_EVIDENCE_STRONG = "DEFECT_EVIDENCE_STRONG"
    NO_DEFECT_EVIDENCE = "NO_DEFECT_EVIDENCE"
    EVIDENCE_IN_REVIEW_BAND = "EVIDENCE_IN_REVIEW_BAND"
    INPUT_QUALITY_LOW = "INPUT_QUALITY_LOW"
    ROI_UNCERTAIN = "ROI_UNCERTAIN"
    BORDER_TOUCHING_CANDIDATE = "BORDER_TOUCHING_CANDIDATE"
    CANDIDATE_TRUNCATED = "CANDIDATE_TRUNCATED"
    MULTIPLE_WEAK_CANDIDATES = "MULTIPLE_WEAK_CANDIDATES"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    MANUAL_DEFERRED = "MANUAL_DEFERRED"


class ErrorCode(StrEnum):
    E_INPUT_MISSING = "E_INPUT_MISSING"
    E_INPUT_DECODE = "E_INPUT_DECODE"
    E_INPUT_SIZE = "E_INPUT_SIZE"
    E_INPUT_UNIFORM = "E_INPUT_UNIFORM"
    E_MASK_SHAPE = "E_MASK_SHAPE"
    E_MASK_MISMATCH = "E_MASK_MISMATCH"
    E_CONFIG_INVALID = "E_CONFIG_INVALID"
    E_PROCESS_INTERNAL = "E_PROCESS_INTERNAL"
    E_STORAGE_WRITE = "E_STORAGE_WRITE"
    E_SEALED_DENIED = "E_SEALED_DENIED"
    E_PROFILE_UNKNOWN = "E_PROFILE_UNKNOWN"
    E_BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"
    E_TOOL_FAILED = "E_TOOL_FAILED"


@dataclass(frozen=True)
class ComponentFeature:
    index: int
    area_px: int
    bbox: tuple[int, int, int, int]
    area_ratio: float
    aspect_ratio: float
    elongation: float
    solidity: float
    mean_contrast_gray: float
    max_contrast_gray: float
    integrated_contrast: float
    border_distance_px: float
    touches_roi_border: bool
    evidence: float

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class QualityInfo:
    blur_variance: float
    mean_gray: float
    saturated_ratio: float
    flags: tuple[ReasonCode, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class RoiInfo:
    mode: str
    bbox: tuple[int, int, int, int]
    area_px: int
    area_ratio: float
    uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class SampleResult:
    run_id: str
    sample_id: str
    original_id: str
    split: str
    label: str
    status: Status
    score: float
    reasons: tuple[ReasonCode, ...] = ()
    components: tuple[ComponentFeature, ...] = ()
    quality: QualityInfo | None = None
    roi: RoiInfo | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    error_code: ErrorCode | None = None
    error_stage: Stage | None = None
    input_sha256: str = ""
    mask_sha256: str = ""
    overlay_path: str = ""
    config_hash: str = ""
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class EngineObservation:
    """결합(combined) 검사에서 엔진 하나의 원본 결과.

    기계 판정(``SampleResult.status``)은 그대로 보존하며, 결합 결과가 점수·성분을
    덮어써도 엔진별 증거가 소실되지 않도록 별도 기록용이다(P0-3). ``engine`` 은
    ``opencv``/``yolo-det``/``yolo-seg`` 같은 실제 엔진 이름이다.
    """

    engine: str
    result: SampleResult
    task: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "ComponentFeature",
    "EngineObservation",
    "ErrorCode",
    "QualityInfo",
    "RESULT_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "ReasonCode",
    "RoiInfo",
    "RunStatus",
    "STAGE_ORDER",
    "SampleResult",
    "Stage",
    "Status",
]
