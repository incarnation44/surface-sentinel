"""검사 API 요청·응답 스키마 (E4, ADR 0007).

상태·사유 문자열은 :mod:`surface_sentinel.contracts`의 ``StrEnum`` 값을 그대로
직렬화한다(예: ``"REVIEW"``, ``"DEFECT_EVIDENCE_STRONG"``). API 전용 문자열
사본을 새로 만들지 않으므로 contracts와 어긋날 수 없다.

모든 비율은 :class:`~surface_sentinel.evaluation.metrics.Ratio`의 분모를 함께
보고한다. 분모가 0이면 ``value``는 ``None``이다.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ..contracts import ComponentFeature, ReasonCode, Status


class EngineName(StrEnum):
    """검사 엔진 이름. 값은 :data:`surface_sentinel.records.store.ENGINES`와 같다."""

    OPENCV = "opencv"
    YOLO_DET = "yolo-det"
    YOLO_SEG = "yolo-seg"
    COMBINED = "combined"


class ReviewDecision(StrEnum):
    """사람 판정 값. 기계 판정의 REVIEW/ERROR는 사람 판정이 아니다."""

    OK = "OK"
    NG = "NG"


class ComponentOut(BaseModel):
    """검사 결과의 컴포넌트 특징 한 건(contracts와 같은 필드 이름)."""

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

    @classmethod
    def from_feature(cls, feature: ComponentFeature) -> ComponentOut:
        """``ComponentFeature``를 그대로 옮긴다."""

        return cls(**feature.to_dict())


class InspectResponse(BaseModel):
    """``POST /api/inspect`` 응답.

    ``status``/``reasons``/``components``는 **이번 요청의 검사 결과**다.
    ``inspection_id``는 DB에 남은 행(같은 이미지면 기존 행)이고,
    ``deduplicated``가 참이면 새 행을 만들지 않았다는 뜻이다. 저장된 기계
    판정은 ``GET /api/inspections/{id}``로 확인한다(불변).
    """

    inspection_id: int
    run_id: str
    sample_id: str
    status: Status
    score: float
    reasons: list[ReasonCode]
    components: list[ComponentOut]
    engine: EngineName
    model_version: str
    overlay_url: str | None
    deduplicated: bool


class InspectionItem(BaseModel):
    """``GET /api/inspections`` 목록의 한 행."""

    inspection_id: int
    run_id: str
    sample_id: str
    label: str
    split: str
    status: Status
    score: float
    engine: EngineName
    model_version: str
    reasons: list[str]
    overlay_url: str | None
    created_at: str


class InspectionList(BaseModel):
    """페이지네이션 응답. ``total``은 필터 적용 후 전체 건수다."""

    total: int
    limit: int
    offset: int
    items: list[InspectionItem]


class ReviewItem(BaseModel):
    """사람 판정 한 건(append-only)."""

    review_id: int
    decision: ReviewDecision
    reviewer: str
    reason: str
    created_at: str


class InspectionDetail(InspectionItem):
    """``GET /api/inspections/{id}`` 응답. 리뷰 이력을 함께 준다."""

    image_sha256: str
    config_hash: str
    overlay_path: str
    timing_ms: dict[str, float]
    reviews: list[ReviewItem]


class ReviewRequest(BaseModel):
    """``POST /api/inspections/{id}/review`` 본문. 기계 판정은 바꾸지 않는다."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    reviewer: str = ""
    reason: str = ""


class ReviewResponse(BaseModel):
    """사람 판정 append 결과. ``machine_status``는 기록된 기계 판정 그대로다."""

    inspection_id: int
    review_id: int
    decision: ReviewDecision
    machine_status: Status
    reviews: int


class RatioOut(BaseModel):
    """분모가 명시된 비율(:meth:`Ratio.as_dict`와 같은 필드)."""

    definition: str
    numerator: float | None
    denominator: float | None
    value: float | None
    ci95_low: float | None
    ci95_high: float | None


class ReviewStats(BaseModel):
    """리뷰 집계: append 건수와 판정이 있는 검사 건수."""

    decisions: int
    reviewed_inspections: int


class StatsResponse(BaseModel):
    """``GET /api/stats`` 응답. 모든 비율에 분모가 함께 온다."""

    total: int
    by_status: dict[str, int]
    reviews: ReviewStats
    overlays: int
    ratios: dict[str, RatioOut]
    by_engine: dict[str, dict[str, int]]


__all__ = [
    "ComponentOut",
    "EngineName",
    "InspectResponse",
    "InspectionDetail",
    "InspectionItem",
    "InspectionList",
    "RatioOut",
    "ReviewDecision",
    "ReviewItem",
    "ReviewRequest",
    "ReviewResponse",
    "ReviewStats",
    "StatsResponse",
]
