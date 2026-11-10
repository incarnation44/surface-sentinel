"""파이프라인 단계 사이에서 주고받는 값 타입.

numpy 배열은 동결 dataclass에 담기므로 ``eq=False``를 사용한다. 배열
동등성은 테스트에서 명시적으로 ``numpy.array_equal``로 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import (
    ComponentFeature,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Status,
)


@dataclass(frozen=True, eq=False)
class InputMetadata:
    """입력 검증 단계가 확정한 메타데이터."""

    image_path: str
    mask_path: str
    label: str
    split: str
    width: int
    height: int
    channels: int
    image_sha256: str
    mask_sha256: str
    defect_pixels: int


@dataclass(frozen=True, eq=False)
class LoadedImage:
    """원본 BGR과 회색조 표현."""

    bgr: np.ndarray
    gray: np.ndarray


@dataclass(frozen=True, eq=False)
class RoiResult:
    """관심 영역 마스크와 표시용 경계 상자."""

    mode: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area_px: int
    area_ratio: float
    uncertain: bool

    def to_info(self) -> RoiInfo:
        return RoiInfo(
            mode=self.mode,
            bbox=self.bbox,
            area_px=self.area_px,
            area_ratio=self.area_ratio,
            uncertain=self.uncertain,
        )


@dataclass(frozen=True, eq=False)
class PreprocessResult:
    """배경 보정 응답, 강건 z 지도, 품질 정보."""

    response: np.ndarray
    z_map: np.ndarray
    quality: QualityInfo
    bg_kernel: int


@dataclass(frozen=True, eq=False)
class CandidateResult:
    """후보 픽셀 마스크."""

    mask: np.ndarray


@dataclass(frozen=True, eq=False)
class RefinedComponent:
    """정제 단계에서 유지된 연결 요소."""

    index: int
    area_px: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    mean_contrast_gray: float
    max_contrast_gray: float
    integrated_contrast: float
    border_distance_px: float
    touches_roi_border: bool
    evidence: float


@dataclass(frozen=True, eq=False)
class RefineResult:
    """정제된 컴포넌트 목록과 절단 여부."""

    components: tuple[RefinedComponent, ...]
    combined_mask: np.ndarray
    truncated: bool
    total_components: int


@dataclass(frozen=True, eq=False)
class AnalyzeResult:
    """계약 특징 표."""

    components: tuple[ComponentFeature, ...]


@dataclass(frozen=True, eq=False)
class ScoreResult:
    """이미지 점수와 최소 타당성을 통과한 컴포넌트."""

    score: float
    valid_components: tuple[ComponentFeature, ...]


@dataclass(frozen=True, eq=False)
class DecisionResult:
    """상태 계약 판정 결과."""

    status: Status
    reasons: tuple[ReasonCode, ...]


@dataclass(frozen=True, eq=False)
class PipelineDebug:
    """검사 중간 산출물. 저장 여부와 무관하게 호출자에게 돌려준다."""

    gray: np.ndarray
    roi_mask: np.ndarray
    response: np.ndarray
    z_map: np.ndarray
    candidate_mask: np.ndarray
    refined_mask: np.ndarray


@dataclass(frozen=True, eq=False)
class InspectionOutcome:
    """검사 결과와 선택적 중간 산출물."""

    result: SampleResult
    debug: PipelineDebug | None = None


__all__ = [
    "AnalyzeResult",
    "CandidateResult",
    "DecisionResult",
    "InputMetadata",
    "InspectionOutcome",
    "LoadedImage",
    "PipelineDebug",
    "PreprocessResult",
    "RefinedComponent",
    "RefineResult",
    "RoiResult",
    "ScoreResult",
]
