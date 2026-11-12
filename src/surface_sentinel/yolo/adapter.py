"""YOLO 탐지 → 기존 ``Status`` 계약 어댑터.

새 판정 체계를 만들지 않는다. 탐지 결과를 ``ComponentFeature``/``SampleResult``
라는 기존 계약에 채워 넣고, 임계값 정책은 :class:`YoloDecisionPolicy` 하나로
고정한다. 임계값은 tune(또는 tune-val)에서만 조정하며 test 로 튜닝하지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..contracts import ComponentFeature, ReasonCode, SampleResult, Status
from .inference import Detection


@dataclass(frozen=True)
class YoloDecisionPolicy:
    """탐지 점수 → 제품 판정 정책.

    ``review_lo`` 미만이면 OK, ``ng_conf`` 이상이면 NG, 사이는 REVIEW.
    두 값은 tune-val 에서만 조정한다 [DESIGN].
    """

    ng_conf: float = 0.50
    review_lo: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_lo <= self.ng_conf <= 1.0:
            raise ValueError(
                "임계값은 0 <= review_lo <= ng_conf <= 1 이어야 합니다: "
                f"review_lo={self.review_lo} ng_conf={self.ng_conf}"
            )


def decide_from_detections(
    detections: Sequence[Detection],
    policy: YoloDecisionPolicy | None = None,
) -> tuple[Status, tuple[ReasonCode, ...], float]:
    """탐지 목록 → ``(Status, reasons, score)``.

    ``score`` 는 최대 탐지 점수이며, 탐지가 없으면 0.0 이다.
    """

    effective = policy if policy is not None else YoloDecisionPolicy()
    score = max((float(detection.score) for detection in detections), default=0.0)
    if not detections or score < effective.review_lo:
        return Status.OK, (ReasonCode.NO_DEFECT_EVIDENCE,), score
    if score >= effective.ng_conf:
        return Status.NG, (ReasonCode.DEFECT_EVIDENCE_STRONG,), score
    return Status.REVIEW, (ReasonCode.EVIDENCE_IN_REVIEW_BAND,), score


def _component_from_detection(
    detection: Detection,
    index: int,
    *,
    image_w: int,
    image_h: int,
) -> ComponentFeature:
    """탐지 하나를 ``ComponentFeature`` 로 변환한다.

    면적·종횡비·solidity 는 마스크/박스에서 계산한다. 대비(contrast) 계열은
    YOLO 경로에서 그레이 응답을 계산하지 않으므로 0.0 placeholder 로 둔다
    [DESIGN]. 비율의 분모는 이미지 전체 면적이며, 0이면 호출 전에 거부한다.
    """

    x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    bbox_area = width * height

    mask = detection.mask
    if mask is not None:
        area_px = int(np.count_nonzero(mask))
        solidity = float(area_px / bbox_area) if bbox_area > 0 else 0.0
    else:
        area_px = bbox_area
        solidity = 1.0  # [DESIGN] 마스크 없음 → 박스가 곧 영역

    total_px = image_w * image_h
    area_ratio = float(area_px / total_px)
    aspect_ratio = float(width / height) if height > 0 else 0.0
    elongation = float(max(width, height) / min(width, height)) if min(width, height) > 0 else 1.0
    border_distance = float(min(x1, y1, image_w - x2, image_h - y2))

    return ComponentFeature(
        index=index,
        area_px=area_px,
        bbox=(x1, y1, x2, y2),
        area_ratio=area_ratio,
        aspect_ratio=aspect_ratio,
        elongation=max(1.0, elongation),
        solidity=min(1.0, max(0.0, solidity)),
        mean_contrast_gray=0.0,
        max_contrast_gray=0.0,
        integrated_contrast=0.0,
        border_distance_px=max(0.0, border_distance),
        touches_roi_border=border_distance <= 0.0,
        evidence=float(detection.score),
    )


def to_sample_result(
    detections: Sequence[Detection],
    *,
    run_id: str,
    sample_id: str,
    image_size: tuple[int, int],
    split: str = "tune",
    label: str = "unknown",
    config_hash: str = "",
    model_version: str = "",
    policy: YoloDecisionPolicy | None = None,
    timing_ms: dict[str, float] | None = None,
) -> SampleResult:
    """탐지 목록 → ``SampleResult``(기존 계약 필드 전부 채움).

    ``image_size`` 는 ``(width, height)`` 원본 픽셀 크기다. 0이면 비율의
    분모가 없으므로 ``ValueError`` 로 거부한다. ``model_version`` 이 있으면
    ``config_hash`` 뒤에 병기해 모델 교체를 추적할 수 있게 한다.
    """

    image_w, image_h = int(image_size[0]), int(image_size[1])
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"이미지 크기가 0입니다: image_size={image_size}")

    ordered = sorted(
        detections,
        key=lambda detection: (-float(detection.score), detection.bbox_xyxy),
    )
    status, reasons, score = decide_from_detections(ordered, policy)
    components = tuple(
        _component_from_detection(
            detection, index, image_w=image_w, image_h=image_h
        )
        for index, detection in enumerate(ordered)
    )

    effective_hash = config_hash
    if model_version:
        effective_hash = (
            f"{config_hash}|yolo:{model_version}" if config_hash else f"yolo:{model_version}"
        )

    return SampleResult(
        run_id=run_id,
        sample_id=sample_id,
        original_id=sample_id,
        split=split,
        label=label,
        status=status,
        score=score,
        reasons=reasons,
        components=components,
        timing_ms=dict(timing_ms) if timing_ms is not None else {},
        config_hash=effective_hash,
    )


__all__ = [
    "YoloDecisionPolicy",
    "decide_from_detections",
    "to_sample_result",
]
