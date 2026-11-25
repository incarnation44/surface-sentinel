"""성능 비교용 단순 기준선(B0 항상OK · B1 Otsu · B2 단순 임계값) 독립 구현.

**독립 구현이다.** 제안 파이프라인의 단계 함수(``prepare_roi``,
``preprocess``, ``refine_mask`` …)를 부르지 않는다. 기준선이 제안 파이프라인의
버그를 물려받으면 비교가 성립하지 않기 때문이다. 여기서는 OpenCV를 직접
부른다. 공유하는 것은 **공정성 때문에 같은 값을 쓰는 세 파라미터**뿐이다
(``roi.dark_floor``, ``candidates.min_area_px``, ``candidates.min_contrast_gray``).

| ID | 구성 | 판정 |
|---|---|---|
| B0 | 항상 OK | 판정 없음(유병률/하한) |
| B1 | ROI + median 3 + Otsu(ROI 내부) + 최소 면적 | 이진 |
| B2 | 배경 보정(median 31) + 고정 임계 + 최소 면적 | 이진 |

B1/B2는 REVIEW를 내지 않는다. 즉 이 기준선들은 "보류"라는 선택지를 갖지
못하므로 review 지표가 0으로 나오는 것이 정상이며, 그 사실을 표에 적는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig
from ..contracts import Status
from .cache import CachedComponent, CachedSample
from .measure import SampleOutcome, VariantReport, summarise

B0_NAME = "B0"
B1_NAME = "B1"
B2_NAME = "B2"

BG_MEDIAN_KERNEL = 31


def _read_gray(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"이미지가 비어 있습니다: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"이미지 디코드에 실패했습니다: {path}")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ellipse(size: int) -> np.ndarray:
    kernel = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))


def _largest_surface(gray: np.ndarray, dark_floor: int) -> np.ndarray:
    """B1의 ROI: 밝은 표면 → 닫힘 → 최대 연결요소 → 채움 → 침식."""

    surface = (gray > dark_floor).astype(np.uint8) * 255
    closed = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, _ellipse(7))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return np.zeros(gray.shape, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(gray.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=2)


def _components_from_mask(
    mask: np.ndarray,
    *,
    min_area_px: int,
) -> tuple[CachedComponent, ...]:
    """이진 마스크에서 최소 면적을 통과한 컴포넌트를 잘라 담는다."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[CachedComponent] = []
    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        bbox = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        x, y, width, height = bbox
        component_mask = (labels == label).astype(np.uint8) * 255
        components.append(
            CachedComponent(
                area_px=area_px,
                bbox=bbox,
                mean_contrast_gray=0.0,
                max_contrast_gray=0.0,
                integrated_contrast=0.0,
                border_distance_px=0.0,
                touches_roi_border=False,
                mask=np.ascontiguousarray(component_mask[y : y + height, x : x + width]),
            )
        )
    return tuple(components)


@dataclass(frozen=True)
class BaselineDetection:
    """기준선 하나가 샘플마다 낸 것(마스크 컴포넌트와 점수)."""

    components: tuple[CachedComponent, ...]
    score: float


def b1_detect(gray: np.ndarray, config: AppConfig) -> BaselineDetection:
    """ROI + median + Otsu + 최소 면적. Otsu는 **ROI 내부 픽셀에만** 적용한다."""

    roi = _largest_surface(gray, config.roi.dark_floor)
    blurred = cv2.medianBlur(gray, 3)
    values = blurred[roi > 0]
    if values.size == 0:
        return BaselineDetection(components=(), score=0.0)
    threshold, _ = cv2.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    foreground = ((blurred < float(threshold)) & (roi > 0)).astype(np.uint8) * 255
    components = _components_from_mask(
        foreground,
        min_area_px=config.candidates.min_area_px,
    )
    return BaselineDetection(
        components=components,
        score=1.0 if components else 0.0,
    )


def b2_detect(gray: np.ndarray, config: AppConfig) -> BaselineDetection:
    """배경 보정(median 31) + 고정 임계 + 최소 면적. ROI를 쓰지 않는다."""

    kernel = BG_MEDIAN_KERNEL
    limit = min(gray.shape[:2])
    if limit % 2 == 0:
        limit -= 1
    kernel = max(3, min(kernel, limit))
    background = cv2.medianBlur(gray, kernel)
    dark = background.astype(np.int16) - gray.astype(np.int16)
    bright = gray.astype(np.int16) - background.astype(np.int16)
    response = np.maximum(dark, bright)
    mask = (response >= float(config.candidates.min_contrast_gray)).astype(np.uint8) * 255
    components = _components_from_mask(mask, min_area_px=config.candidates.min_area_px)
    return BaselineDetection(components=components, score=1.0 if components else 0.0)


def _b0(samples: Sequence[CachedSample]) -> list[SampleOutcome]:
    return [
        SampleOutcome(
            sample_id=sample.sample_id,
            label=sample.label,
            status=Status.OK,
            score=0.0,
            reasons=("NO_DEFECT_EVIDENCE",),
            predicted=(),
        )
        for sample in samples
    ]


def _detector_outcomes(
    samples: Sequence[CachedSample],
    *,
    raw_root: Path,
    config: AppConfig,
    detector: Any,
) -> list[SampleOutcome]:
    outcomes: list[SampleOutcome] = []
    for sample in samples:
        try:
            gray = _read_gray(raw_root / sample.image_path)
        except ValueError:
            outcomes.append(
                SampleOutcome(
                    sample_id=sample.sample_id,
                    label=sample.label,
                    status=Status.ERROR,
                    score=0.0,
                    reasons=(),
                    predicted=(),
                )
            )
            continue
        detection = detector(gray, config)
        status = Status.NG if detection.components else Status.OK
        outcomes.append(
            SampleOutcome(
                sample_id=sample.sample_id,
                label=sample.label,
                status=status,
                score=float(detection.score),
                reasons=(
                    ("DEFECT_EVIDENCE_STRONG",)
                    if status is Status.NG
                    else ("NO_DEFECT_EVIDENCE",)
                ),
                predicted=detection.components,
            )
        )
    return outcomes


BASELINE_NOTES: dict[str, tuple[str, ...]] = {
    B0_NAME: ("항상 OK. 유병률/하한 기준선이며 마스크를 내지 않는다.",),
    B1_NAME: (
        "ROI + median3 + Otsu(ROI 내부) + 최소 면적. 독립 구현(파이프라인 미사용).",
        "REVIEW를 내지 않는 이진 기준선이다.",
    ),
    B2_NAME: (
        "median31 배경 보정 + 고정 임계 + 최소 면적. ROI·정규화 없음. 독립 구현.",
        "REVIEW를 내지 않는 이진 기준선이다.",
    ),
}


def evaluate_baselines(
    samples: Sequence[CachedSample],
    config: AppConfig,
    *,
    raw_root: str | Path = Path("data/raw/kolektorsdd2"),
    which: Sequence[str] = (B0_NAME, B1_NAME, B2_NAME),
) -> list[VariantReport]:
    """B0/B1/B2를 같은 표본 위에서 평가한다."""

    root = Path(raw_root)
    reports: list[VariantReport] = []
    for name in which:
        if name == B0_NAME:
            outcomes = _b0(samples)
            config_hash = "b0-fixed"
        elif name == B1_NAME:
            outcomes = _detector_outcomes(
                samples, raw_root=root, config=config, detector=b1_detect
            )
            config_hash = f"b1-{config.config_hash[:8]}"
        elif name == B2_NAME:
            outcomes = _detector_outcomes(
                samples, raw_root=root, config=config, detector=b2_detect
            )
            config_hash = f"b2-{config.config_hash[:8]}"
        else:
            raise ValueError(f"알 수 없는 기준선입니다: {name!r}")
        reports.append(
            summarise(
                name,
                samples,
                outcomes,
                config_hash=config_hash,
                overrides={
                    "baseline": {
                        "dark_floor": config.roi.dark_floor,
                        "min_area_px": config.candidates.min_area_px,
                        "min_contrast_gray": config.candidates.min_contrast_gray,
                    }
                },
                notes=BASELINE_NOTES[name],
            )
        )
    return reports


__all__ = [
    "B0_NAME",
    "B1_NAME",
    "B2_NAME",
    "BASELINE_NOTES",
    "BaselineDetection",
    "b1_detect",
    "b2_detect",
    "evaluate_baselines",
]
