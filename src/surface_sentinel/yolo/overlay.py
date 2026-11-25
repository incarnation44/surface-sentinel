"""YOLO 탐지 결과 시각화 오버레이 PNG 생성 모듈.

원본 이미지 위에 바운딩 박스, 신뢰도 점수, 세그멘테이션 마스크를 반투명 합성해 저장한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .inference import Detection

DEFAULT_COLOR = (0, 0, 255)
#: 마스크 반투명 합성 비율(0~1).
DEFAULT_MASK_ALPHA = 0.4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_THICKNESS = 1


def draw_detections(
    bgr: np.ndarray,
    detections: Sequence[Detection],
    *,
    color: tuple[int, int, int] = DEFAULT_COLOR,
    mask_alpha: float = DEFAULT_MASK_ALPHA,
    thickness: int = 2,
) -> np.ndarray:
    """박스와 점수 텍스트를 그리고, seg 마스크는 반투명으로 합성한 새 배열을 돌려준다.

    입력 배열은 변경하지 않는다. 마스크가 있으면 ``cv2.addWeighted`` 로
    색을 입히고, 그 위에 박스·텍스트를 얹는다.
    """

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"BGR 3채널 이미지가 필요합니다: shape={bgr.shape}")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError(f"mask_alpha 는 0~1 입니다: {mask_alpha}")

    canvas = bgr.copy()
    masks = [detection.mask for detection in detections if detection.mask is not None]
    if masks:
        layer = canvas.copy()
        for mask in masks:
            if mask.shape[:2] != canvas.shape[:2]:
                raise ValueError(
                    f"마스크 크기가 이미지와 다릅니다: mask={mask.shape} image={canvas.shape}"
                )
            layer[mask > 0] = color
        canvas = cv2.addWeighted(canvas, 1.0 - mask_alpha, layer, mask_alpha, 0.0)

    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text_origin = (max(0, x1), max(8, y1 - 4))
        cv2.putText(
            canvas,
            f"{detection.score:.2f}",
            text_origin,
            FONT,
            FONT_SCALE,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
    return canvas


def save_overlay(
    bgr: np.ndarray,
    detections: Sequence[Detection],
    path: str | Path,
    *,
    color: tuple[int, int, int] = DEFAULT_COLOR,
    mask_alpha: float = DEFAULT_MASK_ALPHA,
    thickness: int = 2,
) -> Path:
    """오버레이를 PNG 로 저장하고 경로를 돌려준다(부모 디렉터리 자동 생성)."""

    target = Path(path)
    if target.suffix.lower() != ".png":
        target = target.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_detections(
        bgr,
        detections,
        color=color,
        mask_alpha=mask_alpha,
        thickness=thickness,
    )
    from ..io.imageio import imwrite_png_atomic  # noqa: PLC0415

    if not imwrite_png_atomic(target, overlay):
        raise OSError(f"오버레이 저장 실패: {target}")
    return target


__all__ = ["DEFAULT_COLOR", "DEFAULT_MASK_ALPHA", "draw_detections", "save_overlay"]
