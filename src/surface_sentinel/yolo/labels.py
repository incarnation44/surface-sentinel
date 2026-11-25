"""GT 이진 마스크 연결 요소를 YOLO detection bbox 및 segmentation 폴리곤 라벨로 변환.

GT 마스크의 연결 요소를 컴포넌트로 추출해 detection용 bbox와
segmentation용 폴리곤을 만든다. 좌표는 이미지 크기로 정규화한다.
폴리곤이 퇴화(점·선)하면 bbox 네 꼭짓점으로 대체해 인스턴스 수를 보존한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: 단일 클래스(결함) 인덱스. ultralytics names와 일치해야 한다.
DEFECT_CLASS_INDEX = 0


@dataclass(frozen=True)
class MaskComponent:
    """마스크 연결 요소 하나. 좌표는 픽셀 단위."""

    bbox_xywh: tuple[int, int, int, int]
    polygon: tuple[tuple[int, int], ...]
    area_px: int


def mask_to_components(mask: np.ndarray, *, min_area_px: int = 1) -> list[MaskComponent]:
    """이진 마스크에서 연결 요소를 추출한다.

    ``min_area_px`` 미만 컴포넌트는 제외한다. 반환 순서는 bbox (y, x) 오름차순으로
    결정론적이다.
    """

    if mask.ndim != 2:
        raise ValueError(f"마스크는 2차원이어야 합니다: shape={mask.shape}")
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[MaskComponent] = []
    for index in range(1, count):  # 0은 배경
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_mask = (labels[y : y + h, x : x + w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon: tuple[tuple[int, int], ...] = ()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 3:
                polygon = tuple((int(px) + x, int(py) + y) for px, py in largest.reshape(-1, 2))
        if not polygon:
            polygon = ((x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1))
        components.append(MaskComponent(bbox_xywh=(x, y, w, h), polygon=polygon, area_px=area))
    components.sort(key=lambda c: (c.bbox_xywh[1], c.bbox_xywh[0]))
    return components


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def det_lines(components: list[MaskComponent], *, width: int, height: int) -> list[str]:
    """detection 라벨 줄: ``0 cx cy w h``(정규화, 소수 6자리)."""

    lines: list[str] = []
    for component in components:
        x, y, w, h = component.bbox_xywh
        cx = _clip01((x + w / 2) / width)
        cy = _clip01((y + h / 2) / height)
        nw = _clip01(w / width)
        nh = _clip01(h / height)
        lines.append(f"{DEFECT_CLASS_INDEX} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def seg_lines(components: list[MaskComponent], *, width: int, height: int) -> list[str]:
    """segmentation 라벨 줄: ``0 x1 y1 x2 y2 ...``(정규화, 소수 6자리)."""

    lines: list[str] = []
    for component in components:
        coords: list[str] = []
        for px, py in component.polygon:
            coords.append(f"{_clip01(px / width):.6f}")
            coords.append(f"{_clip01(py / height):.6f}")
        lines.append(f"{DEFECT_CLASS_INDEX} " + " ".join(coords))
    return lines


__all__ = [
    "DEFECT_CLASS_INDEX",
    "MaskComponent",
    "det_lines",
    "mask_to_components",
    "seg_lines",
]