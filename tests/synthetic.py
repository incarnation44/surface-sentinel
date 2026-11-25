"""테스트 공용 결정론적 합성 이미지·결함 마스크 생성기(실제 KSDD2 데이터 불필요).

실제 KSDD2 이미지는 사용하지 않는다. 모든 픽셀은 시드가 고정된 numpy/OpenCV
연산으로 만들며, 120x320 크기와 어두운 테두리, 표면 잡음(std≈2)을 기본값으로
쓴다. 이 모듈은 pytest가 수집하지 않는 이름이다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HEIGHT = 120
WIDTH = 320
BORDER = 8
SURFACE_MEAN = 120
SURFACE_NOISE_STD = 2.0


def surface_gray(
    seed: int = 0,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    border: int = BORDER,
    mean: int = SURFACE_MEAN,
    noise_std: float = SURFACE_NOISE_STD,
) -> np.ndarray:
    """어두운 테두리와 잡음 표면을 가진 회색조 이미지를 만든다."""

    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.uint8)
    inner_height = height - 2 * border
    inner_width = width - 2 * border
    if inner_height <= 0 or inner_width <= 0:
        return image
    noise = rng.normal(0.0, noise_std, (inner_height, inner_width))
    inner = np.clip(mean + noise, 0, 255).astype(np.uint8)
    image[border : border + inner_height, border : border + inner_width] = inner
    return image


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def fill_rect(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: int,
    *,
    gray: bool = False,
) -> np.ndarray:
    """이미지를 복사해 사각형 영역을 지정 값으로 채운다."""

    result = image.copy()
    if gray:
        result[y : y + height, x : x + width] = value
    else:
        result[y : y + height, x : x + width, :] = value
    return result


def rect_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255
    return mask


def add_dark_rect(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: int = 20,
) -> np.ndarray:
    gray = image.ndim == 2
    return fill_rect(image, x, y, width, height, value, gray=gray)


def add_bright_rect(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: int = 230,
) -> np.ndarray:
    gray = image.ndim == 2
    return fill_rect(image, x, y, width, height, value, gray=gray)


def add_defect_with_mask(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: int,
    *,
    bright: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    defect = add_bright_rect(image, x, y, width, height, value) if bright else add_dark_rect(
        image, x, y, width, height, value
    )
    return defect, rect_mask(image.shape, x, y, width, height)


def write_png(path: str | Path, image: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success, f"PNG 인코드 실패: {target}"
    buffer.tofile(str(target))
    return target


def good_image() -> np.ndarray:
    return to_bgr(surface_gray(seed=11))


def dark_defect_image() -> tuple[np.ndarray, np.ndarray]:
    image = to_bgr(surface_gray(seed=12))
    return add_defect_with_mask(image, 150, 50, 12, 12, 20)


def bright_defect_image() -> tuple[np.ndarray, np.ndarray]:
    image = to_bgr(surface_gray(seed=13))
    return add_defect_with_mask(image, 150, 50, 12, 12, 230, bright=True)


__all__ = [
    "BORDER",
    "HEIGHT",
    "SURFACE_MEAN",
    "SURFACE_NOISE_STD",
    "WIDTH",
    "add_bright_rect",
    "add_dark_rect",
    "add_defect_with_mask",
    "bright_defect_image",
    "dark_defect_image",
    "fill_rect",
    "good_image",
    "rect_mask",
    "surface_gray",
    "to_bgr",
    "write_png",
]
