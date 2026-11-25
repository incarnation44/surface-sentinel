"""Windows 비ASCII(한글 등) 경로에서도 깨지지 않는 안전한 이미지 읽기·원자적 PNG 저장 모듈.

``cv2.imread``/``cv2.imwrite`` 는 Windows 에서 비ASCII(한글 등) 경로를
열지 못한다. 읽기는 ``np.fromfile`` + ``cv2.imdecode``, 쓰기는
``cv2.imencode`` + pid/uuid 임시 파일 + fsync + ``os.replace`` 원자 교체로
대체한다. 실패 시 임시 파일을 남기지 않는다.

계약은 OpenCV 와 동일하게 유지한다: 읽기 실패는 ``None``, 쓰기 실패는
``False`` 를 돌려준다(호출부의 기존 분기를 그대로 쓴다).
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """경로 인코딩과 무관하게 이미지를 디코드한다. 실패 시 ``None``.

    디코드 전 픽셀 상한 가드는 호출부(``io.image_guard``)가 별도로 적용한다.
    """

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_png_atomic(path: str | Path, image: np.ndarray) -> bool:
    """PNG 를 원자적으로 기록한다. 성공 ``True``, 실패 ``False``.

    pid/uuid 임시 파일에 쓴 뒤 fsync + ``os.replace`` 로 게시하고,
    실패 시 임시 파일을 정리한다(부분 PNG 가 정상 산출물로 보이지 않는다).
    """

    target = Path(path)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return False
    tmp = target.parent / f".tmp-{target.name}-{os.getpid()}-{uuid.uuid4().hex}.png"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as handle:
            handle.write(buf.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        return True
    except OSError:
        return False
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


__all__ = ["imread_unicode", "imwrite_png_atomic"]