"""오버레이와 결과 JSON, 선택적 단계 이미지의 원자 저장."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ..config import AppConfig
from ..contracts import STAGE_ORDER, ErrorCode, SampleResult, Stage
from .errors import PipelineError
from .types import RefinedComponent

_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

OVERLAY_COLOR_ROI = (0, 200, 0)
OVERLAY_COLOR_COMPONENT = (0, 0, 255)
OVERLAY_COLOR_BBOX = (0, 220, 220)
OVERLAY_COLOR_TEXT = (255, 255, 255)
OVERLAY_COLOR_TEXT_BG = (0, 0, 0)


def sanitize_sample_id(sample_id: str) -> str:
    """파일 이름에 안전한 sample_id 조각을 만든다."""

    cleaned = _FILENAME_UNSAFE.sub("_", sample_id).strip("._")
    return cleaned or "sample"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise PipelineError(
            ErrorCode.E_STORAGE_WRITE,
            Stage.SAVE_EVIDENCE,
            "PNG 인코드에 실패했습니다.",
        )
    return buffer.tobytes()


def _draw_overlay(
    bgr: np.ndarray,
    roi_mask: np.ndarray,
    components: tuple[RefinedComponent, ...],
    status_text: str,
    scale: int,
) -> np.ndarray:
    canvas = bgr.copy()
    roi_contours, _ = cv2.findContours(
        roi_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if roi_contours:
        cv2.drawContours(canvas, roi_contours, -1, OVERLAY_COLOR_ROI, 1)

    for component in components:
        contours, _ = cv2.findContours(
            component.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            cv2.drawContours(canvas, contours, -1, OVERLAY_COLOR_COMPONENT, 1)
        x, y, width, height = component.bbox
        cv2.rectangle(
            canvas,
            (x, y),
            (x + width - 1, y + height - 1),
            OVERLAY_COLOR_BBOX,
            1,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        status_text,
        font,
        font_scale,
        thickness,
    )
    cv2.rectangle(
        canvas,
        (2, 2),
        (2 + text_width + 4, 2 + text_height + baseline + 4),
        OVERLAY_COLOR_TEXT_BG,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        canvas,
        status_text,
        (4, 4 + text_height),
        font,
        font_scale,
        OVERLAY_COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )
    if scale > 1:
        canvas = cv2.resize(
            canvas,
            (canvas.shape[1] * scale, canvas.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return canvas


def _normalized_response(response: np.ndarray) -> np.ndarray:
    finite = response[np.isfinite(response)]
    maximum = float(np.max(finite)) if finite.size else 0.0
    if maximum <= 0.0:
        return np.zeros(response.shape, dtype=np.uint8)
    scaled = np.clip(response.astype(np.float32) * (255.0 / maximum), 0.0, 255.0)
    return scaled.astype(np.uint8)


def save_evidence(
    result: SampleResult,
    *,
    bgr: np.ndarray,
    roi_mask: np.ndarray,
    components: tuple[RefinedComponent, ...],
    response: np.ndarray,
    candidate_mask: np.ndarray,
    output_dir: str | Path,
    timing_ms: dict[str, float],
    config: AppConfig,
    save_stages: bool,
    stage_start: float,
) -> SampleResult:
    """오버레이/결과 JSON/선택 단계 이미지를 원자적으로 저장한다."""

    directory = Path(output_dir)
    safe_id = sanitize_sample_id(result.sample_id)
    overlay_path = directory / f"{safe_id}_overlay.png"
    result_path = directory / f"{safe_id}_result.json"

    try:
        status_text = f"{result.status.value} score={result.score:.4f}"
        overlay = _draw_overlay(
            bgr,
            roi_mask,
            components,
            status_text,
            int(config.artifacts.overlay_scale),
        )
        _atomic_write_bytes(overlay_path, _encode_png(overlay))

        if save_stages:
            _atomic_write_bytes(
                directory / f"{safe_id}_response.png",
                _encode_png(_normalized_response(response)),
            )
            _atomic_write_bytes(
                directory / f"{safe_id}_candidates.png",
                _encode_png(candidate_mask),
            )
    except OSError as exc:
        raise PipelineError(
            ErrorCode.E_STORAGE_WRITE,
            Stage.SAVE_EVIDENCE,
            f"증거 파일을 저장하지 못했습니다: {exc}",
        ) from exc

    timing_ms[Stage.SAVE_EVIDENCE.value] = (time.perf_counter() - stage_start) * 1000.0
    timing_ms["total"] = float(
        sum(float(timing_ms[stage.value]) for stage in STAGE_ORDER)
    )
    updated = replace(result, overlay_path=str(overlay_path), timing_ms=timing_ms)

    try:
        payload = (
            json.dumps(
                updated.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _atomic_write_bytes(result_path, payload.encode("utf-8"))
    except OSError as exc:
        raise PipelineError(
            ErrorCode.E_STORAGE_WRITE,
            Stage.SAVE_EVIDENCE,
            f"결과 JSON을 저장하지 못했습니다: {exc}",
        ) from exc

    return updated


__all__ = ["sanitize_sample_id", "save_evidence"]
