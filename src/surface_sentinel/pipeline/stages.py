"""고정 10단계 검사 파이프라인의 단계 함수.

각 함수는 한 단계만 수행하며 오케스트레이션은 ``inspect`` 모듈이 맡는다.
사람이 읽는 예외 메시지는 한국어, 식별자와 상태 코드는 영어 원형을 쓴다.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace as dataclass_replace
from pathlib import Path

import cv2
import numpy as np

from ..config import REVIEW_BAND_MAX, AppConfig
from ..contracts import (
    ComponentFeature,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    Stage,
    Status,
)
from .errors import PipelineError
from .types import (
    AnalyzeResult,
    CandidateResult,
    DecisionResult,
    InputMetadata,
    LoadedImage,
    PreprocessResult,
    RefinedComponent,
    RefineResult,
    RoiResult,
    ScoreResult,
)

VALID_LABELS = ("unknown", "defect", "good")
VALID_SPLITS = ("tune", "test", "official_test")
SEALED_SPLITS = frozenset({"test", "official_test"})

#: 경계 링 계산에 쓰는 침식 커널 크기. 설계 고정값이다.
BORDER_RING_KERNEL = 3
#: refine 단계 재임계 뒤 적용하는 고정 open 커널 크기. 설계 고정값이다.
REFINE_OPEN_KERNEL = 3


def _ellipse(size: int) -> np.ndarray:
    kernel = max(1, int(size))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        channels = int(image.shape[2])
        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"지원하지 않는 이미지 shape입니다: {image.shape}")


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        channels = int(image.shape[2])
        if channels == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        if channels == 3:
            return image
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"지원하지 않는 이미지 shape입니다: {image.shape}")


def _channels(image: np.ndarray) -> int:
    if image.ndim == 2:
        return 1
    if image.ndim == 3 and int(image.shape[2]) in (1, 3, 4):
        return int(image.shape[2])
    raise ValueError(f"지원하지 않는 이미지 shape입니다: {image.shape}")


def _read_image_file(path: Path, stage: Stage) -> tuple[np.ndarray, str]:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise PipelineError(
            ErrorCode.E_INPUT_DECODE,
            stage,
            f"이미지 파일을 읽지 못했습니다: {path}",
        ) from exc
    if buffer.size == 0:
        raise PipelineError(
            ErrorCode.E_INPUT_DECODE,
            stage,
            f"이미지 파일이 비어 있습니다: {path}",
        )
    digest = hashlib.sha256(buffer.tobytes()).hexdigest()
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PipelineError(
            ErrorCode.E_INPUT_DECODE,
            stage,
            f"이미지 디코드에 실패했습니다: {path}",
        )
    return image, digest


def _mask_defect_pixels(mask: np.ndarray) -> int:
    if mask.ndim == 2:
        return int(np.count_nonzero(mask))
    if mask.ndim == 3:
        return int(np.count_nonzero(np.any(mask != 0, axis=2)))
    raise PipelineError(
        ErrorCode.E_MASK_SHAPE,
        Stage.VALIDATE_INPUT,
        f"지원하지 않는 마스크 차원입니다: shape={mask.shape}",
    )


def validate_input(
    image_path: str | Path,
    *,
    mask_path: str | Path | None,
    label: str,
    split: str,
    config: AppConfig,
    allow_sealed: bool = False,
) -> tuple[InputMetadata, np.ndarray]:
    """입력 계약을 검사하고 디코드된 원본 배열을 돌려준다.

    봉인 split 거부는 파일을 읽기 전에 수행한다. ``allow_sealed=True``는
    T08 봉인 측정 경로가 승인했을 때만 전달한다(기본 False = 거부 유지).
    """

    if label not in VALID_LABELS:
        raise ValueError(f"알 수 없는 label입니다: {label!r}")
    if split not in VALID_SPLITS:
        raise ValueError(f"알 수 없는 split입니다: {split!r}")
    if split in SEALED_SPLITS and not allow_sealed:
        raise PipelineError(
            ErrorCode.E_SEALED_DENIED,
            Stage.VALIDATE_INPUT,
            f"봉인된 split은 inspect에서 열 수 없습니다: {split}",
        )

    source = Path(image_path)
    if not source.exists() or not source.is_file():
        raise PipelineError(
            ErrorCode.E_INPUT_MISSING,
            Stage.VALIDATE_INPUT,
            f"이미지 파일을 찾을 수 없습니다: {source}",
        )

    image, image_sha256 = _read_image_file(source, Stage.VALIDATE_INPUT)
    try:
        gray = _to_gray(image)
        channels = _channels(image)
    except ValueError as exc:
        raise PipelineError(
            ErrorCode.E_INPUT_DECODE,
            Stage.VALIDATE_INPUT,
            str(exc),
        ) from exc

    height = int(gray.shape[0])
    width = int(gray.shape[1])
    if width < config.input.min_width or height < config.input.min_height:
        raise PipelineError(
            ErrorCode.E_INPUT_SIZE,
            Stage.VALIDATE_INPUT,
            (
                f"이미지 크기가 최소값보다 작습니다: {width}x{height} "
                f"(최소 {config.input.min_width}x{config.input.min_height})"
            ),
        )

    std = float(np.std(gray, ddof=0))
    if std < config.input.min_std:
        raise PipelineError(
            ErrorCode.E_INPUT_UNIFORM,
            Stage.VALIDATE_INPUT,
            f"이미지 표준편차가 너무 낮습니다: {std:.4f} (최소 {config.input.min_std})",
        )

    mask_sha256 = ""
    defect_pixels = 0
    mask_source = ""
    if mask_path is not None:
        mask_file = Path(mask_path)
        mask_source = str(mask_file)
        if not mask_file.exists() or not mask_file.is_file():
            raise PipelineError(
                ErrorCode.E_INPUT_MISSING,
                Stage.VALIDATE_INPUT,
                f"마스크 파일을 찾을 수 없습니다: {mask_file}",
            )
        mask, mask_sha256 = _read_image_file(mask_file, Stage.VALIDATE_INPUT)
        if mask.ndim < 2 or tuple(mask.shape[:2]) != (height, width):
            raise PipelineError(
                ErrorCode.E_MASK_SHAPE,
                Stage.VALIDATE_INPUT,
                (
                    f"마스크 크기가 이미지와 다릅니다: 이미지={(height, width)} "
                    f"마스크={mask.shape[:2]}"
                ),
            )
        defect_pixels = _mask_defect_pixels(mask)
        if label == "defect" and defect_pixels == 0:
            raise PipelineError(
                ErrorCode.E_MASK_MISMATCH,
                Stage.VALIDATE_INPUT,
                "label=defect인데 마스크에 결함 픽셀이 없습니다.",
            )
        if label == "good" and defect_pixels > 0:
            raise PipelineError(
                ErrorCode.E_MASK_MISMATCH,
                Stage.VALIDATE_INPUT,
                "label=good인데 마스크에 결함 픽셀이 있습니다.",
            )

    metadata = InputMetadata(
        image_path=str(source),
        mask_path=mask_source,
        label=label,
        split=split,
        width=width,
        height=height,
        channels=channels,
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        defect_pixels=defect_pixels,
    )
    return metadata, image


def load_image(decoded: np.ndarray, metadata: InputMetadata) -> LoadedImage:
    """검증 단계에서 디코드한 배열에서 BGR과 회색조를 만든다."""

    try:
        gray = _to_gray(decoded)
        bgr = _to_bgr(decoded)
    except ValueError as exc:
        raise PipelineError(
            ErrorCode.E_INPUT_DECODE,
            Stage.LOAD_IMAGE,
            str(exc),
        ) from exc
    if gray.shape[:2] != (metadata.height, metadata.width):
        raise PipelineError(
            ErrorCode.E_INPUT_DECODE,
            Stage.LOAD_IMAGE,
            (
                "디코드된 이미지 크기가 검증 메타데이터와 다릅니다: "
                f"메타데이터={(metadata.height, metadata.width)} 실제={gray.shape[:2]}"
            ),
        )
    return LoadedImage(
        bgr=np.ascontiguousarray(bgr),
        gray=np.ascontiguousarray(gray),
    )


def _empty_roi(mode: str) -> RoiResult:
    return RoiResult(
        mode=mode,
        mask=np.zeros((1, 1), dtype=np.uint8),
        bbox=(0, 0, 0, 0),
        area_px=0,
        area_ratio=0.0,
        uncertain=True,
    )


def _prepare_auto_roi(gray: np.ndarray, config: AppConfig) -> RoiResult:
    height, width = gray.shape[:2]
    surface = (gray > config.roi.dark_floor).astype(np.uint8) * 255
    closed = cv2.morphologyEx(
        surface,
        cv2.MORPH_CLOSE,
        _ellipse(config.roi.close_kernel),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        empty = _empty_roi("auto")
        return RoiResult(
            mode=empty.mode,
            mask=np.zeros((height, width), dtype=np.uint8),
            bbox=empty.bbox,
            area_px=empty.area_px,
            area_ratio=empty.area_ratio,
            uncertain=empty.uncertain,
        )

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((height, width), dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    eroded = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=2)

    area_px = int(np.count_nonzero(eroded))
    if area_px == 0:
        return RoiResult(
            mode="auto",
            mask=eroded,
            bbox=(0, 0, 0, 0),
            area_px=0,
            area_ratio=0.0,
            uncertain=True,
        )

    x, y, w, h = cv2.boundingRect(eroded)
    area_ratio = area_px / float(height * width)
    uncertain = area_ratio < config.roi.min_area_ratio

    margin = config.roi.margin_px
    left = max(int(x) - margin, 0)
    top = max(int(y) - margin, 0)
    right = min(int(x) + int(w) + margin, width)
    bottom = min(int(y) + int(h) + margin, height)
    bbox = (left, top, max(right - left, 0), max(bottom - top, 0))
    return RoiResult(
        mode="auto",
        mask=eroded,
        bbox=bbox,
        area_px=area_px,
        area_ratio=area_ratio,
        uncertain=uncertain,
    )


def _prepare_fixed_roi(gray: np.ndarray, config: AppConfig) -> RoiResult:
    height, width = gray.shape[:2]
    inset = int(config.roi.fixed_margin_px)
    rect_width = width - 2 * inset
    rect_height = height - 2 * inset
    mask = np.zeros((height, width), dtype=np.uint8)
    valid = rect_width > 0 and rect_height > 0
    if valid:
        if inset == 0:
            mask[:, :] = 255
        else:
            mask[inset : inset + rect_height, inset : inset + rect_width] = 255
    area_px = int(np.count_nonzero(mask))
    area_ratio = area_px / float(height * width) if height * width else 0.0
    uncertain = (not valid) or area_ratio < config.roi.min_area_ratio
    bbox = (inset, inset, rect_width, rect_height)
    return RoiResult(
        mode="fixed",
        mask=mask,
        bbox=bbox,
        area_px=area_px,
        area_ratio=area_ratio,
        uncertain=uncertain,
    )


def prepare_roi(gray: np.ndarray, config: AppConfig) -> RoiResult:
    """표면 영역을 찾아 ROI 마스크와 경계 상자를 만든다.

    ROI가 비어도 예외를 내지 않는다. 이 경우 ``uncertain=True``이며 이후
    단계는 0 응답으로 진행한다.
    """

    if config.roi.mode == "fixed":
        return _prepare_fixed_roi(gray, config)
    return _prepare_auto_roi(gray, config)


def _background_kernel(height: int, width: int, config: AppConfig) -> int:
    ratio = config.preprocess.bg_kernel_ratio
    kernel = int(round(ratio * min(height, width)))
    kernel = max(kernel, config.preprocess.bg_kernel_min)
    kernel = min(kernel, config.preprocess.bg_kernel_max)
    if kernel % 2 == 0:
        kernel += 1
    limit = min(height, width)
    if limit % 2 == 0:
        limit -= 1
    kernel = min(kernel, limit)
    return max(kernel, 3)


def _blank_float(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.float32)


def preprocess(gray: np.ndarray, roi: RoiResult, config: AppConfig) -> PreprocessResult:
    """median, black-hat/top-hat, 강건 z 정규화, 품질 플래그를 계산한다."""

    height, width = gray.shape[:2]
    cfg = config.preprocess
    if not cfg.use_dark_channel and not cfg.use_bright_channel:
        raise PipelineError(
            ErrorCode.E_CONFIG_INVALID,
            Stage.PREPROCESS,
            "preprocess 채널이 모두 비활성화되었습니다.",
        )

    gray_blur = (
        cv2.medianBlur(gray, int(cfg.median_kernel)) if cfg.median_kernel > 1 else gray
    )

    kernel_size = _background_kernel(height, width, config)
    kernel = _ellipse(kernel_size)

    dark = _blank_float(height, width)
    bright = _blank_float(height, width)
    if cfg.use_dark_channel:
        dark = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    if cfg.use_bright_channel:
        bright = cv2.morphologyEx(gray_blur, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    if cfg.use_dark_channel and cfg.use_bright_channel:
        response = np.maximum(dark, bright)
    elif cfg.use_dark_channel:
        response = dark
    else:
        response = bright
    response = np.ascontiguousarray(response, dtype=np.float32)

    roi_bool = roi.mask > 0
    response[~roi_bool] = 0.0

    z_map = _blank_float(height, width)
    values = response[roi_bool]
    if values.size == 0:
        z_map[:] = 0.0
    else:
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        sigma = 1.4826 * mad
        denom = max(sigma, float(cfg.noise_floor_gray))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_map = (response / denom).astype(np.float32)
        z_map[~roi_bool] = 0.0

    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(z_map)):
        raise PipelineError(
            ErrorCode.E_PROCESS_INTERNAL,
            Stage.PREPROCESS,
            "전처리 응답 또는 z 지도에 유한하지 않은 값이 있습니다.",
        )

    if roi_bool.any():
        quality_values = gray[roi_bool]
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_values = laplacian[roi_bool]
    else:
        quality_values = gray.reshape(-1)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_values = laplacian.reshape(-1)

    blur_variance = float(np.var(laplacian_values, ddof=0))
    mean_gray = float(np.mean(quality_values))
    saturated = (quality_values == 0) | (quality_values == 255)
    saturated_ratio = float(np.count_nonzero(saturated) / quality_values.size)
    low_quality = (
        blur_variance < config.quality.blur_min_var
        or mean_gray < config.quality.dark_mean_min
        or mean_gray > config.quality.bright_mean_max
        or saturated_ratio > config.quality.saturated_ratio_max
    )
    flags: tuple[ReasonCode, ...] = (ReasonCode.INPUT_QUALITY_LOW,) if low_quality else ()
    quality = QualityInfo(
        blur_variance=blur_variance,
        mean_gray=mean_gray,
        saturated_ratio=saturated_ratio,
        flags=flags,
    )
    return PreprocessResult(
        response=response,
        z_map=z_map,
        quality=quality,
        bg_kernel=kernel_size,
    )


def generate_candidates(
    z_map: np.ndarray,
    response: np.ndarray,
    roi_mask: np.ndarray,
    config: AppConfig,
) -> CandidateResult:
    """z 임계값과 대비 하한을 동시에 통과한 후보 픽셀을 만든다."""

    cfg = config.candidates
    candidate = (
        (z_map >= cfg.z_threshold)
        & (response >= cfg.min_contrast_gray)
        & (roi_mask > 0)
    )
    mask = candidate.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(cfg.open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(cfg.close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(cfg.merge_kernel))
    mask[roi_mask == 0] = 0
    return CandidateResult(mask=np.ascontiguousarray(mask))


def _component_evidence(
    area_px: int,
    mean_contrast_gray: float,
    config: AppConfig,
) -> float:
    area_norm = min(float(area_px) / config.scoring.ng_area_px, 1.5)
    contrast_norm = min(mean_contrast_gray / config.scoring.ng_contrast_gray, 1.5)
    return float(min(math.sqrt(area_norm * contrast_norm), 1.5))


def refine_mask(
    candidate_mask: np.ndarray,
    z_map: np.ndarray,
    response: np.ndarray,
    roi_mask: np.ndarray,
    config: AppConfig,
) -> RefineResult:
    """후보를 재임계하고 연결 요소를 정제해 상위 후보만 남긴다."""

    cfg = config.candidates
    refined = (
        (candidate_mask > 0)
        & (z_map >= cfg.z_threshold)
        & (response >= cfg.min_contrast_gray)
    ).astype(np.uint8) * 255
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        _ellipse(REFINE_OPEN_KERNEL),
    )

    roi_bool = roi_mask > 0
    roi_eroded = cv2.erode(
        (roi_bool.astype(np.uint8) * 255),
        np.ones((BORDER_RING_KERNEL, BORDER_RING_KERNEL), np.uint8),
        iterations=1,
    )
    ring = roi_bool & (roi_eroded == 0)
    distance = cv2.distanceTransform(
        (roi_bool.astype(np.uint8) * 255),
        cv2.DIST_L2,
        5,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    components: list[RefinedComponent] = []
    total_components = 0
    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < cfg.min_area_px:
            continue
        total_components += 1
        component_mask = labels == label
        values = response[component_mask]
        mean_contrast = float(np.mean(values)) if values.size else 0.0
        max_contrast = float(np.max(values)) if values.size else 0.0
        integrated_contrast = float(np.sum(values)) if values.size else 0.0
        touches = bool(np.any(component_mask & ring))
        border_distance = float(np.min(distance[component_mask]))
        bbox = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        components.append(
            RefinedComponent(
                index=0,
                area_px=area_px,
                bbox=bbox,
                mask=(component_mask.astype(np.uint8) * 255),
                mean_contrast_gray=mean_contrast,
                max_contrast_gray=max_contrast,
                integrated_contrast=integrated_contrast,
                border_distance_px=border_distance,
                touches_roi_border=touches,
                evidence=_component_evidence(area_px, mean_contrast, config),
            )
        )

    components.sort(
        key=lambda item: (item.evidence, item.area_px, -item.bbox[1], -item.bbox[0]),
        reverse=True,
    )
    truncated = total_components > cfg.max_candidates
    kept = components[: cfg.max_candidates]

    combined = np.zeros_like(refined)
    indexed: list[RefinedComponent] = []
    for index, component in enumerate(kept):
        indexed.append(dataclass_replace(component, index=index))
        combined[component.mask > 0] = 255
    return RefineResult(
        components=tuple(indexed),
        combined_mask=np.ascontiguousarray(combined),
        truncated=truncated,
        total_components=total_components,
    )


def analyze_components(
    refined: RefineResult,
    roi: RoiResult,
    response: np.ndarray,
    config: AppConfig,
) -> AnalyzeResult:
    """유지된 컴포넌트마다 계약 특징을 계산한다."""

    features: list[ComponentFeature] = []
    roi_area = roi.area_px
    for component in refined.components:
        width = component.bbox[2]
        height = component.bbox[3]
        area_ratio = float(component.area_px) / roi_area if roi_area > 0 else 0.0
        aspect_ratio = max(width, height) / max(min(width, height), 1)

        contours, _ = cv2.findContours(
            component.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        solidity = 0.0
        elongation = 1.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0.0:
                solidity = min(max(contour_area / hull_area, 0.0), 1.0)
            (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
            long_side = max(float(rect_width), float(rect_height))
            short_side = min(float(rect_width), float(rect_height))
            elongation = long_side / max(short_side, 1e-6)

        features.append(
            ComponentFeature(
                index=component.index,
                area_px=component.area_px,
                bbox=component.bbox,
                area_ratio=area_ratio,
                aspect_ratio=float(aspect_ratio),
                elongation=float(elongation),
                solidity=float(solidity),
                mean_contrast_gray=component.mean_contrast_gray,
                max_contrast_gray=component.max_contrast_gray,
                integrated_contrast=component.integrated_contrast,
                border_distance_px=component.border_distance_px,
                touches_roi_border=component.touches_roi_border,
                evidence=_component_evidence(
                    component.area_px,
                    component.mean_contrast_gray,
                    config,
                ),
            )
        )
    return AnalyzeResult(components=tuple(features))


def calculate_score(components: tuple[ComponentFeature, ...], config: AppConfig) -> ScoreResult:
    """최소 타당성을 통과한 컴포넌트의 최대 증거를 이미지 점수로 쓴다."""

    valid = tuple(
        component
        for component in components
        if component.area_px >= config.candidates.min_area_px
        and component.mean_contrast_gray >= config.candidates.min_contrast_gray
    )
    score = max((component.evidence for component in valid), default=0.0)
    if not math.isfinite(score):
        raise PipelineError(
            ErrorCode.E_PROCESS_INTERNAL,
            Stage.CALCULATE_SCORE,
            "이미지 점수가 유한하지 않습니다.",
        )
    return ScoreResult(score=float(score), valid_components=valid)


def decide_status(
    scored: ScoreResult,
    quality: QualityInfo,
    roi: RoiResult,
    refined: RefineResult,
    config: AppConfig,
) -> DecisionResult:
    """설계 3.2의 우선순위 진리표를 그대로 적용한다."""

    scoring = config.scoring
    decision = config.decision
    score = scored.score
    valid = scored.valid_components

    multi_weak = (
        sum(
            1
            for component in valid
            if max(REVIEW_BAND_MAX - scoring.multi_candidate_margin, 0.0)
            <= component.evidence
            < scoring.ng_min
        )
        >= 2
    )
    border_touch = any(component.touches_roi_border for component in valid)
    quality_low = bool(quality.flags)
    roi_uncertain = bool(roi.uncertain)
    border_flag = border_touch and decision.border_touch_escalates
    truncated_flag = refined.truncated and decision.truncation_escalates
    multi_flag = multi_weak and decision.multi_candidate_escalates
    effective_flags = (
        quality_low or roi_uncertain or border_flag or truncated_flag or multi_flag
    )

    if valid and score >= scoring.ng_min:
        return DecisionResult(Status.NG, (ReasonCode.DEFECT_EVIDENCE_STRONG,))
    if not valid and not effective_flags:
        return DecisionResult(Status.OK, (ReasonCode.NO_DEFECT_EVIDENCE,))
    if quality_low:
        return DecisionResult(Status.REVIEW, (ReasonCode.INPUT_QUALITY_LOW,))
    if roi_uncertain:
        return DecisionResult(Status.REVIEW, (ReasonCode.ROI_UNCERTAIN,))
    if border_flag and score >= REVIEW_BAND_MAX:
        return DecisionResult(Status.REVIEW, (ReasonCode.BORDER_TOUCHING_CANDIDATE,))
    if truncated_flag:
        return DecisionResult(Status.REVIEW, (ReasonCode.CANDIDATE_TRUNCATED,))
    if multi_flag:
        return DecisionResult(Status.REVIEW, (ReasonCode.MULTIPLE_WEAK_CANDIDATES,))
    return DecisionResult(Status.REVIEW, (ReasonCode.EVIDENCE_IN_REVIEW_BAND,))


__all__ = [
    "BORDER_RING_KERNEL",
    "REFINE_OPEN_KERNEL",
    "SEALED_SPLITS",
    "VALID_LABELS",
    "VALID_SPLITS",
    "analyze_components",
    "calculate_score",
    "decide_status",
    "generate_candidates",
    "load_image",
    "prepare_roi",
    "preprocess",
    "refine_mask",
    "validate_input",
]
