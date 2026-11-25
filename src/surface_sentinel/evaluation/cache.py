"""평가용 파이프라인 1회 실행 캐시 — 후보 파라미터 재평가 시 영상처리를 반복하지 않기 위해 사용.

T05의 grid/ablation은 후보 config마다 ``calculate_score``·``decide_status``만
다시 부르면 된다. 그런데 이 두 함수는 ``ComponentFeature``(면적·평균 대비)와
decision 입력(품질 플래그, ``roi.uncertain``, ``refined.truncated``,
``touches_roi_border``)만 있으면 충분하다. 그래서 파이프라인 단계 함수를
직접 호출해 그 재료를 **샘플당 1회** 만들어 둔다(``inspect_image`` 재실행 금지).

캐시에는 판정에 쓰지 않고 지표에만 쓰는 것도 함께 넣는다.

* 컴포넌트 마스크(bbox로 잘라서) — 영역 지표(IoU)용
* GT 마스크의 8-연결 컴포넌트 — 영역 지표 분모용
* 오버레이 경로 — 실패 사례 색인용

마스크는 전부 bbox 크기로 잘라 저장한다(전체 크기 배열을 샘플 수만큼 들고
있으면 700장 × 145KB가 넘는다).
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import math
import os
import time
import uuid
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig
from ..contracts import STAGE_ORDER, SampleResult, Status
from ..io.image_guard import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    ImageHeaderError,
    ImagePixelLimitError,
    check_image_pixel_budget,
)
from ..io.manifest import ManifestRow
from ..pipeline.errors import PipelineError
from ..pipeline.evidence import sanitize_sample_id, save_evidence
from ..pipeline.stages import (
    analyze_components,
    calculate_score,
    decide_status,
    generate_candidates,
    load_image,
    prepare_roi,
    preprocess,
    refine_mask,
    validate_input,
)
from .sampling import SamplePlan, sample_ids

CACHE_VERSION = "t05-cache-v2"
_CACHE_MAGIC = b"SSCACHE2\n"
#: 캐시 압축 파일 크기 상한(128 MiB, 무제한 메모리 읽기 방어).
MAX_CACHE_COMPRESSED_BYTES = 128 * 1024 * 1024
#: 캐시 압축 해제 크기 상한(512 MiB, zlib 폭탄 방어).
MAX_CACHE_DECOMPRESSED_BYTES = 512 * 1024 * 1024
CACHE_DIR = Path("data/derived/t05_cache")
OVERLAY_DIR = Path("data/derived/t05_overlays")
DEFAULT_RAW_ROOT = Path("data/raw/kolektorsdd2")


@dataclass(frozen=True, eq=False)
class CachedComponent:
    """컴포넌트 하나. ``mask``는 ``bbox`` 크기로 잘린 uint8(0/255) 배열이다."""

    area_px: int
    bbox: tuple[int, int, int, int]
    mean_contrast_gray: float
    max_contrast_gray: float
    integrated_contrast: float
    border_distance_px: float
    touches_roi_border: bool
    mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_px": self.area_px,
            "bbox": list(self.bbox),
            "mean_contrast_gray": self.mean_contrast_gray,
            "max_contrast_gray": self.max_contrast_gray,
            "integrated_contrast": self.integrated_contrast,
            "border_distance_px": self.border_distance_px,
            "touches_roi_border": self.touches_roi_border,
        }


@dataclass(frozen=True, eq=False)
class CachedSample:
    """샘플 하나의 재계산 재료와 기준 config 판정 결과."""

    sample_id: str
    original_id: str
    label: str
    split: str
    source_split: str
    image_path: str
    mask_path: str
    defect_pixels: int
    status: Status
    score: float
    reasons: tuple[str, ...]
    quality_flags: tuple[str, ...]
    blur_variance: float
    mean_gray: float
    saturated_ratio: float
    roi_uncertain: bool
    truncated: bool
    components: tuple[CachedComponent, ...]
    gt_components: tuple[CachedComponent, ...]
    overlay: str
    error_code: str | None
    error_stage: str | None

    @property
    def gt_area_px(self) -> int:
        return sum(component.area_px for component in self.gt_components)


@dataclass(frozen=True)
class CacheMeta:
    """캐시 정체성. 이 값이 다르면 캐시를 쓰지 않는다."""

    version: str
    config_hash: str
    split: str
    split_sha256: str
    sample_ids_sha256: str
    samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.config_hash,
            "split": self.split,
            "split_sha256": self.split_sha256,
            "sample_ids_sha256": self.sample_ids_sha256,
            "samples": self.samples,
        }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(ids))


def cache_meta(plan: SamplePlan, config: AppConfig, *, split_sha256: str = "") -> CacheMeta:
    ids = sample_ids(plan)
    return CacheMeta(
        version=CACHE_VERSION,
        config_hash=config.config_hash,
        split=plan.split,
        split_sha256=split_sha256,
        sample_ids_sha256=_ids_sha256(ids),
        samples=len(ids),
    )


def cache_path(meta: CacheMeta, directory: str | Path = CACHE_DIR) -> Path:
    name = (
        f"{meta.version}-{meta.split}-{meta.config_hash[:12]}"
        f"-{meta.sample_ids_sha256[:12]}.sscache"
    )
    return Path(directory) / name


def _read_mask(path: Path) -> np.ndarray | None:
    try:
        check_image_pixel_budget(path)
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ImageHeaderError, ImagePixelLimitError):
        return None
    if buffer.size == 0:
        return None
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 3:
        decoded = np.any(decoded != 0, axis=2).astype(np.uint8) * 255
    return (decoded > 0).astype(np.uint8) * 255


def _resolve_raw_path(raw_root: Path, relative: str, *, field: str) -> Path:
    rel = Path(relative)
    if not relative or rel.is_absolute() or rel.anchor or rel.drive:
        raise ValueError(
            f"manifest {field}는 raw_root 내부 상대 경로만 허용합니다: {relative!r}"
        )
    base = raw_root.resolve()
    candidate = (base / rel).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(
            f"manifest {field}가 raw_root 밖을 가리킵니다: {relative!r}"
        )
    return candidate


def _cropped(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    return np.ascontiguousarray(mask[y : y + height, x : x + width])


def _gt_components(mask_path: str) -> tuple[CachedComponent, ...]:
    """GT 마스크의 8-연결 컴포넌트. 설계 §4.3의 GT 정의 그대로다."""

    if not mask_path:
        return ()
    mask = _read_mask(Path(mask_path))
    if mask is None:
        return ()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[CachedComponent] = []
    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px <= 0:
            continue
        bbox = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
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
                mask=_cropped(component_mask, bbox),
            )
        )
    return tuple(components)


def _error_sample(
    row: ManifestRow,
    *,
    split: str,
    error: PipelineError,
    mask_path: Path | None = None,
) -> CachedSample:
    return CachedSample(
        sample_id=row.sample_id,
        original_id=row.original_id,
        label=row.label,
        split=split,
        source_split=row.source_split,
        image_path=row.image_path,
        mask_path=row.mask_path,
        defect_pixels=row.defect_pixels,
        status=Status.ERROR,
        score=0.0,
        reasons=(),
        quality_flags=(),
        blur_variance=0.0,
        mean_gray=0.0,
        saturated_ratio=0.0,
        roi_uncertain=True,
        truncated=False,
        components=(),
        gt_components=_gt_components(str(mask_path)) if mask_path is not None else (),
        overlay="",
        error_code=str(error.error_code),
        error_stage=str(error.error_stage),
    )


def inspect_once(
    row: ManifestRow,
    config: AppConfig,
    *,
    raw_root: str | Path,
    split: str,
    overlay_dir: str | Path | None,
    allow_sealed: bool = False,
) -> CachedSample:
    """한 샘플을 10단계로 한 번 검사하고 재계산 재료를 남긴다."""

    root = Path(raw_root)
    image_path = _resolve_raw_path(root, row.image_path, field="image_path")
    mask_path = (
        _resolve_raw_path(root, row.mask_path, field="mask_path")
        if row.mask_path
        else None
    )
    try:
        metadata, decoded = validate_input(
            image_path,
            mask_path=mask_path,
            label=row.label,
            split=split,
            config=config,
            allow_sealed=allow_sealed,
        )
        loaded = load_image(decoded, metadata)
        roi = prepare_roi(loaded.gray, config)
        pre = preprocess(loaded.gray, roi, config)
        candidate = generate_candidates(pre.z_map, pre.response, roi.mask, config)
        refined = refine_mask(candidate.mask, pre.z_map, pre.response, roi.mask, config)
        analyzed = analyze_components(refined, roi, pre.response, config)
        scored = calculate_score(analyzed.components, config)
        decision = decide_status(scored, pre.quality, roi, refined, config)
    except PipelineError as exc:
        return _error_sample(row, split=split, error=exc, mask_path=mask_path)

    overlay = ""
    if overlay_dir is not None:
        timing = {stage.value: 0.0 for stage in STAGE_ORDER}
        timing["total"] = 0.0
        result = SampleResult(
            run_id=f"t05-{config.config_hash[:8]}",
            sample_id=row.sample_id,
            original_id=row.original_id,
            split=split,
            label=row.label,
            status=decision.status,
            score=scored.score,
            reasons=decision.reasons,
            components=analyzed.components,
            quality=pre.quality,
            roi=roi.to_info(),
            timing_ms=timing,
            input_sha256=metadata.image_sha256,
            mask_sha256=metadata.mask_sha256,
            overlay_path="",
            config_hash=config.config_hash,
        )
        saved = save_evidence(
            result,
            bgr=loaded.bgr,
            roi_mask=roi.mask,
            components=refined.components,
            response=pre.response,
            candidate_mask=candidate.mask,
            output_dir=overlay_dir,
            timing_ms=timing,
            config=config,
            save_stages=False,
            stage_start=time.perf_counter(),
        )
        overlay = saved.overlay_path
        stray = Path(overlay_dir) / f"{sanitize_sample_id(row.sample_id)}_result.json"
        if stray.is_file():
            stray.unlink()

    components: list[CachedComponent] = []
    for refined_component, feature in zip(
        refined.components, analyzed.components, strict=True
    ):
        components.append(
            CachedComponent(
                area_px=int(feature.area_px),
                bbox=tuple(int(value) for value in feature.bbox),  # type: ignore[arg-type]
                mean_contrast_gray=float(feature.mean_contrast_gray),
                max_contrast_gray=float(feature.max_contrast_gray),
                integrated_contrast=float(feature.integrated_contrast),
                border_distance_px=float(feature.border_distance_px),
                touches_roi_border=bool(feature.touches_roi_border),
                mask=_cropped(refined_component.mask, refined_component.bbox),
            )
        )

    return CachedSample(
        sample_id=row.sample_id,
        original_id=row.original_id,
        label=row.label,
        split=split,
        source_split=row.source_split,
        image_path=row.image_path,
        mask_path=row.mask_path,
        defect_pixels=row.defect_pixels,
        status=decision.status,
        score=float(scored.score),
        reasons=tuple(reason.value for reason in decision.reasons),
        quality_flags=tuple(flag.value for flag in pre.quality.flags),
        blur_variance=float(pre.quality.blur_variance),
        mean_gray=float(pre.quality.mean_gray),
        saturated_ratio=float(pre.quality.saturated_ratio),
        roi_uncertain=bool(roi.uncertain),
        truncated=bool(refined.truncated),
        components=tuple(components),
        gt_components=_gt_components(str(mask_path)) if mask_path else (),
        overlay=overlay,
        error_code=None,
        error_stage=None,
    )


def build_cache(
    plan: SamplePlan,
    config: AppConfig,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    overlay_dir: str | Path | None = None,
    progress: Any = None,
    allow_sealed: bool = False,
) -> tuple[CachedSample, ...]:
    """표본 전체를 순서대로 검사한다. 순서는 표본 순서 그대로(결정론)."""

    samples: list[CachedSample] = []
    total = len(plan.rows)
    for index, row in enumerate(plan.rows, start=1):
        samples.append(
            inspect_once(
                row,
                config,
                raw_root=raw_root,
                split=plan.split,
                overlay_dir=overlay_dir,
                allow_sealed=allow_sealed,
            )
        )
        if progress is not None:
            progress(index, total, row.sample_id)
    return tuple(samples)


def _encode_component(component: CachedComponent) -> dict[str, Any]:
    mask_u8 = np.ascontiguousarray(component.mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError(f"캐시 마스크는 2차원이어야 합니다: shape={mask_u8.shape}")
    return {
        "area_px": int(component.area_px),
        "bbox": [int(v) for v in component.bbox],
        "mean_contrast_gray": float(component.mean_contrast_gray),
        "max_contrast_gray": float(component.max_contrast_gray),
        "integrated_contrast": float(component.integrated_contrast),
        "border_distance_px": float(component.border_distance_px),
        "touches_roi_border": bool(component.touches_roi_border),
        "mask_shape": [int(mask_u8.shape[0]), int(mask_u8.shape[1])],
        "mask_b64": base64.b64encode(mask_u8.tobytes()).decode("ascii"),
    }


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_non_negative_int(value: Any, field: str) -> int:
    if not _is_strict_int(value) or value < 0:
        raise ValueError(f"캐시 정수 필드가 유효하지 않습니다({field}): {value!r}")
    return value


def _require_finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"캐시 실수 필드가 유효하지 않습니다({field}): {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"캐시 실수 필드가 유한하지 않습니다({field}): {value!r}")
    return out


def _decode_component(raw: Any) -> CachedComponent:
    if not isinstance(raw, dict):
        raise ValueError("캐시 컴포넌트 형식이 올바르지 않습니다.")
    bbox_raw = raw.get("bbox")
    shape_raw = raw.get("mask_shape")
    mask_b64 = raw.get("mask_b64")
    if (
        not isinstance(bbox_raw, list)
        or len(bbox_raw) != 4
        or not all(_is_strict_int(v) for v in bbox_raw)
        or not isinstance(shape_raw, list)
        or len(shape_raw) != 2
        or not all(_is_strict_int(v) for v in shape_raw)
        or not isinstance(mask_b64, str)
    ):
        raise ValueError("캐시 컴포넌트 메타데이터가 손상됐습니다.")
    h, w = shape_raw[0], shape_raw[1]
    if (
        h < 0
        or w < 0
        or h > MAX_IMAGE_DIMENSION
        or w > MAX_IMAGE_DIMENSION
        or h * w > MAX_IMAGE_PIXELS
    ):
        raise ValueError(f"캐시 마스크 크기가 허용 범위를 벗어났습니다: {(h, w)}")
    expected_bytes = h * w
    max_b64_len = ((expected_bytes + 2) // 3) * 4
    if len(mask_b64) > max_b64_len:
        raise ValueError("캐시 마스크 base64 길이가 shape 상한을 초과했습니다.")
    try:
        mask_bytes = base64.b64decode(mask_b64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("캐시 마스크 base64 디코딩에 실패했습니다.") from exc
    if len(mask_bytes) != expected_bytes:
        raise ValueError("캐시 마스크 바이트 길이가 shape와 일치하지 않습니다.")
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w)).copy()
    return CachedComponent(
        area_px=_require_non_negative_int(raw["area_px"], "area_px"),
        bbox=(bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]),
        mean_contrast_gray=_require_finite_float(
            raw["mean_contrast_gray"], "mean_contrast_gray"
        ),
        max_contrast_gray=_require_finite_float(
            raw["max_contrast_gray"], "max_contrast_gray"
        ),
        integrated_contrast=_require_finite_float(
            raw["integrated_contrast"], "integrated_contrast"
        ),
        border_distance_px=_require_finite_float(
            raw["border_distance_px"], "border_distance_px"
        ),
        touches_roi_border=bool(raw["touches_roi_border"]),
        mask=mask,
    )


def _encode_sample(sample: CachedSample) -> dict[str, Any]:
    return {
        "sample_id": str(sample.sample_id),
        "original_id": str(sample.original_id),
        "label": str(sample.label),
        "split": str(sample.split),
        "source_split": str(sample.source_split),
        "image_path": str(sample.image_path),
        "mask_path": str(sample.mask_path),
        "defect_pixels": int(sample.defect_pixels),
        "status": sample.status.value if isinstance(sample.status, Status) else str(sample.status),
        "score": float(sample.score),
        "reasons": [str(r) for r in sample.reasons],
        "quality_flags": [str(f) for f in sample.quality_flags],
        "blur_variance": float(sample.blur_variance),
        "mean_gray": float(sample.mean_gray),
        "saturated_ratio": float(sample.saturated_ratio),
        "roi_uncertain": bool(sample.roi_uncertain),
        "truncated": bool(sample.truncated),
        "components": [_encode_component(c) for c in sample.components],
        "gt_components": [_encode_component(c) for c in sample.gt_components],
        "overlay": str(sample.overlay),
        "error_code": None if sample.error_code is None else str(sample.error_code),
        "error_stage": None if sample.error_stage is None else str(sample.error_stage),
    }


def _decode_sample(raw: Any) -> CachedSample:
    if not isinstance(raw, dict):
        raise ValueError("캐시 샘플 형식이 올바르지 않습니다.")
    try:
        comps_raw = raw["components"]
        gt_raw = raw["gt_components"]
        reasons_raw = raw["reasons"]
        flags_raw = raw["quality_flags"]
        if (
            not isinstance(comps_raw, list)
            or not isinstance(gt_raw, list)
            or not isinstance(reasons_raw, list)
            or not isinstance(flags_raw, list)
        ):
            raise ValueError("캐시 목록 필드가 배열이 아닙니다.")
        return CachedSample(
            sample_id=str(raw["sample_id"]),
            original_id=str(raw["original_id"]),
            label=str(raw["label"]),
            split=str(raw["split"]),
            source_split=str(raw["source_split"]),
            image_path=str(raw["image_path"]),
            mask_path=str(raw["mask_path"]),
            defect_pixels=_require_non_negative_int(raw["defect_pixels"], "defect_pixels"),
            status=Status(str(raw["status"])),
            score=_require_finite_float(raw["score"], "score"),
            reasons=tuple(str(r) for r in reasons_raw),
            quality_flags=tuple(str(f) for f in flags_raw),
            blur_variance=_require_finite_float(raw["blur_variance"], "blur_variance"),
            mean_gray=_require_finite_float(raw["mean_gray"], "mean_gray"),
            saturated_ratio=_require_finite_float(raw["saturated_ratio"], "saturated_ratio"),
            roi_uncertain=bool(raw["roi_uncertain"]),
            truncated=bool(raw["truncated"]),
            components=tuple(_decode_component(c) for c in comps_raw),
            gt_components=tuple(_decode_component(c) for c in gt_raw),
            overlay=str(raw["overlay"]),
            error_code=None if raw.get("error_code") is None else str(raw["error_code"]),
            error_stage=None if raw.get("error_stage") is None else str(raw["error_stage"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"캐시 샘플 역직렬화에 실패했습니다: {exc}") from exc


def save_cache(
    samples: Sequence[CachedSample],
    meta: CacheMeta,
    path: str | Path,
) -> Path:
    """안전한 JSON+zlib 포맷으로 캐시를 원자적으로 저장한다(pickle 미사용)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta.as_dict(),
        "samples": [_encode_sample(sample) for sample in samples],
    }
    encoded = _CACHE_MAGIC + zlib.compress(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=6,
    )
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return target


def load_cache(path: str | Path, meta: CacheMeta) -> tuple[CachedSample, ...]:
    """안전한 JSON+zlib 캐시를 크기·구조 검증 후 복원한다(임의 역직렬화·zlib 폭탄 불가)."""

    target = Path(path)
    with target.open("rb") as handle:
        raw = handle.read(MAX_CACHE_COMPRESSED_BYTES + 1)
    if len(raw) > MAX_CACHE_COMPRESSED_BYTES:
        raise ValueError(
            f"캐시 파일 크기 상한({MAX_CACHE_COMPRESSED_BYTES} bytes)을 초과했습니다: {target}"
        )
    if not raw.startswith(_CACHE_MAGIC):
        raise ValueError(f"지원하지 않거나 손상된 캐시 포맷입니다: {target}")
    try:
        decomp = zlib.decompressobj()
        decompressed = decomp.decompress(
            raw[len(_CACHE_MAGIC) :],
            MAX_CACHE_DECOMPRESSED_BYTES + 1,
        )
        if (
            len(decompressed) > MAX_CACHE_DECOMPRESSED_BYTES
            or not decomp.eof
            or decomp.unconsumed_tail
            or decomp.unused_data
        ):
            raise ValueError(
                f"캐시 압축 해제 크기 상한({MAX_CACHE_DECOMPRESSED_BYTES} bytes) 초과 "
                f"또는 스트림 손상: {target}"
            )
        payload = json.loads(decompressed.decode("utf-8"))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"캐시 파일이 손상됐습니다: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"캐시 최상위 구조가 객체가 아닙니다: {target}")
    stored = payload.get("meta")
    if stored != meta.as_dict():
        raise ValueError(f"캐시 정체성이 다릅니다: {target}")
    samples_raw = payload.get("samples")
    if not isinstance(samples_raw, list):
        raise ValueError(f"캐시 samples 필드가 배열이 아닙니다: {target}")
    return tuple(_decode_sample(item) for item in samples_raw)


def cache_summary(samples: Sequence[CachedSample]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[str(sample.status)] = counts.get(str(sample.status), 0) + 1
    return {
        "samples": len(samples),
        "defect": sum(1 for sample in samples if sample.label == "defect"),
        "good": sum(1 for sample in samples if sample.label == "good"),
        "status": counts,
        "components": sum(len(sample.components) for sample in samples),
        "gt_components": sum(len(sample.gt_components) for sample in samples),
        "errors": sum(1 for sample in samples if sample.error_code),
    }


def write_meta(meta: CacheMeta, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(meta.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "CACHE_DIR",
    "CACHE_VERSION",
    "CacheMeta",
    "CachedComponent",
    "CachedSample",
    "DEFAULT_RAW_ROOT",
    "OVERLAY_DIR",
    "build_cache",
    "cache_meta",
    "cache_path",
    "cache_summary",
    "inspect_once",
    "load_cache",
    "save_cache",
    "write_meta",
]
