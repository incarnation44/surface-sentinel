"""샘플별 1회 캐시: 10단계 파이프라인을 **한 번만** 돌리고 재계산 재료를 남긴다.

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

import hashlib
import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig
from ..contracts import STAGE_ORDER, SampleResult, Status
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

CACHE_VERSION = "t05-cache-v1"
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
    name = f"{meta.version}-{meta.split}-{meta.config_hash[:12]}-{meta.sample_ids_sha256[:12]}.pkl"
    return Path(directory) / name


def _read_mask(path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 3:
        decoded = np.any(decoded != 0, axis=2).astype(np.uint8) * 255
    return (decoded > 0).astype(np.uint8) * 255


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
        gt_components=_gt_components(row.mask_path)
        if row.mask_path
        else (),
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
) -> CachedSample:
    """한 샘플을 10단계로 한 번 검사하고 재계산 재료를 남긴다."""

    root = Path(raw_root)
    image_path = root / row.image_path
    mask_path = root / row.mask_path if row.mask_path else None
    try:
        metadata, decoded = validate_input(
            image_path,
            mask_path=mask_path,
            label=row.label,
            split=split,
            config=config,
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
        return _error_sample(row, split=split, error=exc)

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
            )
        )
        if progress is not None:
            progress(index, total, row.sample_id)
    return tuple(samples)


def save_cache(
    samples: Sequence[CachedSample],
    meta: CacheMeta,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta.as_dict(), "samples": list(samples)}
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    temporary.replace(target)
    return target


def load_cache(path: str | Path, meta: CacheMeta) -> tuple[CachedSample, ...]:
    target = Path(path)
    with target.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - 로컬에서 만든 캐시만 읽는다
    stored = payload.get("meta")
    if stored != meta.as_dict():
        raise ValueError(f"캐시 정체성이 다릅니다: {target}")
    return tuple(payload["samples"])


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
