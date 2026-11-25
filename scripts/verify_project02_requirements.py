"""PROJECT 02 과제 요건 실측 검증 스크립트.

봉인 분할(``test``, ``official_test``)에 절대 접근하지 않고,
``tune`` 분할 중 ``source_split == "train"`` (``data/raw/kolektorsdd2/train``)
가용 이미지 1,422장(defect 146 + good 1,276)을 대상으로 다음을 측정한다:

1. 전처리 3종 비교 (``configs/base.toml`` r20 동일 후단 조건):
   - CLAHE 대비 강조 (``cv2.createCLAHE``)
   - Sobel 에지 그라디언트 (``cv2.Sobel``)
   - 정본 Dual Morphology (``Black-hat + Top-hat + Median Blur``)
   -> ``docs/evidence/project02_preprocess_comparison.json`` 저장

2. 정본 파이프라인 검출 ``NG`` 및 ``false-NG`` 사례의 ``cv2.minAreaRect`` 기하 계측
   (``area_px``, ``length_px``, ``width_px``, ``angle_deg``, ``mean_contrast_gray``,
   ``score``, ``status``, ``label``, ``failure_reason``)
   -> ``docs/evidence/failure_cases/project02_ng_and_false_ng_cases.csv`` 저장

실행(PowerShell): ``uv run python scripts/verify_project02_requirements.py``
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from surface_sentinel.config import AppConfig, load_config
from surface_sentinel.contracts import QualityInfo, ReasonCode, Status
from surface_sentinel.evaluation.cache import CachedComponent, _gt_components
from surface_sentinel.evaluation.measure import component_iou
from surface_sentinel.evaluation.metrics import (
    Confusion,
    GtMatch,
    PixelCounts,
    image_metrics,
    localization_metrics,
    recall_at,
)
from surface_sentinel.evaluation.runner import load_split_rows
from surface_sentinel.pipeline.stages import (
    _background_kernel,
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
from surface_sentinel.pipeline.types import PreprocessResult, RoiResult

CONFIG_PATH = Path("configs/base.toml")
RAW_ROOT = Path("data/raw/kolektorsdd2")
OUT_JSON = Path("docs/evidence/project02_preprocess_comparison.json")
OUT_CSV = Path("docs/evidence/failure_cases/project02_ng_and_false_ng_cases.csv")

METHODS: tuple[tuple[str, str], ...] = (
    ("clahe", "CLAHE 대비 강조 (cv2.createCLAHE)"),
    ("sobel", "Sobel 에지 그라디언트 (cv2.Sobel)"),
    ("dual_morphology", "정본 Dual Morphology (Black-hat + Top-hat + Median Blur)"),
)


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    label: str
    source_split: str
    gray: np.ndarray
    roi: RoiResult
    quality: QualityInfo
    bg_kernel: int
    gt_components: tuple[CachedComponent, ...]
    gt_area_px: int


def _compute_quality(gray: np.ndarray, roi_mask: np.ndarray, config: AppConfig) -> QualityInfo:
    roi_bool = roi_mask > 0
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
    return QualityInfo(
        blur_variance=blur_variance,
        mean_gray=mean_gray,
        saturated_ratio=saturated_ratio,
        flags=flags,
    )


def _normalize_z(
    response: np.ndarray,
    roi_mask: np.ndarray,
    noise_floor_gray: float,
) -> np.ndarray:
    height, width = response.shape[:2]
    roi_bool = roi_mask > 0
    resp = np.ascontiguousarray(response, dtype=np.float32)
    resp[~roi_bool] = 0.0
    z_map = np.zeros((height, width), dtype=np.float32)
    values = resp[roi_bool]
    if values.size > 0:
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        sigma = 1.4826 * mad
        denom = max(sigma, float(noise_floor_gray))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_map = (resp / denom).astype(np.float32)
        z_map[~roi_bool] = 0.0
    return z_map


def _run_preprocess_variant(
    method: str,
    sample: PreparedSample,
    config: AppConfig,
) -> PreprocessResult:
    if method == "dual_morphology":
        return preprocess(sample.gray, sample.roi, config)

    gray = sample.gray
    roi_bool = sample.roi.mask > 0
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        vals = enhanced[roi_bool]
        med_bg = float(np.median(vals)) if vals.size > 0 else 0.0
        response = np.abs(enhanced.astype(np.float32) - med_bg)
    elif method == "sobel":
        gray_blur = cv2.medianBlur(gray, int(config.preprocess.median_kernel))
        gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
        response = cv2.magnitude(gx, gy)
    else:
        raise ValueError(f"알 수 없는 전처리 방식: {method!r}")

    response = np.ascontiguousarray(response, dtype=np.float32)
    response[~roi_bool] = 0.0
    z_map = _normalize_z(response, sample.roi.mask, config.preprocess.noise_floor_gray)
    return PreprocessResult(
        response=response,
        z_map=z_map,
        quality=sample.quality,
        bg_kernel=sample.bg_kernel,
    )


def _rect_geometry(mask: np.ndarray) -> tuple[float, float, float]:
    """cv2.minAreaRect 기반 (length_px, width_px, angle_deg)를 반환한다."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect
    long_side = max(float(rw), float(rh))
    short_side = min(float(rw), float(rh))

    box = cv2.boxPoints(rect)
    edge0 = box[1] - box[0]
    edge1 = box[2] - box[1]
    len0 = float(np.hypot(edge0[0], edge0[1]))
    len1 = float(np.hypot(edge1[0], edge1[1]))
    major = edge0 if len0 >= len1 else edge1
    angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    return long_side, short_side, angle


def _classify_failure_reason(
    *,
    label: str,
    elongation: float,
    component_count: int,
    edge_margin_px: int,
    touches_roi_border: bool,
    gray_mean_comp: float,
    gray_max_comp: float,
) -> str:
    if label == "defect":
        if elongation >= 3.0:
            return "정상 검출 (선형 스크래치 결함)"
        return "정상 검출 (국소 표면 결함)"

    if gray_max_comp >= 200.0 or gray_mean_comp >= 150.0:
        return "조명 반사 (고휘도 반사·포화 영역)"
    if touches_roi_border or edge_margin_px <= 12 or (gray_mean_comp <= 25.0 and elongation >= 5.0):
        return "경계 인접 (치구·가장자리 경계선)"
    if component_count >= 3:
        return "미세 표면 결 다발 (반복 가공 무늬·다중 후보)"
    return "국소 표면 음영/오염"


def _load_unsealed_tune_train(config: AppConfig) -> list[PreparedSample]:
    # 봉인 분할(test, official_test)을 열지 않도록 tune split 중 source_split=="train"만 사용한다.
    rows = [
        row
        for row in load_split_rows("tune", allow_sealed=False)
        if row.source_split == "train"
    ]
    samples: list[PreparedSample] = []
    for row in rows:
        image_path = RAW_ROOT / row.image_path
        mask_path = RAW_ROOT / row.mask_path
        meta, decoded = validate_input(
            image_path,
            mask_path=mask_path,
            label=row.label,
            split="tune",
            config=config,
            allow_sealed=False,
        )
        loaded = load_image(decoded, meta)
        roi = prepare_roi(loaded.gray, config)
        quality = _compute_quality(loaded.gray, roi.mask, config)
        bg_kernel = _background_kernel(loaded.gray.shape[0], loaded.gray.shape[1], config)
        gt_components = _gt_components(str(mask_path)) if row.label == "defect" else ()
        gt_area_px = sum(c.area_px for c in gt_components)
        samples.append(
            PreparedSample(
                sample_id=row.sample_id,
                label=row.label,
                source_split=row.source_split,
                gray=loaded.gray,
                roi=roi,
                quality=quality,
                bg_kernel=bg_kernel,
                gt_components=gt_components,
                gt_area_px=gt_area_px,
            )
        )
    return samples


def _to_cached_component(
    bbox: tuple[int, int, int, int],
    full_mask: np.ndarray,
    area_px: int,
    mean_contrast_gray: float,
    max_contrast_gray: float,
    integrated_contrast: float,
    border_distance_px: float,
    touches_roi_border: bool,
) -> CachedComponent:
    x, y, w, h = bbox
    patch = np.ascontiguousarray(full_mask[y : y + h, x : x + w])
    return CachedComponent(
        area_px=area_px,
        bbox=bbox,
        mean_contrast_gray=mean_contrast_gray,
        max_contrast_gray=max_contrast_gray,
        integrated_contrast=integrated_contrast,
        border_distance_px=border_distance_px,
        touches_roi_border=touches_roi_border,
        mask=patch,
    )


def main() -> int:
    config = load_config(CONFIG_PATH)
    samples = _load_unsealed_tune_train(config)
    if not samples:
        raise SystemExit("가용 unsealed tune-train 이미지가 없습니다.")

    # 첫 호출 JIT/OpenCV 내부 캐시 편향을 없애기 위해 3개 방식 1장 워밍업
    for method_id, _ in METHODS:
        _run_preprocess_variant(method_id, samples[0], config)

    method_results: dict[str, Any] = {}
    ng_case_rows: list[dict[str, Any]] = []

    for method_id, method_label in METHODS:
        pre_times: list[float] = []
        tot_times: list[float] = []
        pairs: list[tuple[str, str]] = []
        matches: list[GtMatch] = []
        predicted_components_total = 0
        gt_components_total = 0

        for sample in samples:
            t0 = time.perf_counter()
            pre = _run_preprocess_variant(method_id, sample, config)
            t_pre = time.perf_counter()
            cand = generate_candidates(pre.z_map, pre.response, sample.roi.mask, config)
            ref = refine_mask(cand.mask, pre.z_map, pre.response, sample.roi.mask, config)
            ana = analyze_components(ref, sample.roi, pre.response, config)
            sc = calculate_score(ana.components, config)
            dec = decide_status(sc, pre.quality, sample.roi, ref, config)
            t1 = time.perf_counter()

            pre_times.append((t_pre - t0) * 1000.0)
            tot_times.append((t1 - t0) * 1000.0)
            pairs.append((sample.label, dec.status.value))
            predicted_components_total += len(sc.valid_components)
            gt_components_total += len(sample.gt_components)

            if sample.gt_components:
                valid_indices = {f.index for f in sc.valid_components}
                pred_cached = [
                    _to_cached_component(
                        c.bbox,
                        c.mask,
                        c.area_px,
                        c.mean_contrast_gray,
                        c.max_contrast_gray,
                        c.integrated_contrast,
                        c.border_distance_px,
                        c.touches_roi_border,
                    )
                    for c in ref.components
                    if c.index in valid_indices
                ]
                for gt in sample.gt_components:
                    best = max(
                        (component_iou(gt, p) for p in pred_cached),
                        default=0.0,
                    )
                    matches.append(GtMatch(iou=best, image_status=dec.status.value))

            if method_id == "dual_morphology" and dec.status is Status.NG and ref.components:
                top_comp = ref.components[0]
                top_feat = ana.components[0]
                length_px, width_px, angle_deg = _rect_geometry(top_comp.mask)
                comp_pixels = sample.gray[top_comp.mask > 0]
                gray_mean_comp = float(np.mean(comp_pixels)) if comp_pixels.size else 0.0
                gray_max_comp = float(np.max(comp_pixels)) if comp_pixels.size else 0.0
                ih, iw = sample.gray.shape[:2]
                bx, by, bw, bh = top_comp.bbox
                edge_margin_px = min(bx, by, max(iw - (bx + bw), 0), max(ih - (by + bh), 0))
                reason = _classify_failure_reason(
                    label=sample.label,
                    elongation=top_feat.elongation,
                    component_count=len(ana.components),
                    edge_margin_px=edge_margin_px,
                    touches_roi_border=top_comp.touches_roi_border,
                    gray_mean_comp=gray_mean_comp,
                    gray_max_comp=gray_max_comp,
                )
                ng_case_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "label": sample.label,
                        "status": dec.status.value,
                        "case_type": "true_ng" if sample.label == "defect" else "false_ng",
                        "score": round(float(sc.score), 6),
                        "area_px": int(top_comp.area_px),
                        "length_px": round(length_px, 2),
                        "width_px": round(width_px, 2),
                        "angle_deg": round(angle_deg, 2),
                        "elongation": round(float(top_feat.elongation), 3),
                        "mean_contrast_gray": round(float(top_comp.mean_contrast_gray), 3),
                        "component_count": len(ana.components),
                        "edge_margin_px": int(edge_margin_px),
                        "gray_mean_comp": round(gray_mean_comp, 2),
                        "gray_max_comp": round(gray_max_comp, 2),
                        "failure_reason": reason,
                    }
                )

        confusion = Confusion.from_pairs(pairs)
        img_metrics = image_metrics(confusion)
        loc = localization_metrics(matches, PixelCounts())
        r050 = recall_at(matches, 0.50)
        p_def = confusion.evaluated("defect")
        n_good = confusion.evaluated("good")
        total = p_def + n_good

        method_results[method_id] = {
            "label": method_label,
            "counts": {
                "total": total,
                "defect": p_def,
                "good": n_good,
                "gt_components": gt_components_total,
                "predicted_components": predicted_components_total,
                "confusion": confusion.as_dict(),
            },
            "metrics": {
                "escape": img_metrics["defect_escape_rate"].as_dict(),
                "false_ng": img_metrics["good_false_ng_rate"].as_dict(),
                "review": img_metrics["review_rate"].as_dict(),
                "recall_iou0.10": loc["localization_recall_iou0.10"].as_dict(),
                "recall_iou0.30": loc["localization_recall_iou0.30"].as_dict(),
                "recall_iou0.50": r050.as_dict(),
            },
            "timing_ms": {
                "preprocess_p50_ms": round(float(np.percentile(pre_times, 50)), 3),
                "preprocess_p95_ms": round(float(np.percentile(pre_times, 95)), 3),
                "pipeline_p50_ms": round(float(np.percentile(tot_times, 50)), 3),
                "pipeline_p95_ms": round(float(np.percentile(tot_times, 95)), 3),
            },
        }

    ng_case_rows.sort(key=lambda r: (r["case_type"], -r["score"], r["sample_id"]))
    false_ng_reasons: dict[str, int] = {}
    for r in ng_case_rows:
        if r["case_type"] == "false_ng":
            false_ng_reasons[r["failure_reason"]] = false_ng_reasons.get(r["failure_reason"], 0) + 1

    payload = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": CONFIG_PATH.as_posix(),
        "config_hash": config.config_hash,
        "dataset_scope": {
            "split": "tune",
            "source_split": "train",
            "allow_sealed": False,
            "note": (
                "봉인 분할(test 1,334장, official_test 1,004장)과 겹치지 않는 "
                "data/raw/kolektorsdd2/train 내 unsealed tune 표본 전수"
            ),
            "total": len(samples),
            "defect": sum(1 for s in samples if s.label == "defect"),
            "good": sum(1 for s in samples if s.label == "good"),
        },
        "methods": method_results,
        "ng_case_summary": {
            "total_ng": len(ng_case_rows),
            "true_ng": sum(1 for r in ng_case_rows if r["case_type"] == "true_ng"),
            "false_ng": sum(1 for r in ng_case_rows if r["case_type"] == "false_ng"),
            "false_ng_reason_counts": false_ng_reasons,
            "csv_path": OUT_CSV.as_posix(),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "status",
        "case_type",
        "score",
        "area_px",
        "length_px",
        "width_px",
        "angle_deg",
        "elongation",
        "mean_contrast_gray",
        "component_count",
        "edge_margin_px",
        "gray_mean_comp",
        "gray_max_comp",
        "failure_reason",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ng_case_rows)

    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved CSV ({len(ng_case_rows)} rows): {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
