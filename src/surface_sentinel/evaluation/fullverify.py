"""전수 tune 검증 상세 분석 (2026-11-11).

전수 측정 산출물(``docs/evidence/tune_fullverify.jsonl``)과 후보별 캐시를
**읽기 전용**으로 재사용한다. 파이프라인 재실행은 escape 샘플의 원인을
분류할 때 escape 장수만큼만 수행한다.

산출물은 ``docs/evidence/tune_fullverify_details.json`` 하나다.
봉인 분할(``test``/``official_test``)은 이름 단계에서 거부한다.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig, load_config
from ..contracts import Status
from ..pipeline.stages import (
    generate_candidates,
    load_image,
    prepare_roi,
    preprocess,
    refine_mask,
    validate_input,
)
from .cache import (
    CACHE_DIR,
    DEFAULT_RAW_ROOT,
    CachedSample,
    cache_meta,
    cache_path,
    load_cache,
)
from .measure import component_iou, evaluate_cached
from .metrics import (
    IOU_MATCH_FULL,
    IOU_MATCH_RECALL,
    IOU_MATCH_STRICT,
    Confusion,
    GtMatch,
    iou_distribution,
    recall_at,
)
from .recalibration import SEED, load_candidate_configs
from .runner import load_split_rows, split_sha256
from .sampling import build_sample_plan

DEFAULT_JSONL = Path("docs/evidence/tune_fullverify.jsonl")
DEFAULT_CANDIDATE_DIR = Path("configs/candidates/fullverify")
DEFAULT_OUT = Path("docs/evidence/tune_fullverify_details.json")
DEFAULT_BASELINE_JSONL = Path("docs/evidence/tune_grid2.jsonl")
DEFAULT_SPLITS_DIR = Path("data/splits")
FULLVERIFY_GOOD_TARGET = 1787
SEALED_SPLITS = frozenset({"test", "official_test"})


class FullVerifyError(RuntimeError):
    """상세 분석을 시작할 수 없을 때의 오류."""


def _count_nonzero(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def classify_escape(
    sample: CachedSample,
    config: AppConfig,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
) -> dict[str, Any]:
    """escape 샘플 하나의 원인을 분류한다.

    우선순위: ``roi_empty`` → ``no_candidate_pixels`` → ``morphology_erased``
    → ``all_below_min_area`` → ``all_below_min_contrast`` → ``other``.
    """

    root = Path(raw_root)
    image_path = root / sample.image_path
    mask_path = root / sample.mask_path if sample.mask_path else None
    metadata, decoded = validate_input(
        image_path,
        mask_path=mask_path,
        label=sample.label,
        split=sample.split,
        config=config,
    )
    loaded = load_image(decoded, metadata)
    roi = prepare_roi(loaded.gray, config)
    pre = preprocess(loaded.gray, roi, config)
    candidate = generate_candidates(pre.z_map, pre.response, roi.mask, config)
    refined = refine_mask(
        candidate.mask, pre.z_map, pre.response, roi.mask, config
    )

    z_pass = pre.z_map >= config.candidates.z_threshold
    contrast_pass = pre.response >= config.candidates.min_contrast_gray
    roi_pass = roi.mask > 0
    both = z_pass & contrast_pass & roi_pass
    z_only = _count_nonzero(z_pass & ~contrast_pass & roi_pass)
    contrast_only = _count_nonzero(~z_pass & contrast_pass & roi_pass)

    cc_count, _, cc_stats, _ = cv2.connectedComponentsWithStats(
        candidate.mask, connectivity=8
    )
    cc_areas = [
        int(cc_stats[index, cv2.CC_STAT_AREA])
        for index in range(1, int(cc_count))
    ]
    refined_means = [
        float(component.mean_contrast_gray) for component in refined.components
    ]
    diagnostics = {
        "roi_area_px": int(roi.area_px),
        "roi_uncertain": bool(roi.uncertain),
        "z_only_pixels": z_only,
        "contrast_only_pixels": contrast_only,
        "both_pixels": _count_nonzero(both),
        "candidate_pixels": _count_nonzero(candidate.mask),
        "refined_pixels": _count_nonzero(refined.combined_mask),
        "cc_count": max(int(cc_count) - 1, 0),
        "max_cc_area": max(cc_areas, default=0),
        "refined_components": len(refined.components),
        "max_refined_mean_contrast": max(refined_means, default=0.0),
        "min_area_px": int(config.candidates.min_area_px),
        "min_contrast_gray": float(config.candidates.min_contrast_gray),
    }
    cause = _classify_cause(diagnostics)
    return {
        "sample_id": sample.sample_id,
        "label": sample.label,
        "image_path": sample.image_path,
        "cause": cause,
        "diagnostics": diagnostics,
        "truncated": bool(sample.truncated),
        "quality_flags": list(sample.quality_flags),
    }


def _classify_cause(diagnostics: dict[str, Any]) -> str:
    """진단 수치만으로 원인을 고른다(테스트 가능한 순수 함수)."""

    if int(diagnostics["roi_area_px"]) <= 0:
        return "roi_empty"
    if int(diagnostics["both_pixels"]) <= 0:
        return "no_candidate_pixels"
    if int(diagnostics["candidate_pixels"]) <= 0:
        return "morphology_erased"
    if int(diagnostics["cc_count"]) <= 0:
        return "other"
    if int(diagnostics["max_cc_area"]) < int(diagnostics["min_area_px"]):
        return "all_below_min_area"
    if (
        int(diagnostics["refined_components"]) <= 0
        and int(diagnostics.get("refined_pixels", 0)) <= 0
    ):
        return "morphology_erased"
    if int(diagnostics["refined_components"]) > 0 and (
        float(diagnostics["max_refined_mean_contrast"])
        < float(diagnostics["min_contrast_gray"])
    ):
        return "all_below_min_contrast"
    return "other"


def review_breakdown(samples: Sequence[CachedSample]) -> dict[str, Any]:
    """REVIEW 샘플의 ReasonCode 분포를 label별로 센다."""

    reasons = {
        "defect": Counter(),
        "good": Counter(),
        "total": Counter(),
    }
    counts = {"defect": 0, "good": 0, "total": 0}
    for sample in samples:
        if sample.status != Status.REVIEW:
            continue
        label = sample.label if sample.label in ("defect", "good") else "other"
        if label == "other":
            continue
        counts[label] += 1
        counts["total"] += 1
        for reason in sample.reasons:
            reasons[label][reason] += 1
            reasons["total"][reason] += 1
    return {
        "samples": counts,
        "reasons": {
            label: dict(sorted(bucket.items())) for label, bucket in reasons.items()
        },
    }


def iou_details(
    samples: Sequence[CachedSample],
    config: AppConfig,
) -> dict[str, Any]:
    """GT별 best IoU와 recall@0.10/0.30/0.50, IoU 분포를 낸다."""

    report = evaluate_cached(samples, config)
    outcomes = {outcome.sample_id: outcome for outcome in report.outcomes}
    matches: list[GtMatch] = []
    for sample in samples:
        if sample.label != "defect":
            continue
        outcome = outcomes.get(sample.sample_id)
        if outcome is None:
            continue
        for gt in sample.gt_components:
            best = 0.0
            for predicted in outcome.predicted:
                best = max(best, component_iou(gt, predicted))
            matches.append(GtMatch(iou=best, image_status=outcome.status.value))
    return {
        "gt_components": len(matches),
        "recall_at": {
            f"{threshold:.2f}": recall_at(matches, threshold).as_dict()
            for threshold in (IOU_MATCH_RECALL, IOU_MATCH_STRICT, IOU_MATCH_FULL)
        },
        "distribution": iou_distribution(matches),
    }


def _load_cached_samples(
    plan: Any,
    config: AppConfig,
    *,
    split_hash: str,
    cache_dir: str | Path,
) -> tuple[CachedSample, ...]:
    meta = cache_meta(plan, config, split_sha256=split_hash)
    path = cache_path(meta, cache_dir)
    if not path.is_file():
        raise FullVerifyError(f"전수 측정 캐시가 없습니다: {path}")
    return load_cache(path, meta)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FullVerifyError(f"JSONL이 없습니다: {target}")
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metric_value(row: dict[str, Any], name: str) -> float | None:
    metric = row.get("metrics", {}).get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    return None if value is None else float(value)


def _delta(full: float | None, baseline: float | None) -> float | None:
    if full is None or baseline is None:
        return None
    return full - baseline


def build_details(
    *,
    jsonl_path: str | Path = DEFAULT_JSONL,
    candidate_dir: str | Path = DEFAULT_CANDIDATE_DIR,
    cache_dir: str | Path = CACHE_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    good_target: int = FULLVERIFY_GOOD_TARGET,
    seed: str = SEED,
    baseline_jsonl: str | Path = DEFAULT_BASELINE_JSONL,
) -> dict[str, Any]:
    """후보별 상세 분석 묶음을 만든다. 캐시만 읽고, escape만 재실행한다."""

    rows = _read_jsonl(jsonl_path)
    if not rows:
        raise FullVerifyError(f"JSONL에 후보가 없습니다: {jsonl_path}")
    configs = dict(load_candidate_configs(candidate_dir))
    split_rows = load_split_rows("tune", splits_dir=splits_dir)
    plan = build_sample_plan(
        split_rows, split="tune", good_target=good_target, seed=seed
    )
    split_hash = split_sha256(Path(splits_dir) / "tune.csv")
    baseline = {
        str(row.get("candidate_id")): row for row in _read_jsonl(baseline_jsonl)
    }

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in configs:
            raise FullVerifyError(
                f"후보 TOML이 없습니다: {candidate_id} ({candidate_dir})"
            )
        config = load_config(configs[candidate_id])
        samples = _load_cached_samples(
            plan, config, split_hash=split_hash, cache_dir=cache_dir
        )
        report = evaluate_cached(samples, config, name=candidate_id)
        pairs = [
            (outcome.label, outcome.status.value) for outcome in report.outcomes
        ]
        confusion = Confusion.from_pairs(pairs)
        iou = iou_details(samples, config)
        escapes = [
            classify_escape(sample, config, raw_root=raw_root)
            for sample in samples
            if sample.label == "defect" and sample.status == Status.OK
        ]
        cause_counts = Counter(item["cause"] for item in escapes)
        base_row = baseline.get(candidate_id, {})
        full_values = {
            "escape": _metric_value(row, "escape"),
            "false_ng": _metric_value(row, "false_ng"),
            "review": _metric_value(row, "review"),
            "recall_iou0.10": _metric_value(row, "localization_recall_iou0.10"),
        }
        baseline_keys = {
            "escape": "escape",
            "false_ng": "false_ng",
            "review": "review",
            "recall_iou0.10": "localization_recall_iou0.10",
        }
        delta_714 = {
            name: _delta(
                full_values[name],
                _metric_value(base_row, baseline_keys[name]),
            )
            for name in full_values
        }
        candidates.append(
            {
                "candidate_id": candidate_id,
                "config_hash": config.config_hash,
                "config_path": str(configs[candidate_id]).replace("\\", "/"),
                "samples": len(samples),
                "counts": {
                    "defect": sum(1 for sample in samples if sample.label == "defect"),
                    "good": sum(1 for sample in samples if sample.label == "good"),
                    "gt_components": iou["gt_components"],
                    "errors": sum(1 for sample in samples if sample.error_code),
                    "truncated": sum(1 for sample in samples if sample.truncated),
                },
                "confusion": confusion.as_dict(),
                "metrics": {
                    **{
                        name: report.ratio(metric).as_dict()
                        for name, metric in (
                            ("escape", "defect_escape_rate"),
                            ("false_ng", "good_false_ng_rate"),
                            ("review", "review_rate"),
                        )
                    },
                    "recall_iou0.10": report.ratio(
                        "localization_recall_iou0.10"
                    ).as_dict(),
                    "recall_iou0.30": iou["recall_at"]["0.30"],
                    "recall_iou0.50": iou["recall_at"]["0.50"],
                },
                "delta_714": delta_714,
                "baseline_714": {
                    "samples": base_row.get("samples"),
                    "metrics": {
                        name: _metric_value(base_row, name)
                        for name in (
                            "escape",
                            "false_ng",
                            "review",
                            "localization_recall_iou0.10",
                        )
                    },
                },
                "review_breakdown": review_breakdown(samples),
                "iou": iou,
                "escapes": escapes,
                "escape_causes": dict(sorted(cause_counts.items())),
            }
        )

    return {
        "split": "tune",
        "samples": len(plan.rows),
        "candidates": candidates,
        "totals": {
            "candidates": len(candidates),
            "samples": len(plan.rows),
            "defect": plan.defect_taken,
            "good": plan.good_taken,
            "gt_components": candidates[0]["iou"]["gt_components"] if candidates else 0,
            "errors": sum(item["counts"]["errors"] for item in candidates),
        },
        "inputs": {
            "jsonl": str(Path(jsonl_path)).replace("\\", "/"),
            "candidate_dir": str(Path(candidate_dir)).replace("\\", "/"),
            "cache_dir": str(Path(cache_dir)).replace("\\", "/"),
            "baseline_jsonl": str(Path(baseline_jsonl)).replace("\\", "/"),
            "good_target": good_target,
            "seed": seed,
        },
    }


def write_details(details: dict[str, Any], path: str | Path = DEFAULT_OUT) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "DEFAULT_CANDIDATE_DIR",
    "DEFAULT_JSONL",
    "DEFAULT_OUT",
    "FULLVERIFY_GOOD_TARGET",
    "FullVerifyError",
    "build_details",
    "classify_escape",
    "iou_details",
    "review_breakdown",
    "write_details",
]
