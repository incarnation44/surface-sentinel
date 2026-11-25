"""최종 채택 설정(r20) 통계 검증 모듈 — 이미지 단위 그룹 CI와 대안 후보(r18) 대비 짝 부트스트랩.

`docs/evidence/t05_adopted_verify.json`을 생성해 채택 근거를 정량화한다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig, load_config
from ..contracts import Status
from .bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_BOOTSTRAP_SEED,
    group_bootstrap_ci,
    paired_bootstrap_delta,
)
from .cache import (
    CACHE_DIR,
    DEFAULT_RAW_ROOT,
    CachedSample,
    cache_meta,
    cache_path,
    load_cache,
)
from .measure import VariantReport, component_iou, evaluate_cached
from .recalibration import SEED
from .runner import cached_samples, load_split_rows, split_sha256
from .sampling import build_sample_plan

DEFAULT_ADOPTED_CONFIG = Path("configs/candidates/adopted/t05_adopted_r20.toml")
DEFAULT_COMPARISON_CONFIG = Path("configs/candidates/fullverify/r18_z7_c16_m5_a45_o3.toml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_GOOD_TARGET = 1787
METRIC_NAMES = ("escape", "false_ng", "review", "recall_iou0.10", "recall_iou0.50")


class AdoptedVerifyError(RuntimeError):
    """채택 검증을 시작할 수 없을 때의 오류."""


@dataclass(frozen=True)
class MetricArrays:
    """이미지 순서대로 정렬된 (분자, 분모) 배열."""

    values: dict[str, tuple[list[int], list[int]]]

    def pair(self, name: str) -> tuple[list[int], list[int]]:
        return self.values[name]


def metric_arrays(
    samples: Sequence[CachedSample],
    report: VariantReport,
) -> MetricArrays:
    """지표별 이미지 단위 분자·분모 배열을 만든다.

    * escape: defect 이미지별 OK 여부
    * false_ng: good 이미지별 NG 여부
    * review: 전체 이미지별 REVIEW 여부
    * recall: defect 이미지별 matched GT / GT 수 (GT 0개는 제외)
    """

    if len(samples) != len(report.outcomes):
        raise AdoptedVerifyError("샘플 수와 결과 수가 다릅니다.")
    values: dict[str, tuple[list[int], list[int]]] = {
        name: ([], []) for name in METRIC_NAMES
    }
    for sample, outcome in zip(samples, report.outcomes, strict=True):
        if sample.error_code:
            continue
        if sample.label == "defect":
            numerator, denominator = values["escape"]
            numerator.append(1 if outcome.status == Status.OK else 0)
            denominator.append(1)
        if sample.label == "good":
            numerator, denominator = values["false_ng"]
            numerator.append(1 if outcome.status == Status.NG else 0)
            denominator.append(1)
        numerator, denominator = values["review"]
        numerator.append(1 if outcome.status == Status.REVIEW else 0)
        denominator.append(1)
        if sample.label != "defect" or not sample.gt_components:
            continue
        matched_10 = 0
        matched_50 = 0
        for gt in sample.gt_components:
            best = 0.0
            for predicted in outcome.predicted:
                best = max(best, component_iou(gt, predicted))
            if best >= 0.10:
                matched_10 += 1
            if best >= 0.50:
                matched_50 += 1
        total = len(sample.gt_components)
        for name, matched in (("recall_iou0.10", matched_10), ("recall_iou0.50", matched_50)):
            numerator, denominator = values[name]
            numerator.append(matched)
            denominator.append(total)
    return MetricArrays(values)


def _load_cached(
    plan: Any,
    config: AppConfig,
    *,
    split_hash: str,
    cache_dir: str | Path,
) -> tuple[CachedSample, ...]:
    meta = cache_meta(plan, config, split_sha256=split_hash)
    path = cache_path(meta, cache_dir)
    if not path.is_file():
        raise AdoptedVerifyError(f"채택 캐시가 없습니다: {path}")
    return load_cache(path, meta)


def build_adopted_verify(
    *,
    adopted_config_path: str | Path = DEFAULT_ADOPTED_CONFIG,
    comparison_config_path: str | Path = DEFAULT_COMPARISON_CONFIG,
    cache_dir: str | Path = CACHE_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    good_target: int = DEFAULT_GOOD_TARGET,
    seed: str = SEED,
    b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """채택 캐시를 읽고 그룹 CI·r18 짝 delta를 계산한다."""

    adopted_config = load_config(adopted_config_path)
    comparison_config = load_config(comparison_config_path)
    plan = build_sample_plan(
        load_split_rows("tune", splits_dir=splits_dir),
        split="tune",
        good_target=good_target,
        seed=seed,
    )
    split_hash = split_sha256(Path(splits_dir) / "tune.csv")
    adopted_samples = _load_cached(
        plan, adopted_config, split_hash=split_hash, cache_dir=cache_dir
    )
    comparison_samples = cached_samples(
        plan,
        comparison_config,
        raw_root=raw_root,
        split_hash=split_hash,
        cache_dir=cache_dir,
        overlays=False,
        label="r18",
    )
    adopted_report = evaluate_cached(adopted_samples, adopted_config, name="adopted")
    comparison_report = evaluate_cached(
        comparison_samples, comparison_config, name="r18"
    )
    adopted_arrays = metric_arrays(adopted_samples, adopted_report)
    comparison_arrays = metric_arrays(comparison_samples, comparison_report)

    group_ci = {
        name: group_bootstrap_ci(*adopted_arrays.pair(name), b=b, seed=bootstrap_seed, alpha=alpha)
        for name in METRIC_NAMES
    }
    paired_delta = {
        name: paired_bootstrap_delta(
            *adopted_arrays.pair(name),
            *comparison_arrays.pair(name),
            b=b,
            seed=bootstrap_seed,
            alpha=alpha,
        )
        for name in METRIC_NAMES
    }
    return {
        "split": "tune",
        "samples": len(plan.rows),
        "adopted": {
            "config_path": str(Path(adopted_config_path)).replace("\\", "/"),
            "config_hash": adopted_config.config_hash,
            "cache": str(
                cache_path(
                    cache_meta(plan, adopted_config, split_sha256=split_hash),
                    cache_dir,
                )
            ).replace("\\", "/"),
        },
        "comparison": {
            "config_path": str(Path(comparison_config_path)).replace("\\", "/"),
            "config_hash": comparison_config.config_hash,
            "cache": str(
                cache_path(
                    cache_meta(plan, comparison_config, split_sha256=split_hash),
                    cache_dir,
                )
            ).replace("\\", "/"),
        },
        "bootstrap": {"b": b, "seed": bootstrap_seed, "alpha": alpha},
        "group_ci": group_ci,
        "paired_delta": paired_delta,
    }


def write_adopted_verify(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "AdoptedVerifyError",
    "DEFAULT_ADOPTED_CONFIG",
    "DEFAULT_COMPARISON_CONFIG",
    "DEFAULT_GOOD_TARGET",
    "build_adopted_verify",
    "metric_arrays",
    "write_adopted_verify",
]
