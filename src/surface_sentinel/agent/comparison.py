"""고정 base vs 에이전트 정책 비교 (T09, 설계 §6 승격 조건).

120장 층화 tune 부분집합에서만 비교한다. 봉인 분할은 절대 열지 않는다.
모든 비율은 명시적 분모와 함께 보고한다. 승격은 다섯 조건(리뷰율 5pp 이상
감소, escape 1pp 이하 증가, 추가 p95 지연 1.5초 이하, 잘못된 행동 0, 감사
완전)을 모두 충족할 때만이며, 그렇지 않으면 고정 모드를 유지하고 보고한다.

실행: ``python -m surface_sentinel.agent.comparison --code-sha <sha>``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..pipeline import inspect_image
from .config import AgentConfig, ComparisonConfig, load_agent_config
from .run import load_profiles, run_agent_episode
from .trace import audit_trace

COMPARISON_SCHEMA_VERSION = "agent-comparison-v1"


def stratified_subset(
    rows: Sequence[Mapping[str, str]],
    size: int,
    seed: int,
) -> list[Mapping[str, str]]:
    """label 비례 층화 추출(최대 나머지 할당). 정렬+시드로 결정론적이다."""

    if size <= 0:
        raise ValueError("size는 양수여야 합니다")
    by_label: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    total = len(rows)
    if total == 0:
        raise ValueError("입력 행이 비어 있습니다")
    if size >= total:
        return sorted(rows, key=lambda row: row["sample_id"])

    labels = sorted(by_label)
    quotas = {label: size * len(by_label[label]) / total for label in labels}
    allocation = {label: int(quotas[label]) for label in labels}
    remainder = size - sum(allocation.values())
    order = sorted(labels, key=lambda label: (-(quotas[label] - allocation[label]), label))
    for label in order[:remainder]:
        allocation[label] += 1

    rng = random.Random(seed)
    picked: list[Mapping[str, str]] = []
    for label in labels:
        pool = sorted(by_label[label], key=lambda row: row["sample_id"])
        picked.extend(rng.sample(pool, allocation[label]))
    return sorted(picked, key=lambda row: row["sample_id"])


def _rate(status: str, pool: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    count = sum(1 for sample in pool if sample[key] == status)
    denominator = len(pool)
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _p95(values: Sequence[float]) -> float | None:
    """nearest-rank p95. 빈 목록은 None."""

    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return float(ordered[rank - 1])


def _metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    defect = [sample for sample in samples if sample["label"] == "defect"]
    good = [sample for sample in samples if sample["label"] == "good"]
    return {
        "samples": len(samples),
        "review_fixed": _rate("REVIEW", samples, "fixed_status"),
        "review_agent": _rate("REVIEW", samples, "agent_status"),
        "escape_fixed": _rate("OK", defect, "fixed_status"),
        "escape_agent": _rate("OK", defect, "agent_status"),
        "false_ng_fixed": _rate("NG", good, "fixed_status"),
        "false_ng_agent": _rate("NG", good, "agent_status"),
        "extra_latency_p95_ms": _p95([float(s["agent_extra_ms"]) for s in samples]),
        "extra_latency_denominator": len(samples),
        "wrong_actions": sum(int(s["rejections"]) for s in samples),
        "audit_complete": all(not s["audit_errors"] for s in samples),
        "audit_error_samples": sum(1 for s in samples if s["audit_errors"]),
    }


def _promotion(
    metrics: Mapping[str, Any],
    criteria: ComparisonConfig,
) -> dict[str, Any]:
    review_fixed = metrics["review_fixed"]["rate"]
    review_agent = metrics["review_agent"]["rate"]
    escape_fixed = metrics["escape_fixed"]["rate"]
    escape_agent = metrics["escape_agent"]["rate"]

    review_drop_pp = (
        (review_fixed - review_agent) * 100.0
        if review_fixed is not None and review_agent is not None
        else None
    )
    escape_increase_pp = (
        (escape_agent - escape_fixed) * 100.0
        if escape_fixed is not None and escape_agent is not None
        else None
    )
    p95 = metrics["extra_latency_p95_ms"]

    checks = {
        "review_drop_pp": {
            "value": review_drop_pp,
            "threshold_min": criteria.review_drop_pp_min,
            "pass": review_drop_pp is not None
            and review_drop_pp >= criteria.review_drop_pp_min,
        },
        "escape_increase_pp": {
            "value": escape_increase_pp,
            "threshold_max": criteria.escape_increase_pp_max,
            "pass": escape_increase_pp is not None
            and escape_increase_pp <= criteria.escape_increase_pp_max,
        },
        "extra_p95_ms": {
            "value": p95,
            "threshold_max": criteria.extra_p95_ms_max,
            "pass": p95 is not None and p95 <= criteria.extra_p95_ms_max,
        },
        "wrong_actions": {
            "value": metrics["wrong_actions"],
            "pass": metrics["wrong_actions"] == 0,
        },
        "audit_complete": {
            "value": metrics["audit_complete"],
            "pass": bool(metrics["audit_complete"]),
        },
    }
    promoted = all(check["pass"] for check in checks.values())
    return {
        "promoted": promoted,
        "mode": "agent" if promoted else "fixed",
        "criteria": checks,
    }


def run_comparison(
    *,
    tune_csv: str | Path,
    raw_root: str | Path,
    agent_config: AgentConfig,
    profiles: Mapping[str, Any],
    output_dir: str | Path,
    subset_size: int | None = None,
    seed: int | None = None,
    code_sha: str = "",
) -> dict[str, Any]:
    """층화 tune 부분집합에서 고정 모드와 에이전트 정책을 비교한다."""

    with Path(tune_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    size = agent_config.comparison.subset_size if subset_size is None else subset_size
    chosen_seed = agent_config.comparison.seed if seed is None else seed
    subset = stratified_subset(rows, size, chosen_seed)
    base_config = profiles["base"]
    root = Path(raw_root)
    out = Path(output_dir)

    samples: list[dict[str, Any]] = []
    for row in subset:
        image = root / row["image_path"]
        mask = root / row["mask_path"] if row.get("mask_path") else None
        fixed = inspect_image(
            image,
            base_config,
            mask_path=mask,
            label=row["label"],
            split="tune",
            output_dir=out / "fixed" / row["sample_id"],
            save_stages=False,
        )
        outcome = run_agent_episode(
            image,
            agent_config=agent_config,
            profiles=profiles,
            output_dir=out / "agent",
            label=row["label"],
            split="tune",
            mask_path=mask,
        )
        episode = outcome.episode
        audit_errors = audit_trace(list(episode.events))
        samples.append(
            {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "fixed_status": fixed.result.status.value,
                "agent_status": episode.final_status.value,
                "fixed_total_ms": float(fixed.result.timing_ms.get("total", 0.0)),
                "agent_base_ms": episode.base_latency_ms,
                "agent_extra_ms": episode.extra_latency_ms,
                "actions_used": episode.actions_used,
                "profiles_used": episode.profiles_used,
                "rejections": len(episode.rejections),
                "audit_errors": audit_errors,
                "run_id": outcome.run_id,
            }
        )

    metrics = _metrics(samples)
    promotion = _promotion(metrics, agent_config.comparison)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_sha": code_sha,
        "config_hash": base_config.config_hash,
        "agent_config": agent_config.to_dict(),
        "split": "tune",
        "subset": {
            "size": len(subset),
            "seed": chosen_seed,
            "defect": sum(1 for row in subset if row["label"] == "defect"),
            "good": sum(1 for row in subset if row["label"] == "good"),
            "sample_ids": [row["sample_id"] for row in subset],
        },
        "samples": samples,
        "metrics": metrics,
        "promotion": promotion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="surface_sentinel.agent.comparison",
        description="T09 고정 base vs 에이전트 정책 비교 (tune 층화 부분집합).",
    )
    parser.add_argument("--tune-csv", type=Path, default=Path("data/splits/tune.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/kolektorsdd2"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent.toml"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("configs/profiles"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/agent-comparison"))
    parser.add_argument(
        "--evidence", type=Path, default=Path("docs/evidence/T09_agent_comparison.json")
    )
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--code-sha", default="")
    args = parser.parse_args(argv)

    agent_config = load_agent_config(args.agent_config)
    profiles = load_profiles(args.config, args.profiles_dir)
    report = run_comparison(
        tune_csv=args.tune_csv,
        raw_root=args.raw_root,
        agent_config=agent_config,
        profiles=profiles,
        output_dir=args.output_dir,
        subset_size=args.size,
        seed=args.seed,
        code_sha=args.code_sha,
    )
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = report["metrics"]
    promotion = report["promotion"]
    print(f"비교 완료: {args.evidence}")
    print(
        "부분집합: tune {}장 (defect {} / good {})".format(
            report["subset"]["size"],
            report["subset"]["defect"],
            report["subset"]["good"],
        )
    )
    print(
        "review: fixed {:.4f} ({}/{}) -> agent {:.4f} ({}/{})".format(
            metrics["review_fixed"]["rate"],
            metrics["review_fixed"]["count"],
            metrics["review_fixed"]["denominator"],
            metrics["review_agent"]["rate"],
            metrics["review_agent"]["count"],
            metrics["review_agent"]["denominator"],
        )
    )
    print(
        "escape: fixed {:.4f} ({}/{}) -> agent {:.4f} ({}/{})".format(
            metrics["escape_fixed"]["rate"],
            metrics["escape_fixed"]["count"],
            metrics["escape_fixed"]["denominator"],
            metrics["escape_agent"]["rate"],
            metrics["escape_agent"]["count"],
            metrics["escape_agent"]["denominator"],
        )
    )
    print(f"추가 p95 지연: {metrics['extra_latency_p95_ms']} ms")
    print(f"잘못된 행동: {metrics['wrong_actions']} / 감사 완전: {metrics['audit_complete']}")
    print(f"승격: {promotion['promoted']} (모드: {promotion['mode']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
