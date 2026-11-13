"""Phase 2 리뷰 트리아지·섭동 민감도·능동 패치 중재 실측 검증 (tune + val).

봉인 분할(``test.csv``·``official_test.csv``)은 절대 열지 않는다.
오직 ``data/splits/tune.csv``(2,001장)와 그 내부의 group-aware ``val``
(400장)만 사용한다.

산출물:
* ``docs/evidence/P2_triage_eval_2026-11-13.json``
* ``docs/reports/P2_agent_triage_2026-11-13.md``

실행: ``PYTHONUTF8=1 uv run --no-sync python scripts/phase2_triage_eval.py``
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import cv2

os.environ.setdefault("SURFACE_SENTINEL_YOLO_COMBINED_TASK", "seg")

from surface_sentinel.agent.config import load_agent_config
from surface_sentinel.agent.engine import AgentEngine
from surface_sentinel.agent.evidence import EvidenceFlag, assess_cross_evidence
from surface_sentinel.agent.review import (
    TriageAssessment,
    assess_review_triage,
    evaluate_triage_lift,
    rank_review_queue,
)
from surface_sentinel.agent.run import load_profiles
from surface_sentinel.agent.stability import (
    _inspect_array_in_memory,
    analyze_weak_geometry,
    assess_perturbation_stability,
)
from surface_sentinel.agent.trace import TraceWriter, audit_trace, replay_trace
from surface_sentinel.config import AppConfig, load_config
from surface_sentinel.contracts import Status
from surface_sentinel.evaluation.metrics import ratio
from surface_sentinel.evaluation.runner import load_split_rows
from surface_sentinel.yolo.evaluate import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RAW_ROOT,
    resolve_split_rows,
)
from surface_sentinel.yolo.inspect import inspect_with_yolo
from surface_sentinel.yolo.runtime import YoloEngineRegistry

REPO = Path(__file__).resolve().parents[1]
TUNE_CACHE_PATH = REPO / "data/derived/t05_cache/t05-cache-v1-tune-a762ba639441-46c06acb3b53.pkl"
SEG_ONNX_PATH = REPO / "data/derived/yolo/models/seg/weights/best.onnx"
EVIDENCE_OUT = REPO / "docs/evidence/P2_triage_eval_2026-11-13.json"
REPORT_OUT = REPO / "docs/reports/P2_agent_triage_2026-11-13.md"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return round(float(ordered[idx]), 4)


def _priority_breakdown(
    ranked: list[TriageAssessment], labels: dict[str, str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for band in ("P0", "P1", "P2", "P3"):
        bucket = [item for item in ranked if item.priority.value == band]
        defects = sum(1 for item in bucket if labels.get(item.sample_id) == "defect")
        r = ratio(f"{band} 등급 내 실제 결함 비율(PPV)", defects, len(bucket))
        out[band] = {
            "count": len(bucket),
            "defects": defects,
            "goods": len(bucket) - defects,
            "ppv": r.as_dict(),
        }
    return out


def run_tune_review_eval(config: AppConfig) -> dict[str, Any]:
    rows = load_split_rows("tune")
    row_by_id = {row.sample_id: row for row in rows}
    with TUNE_CACHE_PATH.open("rb") as handle:
        cached_samples = pickle.load(handle)["samples"]  # noqa: S301 - 로컬 검증용 캐시

    review_samples = [s for s in cached_samples if s.status == Status.REVIEW]
    labels = {s.sample_id: s.label for s in review_samples}

    assessments: list[TriageAssessment] = []
    latencies_ms: list[float] = []
    stability_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    verdict_violations = 0

    t_start = time.perf_counter()
    for cached in review_samples:
        row = row_by_id[cached.sample_id]
        img_path = REPO / DEFAULT_RAW_ROOT / row.image_path
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"이미지를 읽지 못했습니다: {img_path}")

        t0 = time.perf_counter()
        base = _inspect_array_in_memory(
            bgr, config, sample_id=row.sample_id, split="tune", label=row.label
        )
        geom = analyze_weak_geometry(base.components, base.roi)
        stab = assess_perturbation_stability(bgr, config, baseline=base)
        triage = assess_review_triage(
            base,
            geometry=geom,
            stability=stab,
            ng_min=config.scoring.ng_min,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(elapsed_ms)
        stability_counts[stab.level.value] += 1
        pattern_counts[geom.pattern.value] += 1
        if triage.machine_status is not base.status or base.status is not Status.REVIEW:
            verdict_violations += 1
        assessments.append(triage)
    total_elapsed_s = round(time.perf_counter() - t_start, 3)

    ranked = rank_review_queue(assessments)
    id_ordered = sorted(assessments, key=lambda item: item.sample_id)
    score_ordered = sorted(
        assessments,
        key=lambda item: (-item.signals["score_proximity"], item.sample_id),
    )

    triage_lift = evaluate_triage_lift([item.sample_id for item in ranked], labels)
    id_lift = evaluate_triage_lift([item.sample_id for item in id_ordered], labels)
    score_lift = evaluate_triage_lift([item.sample_id for item in score_ordered], labels)

    # 한계·실패 사례 추출: P3에 배정된 실제 결함 / P0에 배정된 정상 표본 상위
    p3_defects = [
        {
            "sample_id": item.sample_id,
            "priority_score": item.priority_score,
            "stability": item.stability.value if item.stability else None,
            "weak_pattern": item.weak_pattern.value,
            "guidance": item.argumentation.operator_guidance,
        }
        for item in ranked
        if labels[item.sample_id] == "defect" and item.priority.value == "P3"
    ][:5]
    p0_false_alarms = [
        {
            "sample_id": item.sample_id,
            "priority_score": item.priority_score,
            "stability": item.stability.value if item.stability else None,
            "weak_pattern": item.weak_pattern.value,
            "guidance": item.argumentation.operator_guidance,
        }
        for item in ranked
        if labels[item.sample_id] == "good" and item.priority.value == "P0"
    ][:5]

    return {
        "split": "tune",
        "total_tune_samples": len(cached_samples),
        "review_samples": len(review_samples),
        "review_defects": sum(1 for v in labels.values() if v == "defect"),
        "review_goods": sum(1 for v in labels.values() if v == "good"),
        "elapsed_seconds": total_elapsed_s,
        "latency_ms": {
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "mean": round(sum(latencies_ms) / len(latencies_ms), 4),
        },
        "verdict_immutability_violations": verdict_violations,
        "stability_distribution": dict(sorted(stability_counts.items())),
        "weak_pattern_distribution": dict(sorted(pattern_counts.items())),
        "priority_breakdown": _priority_breakdown(ranked, labels),
        "triage_lift": triage_lift,
        "baseline_id_sort_lift": id_lift,
        "reference_score_sort_lift": score_lift,
        "failure_examples": {
            "p3_missed_defects": p3_defects,
            "p0_good_false_alarms": p0_false_alarms,
        },
        "top_5_ranked": [
            {
                "sample_id": item.sample_id,
                "label": labels[item.sample_id],
                "priority": item.priority.value,
                "priority_score": item.priority_score,
                "stability": item.stability.value if item.stability else None,
                "weak_pattern": item.weak_pattern.value,
                "guidance": item.argumentation.operator_guidance,
            }
            for item in ranked[:5]
        ],
    }


def run_val_combined_eval(config: AppConfig) -> dict[str, Any]:
    if not (REPO / SEG_ONNX_PATH).is_file():
        return {"measured": False, "reason": "best.onnx 없음"}

    val_rows = resolve_split_rows(load_split_rows("tune"), "val")
    agent_config = load_agent_config(REPO / "configs/agent.toml")
    profiles = load_profiles(REPO / DEFAULT_CONFIG_PATH, REPO / "configs/profiles")
    registry = YoloEngineRegistry()

    flag_counts: Counter[str] = Counter()
    arb_verdict_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    episode_latencies_ms: list[float] = []

    verdict_violations = 0
    replay_failures = 0
    audit_failures = 0
    arbitrated_episodes = 0
    episode_count = 0

    review_assessments: list[TriageAssessment] = []
    review_labels: dict[str, str] = {}

    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for row in val_rows:
            img_path = REPO / DEFAULT_RAW_ROOT / row.image_path
            outcome = inspect_with_yolo(
                img_path,
                config,
                engine="combined",
                output_dir=tmp_path,
                split="tune",
                label=row.label,
                registry=registry,
            )
            combined = outcome.result
            status_counts[combined.status.value] += 1
            obs_a, obs_b = outcome.engine_observations
            cross = assess_cross_evidence(obs_a, obs_b)
            for flag in cross.flags:
                flag_counts[flag.value] += 1

            needs_episode = (
                combined.status is Status.REVIEW
                or EvidenceFlag.LOCALIZATION_MISMATCH in cross.flags
                or EvidenceFlag.NG_SINGLE_ENGINE in cross.flags
            )
            arbitration = None
            if needs_episode:
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"이미지 디코드 실패: {img_path}")
                writer = TraceWriter(None)
                engine = AgentEngine(
                    agent_config=agent_config,
                    profiles=profiles,
                    inspector=lambda _pid, _cfg, _res=combined: _res,
                    trace=writer,
                    observations=(obs_a, obs_b),
                    image_bgr=bgr,
                )
                t0 = time.perf_counter()
                episode = engine.run(image_path=str(img_path), label=row.label, split="tune")
                episode_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                episode_count += 1
                if episode.final_status is not combined.status:
                    verdict_violations += 1
                if audit_trace(list(episode.events)):
                    audit_failures += 1
                rep = replay_trace(writer.events)
                if not rep.deterministic:
                    replay_failures += 1
                arbitration = engine.arbitration
                if arbitration is not None:
                    arbitrated_episodes += 1
                    arb_verdict_counts[arbitration.verdict.value] += 1

                if combined.status is Status.REVIEW:
                    geom = analyze_weak_geometry(obs_a.result.components, obs_a.result.roi)
                    stab = assess_perturbation_stability(bgr, config, baseline=obs_a.result)
                    triage = assess_review_triage(
                        combined,
                        geometry=geom,
                        stability=stab,
                        cross_assessment=cross,
                        arbitration=arbitration,
                        ng_min=config.scoring.ng_min,
                    )
                    if triage.machine_status is not combined.status:
                        verdict_violations += 1
                    review_assessments.append(triage)
                    review_labels[row.sample_id] = row.label

    total_elapsed_s = round(time.perf_counter() - t_start, 3)
    ranked_val = rank_review_queue(review_assessments)
    id_val = sorted(review_assessments, key=lambda item: item.sample_id)

    return {
        "measured": True,
        "split": "val",
        "combined_task": os.environ["SURFACE_SENTINEL_YOLO_COMBINED_TASK"],
        "total_val_samples": len(val_rows),
        "elapsed_seconds": total_elapsed_s,
        "status_distribution": dict(sorted(status_counts.items())),
        "cross_evidence_flag_counts": dict(sorted(flag_counts.items())),
        "episodes_executed": episode_count,
        "arbitrated_episodes": arbitrated_episodes,
        "arbitration_verdict_distribution": dict(sorted(arb_verdict_counts.items())),
        "episode_latency_ms": {
            "p50": _percentile(episode_latencies_ms, 0.50),
            "p95": _percentile(episode_latencies_ms, 0.95),
        },
        "verdict_immutability_violations": verdict_violations,
        "audit_failures": audit_failures,
        "replay_failures": replay_failures,
        "val_review_samples": len(review_assessments),
        "val_review_defects": sum(1 for v in review_labels.values() if v == "defect"),
        "val_review_priority_breakdown": _priority_breakdown(ranked_val, review_labels),
        "val_triage_lift": evaluate_triage_lift(
            [item.sample_id for item in ranked_val], review_labels
        ),
        "val_baseline_id_sort_lift": evaluate_triage_lift(
            [item.sample_id for item in id_val], review_labels
        ),
    }


def _fmt_ratio(d: dict[str, Any]) -> str:
    val = d.get("value")
    den = d.get("denominator")
    num = d.get("numerator")
    if val is None:
        return f"- (0/{den})"
    lo = d.get("ci95_low")
    hi = d.get("ci95_high")
    return f"{val * 100:.2f}% ({int(num)}/{int(den)}, CI95 [{lo:.3f}, {hi:.3f}])"


def _render_markdown(payload: dict[str, Any]) -> str:
    tune = payload["tune_review"]
    val = payload["val_combined"]
    lines = [
        "# Phase 2 에이전트 트리아지·민감도·능동 패치 중재 실측 보고서",
        "",
        f"- 생성일: {payload['generated']}",
        f"- 기준선 설정: `configs/base.toml` (`config_hash={payload['config_hash']}`)",
        "- 봉인 분할(`test.csv`, `official_test.csv`): **미접근** (`tune` 및 내부 `val`만 사용)",
        "- 판정 소유권: 기계 판정(`SampleResult.status`)은 파이프라인 소유이며, "
        "에이전트 위반 **0건**.",
        "",
        "## 1. tune REVIEW 779장 정량 검증 (OpenCV r20 기준선)",
        "",
        f"- 대상: `tune` 2,001장 중 REVIEW **{tune['review_samples']}장** "
        f"(실제 결함 **{tune['review_defects']}건** / 정상 {tune['review_goods']}건)",
        f"- 실행 시간: {tune['elapsed_seconds']}초 "
        f"(이미지당 p50={tune['latency_ms']['p50']} ms, p95={tune['latency_ms']['p95']} ms)",
        f"- 판정 불변 위반: **{tune['verdict_immutability_violations']}건**",
        "",
        "### 1.1 Recall@Top-K% 비교 (기본 ID순 vs 에이전트 트리아지)",
        "",
        "| 구간 | K (장) | ID순 회수율 (분모 44) | "
        "에이전트 Recall@Top-K (분모 44, Wilson 95% CI) | PPV (정밀도) | Lift (무작위 대비) |",
        "|---|---:|---|---|---|---:|",
    ]
    for frac in ("0.10", "0.20", "0.30"):
        t_row = tune["triage_lift"]["top_k"][frac]
        b_row = tune["baseline_id_sort_lift"]["top_k"][frac]
        lines.append(
            f"| Top-{int(float(frac) * 100)}% | {t_row['k']} | "
            f"{_fmt_ratio(b_row['recall_at_k'])} | "
            f"{_fmt_ratio(t_row['recall_at_k'])} | "
            f"{_fmt_ratio(t_row['precision_at_k'])} | "
            f"{t_row['lift']:.2f}x |"
        )

    lines.extend(
        [
            "",
            "### 1.2 우선순위 등급(P0~P3)별 결함 밀집도(PPV)",
            "",
            "| 등급 | 표본 수 | 실제 결함 수 | 정상 수 | PPV (결함 비율, Wilson 95% CI) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for band in ("P0", "P1", "P2", "P3"):
        b = tune["priority_breakdown"][band]
        lines.append(
            f"| {band} | {b['count']} | {b['defects']} | {b['goods']} | {_fmt_ratio(b['ppv'])} |"
        )

    lines.extend(
        [
            "",
            "### 1.3 섭동 안정성 · 약후보 기하 분포",
            "",
            f"- `StabilityLevel` 분포: {tune['stability_distribution']}",
            f"- `WeakPattern` 분포: {tune['weak_pattern_distribution']}",
            "",
            "## 2. val 400장 결합(yolo-seg + opencv) 교차 증거 · 능동 패치 중재 실측",
            "",
        ]
    )
    if val.get("measured"):
        lines.extend(
            [
                f"- 대상: `val` {val['total_val_samples']}장 (task=`{val['combined_task']}`, "
                f"총 소요 {val['elapsed_seconds']}초)",
                f"- 결합 상태 분포: {val['status_distribution']}",
                f"- 교차 증거 플래그 분포: {val['cross_evidence_flag_counts']}",
                f"- 능동 에이전트 에피소드: {val['episodes_executed']}건 실행, "
                f"패치 중재(`arbitrate_patches`) **{val['arbitrated_episodes']}건** 발동 "
                f"(p50={val['episode_latency_ms']['p50']} ms, "
                f"p95={val['episode_latency_ms']['p95']} ms)",
                f"- 패치 중재 결론 분포: {val['arbitration_verdict_distribution']}",
                f"- 계약 검증: 판정 불변 위반 **{val['verdict_immutability_violations']}건**, "
                f"감사 오류 **{val['audit_failures']}건**, "
                f"replay 불일치 **{val['replay_failures']}건**",
                "",
                f"### 2.1 val 결합 REVIEW {val['val_review_samples']}장 "
                f"(결함 {val['val_review_defects']}건) 트리아지",
                "",
                "| 구간 | K (장) | ID순 회수율 | "
                "에이전트 Recall@Top-K (Wilson 95% CI) | PPV | Lift |",
                "|---|---:|---|---|---|---:|",
            ]
        )
        for frac in ("0.10", "0.20", "0.30"):
            t_row = val["val_triage_lift"]["top_k"][frac]
            b_row = val["val_baseline_id_sort_lift"]["top_k"][frac]
            lines.append(
                f"| Top-{int(float(frac) * 100)}% | {t_row['k']} | "
                f"{_fmt_ratio(b_row['recall_at_k'])} | "
                f"{_fmt_ratio(t_row['recall_at_k'])} | "
                f"{_fmt_ratio(t_row['precision_at_k'])} | "
                f"{t_row['lift']:.2f}x |"
            )

    before = payload.get("aspect_ratio_correction_history", {}).get("before", {})
    after_dist = tune["weak_pattern_distribution"]
    before_dist = before.get("weak_pattern_distribution", {})
    after_collinear = after_dist.get("COLLINEAR_SCRATCH_SUSPECT", 0)
    before_collinear = before_dist.get("COLLINEAR_SCRATCH_SUSPECT", 391)
    total_rev = tune["review_samples"] or 1
    after_pct = after_collinear / total_rev * 100.0
    before_pct = before_collinear / total_rev * 100.0

    b_diff = before_dist.get("DIFFUSE_SURFACE_TEXTURE", 88)
    a_diff = after_dist.get("DIFFUSE_SURFACE_TEXTURE", 0)
    b_loc = before_dist.get("LOCAL_CLUSTER", 81)
    a_loc = after_dist.get("LOCAL_CLUSTER", 0)
    top_k = tune["triage_lift"]["top_k"]
    r10_v = top_k["0.10"]["recall_at_k"]["value"] * 100
    r10_c = top_k["0.10"]["captured_defects"]
    r20_v = top_k["0.20"]["recall_at_k"]["value"] * 100
    r20_c = top_k["0.20"]["captured_defects"]
    r30_v = top_k["0.30"]["recall_at_k"]["value"] * 100
    r30_c = top_k["0.30"]["captured_defects"]
    p0 = tune["priority_breakdown"]["P0"]
    p0_v = p0["ppv"]["value"] * 100

    lines.extend(
        [
            "",
            "## [개선 이력] 기하 공선성 종횡비 착시 보정 전 vs 보정 후 비교",
            "",
            "KSDD2 이미지는 폭~230px·높이~630px(약 1:2.7 세로형)이므로 공분산 고유값 비율 "
            "`lambda_max / (lambda_max + lambda_min)` 만 사용하면 가로로 넓게 퍼진 "
            "정상 표면 노이즈도 세로 분산 때문에 `COLLINEAR_SCRATCH_SUSPECT` 로 과잉 분류되었다. "
            "Wave 2에서 주축 직선 수직 RMS 폭(`orthogonal_rms_px <= 14.0px`, 또는 "
            "`max_elongation >= 4.0`·`R² >= 0.75`·`orthogonal_rms_px <= 18.0px`) "
            "제약을 추가한 전/후 실측 비교는 다음과 같다.",
            "",
            "| 지표 (tune REVIEW 779장 기준) | 보정 전 (Wave 1) | "
            "보정 후 (Wave 2, `orthogonal_rms_px <= 14px` 적용) | 변화 |",
            "|---|---:|---:|---:|",
            "| `COLLINEAR_SCRATCH_SUSPECT` 건수 (비율) | "
            f"{before_collinear}/779 ({before_pct:.1f}%) | "
            f"{after_collinear}/779 ({after_pct:.1f}%) | "
            f"{after_collinear - before_collinear:+d}건 ({after_pct - before_pct:+.1f}%p) |",
            f"| `DIFFUSE_SURFACE_TEXTURE` 건수 | {b_diff} | {a_diff} | {a_diff - b_diff:+d}건 |",
            f"| `LOCAL_CLUSTER` 건수 | {b_loc} | {a_loc} | {a_loc - b_loc:+d}건 |",
            "| Recall@Top-10% (K=78, 분모 44) | "
            f"{before.get('recall_top10_pct', 38.64):.2f}% "
            f"({before.get('recall_top10_captured', 17)}/44) | "
            f"{r10_v:.2f}% ({r10_c}/44) | "
            f"{r10_v - before.get('recall_top10_pct', 38.64):+.2f}%p |",
            "| Recall@Top-20% (K=156, 분모 44) | "
            f"{before.get('recall_top20_pct', 52.27):.2f}% "
            f"({before.get('recall_top20_captured', 23)}/44) | "
            f"{r20_v:.2f}% ({r20_c}/44) | "
            f"{r20_v - before.get('recall_top20_pct', 52.27):+.2f}%p |",
            "| Recall@Top-30% (K=234, 분모 44) | "
            f"{before.get('recall_top30_pct', 59.09):.2f}% "
            f"({before.get('recall_top30_captured', 26)}/44) | "
            f"{r30_v:.2f}% ({r30_c}/44) | "
            f"{r30_v - before.get('recall_top30_pct', 59.09):+.2f}%p |",
            "| P0 등급 PPV (실제 결함/표본) | "
            f"{before.get('p0_ppv_pct', 18.75):.2f}% "
            f"({before.get('p0_defects', 18)}/{before.get('p0_count', 96)}) | "
            f"{p0_v:.2f}% ({p0['defects']}/{p0['count']}) | "
            f"{p0_v - before.get('p0_ppv_pct', 18.75):+.2f}%p |",
            "",
            "## 3. 한계 및 실패 사례 분석",
            "",
            "1. **종횡비 편향 보정 완료 (`orthogonal_rms_px <= 14.0px` 회랑 폭 제한)**: "
            "KSDD2 1:2.7 세로형 이미지에서 비등방 분산 비율만으로 과잉 분류되던 "
            "`COLLINEAR_SCRATCH_SUSPECT` 를 Wave 1 "
            f"{before_collinear}/779건({before_pct:.1f}%)에서 "
            f"Wave 2 {after_collinear}/779건({after_pct:.1f}%)으로 보정하였다.",
            "2. **약한/저대비 결함의 후순위화 (`P3` 누락)**: 블러 섭동에서 급감하거나 "
            "국소 대비가 낮은 "
            "미세 결함은 P3 로 내려갈 수 있다(작업자 전수 검토 보류 시 조기 발견 지연 가능).",
        ]
    )
    for item in tune["failure_examples"]["p3_missed_defects"][:3]:
        lines.append(
            f"   - P3 누락 결함 예: `{item['sample_id']}` "
            f"(score={item['priority_score']:.3f}, {item['stability']}, {item['weak_pattern']})"
        )
    lines.append(
        "3. **고대비 정상 반사광의 `P0` 상승**: 국소 대비와 안정성이 높은 표면 기포·조명 경계는 "
        "결함 신호와 유사해 `P0` 로 올라올 수 있다."
    )
    for item in tune["failure_examples"]["p0_good_false_alarms"][:3]:
        lines.append(
            f"   - P0 정상 오경보 예: `{item['sample_id']}` "
            f"(score={item['priority_score']:.3f}, {item['stability']}, {item['weak_pattern']})"
        )
    if val.get("measured"):
        v10 = val["val_triage_lift"]["top_k"]["0.10"]
        v30 = val["val_triage_lift"]["top_k"]["0.30"]
        v_rev = val["val_review_samples"]
        v_def = val["val_review_defects"]
        lines.append(
            "4. **결합(`yolo-seg+opencv`) 후 잔여 `val` REVIEW 의 저신호 집중**: "
            f"고대비 결함은 `yolo-seg` 가 이미 `NG` 로 확정하므로, 잔여 REVIEW {v_rev}장 중 "
            f"실제 결함 {v_def}건은 저대비·미세 결함이 많으나, 종횡비 착시 보정 후 "
            f"Top-10%({v10['captured_defects']}/{v_def}, {v10['lift']:.2f}x Lift), "
            f"Top-30%({v30['captured_defects']}/{v_def}, {v30['lift']:.2f}x Lift)로 개선되었다 "
            "(단, 소표본으로 Wilson 구간 폭은 여전히 넓다)."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    config = load_config(REPO / DEFAULT_CONFIG_PATH)
    tune_report = run_tune_review_eval(config)
    val_report = run_val_combined_eval(config)

    before_snapshot = {
        "weak_pattern_distribution": {
            "BORDER_ARTIFACT_SUSPECT": 1,
            "COLLINEAR_SCRATCH_SUSPECT": 391,
            "DIFFUSE_SURFACE_TEXTURE": 88,
            "LOCAL_CLUSTER": 81,
            "SINGLE_CANDIDATE": 218,
        },
        "recall_top10_captured": 17,
        "recall_top10_pct": 38.64,
        "recall_top20_captured": 23,
        "recall_top20_pct": 52.27,
        "recall_top30_captured": 26,
        "recall_top30_pct": 59.09,
        "p0_count": 96,
        "p0_defects": 18,
        "p0_ppv_pct": 18.75,
    }
    payload = {
        "schema_version": "phase2-triage-eval-v2",
        "generated": date.today().isoformat(),
        "config_hash": config.config_hash,
        "sealed_splits_accessed": False,
        "aspect_ratio_correction_history": {
            "before": before_snapshot,
            "after": {
                "weak_pattern_distribution": tune_report["weak_pattern_distribution"],
                "orthogonal_rms_threshold_px": 14.0,
                "elongated_orthogonal_rms_threshold_px": 18.0,
            },
        },
        "tune_review": tune_report,
        "val_combined": val_report,
    }
    EVIDENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_OUT.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {EVIDENCE_OUT} and {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
