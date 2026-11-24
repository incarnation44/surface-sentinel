"""Phase 4 반사실적 경계 해석 · 2D 스트립 공간 핫스팟 · 사람-기계 피드백 캘리브레이션 실측.

봉인 분할(``test``·``official_test``)은 절대 열지 않는다.
오직 ``tune`` 2,001장 캐시(``t05_adopted_r20`` / ``configs/base.toml``)와 ``tune.csv``만
사용해 다음을 실측·저장한다.

1. ``tune`` REVIEW 779장의 닫힌 형태 반사실 병목 분해(면적 병목 vs 경계 전환 가능 비율)
2. 대표 사례(escape ``10090``, REVIEW 경계, NG 확정)의 Leave-One-Out 및 반사실 해
3. ``tune`` 2,001장(NG 249 / REVIEW 779) 6×14 스트립 공간 핫스팟·가장자리(15%) 집중 비율
4. 결정론 시뮬레이션 검토 이력 기반 Phase 4 캘리브레이션 및 하드 샘플 큐레이션 검증

산출물:
* ``docs/evidence/P4_counterfactual_eval_2026-11-13.json``
* ``docs/reports/P4_counterfactual_calibration_2026-11-13.md``
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Any

from surface_sentinel.agent.calibration import analyze_human_feedback
from surface_sentinel.agent.counterfactual import explain_counterfactual
from surface_sentinel.config import AppConfig, load_config
from surface_sentinel.contracts import (
    ComponentFeature,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Status,
)
from surface_sentinel.evaluation.cache import (
    CACHE_DIR,
    DEFAULT_RAW_ROOT,
    CachedSample,
    build_cache,
    cache_meta,
    cache_path,
    load_cache,
    save_cache,
)
from surface_sentinel.evaluation.fullverify import FULLVERIFY_GOOD_TARGET
from surface_sentinel.evaluation.metrics import ratio
from surface_sentinel.evaluation.runner import DEFAULT_SPLITS_DIR, load_split_rows, split_sha256
from surface_sentinel.evaluation.sampling import DEFAULT_SEED, build_sample_plan
from surface_sentinel.pipeline.stages import _component_evidence

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO / "configs/base.toml"
EVIDENCE_OUT = REPO / "docs/evidence/P4_counterfactual_eval_2026-11-13.json"
REPORT_OUT = REPO / "docs/reports/P4_counterfactual_calibration_2026-11-13.md"


def _load_tune_cached_samples(
    cfg: AppConfig, rows: list[Any]
) -> tuple[CachedSample, ...]:
    plan = build_sample_plan(
        rows,
        split="tune",
        good_target=FULLVERIFY_GOOD_TARGET,
        seed=DEFAULT_SEED,
    )
    split_hash = split_sha256(REPO / DEFAULT_SPLITS_DIR / "tune.csv")
    meta = cache_meta(plan, cfg, split_sha256=split_hash)
    target = cache_path(meta, REPO / CACHE_DIR)
    if target.is_file():
        return load_cache(target, meta)
    samples = build_cache(plan, cfg, raw_root=REPO / DEFAULT_RAW_ROOT)
    save_cache(samples, meta, target)
    return samples


def _cached_to_sample_result(cached: Any, cfg: AppConfig, config_hash: str) -> SampleResult:
    comps = tuple(
        ComponentFeature(
            index=idx,
            area_px=int(c.area_px),
            bbox=tuple(int(v) for v in c.bbox),
            area_ratio=0.0,
            aspect_ratio=(float(c.bbox[2]) / float(c.bbox[3])) if c.bbox[3] else 1.0,
            elongation=1.0,
            solidity=1.0,
            mean_contrast_gray=float(c.mean_contrast_gray),
            max_contrast_gray=float(c.max_contrast_gray),
            integrated_contrast=float(c.integrated_contrast),
            border_distance_px=float(c.border_distance_px),
            touches_roi_border=bool(c.touches_roi_border),
            evidence=_component_evidence(int(c.area_px), float(c.mean_contrast_gray), cfg),
        )
        for idx, c in enumerate(cached.components)
    )
    q_flags = tuple(ReasonCode(v) for v in cached.quality_flags)
    reasons = tuple(ReasonCode(v) for v in cached.reasons)
    return SampleResult(
        run_id="tune-cache",
        sample_id=str(cached.sample_id),
        original_id=str(cached.original_id),
        split=str(cached.split),
        label=str(cached.label),
        status=cached.status,
        score=float(cached.score),
        reasons=reasons,
        components=comps,
        quality=QualityInfo(
            blur_variance=float(cached.blur_variance),
            mean_gray=float(cached.mean_gray),
            saturated_ratio=float(cached.saturated_ratio),
            flags=q_flags,
        ),
        roi=RoiInfo(
            mode=cfg.roi.mode,
            bbox=(0, 0, 230, 630),
            area_px=144900,
            area_ratio=1.0,
            uncertain=bool(cached.roi_uncertain),
        ),
        overlay_path=str(cached.overlay),
        config_hash=config_hash,
    )

def _evaluate_review_bottlenecks(
    review_results: list[SampleResult],
    cfg: AppConfig,
) -> dict[str, Any]:
    area_bottleneck_ids: list[str] = []
    boundary_reachable_ids: list[str] = []
    no_component_ids: list[str] = []
    contrast_bottleneck_ids: list[str] = []
    pivotal_count = 0
    area_bottleneck_defects = 0
    boundary_reachable_defects = 0
    no_comp_defects = 0

    sample_examples: dict[str, dict[str, Any]] = {}
    for sr in review_results:
        rep = explain_counterfactual(sr, cfg)
        if rep.pivotal_component_indices:
            pivotal_count += 1
        if not rep.components:
            no_component_ids.append(sr.sample_id)
            if sr.label == "defect":
                no_comp_defects += 1
            continue
        top = max(rep.components, key=lambda c: (c.evidence, -c.index))
        if top.target_contrast_for_ng is None and top.target_area_for_ng is not None:
            area_bottleneck_ids.append(sr.sample_id)
            if sr.label == "defect":
                area_bottleneck_defects += 1
            if "area_bottleneck_sample" not in sample_examples and sr.label == "defect":
                sample_examples["area_bottleneck_sample"] = rep.to_dict()
        elif top.target_area_for_ng is not None and top.target_contrast_for_ng is not None:
            boundary_reachable_ids.append(sr.sample_id)
            if sr.label == "defect":
                boundary_reachable_defects += 1
            if "boundary_reachable_sample" not in sample_examples and sr.label == "defect":
                sample_examples["boundary_reachable_sample"] = rep.to_dict()
        else:
            contrast_bottleneck_ids.append(sr.sample_id)

    total = len(review_results)
    with_comp = total - len(no_component_ids)
    return {
        "total_review": total,
        "with_components": with_comp,
        "without_components": len(no_component_ids),
        "pivotal_samples": pivotal_count,
        "pivotal_ratio": ratio("REVIEW 중 결정적 후보 보유 비율", pivotal_count, total).as_dict(),
        "area_bottleneck_count": len(area_bottleneck_ids),
        "area_bottleneck_defects": area_bottleneck_defects,
        "area_bottleneck_ratio_all_review": ratio(
            "전체 REVIEW 779장 중 면적 병목(대비 상승만으로 NG 불가) 비율",
            len(area_bottleneck_ids),
            total,
        ).as_dict(),
        "area_bottleneck_ratio_with_components": ratio(
            "후보 보유 REVIEW 중 면적 병목 비율",
            len(area_bottleneck_ids),
            with_comp,
        ).as_dict(),
        "boundary_reachable_count": len(boundary_reachable_ids),
        "boundary_reachable_defects": boundary_reachable_defects,
        "boundary_reachable_ratio_all_review": ratio(
            "전체 REVIEW 779장 중 경계 전환 가능(면적 또는 대비 상승 시 NG) 비율",
            len(boundary_reachable_ids),
            total,
        ).as_dict(),
        "boundary_reachable_ratio_with_components": ratio(
            "후보 보유 REVIEW 중 경계 전환 가능 비율",
            len(boundary_reachable_ids),
            with_comp,
        ).as_dict(),
        "contrast_or_invalid_bottleneck_count": len(contrast_bottleneck_ids),
        "no_component_defects": no_comp_defects,
        "representative_examples": sample_examples,
    }


def _evaluate_spatial_hotspot(
    all_results: list[SampleResult],
    dim_by_id: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    cols, rows = 6, 14
    ng_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    rev_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    ng_total, ng_edge = 0, 0
    rev_total, rev_edge = 0, 0
    for sr in all_results:
        if sr.status not in {Status.NG, Status.REVIEW} or not sr.components:
            continue
        w_img, h_img = dim_by_id.get(sr.sample_id, (230, 630))
        for c in sr.components:
            cx = float(c.bbox[0]) + float(c.bbox[2]) / 2.0
            cy = float(c.bbox[1]) + float(c.bbox[3]) / 2.0
            nx = min(max(cx / float(w_img), 0.0), 0.999999)
            ny = min(max(cy / float(h_img), 0.0), 0.999999)
            gx = int(nx * cols)
            gy = int(ny * rows)
            is_edge = nx <= 0.15 or nx >= 0.85
            if sr.status is Status.NG:
                ng_grid[gy][gx] += 1
                ng_total += 1
                if is_edge:
                    ng_edge += 1
            else:
                rev_grid[gy][gx] += 1
                rev_total += 1
                if is_edge:
                    rev_edge += 1
    return {
        "grid_shape": [rows, cols],
        "ng_candidate_hits": ng_total,
        "review_candidate_hits": rev_total,
        "ng_edge_15pct_ratio": ratio(
            "NG 후보 중 좌·우 가장자리(15%) 집중 비율", ng_edge, ng_total
        ).as_dict(),
        "review_edge_15pct_ratio": ratio(
            "REVIEW 후보 중 좌·우 가장자리(15%) 집중 비율", rev_edge, rev_total
        ).as_dict(),
        "ng_grid": ng_grid,
        "review_grid": rev_grid,
    }

def _simulate_calibration_eval(
    review_results: list[SampleResult], cfg: AppConfig
) -> dict[str, Any]:
    rng = random.Random(20261113)
    selected = list(review_results)
    rng.shuffle(selected)
    sampled = selected[:120]

    decisions: list[dict[str, str]] = []
    for idx, sr in enumerate(sampled):
        if sr.label == "defect":
            if idx % 13 == 0:
                decisions.append({
                    "sample_id": sr.sample_id,
                    "final_status": "REVIEW",
                    "reason_code": "INSUFFICIENT_EVIDENCE",
                    "note": "희미한 미세 결함 경계로 재촬영 보류",
                })
            else:
                decisions.append({
                    "sample_id": sr.sample_id,
                    "final_status": "NG",
                    "reason_code": "CONFIRMED_DEFECT",
                    "note": "작업자 결함 확정",
                })
        else:
            decisions.append({
                "sample_id": sr.sample_id,
                "final_status": "OK",
                "reason_code": "FALSE_ALARM",
                "note": "표면 반사/미세 텍스처 오경보 기각",
            })

    summary = analyze_human_feedback(
        review_results,
        decisions,
        ng_min=float(cfg.scoring.ng_min),
    )
    return summary.to_dict()

def _render_markdown(payload: dict[str, Any]) -> str:
    rb = payload["review_bottleneck_breakdown"]
    hs = payload["spatial_hotspot_tune"]
    cal = payload["calibration_simulation"]
    area_r = rb["area_bottleneck_ratio_all_review"]
    bound_r = rb["boundary_reachable_ratio_all_review"]
    ng_edge = hs["ng_edge_15pct_ratio"]
    rev_edge = hs["review_edge_15pct_ratio"]
    no_comp = rb["without_components"]
    tot_rev = rb["total_review"]

    lines = [
        "# Phase 4 반사실적 경계 해석 · 2D 스트립 핫스팟 · 사람-기계 피드백 캘리브레이션",
        "",
        f"- 생성일: {payload['generated']}",
        f"- 기준선 설정: `configs/base.toml` (`config_hash={payload['config_hash']}`)",
        "- 봉인 분할(`test`·`official_test`): **미접근** (`tune` 2,001장 캐시·메타데이터만 사용)",
        "- 판정 소유권: 100% observe-only (기계 판정·설정 불변)",
        "",
        "## 1. `tune` REVIEW 779장 반사실적 병목 분해 (Closed-Form)",
        "",
        f"- 전체 `tune` REVIEW 표본: **{tot_rev}장** "
        f"(후보 보유 {rb['with_components']}장, 무후보 품질/ROI 보류 {no_comp}장)",
        f"- Leave-One-Out 결정적(Pivotal) 후보 보유 샘플: "
        f"**{rb['pivotal_samples']} / {tot_rev}장** "
        f"({rb['pivotal_ratio']['value']*100:.2f}%)",
        "",
        "| 병목 분류 | 건수 | 전체 REVIEW(779) 비율 | 95% Wilson CI | 포함 실제 결함 |",
        "|---|---:|---:|---|---:|",
        f"| 면적 병목 (면적 <= 133px, 대비 상한 33 불가) | {rb['area_bottleneck_count']} | "
        f"{area_r['value']*100:.2f}% ({area_r['numerator']}/{area_r['denominator']}) | "
        f"[{area_r['ci95_low']*100:.2f}%, {area_r['ci95_high']*100:.2f}%] | "
        f"{rb['area_bottleneck_defects']}건 |",
        f"| 경계 전환 가능 (면적 > 133px, 상승 시 NG) | {rb['boundary_reachable_count']} | "
        f"{bound_r['value']*100:.2f}% ({bound_r['numerator']}/{bound_r['denominator']}) | "
        f"[{bound_r['ci95_low']*100:.2f}%, {bound_r['ci95_high']*100:.2f}%] | "
        f"{rb['boundary_reachable_defects']}건 |",
        f"| 무후보(품질/ROI 플래그 보류) | {no_comp} | "
        f"{no_comp/tot_rev*100:.2f}% ({no_comp}/{tot_rev}) | - | "
        f"{rb['no_component_defects']}건 |",
        "",
        "## 2. 대표 사례 반사실적 해석 (escape `10090` / REVIEW 경계 / NG 확정)",
        "",
    ]
    for key, label in (
        ("escape_10090", "Escape 사례 `10090` (OK)"),
        ("ng_representative", "대표 NG 확정 사례"),
    ):
        ex = payload["representative_samples"].get(key)
        if ex:
            lines.append(f"- **{label}** (`{ex['sample_id']}`): {ex['headline_ko']}")
    for key, label in (
        ("area_bottleneck_sample", "REVIEW 면적 병목 결함 사례"),
        ("boundary_reachable_sample", "REVIEW 경계 전환 가능 결함 사례"),
    ):
        ex = rb["representative_examples"].get(key)
        if ex:
            lines.append(f"- **{label}** (`{ex['sample_id']}`): {ex['headline_ko']}")

    lines.extend([
        "",
        "## 3. `tune` 2,001장 2D 스트립 공간 핫스팟 (6×14 격자)",
        "",
        f"- NG 후보 총 {hs['ng_candidate_hits']}건 중 좌·우 가장자리(15%) 집중: "
        f"**{ng_edge['value']*100:.2f}%** "
        f"({ng_edge['numerator']}/{ng_edge['denominator']})",
        f"- REVIEW 후보 총 {hs['review_candidate_hits']}건 중 좌·우 가장자리(15%) 집중: "
        f"**{rev_edge['value']*100:.2f}%** "
        f"({rev_edge['numerator']}/{rev_edge['denominator']})",
        "",
        "## 4. Phase 4 사람-기계 피드백 캘리브레이션 (시드 20261113, 120건 시뮬레이션)",
        "",
        f"- 검토 완료: **{cal['decided_count']} / {cal['total_review_samples']}건**, "
        f"하드 샘플 추출: **{len(cal['hard_examples'])}건**",
        "",
        "| 우선순위 | REVIEW 건수 | 검토 완료 | 사람 확정 NG | 기각 OK | 확정 NG 비율 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for band in ("P0", "P1", "P2", "P3"):
        pb = cal["by_priority"][band]
        pr = pb["confirmed_ng_ratio"]
        pct_txt = (
            "-"
            if pr["value"] is None
            else f"{pr['value']*100:.1f}% ({pr['numerator']}/{pr['denominator']})"
        )
        lines.append(
            f"| {band} | {pb['review_count']} | {pb['decided_count']} | "
            f"{pb['confirmed_ng_count']} | {pb['false_alarm_ok_count']} | {pct_txt} |"
        )
    lines.append("")
    for rec in cal["recommendations_ko"]:
        lines.append(f"- {rec}")
    lines.append("")
    return chr(10).join(lines)

def main() -> int:
    cfg = load_config(BASE_CONFIG_PATH)
    rows = load_split_rows("tune")
    dim_by_id = {row.sample_id: (int(row.width), int(row.height)) for row in rows}
    cached_samples = _load_tune_cached_samples(cfg, rows)

    all_results = [_cached_to_sample_result(c, cfg, cfg.config_hash) for c in cached_samples]
    review_results = [r for r in all_results if r.status is Status.REVIEW]
    ng_results = [r for r in all_results if r.status is Status.NG]
    s_10090 = next(r for r in all_results if r.sample_id == "10090")
    ng_rep = next((r for r in ng_results if r.components), ng_results[0])

    bottleneck_summary = _evaluate_review_bottlenecks(review_results, cfg)
    hotspot_summary = _evaluate_spatial_hotspot(all_results, dim_by_id)
    calibration_summary = _simulate_calibration_eval(review_results, cfg)

    payload = {
        "schema_version": "phase4-counterfactual-eval-v1",
        "generated": date.today().isoformat(),
        "config_hash": cfg.config_hash,
        "sealed_splits_accessed": False,
        "tune_samples": len(all_results),
        "review_bottleneck_breakdown": bottleneck_summary,
        "representative_samples": {
            "escape_10090": explain_counterfactual(s_10090, cfg).to_dict(),
            "ng_representative": explain_counterfactual(ng_rep, cfg).to_dict(),
        },
        "spatial_hotspot_tune": hotspot_summary,
        "calibration_simulation": calibration_summary,
    }
    EVIDENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
    )
    REPORT_OUT.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {EVIDENCE_OUT} and {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
