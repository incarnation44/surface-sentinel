"""리뷰 트리아지·3단 논증 카드·Recall@Top-K(`agent/review.py`) 단위 테스트.

기계 판정 불변, 위험도 우선 정렬, 결정론적 tie-break 및 Lift 신뢰구간 계산을 검증한다.
"""

from __future__ import annotations

from dataclasses import replace

from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.agent.evidence import assess_cross_evidence
from surface_sentinel.agent.review import (
    ReviewPriority,
    assess_review_triage,
    evaluate_triage_lift,
    rank_review_queue,
)
from surface_sentinel.agent.stability import (
    PerturbationSample,
    StabilityLevel,
    StabilityPolicy,
    StabilityReport,
    WeakPattern,
    analyze_weak_geometry,
)
from surface_sentinel.contracts import Status


def test_triage_preserves_machine_status_and_prioritizes_strong_defect() -> None:
    strong_comp = replace(
        make_component(0, (40, 100, 18, 6), area_px=110, evidence=0.47),
        mean_contrast_gray=32.0,
        max_contrast_gray=78.0,
        integrated_contrast=3520.0,
        elongation=4.5,
    )
    strong_res = replace(
        make_result(Status.REVIEW, reasons=["EVIDENCE_IN_REVIEW_BAND"], score=0.47),
        sample_id="defect_cand",
        components=(strong_comp,),
    )
    geom_strong = analyze_weak_geometry(strong_res.components, strong_res.roi)
    stab_strong = StabilityReport(
        level=StabilityLevel.STABLE,
        base_status=Status.REVIEW,
        base_score=0.47,
        perturbations=(
            PerturbationSample("brightness_plus_5pct", Status.REVIEW, 0.46, -0.01, 0.9),
            PerturbationSample("gaussian_blur_3x3", Status.REVIEW, 0.44, -0.03, 0.85),
        ),
        status_preserved_count=2,
        mean_score_retention=0.93617,
        max_abs_score_delta=0.03,
        policy_hash=StabilityPolicy().policy_hash(),
        stability_hash="a" * 64,
    )
    obs_a = make_observation("opencv", Status.REVIEW, components=[strong_comp], score=0.47)
    obs_b = make_observation(
        "yolo-seg", Status.NG, components=[strong_comp], score=0.81, task="seg"
    )
    cross = assess_cross_evidence(obs_a, obs_b)

    t_strong = assess_review_triage(
        strong_res,
        geometry=geom_strong,
        stability=stab_strong,
        cross_assessment=cross,
    )
    assert t_strong.machine_status is Status.REVIEW
    assert t_strong.priority in {ReviewPriority.P0, ReviewPriority.P1}
    assert t_strong.priority_score >= 0.65
    assert t_strong.argumentation.hypothesis
    assert t_strong.argumentation.experiment
    assert t_strong.argumentation.verdict
    assert t_strong.argumentation.operator_guidance

    weak_comp = replace(
        make_component(0, (1, 10, 8, 8), area_px=42, evidence=0.28),
        mean_contrast_gray=17.0,
        max_contrast_gray=24.0,
        integrated_contrast=714.0,
        border_distance_px=1.0,
        touches_roi_border=True,
    )
    weak_res = replace(
        make_result(Status.REVIEW, reasons=["BORDER_TOUCHING_CANDIDATE"], score=0.28),
        sample_id="border_noise",
        components=(weak_comp, replace(weak_comp, index=1, bbox=(2, 30, 8, 8))),
    )
    geom_weak = analyze_weak_geometry(weak_res.components, weak_res.roi)
    assert geom_weak.pattern is WeakPattern.BORDER_ARTIFACT_SUSPECT
    stab_weak = StabilityReport(
        level=StabilityLevel.UNSTABLE,
        base_status=Status.REVIEW,
        base_score=0.28,
        perturbations=(
            PerturbationSample("brightness_plus_5pct", Status.OK, 0.0, -0.28, 0.0),
            PerturbationSample("gaussian_blur_3x3", Status.OK, 0.0, -0.28, 0.0),
        ),
        status_preserved_count=0,
        mean_score_retention=0.0,
        max_abs_score_delta=0.28,
        policy_hash=StabilityPolicy().policy_hash(),
        stability_hash="b" * 64,
    )
    t_weak = assess_review_triage(weak_res, geometry=geom_weak, stability=stab_weak)
    assert t_weak.priority_score < t_strong.priority_score
    assert t_weak.priority in {ReviewPriority.P2, ReviewPriority.P3}

    ranked = rank_review_queue([t_weak, t_strong])
    assert [item.sample_id for item in ranked] == ["defect_cand", "border_noise"]


def test_rank_review_queue_tie_breaks_by_sample_id() -> None:
    r1 = replace(make_result(Status.OK), sample_id="s_b")
    r2 = replace(make_result(Status.OK), sample_id="s_a")
    t1 = assess_review_triage(r1)
    t2 = assess_review_triage(r2)
    ranked = rank_review_queue([t1, t2])
    assert [item.sample_id for item in ranked] == ["s_a", "s_b"]


def test_evaluate_triage_lift_computes_denominators_ci_and_handles_empty() -> None:
    ranked_ids = [f"s{i:02d}" for i in range(10)]
    labels = {f"s{i:02d}": ("defect" if i in {0, 1, 5} else "good") for i in range(10)}
    report = evaluate_triage_lift(ranked_ids, labels, top_fractions=(0.10, 0.20, 0.30))
    assert report["total_review"] == 10
    assert report["total_defects"] == 3
    top20 = report["top_k"]["0.20"]
    assert top20["k"] == 2
    assert top20["captured_defects"] == 2
    assert top20["recall_at_k"]["denominator"] == 3
    assert abs(top20["recall_at_k"]["value"] - (2 / 3)) < 1e-6
    assert top20["recall_at_k"]["ci95_low"] is not None
    assert top20["precision_at_k"]["denominator"] == 2
    assert top20["precision_at_k"]["value"] == 1.0
    assert top20["lift"] > 3.0

    empty = evaluate_triage_lift([], {})
    assert empty["total_review"] == 0
    assert empty["total_defects"] == 0
    assert empty["top_k"]["0.10"]["recall_at_k"]["value"] is None
    assert empty["top_k"]["0.10"]["lift"] is None
