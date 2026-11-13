"""Phase 1 교차 증거 순수 모듈 단위 테스트 (에이전트 확장 제안 §2).

엔진 간 localization(IoU·containment·중심거리·면적비)·ROI 비교, 결정론 greedy
matching, 전용 플래그, dual overlay를 검증한다. 합성 관찰만 사용하며 기계 판정은
만들거나 바꾸지 않는다(observe-only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from agent_fakes import make_component, make_observation
from surface_sentinel.agent.evidence import (
    DUAL_COLOR_A,
    DUAL_COLOR_B,
    EVIDENCE_POLICY_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    CrossEvidenceAssessment,
    EvidenceFlag,
    EvidencePolicy,
    assess_cross_evidence,
    bbox_intersection_area,
    bbox_iou,
    default_evidence_policy,
    draw_dual_overlay,
    greedy_match,
    pair_metrics,
    roi_comparison,
    save_dual_overlay,
)
from surface_sentinel.contracts import RoiInfo, Status

BBox = tuple[int, int, int, int]


def _roi(bbox: BBox, *, uncertain: bool = False) -> RoiInfo:
    x, y, w, h = bbox
    return RoiInfo(
        mode="auto", bbox=bbox, area_px=w * h, area_ratio=0.5, uncertain=uncertain
    )


# ---------------------------------------------------------------------------
# bbox 기하 (ComponentFeature.bbox 는 x, y, w, h 계약)
# ---------------------------------------------------------------------------


def test_bbox_iou_identical_is_one() -> None:
    assert bbox_iou((10, 10, 20, 20), (10, 10, 20, 20)) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero() -> None:
    assert bbox_iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


def test_bbox_iou_half_overlap() -> None:
    # 교집합 200, 합집합 600 → 1/3
    assert bbox_iou((10, 10, 20, 20), (20, 10, 20, 20)) == pytest.approx(1 / 3)


def test_bbox_iou_containment() -> None:
    # 작은 박스가 큰 박스 안에 완전히 포함 → 교집합 100 / 합집합 400
    assert bbox_iou((0, 0, 20, 20), (0, 0, 10, 10)) == pytest.approx(0.25)


def test_bbox_iou_zero_area_is_zero() -> None:
    assert bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 0)) == 0.0


def test_bbox_intersection_area() -> None:
    assert bbox_intersection_area((10, 10, 20, 20), (20, 10, 20, 20)) == 200
    assert bbox_intersection_area((0, 0, 10, 10), (20, 20, 5, 5)) == 0


def test_pair_metrics_uses_component_area() -> None:
    # bbox 겹침은 200이지만 실제 컴포넌트 면적은 300/300 → containment 200/300
    left = make_component(0, (10, 10, 20, 20), area_px=300)
    right = make_component(1, (20, 10, 20, 20), area_px=300)
    match = pair_metrics(left, right)
    assert match.index_a == 0
    assert match.index_b == 1
    assert match.iou == pytest.approx(1 / 3)
    assert match.containment == pytest.approx(200 / 300)
    assert match.center_distance_px == pytest.approx(10.0)
    diagonal = math.sqrt(20**2 + 20**2)
    assert match.center_distance_norm == pytest.approx(10.0 / diagonal)
    assert match.area_ratio == pytest.approx(1.0)


def test_pair_metrics_area_ratio_is_max_over_min() -> None:
    left = make_component(0, (0, 0, 20, 20))
    right = make_component(1, (0, 0, 9, 9))
    match = pair_metrics(left, right)
    assert match.area_ratio == pytest.approx(400 / 81)
    assert match.containment == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 결정론 greedy matching
# ---------------------------------------------------------------------------


def test_greedy_match_one_to_one() -> None:
    a = [make_component(0, (0, 0, 10, 10)), make_component(1, (100, 100, 10, 10))]
    b = [make_component(0, (0, 0, 10, 10)), make_component(1, (100, 100, 10, 10))]
    result = greedy_match(a, b, iou_min=0.10)
    assert [(m.index_a, m.index_b) for m in result.matches] == [(0, 0), (1, 1)]
    assert result.unmatched_a == ()
    assert result.unmatched_b == ()


def test_greedy_match_tie_breaks_by_index() -> None:
    # A0 이 B0·B1 과 같은 IoU(1/3) → 인덱스가 작은 B0 먼저 선택(결정론)
    a = [make_component(0, (0, 0, 10, 10))]
    b = [make_component(0, (5, 0, 10, 10)), make_component(1, (0, 5, 10, 10))]
    result = greedy_match(a, b, iou_min=0.10)
    assert [(m.index_a, m.index_b) for m in result.matches] == [(0, 0)]
    assert result.unmatched_b == (1,)


def test_greedy_match_prefers_higher_iou() -> None:
    # A0-B1 IoU 가 더 높으면 인덱스와 무관하게 그 쌍이 먼저 선택된다.
    a = [make_component(0, (0, 0, 10, 10))]
    b = [make_component(0, (5, 0, 10, 10)), make_component(1, (0, 0, 10, 10))]
    result = greedy_match(a, b, iou_min=0.10)
    assert [(m.index_a, m.index_b) for m in result.matches] == [(0, 1)]
    assert result.unmatched_b == (0,)


def test_greedy_match_threshold_filters() -> None:
    a = [make_component(0, (0, 0, 10, 10))]
    b = [make_component(0, (9, 0, 10, 10))]  # IoU 10/190 ≈ 0.053 < 0.10
    result = greedy_match(a, b, iou_min=0.10)
    assert result.matches == ()
    assert result.unmatched_a == (0,)
    assert result.unmatched_b == (0,)


def test_greedy_match_empty_inputs() -> None:
    result = greedy_match([], [], iou_min=0.10)
    assert result.matches == ()
    assert result.unmatched_a == ()
    assert result.unmatched_b == ()
    one = greedy_match([make_component(0, (0, 0, 10, 10))], [], iou_min=0.10)
    assert one.unmatched_a == (0,)


def test_greedy_match_is_order_independent() -> None:
    a = [make_component(0, (0, 0, 10, 10)), make_component(1, (3, 0, 10, 10))]
    b = [make_component(0, (0, 0, 10, 10)), make_component(1, (3, 0, 10, 10))]
    first = greedy_match(a, b, iou_min=0.10)
    second = greedy_match(list(reversed(a)), list(reversed(b)), iou_min=0.10)
    assert [(m.index_a, m.index_b) for m in first.matches] == [
        (m.index_a, m.index_b) for m in second.matches
    ]


# ---------------------------------------------------------------------------
# ROI 비교
# ---------------------------------------------------------------------------


def test_roi_comparison_iou_and_uncertainty() -> None:
    comparison = roi_comparison(_roi((0, 0, 100, 100)), _roi((0, 0, 100, 100), uncertain=True))
    assert comparison is not None
    assert comparison.iou == pytest.approx(1.0)
    assert comparison.uncertain_a is False
    assert comparison.uncertain_b is True


def test_roi_comparison_missing_roi_is_none() -> None:
    assert roi_comparison(None, _roi((0, 0, 10, 10))) is None
    assert roi_comparison(_roi((0, 0, 10, 10)), None) is None


# ---------------------------------------------------------------------------
# 정책
# ---------------------------------------------------------------------------


def test_default_policy_hash_is_stable() -> None:
    policy = default_evidence_policy()
    assert policy.policy_hash == default_evidence_policy().policy_hash
    assert len(policy.policy_hash) == 64
    assert EVIDENCE_POLICY_VERSION == "evidence-v1"


def test_policy_hash_changes_with_thresholds() -> None:
    base = default_evidence_policy()
    changed = EvidencePolicy(iou_match_min=0.20)
    assert changed.policy_hash != base.policy_hash


def test_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        EvidencePolicy(iou_match_min=0.60, iou_strong_min=0.50)
    with pytest.raises(ValueError):
        EvidencePolicy(iou_match_min=0.0)
    with pytest.raises(ValueError):
        EvidencePolicy(iou_strong_min=1.5)
    with pytest.raises(ValueError):
        EvidencePolicy(area_ratio_max=0.5)
    with pytest.raises(ValueError):
        EvidencePolicy(score_gap_max=-0.1)
    with pytest.raises(ValueError):
        EvidencePolicy(roi_iou_min=1.2)
    with pytest.raises(ValueError):
        EvidencePolicy(center_distance_norm_max=0.0)


# ---------------------------------------------------------------------------
# 교차 평가 플래그
# ---------------------------------------------------------------------------


def _aligned_pair() -> tuple[object, object]:
    component_a = make_component(0, (10, 10, 20, 20))
    component_b = make_component(0, (10, 10, 20, 20))
    obs_a = make_observation(
        "opencv",
        Status.NG,
        components=[component_a],
        score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
        roi=_roi((0, 0, 100, 100)),
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[component_b],
        score=0.9,
        reasons=["DEFECT_EVIDENCE_STRONG"],
        task="det",
        roi=_roi((0, 0, 100, 100)),
    )
    return obs_a, obs_b


def test_aligned_ng_evidence_has_no_problem_flags() -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert assessment.flags == (EvidenceFlag.EVIDENCE_ALIGNED,)
    assert assessment.status_agree is True
    assert assessment.recommended_action == "NONE"
    assert len(assessment.matches) == 1
    assert assessment.matches[0].iou == pytest.approx(1.0)
    assert assessment.engine_a == "opencv"
    assert assessment.engine_b == "yolo-det"
    assert assessment.schema_version == EVIDENCE_SCHEMA_VERSION


def test_both_ok_without_components_is_aligned() -> None:
    obs_a = make_observation("opencv", Status.OK, roi=_roi((0, 0, 100, 100)))
    obs_b = make_observation("yolo-det", Status.OK, task="det", roi=_roi((0, 0, 100, 100)))
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert assessment.flags == (EvidenceFlag.EVIDENCE_ALIGNED,)
    assert assessment.recommended_action == "NONE"


def test_status_mismatch_flags_human_review() -> None:
    obs_a = make_observation("opencv", Status.NG, score=0.9)
    obs_b = make_observation("yolo-det", Status.OK, task="det")
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.STATUS_MISMATCH in assessment.flags
    assert EvidenceFlag.NG_SINGLE_ENGINE in assessment.flags
    assert assessment.status_agree is False
    assert assessment.recommended_action == "HUMAN_REVIEW"


def test_both_ng_is_not_single_engine() -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.NG_SINGLE_ENGINE not in assessment.flags


def test_error_engine_flags_human_review() -> None:
    obs_a = make_observation("opencv", Status.NG, score=0.9)
    obs_b = make_observation(
        "yolo-det", Status.ERROR, task="det", error_code="E_PROCESS_INTERNAL"
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.ERROR_ENGINE in assessment.flags
    assert assessment.recommended_action == "HUMAN_REVIEW"


def test_single_engine_evidence_flag() -> None:
    obs_a = make_observation(
        "opencv", Status.NG, components=[make_component(0, (10, 10, 20, 20))], score=0.9
    )
    obs_b = make_observation("yolo-det", Status.NG, task="det", score=0.9)
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.SINGLE_ENGINE_EVIDENCE in assessment.flags
    assert assessment.recommended_action == "INSPECT_EVIDENCE"


def test_localization_mismatch_when_no_pair_matches() -> None:
    obs_a = make_observation(
        "opencv", Status.NG, components=[make_component(0, (0, 0, 10, 10))], score=0.9
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (200, 200, 10, 10))],
        task="det",
        score=0.9,
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.LOCALIZATION_MISMATCH in assessment.flags
    assert assessment.matches == ()
    assert assessment.unmatched_a == (0,)
    assert assessment.unmatched_b == (0,)
    assert assessment.recommended_action == "INSPECT_EVIDENCE"


def test_unmatched_components_flag() -> None:
    obs_a = make_observation(
        "opencv",
        Status.NG,
        components=[make_component(0, (0, 0, 10, 10)), make_component(1, (50, 50, 10, 10))],
        score=0.9,
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (0, 0, 10, 10))],
        task="det",
        score=0.9,
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.UNMATCHED_COMPONENTS in assessment.flags
    assert assessment.unmatched_a == (1,)
    assert EvidenceFlag.EVIDENCE_ALIGNED not in assessment.flags


def test_weak_match_flag_below_strong_iou() -> None:
    obs_a = make_observation(
        "opencv", Status.NG, components=[make_component(0, (10, 10, 20, 20))], score=0.9
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (20, 10, 20, 20))],
        task="det",
        score=0.9,
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.WEAK_MATCH in assessment.flags
    assert assessment.matches[0].iou == pytest.approx(1 / 3)


def test_center_offset_large_flag() -> None:
    # IoU 0.25 로 매칭되지만 중심거리/대각선 ≈ 0.597 > 0.5
    obs_a = make_observation(
        "opencv", Status.NG, components=[make_component(0, (0, 0, 100, 10))], score=0.9
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (60, 0, 100, 10))],
        task="det",
        score=0.9,
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.CENTER_OFFSET_LARGE in assessment.flags


def test_area_ratio_large_flag() -> None:
    obs_a = make_observation(
        "opencv", Status.NG, components=[make_component(0, (0, 0, 20, 20))], score=0.9
    )
    obs_b = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (0, 0, 9, 9))],
        task="det",
        score=0.9,
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.AREA_RATIO_LARGE in assessment.flags


def test_score_gap_large_flag() -> None:
    obs_a, obs_b = _aligned_pair()
    from dataclasses import replace

    obs_b = replace(obs_b, result=replace(obs_b.result, score=0.1))
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.SCORE_GAP_LARGE in assessment.flags
    assert assessment.score_gap == pytest.approx(0.8)


def test_roi_mismatch_and_uncertain_diff_flags() -> None:
    obs_a = make_observation("opencv", Status.OK, roi=_roi((0, 0, 100, 100)))
    obs_b = make_observation(
        "yolo-det", Status.OK, task="det", roi=_roi((200, 200, 50, 50), uncertain=True)
    )
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert EvidenceFlag.ROI_MISMATCH in assessment.flags
    assert EvidenceFlag.ROI_UNCERTAIN_DIFF in assessment.flags
    assert assessment.roi is not None
    assert assessment.roi.iou == 0.0


def test_flags_are_in_declaration_order() -> None:
    obs_a = make_observation("opencv", Status.NG, score=0.9)
    obs_b = make_observation("yolo-det", Status.OK, task="det")
    assessment = assess_cross_evidence(obs_a, obs_b)
    order = list(EvidenceFlag)
    indices = [order.index(flag) for flag in assessment.flags]
    assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# 해시 결정론
# ---------------------------------------------------------------------------


def test_assessment_hash_is_deterministic() -> None:
    obs_a, obs_b = _aligned_pair()
    first = assess_cross_evidence(obs_a, obs_b)
    second = assess_cross_evidence(obs_a, obs_b)
    assert first.assessment_hash == second.assessment_hash
    assert len(first.assessment_hash) == 64
    assert first.policy_hash == default_evidence_policy().policy_hash


def test_assessment_hash_changes_with_input() -> None:
    obs_a, obs_b = _aligned_pair()
    base = assess_cross_evidence(obs_a, obs_b)
    moved = make_observation(
        "yolo-det",
        Status.NG,
        components=[make_component(0, (11, 10, 20, 20))],
        score=0.9,
        task="det",
        roi=_roi((0, 0, 100, 100)),
    )
    changed = assess_cross_evidence(obs_a, moved)
    assert changed.assessment_hash != base.assessment_hash


def test_assessment_hash_changes_with_policy() -> None:
    obs_a, obs_b = _aligned_pair()
    base = assess_cross_evidence(obs_a, obs_b)
    changed = assess_cross_evidence(obs_a, obs_b, policy=EvidencePolicy(iou_match_min=0.20))
    assert changed.policy_hash != base.policy_hash
    assert changed.assessment_hash != base.assessment_hash


def test_assessment_to_dict_is_json_serializable() -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    payload = assessment.to_dict()
    text = json.dumps(payload, sort_keys=True)
    assert json.loads(text)["assessment_hash"] == assessment.assessment_hash
    assert payload["flags"] == ["EVIDENCE_ALIGNED"]


def test_assessment_is_frozen() -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    assert isinstance(assessment, CrossEvidenceAssessment)
    with pytest.raises(AttributeError):
        assessment.recommended_action = "HUMAN_REVIEW"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# dual overlay
# ---------------------------------------------------------------------------


def _canvas() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (120, 320, 3), dtype=np.uint8)


def test_draw_dual_overlay_marks_both_engines() -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    canvas = _canvas()
    original = canvas.copy()
    drawn = draw_dual_overlay(canvas, obs_a, obs_b, assessment)
    assert drawn.shape == canvas.shape
    assert drawn.dtype == canvas.dtype
    assert np.array_equal(canvas, original)  # 입력 불변
    assert not np.array_equal(drawn, canvas)  # 실제로 그렸다
    # 두 엔진 색이 모두 존재해야 한다.
    assert np.any(np.all(drawn == np.array(DUAL_COLOR_A), axis=-1))
    assert np.any(np.all(drawn == np.array(DUAL_COLOR_B), axis=-1))


def test_draw_dual_overlay_rejects_non_bgr() -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    with pytest.raises(ValueError):
        draw_dual_overlay(np.zeros((10, 10), dtype=np.uint8), obs_a, obs_b, assessment)


def test_save_dual_overlay_writes_png(tmp_path: Path) -> None:
    obs_a, obs_b = _aligned_pair()
    assessment = assess_cross_evidence(obs_a, obs_b)
    target = tmp_path / "sub" / "dual.jpg"  # 확장자는 .png 로 고정
    saved = save_dual_overlay(_canvas(), obs_a, obs_b, assessment, target)
    assert saved.suffix == ".png"
    assert saved.is_file()
    assert saved.stat().st_size > 0
