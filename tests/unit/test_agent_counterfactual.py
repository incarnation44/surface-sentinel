"""반사실적 경계 해석 & Leave-One-Out 귀속도(agent.counterfactual) 단위 테스트."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent_fakes import make_result
from surface_sentinel.agent.counterfactual import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    explain_counterfactual,
)
from surface_sentinel.config import default_config
from surface_sentinel.contracts import ComponentFeature, ReasonCode, Status
from surface_sentinel.pipeline.stages import _component_evidence


def _comp(
    index: int,
    area_px: int,
    mean_contrast_gray: float,
    *,
    bbox: tuple[int, int, int, int] = (20, 30, 12, 12),
    touches_roi_border: bool = False,
) -> ComponentFeature:
    cfg = default_config()
    ev = _component_evidence(area_px, mean_contrast_gray, cfg)
    return ComponentFeature(
        index=index,
        area_px=area_px,
        bbox=bbox,
        area_ratio=0.01,
        aspect_ratio=1.0,
        elongation=1.0,
        solidity=1.0,
        mean_contrast_gray=mean_contrast_gray,
        max_contrast_gray=mean_contrast_gray * 1.5,
        integrated_contrast=area_px * mean_contrast_gray,
        border_distance_px=0.0 if touches_roi_border else 15.0,
        touches_roi_border=touches_roi_border,
        evidence=ev,
    )


def test_closed_form_counterfactual_targets_and_impossibility_proofs() -> None:
    cfg = default_config()
    # 1) 면적 96px (<= 133px) + 대비 17.8: 면적 병목으로 대비 포화(33.0)에서도 NG 도달 불능
    c_bottleneck = _comp(0, area_px=96, mean_contrast_gray=17.8)
    res = replace(
        make_result(
            Status.REVIEW, reasons=["EVIDENCE_IN_REVIEW_BAND"], score=c_bottleneck.evidence
        ),
        sample_id="rev_area_bottleneck",
        components=(c_bottleneck,),
    )
    rep = explain_counterfactual(res, cfg)
    assert rep.machine_status is Status.REVIEW
    assert rep.pivotal_component_indices == (0,)
    attr = rep.components[0]
    assert attr.is_pivotal is True
    assert attr.status_without is Status.OK
    assert attr.score_without == 0.0
    # 대비 고정 시 target_area_for_ng 역산 검증: target 에서는 >= 0.50, target-1 에서는 < 0.50
    assert attr.target_area_for_ng is not None
    assert attr.delta_area_for_ng == attr.target_area_for_ng - 96
    assert _component_evidence(attr.target_area_for_ng, 17.8, cfg) >= cfg.scoring.ng_min
    assert _component_evidence(attr.target_area_for_ng - 1, 17.8, cfg) < cfg.scoring.ng_min
    # 면적 고정 시 대비 상승만으로는 수학적 도달 불능(None)
    assert attr.target_contrast_for_ng is None
    assert attr.delta_contrast_for_ng is None
    assert "도달할 수 없습니다" in attr.explanation_ko

    # 2) 면적 200px (> 133px) + 대비 18.0: 면적/대비 양쪽 모두 닫힌 형태 경계 존재
    c_reachable = _comp(0, area_px=200, mean_contrast_gray=18.0)
    res_reach = replace(
        make_result(Status.REVIEW, reasons=["EVIDENCE_IN_REVIEW_BAND"], score=c_reachable.evidence),
        sample_id="rev_reachable",
        components=(c_reachable,),
    )
    rep_reach = explain_counterfactual(res_reach, cfg)
    a_r = rep_reach.components[0]
    assert a_r.target_area_for_ng is not None
    assert _component_evidence(a_r.target_area_for_ng, 18.0, cfg) >= cfg.scoring.ng_min
    assert _component_evidence(a_r.target_area_for_ng - 1, 18.0, cfg) < cfg.scoring.ng_min
    assert a_r.target_contrast_for_ng is not None
    assert a_r.delta_contrast_for_ng == pytest.approx(a_r.target_contrast_for_ng - 18.0, abs=1e-6)
    assert _component_evidence(200, a_r.target_contrast_for_ng, cfg) >= cfg.scoring.ng_min
    assert _component_evidence(200, a_r.target_contrast_for_ng - 1e-4, cfg) < cfg.scoring.ng_min

    # 3) 무효 영역 (area=30 < 40, contrast=12 < 16)
    c_invalid = _comp(0, area_px=30, mean_contrast_gray=12.0)
    res_inv = replace(
        make_result(Status.OK, score=0.0), sample_id="ok_inv", components=(c_invalid,)
    )
    rep_inv = explain_counterfactual(res_inv, cfg)
    a_inv = rep_inv.components[0]
    assert a_inv.is_valid is False
    assert a_inv.min_area_delta_to_valid == 10
    assert a_inv.min_contrast_delta_to_valid == 4.0
    assert a_inv.target_area_for_ng is None
    assert a_inv.target_contrast_for_ng is None


def test_leave_one_out_pivotal_and_determinism() -> None:
    cfg = default_config()
    c_ng0 = _comp(0, area_px=420, mean_contrast_gray=24.5)
    c_weak1 = _comp(1, area_px=90, mean_contrast_gray=18.0)
    c_ng1 = _comp(1, area_px=500, mean_contrast_gray=26.0)

    # NG 1개 + 약후보 1개 -> #0 은 NG -> REVIEW 로 바꾸는 결정적 후보
    res_mix = replace(
        make_result(Status.NG, reasons=["DEFECT_EVIDENCE_STRONG"], score=c_ng0.evidence),
        sample_id="ng_mix",
        components=(c_ng0, c_weak1),
    )
    rep_mix = explain_counterfactual(res_mix, cfg)
    assert rep_mix.pivotal_component_indices == (0,)
    assert rep_mix.components[0].is_pivotal is True
    assert rep_mix.components[0].status_without is Status.REVIEW
    assert rep_mix.components[1].is_pivotal is False
    assert rep_mix.components[1].status_without is Status.NG

    # NG 2개 -> 어느 하나를 빼도 여전히 NG 이므로 pivotal 없음
    res_dual = replace(
        make_result(Status.NG, reasons=["DEFECT_EVIDENCE_STRONG"], score=c_ng1.evidence),
        sample_id="ng_dual",
        components=(c_ng0, c_ng1),
    )
    rep_dual = explain_counterfactual(res_dual, cfg)
    assert rep_dual.pivotal_component_indices == ()
    assert all(not c.is_pivotal for c in rep_dual.components)

    # 위상도 데이터 및 해시 결정론 + JSON 직렬화 검증
    rep_mix_2 = explain_counterfactual(res_mix, cfg)
    assert rep_mix.counterfactual_hash == rep_mix_2.counterfactual_hash
    payload = rep_mix.to_dict()
    assert payload["schema_version"] == COUNTERFACTUAL_SCHEMA_VERSION
    assert payload["phase_diagram_svg_data"]["ng_polyline"]
    assert len(payload["phase_diagram_svg_data"]["points"]) == 2
    json.dumps(payload, allow_nan=False)

    # ERROR / 0 컴포넌트 OK 안전 처리
    err_res = make_result(Status.ERROR, error_code="E_INPUT_SIZE", score=0.0)
    err_rep = explain_counterfactual(err_res, cfg)
    assert err_rep.components == ()
    assert "ERROR" in err_rep.headline_ko
    ok_res = make_result(Status.OK, reasons=[ReasonCode.NO_DEFECT_EVIDENCE.value], score=0.0)
    ok_rep = explain_counterfactual(ok_res, cfg)
    assert ok_rep.margin_to_ng == pytest.approx(0.50)
    assert ok_rep.machine_status is Status.OK
