"""결함 판정 경계까지의 최소 변화량과 컴포넌트별 기여도(LOO)를 닫힌 식으로 계산하는 해석 모듈.

``pipeline/stages.py`` 의 실제 점수 공식(``_component_evidence``, ``calculate_score``,
``decide_status``)을 그대로 역산하여, 딥러닝 블랙박스가 줄 수 없는 화이트박스
수식 기반 설명력과 수학적 도달 불능 증명을 제공한다.

계약:
* 원본 기계 판정(``SampleResult.status``)과 설정은 절대 변경하지 않는다.
* 외부 네트워크·셸·디스크 쓰기를 사용하지 않는다.
* 동일 입력·설정에서 100% 결정론적인 해시와 수치를 산출한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import REVIEW_BAND_MAX, AppConfig, default_config
from ..contracts import QualityInfo, ReasonCode, SampleResult, Status
from ..pipeline.stages import (
    _component_evidence,
    calculate_score,
    decide_status,
)
from ..pipeline.types import RefineResult, RoiResult
from ..records.fingerprint import canonical_json_sha256

COUNTERFACTUAL_SCHEMA_VERSION = "counterfactual-v1"

#: 2D (면적 × 평균대비) 등위선 위상도 SVG 좌표계 상수.
PHASE_VIEW_BOX = "0 0 600 280"
PHASE_PLOT_X0 = 54.0
PHASE_PLOT_X1 = 576.0
PHASE_PLOT_Y0 = 18.0
PHASE_PLOT_Y1 = 236.0
PHASE_X_MAX = 900.0
PHASE_Y_MAX = 36.0
_SATURATION_CAP = 1.5

@dataclass(frozen=True)
class ComponentAttribution:
    """단일 후보 컴포넌트의 Leave-One-Out 귀속도 및 닫힌 형태 반사실 경계 해."""

    index: int
    area_px: int
    mean_contrast_gray: float
    evidence: float
    is_valid: bool
    score_without: float
    delta_score: float
    status_without: Status
    is_pivotal: bool
    min_area_delta_to_valid: int
    min_contrast_delta_to_valid: float
    target_area_for_ng: int | None
    delta_area_for_ng: int | None
    target_contrast_for_ng: float | None
    delta_contrast_for_ng: float | None
    explanation_ko: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "area_px": int(self.area_px),
            "mean_contrast_gray": round(float(self.mean_contrast_gray), 6),
            "evidence": round(float(self.evidence), 6),
            "is_valid": bool(self.is_valid),
            "score_without": round(float(self.score_without), 6),
            "delta_score": round(float(self.delta_score), 6),
            "status_without": self.status_without.value,
            "is_pivotal": bool(self.is_pivotal),
            "min_area_delta_to_valid": int(self.min_area_delta_to_valid),
            "min_contrast_delta_to_valid": round(float(self.min_contrast_delta_to_valid), 6),
            "target_area_for_ng": self.target_area_for_ng,
            "delta_area_for_ng": self.delta_area_for_ng,
            "target_contrast_for_ng": (
                round(float(self.target_contrast_for_ng), 6)
                if self.target_contrast_for_ng is not None
                else None
            ),
            "delta_contrast_for_ng": (
                round(float(self.delta_contrast_for_ng), 6)
                if self.delta_contrast_for_ng is not None
                else None
            ),
            "explanation_ko": self.explanation_ko,
        }


@dataclass(frozen=True)
class CounterfactualReport:
    """샘플 전체의 반사실적 경계 해석 및 2D 위상도 데이터(observe-only)."""

    sample_id: str
    machine_status: Status
    score: float
    margin_to_ng: float
    margin_to_review_band: float
    pivotal_component_indices: tuple[int, ...]
    components: tuple[ComponentAttribution, ...]
    headline_ko: str
    phase_diagram_svg_data: dict[str, Any]
    counterfactual_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "machine_status": self.machine_status.value,
            "score": round(float(self.score), 6),
            "margin_to_ng": round(float(self.margin_to_ng), 6),
            "margin_to_review_band": round(float(self.margin_to_review_band), 6),
            "pivotal_component_indices": list(self.pivotal_component_indices),
            "components": [comp.to_dict() for comp in self.components],
            "headline_ko": self.headline_ko,
            "phase_diagram_svg_data": self.phase_diagram_svg_data,
            "counterfactual_hash": self.counterfactual_hash,
        }


def _solve_target_area_for_ng(
    area_px: int,
    mean_contrast_gray: float,
    cfg: AppConfig,
) -> tuple[int | None, int | None]:
    """현재 대비를 고정할 때 ``evidence >= ng_min`` 및 유효성을 만족하는 최소 정수 면적."""

    min_area_px = int(cfg.candidates.min_area_px)
    min_contrast_gray = float(cfg.candidates.min_contrast_gray)
    ng_area_px = float(cfg.scoring.ng_area_px)
    ng_contrast_gray = float(cfg.scoring.ng_contrast_gray)
    ng_min = float(cfg.scoring.ng_min)

    if mean_contrast_gray < min_contrast_gray or mean_contrast_gray <= 0.0:
        return None, None

    contrast_norm = min(mean_contrast_gray / ng_contrast_gray, _SATURATION_CAP)
    if contrast_norm <= 0.0:
        return None, None

    max_evidence = min(math.sqrt(_SATURATION_CAP * contrast_norm), _SATURATION_CAP)
    if max_evidence + 1e-12 < ng_min:
        return None, None

    req_area_norm = (ng_min * ng_min) / contrast_norm
    if req_area_norm > _SATURATION_CAP + 1e-12:
        return None, None

    raw_area = req_area_norm * ng_area_px
    cand = max(min_area_px, math.ceil(raw_area))
    while cand > min_area_px and _component_evidence(cand - 1, mean_contrast_gray, cfg) >= ng_min:
        cand -= 1
    max_sat_area = math.ceil(_SATURATION_CAP * ng_area_px) + 1
    while _component_evidence(cand, mean_contrast_gray, cfg) < ng_min and cand <= max_sat_area:
        cand += 1

    if _component_evidence(cand, mean_contrast_gray, cfg) < ng_min:
        return None, None
    return int(cand), int(cand - area_px)


def _solve_target_contrast_for_ng(
    area_px: int,
    mean_contrast_gray: float,
    cfg: AppConfig,
) -> tuple[float | None, float | None]:
    """현재 면적을 고정할 때 ``evidence >= ng_min`` 및 유효성을 만족하는 최소 평균 대비."""

    min_area_px = int(cfg.candidates.min_area_px)
    min_contrast_gray = float(cfg.candidates.min_contrast_gray)
    ng_area_px = float(cfg.scoring.ng_area_px)
    ng_contrast_gray = float(cfg.scoring.ng_contrast_gray)
    ng_min = float(cfg.scoring.ng_min)

    if area_px < min_area_px or area_px <= 0:
        return None, None

    area_norm = min(float(area_px) / ng_area_px, _SATURATION_CAP)
    if area_norm <= 0.0:
        return None, None

    max_evidence = min(math.sqrt(area_norm * _SATURATION_CAP), _SATURATION_CAP)
    if max_evidence + 1e-12 < ng_min:
        return None, None

    req_contrast_norm = (ng_min * ng_min) / area_norm
    if req_contrast_norm > _SATURATION_CAP + 1e-12:
        return None, None

    raw_contrast = max(min_contrast_gray, req_contrast_norm * ng_contrast_gray)
    target_c = round(raw_contrast, 6)
    if _component_evidence(area_px, target_c, cfg) < ng_min:
        target_c = round(raw_contrast + 1e-6, 6)
        while _component_evidence(area_px, target_c, cfg) < ng_min:
            target_c = math.nextafter(target_c, math.inf)

    if _component_evidence(area_px, target_c, cfg) < ng_min:
        return None, None
    return float(target_c), round(float(target_c) - float(mean_contrast_gray), 6)

def _component_explanation_ko(
    *,
    index: int,
    area_px: int,
    mean_contrast_gray: float,
    evidence: float,
    is_valid: bool,
    status_without: Status,
    is_pivotal: bool,
    target_area_for_ng: int | None,
    delta_area_for_ng: int | None,
    target_contrast_for_ng: float | None,
    delta_contrast_for_ng: float | None,
    cfg: AppConfig,
) -> str:
    min_area_px = int(cfg.candidates.min_area_px)
    min_contrast_gray = float(cfg.candidates.min_contrast_gray)
    ng_contrast_gray = float(cfg.scoring.ng_contrast_gray)
    ng_area_px = float(cfg.scoring.ng_area_px)
    ng_min = float(cfg.scoring.ng_min)
    sat_contrast = _SATURATION_CAP * ng_contrast_gray

    if is_valid and evidence >= ng_min:
        margin_ev = evidence - ng_min
        if is_pivotal:
            tail = f"이 후보 하나만 제거하면 판정이 {status_without.value}로 낮아집니다."
        else:
            tail = f"이 후보만 제거해도 다른 근거로 판정은 {status_without.value}로 유지됩니다."
        return (
            f"#{index} ({area_px}px, 대비 {mean_contrast_gray:.1f}): "
            f"NG 임계값({ng_min:.2f})을 +{margin_ev:.4f} 초과했습니다. {tail}"
        )

    if is_pivotal:
        pivotal_txt = f"단독 제거 시 {status_without.value}로 전환되는 결정적 후보입니다."
    else:
        pivotal_txt = f"단독 제거 시 판정은 {status_without.value}로 유지됩니다."

    if target_area_for_ng is not None and delta_area_for_ng is not None:
        if delta_area_for_ng > 0:
            area_txt = (
                f"현재 대비 유지 시 면적이 +{delta_area_for_ng}px"
                f"(총 {target_area_for_ng}px) 커지면 NG가 됩니다"
            )
        else:
            area_txt = f"현재 대비에서 NG 면적 경계는 {target_area_for_ng}px입니다"
    else:
        area_txt = (
            f"현재 대비({mean_contrast_gray:.1f})가 유효 하한({min_contrast_gray:.1f}) 미만이거나 "
            "포화 상한 제한으로 면적 확장만으로는 NG에 도달할 수 없습니다"
        )

    if target_contrast_for_ng is not None and delta_contrast_for_ng is not None:
        if delta_contrast_for_ng > 0:
            contrast_txt = (
                f"현재 면적 유지 시 대비가 +{delta_contrast_for_ng:.2f}"
                f"(총 {target_contrast_for_ng:.2f})에 도달하면 NG가 됩니다."
            )
        else:
            contrast_txt = f"현재 면적에서 NG 대비 경계는 {target_contrast_for_ng:.2f}입니다."
    elif area_px < min_area_px:
        contrast_txt = (
            f"현재 면적({area_px}px)이 최소 유효 면적({min_area_px}px) 미만이어서 "
            "단독 대비 상승만으로는 NG에 도달할 수 없습니다."
        )
    else:
        area_norm = min(float(area_px) / ng_area_px, _SATURATION_CAP)
        max_ev_c = min(math.sqrt(area_norm * _SATURATION_CAP), _SATURATION_CAP)
        contrast_txt = (
            f"현재 면적({area_px}px)만으로는 대비 포화 상한({sat_contrast:.1f})에서도 "
            f"최대 증거가 {max_ev_c:.4f}에 그쳐 단독 대비 상승만으로는 NG에 도달할 수 없습니다."
        )

    return (
        f"#{index} ({area_px}px, 대비 {mean_contrast_gray:.1f}): "
        f"{pivotal_txt} {area_txt}, {contrast_txt}"
    )


def _phase_px(area: float) -> float:
    clamped = min(max(float(area), 0.0), PHASE_X_MAX)
    return round(PHASE_PLOT_X0 + (clamped / PHASE_X_MAX) * (PHASE_PLOT_X1 - PHASE_PLOT_X0), 2)


def _phase_py(contrast: float) -> float:
    clamped = min(max(float(contrast), 0.0), PHASE_Y_MAX)
    return round(PHASE_PLOT_Y1 - (clamped / PHASE_Y_MAX) * (PHASE_PLOT_Y1 - PHASE_PLOT_Y0), 2)


def _iso_curve_points(
    threshold: float,
    cfg: AppConfig,
    *,
    min_area: float,
    min_contrast: float,
) -> list[tuple[float, float]]:
    """등위 증거 곡선(``evidence == threshold``)의 (area_px, contrast) 절점 목록을 만든다."""

    ng_area = float(cfg.scoring.ng_area_px)
    ng_contrast = float(cfg.scoring.ng_contrast_gray)
    sat_contrast = _SATURATION_CAP * ng_contrast
    sat_area = _SATURATION_CAP * ng_area
    k = (threshold * threshold) * ng_area * ng_contrast

    a_sat = max(min_area, k / sat_contrast)
    a_floor = min(PHASE_X_MAX, k / max(min_contrast, 1e-6))

    pts: list[tuple[float, float]] = []
    if a_sat <= PHASE_X_MAX:
        if sat_contrast < PHASE_Y_MAX:
            pts.append((a_sat, PHASE_Y_MAX))
        pts.append((a_sat, min(PHASE_Y_MAX, sat_contrast)))

    curve_end = min(PHASE_X_MAX, a_floor, sat_area)
    if curve_end > a_sat + 1e-6:
        steps = 28
        for step in range(1, steps + 1):
            a = a_sat + (curve_end - a_sat) * (step / float(steps))
            c = max(min_contrast, k / max(a, 1e-6))
            if c <= PHASE_Y_MAX:
                pts.append((a, c))

    if a_floor < PHASE_X_MAX:
        pts.append((PHASE_X_MAX, min_contrast))
    elif pts and pts[-1][0] < PHASE_X_MAX:
        a = PHASE_X_MAX
        c = max(min_contrast, k / min(a, sat_area))
        if c <= PHASE_Y_MAX:
            pts.append((a, c))
    return pts

def _build_phase_diagram_svg_data(
    attributions: tuple[ComponentAttribution, ...],
    cfg: AppConfig,
) -> dict[str, Any]:
    min_area = float(cfg.candidates.min_area_px)
    min_contrast = float(cfg.candidates.min_contrast_gray)
    ng_min = float(cfg.scoring.ng_min)
    sat_contrast = _SATURATION_CAP * float(cfg.scoring.ng_contrast_gray)
    sat_area_for_ng = (ng_min * ng_min) * float(cfg.scoring.ng_area_px) / _SATURATION_CAP

    ng_pts = _iso_curve_points(ng_min, cfg, min_area=min_area, min_contrast=min_contrast)
    ng_polyline = " ".join(f"{_phase_px(a)},{_phase_py(c)}" for a, c in ng_pts)
    ng_poly_pts = list(ng_pts) + [(PHASE_X_MAX, PHASE_Y_MAX)]
    if ng_pts:
        ng_poly_pts.append((ng_pts[0][0], PHASE_Y_MAX))
    ng_polygon = " ".join(f"{_phase_px(a)},{_phase_py(c)}" for a, c in ng_poly_pts)

    rev_pts = _iso_curve_points(REVIEW_BAND_MAX, cfg, min_area=min_area, min_contrast=min_contrast)
    review_polyline = " ".join(f"{_phase_px(a)},{_phase_py(c)}" for a, c in rev_pts)

    component_dots = [
        {
            "index": attr.index,
            "area_px": attr.area_px,
            "mean_contrast_gray": round(attr.mean_contrast_gray, 2),
            "evidence": round(attr.evidence, 4),
            "x": _phase_px(attr.area_px),
            "y": _phase_py(attr.mean_contrast_gray),
            "clamped": (attr.area_px > PHASE_X_MAX or attr.mean_contrast_gray > PHASE_Y_MAX),
            "is_pivotal": attr.is_pivotal,
            "is_valid": attr.is_valid,
            "is_ng": bool(attr.is_valid and attr.evidence >= ng_min),
        }
        for attr in attributions
    ]

    x_ticks = [
        {"value": v, "x": _phase_px(v)}
        for v in (0, int(min_area), round(sat_area_for_ng), 275, 450, 700, int(PHASE_X_MAX))
    ]
    y_ticks = [
        {"value": v, "y": _phase_py(v)}
        for v in (
            0,
            int(min_contrast),
            int(cfg.scoring.ng_contrast_gray),
            int(sat_contrast),
            int(PHASE_Y_MAX),
        )
    ]

    return {
        "view_box": PHASE_VIEW_BOX,
        "plot": {
            "x0": PHASE_PLOT_X0,
            "x1": PHASE_PLOT_X1,
            "y0": PHASE_PLOT_Y0,
            "y1": PHASE_PLOT_Y1,
            "x_max": PHASE_X_MAX,
            "y_max": PHASE_Y_MAX,
        },
        "min_area_px": int(min_area),
        "min_area_x": _phase_px(min_area),
        "min_contrast_gray": min_contrast,
        "min_contrast_y": _phase_py(min_contrast),
        "sat_area_for_ng": round(sat_area_for_ng, 2),
        "sat_area_x": _phase_px(sat_area_for_ng),
        "sat_contrast_gray": sat_contrast,
        "sat_contrast_y": _phase_py(sat_contrast),
        "ng_polyline": ng_polyline,
        "ng_polygon": ng_polygon,
        "review_polyline": review_polyline,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "points": component_dots,
    }

def _leave_one_out_status(
    result: SampleResult,
    remaining_pos_excluded: int,
    cfg: AppConfig,
) -> tuple[float, Status]:
    if result.status is Status.ERROR:
        return float(result.score), Status.ERROR

    remaining = tuple(
        comp for idx, comp in enumerate(result.components) if idx != remaining_pos_excluded
    )
    scored_wo = calculate_score(remaining, cfg)
    quality_obj = (
        result.quality
        if result.quality is not None
        else QualityInfo(blur_variance=100.0, mean_gray=120.0, saturated_ratio=0.0, flags=())
    )
    roi_uncertain = (
        bool(result.roi.uncertain)
        if result.roi is not None
        else (ReasonCode.ROI_UNCERTAIN in result.reasons)
    )
    roi_obj = RoiResult(
        mode=result.roi.mode if result.roi is not None else "auto",
        mask=np.zeros((1, 1), dtype=np.uint8),
        bbox=result.roi.bbox if result.roi is not None else (0, 0, 1, 1),
        area_px=result.roi.area_px if result.roi is not None else 1,
        area_ratio=result.roi.area_ratio if result.roi is not None else 1.0,
        uncertain=roi_uncertain,
    )
    truncated = ReasonCode.CANDIDATE_TRUNCATED in result.reasons
    refined_obj = RefineResult(
        components=(),
        combined_mask=np.zeros((1, 1), dtype=np.uint8),
        truncated=truncated,
        total_components=len(remaining),
    )
    decision_wo = decide_status(scored_wo, quality_obj, roi_obj, refined_obj, cfg)
    return float(scored_wo.score), decision_wo.status


def _build_headline_ko(
    result: SampleResult,
    attributions: tuple[ComponentAttribution, ...],
    pivotal_indices: tuple[int, ...],
    margin_to_ng: float,
    cfg: AppConfig,
) -> str:
    ng_min = float(cfg.scoring.ng_min)
    if result.status is Status.ERROR:
        code = result.error_code.value if result.error_code else "-"
        return (
            f"오류(ERROR: {code}) 상태는 제품 판정이 아니므로 반사실적 경계 해석 대상이 아닙니다."
        )
    if not attributions:
        if result.status is Status.OK:
            return (
                f"유효 후보 컴포넌트가 0개여서 점수 {result.score:.4f}로 OK 판정되었습니다"
                f"(NG 임계값 {ng_min:.2f}까지 여유 {margin_to_ng:+.4f})."
            )
        reasons = ", ".join(r.value for r in result.reasons) or "-"
        return f"후보 컴포넌트는 없으나 사유({reasons})에 의해 REVIEW 보류되었습니다."

    top = max(attributions, key=lambda item: (item.evidence, -item.index))
    pivotal_labels = ", ".join("#" + str(i) for i in pivotal_indices)
    pivotal_text = (
        f"결정적(Pivotal) 후보: {pivotal_labels}"
        if pivotal_indices
        else "단일 제거로 판정이 바뀌는 결정적 후보 없음"
    )
    if result.status is Status.NG:
        return (
            f"NG 확정(점수 {result.score:.4f}, 임계값 초과 {-margin_to_ng:+.4f}) · "
            f"주원인 #{top.index}({top.area_px}px, 대비 {top.mean_contrast_gray:.1f}) · "
            f"{pivotal_text}."
        )
    if top.target_contrast_for_ng is None and top.target_area_for_ng is not None:
        bottleneck = (
            f"대표 후보 #{top.index}({top.area_px}px)은 면적 병목으로 대비 단독 상승만으로는 "
            f"NG 도달 불가(대비 고정 시 면적 {top.delta_area_for_ng:+d}px 필요)"
        )
    elif top.target_area_for_ng is not None and top.target_contrast_for_ng is not None:
        bottleneck = (
            f"대표 후보 #{top.index}은 경계 전환 가능"
            f"(면적 {top.delta_area_for_ng:+d}px 또는 대비 {top.delta_contrast_for_ng:+.2f} 시 NG)"
        )
    else:
        bottleneck = f"대표 후보 #{top.index}은 유효 조건 미달로 단일 축 상승만으로는 NG 도달 불가"
    return (
        f"{result.status.value}(점수 {result.score:.4f}, NG까지 {margin_to_ng:+.4f}) · "
        f"{pivotal_text} · {bottleneck}."
    )

def explain_counterfactual(
    result: SampleResult,
    config: AppConfig | None = None,
) -> CounterfactualReport:
    """``SampleResult`` 에 대한 닫힌 형태 반사실적 경계 해석과 Leave-One-Out 귀속도를 산출한다."""

    cfg = config if config is not None else default_config()
    min_area_px = int(cfg.candidates.min_area_px)
    min_contrast_gray = float(cfg.candidates.min_contrast_gray)
    ng_min = float(cfg.scoring.ng_min)
    base_score = float(result.score)

    attributions: list[ComponentAttribution] = []
    for pos, comp in enumerate(result.components):
        area_px = int(comp.area_px)
        contrast = float(comp.mean_contrast_gray)
        analytic_ev = (
            _component_evidence(area_px, contrast, cfg)
            if (area_px > 0 and contrast > 0)
            else 0.0
        )
        is_valid = (area_px >= min_area_px) and (contrast >= min_contrast_gray)

        score_wo, status_wo = _leave_one_out_status(result, pos, cfg)
        delta_score = round(base_score - score_wo, 6)
        is_pivotal = (result.status is not Status.ERROR) and (status_wo is not result.status)

        min_area_delta = max(0, min_area_px - area_px)
        min_contrast_delta = round(max(0.0, min_contrast_gray - contrast), 6)

        target_area, delta_area = _solve_target_area_for_ng(area_px, contrast, cfg)
        target_contrast, delta_contrast = _solve_target_contrast_for_ng(area_px, contrast, cfg)

        explanation_ko = _component_explanation_ko(
            index=int(comp.index),
            area_px=area_px,
            mean_contrast_gray=contrast,
            evidence=analytic_ev,
            is_valid=is_valid,
            status_without=status_wo,
            is_pivotal=is_pivotal,
            target_area_for_ng=target_area,
            delta_area_for_ng=delta_area,
            target_contrast_for_ng=target_contrast,
            delta_contrast_for_ng=delta_contrast,
            cfg=cfg,
        )
        attributions.append(
            ComponentAttribution(
                index=int(comp.index),
                area_px=area_px,
                mean_contrast_gray=contrast,
                evidence=analytic_ev,
                is_valid=is_valid,
                score_without=round(score_wo, 6),
                delta_score=delta_score,
                status_without=status_wo,
                is_pivotal=is_pivotal,
                min_area_delta_to_valid=min_area_delta,
                min_contrast_delta_to_valid=min_contrast_delta,
                target_area_for_ng=target_area,
                delta_area_for_ng=delta_area,
                target_contrast_for_ng=target_contrast,
                delta_contrast_for_ng=delta_contrast,
                explanation_ko=explanation_ko,
            )
        )

    frozen_attrs = tuple(attributions)
    pivotal_indices = tuple(attr.index for attr in frozen_attrs if attr.is_pivotal)
    margin_to_ng = round(ng_min - base_score, 6)
    margin_to_review_band = round(REVIEW_BAND_MAX - base_score, 6)
    headline_ko = _build_headline_ko(
        result, frozen_attrs, pivotal_indices, margin_to_ng, cfg
    )
    phase_svg = _build_phase_diagram_svg_data(frozen_attrs, cfg)

    hash_payload = {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "config_hash": cfg.config_hash,
        "sample_id": result.sample_id,
        "machine_status": result.status.value,
        "score": round(base_score, 6),
        "margin_to_ng": margin_to_ng,
        "margin_to_review_band": margin_to_review_band,
        "pivotal_component_indices": list(pivotal_indices),
        "components": [attr.to_dict() for attr in frozen_attrs],
    }
    return CounterfactualReport(
        sample_id=result.sample_id,
        machine_status=result.status,
        score=round(base_score, 6),
        margin_to_ng=margin_to_ng,
        margin_to_review_band=margin_to_review_band,
        pivotal_component_indices=pivotal_indices,
        components=frozen_attrs,
        headline_ko=headline_ko,
        phase_diagram_svg_data=phase_svg,
        counterfactual_hash=canonical_json_sha256(hash_payload),
    )


__all__ = [
    "COUNTERFACTUAL_SCHEMA_VERSION",
    "ComponentAttribution",
    "CounterfactualReport",
    "explain_counterfactual",
]
