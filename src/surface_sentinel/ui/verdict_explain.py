"""검사 결과 JSON을 "이 사진의 값 vs 기준" 표로 풀어 상세 화면에 보여주는 설명 계층.

판정을 다시 내리지 않는다. 저장된 상태·사유·후보 수치를 동결 설정의 기준값과 나란히 놓아
왜 OK/NG/REVIEW 인지 숫자로 읽히게만 만든다(읽기 전용, 결정론).
"""

from __future__ import annotations

import math
from typing import Any

from ..config import REVIEW_BAND_MAX, AppConfig


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def explain_verdict(result: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """저장된 결과에서 판정 규칙별 값·기준·충족 여부와 후보 계산표를 만든다."""

    cand = config.candidates
    scoring = config.scoring
    quality_cfg = config.quality
    weak_low = max(REVIEW_BAND_MAX - scoring.multi_candidate_margin, 0.0)

    status = str(result.get("status") or "")
    reasons = [str(r) for r in (result.get("reasons") or [])]
    score = _num(result.get("score"))
    comps_raw = result.get("components") or []

    candidates: list[dict[str, Any]] = []
    for comp in comps_raw if isinstance(comps_raw, list) else []:
        area = int(_num(comp.get("area_px")))
        contrast = _num(comp.get("mean_contrast_gray"))
        area_ok = area >= cand.min_area_px
        contrast_ok = contrast >= cand.min_contrast_gray
        area_norm = min(area / scoring.ng_area_px, 1.5)
        contrast_norm = min(contrast / scoring.ng_contrast_gray, 1.5)
        candidates.append({
            "area": area,
            "contrast": contrast,
            "area_norm": area_norm,
            "contrast_norm": contrast_norm,
            "area_capped": area / scoring.ng_area_px > 1.5,
            "contrast_capped": contrast / scoring.ng_contrast_gray > 1.5,
            "evidence": _num(comp.get("evidence")),
            "valid": area_ok and contrast_ok,
            "why_invalid": (
                "" if area_ok and contrast_ok
                else ("면적 미달" if not area_ok else "대비 미달")
            ),
            "border": bool(comp.get("touches_roi_border")),
        })
    candidates.sort(key=lambda c: c["evidence"], reverse=True)
    valid = [c for c in candidates if c["valid"]]
    weak = [c for c in valid if weak_low <= c["evidence"] < scoring.ng_min]
    top = valid[0] if valid else None

    q = result.get("quality") or {}
    q_flags = [str(f) for f in (q.get("flags") or [])]
    roi = result.get("roi") or {}

    rules: list[dict[str, Any]] = [
        {
            "name": "NG 규칙 · 최고 점수",
            "value": f"{score:.3f}",
            "threshold": f"≥ {scoring.ng_min:.2f}",
            "hit": status == "NG" or (bool(valid) and score >= scoring.ng_min),
            "effect": "NG (불량)",
        },
        {
            "name": (
                f"유효 후보 수 (면적 ≥ {cand.min_area_px}px · 대비 ≥ {cand.min_contrast_gray:g})"
            ),
            "value": f"{len(valid)}개",
            "threshold": "0개면 OK 후보",
            "hit": not valid,
            "effect": "OK (정상)",
        },
        {
            "name": f"약한 후보 ({weak_low:.2f} ≤ 점수 < {scoring.ng_min:.2f})",
            "value": f"{len(weak)}개",
            "threshold": "≥ 2개",
            "hit": "MULTIPLE_WEAK_CANDIDATES" in reasons,
            "effect": "REVIEW",
        },
        {
            "name": f"테두리 접촉 후보 (점수 ≥ {REVIEW_BAND_MAX:.2f})",
            "value": "있음" if any(c["border"] for c in valid) else "없음",
            "threshold": "있음",
            "hit": "BORDER_TOUCHING_CANDIDATE" in reasons,
            "effect": "REVIEW",
        },
        {
            "name": f"후보 개수 상한 ({cand.max_candidates}개)",
            "value": f"{len(candidates)}개 계측",
            "threshold": f"> {cand.max_candidates}개 잘림",
            "hit": "CANDIDATE_TRUNCATED" in reasons,
            "effect": "REVIEW",
        },
        {
            "name": "사진 품질 (흐림·밝기·포화)",
            "value": (
                f"선명도 {_num(q.get('blur_variance')):.0f}"
                f" · 평균밝기 {_num(q.get('mean_gray')):.0f}"
                f" · 포화 {_num(q.get('saturated_ratio')) * 100:.1f}%"
            ),
            "threshold": (
                f"선명도 ≥ {quality_cfg.blur_min_var:g}"
                f" · 밝기 {quality_cfg.dark_mean_min:g}~{quality_cfg.bright_mean_max:g}"
                f" · 포화 ≤ {quality_cfg.saturated_ratio_max * 100:.0f}%"
            ),
            "hit": "INPUT_QUALITY_LOW" in reasons or bool(q_flags),
            "effect": "REVIEW",
        },
        {
            "name": "검사 영역(ROI) 확보",
            "value": f"{_num(roi.get('area_ratio')) * 100:.0f}%",
            "threshold": f"≥ {config.roi.min_area_ratio * 100:.0f}%",
            "hit": "ROI_UNCERTAIN" in reasons or bool(roi.get("uncertain")),
            "effect": "REVIEW",
        },
    ]

    # 한 줄 결론 (숫자 포함)
    if status == "NG" and top:
        headline = (
            f"최고 점수 {score:.3f} ≥ 기준 {scoring.ng_min:.2f} → 불량(NG). "
            f"가장 강한 후보: 면적 {top['area']}px · 대비 {top['contrast']:.1f}"
        )
    elif status == "OK":
        biggest = candidates[0] if candidates else None
        headline = "면적·대비 기준을 동시에 넘는 후보가 0개 → 정상(OK)"
        if biggest:
            headline += (
                f". 가장 강한 흔적도 면적 {biggest['area']}px · 대비 {biggest['contrast']:.1f}로"
                f" {biggest['why_invalid'] or '기준 미달'}"
            )
    elif status == "REVIEW":
        gap = max(scoring.ng_min - score, 0.0)
        if "MULTIPLE_WEAK_CANDIDATES" in reasons:
            headline = (
                f"점수 {weak_low:.2f}~{scoring.ng_min:.2f} 사이 약한 후보가 {len(weak)}개"
                "(기준 2개 이상) → 사람 확인. "
                f"최고 점수 {score:.3f} → NG 기준까지 {gap:.3f} 부족"
            )
        elif "BORDER_TOUCHING_CANDIDATE" in reasons:
            headline = (
                f"점수 {score:.3f} 후보가 검사 영역 테두리에 닿음"
                " → 치구·가장자리일 수 있어 사람 확인"
            )
        elif "CANDIDATE_TRUNCATED" in reasons:
            headline = f"후보가 상한 {cand.max_candidates}개를 넘어 일부가 잘림 → 사람 확인"
        elif "INPUT_QUALITY_LOW" in reasons:
            headline = f"사진 품질 경고({', '.join(q_flags) or '품질 저하'}) → 사람 확인"
        elif "ROI_UNCERTAIN" in reasons:
            headline = "검사 영역을 확실히 잡지 못함 → 사람 확인"
        else:
            headline = (
                f"결함 후보는 있으나 최고 점수 {score:.3f} < NG 기준 {scoring.ng_min:.2f}"
                f" (부족 {gap:.3f}) → 사람 확인"
            )
    else:
        headline = f"검사 오류 ({result.get('error_code') or '원인 미기록'}) — 판정 대상이 아닙니다"

    return {
        "headline": headline,
        "rules": rules,
        "candidates": candidates[:5],
        "candidate_total": len(candidates),
        "top": top,
        "formula": {
            "ng_area": scoring.ng_area_px,
            "ng_contrast": scoring.ng_contrast_gray,
            "ng_min": scoring.ng_min,
        },
    }


__all__ = ["explain_verdict"]
