"""REVIEW 큐를 결함 위험도순으로 정렬하고 3단 근거 카드·Recall@Top-K를 계산하는 트리아지 모듈.

계약:
* 제품 판정(``SampleResult.status``)은 절대 변경하지 않는다(``machine_status`` 불변).
* 모든 점수와 정렬은 결정론적이며 동일 입력에서 동일한 해시를 만든다.
* 모든 비율·리프트 지표는 ``evaluation.metrics.ratio`` 로 분모와 Wilson 95% 구간을
  함께 보고한다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..contracts import ReasonCode, SampleResult, Status
from ..evaluation.metrics import ratio
from ..records.fingerprint import canonical_json_sha256
from .evidence import CrossEvidenceAssessment, EvidenceFlag
from .stability import (
    ArbitrationVerdict,
    PatchArbitration,
    StabilityLevel,
    StabilityReport,
    WeakGeometryReport,
    WeakPattern,
    analyze_weak_geometry,
)

TRIAGE_POLICY_VERSION = "triage-v1"
TRIAGE_SCHEMA_VERSION = "review-triage-v1"


class ReviewPriority(StrEnum):
    """작업자 검토 우선순위 등급(판정이 아니라 큐 정렬용)."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


@dataclass(frozen=True)
class ArgumentationCard:
    """3단 논증 영수증(가설 → 검증 실험 → 정량 결론 → 작업자 지침)."""

    hypothesis: str
    experiment: str
    verdict: str
    operator_guidance: str

    def to_dict(self) -> dict[str, str]:
        return {
            "hypothesis": self.hypothesis,
            "experiment": self.experiment,
            "verdict": self.verdict,
            "operator_guidance": self.operator_guidance,
        }


@dataclass(frozen=True)
class TriageAssessment:
    """REVIEW 샘플(또는 임의 샘플)의 결정론적 우선순위 평가 결과."""

    sample_id: str
    machine_status: Status
    priority: ReviewPriority
    priority_score: float
    stability: StabilityLevel | None
    weak_pattern: WeakPattern
    evidence_flags: tuple[str, ...]
    arbitration_verdict: str | None
    argumentation: ArgumentationCard
    signals: dict[str, float]
    policy_version: str = TRIAGE_POLICY_VERSION
    triage_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRIAGE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "machine_status": self.machine_status.value,
            "priority": self.priority.value,
            "priority_score": round(float(self.priority_score), 6),
            "stability": self.stability.value if self.stability is not None else None,
            "weak_pattern": self.weak_pattern.value,
            "evidence_flags": list(self.evidence_flags),
            "arbitration_verdict": self.arbitration_verdict,
            "argumentation": self.argumentation.to_dict(),
            "signals": {k: round(float(v), 6) for k, v in sorted(self.signals.items())},
            "policy_version": self.policy_version,
            "triage_hash": self.triage_hash,
        }


def _priority_band(score: float) -> ReviewPriority:
    if score >= 0.70:
        return ReviewPriority.P0
    if score >= 0.50:
        return ReviewPriority.P1
    if score >= 0.30:
        return ReviewPriority.P2
    return ReviewPriority.P3


def _build_argumentation(
    result: SampleResult,
    *,
    priority: ReviewPriority,
    priority_score: float,
    geom: WeakGeometryReport,
    stability: StabilityReport | None,
    cross_assessment: CrossEvidenceAssessment | None,
    arbitration: PatchArbitration | None,
    max_contrast: float,
) -> ArgumentationCard:
    reason_texts = [r.value for r in result.reasons]
    flag_texts = [f.value for f in cross_assessment.flags] if cross_assessment else []

    if geom.pattern is WeakPattern.BORDER_ARTIFACT_SUSPECT:
        hypothesis = (
            f"후보 {geom.component_count}개 중 테두리 근접 비율이 {geom.border_touch_ratio:.0%}로 "
            "ROI 경계 반사 아티팩트 가능성이 있습니다."
        )
    elif geom.pattern is WeakPattern.COLLINEAR_SCRATCH_SUSPECT:
        hypothesis = (
            f"후보 {geom.component_count}개가 직선 배열(R²={geom.collinearity_r2:.2f})을 이루어 "
            "끊어진 미세 스크래치 결함 가능성이 있습니다."
        )
    elif geom.pattern is WeakPattern.LOCAL_CLUSTER:
        hypothesis = (
            f"후보 {geom.component_count}개가 국소 영역에 밀집"
            f"(밀집도 {geom.cluster_compactness:.2f})되어 군집 결함 가능성이 있습니다."
        )
    elif ReasonCode.EVIDENCE_IN_REVIEW_BAND in result.reasons:
        hypothesis = (
            f"점수({result.score:.3f})가 검토 경계 구간에 위치하며 국소 대비가 "
            f"{max_contrast:.1f}로 실제 결함 여부를 확인해야 합니다."
        )
    elif geom.pattern is WeakPattern.DIFFUSE_SURFACE_TEXTURE:
        hypothesis = (
            f"후보 {geom.component_count}개가 넓게 분산되어 전역 표면 결/노이즈 가능성이 있습니다."
        )
    else:
        hypothesis = (
            f"사유({', '.join(reason_texts) or '없음'}) 및 교차 플래그"
            f"({', '.join(flag_texts) or '없음'})에 따라 보수적 검토가 필요합니다."
        )

    experiments = [f"약후보 기하 분석(패턴={geom.pattern.value}, N={geom.component_count})"]
    if stability is not None:
        experiments.append(
            f"결정론 섭동 프로브(밝기+5%/블러3x3, 유지율={stability.mean_score_retention:.0%})"
        )
    if cross_assessment is not None:
        experiments.append(
            f"교차 증거 비교({cross_assessment.engine_a} vs {cross_assessment.engine_b})"
        )
    if arbitration is not None:
        experiments.append(f"능동 패치 중재({arbitration.verdict.value})")
    experiment = " · ".join(experiments)

    parts = [
        f"우선순위 점수 {priority_score:.3f}({priority.value})",
        f"기계 점수 {result.score:.3f}",
        f"최대 대비 {max_contrast:.1f}",
    ]
    if stability is not None:
        parts.append(
            f"안정성 {stability.level.value}(유지율 {stability.mean_score_retention:.0%}, "
            f"상태보존 {stability.status_preserved_count}/2)"
        )
    if arbitration is not None:
        parts.append(f"패치 중재 {arbitration.verdict.value}")
    verdict = ", ".join(parts) + f". 원본 기계 판정({result.status.value})은 변경하지 않습니다."

    if priority is ReviewPriority.P0:
        prefix = "P0 최우선 검토"
    elif priority is ReviewPriority.P1:
        prefix = "P1 우선 검토"
    elif priority is ReviewPriority.P2:
        prefix = "P2 일반 검토"
    else:
        prefix = "P3 후순위 검토"

    detail_bits: list[str] = []
    if geom.pattern is WeakPattern.COLLINEAR_SCRATCH_SUSPECT:
        detail_bits.append(
            f"후보 {geom.component_count}개 직선 배열(R²={geom.collinearity_r2:.2f})"
        )
    elif geom.pattern is WeakPattern.BORDER_ARTIFACT_SUSPECT:
        detail_bits.append(f"테두리 접촉 비율 {geom.border_touch_ratio:.0%}")
    else:
        detail_bits.append(f"최대 대비 {max_contrast:.1f}·점수 {result.score:.3f}")
    if stability is not None:
        detail_bits.append(
            f"섭동 후 신호 {stability.mean_score_retention:.0%} 유지({stability.level.value})"
        )
    if arbitration is not None:
        detail_bits.append(arbitration.summary)
    guidance = f"{prefix}: " + ", ".join(detail_bits)

    return ArgumentationCard(
        hypothesis=hypothesis,
        experiment=experiment,
        verdict=verdict,
        operator_guidance=guidance,
    )


def assess_review_triage(
    result: SampleResult,
    *,
    geometry: WeakGeometryReport | None = None,
    stability: StabilityReport | None = None,
    cross_assessment: CrossEvidenceAssessment | None = None,
    arbitration: PatchArbitration | None = None,
    ng_min: float = 0.50,
    review_band_max: float = 0.40,
) -> TriageAssessment:
    """물리적 결함 신호·기하·섭동 안정성·교차 증거를 종합해 검토 우선순위를 산출한다."""

    geom = (
        geometry if geometry is not None else analyze_weak_geometry(result.components, result.roi)
    )

    max_contrast = max((float(c.max_contrast_gray) for c in result.components), default=0.0)
    max_mean_contrast = max((float(c.mean_contrast_gray) for c in result.components), default=0.0)
    max_integrated = max((float(c.integrated_contrast) for c in result.components), default=0.0)
    max_area = max((int(c.area_px) for c in result.components), default=0)

    score_proximity = min(max(float(result.score) / max(float(ng_min), 1e-6), 0.0), 1.0)
    peak_contrast_norm = min(max((max_contrast - 22.0) / 58.0, 0.0), 1.0)
    mean_contrast_norm = min(max((max_mean_contrast - 16.0) / 24.0, 0.0), 1.0)
    integrated_norm = min(max(max_integrated / 5000.0, 0.0), 1.0)
    area_norm = min(max(float(max_area) / 250.0, 0.0), 1.0)
    retention_norm = (
        min(max(float(stability.mean_score_retention), 0.0), 1.0) if stability is not None else 0.5
    )

    raw_score = (
        0.26 * score_proximity
        + 0.24 * peak_contrast_norm
        + 0.12 * mean_contrast_norm
        + 0.10 * integrated_norm
        + 0.04 * area_norm
    )

    if (
        ReasonCode.EVIDENCE_IN_REVIEW_BAND in result.reasons
        or float(result.score) >= review_band_max
    ):
        raw_score += 0.05

    if geom.pattern is WeakPattern.COLLINEAR_SCRATCH_SUSPECT:
        raw_score += 0.08
    elif geom.pattern is WeakPattern.LOCAL_CLUSTER:
        raw_score += 0.05
    elif geom.pattern is WeakPattern.SINGLE_CANDIDATE:
        raw_score += 0.04
    elif geom.pattern is WeakPattern.DIFFUSE_SURFACE_TEXTURE:
        raw_score -= 0.04
    elif geom.pattern is WeakPattern.BORDER_ARTIFACT_SUSPECT:
        raw_score -= 0.22
    elif geom.pattern is WeakPattern.NONE:
        raw_score -= 0.10

    if geom.pattern is not WeakPattern.BORDER_ARTIFACT_SUSPECT and geom.border_touch_ratio > 0.0:
        raw_score -= 0.08 * float(geom.border_touch_ratio)
    if ReasonCode.BORDER_TOUCHING_CANDIDATE in result.reasons:
        raw_score -= 0.08

    if stability is not None:
        if stability.level is StabilityLevel.STABLE:
            raw_score += 0.04 + 0.10 * retention_norm
        elif stability.level is StabilityLevel.SENSITIVE:
            raw_score += 0.05 * retention_norm
        else:
            raw_score -= 0.14

    if cross_assessment is not None:
        flags_set = set(cross_assessment.flags)
        if EvidenceFlag.NG_SINGLE_ENGINE in flags_set:
            raw_score += 0.14
        if EvidenceFlag.STATUS_MISMATCH in flags_set:
            raw_score += 0.06
        if EvidenceFlag.EVIDENCE_ALIGNED in flags_set and cross_assessment.matches:
            raw_score += 0.08
        elif EvidenceFlag.LOCALIZATION_MISMATCH in flags_set:
            raw_score += 0.02

    if arbitration is not None:
        if arbitration.verdict in {ArbitrationVerdict.BOTH_SALIENT, ArbitrationVerdict.ALIGNED}:
            raw_score += 0.08
        elif arbitration.verdict in {
            ArbitrationVerdict.ENGINE_A_SALIENT,
            ArbitrationVerdict.ENGINE_B_SALIENT,
        }:
            raw_score += 0.04
        elif arbitration.verdict is ArbitrationVerdict.BORDER_NOISE_SUSPECT:
            raw_score -= 0.15

    priority_score = round(min(max(raw_score, 0.0), 1.0), 6)
    priority = _priority_band(priority_score)
    evidence_flags = (
        tuple(flag.value for flag in cross_assessment.flags)
        if cross_assessment is not None
        else ()
    )
    arbitration_verdict = arbitration.verdict.value if arbitration is not None else None
    signals = {
        "score_proximity": round(score_proximity, 6),
        "peak_contrast_norm": round(peak_contrast_norm, 6),
        "mean_contrast_norm": round(mean_contrast_norm, 6),
        "integrated_norm": round(integrated_norm, 6),
        "area_norm": round(area_norm, 6),
        "retention_norm": round(retention_norm, 6),
        "collinearity_r2": round(float(geom.collinearity_r2), 6),
        "border_touch_ratio": round(float(geom.border_touch_ratio), 6),
    }
    argumentation = _build_argumentation(
        result,
        priority=priority,
        priority_score=priority_score,
        geom=geom,
        stability=stability,
        cross_assessment=cross_assessment,
        arbitration=arbitration,
        max_contrast=max_contrast,
    )
    hash_payload = {
        "schema_version": TRIAGE_SCHEMA_VERSION,
        "policy_version": TRIAGE_POLICY_VERSION,
        "sample_id": result.sample_id,
        "machine_status": result.status.value,
        "priority": priority.value,
        "priority_score": priority_score,
        "stability": stability.level.value if stability is not None else None,
        "weak_pattern": geom.pattern.value,
        "evidence_flags": list(evidence_flags),
        "arbitration_verdict": arbitration_verdict,
        "signals": signals,
        "argumentation": argumentation.to_dict(),
    }
    return TriageAssessment(
        sample_id=result.sample_id,
        machine_status=result.status,
        priority=priority,
        priority_score=priority_score,
        stability=stability.level if stability is not None else None,
        weak_pattern=geom.pattern,
        evidence_flags=evidence_flags,
        arbitration_verdict=arbitration_verdict,
        argumentation=argumentation,
        signals=signals,
        policy_version=TRIAGE_POLICY_VERSION,
        triage_hash=canonical_json_sha256(hash_payload),
    )


def rank_review_queue(assessments: Sequence[TriageAssessment]) -> list[TriageAssessment]:
    """``(-priority_score, sample_id)`` 기준으로 검토 큐를 결정론 정렬한다."""

    return sorted(assessments, key=lambda item: (-float(item.priority_score), item.sample_id))


def evaluate_triage_lift(
    ranked_sample_ids: Sequence[str],
    labels_by_id: Mapping[str, str],
    *,
    top_fractions: Sequence[float] = (0.10, 0.20, 0.30),
) -> dict[str, Any]:
    """상위 K% 검토 구간에서의 결함 조기 회수율(Recall@Top-K)과 리프트를 계산한다."""

    total = len(ranked_sample_ids)
    total_defects = sum(1 for sid in ranked_sample_ids if labels_by_id.get(sid) == "defect")

    top_k: dict[str, dict[str, Any]] = {}
    for frac in top_fractions:
        if not 0.0 < float(frac) <= 1.0:
            raise ValueError(f"top_fractions 원소는 (0, 1] 범위여야 합니다: {frac!r}")
        if total == 0:
            k = 0
            captured = 0
            expected = 0.0
        else:
            k = min(total, max(1, math.ceil(total * float(frac))))
            captured = sum(
                1 for sid in ranked_sample_ids[:k] if labels_by_id.get(sid) == "defect"
            )
            expected = k / float(total)

        recall_ratio = ratio(
            f"상위 {float(frac):.0%}({k}/{total}) 내 결함 포착 / 전체 REVIEW 결함",
            captured,
            total_defects,
        )
        precision_ratio = ratio(
            f"상위 {float(frac):.0%}({k}/{total}) 내 결함 비율(PPV)",
            captured,
            k,
        )
        lift = (
            round(float(recall_ratio.value) / expected, 6)
            if (recall_ratio.value is not None and expected > 0.0)
            else None
        )
        top_k[f"{float(frac):.2f}"] = {
            "fraction": float(frac),
            "k": k,
            "captured_defects": captured,
            "recall_at_k": recall_ratio.as_dict(),
            "precision_at_k": precision_ratio.as_dict(),
            "random_expected_recall": round(expected, 6),
            "lift": lift,
        }

    return {
        "total_review": total,
        "total_defects": total_defects,
        "baseline_defect_rate": ratio(
            "REVIEW 내 전체 결함 비율", total_defects, total
        ).as_dict(),
        "top_k": top_k,
    }



__all__ = [
    "TRIAGE_POLICY_VERSION",
    "TRIAGE_SCHEMA_VERSION",
    "ArgumentationCard",
    "ReviewPriority",
    "TriageAssessment",
    "assess_review_triage",
    "evaluate_triage_lift",
    "rank_review_queue",
]
