"""Phase 4 사람-기계 피드백 불일치 분석 & 능동 하드샘플 큐레이션 (observe-only).

작업자가 남긴 검토 기록(``review_decisions.jsonl`` 또는 SQLite ``review_decisions``)을
에이전트 우선순위(``P0``~``P3``) 및 사유 코드와 대조해 엔지니어링 피드백 리포트를
생성한다.

계약:
* 자동 재학습·설정 변경·임계값 조정을 절대 수행하지 않는다(100% observe-only).
* 모든 비율은 ``evaluation.metrics.ratio`` 로 분모와 Wilson 95% 구간을 포함한다.
* 외부 네트워크·셸·디스크 쓰기를 사용하지 않는다.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..config import REVIEW_BAND_MAX
from ..contracts import (
    ComponentFeature,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Stage,
    Status,
)
from ..evaluation.metrics import ratio
from ..records.fingerprint import canonical_json_sha256
from .review import assess_review_triage
from .stability import analyze_weak_geometry

CALIBRATION_SCHEMA_VERSION = "agent-calibration-v1"
_PRIORITY_BANDS: tuple[str, ...] = ("P0", "P1", "P2", "P3")


class HardExampleKind(StrEnum):
    """사람 검토와 에이전트 우선순위 간 불일치·난해 하드 샘플 유형."""

    BLINDSPOT_DEFECT = "BLINDSPOT_DEFECT"
    HIGH_PRIO_FALSE_ALARM = "HIGH_PRIO_FALSE_ALARM"
    INSUFFICIENT_EVIDENCE_HOLD = "INSUFFICIENT_EVIDENCE_HOLD"


_KIND_ORDER: dict[str, int] = {
    HardExampleKind.BLINDSPOT_DEFECT.value: 0,
    HardExampleKind.HIGH_PRIO_FALSE_ALARM.value: 1,
    HardExampleKind.INSUFFICIENT_EVIDENCE_HOLD.value: 2,
}


@dataclass(frozen=True)
class CalibrationSummary:
    """Phase 4 사람-기계 피드백 캘리브레이션 및 하드 샘플 큐레이션 보고서."""

    total_review_samples: int
    decided_count: int
    undecided_count: int
    confirmed_ng_ratio: dict[str, Any]
    false_alarm_ok_ratio: dict[str, Any]
    by_priority: dict[str, dict[str, Any]]
    by_reason_code: dict[str, int]
    hard_examples: tuple[dict[str, Any], ...]
    recommendations_ko: tuple[str, ...]
    calibration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "total_review_samples": int(self.total_review_samples),
            "decided_count": int(self.decided_count),
            "undecided_count": int(self.undecided_count),
            "confirmed_ng_ratio": self.confirmed_ng_ratio,
            "false_alarm_ok_ratio": self.false_alarm_ok_ratio,
            "by_priority": self.by_priority,
            "by_reason_code": self.by_reason_code,
            "hard_examples": list(self.hard_examples),
            "recommendations_ko": list(self.recommendations_ko),
            "calibration_hash": self.calibration_hash,
        }

def _record_to_sample_result(item: SampleResult | Mapping[str, Any]) -> SampleResult:
    if isinstance(item, SampleResult):
        return item
    raw_comps = item.get("components") or []
    components = tuple(
        ComponentFeature(
            index=int(c.get("index", idx)),
            area_px=int(c.get("area_px", 0)),
            bbox=tuple(int(x) for x in c.get("bbox", (0, 0, 0, 0))),
            area_ratio=float(c.get("area_ratio", 0.0)),
            aspect_ratio=float(c.get("aspect_ratio", 1.0)),
            elongation=float(c.get("elongation", 1.0)),
            solidity=float(c.get("solidity", 1.0)),
            mean_contrast_gray=float(c.get("mean_contrast_gray", 0.0)),
            max_contrast_gray=float(c.get("max_contrast_gray", 0.0)),
            integrated_contrast=float(c.get("integrated_contrast", 0.0)),
            border_distance_px=float(c.get("border_distance_px", 100.0)),
            touches_roi_border=bool(c.get("touches_roi_border", False)),
            evidence=float(c.get("evidence", 0.0)),
        )
        for idx, c in enumerate(raw_comps)
        if isinstance(c, Mapping)
    )
    q = item.get("quality")
    quality = None
    if isinstance(q, Mapping):
        q_flags: list[ReasonCode] = []
        for val in q.get("flags") or []:
            try:
                q_flags.append(ReasonCode(val))
            except ValueError:
                continue
        quality = QualityInfo(
            blur_variance=float(q.get("blur_variance", 0.0)),
            mean_gray=float(q.get("mean_gray", 0.0)),
            saturated_ratio=float(q.get("saturated_ratio", 0.0)),
            flags=tuple(q_flags),
        )
    r = item.get("roi")
    roi = (
        RoiInfo(
            mode=str(r.get("mode", "auto")),
            bbox=tuple(int(x) for x in r.get("bbox", (0, 0, 0, 0))),
            area_px=int(r.get("area_px", 0)),
            area_ratio=float(r.get("area_ratio", 0.0)),
            uncertain=bool(r.get("uncertain", False)),
        )
        if isinstance(r, Mapping)
        else None
    )
    reasons: list[ReasonCode] = []
    for val in item.get("reasons") or []:
        try:
            reasons.append(ReasonCode(val))
        except ValueError:
            continue
    error_code = None
    if item.get("error_code"):
        try:
            error_code = ErrorCode(item["error_code"])
        except ValueError:
            error_code = None
    error_stage = None
    if item.get("error_stage"):
        try:
            error_stage = Stage(item["error_stage"])
        except ValueError:
            error_stage = None
    return SampleResult(
        run_id=str(item.get("run_id", "")),
        sample_id=str(item.get("sample_id", "")),
        original_id=str(item.get("original_id", item.get("sample_id", ""))),
        split=str(item.get("split", "tune")),
        label=str(item.get("label", "unknown")),
        status=Status(str(item.get("status", "REVIEW"))),
        score=float(item.get("score", 0.0)),
        reasons=tuple(reasons),
        components=components,
        quality=quality,
        roi=roi,
        timing_ms={str(k): float(v) for k, v in (item.get("timing_ms") or {}).items()},
        error_code=error_code,
        error_stage=error_stage,
        input_sha256=str(item.get("input_sha256", "")),
        mask_sha256=str(item.get("mask_sha256", "")),
        overlay_path=str(item.get("overlay_path", "")),
        config_hash=str(item.get("config_hash", "")),
    )


def _normalize_decision(dec: Any) -> tuple[str, str, str, str]:
    if isinstance(dec, Mapping):
        sid = str(dec.get("sample_id", "")).strip()
        human_status = str(dec.get("final_status") or dec.get("decision") or "").strip()
        reason_code = str(dec.get("reason_code") or dec.get("reason") or "OTHER").strip()
        note = str(dec.get("note") or "").strip()
        return sid, human_status, reason_code, note
    sid = str(getattr(dec, "sample_id", "")).strip()
    human_status = str(getattr(dec, "final_status", getattr(dec, "decision", ""))).strip()
    reason_code = str(getattr(dec, "reason_code", getattr(dec, "reason", "OTHER"))).strip()
    note = str(getattr(dec, "note", "")).strip()
    return sid, human_status, reason_code, note

def _build_recommendations_ko(
    *,
    total_review: int,
    decided_count: int,
    by_priority: Mapping[str, Mapping[str, Any]],
    hard_examples: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if total_review == 0:
        return ("REVIEW 대상 샘플이 없어 피드백 캘리브레이션 대상이 비어 있습니다.",)
    if decided_count == 0:
        return (
            "아직 기록된 작업자 검토 결정이 없습니다. 검토 큐에서 결정을 남기면 "
            "우선순위 정합성과 하드 샘플 큐가 자동 집계됩니다.",
        )

    recs: list[str] = []
    blindspots = [h for h in hard_examples if h["kind"] == HardExampleKind.BLINDSPOT_DEFECT.value]
    false_alarms = [
        h for h in hard_examples if h["kind"] == HardExampleKind.HIGH_PRIO_FALSE_ALARM.value
    ]
    holds = [
        h for h in hard_examples if h["kind"] == HardExampleKind.INSUFFICIENT_EVIDENCE_HOLD.value
    ]

    if blindspots:
        ids_txt = ", ".join(str(h["sample_id"]) for h in blindspots[:3])
        recs.append(
            f"[우선순위 사각지대] P2/P3 저위험으로 분류됐으나 사람 확정 NG가 {len(blindspots)}건 "
            f"발생했습니다({ids_txt}). 저대비·미세 면적 후보의 가중치 점검을 권장합니다."
        )
    if false_alarms:
        ids_txt = ", ".join(str(h["sample_id"]) for h in false_alarms[:3])
        recs.append(
            f"[고위험 오경보] P0 최우선으로 분류됐으나 사람 기각(OK)이 {len(false_alarms)}건 "
            f"발생했습니다({ids_txt}). 표면 반사광·결 노이즈 패턴 분리를 권장합니다."
        )
    if holds:
        ids_txt = ", ".join(str(h["sample_id"]) for h in holds[:3])
        recs.append(
            f"[증거 부족 보류] 작업자 보류/증거 부족 결정이 {len(holds)}건 있습니다({ids_txt}). "
            "교차 엔진(YOLO) 재촬영·패치 중재 점검을 권장합니다."
        )

    p0_rate = by_priority["P0"]["confirmed_ng_ratio"].get("value")
    p2_rate = by_priority["P2"]["confirmed_ng_ratio"].get("value")
    p3_rate = by_priority["P3"]["confirmed_ng_ratio"].get("value")
    low_rates = [v for v in (p2_rate, p3_rate) if v is not None]
    if p0_rate is not None and low_rates:
        if p0_rate >= max(low_rates):
            recs.append(
                f"[정합성 양호] P0 사람 확정 NG 비율({p0_rate:.1%})이 P2/P3({max(low_rates):.1%}) "
                "이상으로 에이전트 우선순위 정렬이 작업자 판단과 정합적입니다."
            )
        else:
            recs.append(
                f"[정합성 역전 주의] P0 확정 NG 비율({p0_rate:.1%})이 "
                f"P2/P3({max(low_rates):.1%})보다 "
                "낮아 우선순위 신호 재검토가 필요합니다."
            )

    if not recs:
        recs.append(
            f"검토 완료 {decided_count}건에서 특이 사각지대·오경보 불일치가 발견되지 않았습니다."
        )
    recs.append(
        "모든 권고는 엔지니어 참고용(observe-only)이며 "
        "기계 판정 임계값과 설정을 자동 변경하지 않습니다."
    )
    return tuple(recs)

def analyze_human_feedback(
    records: Sequence[SampleResult | Mapping[str, Any]],
    decisions: Sequence[Any],
    *,
    ng_min: float = 0.50,
    review_band_max: float = REVIEW_BAND_MAX,
) -> CalibrationSummary:
    """REVIEW 큐의 에이전트 우선순위(P0~P3)와 작업자 검토 결정을 대조 분석한다."""

    review_parsed: list[SampleResult] = []
    for rec in records:
        parsed = _record_to_sample_result(rec)
        if parsed.status is Status.REVIEW:
            review_parsed.append(parsed)

    triage_by_id: dict[str, tuple[str, float]] = {}
    score_by_id: dict[str, float] = {}
    for parsed in review_parsed:
        geom = analyze_weak_geometry(parsed.components, parsed.roi)
        triage = assess_review_triage(
            parsed,
            geometry=geom,
            ng_min=ng_min,
            review_band_max=review_band_max,
        )
        triage_by_id[parsed.sample_id] = (triage.priority.value, float(triage.priority_score))
        score_by_id[parsed.sample_id] = float(parsed.score)

    latest_by_id: dict[str, tuple[str, str, str]] = {}
    for dec in decisions:
        sid, human_status, reason_code, note = _normalize_decision(dec)
        if not sid or sid not in triage_by_id:
            continue
        latest_by_id[sid] = (human_status, reason_code, note)

    total_review = len(review_parsed)
    decided_count = len(latest_by_id)
    undecided_count = total_review - decided_count

    confirmed_ng_total = sum(1 for hs, _, _ in latest_by_id.values() if hs == Status.NG.value)
    false_alarm_total = sum(1 for hs, _, _ in latest_by_id.values() if hs == Status.OK.value)

    confirmed_ng_ratio = ratio(
        "검토 결정 중 사람 확정 NG 비율",
        confirmed_ng_total,
        decided_count,
    ).as_dict()
    false_alarm_ok_ratio = ratio(
        "검토 결정 중 사람 기각 OK(오탐) 비율",
        false_alarm_total,
        decided_count,
    ).as_dict()

    reason_counter: Counter[str] = Counter()
    for _, reason_code, _ in latest_by_id.values():
        reason_counter[reason_code] += 1
    by_reason_code = dict(sorted(reason_counter.items()))

    by_priority: dict[str, dict[str, Any]] = {}
    for band in _PRIORITY_BANDS:
        band_ids = [sid for sid, (prio, _) in triage_by_id.items() if prio == band]
        band_decided_ids = [sid for sid in band_ids if sid in latest_by_id]
        band_ng = sum(
            1 for sid in band_decided_ids if latest_by_id[sid][0] == Status.NG.value
        )
        band_ok = sum(
            1 for sid in band_decided_ids if latest_by_id[sid][0] == Status.OK.value
        )
        band_hold = sum(
            1 for sid in band_decided_ids if latest_by_id[sid][0] == Status.REVIEW.value
        )
        by_priority[band] = {
            "review_count": len(band_ids),
            "decided_count": len(band_decided_ids),
            "confirmed_ng_count": band_ng,
            "false_alarm_ok_count": band_ok,
            "hold_count": band_hold,
            "confirmed_ng_ratio": ratio(
                f"{band} 결정 중 사람 확정 NG 비율",
                band_ng,
                len(band_decided_ids),
            ).as_dict(),
        }

    hard_list: list[dict[str, Any]] = []
    for sid, (human_status, reason_code, note) in latest_by_id.items():
        prio, _ = triage_by_id[sid]
        score = score_by_id[sid]
        kind: HardExampleKind | None = None
        if prio in {"P2", "P3"} and human_status == Status.NG.value:
            kind = HardExampleKind.BLINDSPOT_DEFECT
        elif prio == "P0" and human_status == Status.OK.value:
            kind = HardExampleKind.HIGH_PRIO_FALSE_ALARM
        elif human_status == Status.REVIEW.value or reason_code == "INSUFFICIENT_EVIDENCE":
            kind = HardExampleKind.INSUFFICIENT_EVIDENCE_HOLD
        if kind is not None:
            hard_list.append(
                {
                    "sample_id": sid,
                    "kind": kind.value,
                    "priority": prio,
                    "score": round(score, 6),
                    "human_status": human_status,
                    "reason_code": reason_code,
                    "note": note,
                }
            )

    hard_list.sort(
        key=lambda item: (
            _KIND_ORDER.get(item["kind"], 99),
            -float(item["score"]),
            str(item["sample_id"]),
        )
    )
    frozen_hard = tuple(hard_list)
    recs = _build_recommendations_ko(
        total_review=total_review,
        decided_count=decided_count,
        by_priority=by_priority,
        hard_examples=frozen_hard,
    )

    hash_payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "total_review_samples": total_review,
        "decided_count": decided_count,
        "undecided_count": undecided_count,
        "confirmed_ng_ratio": confirmed_ng_ratio,
        "false_alarm_ok_ratio": false_alarm_ok_ratio,
        "by_priority": by_priority,
        "by_reason_code": by_reason_code,
        "hard_examples": list(frozen_hard),
        "recommendations_ko": list(recs),
    }
    return CalibrationSummary(
        total_review_samples=total_review,
        decided_count=decided_count,
        undecided_count=undecided_count,
        confirmed_ng_ratio=confirmed_ng_ratio,
        false_alarm_ok_ratio=false_alarm_ok_ratio,
        by_priority=by_priority,
        by_reason_code=by_reason_code,
        hard_examples=frozen_hard,
        recommendations_ko=recs,
        calibration_hash=canonical_json_sha256(hash_payload),
    )


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationSummary",
    "HardExampleKind",
    "analyze_human_feedback",
]
