"""Phase 4 사람-기계 피드백 캘리브레이션(agent.calibration) 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from surface_sentinel.agent.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    HardExampleKind,
    analyze_human_feedback,
)
from surface_sentinel.config import default_config
from surface_sentinel.review.store import ReviewStore


def _review_record(
    sample_id: str, *, score: float, area: int, max_c: float, mean_c: float, border: bool = False
) -> dict:
    return {
        "sample_id": sample_id,
        "status": "REVIEW",
        "score": score,
        "reasons": ["EVIDENCE_IN_REVIEW_BAND"],
        "components": [
            {
                "index": 0,
                "area_px": area,
                "bbox": [1 if border else 40, 80, 14, 14],
                "area_ratio": 0.01,
                "aspect_ratio": 1.0,
                "elongation": 1.0,
                "solidity": 1.0,
                "mean_contrast_gray": mean_c,
                "max_contrast_gray": max_c,
                "integrated_contrast": float(area * mean_c),
                "border_distance_px": 1.0 if border else 15.0,
                "touches_roi_border": border,
                "evidence": score,
            }
        ],
    }


def test_calibration_empty_decisions_defense() -> None:
    records = [
        _review_record("s_p0", score=0.49, area=320, max_c=95.0, mean_c=35.0),
        _review_record("s_p3", score=0.25, area=42, max_c=20.0, mean_c=17.0, border=True),
        {"sample_id": "s_ok", "status": "OK", "score": 0.0, "components": []},
    ]
    summary = analyze_human_feedback(records, [])
    assert summary.total_review_samples == 2
    assert summary.decided_count == 0
    assert summary.undecided_count == 2
    assert summary.confirmed_ng_ratio["denominator"] == 0
    assert summary.confirmed_ng_ratio["value"] is None
    assert summary.false_alarm_ok_ratio["denominator"] == 0
    assert summary.hard_examples == ()
    assert "아직 기록된 작업자 검토 결정이 없습니다" in summary.recommendations_ko[0]
    payload = summary.to_dict()
    assert payload["schema_version"] == CALIBRATION_SCHEMA_VERSION
    json.dumps(payload, allow_nan=False)


def test_calibration_extracts_hard_examples_ratios_and_latest_decision(tmp_path: Path) -> None:
    records = [
        _review_record("p0_ng", score=0.49, area=320, max_c=95.0, mean_c=35.0),
        _review_record("p0_ok_alarm", score=0.49, area=320, max_c=95.0, mean_c=35.0),
        _review_record("p3_blindspot", score=0.25, area=42, max_c=20.0, mean_c=17.0, border=True),
        _review_record("p3_hold", score=0.26, area=45, max_c=22.0, mean_c=17.5, border=True),
    ]
    cfg_hash = default_config().config_hash
    store = ReviewStore(tmp_path)
    # p0_ng: 처음엔 OK 였다가 마지막 결정이 NG -> 마지막 결정 기준이 반영되어야 한다
    store.append(
        sample_id="p0_ng",
        machine_status="REVIEW",
        final_status="OK",
        reason_code="FALSE_ALARM",
        reviewer="qa",
        config_hash=cfg_hash,
        overlay_sha256="a" * 64,
    )
    store.append(
        sample_id="p0_ng",
        machine_status="REVIEW",
        final_status="NG",
        reason_code="CONFIRMED_DEFECT",
        reviewer="qa",
        config_hash=cfg_hash,
        overlay_sha256="a" * 64,
    )
    store.append(
        sample_id="p0_ok_alarm",
        machine_status="REVIEW",
        final_status="OK",
        reason_code="FALSE_ALARM",
        reviewer="qa",
        config_hash=cfg_hash,
        overlay_sha256="b" * 64,
    )
    store.append(
        sample_id="p3_blindspot",
        machine_status="REVIEW",
        final_status="NG",
        reason_code="CONFIRMED_DEFECT",
        reviewer="qa",
        config_hash=cfg_hash,
        overlay_sha256="c" * 64,
    )
    store.append(
        sample_id="p3_hold",
        machine_status="REVIEW",
        final_status="REVIEW",
        reason_code="INSUFFICIENT_EVIDENCE",
        reviewer="qa",
        config_hash=cfg_hash,
        overlay_sha256="d" * 64,
    )

    summary = analyze_human_feedback(records, store.read_all())
    assert summary.total_review_samples == 4
    assert summary.decided_count == 4
    assert summary.undecided_count == 0
    assert summary.confirmed_ng_ratio["numerator"] == 2
    assert summary.confirmed_ng_ratio["denominator"] == 4
    assert summary.confirmed_ng_ratio["ci95_low"] is not None
    assert summary.false_alarm_ok_ratio["numerator"] == 1
    assert summary.false_alarm_ok_ratio["denominator"] == 4

    by_id = {h["sample_id"]: h for h in summary.hard_examples}
    assert by_id["p3_blindspot"]["kind"] == HardExampleKind.BLINDSPOT_DEFECT.value
    assert by_id["p0_ok_alarm"]["kind"] == HardExampleKind.HIGH_PRIO_FALSE_ALARM.value
    assert by_id["p3_hold"]["kind"] == HardExampleKind.INSUFFICIENT_EVIDENCE_HOLD.value
    assert "p0_ng" not in by_id

    # P0 / P3 비율 분모 확인
    assert summary.by_priority["P0"]["decided_count"] == 2
    assert summary.by_priority["P0"]["confirmed_ng_ratio"]["numerator"] == 1
    assert summary.by_priority["P0"]["confirmed_ng_ratio"]["denominator"] == 2
    assert summary.by_priority["P3"]["decided_count"] == 2
    assert summary.by_priority["P3"]["confirmed_ng_ratio"]["numerator"] == 1

    # SQLite dict 형태 입력도 동일 해시 산출 검증
    sqlite_style = [
        {"sample_id": "p0_ng", "decision": "NG", "reason": "CONFIRMED_DEFECT"},
        {"sample_id": "p0_ok_alarm", "decision": "OK", "reason": "FALSE_ALARM"},
        {"sample_id": "p3_blindspot", "decision": "NG", "reason": "CONFIRMED_DEFECT"},
        {"sample_id": "p3_hold", "decision": "REVIEW", "reason": "INSUFFICIENT_EVIDENCE"},
    ]
    summary2 = analyze_human_feedback(records, sqlite_style)
    assert summary2.calibration_hash == summary.calibration_hash
