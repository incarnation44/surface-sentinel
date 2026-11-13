"""Phase 3 PSI 분포 드리프트 감시(agent.drift) 단위 테스트."""

from __future__ import annotations

import json

from agent_fakes import make_result
from surface_sentinel.agent.drift import (
    DriftPolicy,
    DriftStatus,
    assess_drift,
    extract_observations,
)
from surface_sentinel.contracts import QualityInfo, Status


def _make_records(
    count: int,
    *,
    score: float = 0.02,
    mean_gray: float = 40.0,
    blur_variance: float = 250.0,
    status: str = "OK",
) -> list[dict]:
    return [
        {
            "status": status,
            "score": score,
            "quality": {
                "mean_gray": mean_gray,
                "blur_variance": blur_variance,
                "saturated_ratio": 0.0,
                "flags": [],
            },
        }
        for _ in range(count)
    ]


def test_drift_identical_distribution_has_zero_psi_and_normal() -> None:
    baseline = _make_records(30, score=0.02, mean_gray=40.0, blur_variance=250.0)
    target = _make_records(30, score=0.02, mean_gray=40.0, blur_variance=250.0)
    report = assess_drift(baseline, target)
    assert report.status is DriftStatus.NORMAL
    assert [feat.feature for feat in report.features] == ["score", "mean_gray", "blur_variance"]
    for feat in report.features:
        assert feat.psi == 0.0
        assert feat.status is DriftStatus.NORMAL
    # JSON 직렬화 시 NaN/Inf 없음 확인
    serialized = json.dumps(report.to_dict(), allow_nan=False)
    assert '"NORMAL"' in serialized


def test_drift_detects_watch_and_alert_and_includes_denominators() -> None:
    # 기준선: score 대부분 0.02 (75%) + 0.10 (25%)
    baseline = _make_records(30, score=0.02, mean_gray=45.0, blur_variance=250.0, status="OK") + (
        _make_records(10, score=0.10, mean_gray=45.0, blur_variance=250.0, status="OK")
    )
    # WATCH 구간: 비율 이동으로 0.10 <= PSI < 0.25 유도
    watch_target = (
        _make_records(22, score=0.02, mean_gray=45.0, blur_variance=250.0, status="OK")
        + 
_make_records(18, score=0.10, mean_gray=45.0, blur_variance=250.0, status="OK")
    )
    watch_report = assess_drift(baseline, watch_target)
    assert watch_report.status is DriftStatus.WATCH
    assert 0.10 <= watch_report.features[0].psi < 0.25

    # DRIFT_ALERT 구간: 점수 및 밝기 큰 이동
    alert_target = _make_records(40, score=0.65, mean_gray=130.0, blur_variance=250.0, status="NG")
    alert_report = assess_drift(baseline, alert_target)
    assert alert_report.status is DriftStatus.DRIFT_ALERT
    score_feat = next(f for f in alert_report.features if f.feature == "score")
    gray_feat = next(f for f in alert_report.features if f.feature == "mean_gray")
    assert score_feat.psi >= 0.25
    assert gray_feat.psi >= 0.25
    assert gray_feat.delta_mean == 85.0

    # 상태 비율 분모 및 Wilson CI 확인
    for rate_map in (alert_report.status_rates_baseline, alert_report.status_rates_target):
        for status_name in ("OK", "NG", "REVIEW", "ERROR"):
            entry = rate_map[status_name]
            assert "denominator" in entry
            assert "numerator" in entry
            assert "ci95_low" in entry
    assert alert_report.status_rates_target["NG"]["denominator"] == 40
    assert alert_report.status_rates_target["NG"]["value"] == 1.0


def test_drift_insufficient_sample_prevents_false_alert() -> None:
    baseline = _make_records(50, score=0.02, mean_gray=40.0, blur_variance=250.0)
    small_target = _make_records(19, score=0.95, mean_gray=200.0, blur_variance=10.0, status="NG")
    report = assess_drift(baseline, small_target, policy=DriftPolicy(min_samples=20))
    assert report.status is DriftStatus.INSUFFICIENT_SAMPLE
    assert report.target_total == 19
    assert all(feat.status is DriftStatus.INSUFFICIENT_SAMPLE for feat in report.features)
    assert report.warnings


def test_extract_observations_from_sample_result() -> None:
    res = make_result(Status.REVIEW, score=0.35)
    from dataclasses import replace

    res_q = replace(
        res,
        quality=QualityInfo(blur_variance=120.0, mean_gray=88.0, saturated_ratio=0.01),
    )
    obs = extract_observations([res_q])
    assert len(obs) == 1
    assert obs[0].status == "REVIEW"
    assert obs[0].score == 0.35
    assert obs[0].mean_gray == 88.0
    assert obs[0].blur_variance == 120.0
