"""결함 영역 위치 지표(`recall@IoU`, IoU 분위수·히스토그램) 단위 테스트.

경계값 포함 규칙, 빈 입력 처리, 분위수 보간 및 구간 집계를 검증한다.
"""

from __future__ import annotations

import pytest

from surface_sentinel.evaluation.metrics import (
    IOU_MATCH_FULL,
    GtMatch,
    iou_distribution,
    recall_at,
)


def _match(iou: float, status: str = "NG") -> GtMatch:
    return GtMatch(iou=iou, image_status=status)


def test_recall_at_counts_boundary_value_inclusive() -> None:
    matches = [_match(0.50), _match(0.30), _match(0.10)]
    boundary = recall_at(matches, IOU_MATCH_FULL)
    assert boundary.numerator == 1
    assert boundary.denominator == 3
    assert boundary.value == pytest.approx(1 / 3)
    assert recall_at(matches, 0.30).numerator == 2
    assert recall_at(matches, 0.10).numerator == 3


def test_recall_at_empty_matches_is_undefined() -> None:
    result = recall_at([], 0.50)
    assert result.value is None
    assert result.denominator == 0
    assert result.ci_low is None and result.ci_high is None


def test_iou_distribution_empty_is_deterministic() -> None:
    payload = iou_distribution([])
    assert payload["count"] == 0
    assert payload["mean"] is None
    assert all(value is None for value in payload["quantiles"].values())
    assert [item["count"] for item in payload["histogram"]] == [0, 0, 0, 0, 0, 0]


def test_iou_distribution_quantiles_and_bins() -> None:
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    payload = iou_distribution([_match(value) for value in values])
    assert payload["count"] == 11
    assert payload["mean"] == pytest.approx(0.5)
    assert payload["quantiles"]["p10"] == pytest.approx(0.1)
    assert payload["quantiles"]["p50"] == pytest.approx(0.5)
    assert payload["quantiles"]["p90"] == pytest.approx(0.9)
    assert [item["count"] for item in payload["histogram"]] == [1, 2, 2, 2, 2, 2]
    last = payload["histogram"][-1]
    assert last["lo"] == 0.9 and last["hi"] == 1.0


def test_iou_distribution_last_bin_includes_one() -> None:
    payload = iou_distribution([_match(1.0), _match(0.899)])
    assert payload["histogram"][-1]["count"] == 1
    assert payload["histogram"][-2]["count"] == 1
