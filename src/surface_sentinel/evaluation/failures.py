"""실패 사례 색인 (설계 §4.5).

"임의로 고르지 않는다"가 규칙이다. 그래서 색인 CSV에는 **표본 전체의 실패**를
남기고, 별도 JSON에는 §4.5의 선택 규칙으로 뽑은 사례만 따로 적는다.

| 규칙 | 정의 |
|---|---|
| 고득점 escape 상위 2 | defect → OK, score 내림차순 |
| 고득점 오탐 상위 2 | good → NG, score 내림차순 |
| 가장 큰 REVIEW 클러스터 | REVIEW 사유 코드별 묶음 중 최대 |
| 경계 사례 | score가 ``ok_max`` 또는 ``ng_min``에서 ±0.05 이내 |
| 품질 플래그 사례 | quality flag가 하나라도 있는 샘플 |
| 부분 위치 사례 | 정답은 맞았지만 best IoU가 0.30 미만인 defect |
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .cache import CachedSample
from .measure import SampleOutcome, VariantReport, component_iou

FAILURE_COLUMNS = (
    "sample_id",
    "split",
    "label",
    "status",
    "reason_code",
    "score",
    "overlay",
    "category",
    "status_base",
    "quality_flags",
    "components",
    "best_iou",
)

CATEGORY_ESCAPE = "escape"
CATEGORY_DEFECT_REVIEW = "defect_review"
CATEGORY_FALSE_NG = "false_ng"
CATEGORY_GOOD_REVIEW = "good_review"
CATEGORY_ERROR = "error"

BOUNDARY_MARGIN = 0.05
PARTIAL_IOU_MAX = 0.30


@dataclass(frozen=True)
class FailureRow:
    sample_id: str
    split: str
    label: str
    status: str
    reason_code: str
    score: float
    overlay: str
    category: str
    status_base: str
    quality_flags: str
    components: int
    best_iou: float

    def as_dict(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in FAILURE_COLUMNS}


def _posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _category(label: str, status: str) -> str | None:
    if status == "ERROR":
        return CATEGORY_ERROR
    if label == "defect":
        if status == "OK":
            return CATEGORY_ESCAPE
        if status == "REVIEW":
            return CATEGORY_DEFECT_REVIEW
        return None
    if status == "NG":
        return CATEGORY_FALSE_NG
    if status == "REVIEW":
        return CATEGORY_GOOD_REVIEW
    return None


def _best_iou(sample: CachedSample, outcome: SampleOutcome) -> float:
    best = 0.0
    for gt in sample.gt_components:
        for predicted in outcome.predicted:
            best = max(best, component_iou(gt, predicted))
    return best


def build_failure_rows(report: VariantReport, samples: Sequence[CachedSample]) -> list[FailureRow]:
    """실패(정답이 아닌) 샘플 전체를 색인 행으로 만든다."""

    by_id: Mapping[str, CachedSample] = {sample.sample_id: sample for sample in samples}
    rows: list[FailureRow] = []
    for outcome in report.outcomes:
        category = _category(outcome.label, outcome.status.value)
        if category is None:
            continue
        sample = by_id[outcome.sample_id]
        rows.append(
            FailureRow(
                sample_id=outcome.sample_id,
                split=sample.split,
                label=outcome.label,
                status=outcome.status.value,
                reason_code=outcome.reason_code,
                score=round(float(outcome.score), 6),
                overlay=_posix(sample.overlay) if sample.overlay else "",
                category=category,
                status_base=sample.status.value,
                quality_flags=";".join(sample.quality_flags),
                components=len(sample.components),
                best_iou=round(_best_iou(sample, outcome), 6),
            )
        )
    rows.sort(key=lambda row: (row.category, row.label, -row.score, row.sample_id))
    return rows


def write_failure_csv(rows: Sequence[FailureRow], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FAILURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return target


def _best_iou_by_id(
    report: VariantReport, samples: Sequence[CachedSample]
) -> dict[str, float]:
    by_id = {sample.sample_id: sample for sample in samples}
    return {
        outcome.sample_id: round(_best_iou(by_id[outcome.sample_id], outcome), 6)
        for outcome in report.outcomes
        if outcome.sample_id in by_id
    }


def select_cases(
    rows: Sequence[FailureRow],
    report: VariantReport,
    samples: Sequence[CachedSample],
    config: AppConfig,
) -> dict[str, list[dict[str, Any]]]:
    """§4.5의 규칙으로 대표 사례를 뽑는다. 규칙 이름을 키로 쓴다."""

    def entry(row: FailureRow) -> dict[str, Any]:
        return {
            "sample_id": row.sample_id,
            "label": row.label,
            "status": row.status,
            "reason_code": row.reason_code,
            "score": row.score,
            "category": row.category,
            "overlay": row.overlay,
            "best_iou": row.best_iou,
        }

    def sorted_by_score(category: str) -> list[FailureRow]:
        return sorted(
            (row for row in rows if row.category == category),
            key=lambda row: (-row.score, row.sample_id),
        )

    escapes = sorted_by_score(CATEGORY_ESCAPE)
    false_ng = sorted_by_score(CATEGORY_FALSE_NG)

    review_rows = [
        row
        for row in rows
        if row.category in (CATEGORY_DEFECT_REVIEW, CATEGORY_GOOD_REVIEW)
    ]
    clusters: dict[str, list[FailureRow]] = defaultdict(list)
    for row in review_rows:
        clusters[row.reason_code].append(row)
    largest = max(clusters.items(), key=lambda item: (len(item[1]), item[0]), default=("", []))
    largest_rows = sorted(largest[1], key=lambda row: (-row.score, row.sample_id))

    boundary: list[FailureRow] = []
    for row in rows:
        if row.category == CATEGORY_ERROR:
            continue
        if abs(row.score - config.scoring.ok_max) <= BOUNDARY_MARGIN or abs(
            row.score - config.scoring.ng_min
        ) <= BOUNDARY_MARGIN:
            boundary.append(row)
    boundary.sort(key=lambda row: (abs(row.score - config.scoring.ng_min), row.sample_id))

    quality = [row for row in rows if row.quality_flags]
    quality.sort(key=lambda row: (-row.score, row.sample_id))

    io_us = _best_iou_by_id(report, samples)
    partial = [
        row
        for row in rows
        if row.label == "defect"
        and row.status in {"NG", "REVIEW"}
        and 0.0 < io_us.get(row.sample_id, 0.0) < PARTIAL_IOU_MAX
    ]
    partial.sort(key=lambda row: (io_us.get(row.sample_id, 0.0), row.sample_id))

    return {
        "high_score_escape_top2": [entry(row) for row in escapes[:2]],
        "high_score_false_ng_top2": [entry(row) for row in false_ng[:2]],
        "largest_review_cluster": [entry(row) for row in largest_rows[:5]],
        "largest_review_cluster_reason": largest[0],
        "largest_review_cluster_size": len(largest_rows),
        "boundary_cases": [entry(row) for row in boundary[:5]],
        "quality_flag_cases": [entry(row) for row in quality[:5]],
        "partial_location_cases": [entry(row) for row in partial[:5]],
        "category_counts": dict(Counter(row.category for row in rows)),
        "reason_counts": dict(Counter(row.reason_code for row in rows)),
    }


def reason_distribution(rows: Sequence[FailureRow]) -> dict[str, dict[str, int]]:
    """카테고리 × 사유 코드 분포. 분모(카테고리 크기)는 호출자가 표에 적는다."""

    payload: dict[str, dict[str, int]] = {}
    for category in sorted({row.category for row in rows}):
        payload[category] = dict(
            Counter(row.reason_code for row in rows if row.category == category)
        )
    return payload


__all__ = [
    "BOUNDARY_MARGIN",
    "CATEGORY_DEFECT_REVIEW",
    "CATEGORY_ERROR",
    "CATEGORY_ESCAPE",
    "CATEGORY_FALSE_NG",
    "CATEGORY_GOOD_REVIEW",
    "FAILURE_COLUMNS",
    "PARTIAL_IOU_MAX",
    "FailureRow",
    "build_failure_rows",
    "reason_distribution",
    "select_cases",
    "write_failure_csv",
]
