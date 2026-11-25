"""스코어링 임계값 그리드 탐색 및 Pareto 최적 후보 선별 모듈.

후보는 **판정·점수 계층의 3개 필드**만 흔든다.

1. ``scoring.ng_area_px`` — 면적 정규화 기준
2. ``scoring.ng_contrast_gray`` — 대비 정규화 기준
3. ``scoring.ng_min`` — NG 하한

이 3개를 고른 이유: 이 필드들만이 캐시된 컴포넌트 재료 위에서
evidence·score·판정을 **정확히** 다시 계산할 수 있다(파이프라인 재실행 없음).
``candidates.z_threshold``처럼 후보 마스크 자체를 바꾸는 값은 재실행이
필요하므로 그리드에서 제외했다. 대신 ablation이 그 축을 따로 본다.

F4(2026-11-11): ``scoring.ok_max`` 축은 제거했다. OK 상한은
``REVIEW_BAND_MAX = 0.40``으로 동결됐고 config 값이 아니다.

선택 규칙(§4.4): ``false-NG ≤ 0.20`` 과 ``review ≤ 0.40`` 을 만족하는 후보 중
``escape``가 최소인 것. 만족하는 후보가 없으면 Pareto 점을 보고한다.
``localization recall@IoU0.10 ≥ 0.50`` 은 목표 확인용으로 함께 기록한다.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .cache import CachedSample
from .measure import VariantReport, evaluate_cached

MAX_CANDIDATES = 24

#: 목표치 (T05 확정 사항).
TARGET_ESCAPE = 0.15
TARGET_FALSE_NG = 0.20
TARGET_REVIEW = 0.40
TARGET_LOCALIZATION_RECALL = 0.50

#: 3개 필드의 후보 값. 2 × 3 × 2 = 12 (상한 24 이내).
#: 기본값(800 px, 22 gray, ng_min 0.50)은 S4 동결 값이다.
#: F4(2026-11-11): ``scoring.ok_max`` 축은 제거했다.
GRID_VALUES: dict[str, tuple[float, ...]] = {
    "scoring.ng_area_px": (500.0, 2000.0),
    "scoring.ng_contrast_gray": (24.0, 30.0, 40.0),
    "scoring.ng_min": (0.90, 1.00),
}
GRID_FIELDS: tuple[str, ...] = tuple(GRID_VALUES)


@dataclass(frozen=True)
class Candidate:
    """그리드 후보 하나. ``overrides``는 섹션/필드 형태다."""

    candidate_id: str
    overrides: Mapping[str, float]

    def sectioned(self) -> dict[str, dict[str, float]]:
        payload: dict[str, dict[str, float]] = {}
        for key, value in self.overrides.items():
            section, field = key.split(".", 1)
            payload.setdefault(section, {})[field] = value
        return payload

    def as_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "overrides": dict(self.overrides)}


def build_candidates() -> tuple[Candidate, ...]:
    """12후보를 결정론적 순서로 만든다(사전순 곱, 고정)."""

    combos = list(itertools.product(*(GRID_VALUES[field] for field in GRID_FIELDS)))
    if len(combos) > MAX_CANDIDATES:
        raise ValueError(
            f"후보가 상한을 넘습니다: {len(combos)} > {MAX_CANDIDATES}"
        )
    candidates: list[Candidate] = []
    for index, values in enumerate(combos, start=1):
        candidates.append(
            Candidate(
                candidate_id=f"g{index:02d}",
                overrides={
                    field: float(value)
                    for field, value in zip(GRID_FIELDS, values, strict=True)
                },
            )
        )
    return tuple(candidates)


@dataclass(frozen=True)
class GridRow:
    """그리드 한 칸의 결과."""

    candidate_id: str
    overrides: Mapping[str, float]
    config_hash: str
    report: VariantReport

    def metric(self, name: str) -> dict[str, Any]:
        return self.report.ratio(name).as_dict()

    @property
    def escape(self) -> float | None:
        return self.report.ratio("defect_escape_rate").value

    @property
    def false_ng(self) -> float | None:
        return self.report.ratio("good_false_ng_rate").value

    @property
    def review(self) -> float | None:
        return self.report.ratio("review_rate").value

    @property
    def localization_recall(self) -> float | None:
        return self.report.ratio("localization_recall_iou0.10").value

    def targets(self) -> dict[str, bool | None]:
        escape = self.escape
        false_ng = self.false_ng
        review = self.review
        recall = self.localization_recall
        return {
            "escape_le_0.15": None if escape is None else escape <= TARGET_ESCAPE,
            "false_ng_le_0.20": None if false_ng is None else false_ng <= TARGET_FALSE_NG,
            "review_le_0.40": None if review is None else review <= TARGET_REVIEW,
            "localization_recall_ge_0.50": (
                None if recall is None else recall >= TARGET_LOCALIZATION_RECALL
            ),
        }

    def selection_ready(self) -> bool:
        """§4.4 선택 규칙의 세 조건(escape는 목적함수이므로 제외)."""

        false_ng = self.false_ng
        review = self.review
        if false_ng is None or review is None:
            return False
        return false_ng <= TARGET_FALSE_NG and review <= TARGET_REVIEW

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "overrides": {key: float(value) for key, value in self.overrides.items()},
            "config_hash": self.config_hash,
            "samples": self.report.counts.get("samples"),
            "defect": self.report.counts.get("defect"),
            "good": self.report.counts.get("good"),
            "metrics": {
                "escape": self.metric("defect_escape_rate"),
                "defect_ng": self.metric("defect_ng_rate"),
                "defect_review": self.metric("defect_review_rate"),
                "false_ng": self.metric("good_false_ng_rate"),
                "good_review": self.metric("good_review_rate"),
                "review": self.metric("review_rate"),
                "automation": self.metric("automation_rate"),
                "localization_recall_iou0.10": self.metric("localization_recall_iou0.10"),
            },
            "targets": self.targets(),
            "selection_ready": self.selection_ready(),
        }


def evaluate_grid(
    samples: Sequence[CachedSample],
    base_config: AppConfig,
    *,
    candidates: Sequence[Candidate] | None = None,
) -> list[GridRow]:
    """캐시 위에서 후보별 지표를 계산한다. 파이프라인 재실행은 없다."""

    from .ablation import apply_overrides

    rows: list[GridRow] = []
    for candidate in candidates if candidates is not None else build_candidates():
        config = apply_overrides(base_config, candidate.sectioned())
        report = evaluate_cached(samples, config, name=candidate.candidate_id)
        rows.append(
            GridRow(
                candidate_id=candidate.candidate_id,
                overrides=dict(candidate.overrides),
                config_hash=config.config_hash,
                report=report,
            )
        )
    return rows


def _rank_key(row: GridRow) -> tuple[float, float, float, float, str]:
    escape = row.escape if row.escape is not None else 1.0
    false_ng = row.false_ng if row.false_ng is not None else 1.0
    review = row.review if row.review is not None else 1.0
    recall = row.localization_recall if row.localization_recall is not None else 0.0
    return (escape, false_ng, review, -recall, row.candidate_id)


def pareto_front(rows: Sequence[GridRow]) -> list[GridRow]:
    """escape·false-NG·review를 모두 최소화하는 Pareto 점(비지배 집합)."""

    def vector(row: GridRow) -> tuple[float, float, float]:
        return (
            row.escape if row.escape is not None else 1.0,
            row.false_ng if row.false_ng is not None else 1.0,
            row.review if row.review is not None else 1.0,
        )

    front: list[GridRow] = []
    for row in rows:
        values = vector(row)
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_values = vector(other)
            if (
                all(a <= b for a, b in zip(other_values, values, strict=True))
                and any(a < b for a, b in zip(other_values, values, strict=True))
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=_rank_key)


@dataclass(frozen=True)
class Selection:
    """그리드 선택 결과. 목표를 다 만족하는 후보가 없으면 Pareto를 쓴다."""

    best: GridRow | None
    feasible: tuple[GridRow, ...]
    pareto: tuple[GridRow, ...]
    reason: str


def select(rows: Sequence[GridRow]) -> Selection:
    feasible = tuple(sorted((row for row in rows if row.selection_ready()), key=_rank_key))
    if feasible:
        best = feasible[0]
        reason = (
            "false-NG ≤ 0.20 과 review ≤ 0.40 을 만족하는 후보 중 escape 최소"
            f" (후보 {len(feasible)}/{len(rows)}개가 조건 충족)"
        )
        return Selection(
            best=best,
            feasible=feasible,
            pareto=tuple(pareto_front(rows)),
            reason=reason,
        )
    front = tuple(pareto_front(rows))
    best = min(front, key=minimax_key) if front else None
    reason = (
        "세 조건을 동시에 만족하는 후보가 없어 Pareto 점을 보고한다"
        f" (Pareto {len(front)}개). 표의 '최선'은 목표 대비 최악 위반비"
        "(max(escape/0.15, false-NG/0.20, review/0.40))가 가장 작은 점이다"
    )
    return Selection(best=best, feasible=(), pareto=front, reason=reason)


def minimax_key(row: GridRow) -> tuple[float, str]:
    """목표 대비 최악 위반비. 세 목표를 같은 자로 재는 지표다."""

    escape = row.escape if row.escape is not None else 1.0
    false_ng = row.false_ng if row.false_ng is not None else 1.0
    review = row.review if row.review is not None else 1.0
    worst = max(
        escape / TARGET_ESCAPE,
        false_ng / TARGET_FALSE_NG,
        review / TARGET_REVIEW,
    )
    return (worst, row.candidate_id)


def write_jsonl(rows: Sequence[GridRow], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "GRID_FIELDS",
    "GRID_VALUES",
    "MAX_CANDIDATES",
    "TARGET_ESCAPE",
    "TARGET_FALSE_NG",
    "TARGET_LOCALIZATION_RECALL",
    "TARGET_REVIEW",
    "Candidate",
    "GridRow",
    "Selection",
    "build_candidates",
    "evaluate_grid",
    "minimax_key",
    "pareto_front",
    "read_jsonl",
    "select",
    "write_jsonl",
]
