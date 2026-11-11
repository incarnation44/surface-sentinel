"""후보(candidates) 축 파일럿 (그리드 12후보 **밖**의 보정 실험).

왜 필요한가
-----------
그리드(4개 scoring 필드)만으로는 목표를 만족하는 후보가 나오지 않는다. 이유는
판정 진리표(설계 §3.2)에 있다. `OK`는 **타당한 컴포넌트가 하나도 없을 때만**
나온다. 그런데 기본 설정에서는 표본의 100%가 컴포넌트를 20개(상한)씩 갖고
`CANDIDATE_TRUNCATED`가 걸린다. 그러면 점수가 `ng_min` 미만인 모든 이미지가
`REVIEW`로 떨어져 review rate가 구조적으로 0.7~0.9가 된다.

따라서 실제로 손댈 축은 **후보 생성 임계값**이다. 이 모듈은 그 축을 작은
폭으로 측정한다(파이프라인 실행 3회 + 각 실행 위에서 scoring 3조합 재계산).

상한 회계: 설계 §4.4의 "그리드 최대 24회 실행"에서 이 모듈의 파이프라인 실행은
3회다. 그리드 12후보는 캐시 재계산이라 "실행"이 아니다. 전체 파이프라인 실행
수는 기본 1 + ablation 6 + 파일럿 3 = 10회다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig
from .measure import VariantReport, evaluate_cached

KIND_CANDIDATES = "candidates"
KIND_SCORING = "scoring"


@dataclass(frozen=True)
class PilotAxis:
    """파일럿 축 하나(후보 생성 또는 scoring)."""

    id: str
    kind: str
    overrides: Mapping[str, Mapping[str, Any]]
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "overrides": {section: dict(values) for section, values in self.overrides.items()},
            "note": self.note,
        }


PILOT_CANDIDATES: tuple[PilotAxis, ...] = (
    PilotAxis(
        id="K2",
        kind=KIND_CANDIDATES,
        overrides={"candidates": {"z_threshold": 8.0, "min_contrast_gray": 20.0,
                                  "merge_kernel": 3, "min_area_px": 30}},
        note="후보를 중간 정도로 조인다(merge 7→3, 면적 12→30).",
    ),
    PilotAxis(
        id="K3",
        kind=KIND_CANDIDATES,
        overrides={"candidates": {"z_threshold": 8.0, "min_contrast_gray": 25.0,
                                  "merge_kernel": 3, "min_area_px": 50}},
        note="더 조인다(대비 25, 면적 50). escape↑·review↓ 방향.",
    ),
    PilotAxis(
        id="K4",
        kind=KIND_CANDIDATES,
        overrides={"candidates": {"z_threshold": 9.0, "min_contrast_gray": 25.0,
                                  "merge_kernel": 3, "min_area_px": 40}},
        note="z를 9로 올린 변형(면적 40).",
    ),
)

PILOT_SCORING: tuple[PilotAxis, ...] = (
    PilotAxis(
        id="S1",
        kind=KIND_SCORING,
        overrides={"scoring": {"ng_area_px": 1500.0, "ng_contrast_gray": 28.0,
                               "ng_min": 0.90}},
        note="정규화 기준을 실측 규모에 맞춘다(면적 48→1500 px, 대비 24→28).",
    ),
    PilotAxis(
        id="S2",
        kind=KIND_SCORING,
        overrides={"scoring": {"ng_area_px": 1000.0, "ng_contrast_gray": 25.0,
                               "ng_min": 0.90}},
        note="S1보다 느슨한 정규화.",
    ),
    PilotAxis(
        id="S3",
        kind=KIND_SCORING,
        overrides={"scoring": {"ng_area_px": 800.0, "ng_contrast_gray": 22.0,
                               "ng_min": 0.95}},
        note="ng_min을 0.95로 올려 review를 줄이는 쪽.",
    ),
    PilotAxis(
        id="S4",
        kind=KIND_SCORING,
        overrides={"scoring": {"ng_area_px": 800.0, "ng_contrast_gray": 22.0,
                               "ng_min": 0.50}},
        note="ng_min을 크게 낮춰 review를 줄인다(대신 false-NG가 늘 수 있다).",
    ),
    PilotAxis(
        id="S5",
        kind=KIND_SCORING,
        overrides={"scoring": {"ng_area_px": 800.0, "ng_contrast_gray": 25.0,
                               "ng_min": 0.60}},
        note="S4보다 보수적인 중간점(대비 25, ng_min 0.60).",
    ),
)


def merge_overrides(*parts: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for part in parts:
        for section, values in part.items():
            merged.setdefault(section, {}).update(values)
    return merged


@dataclass(frozen=True)
class PilotRow:
    """(후보 축, scoring 축) 한 칸의 측정 결과."""

    candidate_id: str
    scoring_id: str
    overrides: Mapping[str, Mapping[str, Any]]
    report: VariantReport

    @property
    def config_hash(self) -> str:
        return self.report.config_hash

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

    def targets(self, *, escape_max: float, false_ng_max: float, review_max: float,
                recall_min: float) -> dict[str, bool | None]:
        escape, false_ng, review, recall = (
            self.escape,
            self.false_ng,
            self.review,
            self.localization_recall,
        )
        return {
            "escape": None if escape is None else escape <= escape_max,
            "false_ng": None if false_ng is None else false_ng <= false_ng_max,
            "review": None if review is None else review <= review_max,
            "localization": None if recall is None else recall >= recall_min,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scoring_id": self.scoring_id,
            "overrides": {section: dict(values) for section, values in self.overrides.items()},
            "config_hash": self.config_hash,
            "samples": self.report.counts.get("samples"),
            "metrics": {
                "escape": self.report.ratio("defect_escape_rate").as_dict(),
                "false_ng": self.report.ratio("good_false_ng_rate").as_dict(),
                "review": self.report.ratio("review_rate").as_dict(),
                "localization_recall_iou0.10": self.report.ratio(
                    "localization_recall_iou0.10"
                ).as_dict(),
            },
        }


def pilot_config(
    candidate: PilotAxis,
    scoring: PilotAxis,
    base_config: AppConfig,
) -> AppConfig:
    from .ablation import apply_overrides

    return apply_overrides(base_config, merge_overrides(candidate.overrides, scoring.overrides))


def evaluate_pilot_scoring(
    samples: Sequence[Any],
    candidate: PilotAxis,
    base_config: AppConfig,
    *,
    scoring_axes: Sequence[PilotAxis] = PILOT_SCORING,
) -> list[PilotRow]:
    """한 후보 축 캐시 위에서 scoring 축들을 재계산한다(파이프라인 재실행 없음)."""

    rows: list[PilotRow] = []
    for scoring in scoring_axes:
        overrides = merge_overrides(candidate.overrides, scoring.overrides)
        config = pilot_config(candidate, scoring, base_config)
        report = evaluate_cached(
            samples,
            config,
            name=f"{candidate.id}+{scoring.id}",
            overrides=overrides,
            notes=(f"{candidate.note} / {scoring.note}",),
        )
        rows.append(
            PilotRow(
                candidate_id=candidate.id,
                scoring_id=scoring.id,
                overrides=overrides,
                report=report,
            )
        )
    return rows


def select_pilot(
    rows: Sequence[PilotRow],
    *,
    escape_max: float,
    false_ng_max: float,
    review_max: float,
) -> PilotRow | None:
    """세 목표를 모두 만족하는 행 중 escape 최소·review 최소."""

    feasible = [
        row
        for row in rows
        if (row.escape is not None and row.escape <= escape_max)
        and (row.false_ng is not None and row.false_ng <= false_ng_max)
        and (row.review is not None and row.review <= review_max)
    ]
    if not feasible:
        return None
    feasible.sort(
        key=lambda row: (
            row.escape if row.escape is not None else 1.0,
            row.review if row.review is not None else 1.0,
            row.candidate_id,
            row.scoring_id,
        )
    )
    return feasible[0]


__all__ = [
    "KIND_CANDIDATES",
    "KIND_SCORING",
    "PILOT_CANDIDATES",
    "PILOT_SCORING",
    "PilotAxis",
    "PilotRow",
    "evaluate_pilot_scoring",
    "merge_overrides",
    "pilot_config",
    "select_pilot",
]
