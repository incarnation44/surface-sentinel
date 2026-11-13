"""T05 표본 설계: defect 전수 + good 층화 추출 (설계 §4.4).

표본은 **tune 분할에서만** 고른다. 봉인 분할(``test``/``official_test``) 이름은
여기서 즉시 거부하며, 이 모듈에는 봉인 CSV를 읽는 경로가 없다.

규칙
----
* ``defect``: tune의 결함 이미지 **전수**(분모 = tune 결함 수). 표본 오차가 없다.
* ``good``: ``source_split``(KSDD2 원본 분할)로 층화한 500장.
  배분은 비례 배분 + 최대 나머지 법으로 결정하고, 같은 층 안에서는
  ``sha256(seed:sample_id)`` 순서로 앞에서부터 뽑는다. 난수 생성기를 쓰지
  않으므로 파이썬/넘파이 버전이 달라도 같은 표본이 나온다.

모든 비율은 분모를 함께 보고한다. ``SamplePlan.as_dict()``가 그 근거다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..io.manifest import ManifestRow

SEALED_SPLITS = frozenset({"test", "official_test"})
EVAL_SPLITS = ("tune",)

#: 표본 설계 상수. 보고서에 그대로 적는다.
DEFECT_MODE = "census"
GOOD_TARGET = 500
DEFAULT_SEED = "t05-2026-11-10"


class SampleError(ValueError):
    """표본을 만들 수 없을 때의 오류."""


@dataclass(frozen=True)
class StratumPlan:
    """층 하나의 모집단/요청/실제 수량. ``requested``가 ``available``을 넘지 않는다."""

    stratum: str
    population: int
    requested: int

    @property
    def fraction(self) -> float | None:
        if self.population <= 0:
            return None
        return self.requested / self.population

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "population": self.population,
            "requested": self.requested,
            "fraction": self.fraction,
        }


@dataclass(frozen=True)
class SamplePlan:
    """표본 행과 추출 근거. 모든 수량에 분모가 붙는다."""

    rows: tuple[ManifestRow, ...]
    split: str
    seed: str
    defect_population: int
    good_population: int
    strata: tuple[StratumPlan, ...]

    @property
    def defect_taken(self) -> int:
        return sum(1 for row in self.rows if row.label == "defect")

    @property
    def good_taken(self) -> int:
        return sum(1 for row in self.rows if row.label != "defect")

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seed": self.seed,
            "defect_mode": DEFECT_MODE,
            "defect_population": self.defect_population,
            "defect_taken": self.defect_taken,
            "good_population": self.good_population,
            "good_requested": sum(stratum.requested for stratum in self.strata),
            "good_taken": self.good_taken,
            "total": len(self.rows),
            "strata": [stratum.as_dict() for stratum in self.strata],
            "rule": (
                "defect=전수(분모=tune 결함), good=source_split 층화 "
                f"{GOOD_TARGET}장(비례 배분+최대 나머지, sha256(seed:id) 정렬)"
            ),
        }


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _allocate(populations: dict[str, int], target: int) -> dict[str, int]:
    """비례 배분 + 최대 나머지. 가용량을 넘는 요청은 다른 층으로 넘긴다."""

    total = sum(populations.values())
    if total <= 0:
        return dict.fromkeys(populations, 0)
    budget = min(target, total)
    quotas = {name: budget * count / total for name, count in populations.items()}
    allocation = {name: min(int(value // 1), populations[name]) for name, value in quotas.items()}
    remaining = budget - sum(allocation.values())
    # 나머지는 소수부가 큰 층부터, 동률이면 이름 순(결정론)으로 채운다.
    order = sorted(
        populations,
        key=lambda name: (-(quotas[name] - int(quotas[name] // 1)), name),
    )
    guard = 0
    while remaining > 0 and guard < budget + 1:
        guard += 1
        progressed = False
        for name in order:
            if remaining <= 0:
                break
            if allocation[name] < populations[name]:
                allocation[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation


def build_sample_plan(
    rows: Iterable[ManifestRow],
    *,
    split: str = "tune",
    good_target: int = GOOD_TARGET,
    seed: str = DEFAULT_SEED,
    allow_sealed: bool = False,
) -> SamplePlan:
    """tune 행에서 defect 전수 + good 층화 표본을 만든다.

    ``allow_sealed=True`` 면 봉인 분할도 전수 표본으로 만들 수 있다(T08).
    기본값 False 면 기존 거부 동작을 유지한다.
    """

    sealed_allowed = allow_sealed and split in SEALED_SPLITS
    if split in SEALED_SPLITS and not sealed_allowed:
        raise SampleError(f"봉인된 split에서는 표본을 만들 수 없습니다: {split}")
    if split not in EVAL_SPLITS and not sealed_allowed:
        raise SampleError(f"T05는 {split!r} 분할을 지원하지 않습니다: {list(EVAL_SPLITS)}")
    if good_target < 0:
        raise SampleError(f"good_target은 0 이상이어야 합니다: {good_target}")

    materialised = list(rows)
    if not materialised:
        raise SampleError(f"분할 {split!r}에 행이 없습니다.")
    bad_split = {row.label for row in materialised} - {"good", "defect"}
    if bad_split:
        raise SampleError(f"알 수 없는 label입니다: {sorted(bad_split)}")

    defects = [row for row in materialised if row.label == "defect"]
    goods = [row for row in materialised if row.label == "good"]

    strata_population: dict[str, int] = {}
    for row in goods:
        strata_population[row.source_split] = strata_population.get(row.source_split, 0) + 1
    allocation = _allocate(strata_population, good_target)

    chosen_good: list[ManifestRow] = []
    strata: list[StratumPlan] = []
    for name in sorted(strata_population):
        pool = sorted(
            (row for row in goods if row.source_split == name),
            key=lambda row: _rank(seed, row.sample_id),
        )
        take = allocation[name]
        chosen_good.extend(pool[:take])
        strata.append(
            StratumPlan(stratum=name, population=strata_population[name], requested=take)
        )

    ordered = sorted(defects, key=lambda row: row.sample_id) + sorted(
        chosen_good, key=lambda row: row.sample_id
    )
    duplicates = len({row.sample_id for row in ordered}) != len(ordered)
    if duplicates:
        raise SampleError("표본에 중복 sample_id가 있습니다.")

    return SamplePlan(
        rows=tuple(ordered),
        split=split,
        seed=seed,
        defect_population=len(defects),
        good_population=len(goods),
        strata=tuple(strata),
    )


def sample_ids(plan: SamplePlan) -> Sequence[str]:
    return tuple(row.sample_id for row in plan.rows)


__all__ = [
    "DEFAULT_SEED",
    "DEFECT_MODE",
    "EVAL_SPLITS",
    "GOOD_TARGET",
    "SEALED_SPLITS",
    "SampleError",
    "SamplePlan",
    "StratumPlan",
    "build_sample_plan",
    "sample_ids",
]
