"""tune 내부 group-aware train/val 분할(ADR 0007 D4).

이미지가 아니라 ``group_id`` 단위로 배정해 같은 부품 그룹이 양쪽에
섞이는 누출을 막는다. 그룹 라벨(결함 포함 여부)로 층화하고 시드로
결정론적으로 섞은 뒤 크기 내림차순 탐욕 배정한다.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..io.manifest import ManifestRow
from ..io.splits import build_groups

DEFAULT_SEED = 20261112
DEFAULT_VAL_FRACTION = 0.2

SIDE_TRAIN = "train"
SIDE_VAL = "val"


@dataclass(frozen=True)
class TrainValSplit:
    """그룹 배정 결과와 수량 요약."""

    assignment: dict[str, str]
    seed: int
    val_fraction: float
    counts: dict[str, dict[str, int]]


def group_train_val_split(
    rows: Sequence[ManifestRow],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> TrainValSplit:
    """정본 행을 그룹 단위로 train/val에 배정한다."""

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction은 (0, 1) 범위여야 합니다: {val_fraction}")
    groups = build_groups(rows)
    if not groups:
        raise ValueError("분할할 정본 행이 없습니다.")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    by_label: dict[str, list] = {"defect": [], "good": []}
    for group in shuffled:
        by_label[group.label].append(group)
    for members in by_label.values():
        members.sort(key=lambda group: -group.canonical_size)

    assignment: dict[str, str] = {}
    counts = {
        side: {"rows": 0, "defect": 0, "good": 0, "groups": 0}
        for side in (SIDE_TRAIN, SIDE_VAL)
    }
    for label in ("defect", "good"):
        total = sum(group.canonical_size for group in by_label[label])
        val_target = int(round(val_fraction * total))
        val_count = 0
        for group in by_label[label]:
            side = SIDE_VAL if val_count < val_target else SIDE_TRAIN
            assignment[group.group_id] = side
            counts[side]["rows"] += group.canonical_size
            counts[side][label] += group.canonical_size
            counts[side]["groups"] += 1
            if side == SIDE_VAL:
                val_count += group.canonical_size

    return TrainValSplit(
        assignment=assignment,
        seed=seed,
        val_fraction=val_fraction,
        counts=counts,
    )


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_VAL_FRACTION",
    "SIDE_TRAIN",
    "SIDE_VAL",
    "TrainValSplit",
    "group_train_val_split",
]