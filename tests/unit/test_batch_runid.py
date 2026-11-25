"""배치 `run_id` 생성 및 지연시간 백분위수 계산 단위 테스트.

충돌 시 덮어쓰지 않는 접미사 증가 규칙과 nearest-rank 백분위수 단조성을 검증한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from surface_sentinel.batch import (
    MAX_RUN_ID_ATTEMPTS,
    RUN_ID_PATTERN,
    BatchStorageError,
    allocate_run_id,
    percentile,
)

FIXED_MOMENT = datetime(2026, 11, 10, 7, 8, 9, tzinfo=UTC)
CONFIG_HASH = "0123456789abcdef" + "0" * 48


def _allocate(output_dir: Path) -> str:
    return allocate_run_id(
        output_dir,
        split="tune",
        config_hash=CONFIG_HASH,
        now=FIXED_MOMENT,
    )


def test_allocate_run_id_matches_contract_pattern(tmp_path: Path) -> None:
    run_id = _allocate(tmp_path)

    assert run_id == "20261110T070809Z-01234567-tune"
    assert RUN_ID_PATTERN.match(run_id)


@pytest.mark.parametrize(
    ("run_id", "expected"),
    [
        ("20261110T070809Z-01234567-tune", True),
        ("20261110T070809Z-01234567-tune-01", True),
        ("20261110T070809Z-01234567-test", True),
        ("20261110T070809Z-01234567-official_test-12", True),
        ("20261110T070809Z-01234567-tune-1", False),
        ("2026111T070809Z-01234567-tune", False),
        ("20261110t070809z-01234567-tune", False),
        ("20261110T070809Z-0123456-tune", False),
        ("20261110T070809Z-01234567-eval", False),
    ],
)
def test_run_id_pattern_accepts_only_contract_forms(run_id: str, expected: bool) -> None:
    assert bool(RUN_ID_PATTERN.match(run_id)) is expected


def test_allocate_run_id_increments_instead_of_overwriting(tmp_path: Path) -> None:
    first = _allocate(tmp_path)
    (tmp_path / first).mkdir()

    second = _allocate(tmp_path)
    assert second == f"{first}-01"
    (tmp_path / second).mkdir()

    third = _allocate(tmp_path)
    assert third == f"{first}-02"
    # 기존 디렉터리는 그대로 남아 있다.
    assert sorted(path.name for path in tmp_path.iterdir()) == [first, second]


def test_allocate_run_id_gives_up_when_all_names_taken(tmp_path: Path) -> None:
    first = _allocate(tmp_path)
    (tmp_path / first).mkdir()
    for index in range(1, MAX_RUN_ID_ATTEMPTS):
        (tmp_path / f"{first}-{index:02d}").mkdir()

    with pytest.raises(BatchStorageError):
        _allocate(tmp_path)


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 50) == 5.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0


def test_percentile_is_monotonic_and_rejects_bad_percent() -> None:
    values = [3.0, 1.0, 2.0, 8.0, 5.0]
    assert percentile(values, 50) <= percentile(values, 95)
    with pytest.raises(ValueError):
        percentile(values, 0)
    with pytest.raises(ValueError):
        percentile(values, 101)
