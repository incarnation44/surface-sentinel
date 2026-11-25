"""이미지 단위 그룹·짝 부트스트랩 CI(`evaluation/bootstrap.py`) 단위 테스트.

리샘플링 결정론, 분모 0 이미지 제외, 짝 delta 부호, 채택 검증 CLI의 봉인 차단을 검증한다.
"""

from __future__ import annotations

import pytest

from surface_sentinel.cli import main
from surface_sentinel.evaluation.bootstrap import (
    group_bootstrap_ci,
    paired_bootstrap_delta,
)


def test_group_bootstrap_is_deterministic() -> None:
    num = [1, 0, 1, 1, 0, 1]
    den = [1, 1, 1, 1, 1, 1]
    first = group_bootstrap_ci(num, den, b=200, seed=7)
    second = group_bootstrap_ci(num, den, b=200, seed=7)
    assert first == second
    assert first["value"] == pytest.approx(4 / 6)
    assert first["n"] == 6


def test_group_bootstrap_excludes_zero_denominator() -> None:
    payload = group_bootstrap_ci([1, 1, 0], [1, 1, 0], b=100, seed=11)
    assert payload["n"] == 2
    assert payload["value"] == pytest.approx(1.0)
    assert payload["denominator"] == pytest.approx(2.0)


def test_group_bootstrap_empty_is_undefined() -> None:
    payload = group_bootstrap_ci([], [], b=100, seed=11)
    assert payload["n"] == 0
    assert payload["value"] is None
    assert payload["ci_low"] is None and payload["ci_high"] is None


def test_paired_bootstrap_delta_sign() -> None:
    payload = paired_bootstrap_delta(
        [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], b=100, seed=13
    )
    assert payload["delta"] == pytest.approx(1.0)
    assert payload["ci_low"] == pytest.approx(1.0)
    assert payload["ci_high"] == pytest.approx(1.0)


def test_paired_bootstrap_excludes_zero_denominator() -> None:
    payload = paired_bootstrap_delta(
        [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0], b=100, seed=13
    )
    assert payload["n"] == 2
    assert payload["delta"] == pytest.approx(1.0)


def test_cli_adopted_verify_rejects_sealed_split() -> None:
    assert main(["adopted-verify", "--split", "test"]) == 6
