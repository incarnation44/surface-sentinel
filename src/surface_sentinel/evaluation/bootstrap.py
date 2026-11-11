"""이미지 단위 그룹·짝 부트스트랩 CI (F6, 2026-11-11).

비율의 분모가 이미지마다 다른 경우(예: GT 컴포넌트 수)를 다루기 위해
**이미지 단위로 복원추출**한다. 같은 seed면 같은 결과가 나온다.
분모가 0인 이미지는 계산에서 제외한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_BOOTSTRAP_SEED = 20261111
DEFAULT_ALPHA = 0.05


def _arrays(
    num: Sequence[int | float], den: Sequence[int | float]
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError("num과 den의 길이가 다릅니다.")
    if numerator.ndim != 1:
        raise ValueError("num과 den은 1차원이어야 합니다.")
    return numerator, denominator


def _percentile(values: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def group_bootstrap_ci(
    num: Sequence[int | float],
    den: Sequence[int | float],
    *,
    b: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """이미지별 (분자, 분모) 배열의 비율과 percentile CI.

    분모가 0인 이미지는 제외한다. 남은 이미지가 0장이면 값은 None이다.
    """

    numerator, denominator = _arrays(num, den)
    keep = denominator > 0
    numerator = numerator[keep]
    denominator = denominator[keep]
    n = int(numerator.size)
    if n == 0:
        return {
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "numerator": 0.0,
            "denominator": 0.0,
            "b": b,
            "seed": seed,
            "alpha": alpha,
        }
    point = float(numerator.sum() / denominator.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    sampled_num = numerator[idx].sum(axis=1)
    sampled_den = denominator[idx].sum(axis=1)
    ratios = sampled_num / sampled_den
    low, high = _percentile(ratios, alpha)
    return {
        "value": point,
        "ci_low": low,
        "ci_high": high,
        "n": n,
        "numerator": float(numerator.sum()),
        "denominator": float(denominator.sum()),
        "b": b,
        "seed": seed,
        "alpha": alpha,
    }


def paired_bootstrap_delta(
    num_a: Sequence[int | float],
    den_a: Sequence[int | float],
    num_b: Sequence[int | float],
    den_b: Sequence[int | float],
    *,
    b: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """같은 이미지 인덱스로 A−B 비율 차이의 percentile CI.

    두 배열은 같은 길이·같은 이미지 순서여야 한다. 어느 한쪽이라도
    분모가 0인 이미지는 제외한다.
    """

    a_num, a_den = _arrays(num_a, den_a)
    b_num, b_den = _arrays(num_b, den_b)
    if a_num.shape != b_num.shape:
        raise ValueError("A와 B의 이미지 수가 다릅니다.")
    keep = (a_den > 0) & (b_den > 0)
    a_num, a_den = a_num[keep], a_den[keep]
    b_num, b_den = b_num[keep], b_den[keep]
    n = int(a_num.size)
    if n == 0:
        return {
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "b": b,
            "seed": seed,
            "alpha": alpha,
        }
    a_point = float(a_num.sum() / a_den.sum())
    b_point = float(b_num.sum() / b_den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    ratio_a = a_num[idx].sum(axis=1) / a_den[idx].sum(axis=1)
    ratio_b = b_num[idx].sum(axis=1) / b_den[idx].sum(axis=1)
    deltas = ratio_a - ratio_b
    low, high = _percentile(deltas, alpha)
    return {
        "delta": a_point - b_point,
        "a_value": a_point,
        "b_value": b_point,
        "ci_low": low,
        "ci_high": high,
        "n": n,
        "b": b,
        "seed": seed,
        "alpha": alpha,
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP_B",
    "DEFAULT_BOOTSTRAP_SEED",
    "group_bootstrap_ci",
    "paired_bootstrap_delta",
]
