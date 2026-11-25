"""이미지 단위 그룹·짝 부트스트랩 95% 신뢰구간(CI) 계산 모듈.

비율의 분모가 이미지마다 다른 경우(예: GT 컴포넌트 수)를 다루기 위해
**이미지 단위로 복원추출**한다. 같은 seed면 같은 결과가 나온다.
분모가 0인 이미지는 계산에서 제외한다.

추가(2026-11-17, 감사 5.10): 이미지(또는 그룹) 단위 percentile CI 두 개.

* :func:`balanced_accuracy_bootstrap_ci` — balanced accuracy의 이미지 단위 CI.
  ``metrics.image_metrics``의 Wilson 평균 근사와 달리 **평균 지표 자체**를
  리샘플링하므로 두 성분의 상관·비대칭 표본 수를 반영한다.
* :func:`pixel_ratio_bootstrap_ci` — 픽셀 precision/recall의 이미지 단위 CI.
  같은 이미지 안 픽셀은 공간 상관이 있으므로 픽셀을 독립 시행으로 보는
  Wilson 구간은 근사다. 여기서는 **이미지(그룹)를 복원추출**한다.

두 함수 모두 ``random.Random(seed)``만 쓰므로 같은 seed면 같은 결과다.
한쪽 클래스(또는 분자/분모)가 0인 리샘플은 건너뛰고, 쓸 수 있는 리샘플이
하나도 없으면 구간은 ``None``이다(예외를 던지지 않는다).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import ALL_STATUSES, PixelCounts

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


def _unit_counts(
    rows: Sequence[tuple[int, int, int, int]],
    group_ids: Sequence[str] | None,
) -> list[tuple[int, int, int, int]]:
    """이미 (P, NG_defect, N, OK_good)로 접힌 행들을 그룹 단위로 합산한다."""

    if group_ids is None:
        return list(rows)
    if len(group_ids) != len(rows):
        raise ValueError("group_ids 길이가 입력 길이와 다릅니다.")
    merged: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        bucket = merged.setdefault(str(group_ids[index]), [0, 0, 0, 0])
        for position, value in enumerate(row):
            bucket[position] += value
    return [(bucket[0], bucket[1], bucket[2], bucket[3]) for bucket in merged.values()]


def _balanced_row(label: str, status: str) -> tuple[int, int, int, int]:
    """이미지 하나를 (P, NG_defect, N, OK_good) 기여로 바꾼다. ERROR는 분모 제외."""

    if status not in ALL_STATUSES:
        raise ValueError(f"알 수 없는 상태입니다: {status!r}")
    if label not in ("defect", "good"):
        raise ValueError(f"알 수 없는 라벨입니다: {label!r}")
    if label == "defect":
        evaluated = 0 if status == "ERROR" else 1
        return (evaluated, 1 if status == "NG" else 0, 0, 0)
    evaluated = 0 if status == "ERROR" else 1
    return (0, 0, evaluated, 1 if status == "OK" else 0)


def _resample_balanced(
    units: Sequence[tuple[int, int, int, int]],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    """이미지/그룹 단위 복원추출로 balanced accuracy 표본을 만든다."""

    rng = random.Random(seed)
    count = len(units)
    estimates: list[float] = []
    for _ in range(replicates):
        defect_total = defect_ng = good_total = good_ok = 0
        for index in rng.choices(range(count), k=count):
            unit = units[index]
            defect_total += unit[0]
            defect_ng += unit[1]
            good_total += unit[2]
            good_ok += unit[3]
        # 한쪽 클래스가 0인 리샘플은 balanced accuracy가 정의되지 않는다 → 건너뛴다.
        if defect_total == 0 or good_total == 0:
            continue
        estimates.append(((defect_ng / defect_total) + (good_ok / good_total)) / 2.0)
    return estimates


def balanced_accuracy_bootstrap_ci(
    pairs: Sequence[tuple[str, str]],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_B,
    group_ids: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """balanced accuracy의 이미지(그룹) 단위 percentile CI.

    ``(label, status)`` 쌍을 이미지 하나로 보고 복원추출한다. 점추정값은
    ``(NG_defect/P + OK_good/N) / 2``이고 ``ERROR``는 분모에서 제외한다
    (``metrics.Confusion.evaluated``와 같은 규칙).

    ``group_ids``를 주면 같은 그룹의 이미지가 함께 뽑히는 **그룹 단위**로
    리샘플한다(예: 같은 원본 이미지에서 나온 증강·크롭). 그러면 리샘플 단위
    수는 그룹 수가 된다.

    퇴화 규칙(예외 없이 ``None``):

    * 입력이 비었거나 ``P``/``N`` 중 하나가 0이면 점추정값·구간 모두 ``None``.
    * 리샘플 중 한쪽 클래스 분모가 0인 것은 건너뛰고 ``valid_b``에 남긴다.
      유효한 리샘플이 0개면 구간만 ``None``이다.

    같은 ``seed``면 같은 결과다(``random.Random``).
    """

    rows = [_balanced_row(label, status) for label, status in pairs]
    units = _unit_counts(rows, group_ids)
    defect_total = sum(unit[0] for unit in units)
    defect_ng = sum(unit[1] for unit in units)
    good_total = sum(unit[2] for unit in units)
    good_ok = sum(unit[3] for unit in units)
    payload: dict[str, Any] = {
        "value": None,
        "ci_low": None,
        "ci_high": None,
        "n": len(units),
        "images": len(rows),
        "unit": "image" if group_ids is None else "group",
        "defect_evaluated": defect_total,
        "good_evaluated": good_total,
        "b": replicates,
        "valid_b": 0,
        "skipped_b": 0,
        "seed": seed,
        "alpha": alpha,
    }
    if defect_total == 0 or good_total == 0:
        payload["skipped_b"] = replicates
        return payload
    payload["value"] = ((defect_ng / defect_total) + (good_ok / good_total)) / 2.0
    estimates = _resample_balanced(units, seed=seed, replicates=replicates)
    payload["valid_b"] = len(estimates)
    payload["skipped_b"] = replicates - len(estimates)
    if estimates:
        low, high = _percentile(np.asarray(estimates, dtype=np.float64), alpha)
        payload["ci_low"] = low
        payload["ci_high"] = high
    return payload


def pixel_ratio_bootstrap_ci(
    per_image_counts: Sequence[PixelCounts],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_B,
    group_ids: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """픽셀 precision/recall의 이미지(그룹) 단위 percentile CI.

    이미지별 :class:`~surface_sentinel.evaluation.metrics.PixelCounts`를
    복원추출해 ``tp/pred``(precision)와 ``tp/gt``(recall)를 다시 합산한다.
    같은 이미지 안 픽셀은 공간 상관이 있어 픽셀 단위 Wilson 구간은 근사일
    뿐인데, 이 함수는 **이미지를 독립 단위로** 본다.

    ``group_ids``를 주면 그룹 단위로 리샘플한다(``n``은 그룹 수).

    퇴화 규칙(예외 없이 ``None``): 입력이 비었거나 전체 분모가 0이면 해당
    지표의 값·구간이 ``None``이다. 리샘플 중 분모가 0인 것은 건너뛰고
    ``valid_b``에 남긴다.
    """

    counts = [
        PixelCounts(int(c.gt_pixels), int(c.pred_pixels), int(c.tp_pixels))
        for c in per_image_counts
    ]
    if group_ids is not None and len(group_ids) != len(counts):
        raise ValueError("group_ids 길이가 입력 길이와 다릅니다.")
    units: list[PixelCounts] = []
    if group_ids is None:
        units = counts
    else:
        merged: dict[str, PixelCounts] = {}
        for index, count in enumerate(counts):
            key = str(group_ids[index])
            merged[key] = merged[key].merged(count) if key in merged else count
        units = list(merged.values())

    gt_total = sum(unit.gt_pixels for unit in units)
    pred_total = sum(unit.pred_pixels for unit in units)
    tp_total = sum(unit.tp_pixels for unit in units)

    def _entry(numerator: float, denominator: float) -> dict[str, Any]:
        return {
            "value": (numerator / denominator) if denominator > 0 else None,
            "ci_low": None,
            "ci_high": None,
            "numerator": numerator,
            "denominator": denominator,
        }

    payload: dict[str, Any] = {
        "precision": _entry(tp_total, pred_total),
        "recall": _entry(tp_total, gt_total),
        "n": len(units),
        "images": len(counts),
        "unit": "image" if group_ids is None else "group",
        "b": replicates,
        "valid_b": {"precision": 0, "recall": 0},
        "seed": seed,
        "alpha": alpha,
    }
    if not units:
        return payload

    rng = random.Random(seed)
    count = len(units)
    precision_samples: list[float] = []
    recall_samples: list[float] = []
    for _ in range(replicates):
        gt = pred = tp = 0
        for index in rng.choices(range(count), k=count):
            unit = units[index]
            gt += unit.gt_pixels
            pred += unit.pred_pixels
            tp += unit.tp_pixels
        if pred > 0:
            precision_samples.append(tp / pred)
        if gt > 0:
            recall_samples.append(tp / gt)
    for name, samples in (("precision", precision_samples), ("recall", recall_samples)):
        payload["valid_b"][name] = len(samples)
        if samples:
            low, high = _percentile(np.asarray(samples, dtype=np.float64), alpha)
            payload[name]["ci_low"] = low
            payload[name]["ci_high"] = high
    return payload


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP_B",
    "DEFAULT_BOOTSTRAP_SEED",
    "balanced_accuracy_bootstrap_ci",
    "group_bootstrap_ci",
    "paired_bootstrap_delta",
    "pixel_ratio_bootstrap_ci",
]
