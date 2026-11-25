"""그룹 단위 balanced accuracy·픽셀 비율 부트스트랩 CI 단위 테스트 — 결정론과 퇴화 경계 검증.

기존 Wilson 근사값은 바꾸지 않고, 새 함수의 결정론·퇴화 규칙·리샘플 단위를
확인한다. 봉인 split은 쓰지 않는다(합성 데이터만).
"""

from __future__ import annotations

import pytest

from surface_sentinel.evaluation.bootstrap import (
    balanced_accuracy_bootstrap_ci,
    pixel_ratio_bootstrap_ci,
)
from surface_sentinel.evaluation.metrics import (
    Confusion,
    PixelCounts,
    image_metrics,
    localization_metrics,
)


def _legacy_balanced(pairs: list[tuple[str, str]]):
    return image_metrics(Confusion.from_pairs(pairs))["balanced_accuracy"]


def _width(interval: tuple[float | None, float | None]) -> float:
    low, high = interval
    assert low is not None and high is not None
    return high - low


# ------------------------------------------------------ balanced accuracy CI


def test_balanced_bootstrap_is_deterministic() -> None:
    pairs = [("defect", "NG")] * 8 + [("defect", "OK")] * 2
    pairs += [("good", "OK")] * 8 + [("good", "NG")] * 2
    first = balanced_accuracy_bootstrap_ci(pairs, seed=20261117, replicates=200)
    second = balanced_accuracy_bootstrap_ci(pairs, seed=20261117, replicates=200)
    assert first == second
    assert first["value"] == pytest.approx(0.8)
    assert first["n"] == 20
    assert first["unit"] == "image"
    assert first["valid_b"] == 200


def test_balanced_bootstrap_point_equals_frozen_metric_value() -> None:
    """점추정값은 기존 ``image_metrics`` 값과 같아야 한다(정의 불변)."""

    pairs = [("defect", "NG")] * 4 + [("defect", "OK")]
    pairs += [("good", "OK")] * 190 + [("good", "NG")] * 10
    legacy = _legacy_balanced(pairs)
    payload = balanced_accuracy_bootstrap_ci(pairs, seed=20261117, replicates=300)
    assert payload["value"] == pytest.approx(legacy.value)
    assert payload["value"] == pytest.approx(0.875)
    assert payload["ci_low"] is not None and payload["ci_high"] is not None
    assert payload["ci_low"] < payload["value"] < payload["ci_high"]


def test_balanced_bootstrap_empty_and_single_class_are_none() -> None:
    empty = balanced_accuracy_bootstrap_ci([], seed=7)
    assert empty["value"] is None
    assert empty["ci_low"] is None and empty["ci_high"] is None
    assert empty["n"] == 0
    assert empty["valid_b"] == 0

    single = balanced_accuracy_bootstrap_ci([("defect", "NG")] * 5, seed=7)
    assert single["value"] is None
    assert single["ci_low"] is None and single["ci_high"] is None
    assert single["good_evaluated"] == 0
    assert single["skipped_b"] == single["b"]


def test_balanced_bootstrap_excludes_error_from_denominator() -> None:
    pairs = [("defect", "NG"), ("defect", "ERROR"), ("good", "OK"), ("good", "ERROR")]
    payload = balanced_accuracy_bootstrap_ci(pairs, seed=7, replicates=100)
    assert payload["defect_evaluated"] == 1
    assert payload["good_evaluated"] == 1
    assert payload["value"] == pytest.approx(1.0)
    # ERROR 이미지는 다시 뽑히면서 분모를 흔들 수 없어야 한다 → CI가 1.0에 붙는다.
    assert payload["ci_low"] == pytest.approx(1.0)
    assert payload["ci_high"] == pytest.approx(1.0)


def test_balanced_bootstrap_rejects_unknown_status_and_label() -> None:
    with pytest.raises(ValueError):
        balanced_accuracy_bootstrap_ci([("defect", "PASS")], seed=7)
    with pytest.raises(ValueError):
        balanced_accuracy_bootstrap_ci([("unknown", "OK")], seed=7)


def test_balanced_bootstrap_group_resampling_is_coarser() -> None:
    """그룹 단위 리샘플은 리샘플 단위 수가 줄고 구간이 넓어진다."""

    pairs: list[tuple[str, str]] = []
    groups: list[str] = []
    for group in range(20):
        verdict = "NG" if group % 2 == 0 else "OK"
        for _ in range(10):
            pairs.append(("defect", verdict))
            groups.append(f"d{group:02d}")
    for group in range(20):
        verdict = "OK" if group % 2 == 0 else "NG"
        for _ in range(10):
            pairs.append(("good", verdict))
            groups.append(f"g{group:02d}")

    image = balanced_accuracy_bootstrap_ci(pairs, seed=7, replicates=200)
    group = balanced_accuracy_bootstrap_ci(pairs, seed=7, replicates=200, group_ids=groups)
    assert image["unit"] == "image" and image["n"] == 400
    assert group["unit"] == "group" and group["n"] == 40
    assert group["value"] == pytest.approx(image["value"])
    image_width = _width((image["ci_low"], image["ci_high"]))
    group_width = _width((group["ci_low"], group["ci_high"]))
    assert group_width > 2.0 * image_width


def test_balanced_bootstrap_differs_from_wilson_average() -> None:
    """결함 쪽 ERROR가 많으면 Wilson 평균 구간은 과하게 넓다(근사 한계)."""

    pairs = [("defect", "NG")] * 20 + [("defect", "ERROR")] * 40
    pairs += [("good", "OK")] * 150 + [("good", "NG")] * 10
    legacy = _legacy_balanced(pairs)
    payload = balanced_accuracy_bootstrap_ci(pairs, seed=20261117, replicates=400)
    legacy_width = _width((legacy.ci_low, legacy.ci_high))
    boot_width = _width((payload["ci_low"], payload["ci_high"]))
    assert payload["value"] == pytest.approx(legacy.value)
    assert boot_width < legacy_width


def test_balanced_bootstrap_group_ids_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        balanced_accuracy_bootstrap_ci([("defect", "NG"), ("good", "OK")], seed=1, group_ids=["a"])


# ------------------------------------------------------------ pixel 비율 CI


def _pixel_counts() -> list[PixelCounts]:
    return [
        PixelCounts(gt_pixels=1000, pred_pixels=1000, tp_pixels=600 + 40 * index)
        for index in range(10)
    ]


def _legacy_pixel() -> tuple[dict[str, float | None], dict[str, float | None]]:
    merged = PixelCounts()
    for count in _pixel_counts():
        merged = merged.merged(count)
    metrics = localization_metrics([], merged)
    return metrics["pixel_precision"].as_dict(), metrics["pixel_recall"].as_dict()


def test_pixel_bootstrap_is_deterministic_and_point_matches_legacy() -> None:
    first = pixel_ratio_bootstrap_ci(_pixel_counts(), seed=20261117, replicates=300)
    second = pixel_ratio_bootstrap_ci(_pixel_counts(), seed=20261117, replicates=300)
    assert first == second
    legacy_precision, legacy_recall = _legacy_pixel()
    assert first["precision"]["value"] == pytest.approx(legacy_precision["value"])
    assert first["recall"]["value"] == pytest.approx(legacy_recall["value"])
    assert first["n"] == 10 and first["images"] == 10
    assert first["valid_b"] == {"precision": 300, "recall": 300}


def test_pixel_bootstrap_is_much_wider_than_pixel_wilson() -> None:
    """픽셀을 독립 시행으로 센 Wilson 구간은 공간 상관을 무시해 너무 좁다."""

    legacy_precision, _ = _legacy_pixel()
    payload = pixel_ratio_bootstrap_ci(_pixel_counts(), seed=20261117, replicates=400)
    wilson_width = _width((legacy_precision["ci95_low"], legacy_precision["ci95_high"]))
    boot_width = _width((payload["precision"]["ci_low"], payload["precision"]["ci_high"]))
    assert wilson_width == pytest.approx(0.0162, abs=5e-4)
    assert boot_width > 5.0 * wilson_width
    assert payload["precision"]["ci_low"] < payload["precision"]["value"]
    assert payload["precision"]["value"] < payload["precision"]["ci_high"]


def test_pixel_bootstrap_empty_and_zero_denominator_are_none() -> None:
    empty = pixel_ratio_bootstrap_ci([], seed=1)
    assert empty["n"] == 0
    assert empty["precision"]["value"] is None
    assert empty["recall"]["value"] is None
    assert empty["precision"]["ci_low"] is None and empty["recall"]["ci_low"] is None

    only_gt_counts = [PixelCounts(gt_pixels=10, pred_pixels=0, tp_pixels=0)]
    only_gt = pixel_ratio_bootstrap_ci(only_gt_counts, seed=1)
    assert only_gt["precision"]["value"] is None
    assert only_gt["precision"]["denominator"] == 0
    assert only_gt["recall"]["value"] == pytest.approx(0.0)
    assert only_gt["valid_b"]["precision"] == 0


def test_pixel_bootstrap_group_resampling_is_coarser() -> None:
    counts = _pixel_counts()
    groups = [f"g{index // 2}" for index in range(len(counts))]
    image = pixel_ratio_bootstrap_ci(counts, seed=20261117, replicates=300)
    group = pixel_ratio_bootstrap_ci(counts, seed=20261117, replicates=300, group_ids=groups)
    assert group["unit"] == "group" and group["n"] == 5
    assert group["precision"]["value"] == pytest.approx(image["precision"]["value"])
    image_width = _width((image["precision"]["ci_low"], image["precision"]["ci_high"]))
    group_width = _width((group["precision"]["ci_low"], group["precision"]["ci_high"]))
    assert group_width > image_width


def test_pixel_bootstrap_group_ids_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        pixel_ratio_bootstrap_ci(_pixel_counts(), seed=1, group_ids=["only-one"])
