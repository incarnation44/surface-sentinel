"""이미지 지표(§4.2)와 영역 지표(§4.3).

규칙: **모든 비율은 분모를 함께 갖는다.** ``Ratio``는 분자·분모·값·Wilson
95% 구간·정의를 한 묶음으로 돌려주며, 분모가 0이면 값을 None으로 두고
분모를 0으로 보고한다(0으로 나눈 값을 만들지 않는다).

Wilson 95% 구간(연속성 보정 없음, z=1.96)::

    n = 분모, p = k/n, z = 1.959963984540054
    denom  = 1 + z²/n
    center = (p + z²/(2n)) / denom
    half   = z * sqrt(p(1-p)/n + z²/(4n²)) / denom
    구간   = (center - half, center + half)  # 0..1로 자름

n = 0이면 구간을 정의하지 않는다(0, 1을 보고하지 않는다).

**근사 한계(감사 5.10, 2026-11-17).** Wilson 구간은 "시행이 서로 독립"을
가정한다. 이 모듈에서 그 가정이 성립하는 것은 **이미지 단위 비율**(escape,
false N-G, review 등 — 이미지 하나가 시행 하나)까지다. 다음 두 곳은
독립 가정이 깨지므로 Wilson 값은 **근사 구간**이며, 정확한 95% CI가 필요하면
``evaluation.bootstrap``의 이미지(그룹) 단위 부트스트랩을 함께 쓴다.

* ``balanced_accuracy`` — 두 성분 Wilson 구간의 평균은 평균 지표의 CI가
  아니다(성분 간 상관·비대칭 표본 수를 반영하지 못한다). 정의 문자열도
  "보수적 근사"라고 적혀 있고, 그 문자열과 값은 하위 호환을 위해 그대로 둔다.
* ``pixel_precision``/``pixel_recall`` — 픽셀을 독립 시행으로 세지만 같은
  이미지 안 픽셀은 공간 상관이 있다. 픽셀 수가 크면 구간이 과도하게 좁아진다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Z_95 = 1.959963984540054

DECIDED_STATUSES = ("OK", "NG")
SCORED_STATUSES = ("NG", "REVIEW")
ALL_STATUSES = ("OK", "NG", "REVIEW", "ERROR")


@dataclass(frozen=True)
class Ratio:
    """분모가 명시된 비율. 분모가 0이면 ``value``/구간은 None이다."""

    definition: str
    numerator: float | None
    denominator: float | None
    value: float | None
    ci_low: float | None
    ci_high: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
        }

    def formatted(self) -> str:
        if self.value is None:
            denominator = "-" if self.denominator is None else f"{self.denominator:g}"
            return f"- (분모 {denominator})"
        text = f"{self.value:.4f}"
        if self.ci_low is not None and self.ci_high is not None:
            text += f" [{self.ci_low:.3f}, {self.ci_high:.3f}]"
        denominator = "-" if self.denominator is None else f"{self.denominator:g}"
        return f"{text} (분모 {denominator})"


def wilson_interval(
    successes: float, trials: float, z: float = Z_95
) -> tuple[float | None, float | None]:
    """Wilson 95% 구간. trials가 0이면 (None, None)이다."""

    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1.0 + (z * z) / trials
    center = (proportion + (z * z) / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + (z * z) / (4.0 * trials * trials))
        / denominator
    )
    low = min(max(center - half, 0.0), 1.0)
    high = min(max(center + half, 0.0), 1.0)
    return low, high


def ratio(
    definition: str,
    numerator: float,
    denominator: float,
    *,
    ci: bool = True,
) -> Ratio:
    """비율 하나를 만든다. 분모가 0이면 값과 구간을 None으로 둔다."""

    if denominator <= 0:
        return Ratio(definition, numerator, denominator, None, None, None)
    value = numerator / denominator
    low, high = wilson_interval(numerator, denominator) if ci else (None, None)
    return Ratio(definition, numerator, denominator, value, low, high)


def derived_ratio(
    definition: str,
    value: float | None,
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> Ratio:
    """단일 분모가 없는 파생 지표(예: balanced accuracy)."""

    return Ratio(definition, None, None, value, ci_low, ci_high)


@dataclass(frozen=True)
class Confusion:
    """이미지 단위 혼동행렬. ``ERROR``는 판정이 아니므로 따로 센다."""

    counts: Mapping[str, Mapping[str, int]]

    @staticmethod
    def from_pairs(pairs: Iterable[tuple[str, str]]) -> Confusion:
        counts: dict[str, dict[str, int]] = {}
        for label, status in pairs:
            bucket = counts.setdefault(label, dict.fromkeys(ALL_STATUSES, 0))
            if status not in bucket:
                raise ValueError(f"알 수 없는 상태입니다: {status!r}")
            bucket[status] += 1
        return Confusion(counts)

    def count(self, label: str, status: str) -> int:
        return int(self.counts.get(label, {}).get(status, 0))

    def total(self, label: str) -> int:
        return sum(int(value) for value in self.counts.get(label, {}).values())

    def evaluated(self, label: str) -> int:
        """분모로 쓰는 수: ERROR(판정 아님)를 제외한 이미지 수."""

        return self.total(label) - self.count(label, "ERROR")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for label in sorted(self.counts):
            bucket = {status: self.count(label, status) for status in ALL_STATUSES}
            payload[label] = {
                "total": self.total(label),
                "evaluated": self.evaluated(label),
                "status": bucket,
            }
        return payload


def image_metrics(confusion: Confusion) -> dict[str, Ratio]:
    """§4.2의 이미지 지표 9개. P=결함 이미지 수, N=정상 이미지 수.

    이미지 비율(escape, false N-G, review, automation, decided accuracy)의
    Wilson 구간은 이미지 하나를 시행 하나로 보므로 그대로 쓸 수 있다.

    ``balanced_accuracy``만 예외다. 여기서 내는 구간은 **두 성분 Wilson
    구간의 평균**이라 평균 지표의 엄밀한 95% CI가 아니다(근사). 값·정의
    문자열·구간은 하위 호환을 위해 동결되어 있으므로 바꾸지 않는다.
    정확한 CI가 필요하면
    :func:`surface_sentinel.evaluation.bootstrap.balanced_accuracy_bootstrap_ci`
    를 ``Confusion``의 ``(label, status)`` 쌍에 적용한다.
    """

    p = confusion.evaluated("defect")
    n = confusion.evaluated("good")
    t = p + n
    ok_defect = confusion.count("defect", "OK")
    ng_defect = confusion.count("defect", "NG")
    review_defect = confusion.count("defect", "REVIEW")
    ng_good = confusion.count("good", "NG")
    ok_good = confusion.count("good", "OK")
    review_good = confusion.count("good", "REVIEW")

    escape = ratio("OK_on_defect / P (P=결함 이미지 수)", ok_defect, p)
    defect_ng = ratio("NG_on_defect / P", ng_defect, p)
    defect_review = ratio("REVIEW_on_defect / P", review_defect, p)
    false_ng = ratio("NG_on_good / N (N=정상 이미지 수)", ng_good, n)
    good_review = ratio("REVIEW_on_good / N", review_good, n)
    review = ratio("(REVIEW_defect+REVIEW_good) / T", review_defect + review_good, t)
    automation = ratio("(OK+NG) / T", (ok_defect + ng_defect + ok_good + ng_good), t)
    decided = ok_defect + ng_defect + ok_good + ng_good
    decided_accuracy = ratio("(NG_defect+OK_good) / (OK+NG)", ng_defect + ok_good, decided)

    ok_good_ratio = ratio("OK_on_good / N", ok_good, n)
    balanced_value = (
        (defect_ng.value + ok_good_ratio.value) / 2.0
        if defect_ng.value is not None and ok_good_ratio.value is not None
        else None
    )
    component_interval = _average_interval(defect_ng, ok_good_ratio)
    balanced = derived_ratio(
        "(NG_defect/P + OK_good/N) / 2 — 단일 분모 없음, "
        "구간은 두 성분 Wilson 구간의 평균(보수적 근사)",
        balanced_value,
        ci_low=component_interval[0],
        ci_high=component_interval[1],
    )

    return {
        "defect_escape_rate": escape,
        "defect_ng_rate": defect_ng,
        "defect_review_rate": defect_review,
        "good_false_ng_rate": false_ng,
        "good_review_rate": good_review,
        "review_rate": review,
        "automation_rate": automation,
        "decided_accuracy": decided_accuracy,
        "balanced_accuracy": balanced,
    }


def _average_interval(left: Ratio, right: Ratio) -> tuple[float | None, float | None]:
    if left.ci_low is None or right.ci_low is None:
        return None, None
    return (left.ci_low + right.ci_low) / 2.0, (left.ci_high + right.ci_high) / 2.0


@dataclass(frozen=True)
class GtMatch:
    """GT 컴포넌트 하나와 예측 컴포넌트의 최대 IoU."""

    iou: float
    image_status: str


@dataclass(frozen=True)
class PixelCounts:
    """영역 픽셀 혼동. 합산값으로 precision/recall을 만든다."""

    gt_pixels: int = 0
    pred_pixels: int = 0
    tp_pixels: int = 0

    def merged(self, other: PixelCounts) -> PixelCounts:
        return PixelCounts(
            gt_pixels=self.gt_pixels + other.gt_pixels,
            pred_pixels=self.pred_pixels + other.pred_pixels,
            tp_pixels=self.tp_pixels + other.tp_pixels,
        )


IOU_MATCH_RECALL = 0.10
IOU_MATCH_STRICT = 0.30
IOU_MATCH_FULL = 0.50

#: IoU 분포 히스토그램 경계. 각 구간은 [lo, hi), 마지막 구간만 [lo, hi]다.
IOU_HISTOGRAM_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.0),
)


def recall_at(matches: Sequence[GtMatch], threshold: float) -> Ratio:
    """주어진 IoU 임계값 이상으로 매칭된 GT의 비율(분모=전체 GT)."""

    matched = sum(1 for match in matches if match.iou >= threshold)
    return ratio(
        f"matched_GT(IoU>={threshold:.2f}) / 전체 GT",
        matched,
        len(matches),
    )


def _quantile(values: Sequence[float], q: float) -> float | None:
    """정렬된 값의 선형 보간 분위수. 값이 없으면 None."""

    if not values:
        return None
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * weight)


def iou_distribution(matches: Sequence[GtMatch]) -> dict[str, Any]:
    """GT별 best IoU의 요약(평균·분위수·히스토그램). 결정론, RNG 없음."""

    values = sorted(float(match.iou) for match in matches)
    count = len(values)
    mean = sum(values) / count if count else None
    histogram: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(IOU_HISTOGRAM_BINS):
        last = index == len(IOU_HISTOGRAM_BINS) - 1
        if last:
            hits = sum(1 for value in values if low <= value <= high)
        else:
            hits = sum(1 for value in values if low <= value < high)
        histogram.append({"lo": low, "hi": high, "count": hits})
    return {
        "count": count,
        "mean": mean,
        "quantiles": {
            "p10": _quantile(values, 0.10),
            "p25": _quantile(values, 0.25),
            "p50": _quantile(values, 0.50),
            "p75": _quantile(values, 0.75),
            "p90": _quantile(values, 0.90),
        },
        "histogram": histogram,
    }



def localization_metrics(
    matches: Sequence[GtMatch],
    pixels: PixelCounts,
) -> dict[str, Ratio]:
    """§4.3 영역 지표. 분모 두 개(전체 GT / NG·REVIEW 이미지 안 GT)를 모두 낸다.

    ``pixel_precision``/``pixel_recall``의 Wilson 구간은 **픽셀을 독립
    시행으로 센 근사**다. 같은 이미지 안 픽셀은 공간 상관이 있어 실제
    불확실성보다 좁거나 넓게 나올 수 있다. 정의 문자열·반환값은 하위 호환을
    위해 그대로 두고, 이미지 단위 CI가 필요하면
    :func:`surface_sentinel.evaluation.bootstrap.pixel_ratio_bootstrap_ci`
    를 이미지별 ``PixelCounts``에 적용한다.
    """

    all_gt = len(matches)
    decided_gt = sum(1 for match in matches if match.image_status in SCORED_STATUSES)
    matched_loose = [match for match in matches if match.iou >= IOU_MATCH_RECALL]
    matched_strict = [match for match in matches if match.iou >= IOU_MATCH_STRICT]
    matched_decided = [match for match in matched_loose if match.image_status in SCORED_STATUSES]
    iou_values = [match.iou for match in matched_loose]

    mean_iou = sum(iou_values) / len(iou_values) if iou_values else None
    return {
        "localization_recall_iou0.10": ratio(
            "matched_GT(IoU>=0.10) / 전체 GT", len(matched_loose), all_gt
        ),
        "localization_recall_iou0.10_decided": ratio(
            "matched_GT(IoU>=0.10) / NG·REVIEW 판정 이미지 안 GT",
            len(matched_decided),
            decided_gt,
        ),
        "localization_recall_iou0.30": ratio(
            "matched_GT(IoU>=0.30) / 전체 GT", len(matched_strict), all_gt
        ),
        "localization_recall_iou0.30_decided": ratio(
            "matched_GT(IoU>=0.30) / NG·REVIEW 판정 이미지 안 GT",
            sum(1 for match in matched_strict if match.image_status in SCORED_STATUSES),
            decided_gt,
        ),
        "pixel_precision": ratio(
            "TP_pixels / predicted_pixels", pixels.tp_pixels, pixels.pred_pixels
        ),
        "pixel_recall": ratio("TP_pixels / GT_pixels", pixels.tp_pixels, pixels.gt_pixels),
        "mean_iou_matched": Ratio(
            "matched(IoU>=0.10) 쌍의 평균 IoU — 비율이 아니므로 분모는 쌍 수",
            None,
            len(iou_values),
            mean_iou,
            None,
            None,
        ),
    }


def metrics_to_dict(metrics: Mapping[str, Ratio]) -> dict[str, Any]:
    return {name: value.as_dict() for name, value in metrics.items()}


__all__ = [
    "ALL_STATUSES",
    "Confusion",
    "DECIDED_STATUSES",
    "GtMatch",
    "IOU_MATCH_FULL",
    "IOU_MATCH_RECALL",
    "IOU_MATCH_STRICT",
    "IOU_HISTOGRAM_BINS",
    "PixelCounts",
    "Ratio",
    "SCORED_STATUSES",
    "Z_95",
    "derived_ratio",
    "image_metrics",
    "iou_distribution",
    "localization_metrics",
    "metrics_to_dict",
    "ratio",
    "recall_at",
    "wilson_interval",
]
