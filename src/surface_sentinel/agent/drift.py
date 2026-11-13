"""Phase 3 PSI 분포 드리프트 감시기 (observe-only, 경고 전용).

기준선(tune 2,001장 또는 기준 배치)과 대상 배치의 점수·화질(밝기·블러)·상태 분포
이동을 Population Stability Index(PSI)로 정량 감지한다.

계약:
* 자동 재학습, 임계값 자동 변경, 모델 자동 채택을 절대 수행하지 않는다.
* 소표본(``target_total < policy.min_samples``)에서는 억지 경보를 내지 않고
  ``INSUFFICIENT_SAMPLE`` 을 반환한다.
* 상태 비율은 모두 ``evaluation.metrics.ratio`` 를 통해 분모와 Wilson 95% CI 를 포함한다.
* 외부 네트워크·셸·subprocess·디스크 쓰기를 사용하지 않는다.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ..contracts import Status
from ..evaluation.metrics import ratio
from ..records.fingerprint import canonical_json_sha256

DRIFT_POLICY_VERSION = "drift-v1"
DRIFT_SCHEMA_VERSION = "agent-drift-v1"

SCORE_BIN_EDGES: tuple[float, ...] = (0.0, 0.05, 0.20, 0.40, 0.50, 0.80, 1.20, float("inf"))
MEAN_GRAY_BIN_EDGES: tuple[float, ...] = (0.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 256.0)
BLUR_VARIANCE_BIN_EDGES: tuple[float, ...] = (
    0.0,
    20.0,
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
    float("inf"),
)
_QUALITY_MIN_SAMPLES = 20
_STATUS_ORDER: tuple[str, ...] = (
    Status.OK.value,
    Status.NG.value,
    Status.REVIEW.value,
    Status.ERROR.value,
)


class DriftStatus(StrEnum):
    """PSI 기반 분포 이동 경보 등급."""

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    DRIFT_ALERT = "DRIFT_ALERT"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


@dataclass(frozen=True)
class DriftPolicy:
    """PSI 드리프트 판정 정책 상수."""

    version: str = DRIFT_POLICY_VERSION
    min_samples: int = 20
    watch_psi: float = 0.10
    alert_psi: float = 0.25
    eps: float = 1e-4

    def policy_hash(self) -> str:
        return canonical_json_sha256(asdict(self))


DEFAULT_DRIFT_POLICY = DriftPolicy()


@dataclass(frozen=True)
class DriftObservation:
    """드리프트 계산을 위해 레코드에서 추출한 정규화 관측치."""

    status: str
    score: float | None
    mean_gray: float | None = None
    blur_variance: float | None = None


@dataclass(frozen=True)
class FeatureDrift:
    """단일 피처의 PSI 및 평균 변화량 계측 결과."""

    feature: str
    psi: float
    baseline_count: int
    target_count: int
    baseline_mean: float
    target_mean: float
    delta_mean: float
    status: DriftStatus
    bin_edges: tuple[float, ...]
    baseline_ratios: tuple[float, ...]
    target_ratios: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": round(float(self.psi), 6),
            "baseline_count": self.baseline_count,
            "target_count": self.target_count,
            "baseline_mean": round(float(self.baseline_mean), 6),
            "target_mean": round(float(self.target_mean), 6),
            "delta_mean": round(float(self.delta_mean), 6),
            "status": self.status.value,
            "bin_edges": [
                "inf" if math.isinf(edge) else round(float(edge), 6) for edge in self.bin_edges
            ],
            "baseline_ratios": [round(float(r), 6) for r in self.baseline_ratios],
            "target_ratios": [round(float(r), 6) for r in self.target_ratios],
        }


@dataclass(frozen=True)
class DriftReport:
    """배치 분포 드리프트 종합 감시 보고서."""

    status: DriftStatus
    baseline_total: int
    target_total: int
    features: tuple[FeatureDrift, ...]
    status_rates_baseline: dict[str, dict[str, Any]]
    status_rates_target: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]
    policy_version: str
    drift_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DRIFT_SCHEMA_VERSION,
            "status": self.status.value,
            "baseline_total": self.baseline_total,
            "target_total": self.target_total,
            "features": [feat.to_dict() for feat in self.features],
            "status_rates_baseline": self.status_rates_baseline,
            "status_rates_target": self.status_rates_target,
            "warnings": list(self.warnings),
            "policy_version": self.policy_version,
            "drift_hash": self.drift_hash,
        }


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def extract_observations(records: Iterable[Any]) -> tuple[DriftObservation, ...]:
    """``results.jsonl`` dict, ``SampleResult``, ``CachedSample`` 등에서 관측치를 추출한다."""

    observations: list[DriftObservation] = []
    for item in records:
        if isinstance(item, DriftObservation):
            observations.append(item)
            continue
        if isinstance(item, Mapping):
            raw_status = item.get("status", "")
            status_str = (
                raw_status.value if isinstance(raw_status, Status) else str(raw_status or "")
            )
            score_val = (
                None if status_str == Status.ERROR.value else _finite_float(item.get("score"))
            )
            quality = item.get("quality")
            if isinstance(quality, Mapping):
                mean_gray = _finite_float(quality.get("mean_gray"))
                blur_var = _finite_float(quality.get("blur_variance"))
            else:
                mean_gray = _finite_float(item.get("mean_gray"))
                blur_var = _finite_float(item.get("blur_variance"))
            observations.append(
                DriftObservation(
                    status=status_str,
                    score=score_val,
                    mean_gray=mean_gray,
                    blur_variance=blur_var,
                )
            )
            continue

        raw_status = getattr(item, "status", "")
        status_str = raw_status.value if isinstance(raw_status, Status) else str(raw_status or "")
        score_val = (
            None
            if status_str == Status.ERROR.value
            else _finite_float(getattr(item, "score", None))
        )
        quality = getattr(item, "quality", None)
        if quality is not None:
            mean_gray = _finite_float(getattr(quality, "mean_gray", None))
            blur_var = _finite_float(getattr(quality, "blur_variance", None))
        else:
            mean_gray = _finite_float(getattr(item, "mean_gray", None))
            blur_var = _finite_float(getattr(item, "blur_variance", None))
        observations.append(
            DriftObservation(
                status=status_str,
                score=score_val,
                mean_gray=mean_gray,
                blur_variance=blur_var,
            )
        )
    return tuple(observations)


def _bin_counts(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    num_bins = len(edges) - 1
    counts = [0] * num_bins
    for val in values:
        placed = False
        for idx in range(num_bins):
            lo = edges[idx]
            hi = edges[idx + 1]
            is_last = idx == num_bins - 1
            if idx == 0 and val < lo:
                counts[0] += 1
                placed = True
                break
            if (lo <= val <= hi) if is_last else (lo <= val < hi):
                counts[idx] += 1
                placed = True
                break
        if not placed:
            counts[num_bins - 1] += 1
    return counts


def _normalize_smoothed(ratios: Sequence[float], eps: float) -> list[float]:
    smoothed = [r if r > 0.0 else eps for r in ratios]
    total = sum(smoothed)
    if total <= 0.0:
        return [1.0 / len(ratios)] * len(ratios)
    return [val / total for val in smoothed]


def _classify_psi(psi: float, policy: DriftPolicy) -> DriftStatus:
    if psi >= policy.alert_psi:
        return DriftStatus.DRIFT_ALERT
    if psi >= policy.watch_psi:
        return DriftStatus.WATCH
    return DriftStatus.NORMAL


def _compute_feature_drift(
    feature: str,
    baseline_values: Sequence[float],
    target_values: Sequence[float],
    edges: tuple[float, ...],
    policy: DriftPolicy,
    *,
    insufficient: bool,
) -> FeatureDrift:
    n_base = len(baseline_values)
    n_target = len(target_values)
    base_counts = _bin_counts(baseline_values, edges)
    target_counts = _bin_counts(target_values, edges)

    raw_base = [c / float(n_base) for c in base_counts]
    raw_target = [c / float(n_target) for c in target_counts]

    p = _normalize_smoothed(raw_base, policy.eps)
    q = _normalize_smoothed(raw_target, policy.eps)
    psi_raw = sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))
    psi = round(max(0.0, psi_raw), 6)

    base_mean = round(sum(baseline_values) / float(n_base), 6)
    target_mean = round(sum(target_values) / float(n_target), 6)
    delta_mean = round(target_mean - base_mean, 6)
    feat_status = (
        DriftStatus.INSUFFICIENT_SAMPLE if insufficient else _classify_psi(psi, policy)
    )
    return FeatureDrift(
        feature=feature,
        psi=psi,
        baseline_count=n_base,
        target_count=n_target,
        baseline_mean=base_mean,
        target_mean=target_mean,
        delta_mean=delta_mean,
        status=feat_status,
        bin_edges=edges,
        baseline_ratios=tuple(round(r, 6) for r in raw_base),
        target_ratios=tuple(round(r, 6) for r in raw_target),
    )


def _status_rates(
    observations: Sequence[DriftObservation],
    label_prefix: str,
) -> dict[str, dict[str, Any]]:
    total = len(observations)
    counts = Counter(obs.status for obs in observations)
    return {
        status_name: ratio(
            f"{label_prefix} {status_name} 비율",
            counts.get(status_name, 0),
            total,
        ).as_dict()
        for status_name in _STATUS_ORDER
    }


def assess_drift(
    baseline_records: Iterable[Any],
    target_records: Iterable[Any],
    policy: DriftPolicy = DEFAULT_DRIFT_POLICY,
) -> DriftReport:
    """기준선 대비 대상 배치의 PSI 분포 이동을 계측한다(판정·설정 불변)."""

    base_obs = extract_observations(baseline_records)
    target_obs = extract_observations(target_records)
    baseline_total = len(base_obs)
    target_total = len(target_obs)

    rates_base = _status_rates(base_obs, "기준선")
    rates_target = _status_rates(target_obs, "대상")

    insufficient = baseline_total < policy.min_samples or target_total < policy.min_samples
    warnings: list[str] = []
    if target_total < policy.min_samples:
        warnings.append(
            f"대상 표본 수({target_total}건)가 최소 기준({policy.min_samples}건) 미만이어서 "
            "INSUFFICIENT_SAMPLE 로 처리합니다(억지 경보 금지)."
        )
    if baseline_total < policy.min_samples:
        warnings.append(
            f"기준선 표본 수({baseline_total}건)가 최소 기준({policy.min_samples}건) 미만입니다."
        )

    features: list[FeatureDrift] = []

    base_scores = [obs.score for obs in base_obs if obs.score is not None]
    target_scores = [obs.score for obs in target_obs if obs.score is not None]
    if base_scores and target_scores:
        features.append(
            _compute_feature_drift(
                "score",
                base_scores,
                target_scores,
                SCORE_BIN_EDGES,
                policy,
                insufficient=insufficient,
            )
        )

    base_gray = [obs.mean_gray for obs in base_obs if obs.mean_gray is not None]
    target_gray = [obs.mean_gray for obs in target_obs if obs.mean_gray is not None]
    if len(base_gray) >= _QUALITY_MIN_SAMPLES and len(target_gray) >= _QUALITY_MIN_SAMPLES:
        features.append(
            _compute_feature_drift(
                "mean_gray",
                base_gray,
                target_gray,
                MEAN_GRAY_BIN_EDGES,
                policy,
                insufficient=insufficient,
            )
        )

    base_blur = [obs.blur_variance for obs in base_obs if obs.blur_variance is not None]
    target_blur = [obs.blur_variance for obs in target_obs if obs.blur_variance is not None]
    if len(base_blur) >= _QUALITY_MIN_SAMPLES and len(target_blur) >= _QUALITY_MIN_SAMPLES:
        features.append(
            _compute_feature_drift(
                "blur_variance",
                base_blur,
                target_blur,
                BLUR_VARIANCE_BIN_EDGES,
                policy,
                insufficient=insufficient,
            )
        )

    if insufficient or not features:
        overall = DriftStatus.INSUFFICIENT_SAMPLE
        if not features and not insufficient:
            warnings.append("비교 가능한 수치 피처가 없어 INSUFFICIENT_SAMPLE 로 처리합니다.")
    else:
        max_psi = max(feat.psi for feat in features)
        overall = _classify_psi(max_psi, policy)
        for feat in features:
            if feat.status is DriftStatus.DRIFT_ALERT:
                warnings.append(
                    f"[경보] {feat.feature} PSI={feat.psi:.4f} (>= {policy.alert_psi:.2f}), "
                    f"평균 변화 {feat.delta_mean:+.4f}"
                )
            elif feat.status is DriftStatus.WATCH:
                warnings.append(
                    f"[주의] {feat.feature} PSI={feat.psi:.4f} (>= {policy.watch_psi:.2f}), "
                    f"평균 변화 {feat.delta_mean:+.4f}"
                )

    frozen_features = tuple(features)
    frozen_warnings = tuple(warnings)
    payload = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "status": overall.value,
        "baseline_total": baseline_total,
        "target_total": target_total,
        "features": [feat.to_dict() for feat in frozen_features],
        "status_rates_baseline": rates_base,
        "status_rates_target": rates_target,
        "warnings": list(frozen_warnings),
        "policy_version": policy.version,
    }
    return DriftReport(
        status=overall,
        baseline_total=baseline_total,
        target_total=target_total,
        features=frozen_features,
        status_rates_baseline=rates_base,
        status_rates_target=rates_target,
        warnings=frozen_warnings,
        policy_version=policy.version,
        drift_hash=canonical_json_sha256(payload),
    )


__all__ = [
    "BLUR_VARIANCE_BIN_EDGES",
    "DEFAULT_DRIFT_POLICY",
    "DRIFT_POLICY_VERSION",
    "DRIFT_SCHEMA_VERSION",
    "MEAN_GRAY_BIN_EDGES",
    "SCORE_BIN_EDGES",
    "DriftObservation",
    "DriftPolicy",
    "DriftReport",
    "DriftStatus",
    "FeatureDrift",
    "assess_drift",
    "extract_observations",
]
