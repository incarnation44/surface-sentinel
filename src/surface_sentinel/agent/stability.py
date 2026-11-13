"""Phase 2 안정성·약후보 기하·능동 패치 중재 모듈 (observe-only).

실데이터 KSDD2의 REVIEW 원인(``MULTIPLE_WEAK_CANDIDATES``,
``EVIDENCE_IN_REVIEW_BAND``)과 엔진 간 불일치(``LOCALIZATION_MISMATCH``,
``NG_SINGLE_ENGINE``)를 순수 Python + NumPy + OpenCV 결정론 연산으로
정밀 계측한다.

계약:
* 제품 판정(``SampleResult.status``)은 절대 생성하거나 변경하지 않는다.
* 외부 네트워크·셸·subprocess·디스크 파일 쓰기를 전혀 사용하지 않는다.
* 동일 입력·정책에서는 100% 동일한 해시와 계측치를 산출한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig
from ..contracts import (
    ComponentFeature,
    EngineObservation,
    ErrorCode,
    RoiInfo,
    SampleResult,
    Stage,
    Status,
)
from ..pipeline.stages import (
    _to_gray,
    analyze_components,
    calculate_score,
    decide_status,
    generate_candidates,
    prepare_roi,
    preprocess,
    refine_mask,
)
from ..records.fingerprint import canonical_json_sha256
from .evidence import bbox_center, bbox_iou

STABILITY_POLICY_VERSION = "stability-v1"
GEOMETRY_SCHEMA_VERSION = "weak-geometry-v1"
ARBITRATION_SCHEMA_VERSION = "patch-arbitration-v1"

#: 테두리 근접 판정 거리 기준(px).
_GEOMETRY_BORDER_DIST_PX = 8.0
#: 패치 중재 시 이미지 가장자리 근접 판정 거리 기준(px).
_PATCH_BORDER_MARGIN_PX = 6
#: 패치 주변 링(ring) 여유 폭(px).
_PATCH_RING_PAD_PX = 8


class WeakPattern(StrEnum):
    """약후보 컴포넌트들의 공간 기하 패턴."""

    NONE = "NONE"
    SINGLE_CANDIDATE = "SINGLE_CANDIDATE"
    COLLINEAR_SCRATCH_SUSPECT = "COLLINEAR_SCRATCH_SUSPECT"
    LOCAL_CLUSTER = "LOCAL_CLUSTER"
    BORDER_ARTIFACT_SUSPECT = "BORDER_ARTIFACT_SUSPECT"
    DIFFUSE_SURFACE_TEXTURE = "DIFFUSE_SURFACE_TEXTURE"


class StabilityLevel(StrEnum):
    """결정론 미세 섭동 민감도 등급."""

    STABLE = "STABLE"
    SENSITIVE = "SENSITIVE"
    UNSTABLE = "UNSTABLE"


class ArbitrationVerdict(StrEnum):
    """두 엔진 후보 패치의 교차 중재 결론."""

    ALIGNED = "ALIGNED"
    ENGINE_A_SALIENT = "ENGINE_A_SALIENT"
    ENGINE_B_SALIENT = "ENGINE_B_SALIENT"
    BOTH_SALIENT = "BOTH_SALIENT"
    BORDER_NOISE_SUSPECT = "BORDER_NOISE_SUSPECT"


@dataclass(frozen=True)
class WeakGeometryReport:
    """약후보 기하 가설 분석 결과(결정론 해시 포함)."""

    pattern: WeakPattern
    component_count: int
    collinearity_r2: float
    orthogonal_rms_px: float
    cluster_compactness: float
    border_touch_ratio: float
    max_elongation: float
    max_evidence: float
    geometry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "pattern": self.pattern.value,
            "component_count": self.component_count,
            "collinearity_r2": round(float(self.collinearity_r2), 6),
            "orthogonal_rms_px": round(float(self.orthogonal_rms_px), 6),
            "cluster_compactness": round(float(self.cluster_compactness), 6),
            "border_touch_ratio": round(float(self.border_touch_ratio), 6),
            "max_elongation": round(float(self.max_elongation), 6),
            "max_evidence": round(float(self.max_evidence), 6),
            "geometry_hash": self.geometry_hash,
        }


def analyze_weak_geometry(
    components: Sequence[ComponentFeature],
    roi: RoiInfo | None = None,
) -> WeakGeometryReport:
    """후보 컴포넌트들의 배열·밀집·테두리 편향을 분석해 기하 패턴을 판별한다."""

    count = len(components)
    if count == 0:
        payload = {
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "pattern": WeakPattern.NONE.value,
            "component_count": 0,
            "collinearity_r2": 0.0,
            "orthogonal_rms_px": 0.0,
            "cluster_compactness": 0.0,
            "border_touch_ratio": 0.0,
            "max_elongation": 0.0,
            "max_evidence": 0.0,
        }
        return WeakGeometryReport(
            pattern=WeakPattern.NONE,
            component_count=0,
            collinearity_r2=0.0,
            orthogonal_rms_px=0.0,
            cluster_compactness=0.0,
            border_touch_ratio=0.0,
            max_elongation=0.0,
            max_evidence=0.0,
            geometry_hash=canonical_json_sha256(payload),
        )

    centers = [bbox_center(c.bbox) for c in components]
    if count >= 3:
        pts = np.asarray(centers, dtype=np.float64)
        centered = pts - np.mean(pts, axis=0)
        cov = (centered.T @ centered) / float(count)
        eigvals = np.linalg.eigvalsh(cov)
        lambda_min = max(0.0, float(eigvals[0]))
        lambda_max = max(0.0, float(eigvals[1]))
        total_var = lambda_max + lambda_min
        raw_r2 = 1.0 if total_var <= 1e-12 else (lambda_max / total_var)
        collinearity_r2 = round(min(max(raw_r2, 0.0), 1.0), 6)
        orthogonal_rms_px = round(math.sqrt(max(0.0, lambda_min)), 6)
    else:
        collinearity_r2 = 0.0
        orthogonal_rms_px = 0.0

    if count >= 2:
        distances: list[float] = []
        for i in range(count):
            for j in range(i + 1, count):
                distances.append(
                    math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                )
        mean_dist = sum(distances) / float(len(distances))
        if roi is not None and roi.bbox[2] > 0 and roi.bbox[3] > 0:
            diag = math.hypot(float(roi.bbox[2]), float(roi.bbox[3]))
        else:
            min_x = min(c.bbox[0] for c in components)
            min_y = min(c.bbox[1] for c in components)
            max_x = max(c.bbox[0] + c.bbox[2] for c in components)
            max_y = max(c.bbox[1] + c.bbox[3] for c in components)
            diag = max(math.hypot(float(max_x - min_x), float(max_y - min_y)), 1.0)
        norm_dist = mean_dist / max(diag, 1.0)
        cluster_compactness = round(min(max(1.0 - norm_dist, 0.0), 1.0), 6)
    else:
        cluster_compactness = 0.0

    border_hits = sum(
        1
        for c in components
        if c.touches_roi_border or float(c.border_distance_px) <= _GEOMETRY_BORDER_DIST_PX
    )
    border_touch_ratio = round(border_hits / float(count), 6)
    max_elongation = round(max(float(c.elongation) for c in components), 6)
    max_evidence = round(max(float(c.evidence) for c in components), 6)

    if count == 1:
        pattern = WeakPattern.SINGLE_CANDIDATE
    elif border_touch_ratio >= 0.6:
        pattern = WeakPattern.BORDER_ARTIFACT_SUSPECT
    elif (
        count >= 3 and collinearity_r2 >= 0.85 and orthogonal_rms_px <= 14.0
    ) or (
        count >= 3
        and max_elongation >= 4.0
        and collinearity_r2 >= 0.75
        and orthogonal_rms_px <= 18.0
    ):
        pattern = WeakPattern.COLLINEAR_SCRATCH_SUSPECT
    elif cluster_compactness >= 0.72:
        pattern = WeakPattern.LOCAL_CLUSTER
    else:
        pattern = WeakPattern.DIFFUSE_SURFACE_TEXTURE

    payload = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "pattern": pattern.value,
        "component_count": count,
        "collinearity_r2": collinearity_r2,
        "orthogonal_rms_px": orthogonal_rms_px,
        "cluster_compactness": cluster_compactness,
        "border_touch_ratio": border_touch_ratio,
        "max_elongation": max_elongation,
        "max_evidence": max_evidence,
    }
    return WeakGeometryReport(
        pattern=pattern,
        component_count=count,
        collinearity_r2=collinearity_r2,
        orthogonal_rms_px=orthogonal_rms_px,
        cluster_compactness=cluster_compactness,
        border_touch_ratio=border_touch_ratio,
        max_elongation=max_elongation,
        max_evidence=max_evidence,
        geometry_hash=canonical_json_sha256(payload),
    )


@dataclass(frozen=True)
class StabilityPolicy:
    """결정론 섭동 민감도 프로브 정책 상수(동결 기본값)."""

    version: str = STABILITY_POLICY_VERSION
    brightness_factor: float = 1.05
    blur_ksize: int = 3
    blur_sigma: float = 0.8
    stable_retention_min: float = 0.75
    stable_delta_max: float = 0.10
    sensitive_retention_min: float = 0.55
    sensitive_delta_max: float = 0.22
    zero_base_stable_delta_max: float = 0.05
    zero_base_sensitive_delta_max: float = 0.20

    def policy_hash(self) -> str:
        """정책 버전 및 임계값 전체의 canonical SHA256."""

        return canonical_json_sha256(asdict(self))


DEFAULT_STABILITY_POLICY = StabilityPolicy()


@dataclass(frozen=True)
class PerturbationSample:
    """단일 결정론 섭동에 대한 파이프라인 응답."""

    name: str
    status: Status
    score: float
    score_delta: float
    top_box_iou: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "score": round(float(self.score), 6),
            "score_delta": round(float(self.score_delta), 6),
            "top_box_iou": round(float(self.top_box_iou), 6),
        }


@dataclass(frozen=True)
class StabilityReport:
    """섭동 민감도 프로브 종합 결과(판정 불변·결정론 해시 포함)."""

    level: StabilityLevel
    base_status: Status
    base_score: float
    perturbations: tuple[PerturbationSample, ...]
    status_preserved_count: int
    mean_score_retention: float
    max_abs_score_delta: float
    policy_hash: str
    stability_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "base_status": self.base_status.value,
            "base_score": round(float(self.base_score), 6),
            "perturbations": [item.to_dict() for item in self.perturbations],
            "status_preserved_count": self.status_preserved_count,
            "mean_score_retention": round(float(self.mean_score_retention), 6),
            "max_abs_score_delta": round(float(self.max_abs_score_delta), 6),
            "policy_hash": self.policy_hash,
            "stability_hash": self.stability_hash,
        }


def _top_component(components: Sequence[ComponentFeature]) -> ComponentFeature | None:
    if not components:
        return None
    return max(
        components,
        key=lambda c: (float(c.evidence), int(c.area_px), -int(c.index)),
    )


def _inspect_array_in_memory(
    image: np.ndarray,
    config: AppConfig,
    *,
    sample_id: str = "in-memory",
    split: str = "tune",
    label: str = "unknown",
) -> SampleResult:
    """디스크 파일 쓰기 없이 순수 메모리 내에서 10단계 파이프라인을 실행한다."""

    try:
        gray = np.ascontiguousarray(_to_gray(image))
    except ValueError:
        return SampleResult(
            run_id="in-memory",
            sample_id=sample_id,
            original_id=sample_id,
            split=split,
            label=label,
            status=Status.ERROR,
            score=0.0,
            error_code=ErrorCode.E_INPUT_DECODE,
            error_stage=Stage.LOAD_IMAGE,
            config_hash=config.config_hash,
        )

    height, width = int(gray.shape[0]), int(gray.shape[1])
    if width < config.input.min_width or height < config.input.min_height:
        return SampleResult(
            run_id="in-memory",
            sample_id=sample_id,
            original_id=sample_id,
            split=split,
            label=label,
            status=Status.ERROR,
            score=0.0,
            error_code=ErrorCode.E_INPUT_SIZE,
            error_stage=Stage.VALIDATE_INPUT,
            config_hash=config.config_hash,
        )

    if float(np.std(gray, ddof=0)) < config.input.min_std:
        return SampleResult(
            run_id="in-memory",
            sample_id=sample_id,
            original_id=sample_id,
            split=split,
            label=label,
            status=Status.ERROR,
            score=0.0,
            error_code=ErrorCode.E_INPUT_UNIFORM,
            error_stage=Stage.VALIDATE_INPUT,
            config_hash=config.config_hash,
        )

    roi = prepare_roi(gray, config)
    pre = preprocess(gray, roi, config)
    candidate = generate_candidates(pre.z_map, pre.response, roi.mask, config)
    refined = refine_mask(candidate.mask, pre.z_map, pre.response, roi.mask, config)
    analyzed = analyze_components(refined, roi, pre.response, config)
    scored = calculate_score(analyzed.components, config)
    decision = decide_status(scored, pre.quality, roi, refined, config)
    return SampleResult(
        run_id="in-memory",
        sample_id=sample_id,
        original_id=sample_id,
        split=split,
        label=label,
        status=decision.status,
        score=float(scored.score),
        reasons=decision.reasons,
        components=analyzed.components,
        quality=pre.quality,
        roi=roi.to_info(),
        config_hash=config.config_hash,
    )


def assess_perturbation_stability(
    bgr: np.ndarray,
    config: AppConfig,
    baseline: SampleResult | None = None,
    policy: StabilityPolicy = DEFAULT_STABILITY_POLICY,
) -> StabilityReport:
    """메모리 내에서 2종의 결정론 미세 섭동(밝기 +5%, 3x3 가우시안 블러)을 수행한다.

    입력 배열 ``bgr`` 은 변경하지 않으며, 디스크 파일을 전혀 생성하지 않는다.
    """

    if not isinstance(bgr, np.ndarray) or bgr.ndim not in (2, 3):
        raise ValueError("유효한 2D/3D 이미지 numpy 배열이 필요합니다")

    base = baseline if baseline is not None else _inspect_array_in_memory(bgr, config)
    base_score = round(float(base.score), 6)
    base_top = _top_component(base.components)

    bright_img = np.clip(
        np.rint(bgr.astype(np.float32) * float(policy.brightness_factor)), 0, 255
    ).astype(np.uint8)
    blur_img = cv2.GaussianBlur(
        bgr, (int(policy.blur_ksize), int(policy.blur_ksize)), float(policy.blur_sigma)
    )

    pert_specs = (
        ("brightness_plus_5pct", bright_img),
        ("gaussian_blur_3x3", blur_img),
    )

    samples: list[PerturbationSample] = []
    for name, pert_img in pert_specs:
        pert_res = _inspect_array_in_memory(
            pert_img,
            config,
            sample_id=base.sample_id,
            split=base.split,
            label=base.label,
        )
        pert_score = round(float(pert_res.score), 6)
        delta = round(pert_score - base_score, 6)
        pert_top = _top_component(pert_res.components)
        if base_top is not None and pert_top is not None:
            top_iou = round(bbox_iou(base_top.bbox, pert_top.bbox), 6)
        else:
            top_iou = 0.0
        samples.append(
            PerturbationSample(
                name=name,
                status=pert_res.status,
                score=pert_score,
                score_delta=delta,
                top_box_iou=top_iou,
            )
        )

    perturbations = tuple(samples)
    status_preserved = sum(1 for item in perturbations if item.status is base.status)
    pert_scores = [item.score for item in perturbations]
    max_abs_delta = round(max(abs(item.score_delta) for item in perturbations), 6)

    if base_score > 0.0:
        raw_retention = min(pert_scores) / base_score
        retention = round(min(max(raw_retention, 0.0), 1.5), 6)
        if (
            all(s >= base_score * policy.stable_retention_min for s in pert_scores)
            and max_abs_delta <= policy.stable_delta_max
        ):
            level = StabilityLevel.STABLE
        elif (
            any(s >= base_score * policy.sensitive_retention_min for s in pert_scores)
            and max_abs_delta <= policy.sensitive_delta_max
        ):
            level = StabilityLevel.SENSITIVE
        else:
            level = StabilityLevel.UNSTABLE
    else:
        retention = 1.0 if max(pert_scores) <= policy.zero_base_stable_delta_max else 0.0
        if max_abs_delta <= policy.zero_base_stable_delta_max:
            level = StabilityLevel.STABLE
        elif max_abs_delta <= policy.zero_base_sensitive_delta_max:
            level = StabilityLevel.SENSITIVE
        else:
            level = StabilityLevel.UNSTABLE

    policy_sha = policy.policy_hash()
    payload = {
        "level": level.value,
        "base_status": base.status.value,
        "base_score": base_score,
        "perturbations": [item.to_dict() for item in perturbations],
        "status_preserved_count": status_preserved,
        "mean_score_retention": retention,
        "max_abs_score_delta": max_abs_delta,
        "policy_hash": policy_sha,
    }
    return StabilityReport(
        level=level,
        base_status=base.status,
        base_score=base_score,
        perturbations=perturbations,
        status_preserved_count=status_preserved,
        mean_score_retention=retention,
        max_abs_score_delta=max_abs_delta,
        policy_hash=policy_sha,
        stability_hash=canonical_json_sha256(payload),
    )


@dataclass(frozen=True)
class PatchMetrics:
    """엔진 최상위 후보 패치의 국소 물리 계측값."""

    engine: str
    component_index: int | None
    bbox: tuple[int, int, int, int] | None
    local_contrast: float
    edge_sharpness: float
    touches_border: bool
    salience_index: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "component_index": self.component_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "local_contrast": round(float(self.local_contrast), 6),
            "edge_sharpness": round(float(self.edge_sharpness), 6),
            "touches_border": self.touches_border,
            "salience_index": round(float(self.salience_index), 6),
        }


@dataclass(frozen=True)
class PatchArbitration:
    """두 엔진의 최상위 후보 패치 비교 중재 결과."""

    engine_a: str
    engine_b: str
    verdict: ArbitrationVerdict
    patch_a: PatchMetrics | None
    patch_b: PatchMetrics | None
    summary: str
    arbitration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARBITRATION_SCHEMA_VERSION,
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "verdict": self.verdict.value,
            "patch_a": self.patch_a.to_dict() if self.patch_a is not None else None,
            "patch_b": self.patch_b.to_dict() if self.patch_b is not None else None,
            "summary": self.summary,
            "arbitration_hash": self.arbitration_hash,
        }


def _measure_patch(
    gray: np.ndarray,
    observation: EngineObservation,
) -> PatchMetrics | None:
    comp = _top_component(observation.result.components)
    if comp is None:
        return None
    height, width = int(gray.shape[0]), int(gray.shape[1])
    x, y, w, h = comp.bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x) + max(0, int(w)))
    y1 = min(height, int(y) + max(0, int(h)))
    if x1 <= x0 or y1 <= y0:
        return PatchMetrics(
            engine=observation.engine,
            component_index=comp.index,
            bbox=comp.bbox,
            local_contrast=0.0,
            edge_sharpness=0.0,
            touches_border=bool(comp.touches_roi_border),
            salience_index=0.0,
        )

    patch = gray[y0:y1, x0:x1]
    ox0 = max(0, x0 - _PATCH_RING_PAD_PX)
    oy0 = max(0, y0 - _PATCH_RING_PAD_PX)
    ox1 = min(width, x1 + _PATCH_RING_PAD_PX)
    oy1 = min(height, y1 + _PATCH_RING_PAD_PX)

    outer = gray[oy0:oy1, ox0:ox1]
    mask = np.ones(outer.shape, dtype=bool)
    mask[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    ring_pixels = outer[mask]

    internal_contrast = float(np.percentile(patch, 95) - np.percentile(patch, 5))
    if ring_pixels.size > 0:
        ring_med = float(np.median(ring_pixels))
        ring_diff = max(
            abs(float(np.percentile(patch, 90)) - ring_med),
            abs(float(np.percentile(patch, 10)) - ring_med),
        )
        local_contrast = max(ring_diff, internal_contrast)
    else:
        local_contrast = internal_contrast

    outer_f = outer.astype(np.float32)
    dx = cv2.Sobel(outer_f, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(outer_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(dx, dy)
    patch_grad = grad_mag[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0]
    edge_sharpness = float(np.percentile(patch_grad, 90)) if patch_grad.size > 0 else 0.0

    touches_border = bool(
        comp.touches_roi_border
        or x0 <= _PATCH_BORDER_MARGIN_PX
        or y0 <= _PATCH_BORDER_MARGIN_PX
        or (width - x1) <= _PATCH_BORDER_MARGIN_PX
        or (height - y1) <= _PATCH_BORDER_MARGIN_PX
    )
    norm_contrast = min(local_contrast / 40.0, 1.5)
    norm_edge = min(edge_sharpness / 120.0, 1.5)
    salience_index = round(0.55 * norm_contrast + 0.45 * norm_edge, 6)

    return PatchMetrics(
        engine=observation.engine,
        component_index=comp.index,
        bbox=comp.bbox,
        local_contrast=round(local_contrast, 6),
        edge_sharpness=round(edge_sharpness, 6),
        touches_border=touches_border,
        salience_index=salience_index,
    )


def arbitrate_patches(
    image: np.ndarray,
    obs_a: EngineObservation,
    obs_b: EngineObservation,
) -> PatchArbitration:
    """두 엔진의 최상위 후보 패치와 8px 주변 링을 비교해 중재 결론을 생성한다."""

    gray = _to_gray(image)
    patch_a = _measure_patch(gray, obs_a)
    patch_b = _measure_patch(gray, obs_b)

    sal_a = patch_a.salience_index if patch_a is not None else 0.0
    sal_b = patch_b.salience_index if patch_b is not None else 0.0
    salient_min = 0.35
    dominance_margin = 0.15

    both_present = patch_a is not None and patch_b is not None
    overlap_iou = (
        bbox_iou(patch_a.bbox, patch_b.bbox)
        if (both_present and patch_a.bbox is not None and patch_b.bbox is not None)
        else 0.0
    )

    if patch_a is None and patch_b is None:
        verdict = ArbitrationVerdict.ALIGNED
        summary = f"{obs_a.engine}·{obs_b.engine} 모두 검출 후보가 없어 일치(ALIGNED)함"
    elif both_present and overlap_iou >= 0.30:
        if patch_a.touches_border and patch_b.touches_border and max(sal_a, sal_b) < salient_min:
            verdict = ArbitrationVerdict.BORDER_NOISE_SUSPECT
            summary = (
                f"두 엔진 후보가 겹치나(IoU {overlap_iou:.2f}) 이미지 가장자리 접촉·저현저성"
                f"(최대 {max(sal_a, sal_b):.2f})으로 경계 노이즈 의심"
            )
        else:
            verdict = ArbitrationVerdict.ALIGNED
            summary = (
                f"{obs_a.engine}(대비 {patch_a.local_contrast:.1f}, "
                f"경사도 {patch_a.edge_sharpness:.1f})와 "
                f"{obs_b.engine}(대비 {patch_b.local_contrast:.1f}, "
                f"경사도 {patch_b.edge_sharpness:.1f}) 패치가 "
                f"동일 위치(IoU {overlap_iou:.2f})에서 일치함"
            )
    elif (
        (patch_a is None or patch_a.touches_border)
        and (patch_b is None or patch_b.touches_border)
        and max(sal_a, sal_b) < 0.60
    ):
        verdict = ArbitrationVerdict.BORDER_NOISE_SUSPECT
        active = patch_a if patch_a is not None else patch_b
        assert active is not None
        summary = (
            f"{active.engine} 후보 패치가 경계에 접촉하며 대비 {active.local_contrast:.1f}·"
            f"경사도 {active.edge_sharpness:.1f}로 경계 반사 노이즈가 의심됨"
        )
    elif (
        both_present
        and sal_a >= salient_min
        and sal_b >= salient_min
        and abs(sal_a - sal_b) < dominance_margin
    ):
        verdict = ArbitrationVerdict.BOTH_SALIENT
        summary = (
            f"{obs_a.engine}(현저성 {sal_a:.2f})와 {obs_b.engine}(현저성 {sal_b:.2f}) 모두 "
            "서로 다른 위치에서 뚜렷한 표면 단차를 보임"
        )
    elif sal_a >= sal_b + dominance_margin or (patch_b is None and sal_a >= salient_min):
        verdict = ArbitrationVerdict.ENGINE_A_SALIENT
        assert patch_a is not None
        summary = (
            f"{obs_a.engine} 후보 패치(대비 {patch_a.local_contrast:.1f}, "
            f"경사도 {patch_a.edge_sharpness:.1f})가 {obs_b.engine} 후보보다 "
            "뚜렷한 표면 단차를 보임"
        )
    elif sal_b >= sal_a + dominance_margin or (patch_a is None and sal_b >= salient_min):
        verdict = ArbitrationVerdict.ENGINE_B_SALIENT
        assert patch_b is not None
        summary = (
            f"{obs_b.engine} 후보 패치(대비 {patch_b.local_contrast:.1f}, "
            f"경사도 {patch_b.edge_sharpness:.1f})가 {obs_a.engine} 후보보다 "
            "뚜렷한 표면 단차를 보임"
        )
    elif (patch_a is not None and patch_a.touches_border) or (
        patch_b is not None and patch_b.touches_border
    ):
        verdict = ArbitrationVerdict.BORDER_NOISE_SUSPECT
        summary = "경계 접촉 및 저대비 후보 패치로 경계 노이즈가 의심됨"
    elif sal_a >= sal_b:
        verdict = ArbitrationVerdict.ENGINE_A_SALIENT
        assert patch_a is not None
        summary = (
            f"{obs_a.engine} 후보 패치(대비 {patch_a.local_contrast:.1f}, "
            f"경사도 {patch_a.edge_sharpness:.1f})가 상대적으로 우세함"
        )
    else:
        verdict = ArbitrationVerdict.ENGINE_B_SALIENT
        assert patch_b is not None
        summary = (
            f"{obs_b.engine} 후보 패치(대비 {patch_b.local_contrast:.1f}, "
            f"경사도 {patch_b.edge_sharpness:.1f})가 상대적으로 우세함"
        )

    payload = {
        "schema_version": ARBITRATION_SCHEMA_VERSION,
        "engine_a": obs_a.engine,
        "engine_b": obs_b.engine,
        "verdict": verdict.value,
        "patch_a": patch_a.to_dict() if patch_a is not None else None,
        "patch_b": patch_b.to_dict() if patch_b is not None else None,
        "summary": summary,
    }
    return PatchArbitration(
        engine_a=obs_a.engine,
        engine_b=obs_b.engine,
        verdict=verdict,
        patch_a=patch_a,
        patch_b=patch_b,
        summary=summary,
        arbitration_hash=canonical_json_sha256(payload),
    )


__all__ = [
    "ARBITRATION_SCHEMA_VERSION",
    "DEFAULT_STABILITY_POLICY",
    "GEOMETRY_SCHEMA_VERSION",
    "STABILITY_POLICY_VERSION",
    "ArbitrationVerdict",
    "PatchArbitration",
    "PatchMetrics",
    "PerturbationSample",
    "StabilityLevel",
    "StabilityPolicy",
    "StabilityReport",
    "WeakGeometryReport",
    "WeakPattern",
    "analyze_weak_geometry",
    "arbitrate_patches",
    "assess_perturbation_stability",
]
