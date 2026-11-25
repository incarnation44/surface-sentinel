"""전수 `tune` 분석(`evaluation/fullverify.py`) 단위 테스트 — escape 원인·REVIEW 사유·IoU 분포 검증.

합성 이미지 fixture로 escape 원인·REVIEW 사유·IoU 경계·캐시 정체성·
CLI 종료 코드를 검증한다. 실제 KSDD2 이미지는 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import synthetic as syn
from surface_sentinel.cli import main
from surface_sentinel.config import default_config
from surface_sentinel.contracts import Status
from surface_sentinel.evaluation.cache import (
    CachedComponent,
    CachedSample,
    cache_meta,
    cache_path,
    save_cache,
)
from surface_sentinel.evaluation.fullverify import (
    FullVerifyError,
    _classify_cause,
    _load_cached_samples,
    classify_escape,
    iou_details,
    review_breakdown,
)


@pytest.fixture()
def workdir() -> Path:
    path = Path(".work/unit_fullverify")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _component(bbox: tuple[int, int, int, int], mean_contrast: float = 100.0) -> CachedComponent:
    _, _, width, height = bbox
    mask = np.full((height, width), 255, dtype=np.uint8)
    return CachedComponent(
        area_px=width * height,
        bbox=bbox,
        mean_contrast_gray=mean_contrast,
        max_contrast_gray=mean_contrast,
        integrated_contrast=mean_contrast * width * height,
        border_distance_px=5.0,
        touches_roi_border=False,
        mask=mask,
    )


def _sample(
    name: str,
    *,
    label: str = "defect",
    mask_path: str = "",
    status: Status = Status.OK,
    reasons: tuple[str, ...] = (),
    components: tuple[CachedComponent, ...] = (),
    gt_components: tuple[CachedComponent, ...] = (),
    truncated: bool = False,
) -> CachedSample:
    return CachedSample(
        sample_id=name,
        original_id=name,
        label=label,
        split="tune",
        source_split="train",
        image_path=f"{name}.png",
        mask_path=mask_path,
        defect_pixels=0,
        status=status,
        score=0.0,
        reasons=reasons,
        quality_flags=(),
        blur_variance=1.0,
        mean_gray=120.0,
        saturated_ratio=0.0,
        roi_uncertain=False,
        truncated=truncated,
        components=components,
        gt_components=gt_components,
        overlay="",
        error_code=None,
        error_stage=None,
    )


def _diagnostics(**overrides: Any) -> dict[str, Any]:
    base = {
        "roi_area_px": 30000,
        "both_pixels": 20,
        "candidate_pixels": 20,
        "cc_count": 1,
        "max_cc_area": 100,
        "refined_components": 1,
        "refined_pixels": 20,
        "max_refined_mean_contrast": 30.0,
        "min_area_px": 20,
        "min_contrast_gray": 22.0,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ 원인 분류


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"roi_area_px": 0}, "roi_empty"),
        ({"both_pixels": 0}, "no_candidate_pixels"),
        ({"candidate_pixels": 0}, "morphology_erased"),
        ({"refined_components": 0, "refined_pixels": 0}, "morphology_erased"),
        ({"max_cc_area": 5}, "all_below_min_area"),
        ({"max_refined_mean_contrast": 10.0}, "all_below_min_contrast"),
        ({}, "other"),
    ],
)
def test_classify_cause_priority(overrides: dict[str, Any], expected: str) -> None:
    assert _classify_cause(_diagnostics(**overrides)) == expected


def test_classify_escape_roi_empty(workdir: Path) -> None:
    image = syn.good_image()
    syn.write_png(workdir / "roi.png", image)
    config = replace(
        default_config(),
        roi=replace(default_config().roi, mode="fixed", fixed_margin_px=1000),
    )
    result = classify_escape(
        _sample("roi", label="good"), config, raw_root=workdir
    )
    assert result["cause"] == "roi_empty"
    assert result["diagnostics"]["roi_area_px"] == 0


def test_classify_escape_no_candidate_pixels(workdir: Path) -> None:
    image = syn.good_image()
    syn.write_png(workdir / "empty.png", image)
    config = replace(
        default_config(),
        candidates=replace(default_config().candidates, z_threshold=1_000_000.0),
    )
    result = classify_escape(
        _sample("empty", label="good"), config, raw_root=workdir
    )
    assert result["cause"] == "no_candidate_pixels"
    assert result["diagnostics"]["both_pixels"] == 0
    assert result["diagnostics"]["contrast_only_pixels"] > 0


def test_classify_escape_morphology_erased(workdir: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=21))
    image[60, 160] = (0, 0, 0)
    syn.write_png(workdir / "pixel.png", image)
    syn.write_png(workdir / "pixel_gt.png", syn.rect_mask(image.shape, 160, 60, 1, 1))
    base = default_config()
    config = replace(
        base,
        candidates=replace(
            base.candidates,
            z_threshold=3.0,
            min_contrast_gray=5.0,
            min_area_px=1,
        ),
    )
    result = classify_escape(
        _sample("pixel", mask_path="pixel_gt.png"), config, raw_root=workdir
    )
    assert result["cause"] == "morphology_erased"
    assert result["diagnostics"]["both_pixels"] > 0
    assert result["diagnostics"]["candidate_pixels"] == 0


def test_classify_escape_all_below_min_area(workdir: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=22))
    image[58:62, 158:162] = 0
    syn.write_png(workdir / "rect.png", image)
    syn.write_png(workdir / "rect_gt.png", syn.rect_mask(image.shape, 158, 58, 4, 4))
    base = default_config()
    config = replace(
        base,
        candidates=replace(
            base.candidates,
            z_threshold=3.0,
            min_contrast_gray=5.0,
            min_area_px=1000,
        ),
    )
    result = classify_escape(
        _sample("rect", mask_path="rect_gt.png"), config, raw_root=workdir
    )
    assert result["cause"] == "all_below_min_area"
    assert result["diagnostics"]["max_cc_area"] < 1000


# ---------------------------------------------------------------- REVIEW 분해


def test_review_breakdown_splits_labels_and_reasons() -> None:
    samples = [
        _sample(
            "d1",
            status=Status.REVIEW,
            reasons=("EVIDENCE_IN_REVIEW_BAND", "CANDIDATE_TRUNCATED"),
        ),
        _sample(
            "g1",
            label="good",
            status=Status.REVIEW,
            reasons=("EVIDENCE_IN_REVIEW_BAND",),
        ),
        _sample("ok", status=Status.OK),
    ]
    payload = review_breakdown(samples)
    assert payload["samples"] == {"defect": 1, "good": 1, "total": 2}
    assert payload["reasons"]["defect"]["EVIDENCE_IN_REVIEW_BAND"] == 1
    assert payload["reasons"]["defect"]["CANDIDATE_TRUNCATED"] == 1
    assert payload["reasons"]["good"]["EVIDENCE_IN_REVIEW_BAND"] == 1


# -------------------------------------------------------------------- IoU 상세


def test_iou_details_recall_and_boundary() -> None:
    predicted = _component((0, 0, 10, 10))
    gt_full = _component((0, 0, 10, 10))
    gt_empty = _component((20, 20, 10, 10))
    sample = _sample(
        "iou",
        components=(predicted,),
        gt_components=(gt_full, gt_empty),
    )
    payload = iou_details([sample], default_config())
    assert payload["gt_components"] == 2
    assert payload["recall_at"]["0.50"]["numerator"] == 1
    assert payload["recall_at"]["0.50"]["denominator"] == 2
    assert payload["distribution"]["count"] == 2
    assert payload["distribution"]["histogram"][-1]["count"] == 1
    assert payload["distribution"]["histogram"][0]["count"] == 1


# ------------------------------------------------------------------ 캐시 정체성


def test_load_cached_samples_rejects_meta_mismatch(workdir: Path) -> None:
    plan = SimpleNamespace(rows=[SimpleNamespace(sample_id="s1")], split="tune")
    config = default_config()
    meta = cache_meta(plan, config)
    path = cache_path(meta, workdir)
    save_cache([], replace(meta, samples=999), path)
    with pytest.raises(ValueError):
        _load_cached_samples(plan, config, split_hash="", cache_dir=workdir)


def test_load_cached_samples_missing_cache(workdir: Path) -> None:
    plan = SimpleNamespace(rows=[SimpleNamespace(sample_id="s1")], split="tune")
    with pytest.raises(FullVerifyError):
        _load_cached_samples(
            plan, default_config(), split_hash="", cache_dir=workdir / "missing"
        )


# -------------------------------------------------------------------- CLI 게이트


def test_cli_rejects_sealed_split() -> None:
    assert main(["fullverify-details", "--split", "test"]) == 6


def test_cli_rejects_unknown_split() -> None:
    assert main(["fullverify-details", "--split", "train"]) == 3
