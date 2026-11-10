"""T04 N18: 설정 변경이 실제 inspect 결과에 반영되는지 검증한다."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import synthetic as syn
from surface_sentinel.config import ConfigError, parse_config
from surface_sentinel.contracts import ReasonCode, Status
from surface_sentinel.pipeline import inspect_image
from surface_sentinel.pipeline.evidence import sanitize_sample_id

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = parse_config(
    tomllib.loads((REPO_ROOT / "configs" / "base.toml").read_text(encoding="utf-8"))
)


def _write(tmp_path: Path, name: str, image: np.ndarray) -> Path:
    return syn.write_png(tmp_path / "inputs" / name, image)


def _inspect(
    tmp_path: Path,
    image_path: Path,
    config=BASE,
    *,
    name: str = "out",
):
    return inspect_image(
        image_path,
        config,
        output_dir=tmp_path / name,
    )


def _two_level_defect(tmp_path: Path) -> Path:
    image = syn.to_bgr(syn.surface_gray(seed=121))
    image = syn.add_dark_rect(image, 80, 50, 10, 10, 40)
    image = syn.add_dark_rect(image, 220, 50, 10, 10, 90)
    return _write(tmp_path, "two_levels.png", image)


def _candidate_count(outcome) -> int:
    assert outcome.debug is not None
    return int(np.count_nonzero(outcome.debug.candidate_mask))


def test_z_threshold_changes_candidate_pixel_count(tmp_path: Path) -> None:
    path = _two_level_defect(tmp_path)
    base = replace(
        BASE,
        preprocess=replace(BASE.preprocess, noise_floor_gray=8.0),
    )
    low = replace(base, candidates=replace(base.candidates, z_threshold=3.0))
    high = replace(base, candidates=replace(base.candidates, z_threshold=6.0))

    low_outcome = _inspect(tmp_path, path, low, name="z3")
    high_outcome = _inspect(tmp_path, path, high, name="z6")

    low_count = _candidate_count(low_outcome)
    high_count = _candidate_count(high_outcome)
    assert low_count != high_count
    assert low_count > high_count > 0
    assert low.config_hash != high.config_hash


def test_min_area_threshold_changes_status(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=82))
    image = syn.add_dark_rect(image, 150, 50, 8, 8, 20)
    path = _write(tmp_path, "min_area.png", image)
    strict = replace(
        BASE,
        candidates=replace(BASE.candidates, min_area_px=100),
    )

    default_outcome = _inspect(tmp_path, path, BASE, name="default")
    strict_outcome = _inspect(tmp_path, path, strict, name="strict")

    assert default_outcome.result.status is Status.NG
    assert strict_outcome.result.status is Status.OK
    assert strict_outcome.result.components == ()
    assert strict.config_hash != BASE.config_hash


def test_ng_min_changes_status_but_not_score(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=81))
    image = syn.add_dark_rect(image, 150, 50, 8, 8, 90)
    path = _write(tmp_path, "ng_min.png", image)
    strict = replace(BASE, scoring=replace(BASE.scoring, ng_min=1.4))

    default_outcome = _inspect(tmp_path, path, BASE, name="default")
    strict_outcome = _inspect(tmp_path, path, strict, name="strict")

    assert default_outcome.result.status is Status.NG
    assert strict_outcome.result.status is Status.REVIEW
    assert strict_outcome.result.reasons == (ReasonCode.EVIDENCE_IN_REVIEW_BAND,)
    assert strict_outcome.result.score == pytest.approx(default_outcome.result.score)
    assert BASE.scoring.ok_max < BASE.scoring.ng_min < 1.4


def test_dark_floor_changes_roi_area_and_uncertainty(tmp_path: Path) -> None:
    path = _write(tmp_path, "good.png", syn.good_image())
    high_floor = replace(BASE, roi=replace(BASE.roi, dark_floor=200))

    default_outcome = _inspect(tmp_path, path, BASE, name="default")
    high_outcome = _inspect(tmp_path, path, high_floor, name="high_floor")

    assert default_outcome.result.roi is not None
    assert high_outcome.result.roi is not None
    assert default_outcome.result.roi.area_px > 0
    assert default_outcome.result.roi.uncertain is False
    assert high_outcome.result.roi.area_px == 0
    assert high_outcome.result.roi.uncertain is True
    assert high_outcome.result.roi.area_px != default_outcome.result.roi.area_px


def test_bright_channel_toggle_changes_candidate_count(tmp_path: Path) -> None:
    image = syn.bright_defect_image()[0]
    path = _write(tmp_path, "bright.png", image)
    dark_only = replace(
        BASE,
        preprocess=replace(BASE.preprocess, use_bright_channel=False),
    )

    both_outcome = _inspect(tmp_path, path, BASE, name="both")
    dark_outcome = _inspect(tmp_path, path, dark_only, name="dark_only")

    assert _candidate_count(both_outcome) > 0
    assert _candidate_count(dark_outcome) == 0
    assert both_outcome.result.status is Status.NG
    assert dark_outcome.result.status is Status.OK


def test_config_hash_changes_with_values() -> None:
    assert len(BASE.config_hash) == 64
    int(BASE.config_hash, 16)

    variants = [
        replace(BASE, scoring=replace(BASE.scoring, ng_min=1.1)),
        replace(BASE, roi=replace(BASE.roi, dark_floor=9)),
        replace(BASE, preprocess=replace(BASE.preprocess, use_bright_channel=False)),
        replace(BASE, candidates=replace(BASE.candidates, min_area_px=13)),
        replace(BASE, quality=replace(BASE.quality, blur_min_var=21.0)),
        replace(BASE, artifacts=replace(BASE.artifacts, overlay_scale=3)),
        replace(BASE, decision=replace(BASE.decision, border_touch_escalates=False)),
        replace(BASE, input=replace(BASE.input, min_std=1.1)),
    ]
    hashes = {variant.config_hash for variant in variants}
    assert len(hashes) == len(variants)
    assert BASE.config_hash not in hashes


def test_invalid_configs_are_rejected() -> None:
    data = BASE.to_dict()
    data["preprocess"]["use_dark_channel"] = False
    data["preprocess"]["use_bright_channel"] = False
    with pytest.raises(ConfigError, match="use_dark_channel"):
        parse_config(data)

    data = BASE.to_dict()
    data["candidates"]["z_threshold"] = 0.1
    with pytest.raises(ConfigError, match="z_threshold"):
        parse_config(data)

    data = BASE.to_dict()
    data["candidates"]["open_kernel"] = 4
    with pytest.raises(ConfigError, match="must be odd"):
        parse_config(data)

    data = BASE.to_dict()
    data["candidates"]["min_area_px"] = 0
    with pytest.raises(ConfigError, match="min_area_px"):
        parse_config(data)

    data = BASE.to_dict()
    data["scoring"]["ok_max"] = 1.2
    with pytest.raises(ConfigError, match="ok_max"):
        parse_config(data)

    data = BASE.to_dict()
    data["roi"]["close_kernel"] = 6
    with pytest.raises(ConfigError, match="close_kernel"):
        parse_config(data)


def test_inspect_uses_passed_config_and_records_hash(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=81))
    image = syn.add_dark_rect(image, 150, 50, 8, 8, 90)
    path = _write(tmp_path, "custom_config.png", image)
    custom = replace(BASE, scoring=replace(BASE.scoring, ng_min=1.4))

    outcome = _inspect(tmp_path, path, custom)
    result = outcome.result

    assert result.config_hash == custom.config_hash
    assert result.config_hash != BASE.config_hash
    assert result.status is Status.REVIEW
    assert result.reasons == (ReasonCode.EVIDENCE_IN_REVIEW_BAND,)

    safe_id = sanitize_sample_id(result.sample_id)
    payload = json.loads(
        (Path(result.overlay_path).parent / f"{safe_id}_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["config_hash"] == custom.config_hash
