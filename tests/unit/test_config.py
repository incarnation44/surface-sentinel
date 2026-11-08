"""Configuration schema and hashing tests."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pytest

from surface_sentinel.config import (
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    default_config,
    load_config,
    parse_config,
)

BASE_TOML = """
schema_version = "1.0"

[input]
min_width = 64
min_height = 64
min_std = 1.0

[roi]
mode = "auto"
dark_floor = 8
margin_px = 2
min_area_ratio = 0.20
close_kernel = 7
fixed_margin_px = 8

[preprocess]
median_kernel = 3
bg_kernel_ratio = 0.10
bg_kernel_min = 21
bg_kernel_max = 61
use_dark_channel = true
use_bright_channel = true
noise_floor_gray = 2.0

[candidates]
z_threshold = 4.0
min_contrast_gray = 12.0
open_kernel = 3
close_kernel = 5
merge_kernel = 7
min_area_px = 12
max_candidates = 20

[scoring]
ng_area_px = 48.0
ng_contrast_gray = 24.0
ok_max = 0.55
ng_min = 1.00
multi_candidate_margin = 0.15

[quality]
blur_min_var = 20.0
dark_mean_min = 20.0
bright_mean_max = 235.0
saturated_ratio_max = 0.10

[decision]
border_touch_escalates = true
truncation_escalates = true
multi_candidate_escalates = true

[artifacts]
save_stages = false
overlay_scale = 2
"""


def base_data() -> dict:
    return tomllib.loads(BASE_TOML)


def test_valid_config_hash_is_stable() -> None:
    first = parse_config(base_data())
    second = parse_config(base_data())
    assert first.schema_version == CONFIG_SCHEMA_VERSION
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64
    int(first.config_hash, 16)


def test_default_config_hash_is_stable() -> None:
    assert default_config().config_hash == default_config().config_hash


def test_canonical_toml_round_trips() -> None:
    config = parse_config(base_data())
    reparsed = parse_config(tomllib.loads(config.canonical_toml()))
    assert reparsed == config
    assert reparsed.config_hash == config.config_hash



def test_unknown_top_level_key_rejected() -> None:
    data = base_data()
    data["extra"] = {}
    with pytest.raises(ConfigError, match="unknown top-level"):
        parse_config(data)


def test_unknown_section_key_rejected() -> None:
    data = base_data()
    data["input"]["extra"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(data)


def test_missing_section_rejected() -> None:
    data = base_data()
    del data["quality"]
    with pytest.raises(ConfigError, match="missing section"):
        parse_config(data)


def test_schema_version_mismatch_rejected() -> None:
    data = base_data()
    data["schema_version"] = "2.0"
    with pytest.raises(ConfigError, match="schema_version"):
        parse_config(data)


def test_ok_max_must_be_less_than_ng_min() -> None:
    data = base_data()
    data["scoring"]["ok_max"] = 1.2
    with pytest.raises(ConfigError, match="ok_max must be < ng_min"):
        parse_config(data)


def test_kernels_must_be_odd() -> None:
    data = base_data()
    data["candidates"]["open_kernel"] = 4
    with pytest.raises(ConfigError, match="must be odd"):
        parse_config(data)


def test_background_kernel_range_must_be_ordered() -> None:
    data = base_data()
    data["preprocess"]["bg_kernel_min"] = 61
    data["preprocess"]["bg_kernel_max"] = 21
    with pytest.raises(ConfigError, match="bg_kernel_max"):
        parse_config(data)


def test_bool_is_not_accepted_as_integer() -> None:
    data = base_data()
    data["input"]["min_width"] = True
    with pytest.raises(ConfigError, match="must be an integer"):
        parse_config(data)


def test_invalid_roi_mode_rejected() -> None:
    data = base_data()
    data["roi"]["mode"] = "manual"
    with pytest.raises(ConfigError, match="roi.mode"):
        parse_config(data)


def test_hash_changes_with_value() -> None:
    first = parse_config(base_data())
    data = base_data()
    data["scoring"]["ng_min"] = 1.10
    second = parse_config(data)
    assert first.config_hash != second.config_hash


def test_load_config_from_file(tmp_path: Path) -> None:
    path = tmp_path / "base.toml"
    path.write_text(BASE_TOML, encoding="utf-8")
    config = load_config(path)
    assert config.config_hash == parse_config(base_data()).config_hash


def test_load_config_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[input\nmin_width = 64\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "missing.toml")


def test_parse_does_not_mutate_input() -> None:
    data = base_data()
    snapshot = copy.deepcopy(data)
    parse_config(data)
    assert data == snapshot
