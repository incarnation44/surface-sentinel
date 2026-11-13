"""T09 에이전트 설정 단위 테스트. 설계 §6 계약 값과 동결 base 를 검증한다."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from surface_sentinel.agent.config import (
    default_agent_config,
    load_agent_config,
    parse_agent_config,
)
from surface_sentinel.agent.run import load_profiles
from surface_sentinel.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_CONFIG_PATH = REPO_ROOT / "configs" / "agent.toml"
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.toml"
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"

FROZEN_BASE_HASH = "a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098"


def test_defaults_match_design_contract() -> None:
    config = default_agent_config()
    assert config.budget.max_actions == 3
    assert config.budget.max_profiles == 2
    assert config.budget.extra_latency_ms == 5000.0
    assert config.policy.base_profile == "base"
    assert config.policy.quality_low_profile == "denoise_strong"
    assert config.policy.roi_uncertain_profile == "roi_fixed"
    assert config.comparison.subset_size == 120
    assert config.comparison.review_drop_pp_min == 5.0
    assert config.comparison.escape_increase_pp_max == 1.0
    assert config.comparison.extra_p95_ms_max == 1500.0


def test_repo_agent_toml_matches_defaults() -> None:
    assert load_agent_config(AGENT_CONFIG_PATH) == default_agent_config()


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ConfigError, match="알 수 없는 키"):
        parse_agent_config({"gateway": {}})


def test_unknown_section_key_rejected() -> None:
    with pytest.raises(ConfigError, match="알 수 없는 키"):
        parse_agent_config({"budget": {"max_action": 3}})


@pytest.mark.parametrize(
    "payload",
    [
        {"budget": {"max_actions": 0}},
        {"budget": {"max_actions": 11}},
        {"budget": {"max_profiles": 0}},
        {"budget": {"max_actions": 1, "max_profiles": 2}},
        {"budget": {"extra_latency_ms": 0}},
        {"budget": {"max_actions": "3"}},
        {"budget": {"max_actions": True}},
        {"policy": {"base_profile": ""}},
        {"policy": {"base_profile": "x", "quality_low_profile": "x"}},
        {"comparison": {"subset_size": 0}},
        {"comparison": {"seed": -1}},
        {"comparison": {"extra_p95_ms_max": 0}},
        {"schema_version": "9.9"},
    ],
)
def test_invalid_values_rejected(payload: dict) -> None:
    with pytest.raises(ConfigError):
        parse_agent_config(payload)


def test_base_config_still_frozen() -> None:
    assert load_config(BASE_CONFIG_PATH).config_hash == FROZEN_BASE_HASH


def test_denoise_strong_profile_differs_only_in_preprocess() -> None:
    base = load_config(BASE_CONFIG_PATH)
    profile = load_config(PROFILES_DIR / "denoise_strong.toml")
    assert profile.preprocess.median_kernel == 5
    assert profile.preprocess.noise_floor_gray == 3.0
    assert replace(
        profile,
        preprocess=replace(
            profile.preprocess,
            median_kernel=base.preprocess.median_kernel,
            noise_floor_gray=base.preprocess.noise_floor_gray,
        ),
    ) == base


def test_roi_fixed_profile_differs_only_in_roi_mode() -> None:
    base = load_config(BASE_CONFIG_PATH)
    profile = load_config(PROFILES_DIR / "roi_fixed.toml")
    assert profile.roi.mode == "fixed"
    assert replace(profile, roi=replace(profile.roi, mode=base.roi.mode)) == base


def test_load_profiles_registry() -> None:
    registry = load_profiles(BASE_CONFIG_PATH, PROFILES_DIR)
    assert set(registry) == {"base", "denoise_strong", "roi_fixed"}
    assert registry["base"].config_hash == FROZEN_BASE_HASH
