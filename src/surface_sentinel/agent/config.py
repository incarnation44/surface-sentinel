"""T09 제한된 에이전트 설정 (설계 §6).

``configs/agent.toml`` 은 동결 ``configs/base.toml`` 과 별개 파일이며 base 를
변경하지 않는다. 알 수 없는 키와 범위 위반은 ``ConfigError`` 로 거부한다.
예산 기본값은 설계 계약 값 그대로다: 행동 3회, 프로파일 2개, 추가 지연 5,000 ms.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_type_hints

from ..config import ConfigError

AGENT_SCHEMA_VERSION = "1.0"

_TOP_KEYS = {"schema_version", "budget", "policy", "comparison"}
_BUDGET_KEYS = {"max_actions", "max_profiles", "extra_latency_ms"}
_POLICY_KEYS = {"base_profile", "quality_low_profile", "roi_uncertain_profile"}
_COMPARISON_KEYS = {
    "subset_size",
    "seed",
    "review_drop_pp_min",
    "escape_increase_pp_max",
    "extra_p95_ms_max",
}


@dataclass(frozen=True)
class BudgetConfig:
    """행동/프로파일/추가 지연 예산 (설계 §6 계약 값)."""

    max_actions: int = 3
    max_profiles: int = 2
    extra_latency_ms: float = 5000.0


@dataclass(frozen=True)
class PolicyConfig:
    """정책이 선택할 수 있는 프로파일 ID (레지스트리에 미리 로드된다)."""

    base_profile: str = "base"
    quality_low_profile: str = "denoise_strong"
    roi_uncertain_profile: str = "roi_fixed"


@dataclass(frozen=True)
class ComparisonConfig:
    """고정 base 대비 정책 승격 검사 조건 (설계 §6 비교)."""

    subset_size: int = 120
    seed: int = 20261113
    review_drop_pp_min: float = 5.0
    escape_increase_pp_max: float = 1.0
    extra_p95_ms_max: float = 1500.0


@dataclass(frozen=True)
class AgentConfig:
    schema_version: str = AGENT_SCHEMA_VERSION
    budget: BudgetConfig = BudgetConfig()
    policy: PolicyConfig = PolicyConfig()
    comparison: ComparisonConfig = ComparisonConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_agent_config() -> AgentConfig:
    return AgentConfig()


def load_agent_config(path: str | Path) -> AgentConfig:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"에이전트 설정을 읽을 수 없습니다: {config_path}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"유효하지 않은 TOML 형식입니다({config_path}): {exc}") from exc
    return parse_agent_config(data)


def parse_agent_config(data: Mapping[str, Any]) -> AgentConfig:
    _unknown_keys(data, _TOP_KEYS, "agent")
    schema_version = data.get("schema_version", AGENT_SCHEMA_VERSION)
    if schema_version != AGENT_SCHEMA_VERSION:
        raise ConfigError(f"지원하지 않는 에이전트 schema_version입니다: {schema_version!r}")
    config = AgentConfig(
        schema_version=schema_version,
        budget=_section(data, "budget", _BUDGET_KEYS, BudgetConfig),
        policy=_section(data, "policy", _POLICY_KEYS, PolicyConfig),
        comparison=_section(data, "comparison", _COMPARISON_KEYS, ComparisonConfig),
    )
    _validate(config)
    return config


def _section(
    data: Mapping[str, Any],
    name: str,
    keys: set[str],
    section_type: type[Any],
) -> Any:
    payload = data.get(name, {})
    if not isinstance(payload, Mapping):
        raise ConfigError(f"agent.{name}은(는) 테이블이어야 합니다")
    _unknown_keys(payload, keys, f"agent.{name}")
    hints = get_type_hints(section_type)
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        expected = hints[key]
        if expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"agent.{name}.{key}은(는) 정수(int)여야 합니다")
        elif expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"agent.{name}.{key}은(는) 숫자여야 합니다")
            value = float(value)
        elif expected is str and not isinstance(value, str):
            raise ConfigError(f"agent.{name}.{key}은(는) 문자열이어야 합니다")
        kwargs[key] = value
    return section_type(**kwargs)


def _validate(config: AgentConfig) -> None:
    budget = config.budget
    _require(
        1 <= budget.max_actions <= 10,
        "budget.max_actions는 1 이상 10 이하이어야 합니다",
    )
    _require(
        1 <= budget.max_profiles <= 4,
        "budget.max_profiles는 1 이상 4 이하이어야 합니다",
    )
    _require(
        budget.max_profiles <= budget.max_actions,
        "budget.max_profiles는 budget.max_actions 이하이어야 합니다",
    )
    _require(budget.extra_latency_ms > 0.0, "budget.extra_latency_ms는 0보다 커야 합니다")

    policy = config.policy
    names = ("base_profile", "quality_low_profile", "roi_uncertain_profile")
    ids = tuple(getattr(policy, name) for name in names)
    for name, value in zip(names, ids, strict=True):
        _require(bool(value.strip()), f"policy.{name}은(는) 비어 있을 수 없습니다")
    _require(len(set(ids)) == len(ids), "policy 프로파일 ID는 서로 달라야 합니다")

    comparison = config.comparison
    _require(
        1 <= comparison.subset_size <= 100_000,
        "comparison.subset_size는 1 이상 100000 이하이어야 합니다",
    )
    _require(comparison.seed >= 0, "comparison.seed는 0 이상이어야 합니다")
    _require(
        comparison.review_drop_pp_min >= 0.0,
        "comparison.review_drop_pp_min은 0 이상이어야 합니다",
    )
    _require(
        comparison.escape_increase_pp_max >= 0.0,
        "comparison.escape_increase_pp_max는 0 이상이어야 합니다",
    )
    _require(comparison.extra_p95_ms_max > 0.0, "comparison.extra_p95_ms_max는 0보다 커야 합니다")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _unknown_keys(payload: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"{where}에 알 수 없는 키가 있습니다: {', '.join(unknown)}")


__all__ = [
    "AGENT_SCHEMA_VERSION",
    "AgentConfig",
    "BudgetConfig",
    "ComparisonConfig",
    "PolicyConfig",
    "default_agent_config",
    "load_agent_config",
    "parse_agent_config",
]
