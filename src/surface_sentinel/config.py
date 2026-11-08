"""TOML configuration loading, validation, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

CONFIG_SCHEMA_VERSION = "1.0"


class ConfigError(ValueError):
    """Raised when a configuration file violates the schema."""


@dataclass(frozen=True)
class InputConfig:
    min_width: int = 64
    min_height: int = 64
    min_std: float = 1.0


@dataclass(frozen=True)
class RoiConfig:
    mode: str = "auto"
    dark_floor: int = 8
    margin_px: int = 2
    min_area_ratio: float = 0.20
    close_kernel: int = 7
    fixed_margin_px: int = 8


@dataclass(frozen=True)
class PreprocessConfig:
    median_kernel: int = 3
    bg_kernel_ratio: float = 0.10
    bg_kernel_min: int = 21
    bg_kernel_max: int = 61
    use_dark_channel: bool = True
    use_bright_channel: bool = True
    noise_floor_gray: float = 2.0


@dataclass(frozen=True)
class CandidatesConfig:
    z_threshold: float = 4.0
    min_contrast_gray: float = 12.0
    open_kernel: int = 3
    close_kernel: int = 5
    merge_kernel: int = 7
    min_area_px: int = 12
    max_candidates: int = 20


@dataclass(frozen=True)
class ScoringConfig:
    ng_area_px: float = 48.0
    ng_contrast_gray: float = 24.0
    ok_max: float = 0.55
    ng_min: float = 1.00
    multi_candidate_margin: float = 0.15


@dataclass(frozen=True)
class QualityConfig:
    blur_min_var: float = 20.0
    dark_mean_min: float = 20.0
    bright_mean_max: float = 235.0
    saturated_ratio_max: float = 0.10


@dataclass(frozen=True)
class DecisionConfig:
    border_touch_escalates: bool = True
    truncation_escalates: bool = True
    multi_candidate_escalates: bool = True


@dataclass(frozen=True)
class ArtifactsConfig:
    save_stages: bool = False
    overlay_scale: int = 2


@dataclass(frozen=True)
class AppConfig:
    schema_version: str
    input: InputConfig
    roi: RoiConfig
    preprocess: PreprocessConfig
    candidates: CandidatesConfig
    scoring: ScoringConfig
    quality: QualityConfig
    decision: DecisionConfig
    artifacts: ArtifactsConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_toml(self) -> str:
        lines = [f"schema_version = {_toml_literal(self.schema_version)}", ""]
        for section_name in (
            "input",
            "roi",
            "preprocess",
            "candidates",
            "scoring",
            "quality",
            "decision",
            "artifacts",
        ):
            section = getattr(self, section_name)
            lines.append(f"[{section_name}]")
            for field in fields(section):
                lines.append(f"{field.name} = {_toml_literal(getattr(section, field.name))}")
            lines.append("")
        return "\n".join(lines)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_toml().encode("utf-8")).hexdigest()


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise ConfigError(f"cannot serialize config value {value!r}")



def default_config() -> AppConfig:
    return AppConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        input=InputConfig(),
        roi=RoiConfig(),
        preprocess=PreprocessConfig(),
        candidates=CandidatesConfig(),
        scoring=ScoringConfig(),
        quality=QualityConfig(),
        decision=DecisionConfig(),
        artifacts=ArtifactsConfig(),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"cannot read config: {config_path}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    return parse_config(data)


def parse_config(data: Mapping[str, Any]) -> AppConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("config root must be a table")

    section_types: dict[str, type[Any]] = {
        "input": InputConfig,
        "roi": RoiConfig,
        "preprocess": PreprocessConfig,
        "candidates": CandidatesConfig,
        "scoring": ScoringConfig,
        "quality": QualityConfig,
        "decision": DecisionConfig,
        "artifacts": ArtifactsConfig,
    }
    allowed = set(section_types) | {"schema_version"}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")
    missing = set(section_types) - set(data)
    if missing:
        raise ConfigError(f"missing section(s): {sorted(missing)}")

    schema_version = data.get("schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    sections = {
        name: _build_section(name, section_type, data[name])
        for name, section_type in section_types.items()
    }
    config = AppConfig(schema_version=CONFIG_SCHEMA_VERSION, **sections)
    _validate(config)
    return config


def _build_section(name: str, section_type: type[Any], payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        raise ConfigError(f"[{name}] must be a table")
    hints = get_type_hints(section_type)
    allowed = {field.name for field in fields(section_type)}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {sorted(unknown)}")
    missing = allowed - set(payload)
    if missing:
        raise ConfigError(f"missing key(s) in [{name}]: {sorted(missing)}")
    values = {
        field.name: _coerce(payload[field.name], hints[field.name], f"{name}.{field.name}")
        for field in fields(section_type)
    }
    return section_type(**values)


def _coerce(value: Any, expected: Any, path: str) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    raise ConfigError(f"{path} has unsupported type annotation {expected!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _odd(path: str, value: int, minimum: int = 3) -> None:
    _require(value >= minimum and value % 2 == 1, f"{path} must be odd and >= {minimum}")


def _validate(config: AppConfig) -> None:
    _require(config.input.min_width >= 16, "input.min_width must be >= 16")
    _require(config.input.min_height >= 16, "input.min_height must be >= 16")
    _require(0.0 <= config.input.min_std <= 255.0, "input.min_std must be in [0, 255]")

    _require(config.roi.mode in {"auto", "fixed"}, "roi.mode must be 'auto' or 'fixed'")
    _require(0 <= config.roi.dark_floor <= 254, "roi.dark_floor must be in [0, 254]")
    _require(0 <= config.roi.margin_px <= 64, "roi.margin_px must be in [0, 64]")
    _require(0.0 < config.roi.min_area_ratio <= 1.0, "roi.min_area_ratio must be in (0, 1]")
    _odd("roi.close_kernel", config.roi.close_kernel)
    _require(0 <= config.roi.fixed_margin_px <= 256, "roi.fixed_margin_px must be in [0, 256]")

    _require(config.preprocess.median_kernel in {1, 3, 5}, "preprocess.median_kernel in {1,3,5}")
    _require(0.0 < config.preprocess.bg_kernel_ratio < 1.0, "preprocess.bg_kernel_ratio in (0,1)")
    _odd("preprocess.bg_kernel_min", config.preprocess.bg_kernel_min)
    _odd("preprocess.bg_kernel_max", config.preprocess.bg_kernel_max)
    _require(
        config.preprocess.bg_kernel_max >= config.preprocess.bg_kernel_min,
        "preprocess.bg_kernel_max must be >= bg_kernel_min",
    )
    _require(
        0.0 <= config.preprocess.noise_floor_gray <= 255.0,
        "preprocess.noise_floor_gray must be in [0, 255]",
    )

    _require(0.5 <= config.candidates.z_threshold <= 10.0, "candidates.z_threshold in [0.5,10]")
    _require(
        0.0 <= config.candidates.min_contrast_gray <= 255.0,
        "candidates.min_contrast_gray must be in [0, 255]",
    )
    _odd("candidates.open_kernel", config.candidates.open_kernel)
    _odd("candidates.close_kernel", config.candidates.close_kernel)
    _odd("candidates.merge_kernel", config.candidates.merge_kernel)
    _require(1 <= config.candidates.min_area_px <= 1_000_000, "candidates.min_area_px out of range")
    _require(
        1 <= config.candidates.max_candidates <= 1000,
        "candidates.max_candidates must be in [1, 1000]",
    )

    _require(config.scoring.ng_area_px > 0.0, "scoring.ng_area_px must be > 0")
    _require(config.scoring.ng_contrast_gray > 0.0, "scoring.ng_contrast_gray must be > 0")
    _require(0.0 <= config.scoring.ok_max <= 1.5, "scoring.ok_max must be in [0, 1.5]")
    _require(0.0 <= config.scoring.ng_min <= 1.5, "scoring.ng_min must be in [0, 1.5]")
    _require(config.scoring.ok_max < config.scoring.ng_min, "scoring.ok_max must be < ng_min")
    _require(
        0.0 <= config.scoring.multi_candidate_margin <= 1.5,
        "scoring.multi_candidate_margin must be in [0, 1.5]",
    )

    _require(config.quality.blur_min_var >= 0.0, "quality.blur_min_var must be >= 0")
    _require(0.0 <= config.quality.dark_mean_min <= 255.0, "quality.dark_mean_min in [0,255]")
    _require(0.0 <= config.quality.bright_mean_max <= 255.0, "quality.bright_mean_max in [0,255]")
    _require(
        config.quality.dark_mean_min < config.quality.bright_mean_max,
        "quality.dark_mean_min must be < bright_mean_max",
    )
    _require(
        0.0 <= config.quality.saturated_ratio_max <= 1.0,
        "quality.saturated_ratio_max in [0,1]",
    )

    _require(1 <= config.artifacts.overlay_scale <= 8, "artifacts.overlay_scale must be in [1, 8]")


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "AppConfig",
    "ArtifactsConfig",
    "CandidatesConfig",
    "ConfigError",
    "DecisionConfig",
    "InputConfig",
    "PreprocessConfig",
    "QualityConfig",
    "RoiConfig",
    "ScoringConfig",
    "default_config",
    "load_config",
    "parse_config",
]
