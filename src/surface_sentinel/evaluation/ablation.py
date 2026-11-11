"""구조 ablation A1–A6 (설계 §4.1).

원칙: **기본 판정 임계값은 그대로 둔다.** tune에서 구조만 빼고, 그때 지표가
어떻게 움직이는지 본다. 그래서 여기서 바꾸는 값은 판정 밴드
(``REVIEW_BAND_MAX``/``ng_min``)가 아니라 구조 파라미터뿐이다.

| ID | 취지 | 구현 | 정확도 |
|---|---|---|---|
| A1 | 배경 보정 제거 | ``bg_kernel_ratio``를 최소로 | **근사** (커널 21px로 축소. 제거가 아님) |
| A2 | ROI 제거 | ``roi.mode="fixed"``, ``fixed_margin_px=0`` | config override(전체 영상 ROI) |
| A3 | 컴포넌트 정제 제거 | ``min_area_px=1`` | **근사** (면적 체 거름만 제거) |
| A4 | 대비 하한 제거 | ``min_contrast_gray=0`` | config override |
| A5 | median blur 제거 | ``median_kernel=1`` | config override |
| A6 | 어두운 채널만 사용 | ``use_bright_channel=false`` | config override |

A1은 응답 스케일 자체가 달라진다(black-hat 커널 61px → 21px). 그래서 A1의
점수는 다른 행과 **같은 자로 재면 안 된다.** 표에서 스케일이 다르다는 사실을
명시하고, 해석은 "구조 기여의 방향"으로만 한다.

A3은 ``refine_mask``의 재임계·open 연산을 직접 끌 수 있는 스위치가 없어
`min_area_px=1`로 근사한다. 즉 "면적 체로 거르던 정제"만 제거되고, 재임계와
open은 남는다. 그만큼 A3의 효과는 실제보다 작게 나온다(보수적).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig, parse_config
from .cache import CachedSample
from .measure import VariantReport, evaluate_cached

KIND_CONFIG = "config"
KIND_APPROX = "approximate"


@dataclass(frozen=True)
class AblationSpec:
    id: str
    intent: str
    kind: str
    overrides: Mapping[str, Mapping[str, Any]]
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent,
            "kind": self.kind,
            "overrides": {section: dict(values) for section, values in self.overrides.items()},
            "note": self.note,
        }


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        id="A1",
        intent="배경 보정 제거",
        kind=KIND_APPROX,
        overrides={"preprocess": {"bg_kernel_ratio": 0.001}},
        note=(
            "근사: 배경 커널을 하한 21px로 축소(제거 아님). 응답·점수 스케일이 "
            "달라지므로 지표는 방향성만 본다."
        ),
    ),
    AblationSpec(
        id="A2",
        intent="ROI 제거",
        kind=KIND_CONFIG,
        overrides={"roi": {"mode": "fixed", "fixed_margin_px": 0}},
        note="config override: 전체 영상이 ROI가 된다(경계 링 없음).",
    ),
    AblationSpec(
        id="A3",
        intent="컴포넌트 정제 제거",
        kind=KIND_APPROX,
        overrides={"candidates": {"min_area_px": 1}},
        note=(
            "근사: 정제의 면적 체(min_area)만 제거. 재임계·open은 남으므로 "
            "효과가 실제보다 작게 나온다."
        ),
    ),
    AblationSpec(
        id="A4",
        intent="대비 하한 제거",
        kind=KIND_CONFIG,
        overrides={"candidates": {"min_contrast_gray": 0.0}},
        note="config override: 후보 생성과 점수 타당성의 대비 하한을 0으로.",
    ),
    AblationSpec(
        id="A5",
        intent="median blur 제거",
        kind=KIND_CONFIG,
        overrides={"preprocess": {"median_kernel": 1}},
        note="config override: median 3 → 1(생략). 잡음 민감도를 본다.",
    ),
    AblationSpec(
        id="A6",
        intent="어두운 채널만 사용",
        kind=KIND_CONFIG,
        overrides={"preprocess": {"use_bright_channel": False}},
        note="config override: top-hat(밝은 결함) 채널을 끈다.",
    ),
)

ABLATION_IDS: tuple[str, ...] = tuple(spec.id for spec in ABLATIONS)


def get_spec(ablation_id: str) -> AblationSpec:
    for spec in ABLATIONS:
        if spec.id == ablation_id:
            return spec
    raise ValueError(f"알 수 없는 ablation입니다: {ablation_id!r}")


def apply_overrides(config: AppConfig, overrides: Mapping[str, Mapping[str, Any]]) -> AppConfig:
    """섹션/필드 override를 적용하고 **스키마 검증을 다시 통과시킨다**."""

    payload = config.to_dict()
    for section, values in overrides.items():
        if section not in payload:
            raise ValueError(f"알 수 없는 설정 섹션입니다: {section!r}")
        if not isinstance(payload[section], dict):
            raise ValueError(f"설정 섹션이 아닙니다: {section!r}")
        payload[section].update(values)
    return parse_config(payload)


def spec_config(spec: AblationSpec, base_config: AppConfig) -> AppConfig:
    return apply_overrides(base_config, spec.overrides)


def evaluate_ablation(
    spec: AblationSpec,
    samples: Sequence[CachedSample],
    base_config: AppConfig,
) -> VariantReport:
    """이미 만들어진 ablation 캐시 위에서 지표를 낸다.

    판정 임계값(``scoring.*``)은 건드리지 않는다. 캐시는 이 spec의 config로
    만들어야 한다(구조가 달라 컴포넌트가 달라지기 때문).
    """

    config = spec_config(spec, base_config)
    return evaluate_cached(
        samples,
        config,
        name=spec.id,
        overrides=spec.overrides,
        notes=(f"{spec.intent} — {spec.note}",),
    )


def run_all_ablations(
    samples_by_id: Mapping[str, Sequence[CachedSample]],
    base_config: AppConfig,
) -> list[VariantReport]:
    """``samples_by_id``는 ablation id → 그 config로 만든 캐시여야 한다."""

    reports: list[VariantReport] = []
    for spec in ABLATIONS:
        samples = samples_by_id.get(spec.id)
        if samples is None:
            continue
        reports.append(evaluate_ablation(spec, samples, base_config))
    return reports


def summarize(reports: Sequence[VariantReport]) -> list[dict[str, Any]]:
    """표에 바로 쓰는 행 묶음. 비율은 언제나 분모와 함께 나온다."""

    rows: list[dict[str, Any]] = []
    for report in reports:
        escape = report.ratio("defect_escape_rate")
        false_ng = report.ratio("good_false_ng_rate")
        review = report.ratio("review_rate")
        recall = report.ratio("localization_recall_iou0.10")
        rows.append(
            {
                "id": report.name,
                "config_hash": report.config_hash,
                "escape": escape.as_dict(),
                "false_ng": false_ng.as_dict(),
                "review": review.as_dict(),
                "localization_recall": recall.as_dict(),
                "notes": list(report.notes),
            }
        )
    return rows


__all__ = [
    "ABLATIONS",
    "ABLATION_IDS",
    "KIND_APPROX",
    "KIND_CONFIG",
    "AblationSpec",
    "apply_overrides",
    "evaluate_ablation",
    "get_spec",
    "run_all_ablations",
    "spec_config",
    "summarize",
]
