"""T05 단위 테스트: Wilson 경계, 분모 0, 표본, override, grid 결정론, 봉인 차단."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from surface_sentinel.config import default_config, parse_config
from surface_sentinel.evaluation.ablation import (
    ABLATIONS,
    apply_overrides,
    get_spec,
    spec_config,
)
from surface_sentinel.evaluation.grid import (
    GRID_FIELDS,
    MAX_CANDIDATES,
    build_candidates,
    pareto_front,
    select,
)
from surface_sentinel.evaluation.metrics import (
    Ratio,
    derived_ratio,
    image_metrics,
    ratio,
    wilson_interval,
)
from surface_sentinel.evaluation.runner import (
    SEALED_SPLITS,
    SealedSplitAccess,
    load_split_rows,
)
from surface_sentinel.evaluation.sampling import (
    GOOD_TARGET,
    SampleError,
    build_sample_plan,
)
from surface_sentinel.io.manifest import ManifestRow


def _row(sample_id: str, label: str, source_split: str = "train") -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        original_id=sample_id,
        group_id=f"g{sample_id}",
        image_path=f"train/{sample_id}.png",
        mask_path=f"train/{sample_id}_GT.png" if label == "defect" else "",
        label=label,
        source_split=source_split,
        width=120,
        height=320,
        channels=3,
        format="PNG",
        image_sha256="0" * 64,
        mask_sha256="0" * 64,
        defect_pixels=10 if label == "defect" else 0,
    )


# ---------------------------------------------------------------- Wilson 경계


def test_wilson_zero_trials_is_undefined() -> None:
    assert wilson_interval(0, 0) == (None, None)
    assert wilson_interval(5, 0) == (None, None)


def test_wilson_zero_successes_low_is_zero_high_positive() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high is not None and 0.0 < high < 1.0
    # Wilson 상한은 정확값 0.5를 넘지 않는다(작은 표본에서 보수적).
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_all_successes_high_is_one_low_positive() -> None:
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert low is not None and 0.0 < low < 1.0
    assert low == pytest.approx(0.7225, abs=1e-3)


def test_wilson_single_trial() -> None:
    low, high = wilson_interval(1, 1)
    assert low is not None and high == pytest.approx(1.0)
    assert low < 1.0
    low0, high0 = wilson_interval(0, 1)
    assert low0 == 0.0 and high0 is not None and high0 < 1.0


def test_wilson_symmetric_around_proportion_when_p_is_half() -> None:
    low, high = wilson_interval(50, 100)
    assert low is not None and high is not None
    assert (low + high) / 2 == pytest.approx(0.5, abs=1e-12)
    assert 0.0 <= low < 0.5 < high <= 1.0


def test_wilson_interval_narrows_with_more_trials() -> None:
    low_small, high_small = wilson_interval(5, 10)
    low_big, high_big = wilson_interval(500, 1000)
    assert low_small is not None and high_small is not None
    assert low_big is not None and high_big is not None
    assert (high_big - low_big) < (high_small - low_small)


def test_wilson_stays_inside_unit_interval() -> None:
    for successes, trials in ((0, 214), (214, 214), (1, 3), (7, 9)):
        low, high = wilson_interval(successes, trials)
        assert low is not None and high is not None
        assert 0.0 <= low <= high <= 1.0


# ------------------------------------------------------------- 분모 0과 Ratio


def test_ratio_with_zero_denominator_has_no_value() -> None:
    result = ratio("테스트", 0, 0)
    assert result.denominator == 0
    assert result.value is None
    assert result.ci_low is None and result.ci_high is None
    assert result.formatted() == "- (분모 0)"
    assert result.as_dict()["value"] is None


def test_ratio_keeps_numerator_denominator_and_ci() -> None:
    result = ratio("테스트", 3, 10)
    assert result.numerator == 3
    assert result.denominator == 10
    assert result.value == pytest.approx(0.3)
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low < 0.3 < result.ci_high
    assert "분모 10" in result.formatted()


def test_derived_ratio_has_no_denominator() -> None:
    result = derived_ratio("balanced", 0.5)
    assert isinstance(result, Ratio)
    assert result.denominator is None
    assert result.value == 0.5


def test_image_metrics_denominators_follow_confusion() -> None:
    from surface_sentinel.evaluation.metrics import Confusion

    confusion = Confusion.from_pairs(
        [("defect", "OK"), ("defect", "NG"), ("good", "NG"), ("good", "OK")]
    )
    metrics = image_metrics(confusion)
    assert metrics["defect_escape_rate"].denominator == 2
    assert metrics["good_false_ng_rate"].denominator == 2
    assert metrics["review_rate"].denominator == 4
    assert metrics["defect_escape_rate"].value == pytest.approx(0.5)


def test_image_metrics_no_good_images_gives_no_ratio() -> None:
    from surface_sentinel.evaluation.metrics import Confusion

    confusion = Confusion.from_pairs([("defect", "NG")])
    metrics = image_metrics(confusion)
    assert metrics["good_false_ng_rate"].value is None
    assert metrics["good_false_ng_rate"].denominator == 0
    # 분모가 0이어도 다른 지표는 계산된다.
    assert metrics["defect_ng_rate"].value == pytest.approx(1.0)


# ------------------------------------------------------------------- 표본 설계


def test_sample_plan_takes_all_defects_and_stratifies_good() -> None:
    rows = [_row(f"d{index}", "defect") for index in range(5)]
    rows += [_row(f"g{index}", "good", "train") for index in range(8)]
    rows += [_row(f"h{index}", "good", "test") for index in range(2)]
    plan = build_sample_plan(rows, good_target=4)
    assert plan.defect_taken == 5
    assert plan.defect_population == 5
    assert plan.good_taken == 4
    assert plan.good_population == 10
    strata = {stratum.stratum: stratum.requested for stratum in plan.strata}
    assert strata == {"test": 1, "train": 3}


def test_sample_plan_is_deterministic() -> None:
    rows = [_row(f"d{index}", "defect") for index in range(3)]
    rows += [_row(f"g{index}", "good") for index in range(50)]
    first = build_sample_plan(rows, good_target=10)
    second = build_sample_plan(rows, good_target=10)
    assert [row.sample_id for row in first.rows] == [row.sample_id for row in second.rows]


def test_sample_plan_caps_good_target_at_population() -> None:
    rows = [_row("d0", "defect")] + [_row(f"g{index}", "good") for index in range(3)]
    plan = build_sample_plan(rows, good_target=GOOD_TARGET)
    assert plan.good_taken == 3
    assert sum(stratum.requested for stratum in plan.strata) == 3


def test_sample_plan_rejects_sealed_split() -> None:
    rows = [_row("d0", "defect")]
    with pytest.raises(SampleError):
        build_sample_plan(rows, split="test")
    with pytest.raises(SampleError):
        build_sample_plan(rows, split="official_test")


def test_sample_plan_rejects_empty_rows() -> None:
    with pytest.raises(SampleError):
        build_sample_plan([], split="tune")


# ---------------------------------------------------------------- 봉인 미접근


@pytest.mark.parametrize("split", sorted(SEALED_SPLITS))
def test_load_split_rows_refuses_sealed_split(split: str) -> None:
    with pytest.raises(SealedSplitAccess):
        load_split_rows(split)


def test_load_split_rows_refuses_unknown_split() -> None:
    from surface_sentinel.evaluation.runner import EvalError

    with pytest.raises(EvalError):
        load_split_rows("validation")


def test_evaluation_sources_never_name_sealed_csv() -> None:
    """봉인 CSV를 '여는' 문자열이 코드에 없어야 한다."""

    root = Path(__file__).resolve().parents[2] / "src" / "surface_sentinel" / "evaluation"
    forbidden = ("test.csv", "official_test.csv")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            # 이름만 금지한다: `f"{split}.csv"`처럼 조립하는 경로는 허용.
            assert f'"{needle}"' not in text, f"{path.name}에 {needle} 문자열이 있다"
            assert f"'{needle}'" not in text, f"{path.name}에 {needle} 문자열이 있다"


# ------------------------------------------------------------------ override


def test_apply_overrides_round_trips_through_validation() -> None:
    base = default_config()
    changed = apply_overrides(base, {"candidates": {"z_threshold": 9.0}})
    assert changed.candidates.z_threshold == 9.0
    assert changed.config_hash != base.config_hash
    # 원본은 그대로다(frozen).
    assert base.candidates.z_threshold == 4.0


def test_apply_overrides_rejects_invalid_values() -> None:
    base = default_config()
    with pytest.raises(ValueError):
        apply_overrides(base, {"candidates": {"z_threshold": 0.1}})
    with pytest.raises(ValueError):
        apply_overrides(base, {"nope": {"field": 1}})


def test_apply_overrides_rejects_unknown_field() -> None:
    base = default_config()
    with pytest.raises(ValueError):
        apply_overrides(base, {"scoring": {"unknown_field": 1.0}})


def test_every_ablation_config_is_valid_and_keeps_scoring_thresholds() -> None:
    base = default_config()
    for spec in ABLATIONS:
        config = spec_config(spec, base)
        assert config.scoring.ok_max == base.scoring.ok_max
        assert config.scoring.ng_min == base.scoring.ng_min
        assert config.schema_version == base.schema_version


def test_ablation_specs_cover_a1_to_a6() -> None:
    ids = [spec.id for spec in ABLATIONS]
    assert ids == ["A1", "A2", "A3", "A4", "A5", "A6"]
    assert get_spec("A2").overrides["roi"]["mode"] == "fixed"
    assert get_spec("A4").overrides["candidates"]["min_contrast_gray"] == 0.0
    assert get_spec("A5").overrides["preprocess"]["median_kernel"] == 1
    assert get_spec("A6").overrides["preprocess"]["use_bright_channel"] is False
    assert "근사" in get_spec("A1").note
    assert "근사" in get_spec("A3").note


def test_a1_and_a3_are_marked_as_approximations() -> None:
    for spec in ABLATIONS:
        if spec.id in {"A1", "A3"}:
            assert spec.kind == "approximate"
        else:
            assert spec.kind == "config"


def test_parsed_config_equals_serialised_round_trip() -> None:
    base = default_config()
    payload = base.to_dict()
    payload["scoring"]["ng_area_px"] = 2000
    parsed = parse_config(payload)
    assert parsed.scoring.ng_area_px == 2000.0
    assert parsed.config_hash != base.config_hash


# ---------------------------------------------------------------------- grid


def test_candidates_are_unique_and_within_cap() -> None:
    candidates = build_candidates()
    assert len(candidates) == MAX_CANDIDATES
    assert len({candidate.candidate_id for candidate in candidates}) == MAX_CANDIDATES
    assert len({tuple(sorted(candidate.overrides.items())) for candidate in candidates}) == (
        MAX_CANDIDATES
    )
    for candidate in candidates:
        assert set(candidate.overrides) == set(GRID_FIELDS)


def test_candidates_are_deterministic() -> None:
    first = [(c.candidate_id, tuple(sorted(c.overrides.items()))) for c in build_candidates()]
    second = [(c.candidate_id, tuple(sorted(c.overrides.items()))) for c in build_candidates()]
    assert first == second


class _FakeRow:
    """``select``/``pareto_front``만 시험하기 위한 최소 대역."""

    def __init__(self, name: str, escape: float, false_ng: float, review: float) -> None:
        self.candidate_id = name
        self.escape = escape
        self.false_ng = false_ng
        self.review = review
        self.localization_recall = 0.6
        self.overrides: dict[str, float] = {}
        self.report = None
        self.config_hash = "0" * 64

    def selection_ready(self) -> bool:
        return self.false_ng <= 0.20 and self.review <= 0.40

    def as_dict(self) -> dict[str, float]:
        return {"escape": self.escape, "false_ng": self.false_ng, "review": self.review}


def test_select_prefers_lowest_escape_among_feasible() -> None:
    rows = [
        _FakeRow("a", 0.10, 0.10, 0.30),
        _FakeRow("b", 0.05, 0.15, 0.35),
        _FakeRow("c", 0.00, 0.50, 0.10),
    ]
    selection = select(rows)  # type: ignore[arg-type]
    assert selection.best is not None
    assert selection.best.candidate_id == "b"
    assert len(selection.feasible) == 2


def test_select_reports_pareto_when_nothing_is_feasible() -> None:
    rows = [
        _FakeRow("a", 0.00, 0.90, 0.05),
        _FakeRow("b", 0.50, 0.10, 0.60),
        _FakeRow("c", 0.60, 0.40, 0.70),
    ]
    selection = select(rows)  # type: ignore[arg-type]
    assert selection.feasible == ()
    assert selection.best is not None
    assert "Pareto" in selection.reason
    # c는 a와 b에 지배된다.
    assert "c" not in [row.candidate_id for row in selection.pareto]


def test_pareto_front_keeps_mutually_undominated_points() -> None:
    rows = [
        _FakeRow("a", 0.00, 0.90, 0.05),
        _FakeRow("b", 0.50, 0.10, 0.60),
        _FakeRow("c", 0.60, 0.40, 0.70),
    ]
    front = pareto_front(rows)  # type: ignore[arg-type]
    assert [row.candidate_id for row in front] == ["a", "b"]


def test_grid_values_are_positive() -> None:
    for field, values in __import__(
        "surface_sentinel.evaluation.grid", fromlist=["GRID_VALUES"]
    ).GRID_VALUES.items():
        assert values, field
        assert all(value > 0 for value in values), field


def test_targets_are_the_agreed_numbers() -> None:
    from surface_sentinel.evaluation import grid

    assert grid.TARGET_ESCAPE == 0.15
    assert grid.TARGET_FALSE_NG == 0.20
    assert grid.TARGET_REVIEW == 0.40
    assert grid.TARGET_LOCALIZATION_RECALL == 0.50
    assert math.isclose(grid.TARGET_ESCAPE, 0.15)
