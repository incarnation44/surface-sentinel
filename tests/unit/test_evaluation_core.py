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
    assert base.candidates.z_threshold == 7.6


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
        assert config.scoring.ng_area_px == base.scoring.ng_area_px
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
    expected = 12
    assert len(candidates) == expected
    assert len(candidates) <= MAX_CANDIDATES
    assert len({candidate.candidate_id for candidate in candidates}) == expected
    assert len({tuple(sorted(candidate.overrides.items())) for candidate in candidates}) == (
        expected
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


# ------------------------------------------------------------------ 캐시 보안


_PICKLE_PROBE_EXECUTED = False


def _mark_pickle_probe() -> dict[str, object]:
    global _PICKLE_PROBE_EXECUTED
    _PICKLE_PROBE_EXECUTED = True
    return {"meta": {}, "samples": []}


class _PickleBomb:
    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_mark_pickle_probe, ())


def test_load_cache_rejects_pickle_payload_without_executing(tmp_path: Path) -> None:
    import pickle

    from surface_sentinel.evaluation.cache import CACHE_VERSION, CacheMeta, load_cache

    global _PICKLE_PROBE_EXECUTED
    _PICKLE_PROBE_EXECUTED = False

    target = tmp_path / "evil.sscache"
    target.write_bytes(pickle.dumps(_PickleBomb(), protocol=4))
    meta = CacheMeta(
        version=CACHE_VERSION,
        config_hash="a" * 64,
        split="tune",
        split_sha256="b" * 64,
        sample_ids_sha256="c" * 64,
        samples=0,
    )
    with pytest.raises(ValueError):
        load_cache(target, meta)
    assert _PICKLE_PROBE_EXECUTED is False


def test_save_and_load_cache_roundtrip_preserves_masks_and_fields(tmp_path: Path) -> None:
    import numpy as np

    from surface_sentinel.contracts import Status
    from surface_sentinel.evaluation.cache import (
        CACHE_VERSION,
        CachedComponent,
        CachedSample,
        CacheMeta,
        cache_path,
        load_cache,
        save_cache,
    )

    mask = np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8)
    gt_mask = np.array([[255, 255]], dtype=np.uint8)
    comp = CachedComponent(
        area_px=3,
        bbox=(10, 20, 3, 2),
        mean_contrast_gray=12.5,
        max_contrast_gray=25.0,
        integrated_contrast=37.5,
        border_distance_px=4.0,
        touches_roi_border=True,
        mask=mask,
    )
    gt_comp = CachedComponent(
        area_px=2,
        bbox=(11, 20, 2, 1),
        mean_contrast_gray=0.0,
        max_contrast_gray=0.0,
        integrated_contrast=0.0,
        border_distance_px=0.0,
        touches_roi_border=False,
        mask=gt_mask,
    )
    sample = CachedSample(
        sample_id="s1",
        original_id="orig1",
        label="defect",
        split="tune",
        source_split="train",
        image_path="train/s1.png",
        mask_path="train/s1_GT.png",
        defect_pixels=2,
        status=Status.NG,
        score=0.82,
        reasons=("DEFECT_EVIDENCE_STRONG",),
        quality_flags=(),
        blur_variance=120.5,
        mean_gray=130.2,
        saturated_ratio=0.01,
        roi_uncertain=False,
        truncated=False,
        components=(comp,),
        gt_components=(gt_comp,),
        overlay="s1_overlay.png",
        error_code=None,
        error_stage=None,
    )
    meta = CacheMeta(
        version=CACHE_VERSION,
        config_hash="a" * 64,
        split="tune",
        split_sha256="b" * 64,
        sample_ids_sha256="c" * 64,
        samples=1,
    )
    path = cache_path(meta, tmp_path)
    assert path.suffix != ".pkl"
    saved = save_cache([sample], meta, path)
    loaded = load_cache(saved, meta)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.sample_id == "s1"
    assert got.status is Status.NG
    assert got.reasons == ("DEFECT_EVIDENCE_STRONG",)
    assert got.error_code is None
    assert len(got.components) == 1
    assert got.components[0].bbox == (10, 20, 3, 2)
    assert got.components[0].touches_roi_border is True
    assert np.array_equal(got.components[0].mask, mask)
    assert got.components[0].mask.dtype == np.uint8
    assert len(got.gt_components) == 1
    assert np.array_equal(got.gt_components[0].mask, gt_mask)


def test_save_cache_cleans_unique_temp_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from surface_sentinel.evaluation.cache import CACHE_VERSION, CacheMeta, save_cache

    meta = CacheMeta(
        version=CACHE_VERSION,
        config_hash="a" * 64,
        split="tune",
        split_sha256="b" * 64,
        sample_ids_sha256="c" * 64,
        samples=0,
    )
    target = tmp_path / "cache.sscache"

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        save_cache([], meta, target)
    assert list(tmp_path.iterdir()) == []


def test_load_cache_rejects_zlib_bomb_and_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zlib

    from surface_sentinel.evaluation import cache as cache_mod
    from surface_sentinel.evaluation.cache import CACHE_VERSION, CacheMeta, load_cache

    meta = CacheMeta(
        version=CACHE_VERSION,
        config_hash="a" * 64,
        split="tune",
        split_sha256="b" * 64,
        sample_ids_sha256="c" * 64,
        samples=0,
    )
    # 1) 소형 zlib 폭탄 (압축 수 KB → 해제 1 MiB, 상한을 4096 B로 낮춰 검증)
    bomb = tmp_path / "bomb.sscache"
    bomb.write_bytes(b"SSCACHE2\n" + zlib.compress(b"A" * (1024 * 1024), level=9))
    monkeypatch.setattr(cache_mod, "MAX_CACHE_DECOMPRESSED_BYTES", 4096)
    with pytest.raises(ValueError, match="크기"):
        load_cache(bomb, meta)

    # 2) 압축 파일 자체 크기 상한 초과
    monkeypatch.setattr(cache_mod, "MAX_CACHE_COMPRESSED_BYTES", 16)
    with pytest.raises(ValueError, match="크기"):
        load_cache(bomb, meta)


def test_load_cache_rejects_abnormal_mask_shape_and_non_list_fields(tmp_path: Path) -> None:
    import json
    import zlib

    from surface_sentinel.evaluation.cache import CACHE_VERSION, CacheMeta, load_cache

    meta = CacheMeta(
        version=CACHE_VERSION,
        config_hash="a" * 64,
        split="tune",
        split_sha256="b" * 64,
        sample_ids_sha256="c" * 64,
        samples=1,
    )

    def _write(samples_payload: list[dict[str, object]], name: str) -> Path:
        raw = json.dumps({"meta": meta.as_dict(), "samples": samples_payload}).encode("utf-8")
        p = tmp_path / name
        p.write_bytes(b"SSCACHE2\n" + zlib.compress(raw))
        return p

    base_sample: dict[str, object] = {
        "sample_id": "s1",
        "original_id": "s1",
        "label": "defect",
        "split": "tune",
        "source_split": "train",
        "image_path": "a.png",
        "mask_path": "",
        "defect_pixels": 0,
        "status": "OK",
        "score": 0.0,
        "reasons": [],
        "quality_flags": [],
        "blur_variance": 1.0,
        "mean_gray": 100.0,
        "saturated_ratio": 0.0,
        "roi_uncertain": False,
        "truncated": False,
        "components": [],
        "gt_components": [],
        "overlay": "",
        "error_code": None,
        "error_stage": None,
    }

    # (0, 1_000_000_000) 빈 바이트 shape 공격 거부
    bad_shape = dict(base_sample)
    bad_shape["components"] = [
        {
            "area_px": 0,
            "bbox": [0, 0, 0, 0],
            "mean_contrast_gray": 0.0,
            "max_contrast_gray": 0.0,
            "integrated_contrast": 0.0,
            "border_distance_px": 0.0,
            "touches_roi_border": False,
            "mask_shape": [0, 1_000_000_000],
            "mask_b64": "",
        }
    ]
    with pytest.raises(ValueError):
        load_cache(_write([bad_shape], "huge_zero.sscache"), meta)

    # bool shape 거부
    bool_shape = dict(base_sample)
    bool_shape["components"] = [
        {
            **bad_shape["components"][0],  # type: ignore[index]
            "mask_shape": [True, True],
            "mask_b64": "AA==",
        }
    ]
    with pytest.raises(ValueError):
        load_cache(_write([bool_shape], "bool_shape.sscache"), meta)

    # reasons 가 문자열이면 글자 분해 대신 거부
    str_reasons = dict(base_sample)
    str_reasons["reasons"] = "DEFECT_EVIDENCE_STRONG"
    with pytest.raises(ValueError):
        load_cache(_write([str_reasons], "str_reasons.sscache"), meta)


def test_load_cache_bounded_read_and_strict_int_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import zlib
    from types import SimpleNamespace

    from surface_sentinel.evaluation import cache as cache_mod
    from surface_sentinel.evaluation.cache import CACHE_VERSION, CacheMeta, load_cache

    meta = CacheMeta(
        version=CACHE_VERSION,
        config_hash="a" * 64,
        split="tune",
        split_sha256="b" * 64,
        sample_ids_sha256="c" * 64,
        samples=1,
    )

    def _write_json(text: str, name: str) -> Path:
        p = tmp_path / name
        p.write_bytes(b"SSCACHE2\n" + zlib.compress(text.encode("utf-8")))
        return p

    base_sample: dict[str, object] = {
        "sample_id": "s1",
        "original_id": "s1",
        "label": "defect",
        "split": "tune",
        "source_split": "train",
        "image_path": "a.png",
        "mask_path": "",
        "defect_pixels": 0,
        "status": "OK",
        "score": 0.1,
        "reasons": [],
        "quality_flags": [],
        "blur_variance": 1.0,
        "mean_gray": 100.0,
        "saturated_ratio": 0.0,
        "roi_uncertain": False,
        "truncated": False,
        "components": [
            {
                "area_px": 1,
                "bbox": [0, 0, 1, 1],
                "mean_contrast_gray": 10.0,
                "max_contrast_gray": 12.0,
                "integrated_contrast": 10.0,
                "border_distance_px": 5.0,
                "touches_roi_border": False,
                "mask_shape": [1, 1],
                "mask_b64": "/w==",
            }
        ],
        "gt_components": [],
        "overlay": "",
        "error_code": None,
        "error_stage": None,
    }

    # 1) stat()/read_bytes() 없이 단일 바운드 읽기로 동작하고 상한 초과를 차단해야 한다.
    valid_path = _write_json(
        json.dumps({"meta": meta.as_dict(), "samples": [base_sample]}),
        "ok.sscache",
    )
    oversized_path = tmp_path / "oversized.sscache"
    oversized_path.write_bytes(b"SSCACHE2\n" + b"X" * 64)

    def _forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError("load_cache는 read_bytes() 대신 바운드 read()를 써야 합니다")

    def _lying_stat(self: Path, *args: object, **kwargs: object) -> object:
        return SimpleNamespace(st_size=0)

    monkeypatch.setattr(Path, "read_bytes", _forbid_read_bytes)
    monkeypatch.setattr(Path, "stat", _lying_stat)
    loaded = load_cache(valid_path, meta)
    assert len(loaded) == 1

    monkeypatch.setattr(cache_mod, "MAX_CACHE_COMPRESSED_BYTES", 32)
    with pytest.raises(ValueError, match="크기"):
        load_cache(oversized_path, meta)
    monkeypatch.undo()

    # 2) area_px / defect_pixels strict int (>= 0, bool/음수/float 거부)
    for bad_val in (True, -1, 1.5, "1"):
        s_bad_dp = dict(base_sample)
        s_bad_dp["defect_pixels"] = bad_val
        p = _write_json(
            json.dumps({"meta": meta.as_dict(), "samples": [s_bad_dp]}),
            f"dp_{bad_val}.sscache",
        )
        with pytest.raises(ValueError):
            load_cache(p, meta)

        s_bad_area = dict(base_sample)
        comp_bad = dict(base_sample["components"][0])  # type: ignore[index]
        comp_bad["area_px"] = bad_val
        s_bad_area["components"] = [comp_bad]
        p2 = _write_json(
            json.dumps({"meta": meta.as_dict(), "samples": [s_bad_area]}),
            f"area_{bad_val}.sscache",
        )
        with pytest.raises(ValueError):
            load_cache(p2, meta)

    # 3) 비유한(NaN/Infinity) float 필드 거부
    for field in ("score", "blur_variance", "mean_gray", "saturated_ratio"):
        s_nan = dict(base_sample)
        s_nan[field] = float("nan")
        p = _write_json(
            json.dumps({"meta": meta.as_dict(), "samples": [s_nan]}, allow_nan=True),
            f"nan_{field}.sscache",
        )
        with pytest.raises(ValueError):
            load_cache(p, meta)

    for c_field in (
        "mean_contrast_gray",
        "max_contrast_gray",
        "integrated_contrast",
        "border_distance_px",
    ):
        s_inf = dict(base_sample)
        comp_inf = dict(base_sample["components"][0])  # type: ignore[index]
        comp_inf[c_field] = float("inf")
        s_inf["components"] = [comp_inf]
        p = _write_json(
            json.dumps({"meta": meta.as_dict(), "samples": [s_inf]}, allow_nan=True),
            f"inf_{c_field}.sscache",
        )
        with pytest.raises(ValueError):
            load_cache(p, meta)
