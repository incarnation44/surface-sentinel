"""T05 통합 테스트: 합성 dataset에서 전 경로(캐시→지표→그리드→보고서)를 확인.

실제 KSDD2 이미지를 쓰지 않는다. ``tests/batch_data.build_dataset``이 만드는
합성 PNG만 쓴다(봉인 분할은 만들지도 읽지도 않는다).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import batch_data
from surface_sentinel.config import load_config
from surface_sentinel.evaluation.ablation import apply_overrides
from surface_sentinel.evaluation.cache import cache_meta, load_cache, save_cache
from surface_sentinel.evaluation.grid import evaluate_grid, write_jsonl
from surface_sentinel.evaluation.measure import evaluate_cached
from surface_sentinel.evaluation.runner import load_split_rows, run_evaluation
from surface_sentinel.evaluation.sampling import build_sample_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.toml"


def _samples(tmp_path: Path, *, good: int = 3, defect: int = 2):
    dataset = batch_data.build_dataset(tmp_path / "data", good=good, defect=defect)
    plan = build_sample_plan(load_split_rows("tune", splits_dir=dataset.splits))
    return dataset, plan


def test_evaluate_cached_is_deterministic(tmp_path: Path) -> None:
    dataset, plan = _samples(tmp_path)
    config = load_config(BASE_CONFIG)
    from surface_sentinel.evaluation.cache import build_cache

    samples = build_cache(plan, config, raw_root=dataset.raw, overlay_dir=None)
    first = evaluate_cached(samples, config, name="P").as_dict()
    second = evaluate_cached(samples, config, name="P").as_dict()
    assert first == second
    assert first["counts"]["samples"] == len(plan.rows)


def test_grid_is_deterministic_and_writes_same_jsonl(tmp_path: Path) -> None:
    dataset, plan = _samples(tmp_path)
    config = load_config(BASE_CONFIG)
    from surface_sentinel.evaluation.cache import build_cache

    samples = build_cache(plan, config, raw_root=dataset.raw, overlay_dir=None)
    first = [row.as_dict() for row in evaluate_grid(samples, config)]
    second = [row.as_dict() for row in evaluate_grid(samples, config)]
    assert first == second

    left = write_jsonl(evaluate_grid(samples, config), tmp_path / "grid_a.jsonl")
    right = write_jsonl(evaluate_grid(samples, config), tmp_path / "grid_b.jsonl")
    assert left.read_bytes() == right.read_bytes()
    lines = [json.loads(line) for line in left.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 12
    defect_count = sum(1 for row in plan.rows if row.label == "defect")
    for line in lines:
        # 모든 비율에 분모가 있다(escape·review의 분모는 서로 다르다).
        for key in ("escape", "false_ng", "review", "localization_recall_iou0.10"):
            assert "denominator" in line["metrics"][key]
        assert line["metrics"]["escape"]["denominator"] in {0, defect_count}
        assert line["metrics"]["review"]["denominator"] in {0, len(plan.rows)}


def test_cache_round_trip_requires_matching_meta(tmp_path: Path) -> None:
    dataset, plan = _samples(tmp_path)
    config = load_config(BASE_CONFIG)
    from surface_sentinel.evaluation.cache import build_cache

    samples = build_cache(plan, config, raw_root=dataset.raw, overlay_dir=None)
    meta = cache_meta(plan, config, split_sha256="deadbeef")
    path = save_cache(samples, meta, tmp_path / "cache" / "x.pkl")
    assert len(load_cache(path, meta)) == len(samples)

    other = cache_meta(plan, apply_overrides(config, {"scoring": {"ng_min": 0.6}}))
    try:
        load_cache(path, other)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("캐시 정체성이 다르면 거부해야 한다")


def test_run_evaluation_writes_all_artifacts(tmp_path: Path) -> None:
    dataset, _plan = _samples(tmp_path, good=3, defect=3)
    result = run_evaluation(
        split="tune",
        config_path=BASE_CONFIG,
        splits_dir=dataset.splits,
        raw_root=dataset.raw,
        evidence_dir=tmp_path / "evidence",
        report_dir=tmp_path / "reports",
        candidate_dir=tmp_path / "candidates",
        cache_dir=tmp_path / "cache",
        ablations=False,
        baselines=True,
        pilot=False,
        use_cache=True,
        overlays=True,
    )

    assert result.plan.defect_taken == 3
    assert result.plan.good_taken == 3
    # P + B0/B1/B2
    assert {report.name for report in result.reports} == {"P", "B0", "B1", "B2"}
    assert len(result.grid) == 12
    assert result.adopted_label

    paths = result.paths
    assert Path(paths["grid"]).is_file()
    assert Path(paths["variants"]).is_file()
    assert Path(paths["sample_plan"]).is_file()
    assert Path(paths["failure_index"]).is_file()
    assert Path(paths["failure_selection"]).is_file()
    assert Path(paths["report"]).is_file()

    with Path(paths["failure_index"]).open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header[:7] == [
        "sample_id",
        "split",
        "label",
        "status",
        "reason_code",
        "score",
        "overlay",
    ]

    report_text = Path(paths["report"]).read_text(encoding="utf-8")
    for heading in ("## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 6."):
        assert heading in report_text
    assert "Wilson" in report_text

    # 추천 후보 TOML은 후보 디렉터리에만 쓰고 base.toml은 건드리지 않는다.
    candidates = list((tmp_path / "candidates").glob("*.toml"))
    assert candidates
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        assert text.startswith('schema_version = "1.0"')
    assert BASE_CONFIG.read_text(encoding="utf-8").startswith('schema_version = "1.0"')


def test_run_evaluation_refuses_sealed_split(tmp_path: Path) -> None:
    import pytest

    from surface_sentinel.evaluation.runner import SealedSplitAccess

    dataset, _plan = _samples(tmp_path)
    with pytest.raises(SealedSplitAccess):
        run_evaluation(
            split="test",
            config_path=BASE_CONFIG,
            splits_dir=dataset.splits,
            raw_root=dataset.raw,
            evidence_dir=tmp_path / "evidence",
            report_dir=tmp_path / "reports",
            candidate_dir=tmp_path / "candidates",
            cache_dir=tmp_path / "cache",
            ablations=False,
            baselines=False,
            pilot=False,
            overlays=False,
        )
    assert not (tmp_path / "evidence").exists()
