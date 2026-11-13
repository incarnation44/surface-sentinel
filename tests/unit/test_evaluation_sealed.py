"""T08 봉인 메커니즘 단위/통합 테스트 (설계 §2.5, §4.1, §4.4).

실제 KSDD2 봉인 데이터는 쓰지 않는다. ``tests/batch_data.build_dataset``이 만드는
합성 PNG와, 그 행을 복사한 합성 봉인 CSV만 ``tmp_path``에서 쓴다. 실제
``data/splits``·``data/raw``는 읽지도 쓰지도 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import batch_data
from surface_sentinel.cli import main
from surface_sentinel.config import load_config
from surface_sentinel.evaluation.runner import (
    SEALED_SPLITS,
    EvalError,
    SealedSplitAccess,
    load_split_rows,
)
from surface_sentinel.evaluation.sampling import SampleError, build_sample_plan
from surface_sentinel.evaluation.sealed import (
    ensure_sealed_access,
    run_sealed_measurement,
)
from surface_sentinel.io.manifest import load_manifest, write_manifest_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.toml"


def _sealed_dataset(
    tmp_path: Path, split: str = "test", *, good: int = 2, defect: int = 2
):
    """합성 dataset을 만들고 tune 행을 복사해 봉인 CSV를 tmp_path에 쓴다."""

    dataset = batch_data.build_dataset(tmp_path / "ds", good=good, defect=defect)
    rows = load_manifest(dataset.tune_csv)
    write_manifest_csv(rows, dataset.splits / f"{split}.csv")
    return dataset


# ------------------------------------------------------- ensure_sealed_access


def test_ensure_sealed_access_non_sealed_returns_none(tmp_path: Path) -> None:
    result = ensure_sealed_access(
        "tune",
        splits_dir=tmp_path,
        allow_sealed=False,
        config_hash="cfg",
        code_sha="sha",
        purpose="p",
    )
    assert result is None
    assert not (tmp_path / "tune.lock.json").exists()


@pytest.mark.parametrize("split", sorted(SEALED_SPLITS))
def test_ensure_sealed_access_without_flag_raises_exit_six(
    tmp_path: Path, split: str
) -> None:
    with pytest.raises(SealedSplitAccess) as exc:
        ensure_sealed_access(
            split,
            splits_dir=tmp_path,
            allow_sealed=False,
            config_hash="cfg",
            code_sha="sha",
            purpose="p",
        )
    assert exc.value.exit_code == 6
    assert not (tmp_path / f"{split}.lock.json").exists()


def test_ensure_sealed_access_first_access_creates_lock(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test")
    payload = ensure_sealed_access(
        "test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="cfgA",
        code_sha="sha1",
        purpose="T08",
    )
    assert payload["access_count"] == 1
    assert payload["reuse"] is False
    assert payload["post_seal_config_change"] is False
    assert payload["split_sha256"]
    assert payload["config_hash"] == "cfgA"
    assert payload["code_sha"] == "sha1"
    assert payload["purpose"] == "T08"
    assert payload["created"]
    assert (dataset.splits / "test.lock.json").is_file()


def test_ensure_sealed_access_second_access_increments_reuse(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test")
    ensure_sealed_access(
        "test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="cfgA",
        code_sha="sha1",
        purpose="T08",
    )
    second = ensure_sealed_access(
        "test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="cfgA",
        code_sha="sha2",
        purpose="T08 rerun",
    )
    assert second["access_count"] == 2
    assert second["reuse"] is True
    assert second["post_seal_config_change"] is False
    assert len(second["accesses"]) == 2


def test_ensure_sealed_access_config_change_is_recorded(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test")
    ensure_sealed_access(
        "test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="cfgA",
        code_sha="sha",
        purpose="p",
    )
    second = ensure_sealed_access(
        "test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="cfgB",
        code_sha="sha",
        purpose="p",
    )
    assert second["post_seal_config_change"] is True


def test_ensure_sealed_access_official_test_uses_separate_lock(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test")
    rows = load_manifest(dataset.tune_csv)
    write_manifest_csv(rows, dataset.splits / "official_test.csv")
    ensure_sealed_access(
        "test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="c",
        code_sha="s",
        purpose="p",
    )
    ensure_sealed_access(
        "official_test",
        splits_dir=dataset.splits,
        allow_sealed=True,
        config_hash="c",
        code_sha="s",
        purpose="p",
    )
    test_lock = json.loads((dataset.splits / "test.lock.json").read_text(encoding="utf-8"))
    official_lock = json.loads(
        (dataset.splits / "official_test.lock.json").read_text(encoding="utf-8")
    )
    assert test_lock["access_count"] == 1
    assert official_lock["access_count"] == 1
    assert test_lock["split"] == "test"
    assert official_lock["split"] == "official_test"


# ----------------------------------------------- load_split_rows / sample plan


@pytest.mark.parametrize("split", sorted(SEALED_SPLITS))
def test_load_split_rows_default_still_rejects(tmp_path: Path, split: str) -> None:
    dataset = _sealed_dataset(tmp_path, split)
    with pytest.raises(SealedSplitAccess):
        load_split_rows(split, splits_dir=dataset.splits)


def test_load_split_rows_allow_sealed_reads(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test", good=2, defect=2)
    rows = load_split_rows("test", splits_dir=dataset.splits, allow_sealed=True)
    assert len(rows) == 4


def test_build_sample_plan_default_still_rejects(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test")
    rows = load_manifest(dataset.splits / "test.csv")
    with pytest.raises(SampleError):
        build_sample_plan(rows, split="test")


def test_build_sample_plan_allow_sealed_full_census(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test", good=3, defect=2)
    rows = load_manifest(dataset.splits / "test.csv")
    plan = build_sample_plan(rows, split="test", good_target=3, allow_sealed=True)
    assert plan.defect_taken == 2
    assert plan.good_taken == 3
    assert plan.split == "test"


# ---------------------------------------------------- run_sealed_measurement


def _run_kwargs(tmp_path: Path, dataset) -> dict:
    return {
        "config_path": BASE_CONFIG,
        "splits_dir": dataset.splits,
        "raw_root": dataset.raw,
        "evidence_dir": tmp_path / "evidence",
        "report_dir": tmp_path / "reports",
        "cache_dir": tmp_path / "cache",
        "overlays": False,
    }


def test_run_sealed_measurement_rejects_non_sealed(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test")
    with pytest.raises(EvalError):
        run_sealed_measurement(split="tune", **_run_kwargs(tmp_path, dataset))


def test_run_sealed_measurement_test_runs_b0_b1_p(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test", good=2, defect=2)
    payload = run_sealed_measurement(split="test", **_run_kwargs(tmp_path, dataset))
    assert set(payload["variants"]) == {"B0", "B1", "P"}
    assert payload["sample_plan"]["defect_taken"] == 2
    assert payload["sample_plan"]["good_taken"] == 2
    assert payload["lock"]["access_count"] == 1
    assert (tmp_path / "evidence" / "T08_sealed_test.json").is_file()
    assert (dataset.splits / "test.lock.json").is_file()
    assert (
        tmp_path / "evidence" / "failure_cases" / "T08_test_failure_index.csv"
    ).is_file()
    reports = list((tmp_path / "reports").glob("T08_sealed_test_*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "Wilson" in text
    assert "B0" in text and "B1" in text
    # 모든 비율에 분모가 붙는다.
    assert "분모" in text
    # 목표 판정이 실제 값과 함께 있다.
    assert "요약 판정" in text


def test_run_sealed_measurement_official_test_p_only(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "official_test", good=2, defect=2)
    payload = run_sealed_measurement(
        split="official_test", **_run_kwargs(tmp_path, dataset)
    )
    assert set(payload["variants"]) == {"P"}
    assert (dataset.splits / "official_test.lock.json").is_file()
    assert not (dataset.splits / "test.lock.json").exists()


def test_run_sealed_measurement_goals_report_actual_values(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test", good=2, defect=2)
    payload = run_sealed_measurement(split="test", **_run_kwargs(tmp_path, dataset))
    goals = payload["goals"]
    assert "escape 점추정" in goals
    assert "false-NG" in goals
    for goal in goals.values():
        assert "value" in goal
        assert "pass" in goal
        assert goal["pass"] in (True, False, None)


def test_run_sealed_measurement_is_deterministic(tmp_path: Path) -> None:
    dataset = _sealed_dataset(tmp_path, "test", good=2, defect=2)
    first = run_sealed_measurement(split="test", **_run_kwargs(tmp_path, dataset))
    second = run_sealed_measurement(split="test", **_run_kwargs(tmp_path, dataset))
    assert first["variants"] == second["variants"]
    assert first["goals"] == second["goals"]
    # 두 번째 접근은 reuse 로 기록된다.
    assert second["lock"]["access_count"] == 2
    assert second["lock"]["reuse"] is True


# ------------------------------------------------------------------- CLI 계약


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_cli_eval_sealed_without_flag_exit_six(
    tmp_path: Path, split: str, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _sealed_dataset(tmp_path, split)
    code = main(
        [
            "eval",
            "--split",
            split,
            "--config",
            str(BASE_CONFIG),
            "--splits-dir",
            str(dataset.splits),
            "--raw-root",
            str(dataset.raw),
            "--evidence-dir",
            str(tmp_path / "e"),
            "--report-dir",
            str(tmp_path / "r"),
        ]
    )
    assert code == 6
    captured = capsys.readouterr()
    assert "봉인" in captured.err
    assert not (dataset.splits / f"{split}.lock.json").exists()
    assert not (tmp_path / "e").exists()


def test_cli_eval_tune_unaffected_by_flag(tmp_path: Path) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=2, defect=2)
    config = load_config(BASE_CONFIG)
    rows = load_split_rows("tune", splits_dir=dataset.splits)
    plan = build_sample_plan(rows, split="tune")
    assert plan.split == "tune"
    assert config.config_hash
