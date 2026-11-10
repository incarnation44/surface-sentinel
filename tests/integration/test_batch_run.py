"""T06 batch 통합 테스트. 합성 dataset만 사용한다.

봉인 test/official_test는 접근하지 않는다. 모든 run은 ``tmp_path`` 아래에서
만들어지고 저장소의 ``runs/``나 ``data/``를 건드리지 않는다.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

import batch_data
from surface_sentinel.batch import RUN_ID_PATTERN
from surface_sentinel.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.toml"

EXPECTED_RESULT_FIELDS = (
    "run_id",
    "sample_id",
    "original_id",
    "split",
    "label",
    "status",
    "score",
    "reasons",
    "components",
    "quality",
    "roi",
    "timing_ms",
    "error_code",
    "error_stage",
    "input_sha256",
    "mask_sha256",
    "overlay_path",
    "config_hash",
)

EXPECTED_SUMMARY_COLUMNS = (
    "run_id",
    "sample_id",
    "original_id",
    "split",
    "label",
    "status",
    "score",
    "top_area_px",
    "top_contrast_gray",
    "top_evidence",
    "reason_codes",
    "component_count",
    "duration_ms",
    "error_code",
    "overlay_path",
)

EXPECTED_RUN_FIELDS = (
    "run_id",
    "mode",
    "split",
    "config_path",
    "config_hash",
    "code",
    "env",
    "manifest_sha256",
    "split_sha256",
    "started_at",
    "finished_at",
    "status",
    "counts",
    "timing",
    "artifacts",
)


def _batch_args(dataset: batch_data.SyntheticDataset, out_dir: Path, *extra: str) -> list[str]:
    return [
        "batch",
        "--split",
        "tune",
        "--config",
        str(BASE_CONFIG),
        "--manifest",
        str(dataset.manifest),
        "--splits-dir",
        str(dataset.splits),
        "--raw-root",
        str(dataset.raw),
        "--output-dir",
        str(out_dir),
        *extra,
    ]


def _run_dirs(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    return sorted(
        path
        for path in out_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".tmp-")
    )


def _read_results(run_dir: Path) -> list[dict[str, object]]:
    text = (run_dir / "results.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_run_json(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def test_batch_tune_completes_with_full_run_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=2, defect=2)
    out_dir = tmp_path / "runs"

    code = main(_batch_args(dataset, out_dir, "--limit", "4"))

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "COMPLETE" in captured.out

    runs = _run_dirs(out_dir)
    assert len(runs) == 1
    run_dir = runs[0]
    assert RUN_ID_PATTERN.match(run_dir.name)
    assert run_dir.name.endswith("-tune")

    for relative in (
        "run.json",
        "results.jsonl",
        "summary.csv",
        "summary.json",
        "config_snapshot.toml",
        "manifest_snapshot.csv",
        "logs/run.log.jsonl",
    ):
        assert (run_dir / relative).is_file(), relative
    assert (run_dir / "overlays").is_dir()
    assert [path for path in out_dir.iterdir() if path.name.startswith(".tmp-")] == []

    payload = _read_run_json(run_dir)
    assert payload["status"] == "COMPLETE"
    assert payload["mode"] == "batch"
    assert payload["split"] == "tune"
    assert payload["run_id"] == run_dir.name
    assert payload["timing"]["p50_ms"] <= payload["timing"]["p95_ms"]
    assert payload["timing"]["samples"] == 4

    records = _read_results(run_dir)
    summary_rows = _read_summary_rows(run_dir)
    counts = payload["counts"]
    assert len(records) == 4
    assert len(records) == len(summary_rows)
    assert len(records) == counts["samples"] == counts["results"] == counts["summary_rows"]
    assert counts["ok"] + counts["ng"] + counts["review"] + counts["error"] == 4
    assert payload["artifacts"]["overlay_count"] == counts["overlays"]

    for record in records:
        overlay = run_dir / str(record["overlay_path"])
        assert overlay.is_file(), overlay
        assert overlay.suffix == ".png"
    # run 디렉터리 계약에 없는 per-sample 결과 JSON이 남지 않는다.
    assert all(path.suffix == ".png" for path in (run_dir / "overlays").iterdir())

    log_lines = [
        json.loads(line)
        for line in (run_dir / "logs" / "run.log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert log_lines[0]["event"] == "run_start"
    assert log_lines[-1]["event"] == "run_end"
    assert log_lines[-1]["status"] == "COMPLETE"
    assert sum(1 for line in log_lines if line["event"] == "sample") == 4

    snapshot = (run_dir / "config_snapshot.toml").read_text(encoding="utf-8")
    assert snapshot.startswith('schema_version = "1.0"')


def test_output_files_match_design_schema_exactly(tmp_path: Path) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"

    assert main(_batch_args(dataset, out_dir)) == 0
    run_dir = _run_dirs(out_dir)[0]

    first_record = _read_results(run_dir)[0]
    assert tuple(first_record) == EXPECTED_RESULT_FIELDS

    with (run_dir / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    assert header == EXPECTED_SUMMARY_COLUMNS

    assert tuple(_read_run_json(run_dir)) == EXPECTED_RUN_FIELDS

    payload = _read_run_json(run_dir)
    assert payload["config_hash"] == first_record["config_hash"]
    assert payload["run_id"] == first_record["run_id"]
    assert payload["started_at"] <= payload["finished_at"]


def test_replace_failure_leaves_no_final_directory_and_exits_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=2, defect=1)
    out_dir = tmp_path / "runs"
    out_dir.mkdir()

    real_replace = os.replace

    def fail_on_final_replace(source: object, destination: object) -> None:
        if Path(destination).parent == out_dir:
            raise OSError("의도된 저장 실패")
        real_replace(source, destination)

    monkeypatch.setattr("surface_sentinel.io.atomic.os.replace", fail_on_final_replace)

    code = main(_batch_args(dataset, out_dir))

    captured = capsys.readouterr()
    assert code == 5
    assert "저장" in captured.err
    assert _run_dirs(out_dir) == []
    assert list(out_dir.iterdir()) == []
    assert list(out_dir.rglob("run.json")) == []


def test_count_mismatch_fails_without_complete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"

    monkeypatch.setattr(
        "surface_sentinel.batch.runner.artifacts.count_summary_rows",
        lambda path: 999,
    )

    code = main(_batch_args(dataset, out_dir))

    captured = capsys.readouterr()
    assert code == 5
    assert "일치하지 않아" in captured.err
    assert _run_dirs(out_dir) == []
    assert list(out_dir.rglob("run.json")) == []


def test_sample_error_completes_with_errors_and_exits_four(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=2, defect=1, corrupt=1)
    out_dir = tmp_path / "runs"

    code = main(_batch_args(dataset, out_dir))

    captured = capsys.readouterr()
    assert code == 4, captured.err
    assert "COMPLETE_WITH_ERRORS" in captured.out

    run_dir = _run_dirs(out_dir)[0]
    payload = _read_run_json(run_dir)
    assert payload["status"] == "COMPLETE_WITH_ERRORS"
    assert payload["counts"]["error"] == 1
    assert payload["counts"]["samples"] == 4

    records = _read_results(run_dir)
    summary_rows = _read_summary_rows(run_dir)
    assert len(records) == len(summary_rows) == payload["counts"]["samples"]
    errored = [record for record in records if record["status"] == "ERROR"]
    assert len(errored) == 1
    assert errored[0]["error_code"] == "E_INPUT_DECODE"
    assert errored[0]["overlay_path"] == ""
    assert errored[0]["sample_id"] in dataset.error_sample_ids


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_sealed_split_is_denied_with_exit_six_before_any_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    split: str,
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"

    args = _batch_args(dataset, out_dir)
    args[args.index("--split") + 1] = split
    code = main(args)

    captured = capsys.readouterr()
    assert code == 6
    assert captured.out == ""
    assert "봉인" in captured.err
    # 봉인 분할 파일은 애초에 존재하지 않는다. 접근했다면 exit 3이 났을 것이다.
    assert not (dataset.splits / f"{split}.csv").exists()
    assert not out_dir.exists()


def test_run_id_collision_increments_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"
    fixed_moment = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("surface_sentinel.batch.runner._utc_now", lambda: fixed_moment)

    assert main(_batch_args(dataset, out_dir)) == 0
    first = _run_dirs(out_dir)[0]
    first_payload = (first / "run.json").read_bytes()

    assert main(_batch_args(dataset, out_dir)) == 0
    second = _run_dirs(out_dir)[1]

    assert first.name.startswith("20261110T120000Z-")
    assert first.name.endswith("-tune")
    assert second.name == f"{first.name}-01"
    assert first_payload == (first / "run.json").read_bytes()
    assert _read_run_json(second)["run_id"] == second.name


def test_explicit_existing_run_id_is_not_overwritten(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"
    fixed_id = "20261110T120000Z-01234567-tune"

    assert main(_batch_args(dataset, out_dir, "--run-id", fixed_id)) == 0
    run_dir = out_dir / fixed_id
    original = (run_dir / "run.json").read_bytes()

    code = main(_batch_args(dataset, out_dir, "--run-id", fixed_id))

    captured = capsys.readouterr()
    assert code == 5
    assert "덮어쓰지 않습니다" in captured.err
    assert (run_dir / "run.json").read_bytes() == original
    assert _run_dirs(out_dir) == [run_dir]


def test_invalid_run_id_and_limit_are_usage_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"

    assert main(_batch_args(dataset, out_dir, "--run-id", "not-a-run-id")) == 2
    assert main(_batch_args(dataset, out_dir, "--limit", "0")) == 2

    captured = capsys.readouterr()
    assert "오류" in captured.err
    assert not out_dir.exists()


def test_missing_manifest_is_input_error(tmp_path: Path) -> None:
    dataset = batch_data.build_dataset(tmp_path / "ds", good=1, defect=1)
    out_dir = tmp_path / "runs"
    args = _batch_args(dataset, out_dir)
    args[args.index("--manifest") + 1] = str(tmp_path / "missing.csv")

    assert main(args) == 3
    assert not out_dir.exists()
