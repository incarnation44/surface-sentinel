"""그룹 단위 `tune`/`test` 분할 및 누출 검사 단위 테스트.

부품 그룹 단위 60/40 분할, 결함 수 하한 가드, 해시 동결, 그룹/이미지/ID 누출 탐지를 검증한다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from surface_sentinel.io.manifest import ManifestRow, write_manifest_csv
from surface_sentinel.io.splits import (
    GroupAssignment,
    SplitError,
    assign_groups,
    build_groups,
    verify_splits,
    write_splits,
)


def _row(index: int, label: str, group_id: str, source_split: str = "train") -> ManifestRow:
    sample_id = f"{index:05d}"
    digest = hashlib.sha256(f"image-{index}".encode()).hexdigest()
    if label == "defect":
        mask_path = f"train/{sample_id}_GT.png"
        mask_sha = hashlib.sha256(f"mask-{index}".encode()).hexdigest()
        defect_pixels = 10
    else:
        mask_path = ""
        mask_sha = ""
        defect_pixels = 0
    return ManifestRow(
        sample_id=sample_id,
        original_id=sample_id,
        group_id=group_id,
        image_path=f"{source_split}/{sample_id}.png",
        mask_path=mask_path,
        label=label,
        source_split=source_split,
        width=64,
        height=64,
        channels=3,
        format="PNG",
        image_sha256=digest,
        mask_sha256=mask_sha,
        defect_pixels=defect_pixels,
    )


def _dataset(
    defect_groups: int,
    defect_group_size: int,
    good_groups: int,
    good_group_size: int,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    index = 0
    for group_index in range(defect_groups):
        group_id = f"g{group_index:010x}"
        for _ in range(defect_group_size):
            rows.append(_row(index, "defect", group_id))
            index += 1
    for group_index in range(good_groups):
        group_id = f"g{group_index + 1000:010x}"
        for _ in range(good_group_size):
            rows.append(_row(index, "good", group_id))
            index += 1
    return rows


def test_assign_groups_is_deterministic_and_60_40() -> None:
    rows = _dataset(10, 10, 20, 10)
    groups = build_groups(rows, [])

    first = assign_groups(groups, seed=42, p_test_min=1)
    second = assign_groups(groups, seed=42, p_test_min=1)

    assert first.assignment == second.assignment
    assert first.adjustments == second.adjustments
    assert isinstance(first, GroupAssignment)

    test_defect = sum(
        group.canonical_size
        for group in groups
        if group.label == "defect" and first.assignment[group.group_id] == "test"
    )
    test_good = sum(
        group.canonical_size
        for group in groups
        if group.label == "good" and first.assignment[group.group_id] == "test"
    )
    assert test_defect == 40
    assert test_good == 80
    assert first.targets["defect"]["test_target"] == 40
    assert first.targets["good"]["test_target"] == 80

    tune_groups = {group.group_id for group in groups if first.assignment[group.group_id] == "tune"}
    test_groups = {group.group_id for group in groups if first.assignment[group.group_id] == "test"}
    assert tune_groups.isdisjoint(test_groups)
    assert len(tune_groups | test_groups) == len(groups)


def test_assign_groups_p_test_guard_moves_defect_groups() -> None:
    rows = _dataset(12, 10, 20, 10)
    groups = build_groups(rows, [])

    assignment = assign_groups(groups, seed=42, p_test_min=100)

    test_defect = sum(
        group.canonical_size
        for group in groups
        if group.label == "defect" and assignment.assignment[group.group_id] == "test"
    )
    assert test_defect >= 100
    assert assignment.adjustments
    assert all(item["reason"] == "P_test_guard" for item in assignment.adjustments)
    assert all(item["from"] == "tune" and item["to"] == "test" for item in assignment.adjustments)
    moved = {item["group_id"] for item in assignment.adjustments}
    assert all(assignment.assignment[group_id] == "test" for group_id in moved)


def test_assign_groups_raises_when_p_test_impossible() -> None:
    rows = _dataset(5, 10, 10, 10)
    groups = build_groups(rows, [])
    with pytest.raises(SplitError, match="P_test"):
        assign_groups(groups, seed=42, p_test_min=100)


def _write_clean_splits(tmp_path: Path) -> dict[str, object]:
    rows = _dataset(25, 10, 75, 10)
    groups = build_groups(rows, [])
    assignment = assign_groups(groups, seed=42, p_test_min=100)
    out_dir = tmp_path / "splits"
    report = write_splits(
        rows,
        assignment.assignment,
        out_dir,
        adjustments=assignment.adjustments,
    )
    return {"rows": rows, "out_dir": out_dir, "report": report, "assignment": assignment}


def test_write_splits_files_hashes_and_report(tmp_path: Path) -> None:
    fixture = _write_clean_splits(tmp_path)
    out_dir = fixture["out_dir"]
    report = fixture["report"]
    assert isinstance(out_dir, Path)
    assert isinstance(report, dict)

    for name in ("tune.csv", "test.csv", "official_test.csv"):
        path = out_dir / name
        assert path.is_file()
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        header = raw.split(b"\n", 1)[0].decode("utf-8")
        assert header.startswith("sample_id,original_id,group_id,")

    test_sha = hashlib.sha256((out_dir / "test.csv").read_bytes()).hexdigest()
    assert (out_dir / "test.sha256").read_text(encoding="utf-8") == f"{test_sha}  test.csv\n"

    assert report["seed"] == 42
    assert report["test_ratio_target"] == 0.4
    assert report["manifest"]["canonical"]["total_rows"] == 1000
    assert report["splits"]["test"]["rows"] == 400
    assert report["splits"]["test"]["defect"] == 100
    assert report["official_test_overlap"]["internal_test"]["samples"] == 0

    payload = {
        "seed": report["seed"],
        "test_ratio_target": report["test_ratio_target"],
        "p_test_min": report["p_test_min"],
        "dhash_threshold": report["dhash_threshold"],
        "tune_sha256": report["splits"]["tune"]["sha256"],
        "test_sha256": report["splits"]["test"]["sha256"],
        "official_test_sha256": report["splits"]["official_test"]["sha256"],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert report["split_hash"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert "sha256(canonical_json" in report["split_hash_recipe"]

    parsed = json.loads((out_dir / "split_report.json").read_text(encoding="utf-8"))
    assert parsed["split_hash"] == report["split_hash"]


def test_verify_splits_clean_passes_without_dataset_counts(tmp_path: Path) -> None:
    fixture = _write_clean_splits(tmp_path)
    rows = fixture["rows"]
    out_dir = fixture["out_dir"]
    assert isinstance(rows, list)
    assert isinstance(out_dir, Path)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert result.violations == ()
    assert result.ok is True
    assert result.p_test == 100
    assert result.test_ratio == pytest.approx(0.4)
    assert result.l_checks["L1"]["value"] == 0
    assert result.l_checks["L2"]["value"] == 0
    assert result.l_checks["L3"]["value"] == 0
    assert result.l_checks["L4"]["status"] == "PASS"
    assert result.l_checks["L5"]["status"] == "PASS"
    assert result.l_checks["L6"]["status"] == "NOT_APPLICABLE"
    assert result.l_checks["L7"]["status"] == "REPORTED"
    assert result.l_checks["L8"]["status"] == "PASS"
    assert result.l_checks["official_test_identity"]["status"] == "PASS"
    assert result.l_checks["official_test_identity"]["missing"] == 0
    assert result.l_checks["official_test_identity"]["extra"] == 0
    assert result.l_checks["split_report"]["status"] == "PASS"
    assert result.report is not None
    assert result.report["split_hash"]


def _write_split_fixture(
    tmp_path: Path,
    tune_rows: list[ManifestRow],
    test_rows: list[ManifestRow],
    manifest_rows: list[ManifestRow],
    official_rows: list[ManifestRow] | None = None,
) -> tuple[Path, list[ManifestRow]]:
    out_dir = tmp_path / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(tune_rows, out_dir / "tune.csv")
    write_manifest_csv(test_rows, out_dir / "test.csv")
    write_manifest_csv(official_rows or [], out_dir / "official_test.csv")
    test_sha = hashlib.sha256((out_dir / "test.csv").read_bytes()).hexdigest()
    (out_dir / "test.sha256").write_text(f"{test_sha}  test.csv\n", encoding="utf-8")
    return out_dir, manifest_rows


def _base_rows() -> list[ManifestRow]:
    return [
        _row(1, "defect", "g0000000001"),
        _row(2, "defect", "g0000000002"),
        _row(3, "good", "g0000000003"),
        _row(4, "good", "g0000000004"),
    ]


def test_verify_splits_detects_group_leak(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    leaked = ManifestRow(**{**test[0].__dict__, "group_id": tune[0].group_id})
    test[0] = leaked
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("L3 위반" in message for message in result.violations)


def test_verify_splits_detects_image_hash_leak(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    leaked = ManifestRow(**{**test[0].__dict__, "image_sha256": tune[0].image_sha256})
    test[0] = leaked
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("L2 위반" in message for message in result.violations)


def test_verify_splits_detects_original_id_leak(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[0], manifest_rows[1], manifest_rows[3]]
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("L1 위반" in message for message in result.violations)


def test_verify_splits_detects_group_id_mismatch_with_manifest(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    mismatch = ManifestRow(**{**test[0].__dict__, "group_id": "g00000000ff"})
    test[0] = mismatch
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("L4 위반" in message for message in result.violations)


def test_verify_splits_detects_label_mask_mismatch(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    broken = ManifestRow(**{**test[0].__dict__, "label": "good"})
    test[0] = broken
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("L8 위반" in message for message in result.violations)


def test_verify_splits_detects_test_hash_mismatch(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)
    (out_dir / "test.sha256").write_text("deadbeef  test.csv\n", encoding="utf-8")

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("L5 위반" in message for message in result.violations)


def test_verify_splits_detects_missing_canonical_row(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0]]
    test = [manifest_rows[1]]
    out_dir, rows = _write_split_fixture(tmp_path, tune, test, manifest_rows)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)
    assert any("포함되지 않은 정본" in message for message in result.violations)


def test_verify_splits_enforces_real_dataset_counts_when_requested(tmp_path: Path) -> None:
    fixture = _write_clean_splits(tmp_path)
    rows = fixture["rows"]
    out_dir = fixture["out_dir"]
    assert isinstance(rows, list)
    assert isinstance(out_dir, Path)

    result = verify_splits(rows, out_dir, enforce_dataset_counts=True)
    assert any("데이터 수량 불일치" in message for message in result.violations)


def test_verify_splits_detects_missing_official_test_rows(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    manifest_rows[0] = replace(manifest_rows[0], source_split="test")
    manifest_rows[1] = replace(manifest_rows[1], source_split="test")
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    out_dir, rows = _write_split_fixture(
        tmp_path, tune, test, manifest_rows, official_rows=[]
    )

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert any("official_test.csv 누락" in message for message in result.violations)
    assert result.l_checks["official_test_identity"]["missing"] == 2
    assert result.counts["official_test_identity"]["expected_rows"] == 2
    assert result.counts["official_test_identity"]["actual_rows"] == 0


def test_verify_splits_detects_extra_official_test_rows(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    out_dir, rows = _write_split_fixture(
        tmp_path, tune, test, manifest_rows, official_rows=[manifest_rows[0]]
    )

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert any("official_test.csv 추가" in message for message in result.violations)
    assert result.l_checks["official_test_identity"]["extra"] == 1


def test_verify_splits_detects_official_test_row_mismatch(tmp_path: Path) -> None:
    manifest_rows = _base_rows()
    manifest_rows[0] = replace(manifest_rows[0], source_split="test")
    tune = [manifest_rows[0], manifest_rows[2]]
    test = [manifest_rows[1], manifest_rows[3]]
    official = [replace(manifest_rows[0], defect_pixels=99)]
    out_dir, rows = _write_split_fixture(
        tmp_path, tune, test, manifest_rows, official_rows=official
    )

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert any(
        "official_test.csv" in message and "defect_pixels" in message
        for message in result.violations
    )
    assert result.l_checks["official_test_identity"]["mismatch"] >= 1


def test_verify_splits_detects_report_csv_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_clean_splits(tmp_path)
    rows = fixture["rows"]
    out_dir = fixture["out_dir"]
    assert isinstance(rows, list)
    assert isinstance(out_dir, Path)
    with (out_dir / "tune.csv").open("ab") as handle:
        handle.write(b"\n")

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert any(
        "split_report.json tune sha256" in message for message in result.violations
    )
    assert result.l_checks["split_report"]["status"] == "FAIL"


def test_verify_splits_detects_tampered_split_hash(tmp_path: Path) -> None:
    fixture = _write_clean_splits(tmp_path)
    rows = fixture["rows"]
    out_dir = fixture["out_dir"]
    assert isinstance(rows, list)
    assert isinstance(out_dir, Path)
    report_path = out_dir / "split_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["split_hash"] = "0" * 64
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert any("split_hash" in message for message in result.violations)
    assert result.l_checks["split_report"]["status"] == "FAIL"


def test_verify_splits_requires_split_report(tmp_path: Path) -> None:
    fixture = _write_clean_splits(tmp_path)
    rows = fixture["rows"]
    out_dir = fixture["out_dir"]
    assert isinstance(rows, list)
    assert isinstance(out_dir, Path)
    (out_dir / "split_report.json").unlink()

    result = verify_splits(rows, out_dir, enforce_dataset_counts=False)

    assert any("split_report.json이 없습니다" in message for message in result.violations)
    assert result.l_checks["split_report"]["status"] == "FAIL"
