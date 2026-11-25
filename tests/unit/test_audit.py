"""데이터셋 감사(`io/audit.py`) 단위 테스트.

손상 이미지·마스크 형상 불일치·빈 결함 마스크 탐지와 contact sheet·감사 보고서 생성을 검증한다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from surface_sentinel.io.audit import (
    verify_dataset,
    write_audit_json,
    write_audit_markdown,
    write_contact_sheet,
)
from surface_sentinel.io.manifest import ManifestRow, build_manifest, load_manifest
from surface_sentinel.io.splits import SplitVerifyResult


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))


def _image(seed: int, height: int = 64, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _mask(defect: bool, height: int = 64, width: int = 64) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    if defect:
        mask[20:30, 20:30] = 255
    return mask


def _make_dataset(root: Path) -> tuple[Path, Path]:
    train = root / "train"
    (root / "test").mkdir(parents=True, exist_ok=True)
    for index, defect in ((1, True), (2, True), (3, True), (11, False), (12, False)):
        _write_png(train / f"{index}.png", _image(index))
        _write_png(train / f"{index}_GT.png", _mask(defect))
    # 2% 그룹 상한이 유효하도록 서로 다른 dHash의 정상 행을 추가한다.
    for offset in range(60):
        seed = 5000 + offset
        _write_png(train / f"f{seed:05d}.png", _image(seed))
        _write_png(train / f"f{seed:05d}_GT.png", _mask(False))
    manifest = root.parent / "dataset_manifest.csv"
    result = build_manifest(root, manifest, None)
    assert result.success is True
    return root, manifest


def _split_result(rows: list[ManifestRow]) -> SplitVerifyResult:
    defect = sum(1 for row in rows if row.label == "defect")
    good = sum(1 for row in rows if row.label == "good")
    checks = {
        f"L{index}": {"status": "PASS", "value": 0, "detail": "테스트"}
        for index in range(1, 9)
    }
    return SplitVerifyResult(
        ok=True,
        violations=(),
        warnings=(),
        counts={
            "tune": {"total_rows": len(rows), "defect": defect, "good": good},
            "test": {"total_rows": 0, "defect": 0, "good": 0},
            "official_test": {"total_rows": 0, "defect": 0, "good": 0},
        },
        p_test=0,
        test_ratio=0.0,
        overlap={
            "internal_test": {"samples": 0, "groups": 0},
            "internal_tune": {"samples": 0, "groups": 0},
        },
        l_checks=checks,
        report={
            "seed": 42,
            "split_hash": "deadbeef",
            "split_hash_recipe": "test",
            "dhash_threshold": 2,
            "splits": {
                "tune": {
                    "rows": len(rows),
                    "defect": defect,
                    "good": good,
                    "sha256": "a" * 64,
                },
                "test": {"rows": 0, "defect": 0, "good": 0, "sha256": "b" * 64},
                "official_test": {
                    "rows": 0,
                    "defect": 0,
                    "good": 0,
                    "sha256": "c" * 64,
                },
            },
        },
    )


def test_verify_dataset_passes_on_consistent_data(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)

    audit = verify_dataset(raw, rows, _split_result(rows))

    assert audit.ok is True
    assert audit.violations == ()
    assert audit.image_stats["image_count"] == 65
    assert audit.image_stats["width_min"] == 64
    assert audit.image_stats["width_max"] == 64
    assert audit.image_stats["channels"] == {"3": 65}
    assert audit.image_stats["dtypes"] == {"uint8": 65}
    assert audit.mask_stats["mask_count"] == 65
    assert audit.mask_stats["good_masks_empty"] == 62
    assert audit.mask_stats["defect_masks_nonempty"] == 3
    assert audit.group_stats["dhash_threshold"] == 2
    assert audit.sealed_test_accessed is False
    assert any("마스크 sha256" in warning for warning in audit.warnings)


def test_verify_dataset_detects_corrupt_image(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    (raw / "train" / "1.png").write_bytes(b"broken")

    audit = verify_dataset(raw, rows, _split_result(rows))

    assert audit.ok is False
    assert any("이미지 디코드 실패" in message for message in audit.violations)


def test_verify_dataset_detects_mask_shape_mismatch(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    _write_png(raw / "train" / "1_GT.png", _mask(True, height=32, width=32))

    audit = verify_dataset(raw, rows, _split_result(rows))

    assert audit.ok is False
    assert any("마스크 크기 불일치" in message for message in audit.violations)


def test_verify_dataset_detects_empty_defect_mask(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    _write_png(raw / "train" / "1_GT.png", _mask(False))

    audit = verify_dataset(raw, rows, _split_result(rows))

    assert audit.ok is False
    assert any("L8 라벨 불일치" in message for message in audit.violations)


def test_verify_dataset_detects_nonempty_good_mask(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    _write_png(raw / "train" / "11_GT.png", _mask(True))

    audit = verify_dataset(raw, rows, _split_result(rows))

    assert audit.ok is False
    assert any("L8 라벨 불일치" in message for message in audit.violations)


def test_contact_sheet_uses_tune_rows_and_rejects_test(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)

    sheet_path = tmp_path / "contact.png"
    info = write_contact_sheet(rows, raw, sheet_path)
    assert sheet_path.is_file()
    assert info["total_rows"] == 65
    assert info["defect_rows"] == 3
    assert info["good_rows"] == 62
    image = cv2.imread(str(sheet_path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape[0] > 0

    with pytest.raises(ValueError, match="test"):
        write_contact_sheet(
            rows,
            raw,
            tmp_path / "forbidden.png",
            test_sample_ids={rows[0].sample_id},
        )


def test_audit_reports_are_written(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    audit = verify_dataset(raw, rows, _split_result(rows))

    json_path = tmp_path / "audit.json"
    md_path = tmp_path / "audit.md"
    write_audit_json(audit, json_path)
    write_audit_markdown(audit, md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["sealed_test_accessed"] is False
    markdown = md_path.read_text(encoding="utf-8")
    assert "# T03 데이터 감사" in markdown
    assert "## 2. 공식 수량 vs 로컬 측정" in markdown
    assert "## 7. 누출 검사 L1–L8" in markdown
    assert "증거 파일을 참조한다" in markdown
    assert "PASS" in markdown
    assert "a" * 64 in markdown
    assert "b" * 64 in markdown
    assert "c" * 64 in markdown
    assert "독립 감사를 제한한다" in markdown


def test_verify_dataset_uses_split_report_dhash_threshold(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    base = _split_result(rows)
    report = dict(base.report or {})
    report["dhash_threshold"] = 0
    split_result = replace(base, report=report)

    audit = verify_dataset(raw, rows, split_result)

    assert audit.group_stats["dhash_threshold"] == 0


def test_audit_fails_when_split_verification_fails(tmp_path: Path) -> None:
    raw, manifest = _make_dataset(tmp_path / "raw")
    rows = load_manifest(manifest)
    split_result = SplitVerifyResult(
        ok=False,
        violations=("고의 위반",),
        warnings=(),
        counts={},
        p_test=0,
        test_ratio=0.0,
        overlap={},
        l_checks={},
        report=None,
    )

    audit = verify_dataset(raw, rows, split_result)

    assert audit.ok is False
