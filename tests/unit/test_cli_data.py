"""CLI `data` 서브커맨드 단위 테스트.

합성 데이터셋에서 `build-manifest`, `make-splits`, `verify` 명령의 정상 동작과 도움말을 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from surface_sentinel.cli import main


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))


def _make_dataset(root: Path, defect_count: int = 110, good_count: int = 300) -> None:
    train = root / "train"
    (root / "test").mkdir(parents=True, exist_ok=True)
    for index in range(defect_count):
        image = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[20:30, 20:30] = 255
        _write_png(train / f"d{index:04d}.png", image)
        _write_png(train / f"d{index:04d}_GT.png", mask)
    for index in range(good_count):
        image = np.random.default_rng(10_000 + index).integers(
            0, 256, (64, 64, 3), dtype=np.uint8
        )
        _write_png(train / f"g{index:04d}.png", image)
        _write_png(train / f"g{index:04d}_GT.png", np.zeros((64, 64), np.uint8))


def test_data_help_is_korean_and_lists_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "데이터" in output
    assert "build-manifest" in output
    assert "make-splits" in output
    assert "verify" in output


def test_data_without_subcommand_returns_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["data"]) == 2
    assert "data 하위 명령" in capsys.readouterr().err


def test_data_commands_run_end_to_end_on_synthetic_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = tmp_path / "raw"
    _make_dataset(raw)
    manifest = tmp_path / "manifests" / "dataset_manifest.csv"
    errors = tmp_path / "manifests" / "build_errors.json"

    assert (
        main(
            [
                "data",
                "build-manifest",
                "--raw",
                str(raw),
                "--manifest",
                str(manifest),
                "--errors",
                str(errors),
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "manifest 생성 완료" in build_output
    assert (
        "전체 410행 (결함 110 포함) / 정본 410 (결함 110 + 정상 300)"
        in build_output
    )
    assert manifest.is_file()
    assert errors.is_file()

    splits = tmp_path / "splits"
    assert (
        main(
            [
                "data",
                "make-splits",
                "--manifest",
                str(manifest),
                "--out-dir",
                str(splits),
                "--seed",
                "42",
            ]
        )
        == 0
    )
    split_output = capsys.readouterr().out
    assert "분할 생성 완료" in split_output
    assert (splits / "test.sha256").is_file()

    # 합성 데이터는 공식 수량(3,335/2,331/1,004)과 test 비율 계약을
    # 만족하지 않으므로 verify는 실패해야 한다. 실패를 숨기지 않는다.
    audit_json = tmp_path / "audit" / "T03_audit.json"
    audit_md = tmp_path / "T03_data_audit.md"
    sheet = tmp_path / "derived" / "tune_contact_sheet.png"
    assert (
        main(
            [
                "data",
                "verify",
                "--raw",
                str(raw),
                "--manifest",
                str(manifest),
                "--splits-dir",
                str(splits),
                "--audit-json",
                str(audit_json),
                "--audit-md",
                str(audit_md),
                "--contact-sheet",
                str(sheet),
            ]
        )
        == 3
    )
    verify_output = capsys.readouterr()
    assert "분할 검증: FAIL" in verify_output.out
    assert "데이터 수량 불일치" in verify_output.out
    assert audit_json.is_file()
    assert audit_md.is_file()
    assert sheet.is_file()
