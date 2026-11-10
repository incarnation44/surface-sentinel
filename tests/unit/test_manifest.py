"""manifest 파일명 파싱, dHash, 빌드, 로드 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from surface_sentinel.io.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    ManifestRow,
    build_manifest,
    canonical_rows,
    compute_groups,
    counts,
    dhash64,
    discover_pairs,
    load_manifest,
    parse_png_stem,
    write_manifest_csv,
)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))


def _plateau_image(seed: int, height: int = 64, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    levels = rng.choice(np.array([20, 90, 160, 240], dtype=np.uint8), size=8)
    profile = np.repeat(levels, width // 8)[:width]
    gray = np.tile(profile, (height, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mask(defect: bool, height: int = 64, width: int = 64) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    if defect:
        mask[20:30, 20:30] = 255
    return mask


def _row(
    sample_id: str,
    *,
    original_id: str | None = None,
    group_id: str = "g0000000001",
    label: str = "good",
    mask_path: str = "",
    mask_sha256: str = "",
    defect_pixels: int = 0,
    source_split: str = "train",
    image_sha256: str | None = None,
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        original_id=original_id if original_id is not None else sample_id,
        group_id=group_id,
        image_path=f"{source_split}/{sample_id}.png",
        mask_path=mask_path,
        label=label,
        source_split=source_split,
        width=64,
        height=64,
        channels=3,
        format="PNG",
        image_sha256=image_sha256 or hashlib.sha256(sample_id.encode()).hexdigest(),
        mask_sha256=mask_sha256,
        defect_pixels=defect_pixels,
    )


@pytest.mark.parametrize(
    ("stem", "kind", "base_id", "copy"),
    [
        ("10023", "image", "10023", False),
        ("10023_GT", "mask", "10023", False),
        ("10301 (copy)", "image", "10301", True),
        ("10301_GT (copy)", "mask", "10301", True),
    ],
)
def test_parse_png_stem(stem: str, kind: str, base_id: str, copy: bool) -> None:
    parsed = parse_png_stem(stem)
    assert (parsed.kind, parsed.base_id, parsed.copy) == (kind, base_id, copy)


@pytest.mark.parametrize("stem", ["_GT", " (copy)", "GT_"])
def test_parse_png_stem_rejects_unusable_names(stem: str) -> None:
    # "GT_"는 이미지로 해석되므로 별도 확인한다.
    if stem == "GT_":
        assert parse_png_stem(stem).kind == "image"
        return
    with pytest.raises(ManifestError):
        parse_png_stem(stem)


def test_dhash_identical_and_inverted() -> None:
    image = _plateau_image(7)
    assert dhash64(image) == dhash64(image.copy())
    distance = int(np.bitwise_count(np.uint64(dhash64(image) ^ dhash64(255 - image))))
    assert distance == 64


def test_dhash_different_patterns_differ() -> None:
    first = dhash64(_plateau_image(1))
    second = dhash64(_plateau_image(2))
    assert first != second
    assert 0 <= first < 2**64


def test_compute_groups_exact_and_near() -> None:
    sha_a = hashlib.sha256(b"a").hexdigest()
    sha_b = hashlib.sha256(b"b").hexdigest()
    groups = compute_groups([sha_a, sha_a, sha_b], [0b1010, 0b1010, 0b0101], threshold=0)
    assert sorted(len(group) for group in groups) == [1, 2]


def test_discover_pairs_records_errors_and_pyb(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "test").mkdir(parents=True)
    _write_png(root / "train" / "a.png", _plateau_image(1))
    _write_png(root / "train" / "_GT.png", _mask(True))
    _write_png(root / "train" / "b_GT.png", _mask(True))
    (root / "split_weakly_0.pyb").write_bytes(b"stub")

    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    stats: dict[str, int] = {}
    pairs = discover_pairs(root, errors=errors, warnings=warnings, stats=stats)

    assert [pair.base_id for pair in pairs] == ["a"]
    assert pairs[0].mask_path is None
    assert stats["pyb_files"] == 1
    assert {error["kind"] for error in errors} == {"unknown_png", "mask_without_image"}
    assert [warning["kind"] for warning in warnings] == ["mask_missing"]


def _add_filler_images(root: Path, count: int, seed_start: int) -> None:
    """2% 그룹 상한이 유효하도록 서로 다른 dHash를 가진 정상 행을 추가한다."""

    train = root / "train"
    for offset in range(count):
        seed = seed_start + offset
        image = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        _write_png(train / f"f{seed:05d}.png", image)
        _write_png(train / f"f{seed:05d}_GT.png", _mask(False))


def _build_synthetic_dataset(root: Path) -> Path:
    train = root / "train"
    (root / "test").mkdir(parents=True, exist_ok=True)
    _write_png(train / "100.png", _plateau_image(100))
    _write_png(train / "100_GT.png", _mask(True))
    _write_png(train / "200.png", _plateau_image(200))
    _write_png(train / "200_GT.png", _mask(False))
    _write_png(train / "300.png", _plateau_image(300))
    _write_png(train / "300_GT.png", _mask(True))
    _write_png(train / "300 (copy).png", _plateau_image(300))
    _write_png(train / "300_GT (copy).png", _mask(True))

    near = _plateau_image(300).copy()
    near[32, 32] = (int(near[32, 32, 0]) + 1) % 256
    _write_png(train / "301.png", near)
    _write_png(train / "301_GT.png", _mask(True))
    return root


def test_build_manifest_counts_groups_and_original_ids(tmp_path: Path) -> None:
    raw = _build_synthetic_dataset(tmp_path / "raw")
    _add_filler_images(raw, 150, 1000)
    manifest = tmp_path / "data" / "dataset_manifest.csv"
    errors_path = tmp_path / "data" / "build_errors.json"

    result = build_manifest(raw, manifest, errors_path)

    assert result.success is True
    assert result.manifest_path == manifest
    assert result.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result.counts["total_rows"] == 155
    assert result.counts["canonical_rows"] == 154
    assert result.counts["defect"] == 4
    assert result.counts["good"] == 151
    assert result.counts["duplicate_rows"] == 1
    assert result.counts["max_group_rows"] == 3
    assert result.counts["max_group_canonical"] == 2
    assert result.counts["pyb_files"] == 0

    rows = {row.sample_id: row for row in result.rows}
    assert rows["300 (copy)"].original_id == "300"
    assert rows["300 (copy)"].group_id == rows["300"].group_id
    assert rows["301"].group_id == rows["300"].group_id
    assert rows["100"].group_id != rows["200"].group_id
    assert rows["100"].label == "defect"
    assert rows["100"].defect_pixels == 100
    assert rows["200"].label == "good"
    assert rows["200"].defect_pixels == 0
    assert rows["300 (copy)"].mask_path == "train/300_GT (copy).png"

    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["group_stats"]["dhash_threshold"] == 2

    loaded = load_manifest(manifest)
    assert loaded == list(result.rows)
    canonical = canonical_rows(loaded)
    summary = counts(canonical)
    assert summary["total_rows"] == 154
    assert summary["defect"] == 3
    assert summary["good"] == 151
    assert result.counts["groups"] == len({row.group_id for row in loaded})


def test_build_manifest_decode_failure_is_fatal(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "test").mkdir(parents=True)
    _write_png(raw / "train" / "a.png", _plateau_image(1))
    _write_png(raw / "train" / "a_GT.png", _mask(True))
    (raw / "train" / "a.png").write_bytes(b"not a png")

    manifest = tmp_path / "dataset_manifest.csv"
    errors_path = tmp_path / "build_errors.json"
    result = build_manifest(raw, manifest, errors_path)

    assert result.success is False
    assert result.manifest_path is None
    assert not manifest.exists()
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert any(error["kind"] == "image_decode" for error in payload["errors"])


def test_build_manifest_missing_mask_is_warning(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "test").mkdir(parents=True)
    _write_png(raw / "train" / "a.png", _plateau_image(1))
    _add_filler_images(raw, 60, 2000)

    result = build_manifest(raw, tmp_path / "manifest.csv", tmp_path / "errors.json")

    assert result.success is True
    assert result.counts["mask_missing"] == 1
    row = next(item for item in result.rows if item.sample_id == "a")
    assert row.label == "good"
    assert row.mask_path == ""
    assert any(warning["kind"] == "mask_missing" for warning in result.warnings)


def test_load_manifest_rejects_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,original_id\n" + "a,a\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="헤더"):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    rows = [_row("a"), _row("a")]
    path = tmp_path / "manifest.csv"
    write_manifest_csv(rows, path)
    with pytest.raises(ManifestError, match="중복"):
        load_manifest(path)


def test_load_manifest_rejects_bad_label(tmp_path: Path) -> None:
    row = _row("a", label="unknown")
    path = tmp_path / "manifest.csv"
    write_manifest_csv([row], path)
    with pytest.raises(ManifestError, match="label"):
        load_manifest(path)


def test_load_manifest_rejects_defect_without_mask(tmp_path: Path) -> None:
    row = _row("a", label="defect", mask_path="", defect_pixels=10)
    path = tmp_path / "manifest.csv"
    write_manifest_csv([row], path)
    with pytest.raises(ManifestError, match="mask_path"):
        load_manifest(path)


def test_load_manifest_rejects_bad_hash_length(tmp_path: Path) -> None:
    row = _row("a", image_sha256="abc")
    path = tmp_path / "manifest.csv"
    write_manifest_csv([row], path)
    with pytest.raises(ManifestError, match="image_sha256"):
        load_manifest(path)


def test_load_manifest_rejects_unknown_original_id(tmp_path: Path) -> None:
    row = _row("a", original_id="missing")
    path = tmp_path / "manifest.csv"
    write_manifest_csv([row], path)
    with pytest.raises(ManifestError, match="original_id"):
        load_manifest(path)


def test_manifest_columns_constant_is_stable() -> None:
    assert MANIFEST_COLUMNS[0] == "sample_id"
    assert MANIFEST_COLUMNS[-1] == "defect_pixels"
    assert len(MANIFEST_COLUMNS) == 14
