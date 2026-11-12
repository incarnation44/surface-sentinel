"""YOLO 라벨 export·group-aware 분할 단위 테스트(합성 데이터, 실데이터 미사용)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from surface_sentinel.io.manifest import ManifestRow, write_manifest_csv
from surface_sentinel.yolo.dataset import export_yolo_dataset
from surface_sentinel.yolo.labels import det_lines, mask_to_components, seg_lines
from surface_sentinel.yolo.split import group_train_val_split


def _row(
    sample_id: str,
    group_id: str,
    label: str,
    *,
    width: int = 64,
    height: int = 64,
) -> ManifestRow:
    defect = label == "defect"
    return ManifestRow(
        sample_id=sample_id,
        original_id=sample_id,
        group_id=group_id,
        image_path=f"train/{sample_id}.png",
        mask_path=f"train/{sample_id}_GT.png" if defect else "",
        label=label,
        source_split="train",
        width=width,
        height=height,
        channels=3,
        format="PNG",
        image_sha256=hashlib.sha256(sample_id.encode()).hexdigest(),
        mask_sha256="1" * 64 if defect else "",
        defect_pixels=10 if defect else 0,
    )


# ---------------------------------------------------------------- 마스크 → 컴포넌트


def test_mask_to_components_two_rects() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:20, 30:50] = 255  # x=30, y=10, w=20, h=10
    mask[60:80, 150:190] = 255  # x=150, y=60, w=40, h=20
    components = mask_to_components(mask)
    assert len(components) == 2
    by_x = sorted(components, key=lambda c: c.bbox_xywh[0])
    assert by_x[0].bbox_xywh == (30, 10, 20, 10)
    assert by_x[0].area_px == 200
    assert by_x[1].bbox_xywh == (150, 60, 40, 20)
    assert by_x[1].area_px == 800


def test_mask_to_components_min_area_filter() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0, 0] = 255  # 1px
    mask[10:14, 10:14] = 255  # 16px
    assert len(mask_to_components(mask, min_area_px=1)) == 2
    assert len(mask_to_components(mask, min_area_px=4)) == 1


def test_empty_mask_gives_no_components() -> None:
    assert mask_to_components(np.zeros((32, 32), dtype=np.uint8)) == []


# ---------------------------------------------------------------- 라벨 직렬화


def test_det_lines_normalized() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:20, 30:50] = 255
    lines = det_lines(mask_to_components(mask), width=200, height=100)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    cx, cy, w, h = (float(v) for v in parts[1:])
    assert cx == pytest.approx(40 / 200)
    assert cy == pytest.approx(15 / 100)
    assert w == pytest.approx(20 / 200)
    assert h == pytest.approx(10 / 100)
    assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))


def test_seg_lines_polygon_normalized() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:20, 30:50] = 255
    lines = seg_lines(mask_to_components(mask), width=200, height=100)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert len(coords) >= 6  # 최소 삼각형
    assert len(coords) % 2 == 0
    assert all(0.0 <= v <= 1.0 for v in coords)


def test_seg_degenerate_component_falls_back_to_bbox() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 10:20] = 255  # 가로 1px 선 → 폴리곤 퇴화
    components = mask_to_components(mask)
    assert len(components) == 1
    lines = seg_lines(components, width=64, height=64)
    coords = [float(v) for v in lines[0].split()[1:]]
    assert len(coords) == 8  # bbox 네 꼭짓점


def test_good_image_has_empty_labels() -> None:
    assert det_lines([], width=64, height=64) == []
    assert seg_lines([], width=64, height=64) == []


# ---------------------------------------------------------------- group-aware 분할


def _split_rows() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for g in range(10):
        label = "defect" if g < 4 else "good"
        for i in range(3):
            rows.append(_row(f"s{g}_{i}", f"g{g:03d}", label))
    return rows


def test_group_split_no_group_leak() -> None:
    split = group_train_val_split(_split_rows(), val_fraction=0.2, seed=7)
    train_groups = {g for g, side in split.assignment.items() if side == "train"}
    val_groups = {g for g, side in split.assignment.items() if side == "val"}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == {f"g{g:03d}" for g in range(10)}


def test_group_split_deterministic() -> None:
    a = group_train_val_split(_split_rows(), val_fraction=0.2, seed=7)
    b = group_train_val_split(_split_rows(), val_fraction=0.2, seed=7)
    assert a.assignment == b.assignment


def test_group_split_val_fraction_and_strata() -> None:
    split = group_train_val_split(_split_rows(), val_fraction=0.2, seed=7)
    assert split.counts["val"]["rows"] > 0
    assert split.counts["train"]["rows"] + split.counts["val"]["rows"] == 30
    # 결함/정상 양쪽 모두 val에 존재(층화)
    assert split.counts["val"]["defect"] > 0
    assert split.counts["val"]["good"] > 0
    ratio = split.counts["val"]["rows"] / 30
    assert 0.1 <= ratio <= 0.35


def test_group_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        group_train_val_split(_split_rows(), val_fraction=0.0)
    with pytest.raises(ValueError):
        group_train_val_split(_split_rows(), val_fraction=1.0)


# ---------------------------------------------------------------- export 종단


def _write_synthetic_dataset(root: Path, rows: list[ManifestRow]) -> None:
    for row in rows:
        image = np.full((row.height, row.width, 3), 128, dtype=np.uint8)
        image_path = root / row.image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), image)
        if row.label == "defect":
            mask = np.zeros((row.height, row.width), dtype=np.uint8)
            mask[10:20, 10:30] = 255
            cv2.imwrite(str(root / row.mask_path), mask)


def test_export_end_to_end(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    rows = _split_rows()
    _write_synthetic_dataset(raw_root, rows)
    split_csv = tmp_path / "tune.csv"
    write_manifest_csv(rows, split_csv)
    out_dir = tmp_path / "yolo"

    report = export_yolo_dataset(
        split_csv=split_csv,
        raw_root=raw_root,
        out_dir=out_dir,
        val_fraction=0.2,
        seed=7,
    )

    for task in ("det", "seg"):
        for side in ("train", "val"):
            images = sorted((out_dir / task / "images" / side).glob("*.png"))
            labels = sorted((out_dir / task / "labels" / side).glob("*.txt"))
            assert len(images) == len(labels)
            assert len(images) == report.split.counts[side]["rows"]
        yaml_text = (out_dir / task / "dataset.yaml").read_text(encoding="utf-8")
        assert "names:" in yaml_text
        assert "defect" in yaml_text

    # 결함 이미지는 라벨이 비지 않고, 정상 이미지는 빈 라벨
    defect_row = next(r for r in rows if r.label == "defect")
    side = report.split.assignment[defect_row.group_id]
    det_label = out_dir / "det" / "labels" / side / f"{defect_row.sample_id}.txt"
    assert det_label.read_text(encoding="utf-8").strip() != ""
    good_row = next(r for r in rows if r.label == "good")
    side = report.split.assignment[good_row.group_id]
    good_label = out_dir / "det" / "labels" / side / f"{good_row.sample_id}.txt"
    assert good_label.read_text(encoding="utf-8").strip() == ""

    # split.json 존재·그룹 배정 기록
    assert report.split_json_path.is_file()
    assert report.label_stats["defect_images"] == 12


# ---------------------------------------------------------------- CLI


def test_cli_yolo_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from surface_sentinel.cli import main

    raw_root = tmp_path / "raw"
    rows = _split_rows()
    _write_synthetic_dataset(raw_root, rows)
    split_csv = tmp_path / "tune.csv"
    write_manifest_csv(rows, split_csv)
    out_dir = tmp_path / "yolo"
    code = main(
        [
            "yolo",
            "export",
            "--split-csv",
            str(split_csv),
            "--raw-root",
            str(raw_root),
            "--out-dir",
            str(out_dir),
            "--seed",
            "7",
        ]
    )
    assert code == 0
    assert "YOLO export 완료" in capsys.readouterr().out
    assert (out_dir / "det" / "dataset.yaml").is_file()
    assert (out_dir / "seg" / "dataset.yaml").is_file()


def test_cli_yolo_export_rejects_unknown_task(tmp_path: Path) -> None:
    from surface_sentinel.cli import main

    code = main(["yolo", "export", "--tasks", "det,bogus", "--out-dir", str(tmp_path)])
    assert code == 2