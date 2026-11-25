"""누적 결함 공간 히트맵(`analysis/defect_map.py`) 단위 테스트.

픽셀-그리드 좌표 매핑, 정상/제외 샘플 필터링, 결정론적 통계·렌더링 및 CLI 실행을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from surface_sentinel.analysis import defect_map
from surface_sentinel.cli import main
from surface_sentinel.io.manifest import MANIFEST_COLUMNS, ManifestRow


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))


def _row(
    sample_id: str,
    mask_path: str,
    *,
    label: str = "defect",
    height: int = 100,
    width: int = 200,
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        original_id=sample_id,
        group_id=f"g-{sample_id}",
        image_path=mask_path.replace("_GT", ""),
        mask_path=mask_path,
        label=label,
        source_split="train",
        width=width,
        height=height,
        channels=3,
        format="PNG",
        image_sha256="0" * 64,
        mask_sha256="1" * 64,
        defect_pixels=1,
    )


def test_first_pixel_maps_to_first_bin(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    mask = np.zeros((100, 200), np.uint8)
    mask[0, 0] = 255
    _write_png(raw / "train" / "a_GT.png", mask)
    result = defect_map.accumulate([_row("a", "train/a_GT.png")], raw, set())
    assert result.samples == 1
    assert result.total_defect_pixels == 1
    assert result.pixel_hits[0, 0] == 1
    assert int(result.pixel_hits.sum()) == 1
    assert result.sample_hits[0, 0] == 1
    assert result.nonzero_bins == 1


def test_last_pixel_maps_to_last_bin(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    mask = np.zeros((100, 200), np.uint8)
    mask[99, 199] = 255
    _write_png(raw / "train" / "a_GT.png", mask)
    result = defect_map.accumulate([_row("a", "train/a_GT.png")], raw, set())
    assert result.pixel_hits[63, 63] == 1
    assert int(result.pixel_hits.sum()) == 1


def test_excluded_and_good_rows_are_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    mask = np.zeros((100, 200), np.uint8)
    mask[10, 10] = 255
    _write_png(raw / "train" / "a_GT.png", mask)
    rows = [
        _row("a", "train/a_GT.png"),
        _row("10301 (copy)", "train/a_GT.png"),
        _row("good1", "train/a_GT.png", label="good"),
    ]
    result = defect_map.accumulate(rows, raw, {"10301 (copy)"})
    assert result.samples == 1
    assert result.total_defect_pixels == 1
    assert result.nonzero_bins == 1


def test_empty_rows_render_black_and_empty_stats(tmp_path: Path) -> None:
    result = defect_map.accumulate([], tmp_path, set())
    assert result.samples == 0
    png = tmp_path / "out" / "heatmap.png"
    defect_map.render_heatmap(result.pixel_hits, png)
    image = cv2.imread(str(png))
    assert image is not None
    assert int(image.max()) == 0
    payload = defect_map.write_stats(result, 64, tmp_path / "out" / "stats.json")
    assert payload["defect_samples"] == 0
    assert payload["top_bins"] == []


def test_load_excluded_ids(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    payload = {"counts": {"non_canonical_sample_ids": ["10301 (copy)"]}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert defect_map.load_excluded_ids(path) == {"10301 (copy)"}


def test_stats_are_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    mask = np.zeros((100, 200), np.uint8)
    mask[50, 100] = 255
    _write_png(raw / "train" / "a_GT.png", mask)
    result = defect_map.accumulate([_row("a", "train/a_GT.png")], raw, set())
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    defect_map.write_stats(result, 64, first)
    defect_map.write_stats(result, 64, second)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["top_bins"][0]["rank"] == 1
    assert payload["top_bins"][0]["sample_coverage"] == 1.0


def test_cli_defect_map_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    mask = np.zeros((100, 200), np.uint8)
    mask[99, 199] = 255
    _write_png(raw / "train" / "a_GT.png", mask)
    header = ",".join(MANIFEST_COLUMNS)
    row = (
        "a,a,g-a,train/a.png,train/a_GT.png,defect,train,200,100,3,PNG,"
        + "0" * 64
        + ","
        + "1" * 64
        + ",1"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(header + "\n" + row + "\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"counts": {"non_canonical_sample_ids": []}}), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    code = main(
        [
            "defect-map",
            "--manifest",
            str(manifest),
            "--raw-root",
            str(raw),
            "--audit",
            str(audit),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    assert (out_dir / "heatmap.png").is_file()
    payload = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert payload["defect_samples"] == 1
    assert payload["nonzero_bins"] == 1
