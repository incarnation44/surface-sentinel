"""데이터셋 전체 결함 마스크 좌표를 정규화 그리드에 누적해 공간 핫스팟 히트맵을 생성하는 모듈.

manifest의 defect 행 전체를 읽어 마스크 좌표를 정규화 그리드에 누적하고,
단일 히트맵 PNG와 통계 JSON을 만든다. 판정 파이프라인과 무관한 읽기 전용
분석 산출물이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from surface_sentinel.io.manifest import ManifestRow

DEFAULT_GRID = 64
STATS_SCHEMA_VERSION = "1.0"


class DefectMapError(ValueError):
    """결함 지도 입력이 계약과 다를 때."""


@dataclass(frozen=True)
class DefectMapResult:
    """누적 결과."""

    samples: int
    total_defect_pixels: int
    nonzero_bins: int
    pixel_hits: np.ndarray
    sample_hits: np.ndarray


def load_excluded_ids(audit_path: Path) -> set[str]:
    """감사 JSON에서 비정본 sample_id 목록을 읽는다."""
    try:
        payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DefectMapError(f"감사 JSON을 읽을 수 없습니다: {audit_path}") from exc
    except json.JSONDecodeError as exc:
        raise DefectMapError(f"감사 JSON 형식이 아닙니다: {audit_path}") from exc
    try:
        ids = payload["counts"]["non_canonical_sample_ids"]
    except (KeyError, TypeError) as exc:
        raise DefectMapError(
            f"감사 JSON에 counts.non_canonical_sample_ids가 없습니다: {audit_path}"
        ) from exc
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise DefectMapError(
            f"감사 JSON의 non_canonical_sample_ids가 문자열 리스트가 아닙니다: {audit_path}"
        )
    return set(ids)


def accumulate(
    rows: list[ManifestRow],
    raw_root: Path,
    excluded: set[str],
    grid: int = DEFAULT_GRID,
) -> DefectMapResult:
    """defect 행의 마스크 픽셀을 정규화 그리드에 누적한다."""
    pixel_hits = np.zeros((grid, grid), np.uint64)
    sample_hits = np.zeros((grid, grid), np.uint64)
    samples = 0
    total_pixels = 0
    for row in rows:
        if row.sample_id in excluded or row.label != "defect":
            continue
        mask_file = Path(raw_root) / row.mask_path
        from ..io.imageio import imread_unicode  # noqa: PLC0415

        mask = imread_unicode(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise DefectMapError(f"마스크를 디코드하지 못했습니다: {mask_file}")
        if mask.shape != (row.height, row.width):
            raise DefectMapError(
                f"마스크 크기가 manifest와 다릅니다: {mask_file} "
                f"기대=({row.height}, {row.width}) 실제={mask.shape}"
            )
        samples += 1
        ys, xs = np.nonzero(mask)
        total_pixels += int(ys.size)
        if ys.size == 0:
            continue
        gy = np.minimum(ys * grid // row.height, grid - 1)
        gx = np.minimum(xs * grid // row.width, grid - 1)
        np.add.at(pixel_hits, (gy, gx), 1)
        pairs = np.unique(np.stack([gy, gx], axis=1), axis=0)
        np.add.at(sample_hits, (pairs[:, 0], pairs[:, 1]), 1)
    return DefectMapResult(
        samples=samples,
        total_defect_pixels=total_pixels,
        nonzero_bins=int((sample_hits > 0).sum()),
        pixel_hits=pixel_hits,
        sample_hits=sample_hits,
    )


def render_heatmap(pixel_hits: np.ndarray, out_png: Path, size: int = 1280) -> None:
    """누적 그리드를 컬러 히트맵 PNG로 저장한다. 누적 0이면 검은 이미지."""
    peak = int(pixel_hits.max())
    if peak == 0:
        colored = np.zeros((*pixel_hits.shape, 3), np.uint8)
    else:
        scaled = (pixel_hits.astype(np.float64) / peak * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    resized = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", resized)
    if not success:
        raise OSError(f"히트맵 PNG 인코딩에 실패했습니다: {out_png}")
    buffer.tofile(str(out_png))


def write_stats(result: DefectMapResult, grid: int, out_json: Path) -> dict[str, Any]:
    """통계 JSON을 결정론적 순서로 저장하고 내용을 반환한다."""
    ranked: list[list[int]] = sorted(
        (pos for pos in np.argwhere(result.sample_hits > 0).tolist()),
        key=lambda pos: (-int(result.sample_hits[pos[0], pos[1]]), pos[0], pos[1]),
    )[:10]
    top_bins = [
        {
            "rank": rank + 1,
            "y_pct": round((pos[0] + 0.5) / grid * 100.0, 2),
            "x_pct": round((pos[1] + 0.5) / grid * 100.0, 2),
            "sample_coverage": (
                round(int(result.sample_hits[pos[0], pos[1]]) / result.samples, 4)
                if result.samples
                else 0.0
            ),
        }
        for rank, pos in enumerate(ranked)
    ]
    payload: dict[str, Any] = {
        "schema_version": STATS_SCHEMA_VERSION,
        "grid": grid,
        "defect_samples": result.samples,
        "total_defect_pixels": result.total_defect_pixels,
        "nonzero_bins": result.nonzero_bins,
        "top_bins": top_bins,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
