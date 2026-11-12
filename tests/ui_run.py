"""UI 테스트용 최소 run 디렉터리 factory (T07).

합성 PNG와 손으로 만든 결과 레코드만 쓴다. 실제 데이터셋이나 저장소의
``runs/``를 건드리지 않는다. 모든 경로는 ``tmp_path`` 아래다.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

RUN_ID = "20261112T000000Z-a762ba63-tune"
CONFIG_HASH = "a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098"

MANIFEST_COLUMNS = (
    "sample_id",
    "original_id",
    "group_id",
    "image_path",
    "mask_path",
    "label",
    "source_split",
    "width",
    "height",
    "channels",
    "format",
    "image_sha256",
    "mask_sha256",
    "defect_pixels",
)

#: 기본 샘플 구성: OK/NG/REVIEW/ERROR와 escape·오탐 실패 링크를 모두 커버한다.
DEFAULT_SAMPLES: tuple[dict[str, Any], ...] = (
    {"sample_id": "70001", "label": "good", "status": "OK", "score": 0.0},
    {"sample_id": "70002", "label": "good", "status": "REVIEW", "score": 0.30},
    {"sample_id": "70003", "label": "good", "status": "NG", "score": 1.10},
    {"sample_id": "80001", "label": "defect", "status": "NG", "score": 1.20},
    {"sample_id": "80002", "label": "defect", "status": "REVIEW", "score": 0.35},
    {"sample_id": "80003", "label": "defect", "status": "OK", "score": 0.05},
    {"sample_id": "60001", "label": "good", "status": "ERROR", "score": 0.0},
)


def _png_bytes(value: int) -> bytes:
    image = np.full((8, 8), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("합성 PNG 인코드 실패")
    return buffer.tobytes()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component() -> dict[str, Any]:
    return {
        "index": 0,
        "area_px": 48,
        "bbox": [1, 1, 4, 4],
        "area_ratio": 0.01,
        "aspect_ratio": 1.0,
        "elongation": 1.0,
        "solidity": 0.9,
        "mean_contrast_gray": 30.0,
        "max_contrast_gray": 60.0,
        "integrated_contrast": 1440.0,
        "border_distance_px": 5.0,
        "touches_roi_border": False,
        "evidence": 1.2,
    }


def _result_record(run_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    sample_id = spec["sample_id"]
    status = spec["status"]
    is_error = status == "ERROR"
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "original_id": sample_id,
        "split": "tune",
        "label": spec["label"],
        "status": status,
        "score": spec["score"],
        "reasons": [] if is_error else ["NO_DEFECT_EVIDENCE"],
        "components": [] if status in ("OK", "ERROR") else [_component()],
        "quality": None,
        "roi": None,
        "timing_ms": {"total": 1.0},
        "error_code": "E_INPUT_DECODE" if is_error else None,
        "error_stage": "load_image" if is_error else None,
        "input_sha256": _sha256_text(sample_id),
        "mask_sha256": _sha256_text(sample_id + "_mask"),
        "overlay_path": "" if is_error else f"overlays/{sample_id}_overlay.png",
        "config_hash": CONFIG_HASH,
    }


def make_run(
    runs_dir: Path,
    raw_root: Path,
    *,
    run_id: str = RUN_ID,
    samples: tuple[dict[str, Any], ...] = DEFAULT_SAMPLES,
    with_stages: bool = True,
) -> Path:
    """run 디렉터리 하나와 원본 이미지를 만들고 run_dir을 돌려준다."""

    run_dir = runs_dir / run_id
    (run_dir / "overlays").mkdir(parents=True)
    records = [_result_record(run_id, spec) for spec in samples]

    counts = {
        "samples": len(records),
        "results": len(records),
        "ok": sum(1 for r in records if r["status"] == "OK"),
        "ng": sum(1 for r in records if r["status"] == "NG"),
        "review": sum(1 for r in records if r["status"] == "REVIEW"),
        "error": sum(1 for r in records if r["status"] == "ERROR"),
        "overlays": sum(1 for r in records if r["overlay_path"]),
    }
    run_payload = {
        "run_id": run_id,
        "mode": "batch",
        "split": "tune",
        "config_path": "configs/base.toml",
        "config_hash": CONFIG_HASH,
        "code": {"version": "0.1.0"},
        "env": {"python": "3.11"},
        "manifest_sha256": "0" * 64,
        "split_sha256": "0" * 64,
        "started_at": "2026-11-12T00:00:00.000Z",
        "finished_at": "2026-11-12T00:00:01.000Z",
        "status": "COMPLETE",
        "counts": counts,
        "timing": {"p50_ms": 1.0, "p95_ms": 2.0, "wall_ms": 3.0},
        "artifacts": {"overlays_dir": "overlays"},
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )

    manifest_rows = []
    for spec in samples:
        sample_id = spec["sample_id"]
        image_path = f"tune/{sample_id}.png"
        original = raw_root / image_path
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(_png_bytes(128))
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "original_id": sample_id,
                "group_id": sample_id,
                "image_path": image_path,
                "mask_path": "",
                "label": spec["label"],
                "source_split": "tune",
                "width": "8",
                "height": "8",
                "channels": "1",
                "format": "png",
                "image_sha256": _sha256_text(sample_id),
                "mask_sha256": "",
                "defect_pixels": "0",
            }
        )
    with (run_dir / "manifest_snapshot.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    for record in records:
        if record["overlay_path"]:
            (run_dir / record["overlay_path"]).write_bytes(_png_bytes(200))
    if with_stages:
        stages = run_dir / "stages"
        stages.mkdir()
        (stages / "70002_response.png").write_bytes(_png_bytes(90))
        (stages / "70002_candidates.png").write_bytes(_png_bytes(10))
    return run_dir


__all__ = ["CONFIG_HASH", "DEFAULT_SAMPLES", "MANIFEST_COLUMNS", "RUN_ID", "make_run"]