"""`tune` 분할 CSV와 마스크를 ultralytics 학습용 YOLO 디렉터리(det/seg)로 내보내는 모듈.

분할 CSV(tune)를 읽어 group-aware train/val로 나누고, task별(det/seg)
루트에 이미지(하드링크, 실패 시 복사)와 라벨을 쓴다. 각 루트에
``dataset.yaml``, 전체 루트에 ``split.json``을 남긴다.

주의: export는 ``<out>/<task>/images``와 ``<out>/<task>/labels``를
재생성한다(이전 export의 잔여 파일 방지). 파생 산출물 전용 경로에만 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from ..io.manifest import ManifestRow, load_manifest
from .labels import det_lines, mask_to_components, seg_lines
from .split import DEFAULT_SEED, DEFAULT_VAL_FRACTION, TrainValSplit, group_train_val_split

TASKS = ("det", "seg")


@dataclass(frozen=True)
class ExportReport:
    """export 결과 요약."""

    out_dir: Path
    tasks: tuple[str, ...]
    split: TrainValSplit
    label_stats: dict[str, Any]
    split_json_path: Path


def _link_or_copy(source: Path, target: Path) -> None:
    """하드링크를 우선 시도하고, 실패하면 복사한다(Windows 권한·볼륨 차이 대응)."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _dataset_yaml(task_root: Path) -> str:
    """ultralytics dataset.yaml 내용. 경로는 POSIX 형식으로 쓴다."""

    return (
        f"path: {task_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: defect\n"
    )


def export_yolo_dataset(
    *,
    split_csv: str | Path,
    raw_root: str | Path,
    out_dir: str | Path,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    tasks: tuple[str, ...] = TASKS,
    min_area_px: int = 1,
) -> ExportReport:
    """분할 CSV를 YOLO 학습 데이터셋으로 export한다."""

    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"알 수 없는 task입니다: {unknown} (가능: {TASKS})")
    rows = load_manifest(split_csv)
    split = group_train_val_split(rows, val_fraction=val_fraction, seed=seed)
    raw = Path(raw_root)
    out = Path(out_dir)

    # 이전 export의 잔여 파일이 섞이지 않도록 task별 images/labels를 재생성한다.
    for task in tasks:
        for kind in ("images", "labels"):
            stale = out / task / kind
            if stale.exists():
                shutil.rmtree(stale)

    areas: list[int] = []
    degenerate = 0
    defect_images = 0
    for row in rows:
        side = split.assignment[row.group_id]
        components = _components_for(row, raw, min_area_px=min_area_px)
        if row.label == "defect":
            defect_images += 1
            areas.extend(component.area_px for component in components)
            degenerate += sum(1 for c in components if len(c.polygon) == 4)
        for task in tasks:
            image_target = out / task / "images" / side / f"{row.sample_id}.png"
            _link_or_copy(raw / row.image_path, image_target)
            lines = (
                det_lines(components, width=row.width, height=row.height)
                if task == "det"
                else seg_lines(components, width=row.width, height=row.height)
            )
            label_target = out / task / "labels" / side / f"{row.sample_id}.txt"
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
            )

    for task in tasks:
        (out / task / "dataset.yaml").write_text(
            _dataset_yaml(out / task), encoding="utf-8", newline="\n"
        )

    label_stats: dict[str, Any] = {
        "defect_images": defect_images,
        "good_images": len(rows) - defect_images,
        "total_components": len(areas),
        "bbox_fallback_polygons": degenerate,
        "area_px_min": min(areas) if areas else 0,
        "area_px_median": statistics.median(areas) if areas else 0,
        "area_px_max": max(areas) if areas else 0,
        "min_area_px": min_area_px,
    }
    split_payload = {
        "schema": "yolo-split-v1",
        "seed": seed,
        "val_fraction": val_fraction,
        "split_csv": str(Path(split_csv).resolve().as_posix()),
        "split_csv_sha256": hashlib.sha256(Path(split_csv).read_bytes()).hexdigest(),
        "tasks": list(tasks),
        "assignment": split.assignment,
        "counts": split.counts,
        "label_stats": label_stats,
    }
    split_json_path = out / "split.json"
    out.mkdir(parents=True, exist_ok=True)
    with split_json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(split_payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")

    return ExportReport(
        out_dir=out,
        tasks=tuple(tasks),
        split=split,
        label_stats=label_stats,
        split_json_path=split_json_path,
    )


def _components_for(row: ManifestRow, raw_root: Path, *, min_area_px: int) -> list:
    """결함 행이면 GT 마스크에서 컴포넌트를 추출한다. 정상이면 빈 목록."""

    if row.label != "defect":
        return []
    mask = cv2.imread(str(raw_root / row.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"마스크를 읽을 수 없습니다: {row.sample_id} {row.mask_path}")
    if mask.shape != (row.height, row.width):
        raise ValueError(
            f"마스크 크기가 manifest와 다릅니다: {row.sample_id} "
            f"mask={mask.shape} 기대=({row.height}, {row.width})"
        )
    return mask_to_components(mask, min_area_px=min_area_px)


__all__ = ["TASKS", "ExportReport", "export_yolo_dataset"]