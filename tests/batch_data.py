"""T06 배치 테스트가 공유하는 결정론적 합성 dataset 빌더.

실제 KSDD2 이미지는 쓰지 않는다. ``tests/synthetic.py``가 만드는 합성 PNG를
``raw/train``에 쓰고, 그 바이트 해시로 manifest와 tune.csv를 만든다.
이 모듈은 pytest가 수집하지 않는 이름이다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import synthetic as syn
from surface_sentinel.io.manifest import ManifestRow, write_manifest_csv


@dataclass(frozen=True)
class SyntheticDataset:
    """배치 테스트용 소량 dataset 경로 묶음."""

    root: Path
    raw: Path
    splits: Path
    manifest: Path
    sample_ids: tuple[str, ...]
    error_sample_ids: tuple[str, ...]

    @property
    def tune_csv(self) -> Path:
        return self.splits / "tune.csv"


def _encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    assert success, "합성 PNG 인코드에 실패했습니다."
    return buffer.tobytes()


def _row(
    *,
    sample_id: str,
    index: int,
    image_path: str,
    mask_path: str,
    label: str,
    width: int,
    height: int,
    channels: int,
    image_sha256: str,
    mask_sha256: str,
    defect_pixels: int,
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        original_id=sample_id,
        group_id=f"g{index:010x}",
        image_path=image_path,
        mask_path=mask_path,
        label=label,
        source_split="train",
        width=width,
        height=height,
        channels=channels,
        format="PNG",
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        defect_pixels=defect_pixels,
    )


def build_dataset(
    root: str | Path,
    *,
    good: int = 2,
    defect: int = 2,
    corrupt: int = 0,
) -> SyntheticDataset:
    """합성 good/defect/corrupt 행을 가진 manifest와 tune.csv를 만든다."""

    base = Path(root)
    raw = base / "raw"
    splits = base / "splits"
    manifest = base / "manifest.csv"
    train = raw / "train"
    train.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    sample_ids: list[str] = []
    error_ids: list[str] = []
    index = 0

    for number in range(good):
        sample_id = f"7000{number}"
        image = syn.good_image()
        payload = _encode_png(image)
        (train / f"{sample_id}.png").write_bytes(payload)
        rows.append(
            _row(
                sample_id=sample_id,
                index=index,
                image_path=f"train/{sample_id}.png",
                mask_path="",
                label="good",
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                channels=int(image.shape[2]),
                image_sha256=hashlib.sha256(payload).hexdigest(),
                mask_sha256="",
                defect_pixels=0,
            )
        )
        sample_ids.append(sample_id)
        index += 1

    for number in range(defect):
        sample_id = f"8000{number}"
        image, mask = syn.dark_defect_image()
        payload = _encode_png(image)
        mask_payload = _encode_png(mask)
        (train / f"{sample_id}.png").write_bytes(payload)
        (train / f"{sample_id}_GT.png").write_bytes(mask_payload)
        rows.append(
            _row(
                sample_id=sample_id,
                index=index,
                image_path=f"train/{sample_id}.png",
                mask_path=f"train/{sample_id}_GT.png",
                label="defect",
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                channels=int(image.shape[2]),
                image_sha256=hashlib.sha256(payload).hexdigest(),
                mask_sha256=hashlib.sha256(mask_payload).hexdigest(),
                defect_pixels=int(np.count_nonzero(mask)),
            )
        )
        sample_ids.append(sample_id)
        index += 1

    for number in range(corrupt):
        sample_id = f"6000{number}"
        payload = b"not a png"
        (train / f"{sample_id}.png").write_bytes(payload)
        rows.append(
            _row(
                sample_id=sample_id,
                index=index,
                image_path=f"train/{sample_id}.png",
                mask_path="",
                label="good",
                width=syn.WIDTH,
                height=syn.HEIGHT,
                channels=3,
                image_sha256=hashlib.sha256(payload).hexdigest(),
                mask_sha256="",
                defect_pixels=0,
            )
        )
        sample_ids.append(sample_id)
        error_ids.append(sample_id)
        index += 1

    write_manifest_csv(rows, manifest)
    write_manifest_csv(rows, splits / "tune.csv")

    return SyntheticDataset(
        root=base,
        raw=raw,
        splits=splits,
        manifest=manifest,
        sample_ids=tuple(sample_ids),
        error_sample_ids=tuple(error_ids),
    )


__all__ = ["SyntheticDataset", "build_dataset"]
