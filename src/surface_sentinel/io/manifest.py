"""KSDD2 반입본에서 이미지/마스크 페어를 찾고 manifest를 만든다.

파일 이름 규칙(실제 반입본에서 확인):

- 일반 이미지: ``<id>.png``
- 일반 마스크: ``<id>_GT.png``
- 복사본 이미지: ``<id> (copy).png``
- 복사본 마스크: ``<id>_GT (copy).png``

복사본은 ``_GT``가 `` (copy)`` 앞에 들어간다. 따라서 stem 뒤에 단순히
``_GT.png``를 붙여 마스크를 찾으면 복사본 마스크를 놓치고
``<id>_GT (copy).png``를 이미지로 오인한다. 이 모듈은 stem을 먼저
`` (copy)``로 분리한 뒤 ``_GT`` 접미사를 해석한다.

manifest의 ``image_path``/``mask_path``는 ``raw_root`` 기준 상대 POSIX
경로다. 따라서 manifest와 분할 CSV는 원본 디렉터리를 옮겨도 재사용할 수
있고, 감사 시 ``raw_root / path``로 해석한다.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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

COPY_SUFFIX = " (copy)"
MASK_SUFFIX = "_GT"
LABEL_DEFECT = "defect"
LABEL_GOOD = "good"
SOURCE_SPLITS = ("train", "test")

DHASH_THRESHOLD = 2
GROUP_SIZE_CAP_RATIO = 0.02

_FATAL_PAIR_ERROR_KINDS = frozenset(
    {
        "duplicate_image",
        "duplicate_mask",
        "mask_without_image",
        "unknown_png",
        "split_dir_missing",
    }
)


class ManifestError(ValueError):
    """manifest 입력이나 파일 이름 규칙이 계약을 위반할 때 발생한다."""


@dataclass(frozen=True)
class ParsedStem:
    """PNG stem을 해석한 결과."""

    kind: str  # "image" 또는 "mask"
    base_id: str
    copy: bool


@dataclass(frozen=True)
class ImageMaskPair:
    """이미지 하나와 그에 대응하는 마스크(없을 수 있음)."""

    source_split: str
    base_id: str
    copy: bool
    image_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class ManifestRow:
    """manifest CSV 한 행. 필드 이름과 순서는 ``MANIFEST_COLUMNS``와 같다."""

    sample_id: str
    original_id: str
    group_id: str
    image_path: str
    mask_path: str
    label: str
    source_split: str
    width: int
    height: int
    channels: int
    format: str
    image_sha256: str
    mask_sha256: str
    defect_pixels: int


@dataclass(frozen=True)
class BuildResult:
    """manifest 생성 결과와 요약 통계."""

    success: bool
    manifest_path: Path | None
    manifest_sha256: str
    counts: dict[str, int]
    group_stats: dict[str, Any]
    errors: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    rows: tuple[ManifestRow, ...] = ()


def parse_png_stem(stem: str) -> ParsedStem:
    """PNG stem을 이미지/마스크로 분류한다.

    `` (copy)``를 먼저 제거하고 그 다음 ``_GT``를 확인한다. 그래야
    ``10301_GT (copy)``를 복사본 마스크로 올바르게 해석한다.
    """

    if not stem:
        raise ManifestError("빈 파일 이름은 해석할 수 없습니다.")

    copy = stem.endswith(COPY_SUFFIX)
    base = stem[: -len(COPY_SUFFIX)] if copy else stem
    if not base:
        raise ManifestError(f"파일 이름 규칙을 해석할 수 없습니다: {stem!r}")

    if base.endswith(MASK_SUFFIX):
        kind = "mask"
        base_id = base[: -len(MASK_SUFFIX)]
    else:
        kind = "image"
        base_id = base

    if not base_id:
        raise ManifestError(f"파일 이름 규칙을 해석할 수 없습니다: {stem!r}")
    return ParsedStem(kind=kind, base_id=base_id, copy=copy)


def discover_pairs(
    raw_root: str | Path,
    *,
    errors: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    stats: dict[str, int] | None = None,
) -> list[ImageMaskPair]:
    """``raw_root`` 아래 train/test PNG를 페어링한다.

    오류는 ``errors``, 경고는 ``warnings``, 개수 통계는 ``stats``에 누적한다.
    ``split_weakly_*.pyb`` 파일은 사용하지 않지만 개수만 기록한다.
    """

    root = Path(raw_root)
    error_list = errors if errors is not None else []
    warning_list = warnings if warnings is not None else []
    stat_map = stats if stats is not None else {}

    if not root.is_dir():
        raise ManifestError(f"원본 디렉터리를 찾을 수 없습니다: {root}")

    stat_map["pyb_files"] = len(list(root.glob("split_weakly_*.pyb")))
    stat_map["png_files"] = 0

    images: dict[tuple[str, str, bool], Path] = {}
    masks: dict[tuple[str, str, bool], Path] = {}

    for source_split in SOURCE_SPLITS:
        split_dir = root / source_split
        if not split_dir.is_dir():
            error_list.append(
                {
                    "kind": "split_dir_missing",
                    "path": str(split_dir),
                    "message": f"공식 분할 디렉터리가 없습니다: {split_dir}",
                }
            )
            continue
        for path in sorted(split_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            stat_map["png_files"] += 1
            try:
                parsed = parse_png_stem(path.stem)
            except ManifestError as exc:
                error_list.append(
                    {
                        "kind": "unknown_png",
                        "path": str(path),
                        "message": str(exc),
                    }
                )
                continue

            key = (source_split, parsed.base_id, parsed.copy)
            target = images if parsed.kind == "image" else masks
            duplicate_kind = "duplicate_image" if parsed.kind == "image" else "duplicate_mask"
            if key in target:
                error_list.append(
                    {
                        "kind": duplicate_kind,
                        "path": str(path),
                        "other": str(target[key]),
                        "message": (
                            f"같은 키에 {parsed.kind} 파일이 둘 이상 있습니다: "
                            f"{source_split}/{parsed.base_id} (copy={parsed.copy})"
                        ),
                    }
                )
                continue
            target[key] = path

    pairs: list[ImageMaskPair] = []
    for key in sorted(set(images) | set(masks)):
        source_split, base_id, copy = key
        image_path = images.get(key)
        mask_path = masks.get(key)
        if image_path is None:
            error_list.append(
                {
                    "kind": "mask_without_image",
                    "path": str(mask_path) if mask_path is not None else "",
                    "message": (
                        f"이미지 없이 마스크만 있습니다: {source_split}/{base_id} "
                        f"(copy={copy})"
                    ),
                }
            )
            continue
        if mask_path is None:
            warning_list.append(
                {
                    "kind": "mask_missing",
                    "path": str(image_path),
                    "sample_id": image_path.stem,
                    "message": f"이미지에 대응하는 마스크가 없습니다: {image_path}",
                }
            )
        pairs.append(
            ImageMaskPair(
                source_split=source_split,
                base_id=base_id,
                copy=copy,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    stat_map["pair_count"] = len(pairs)
    return pairs


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ManifestError(f"지원하지 않는 이미지 차원입니다: shape={image.shape}")
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ManifestError(f"지원하지 않는 채널 수입니다: {channels}")


def dhash64(image: np.ndarray) -> int:
    """9x8 INTER_AREA 축소 후 가로 인접 비교로 64bit dHash를 만든다."""

    gray = _to_gray(image)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


class _UnionFind:
    __slots__ = ("_parent", "_rank")

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, index: int) -> int:
        parent = self._parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank = self._rank
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1


def compute_groups(
    image_sha256: Sequence[str],
    dhashes: Sequence[int],
    threshold: int = DHASH_THRESHOLD,
) -> list[list[int]]:
    """정확히 같은 image sha와 dHash Hamming ``threshold`` 이하를 합친다.

    반환값은 각 그룹의 인덱스 목록이며, 그룹은 첫 인덱스 기준으로 정렬된다.
    """

    if len(image_sha256) != len(dhashes):
        raise ManifestError("image_sha256와 dHash 길이가 다릅니다.")
    count = len(image_sha256)
    if count == 0:
        return []

    union_find = _UnionFind(count)

    seen_by_sha: dict[str, int] = {}
    for index, digest in enumerate(image_sha256):
        previous = seen_by_sha.get(digest)
        if previous is None:
            seen_by_sha[digest] = index
        else:
            union_find.union(index, previous)

    if count > 1 and threshold >= 0:
        hashes = np.asarray(dhashes, dtype=np.uint64)
        for index in range(count - 1):
            distances = np.bitwise_count(hashes[index] ^ hashes[index + 1 :])
            for offset in np.nonzero(distances <= threshold)[0]:
                union_find.union(index, index + 1 + int(offset))

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(union_find.find(index), []).append(index)
    return sorted(grouped.values(), key=lambda group: group[0])


def read_png(path: str | Path) -> tuple[np.ndarray, str]:
    """PNG 파일을 디코드하고 파일 바이트 sha256과 함께 반환한다."""

    source = Path(path)
    buffer = np.fromfile(str(source), dtype=np.uint8)
    if buffer.size == 0:
        raise ManifestError(f"빈 파일입니다: {source}")
    digest = hashlib.sha256(buffer.tobytes()).hexdigest()
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ManifestError(f"PNG 디코드에 실패했습니다: {source}")
    return image, digest


def _mask_defect_pixels(mask: np.ndarray) -> int:
    if mask.ndim == 2:
        return int(np.count_nonzero(mask))
    if mask.ndim == 3:
        return int(np.count_nonzero(np.any(mask != 0, axis=2)))
    raise ManifestError(f"지원하지 않는 마스크 차원입니다: shape={mask.shape}")


@dataclass(frozen=True)
class _DecodedPair:
    pair: ImageMaskPair
    width: int
    height: int
    channels: int
    image_sha256: str
    mask_sha256: str
    defect_pixels: int
    dhash: int


def _zero_counts(pyb_files: int = 0) -> dict[str, int]:
    return {
        "total_rows": 0,
        "canonical_rows": 0,
        "defect": 0,
        "good": 0,
        "official_train": 0,
        "official_test": 0,
        "duplicate_rows": 0,
        "groups": 0,
        "max_group_rows": 0,
        "max_group_canonical": 0,
        "mask_missing": 0,
        "mask_shape_mismatch": 0,
        "decode_errors": 0,
        "pyb_files": pyb_files,
    }


def _default_group_stats() -> dict[str, Any]:
    return {
        "dhash_threshold": DHASH_THRESHOLD,
        "group_size_cap_ratio": GROUP_SIZE_CAP_RATIO,
        "group_size_cap_rows": 0.0,
        "group_rebuild_applied": False,
        "groups": 0,
        "groups_over_cap": 0,
        "max_group_rows": 0,
        "max_group_canonical": 0,
    }


def _write_errors_json(
    errors_path: Path | None,
    *,
    success: bool,
    raw_root: Path,
    manifest_path: Path | None,
    counts: dict[str, int],
    group_stats: dict[str, Any],
    errors: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> None:
    if errors_path is None:
        return
    payload = {
        "success": success,
        "raw_root": str(raw_root),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "counts": counts,
        "group_stats": group_stats,
        "errors": list(errors),
        "warnings": list(warnings),
    }
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_manifest_csv(rows: Iterable[ManifestRow], path: str | Path) -> str:
    """manifest/분할 CSV를 결정론적으로 쓰고 파일 sha256을 반환한다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in MANIFEST_COLUMNS})
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _canonical_key(decoded: _DecodedPair) -> tuple[bool, int, str]:
    stem = decoded.pair.image_path.stem
    return (decoded.pair.copy, len(stem), stem)


def build_manifest(
    raw_root: str | Path,
    manifest_path: str | Path,
    errors_path: str | Path | None = None,
) -> BuildResult:
    """반입본 전체를 디코드하고 manifest를 만든다.

    디코드 실패는 치명적이며 manifest를 쓰지 않는다. 마스크 누락과 크기
    불일치는 경고로 기록하되 manifest는 쓴다.
    """

    root = Path(raw_root)
    manifest_target = Path(manifest_path)
    errors_target = Path(errors_path) if errors_path is not None else None

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stats: dict[str, int] = {}

    try:
        pairs = discover_pairs(root, errors=errors, warnings=warnings, stats=stats)
    except ManifestError as exc:
        errors.append({"kind": "raw_root", "path": str(root), "message": str(exc)})
        summary = _zero_counts()
        group_stats = _default_group_stats()
        _write_errors_json(
            errors_target,
            success=False,
            raw_root=root,
            manifest_path=None,
            counts=summary,
            group_stats=group_stats,
            errors=errors,
            warnings=warnings,
        )
        return BuildResult(False, None, "", summary, group_stats, tuple(errors), tuple(warnings))

    pyb_files = stats.get("pyb_files", 0)
    fatal = [error for error in errors if error["kind"] in _FATAL_PAIR_ERROR_KINDS]
    if not pairs and not fatal:
        errors.append(
            {
                "kind": "no_pairs",
                "path": str(root),
                "message": "이미지/마스크 페어를 하나도 찾지 못했습니다.",
            }
        )
    if fatal or not pairs:
        summary = _zero_counts(pyb_files)
        group_stats = _default_group_stats()
        _write_errors_json(
            errors_target,
            success=False,
            raw_root=root,
            manifest_path=None,
            counts=summary,
            group_stats=group_stats,
            errors=errors,
            warnings=warnings,
        )
        return BuildResult(False, None, "", summary, group_stats, tuple(errors), tuple(warnings))

    decoded: list[_DecodedPair] = []
    decode_errors: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            image, image_sha256 = read_png(pair.image_path)
        except (ManifestError, OSError) as exc:
            decode_errors.append(
                {
                    "kind": "image_decode",
                    "path": str(pair.image_path),
                    "message": str(exc),
                }
            )
            continue

        height = int(image.shape[0])
        width = int(image.shape[1])
        channels = 1 if image.ndim == 2 else int(image.shape[2])

        mask_sha256 = ""
        defect_pixels = 0
        if pair.mask_path is not None:
            try:
                mask, mask_sha256 = read_png(pair.mask_path)
            except (ManifestError, OSError) as exc:
                decode_errors.append(
                    {
                        "kind": "mask_decode",
                        "path": str(pair.mask_path),
                        "message": str(exc),
                    }
                )
                continue
            if mask.ndim < 2 or tuple(mask.shape[:2]) != (height, width):
                warnings.append(
                    {
                        "kind": "mask_shape_mismatch",
                        "path": str(pair.mask_path),
                        "sample_id": pair.image_path.stem,
                        "image_shape": [height, width],
                        "mask_shape": list(mask.shape),
                        "message": (
                            f"마스크 크기가 이미지와 다릅니다: {pair.mask_path} "
                            f"image={(height, width)} mask={mask.shape[:2]}"
                        ),
                    }
                )
            defect_pixels = _mask_defect_pixels(mask)

        decoded.append(
            _DecodedPair(
                pair=pair,
                width=width,
                height=height,
                channels=channels,
                image_sha256=image_sha256,
                mask_sha256=mask_sha256,
                defect_pixels=defect_pixels,
                dhash=dhash64(image),
            )
        )

    if decode_errors:
        errors.extend(decode_errors)
        summary = _zero_counts(pyb_files)
        summary["decode_errors"] = len(decode_errors)
        group_stats = _default_group_stats()
        _write_errors_json(
            errors_target,
            success=False,
            raw_root=root,
            manifest_path=None,
            counts=summary,
            group_stats=group_stats,
            errors=errors,
            warnings=warnings,
        )
        return BuildResult(False, None, "", summary, group_stats, tuple(errors), tuple(warnings))

    image_shas = [item.image_sha256 for item in decoded]
    dhashes = [item.dhash for item in decoded]
    threshold = DHASH_THRESHOLD
    groups = compute_groups(image_shas, dhashes, threshold)
    cap_rows = GROUP_SIZE_CAP_RATIO * len(decoded)
    over_cap = [group for group in groups if len(group) > cap_rows]
    rebuild_applied = False
    if over_cap and threshold > 0:
        threshold = 0
        rebuild_applied = True
        groups = compute_groups(image_shas, dhashes, threshold)
        over_cap = [group for group in groups if len(group) > cap_rows]

    if over_cap:
        largest = max(len(group) for group in over_cap)
        errors.append(
            {
                "kind": "group_size_cap",
                "message": (
                    "dHash threshold 0에서도 전체 행의 2%를 넘는 그룹이 있어 "
                    f"중단합니다. 최대 그룹 {largest}행, 허용 {cap_rows:.2f}행. "
                    "중복/부품 식별 설계의 모호성으로 보고합니다."
                ),
                "groups_over_cap": len(over_cap),
            }
        )
        summary = _zero_counts(pyb_files)
        group_stats = _default_group_stats()
        group_stats["dhash_threshold"] = threshold
        group_stats["group_size_cap_rows"] = cap_rows
        group_stats["group_rebuild_applied"] = rebuild_applied
        group_stats["groups_over_cap"] = len(over_cap)
        group_stats["max_group_rows"] = largest
        _write_errors_json(
            errors_target,
            success=False,
            raw_root=root,
            manifest_path=None,
            counts=summary,
            group_stats=group_stats,
            errors=errors,
            warnings=warnings,
        )
        return BuildResult(False, None, "", summary, group_stats, tuple(errors), tuple(warnings))

    canonical_by_sha: dict[str, int] = {}
    for index, item in enumerate(decoded):
        current = canonical_by_sha.get(item.image_sha256)
        if current is None or _canonical_key(item) < _canonical_key(decoded[current]):
            canonical_by_sha[item.image_sha256] = index

    original_ids = [
        decoded[canonical_by_sha[item.image_sha256]].pair.image_path.stem
        for item in decoded
    ]

    group_ids: dict[int, str] = {}
    max_group_canonical = 0
    for group in groups:
        origins = sorted({original_ids[index] for index in group})
        digest = hashlib.sha256("\n".join(origins).encode("utf-8")).hexdigest()
        group_id = "g" + digest[:10]
        canonical_count = sum(
            1
            for index in group
            if decoded[index].pair.image_path.stem == original_ids[index]
        )
        max_group_canonical = max(max_group_canonical, canonical_count)
        for index in group:
            group_ids[index] = group_id

    rows: list[ManifestRow] = []
    for index, item in enumerate(decoded):
        sample_id = item.pair.image_path.stem
        mask_path = (
            item.pair.mask_path.relative_to(root).as_posix()
            if item.pair.mask_path is not None
            else ""
        )
        rows.append(
            ManifestRow(
                sample_id=sample_id,
                original_id=original_ids[index],
                group_id=group_ids[index],
                image_path=item.pair.image_path.relative_to(root).as_posix(),
                mask_path=mask_path,
                label=LABEL_DEFECT if item.defect_pixels > 0 else LABEL_GOOD,
                source_split=item.pair.source_split,
                width=item.width,
                height=item.height,
                channels=item.channels,
                format="PNG",
                image_sha256=item.image_sha256,
                mask_sha256=item.mask_sha256,
                defect_pixels=item.defect_pixels,
            )
        )
    rows.sort(key=lambda row: (row.source_split, row.sample_id))

    manifest_sha256 = write_manifest_csv(rows, manifest_target)

    summary = counts(rows)
    summary["groups"] = len(groups)
    summary["max_group_rows"] = max((len(group) for group in groups), default=0)
    summary["max_group_canonical"] = max_group_canonical
    summary["mask_missing"] = sum(1 for warning in warnings if warning["kind"] == "mask_missing")
    summary["mask_shape_mismatch"] = sum(
        1 for warning in warnings if warning["kind"] == "mask_shape_mismatch"
    )
    summary["decode_errors"] = 0
    summary["pyb_files"] = pyb_files

    group_stats = {
        "dhash_threshold": threshold,
        "group_size_cap_ratio": GROUP_SIZE_CAP_RATIO,
        "group_size_cap_rows": cap_rows,
        "group_rebuild_applied": rebuild_applied,
        "groups": len(groups),
        "groups_over_cap": 0,
        "max_group_rows": summary["max_group_rows"],
        "max_group_canonical": max_group_canonical,
    }

    _write_errors_json(
        errors_target,
        success=True,
        raw_root=root,
        manifest_path=manifest_target,
        counts=summary,
        group_stats=group_stats,
        errors=errors,
        warnings=warnings,
    )

    return BuildResult(
        True,
        manifest_target,
        manifest_sha256,
        summary,
        group_stats,
        tuple(errors),
        tuple(warnings),
        tuple(rows),
    )


def _parse_int(value: str, column: str, line_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            f"{line_number}행의 {column} 값이 정수가 아닙니다: {value!r}"
        ) from exc


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """manifest CSV를 읽고 스키마와 행 단위 계약을 검증한다."""

    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ManifestError(f"manifest를 읽을 수 없습니다: {source}") from exc

    with handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ManifestError(f"manifest가 비어 있습니다: {source}") from exc
        if header != MANIFEST_COLUMNS:
            raise ManifestError(
                f"manifest 헤더가 계약과 다릅니다: {source} "
                f"기대={MANIFEST_COLUMNS} 실제={header}"
            )

        rows: list[ManifestRow] = []
        for line_number, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(MANIFEST_COLUMNS):
                raise ManifestError(
                    f"{line_number}행의 열 수가 {len(MANIFEST_COLUMNS)}개가 아닙니다: {len(values)}"
                )
            payload = dict(zip(MANIFEST_COLUMNS, values, strict=True))
            rows.append(
                ManifestRow(
                    sample_id=payload["sample_id"],
                    original_id=payload["original_id"],
                    group_id=payload["group_id"],
                    image_path=payload["image_path"],
                    mask_path=payload["mask_path"],
                    label=payload["label"],
                    source_split=payload["source_split"],
                    width=_parse_int(payload["width"], "width", line_number),
                    height=_parse_int(payload["height"], "height", line_number),
                    channels=_parse_int(payload["channels"], "channels", line_number),
                    format=payload["format"],
                    image_sha256=payload["image_sha256"],
                    mask_sha256=payload["mask_sha256"],
                    defect_pixels=_parse_int(
                        payload["defect_pixels"], "defect_pixels", line_number
                    ),
                )
            )

    sample_ids: set[str] = set()
    for row in rows:
        if not row.sample_id:
            raise ManifestError("sample_id가 비어 있는 행이 있습니다.")
        if row.sample_id in sample_ids:
            raise ManifestError(f"sample_id가 중복되었습니다: {row.sample_id}")
        sample_ids.add(row.sample_id)

    for row in rows:
        if not row.original_id:
            raise ManifestError(f"{row.sample_id}: original_id가 비어 있습니다.")
        if row.original_id not in sample_ids:
            raise ManifestError(
                f"{row.sample_id}: original_id가 manifest의 sample_id에 없습니다: "
                f"{row.original_id}"
            )
        if not row.group_id:
            raise ManifestError(f"{row.sample_id}: group_id가 비어 있습니다.")
        if row.label not in (LABEL_DEFECT, LABEL_GOOD):
            raise ManifestError(f"{row.sample_id}: 알 수 없는 label입니다: {row.label}")
        if row.source_split not in SOURCE_SPLITS:
            raise ManifestError(
                f"{row.sample_id}: 알 수 없는 source_split입니다: {row.source_split}"
            )
        if len(row.image_sha256) != 64:
            raise ManifestError(
                f"{row.sample_id}: image_sha256 길이가 64가 아닙니다: {row.image_sha256!r}"
            )
        if row.mask_path:
            if len(row.mask_sha256) != 64:
                raise ManifestError(
                    f"{row.sample_id}: mask_sha256 길이가 64가 아닙니다: {row.mask_sha256!r}"
                )
        elif row.mask_sha256:
            raise ManifestError(
                f"{row.sample_id}: mask_path가 비었는데 mask_sha256이 있습니다."
            )
        if row.defect_pixels < 0:
            raise ManifestError(
                f"{row.sample_id}: defect_pixels가 음수입니다: {row.defect_pixels}"
            )
        if row.label == LABEL_DEFECT and not row.mask_path:
            raise ManifestError(
                f"{row.sample_id}: 결함 행에 mask_path가 비어 있습니다."
            )

    return rows


def canonical_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """``sample_id == original_id``인 정본 행만 반환한다."""

    return [row for row in rows if row.sample_id == row.original_id]


def counts(rows: Iterable[ManifestRow]) -> dict[str, int]:
    """행 목록의 수량 요약을 계산한다."""

    materialized = list(rows)
    return {
        "total_rows": len(materialized),
        "canonical_rows": sum(1 for row in materialized if row.sample_id == row.original_id),
        "defect": sum(1 for row in materialized if row.label == LABEL_DEFECT),
        "good": sum(1 for row in materialized if row.label == LABEL_GOOD),
        "official_train": sum(1 for row in materialized if row.source_split == "train"),
        "official_test": sum(1 for row in materialized if row.source_split == "test"),
        "duplicate_rows": sum(1 for row in materialized if row.sample_id != row.original_id),
    }


__all__ = [
    "COPY_SUFFIX",
    "DHASH_THRESHOLD",
    "GROUP_SIZE_CAP_RATIO",
    "ImageMaskPair",
    "MANIFEST_COLUMNS",
    "ManifestError",
    "ManifestRow",
    "BuildResult",
    "ParsedStem",
    "build_manifest",
    "canonical_rows",
    "compute_groups",
    "counts",
    "dhash64",
    "discover_pairs",
    "load_manifest",
    "parse_png_stem",
    "read_png",
    "write_manifest_csv",
]
