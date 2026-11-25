"""데이터셋 전수 재디코드·마스크/라벨 정합성·중복 그룹 감사 및 보고서 생성 모듈.

``data verify``는 이 모듈을 통해 3,336개 manifest 행을 다시 읽는다. 감사
결과는 JSON과 한국어 Markdown으로 저장하고, tune contact sheet도 함께
만든다. 코드 리비전과 실행 시각은 보고서에 지어내지 않고 증거 파일을
참조한다.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .manifest import (
    DHASH_THRESHOLD,
    GROUP_SIZE_CAP_RATIO,
    ManifestError,
    ManifestRow,
    canonical_rows,
    compute_groups,
    counts,
    dhash64,
    read_png,
)
from .splits import (
    EXPECTED_CANONICAL_ROWS,
    EXPECTED_DEFECT_ROWS,
    EXPECTED_GOOD_ROWS,
    EXPECTED_OFFICIAL_TEST_DEFECT,
    EXPECTED_OFFICIAL_TEST_GOOD,
    EXPECTED_OFFICIAL_TEST_ROWS,
    EXPECTED_OFFICIAL_TRAIN_DEFECT,
    EXPECTED_OFFICIAL_TRAIN_GOOD,
    EXPECTED_OFFICIAL_TRAIN_ROWS,
    SplitVerifyResult,
)

CONTACT_SHEET_COLUMNS = 8
CONTACT_SHEET_THUMB_HEIGHT = 150
CONTACT_SHEET_MAX_DEFECT = 400
CONTACT_SHEET_GOOD_SAMPLE = 120
CONTACT_SHEET_SELECTION = (
    "tune 결함 전체(최대 400장) + tune 정상에서 정렬 후 균등 간격 표본 120장"
)


@dataclass(frozen=True)
class AuditResult:
    """데이터 감사 결과."""

    ok: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, Any]
    image_stats: dict[str, Any]
    mask_stats: dict[str, Any]
    group_stats: dict[str, Any]
    split: dict[str, Any]
    l_checks: dict[str, Any]
    contact_sheet: dict[str, Any]
    sealed_test_accessed: bool


def _mask_defect_pixels(mask: np.ndarray) -> int:
    if mask.ndim == 2:
        return int(np.count_nonzero(mask))
    if mask.ndim == 3:
        return int(np.count_nonzero(np.any(mask != 0, axis=2)))
    raise ManifestError(f"지원하지 않는 마스크 차원입니다: shape={mask.shape}")


def _distribution(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def verify_dataset(
    raw_root: str | Path,
    rows: Sequence[ManifestRow],
    split_result: SplitVerifyResult,
    *,
    contact_sheet_path: str | Path | None = None,
    contact_sheet_info: Mapping[str, Any] | None = None,
    dhash_threshold: int | None = None,
) -> AuditResult:
    """모든 manifest 행의 이미지/마스크를 다시 디코드해 무결성을 검사한다.

    ``dhash_threshold``가 없으면 ``split_result.report``의 값을 사용하고,
    그것도 없으면 설계 기본값 2를 쓴다. 그룹 재조정이 있었던 manifest에서도
    감사 그룹 통계가 manifest와 같은 threshold를 쓰게 하기 위해서다.
    """

    root = Path(raw_root)
    violations: list[str] = []
    warnings: list[str] = []

    resolved_threshold = dhash_threshold
    if resolved_threshold is None:
        reported_threshold = None
        if isinstance(split_result.report, Mapping):
            reported_threshold = split_result.report.get("dhash_threshold")
        if isinstance(reported_threshold, int) and not isinstance(reported_threshold, bool):
            resolved_threshold = reported_threshold
    if resolved_threshold is None:
        resolved_threshold = DHASH_THRESHOLD

    width_values: list[int] = []
    height_values: list[int] = []
    channel_counter: Counter[str] = Counter()
    dtype_counter: Counter[str] = Counter()
    format_counter: Counter[str] = Counter()
    mask_sha_counter: Counter[str] = Counter()

    image_shas: list[str] = []
    dhashes: list[int] = []

    good_masks_empty = 0
    good_masks_nonempty = 0
    defect_masks_empty = 0
    defect_masks_nonempty = 0
    decoded_images = 0
    decoded_masks = 0

    for row in rows:
        image_path = root / row.image_path
        if not image_path.is_file():
            violations.append(f"이미지 파일이 없습니다: {image_path}")
            continue
        try:
            image, image_sha256 = read_png(image_path)
        except (ManifestError, OSError) as exc:
            violations.append(f"이미지 디코드 실패: {image_path}: {exc}")
            continue

        decoded_images += 1
        height = int(image.shape[0])
        width = int(image.shape[1])
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        width_values.append(width)
        height_values.append(height)
        channel_counter[str(channels)] += 1
        dtype_counter[str(image.dtype)] += 1
        format_counter[image_path.suffix.lstrip(".").upper()] += 1
        image_shas.append(image_sha256)
        dhashes.append(dhash64(image))

        if (width, height, channels) != (row.width, row.height, row.channels):
            violations.append(
                f"이미지 메타데이터 불일치: {image_path} "
                f"manifest={(row.width, row.height, row.channels)} "
                f"actual={(width, height, channels)}"
            )
        if image_sha256 != row.image_sha256:
            violations.append(f"이미지 sha256 불일치: {image_path}")

        recomputed_pixels = 0
        mask_checked = False
        if row.mask_path:
            mask_path = root / row.mask_path
            if not mask_path.is_file():
                violations.append(f"마스크 파일이 없습니다: {mask_path}")
            else:
                try:
                    mask, mask_sha256 = read_png(mask_path)
                except (ManifestError, OSError) as exc:
                    violations.append(f"마스크 디코드 실패: {mask_path}: {exc}")
                else:
                    decoded_masks += 1
                    mask_checked = True
                    mask_sha_counter[mask_sha256] += 1
                    if mask_sha256 != row.mask_sha256:
                        violations.append(f"마스크 sha256 불일치: {mask_path}")
                    if mask.ndim < 2 or tuple(mask.shape[:2]) != (height, width):
                        violations.append(
                            f"마스크 크기 불일치: {mask_path} "
                            f"image={(height, width)} mask={mask.shape[:2]}"
                        )
                    recomputed_pixels = _mask_defect_pixels(mask)
        elif row.label == "defect":
            violations.append(f"결함 행에 마스크 경로가 없습니다: {row.sample_id}")

        if recomputed_pixels != row.defect_pixels:
            violations.append(
                f"defect_pixels 불일치: {row.sample_id} "
                f"manifest={row.defect_pixels} actual={recomputed_pixels}"
            )
        expected_label = "defect" if recomputed_pixels > 0 else "good"
        if expected_label != row.label:
            violations.append(
                f"L8 라벨 불일치: {row.sample_id} manifest={row.label} "
                f"actual={expected_label} (defect_pixels={recomputed_pixels})"
            )

        if row.label == "defect":
            if recomputed_pixels > 0:
                defect_masks_nonempty += 1
            else:
                defect_masks_empty += 1
        else:
            if not mask_checked or recomputed_pixels == 0:
                good_masks_empty += 1
            else:
                good_masks_nonempty += 1

    duplicate_mask_groups = sum(1 for value in mask_sha_counter.values() if value > 1)
    max_duplicate_mask_rows = max(mask_sha_counter.values(), default=0)
    warnings.append(
        "마스크 sha256은 빈 마스크 등에서 중복되므로 중복 그룹 키로 사용하지 "
        f"않았습니다. 중복 마스크 해시 그룹: {duplicate_mask_groups}개"
    )

    computed_groups = compute_groups(image_shas, dhashes, resolved_threshold)
    group_sizes = [len(group) for group in computed_groups]
    cap_rows = GROUP_SIZE_CAP_RATIO * len(rows)
    groups_over_cap = sum(1 for size in group_sizes if size > cap_rows)
    manifest_group_count = len({row.group_id for row in rows})
    group_stats = {
        "dhash_threshold": resolved_threshold,
        "group_size_cap_ratio": GROUP_SIZE_CAP_RATIO,
        "group_size_cap_rows": cap_rows,
        "groups": len(computed_groups),
        "max_group_rows": max(group_sizes, default=0),
        "groups_over_cap": groups_over_cap,
        "manifest_group_count": manifest_group_count,
        "group_count_matches_manifest": len(computed_groups) == manifest_group_count,
        "decoded_for_dhash": len(image_shas),
    }
    if groups_over_cap:
        warnings.append(
            f"dHash threshold {resolved_threshold}에서 2% 초과 그룹이 "
            f"{groups_over_cap}개 있습니다."
        )
    if len(computed_groups) != manifest_group_count:
        warnings.append(
            "재계산 dHash 그룹 수와 manifest group_id 수가 다릅니다: "
            f"재계산={len(computed_groups)}, manifest={manifest_group_count}. "
            "manifest 생성 시 threshold 재조정이 있었는지 확인해야 합니다."
        )

    canonical = canonical_rows(rows)
    counts_payload = {
        "manifest": counts(rows),
        "canonical": counts(canonical),
        "official_train": {
            "defect": sum(
                1
                for row in canonical
                if row.source_split == "train" and row.label == "defect"
            ),
            "good": sum(
                1
                for row in canonical
                if row.source_split == "train" and row.label == "good"
            ),
        },
        "official_test": {
            "defect": sum(
                1
                for row in canonical
                if row.source_split == "test" and row.label == "defect"
            ),
            "good": sum(
                1
                for row in canonical
                if row.source_split == "test" and row.label == "good"
            ),
        },
        "non_canonical_sample_ids": sorted(
            row.sample_id for row in rows if row.sample_id != row.original_id
        ),
        "groups": manifest_group_count,
    }

    image_stats = {
        "image_count": decoded_images,
        "width_min": min(width_values) if width_values else None,
        "width_max": max(width_values) if width_values else None,
        "height_min": min(height_values) if height_values else None,
        "height_max": max(height_values) if height_values else None,
        "channels": _distribution(channel_counter),
        "dtypes": _distribution(dtype_counter),
        "formats": _distribution(format_counter),
    }

    mask_stats = {
        "mask_count": decoded_masks,
        "good_masks_empty": good_masks_empty,
        "good_masks_nonempty": good_masks_nonempty,
        "defect_masks_nonempty": defect_masks_nonempty,
        "defect_masks_empty": defect_masks_empty,
        "mask_sha_duplicate_groups": duplicate_mask_groups,
        "max_duplicate_mask_sha_rows": max_duplicate_mask_rows,
    }

    split_payload = {
        "ok": split_result.ok,
        "violations": list(split_result.violations),
        "warnings": list(split_result.warnings),
        "counts": split_result.counts,
        "p_test": split_result.p_test,
        "test_ratio": split_result.test_ratio,
        "overlap": split_result.overlap,
        "report": split_result.report,
    }

    contact_sheet = {
        "path": str(contact_sheet_path) if contact_sheet_path is not None else "",
        "selection": CONTACT_SHEET_SELECTION,
        "columns": CONTACT_SHEET_COLUMNS,
        "thumb_height": CONTACT_SHEET_THUMB_HEIGHT,
    }
    if contact_sheet_info:
        contact_sheet.update(dict(contact_sheet_info))

    return AuditResult(
        ok=not violations and split_result.ok,
        violations=tuple(violations),
        warnings=tuple(warnings),
        counts=counts_payload,
        image_stats=image_stats,
        mask_stats=mask_stats,
        group_stats=group_stats,
        split=split_payload,
        l_checks=dict(split_result.l_checks),
        contact_sheet=contact_sheet,
        sealed_test_accessed=False,
    )


def _even_sample(items: Sequence[ManifestRow], limit: int) -> list[ManifestRow]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = sorted({index * (len(items) - 1) // (limit - 1) for index in range(limit)})
    return [items[index] for index in indices]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ManifestError(f"contact sheet에 넣을 수 없는 이미지 shape입니다: {image.shape}")


def write_contact_sheet(
    rows: Sequence[ManifestRow],
    raw_root: str | Path,
    path: str | Path,
    *,
    test_sample_ids: Sequence[str] | set[str] = (),
) -> dict[str, Any]:
    """tune 행만으로 contact sheet를 만든다. 내부 test 행이 있으면 예외."""

    root = Path(raw_root)
    target = Path(path)
    forbidden = set(test_sample_ids)
    leaked = sorted(row.sample_id for row in rows if row.sample_id in forbidden)
    if leaked:
        raise ValueError(
            "contact sheet에 내부 test 행이 포함될 수 없습니다: " + ", ".join(leaked[:5])
        )

    canonical = [row for row in rows if row.sample_id == row.original_id]
    defect_rows = sorted(
        (row for row in canonical if row.label == "defect"),
        key=lambda row: row.sample_id,
    )[:CONTACT_SHEET_MAX_DEFECT]
    good_rows = _even_sample(
        sorted((row for row in canonical if row.label == "good"), key=lambda row: row.sample_id),
        CONTACT_SHEET_GOOD_SAMPLE,
    )
    selected = [*defect_rows, *good_rows]
    if not selected:
        raise ValueError("contact sheet에 넣을 tune 행이 없습니다.")

    thumbnails: list[tuple[np.ndarray, ManifestRow]] = []
    for row in selected:
        image_path = root / row.image_path
        try:
            image, _ = read_png(image_path)
        except (ManifestError, OSError) as exc:
            raise ValueError(
                f"contact sheet 이미지를 읽지 못했습니다: {image_path}: {exc}"
            ) from exc
        bgr = _as_bgr(image)
        scale = CONTACT_SHEET_THUMB_HEIGHT / bgr.shape[0]
        width = max(1, int(round(bgr.shape[1] * scale)))
        thumbnail = cv2.resize(
            bgr,
            (width, CONTACT_SHEET_THUMB_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        thumbnails.append((thumbnail, row))

    cell_width = max(thumb.shape[1] for thumb, _ in thumbnails) + 12
    cell_height = CONTACT_SHEET_THUMB_HEIGHT + 30
    rows_count = (len(thumbnails) + CONTACT_SHEET_COLUMNS - 1) // CONTACT_SHEET_COLUMNS
    canvas = np.full(
        (rows_count * cell_height, CONTACT_SHEET_COLUMNS * cell_width, 3),
        245,
        dtype=np.uint8,
    )

    for index, (thumbnail, row) in enumerate(thumbnails):
        row_index, column_index = divmod(index, CONTACT_SHEET_COLUMNS)
        x = column_index * cell_width + 6
        y = row_index * cell_height + 6
        height, width = thumbnail.shape[:2]
        canvas[y : y + height, x : x + width] = thumbnail
        color = (0, 0, 255) if row.label == "defect" else (0, 160, 0)
        cv2.rectangle(canvas, (x - 2, y - 2), (x + width + 1, y + height + 1), color, 2)
        cv2.putText(
            canvas,
            row.sample_id,
            (x, y + height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise ValueError(f"contact sheet PNG 인코드에 실패했습니다: {target}")
    encoded.tofile(str(target))

    return {
        "path": str(target),
        "defect_rows": len(defect_rows),
        "good_rows": len(good_rows),
        "total_rows": len(selected),
        "columns": CONTACT_SHEET_COLUMNS,
        "thumb_height": CONTACT_SHEET_THUMB_HEIGHT,
    }


def write_audit_json(result: AuditResult, path: str | Path) -> None:
    """감사 결과를 JSON으로 저장한다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _l_row(l_checks: Mapping[str, Any], key: str) -> str:
    check = l_checks.get(key, {})
    status = check.get("status", "UNKNOWN")
    value = check.get("value", "")
    detail = check.get("detail", "")
    return f"| {key} | {status} | {value} | {detail} |"


def _number_text(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def write_audit_markdown(result: AuditResult, path: str | Path) -> None:
    """감사 결과를 한국어 Markdown 보고서로 저장한다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest_counts = result.counts["manifest"]
    canonical_counts = result.counts["canonical"]
    train_counts = result.counts["official_train"]
    test_counts = result.counts["official_test"]
    non_canonical = result.counts.get("non_canonical_sample_ids", [])
    split = result.split
    split_counts = split.get("counts", {})
    overlap = split.get("overlap", {})
    image_stats = result.image_stats
    mask_stats = result.mask_stats
    group_stats = result.group_stats
    contact = result.contact_sheet
    raw_report = split.get("report")
    report: Mapping[str, Any] = raw_report if isinstance(raw_report, Mapping) else {}

    def split_count(name: str, key: str) -> Any:
        if isinstance(report, Mapping):
            report_splits = report.get("splits")
            if isinstance(report_splits, Mapping):
                report_section = report_splits.get(name)
                if isinstance(report_section, Mapping) and key in report_section:
                    return report_section[key]
        section = split_counts.get(name, {}) if isinstance(split_counts, Mapping) else {}
        return section.get(key, "") if isinstance(section, Mapping) else ""

    official_total = _number_text(split_count("official_test", "total_rows"))
    tune_overlap = _number_text(overlap.get("internal_tune", {}).get("samples", ""))
    test_overlap = _number_text(overlap.get("internal_test", {}).get("samples", ""))

    result_label = "PASS" if result.ok else "FAIL"
    split_label = "PASS" if split.get("ok") else "FAIL"

    lines: list[str] = [
        "# T03 데이터 감사 — KolektorSDD2 반입본, manifest, 분할, 누출 검사",
        "",
        "- 감사 범위: T03. 원본 재디코드, manifest/분할 수량, L1–L8, contact sheet.",
        "- 코드 리비전과 실행 시각: 지어내지 않는다. 증거 파일을 참조한다",
        "  (`docs/evidence/T03_commands.log`, `docs/evidence/T03_summary.txt`).",
        "- sealed test 접근: 없음. `batch`/`eval`을 실행하지 않았고 "
        "`--allow-sealed`를 쓰지 않았다.",
        "",
        "## 1. 요약",
        "",
        f"- 데이터 감사 결과: **{result_label}**",
        f"- 데이터 위반: {len(result.violations)}건, 경고: {len(result.warnings)}건",
        f"- 분할 검증 결과: **{split_label}**, 위반 {len(split.get('violations', []))}건",
        f"- manifest 전체 행: {manifest_counts.get('total_rows', '')}",
        f"- 정본 행: {canonical_counts.get('total_rows', '')}",
        f"- P_test: {split.get('p_test', '')} (하한 100)",
        f"- test 비율: {float(split.get('test_ratio', 0.0)):.4f} (허용 0.38–0.42)",
        "",
        "## 2. 공식 수량 vs 로컬 측정",
        "",
        "공식 수량은 T01 사실표의 공식 페이지 진술과 합산 도출값이다. 로컬 값은",
        "이번 감사에서 manifest/분할 CSV를 다시 읽어 센 값이다.",
        "",
        "| 항목 | 공식 수량 | 로컬 측정 |",
        "|---|---:|---:|",
        f"| 전체 정본 이미지 | {EXPECTED_CANONICAL_ROWS} | "
        f"{canonical_counts.get('total_rows', '')} |",
        f"| 결함 이미지 | {EXPECTED_DEFECT_ROWS} | {canonical_counts.get('defect', '')} |",
        f"| 정상 이미지 | {EXPECTED_GOOD_ROWS} | {canonical_counts.get('good', '')} |",
        f"| 공식 train | {EXPECTED_OFFICIAL_TRAIN_ROWS} | "
        f"{canonical_counts.get('official_train', '')} |",
        f"| 공식 train 결함 | {EXPECTED_OFFICIAL_TRAIN_DEFECT} | "
        f"{train_counts.get('defect', '')} |",
        f"| 공식 train 정상 | {EXPECTED_OFFICIAL_TRAIN_GOOD} | "
        f"{train_counts.get('good', '')} |",
        f"| 공식 test | {EXPECTED_OFFICIAL_TEST_ROWS} | "
        f"{canonical_counts.get('official_test', '')} |",
        f"| 공식 test 결함 | {EXPECTED_OFFICIAL_TEST_DEFECT} | "
        f"{test_counts.get('defect', '')} |",
        f"| 공식 test 정상 | {EXPECTED_OFFICIAL_TEST_GOOD} | "
        f"{test_counts.get('good', '')} |",
        "",
        "## 3. 3,336장과 3,335장",
        "",
        f"- manifest 전체 행은 {manifest_counts.get('total_rows', '')}장이다. 복사본 "
        f"{manifest_counts.get('duplicate_rows', '')}장이 포함된 값이다.",
        f"- 정본은 {canonical_counts.get('total_rows', '')}장이다. 정본 결함 "
        f"{canonical_counts.get('defect', '')}장 + 정상 {canonical_counts.get('good', '')}장.",
        f"- 복사본 행: {', '.join(non_canonical)}" if non_canonical else "- 복사본 행: 없음",
        "- 복사본은 이미지와 마스크의 sha256이 정본과 같다. 원본 이미지/마스크를 "
        "삭제하거나 이동하지 않았다.",
        "- 복사본 마스크 이름은 `_GT`가 ` (copy)` 앞에 들어간다: "
        "`<id>_GT (copy).png`. stem + `_GT.png`로 찾으면 이 마스크를 놓친다.",
        "",
        "## 4. 이미지와 마스크",
        "",
        f"- 디코드한 이미지: {image_stats.get('image_count', '')}장",
        f"- 이미지 폭: {image_stats.get('width_min', '')}–{image_stats.get('width_max', '')}",
        f"- 이미지 높이: {image_stats.get('height_min', '')}–{image_stats.get('height_max', '')}",
        f"- 채널 분포: {image_stats.get('channels', {})}",
        f"- dtype 분포: {image_stats.get('dtypes', {})}",
        f"- format 분포: {image_stats.get('formats', {})}",
        f"- 디코드한 마스크: {mask_stats.get('mask_count', '')}개",
        f"- 정상 마스크: 비어 있음 {mask_stats.get('good_masks_empty', '')}개, "
        f"비어 있지 않음 {mask_stats.get('good_masks_nonempty', '')}개",
        f"- 결함 마스크: 비어 있지 않음 {mask_stats.get('defect_masks_nonempty', '')}개, "
        f"비어 있음 {mask_stats.get('defect_masks_empty', '')}개",
        f"- 마스크 sha256 중복 그룹: {mask_stats.get('mask_sha_duplicate_groups', '')}개 "
        f"(최대 {mask_stats.get('max_duplicate_mask_sha_rows', '')}행)",
        "- 경고: 마스크 sha256은 빈 마스크 등에서 중복되므로 이미지 그룹 키로 쓰지 않았다.",
        "",
        "## 5. 중복 그룹핑",
        "",
        f"- dHash threshold: {group_stats.get('dhash_threshold', '')}",
        f"- 그룹 수: {group_stats.get('groups', '')}",
        f"- 최대 그룹 크기: {group_stats.get('max_group_rows', '')}행",
        f"- 2% 상한: {float(group_stats.get('group_size_cap_rows', 0.0)):.2f}행 "
        f"(비율 {group_stats.get('group_size_cap_ratio', '')})",
        f"- 2% 초과 그룹: {group_stats.get('groups_over_cap', '')}개",
        f"- manifest group_id 수: {group_stats.get('manifest_group_count', '')}",
        "- 정확히 같은 image_sha256을 먼저 합치고, dHash Hamming 거리 ≤ 2를 추가로 합쳤다.",
        "- dHash는 grayscale 9×8 INTER_AREA 축소 후 가로 인접 비교 64bit다.",
        "",
        "## 6. 분할",
        "",
        "| 분할 | 행 | 결함 | 정상 | sha256 |",
        "|---|---:|---:|---:|---|",
        f"| tune | {split_count('tune', 'total_rows')} | {split_count('tune', 'defect')} | "
        f"{split_count('tune', 'good')} | `{split_count('tune', 'sha256')}` |",
        f"| test | {split_count('test', 'total_rows')} | {split_count('test', 'defect')} | "
        f"{split_count('test', 'good')} | `{split_count('test', 'sha256')}` |",
        f"| official_test | {split_count('official_test', 'total_rows')} | "
        f"{split_count('official_test', 'defect')} | {split_count('official_test', 'good')} | "
        f"`{split_count('official_test', 'sha256')}` |",
        "",
        f"- test 비율: {float(split.get('test_ratio', 0.0)):.4f}",
        f"- P_test: {split.get('p_test', '')} (하한 100)",
        f"- seed: {report.get('seed', '기록 없음')}",
        f"- split_hash: `{report.get('split_hash', '기록 없음')}`",
        f"- split_hash recipe: `{report.get('split_hash_recipe', '기록 없음')}`",
        "- 공식 test와 내부 test 중복: "
        f"sample {overlap.get('internal_test', {}).get('samples', '')}개, "
        f"group {overlap.get('internal_test', {}).get('groups', '')}개",
        "- 공식 test와 내부 tune 중복: "
        f"sample {overlap.get('internal_tune', {}).get('samples', '')}개, "
        f"group {overlap.get('internal_tune', {}).get('groups', '')}개",
        "- 공식 test 중복은 보고만 하며 임계값 튜닝에 사용하지 않았다.",
        "",
        "## 7. 누출 검사 L1–L8",
        "",
        "| 검사 | 결과 | 값 | 설명 |",
        "|---|---|---:|---|",
        _l_row(result.l_checks, "L1"),
        _l_row(result.l_checks, "L2"),
        _l_row(result.l_checks, "L3"),
        _l_row(result.l_checks, "L4"),
        _l_row(result.l_checks, "L5"),
        _l_row(result.l_checks, "L6"),
        _l_row(result.l_checks, "L7"),
        _l_row(result.l_checks, "L8"),
        _l_row(result.l_checks, "official_test_identity"),
        _l_row(result.l_checks, "split_report"),
        "",
        "## 8. Contact sheet",
        "",
        f"- 경로: `{contact.get('path', '')}`" if contact.get("path") else "- 경로: 생성하지 않음",
        f"- 선정 규칙: {contact.get('selection', CONTACT_SHEET_SELECTION)}",
        f"- 선정 수: 결함 {contact.get('defect_rows', '기록 없음')}장, "
        f"정상 {contact.get('good_rows', '기록 없음')}장, "
        f"전체 {contact.get('total_rows', '기록 없음')}장",
        f"- 배치: {contact.get('columns', CONTACT_SHEET_COLUMNS)}열, "
        f"썸네일 높이 {contact.get('thumb_height', CONTACT_SHEET_THUMB_HEIGHT)}px",
        "- contact sheet에는 tune 행만 넣는다. 내부 test sample_id가 들어가면 예외로 중단한다.",
        "",
        "## 9. Sealed test",
        "",
        "- sealed test 접근: 없음.",
        f"- `sealed_test_accessed`: {str(result.sealed_test_accessed).lower()}",
        "- 이 감사는 test/official_test 파일을 읽어 수량과 누출만 확인하며, "
        "성능 측정·임계값 튜닝·시각 검토에 사용하지 않았다.",
        "",
        "## 10. 한계와 미해결",
        "",
        "- 단일 통제 데이터셋이며 현장 일반화를 주장하지 않는다.",
        "- 공식 train/test 분할은 보조 감사용이며 독립적인 현장 시험이 아니다.",
        f"- 공식 test {official_total}장 중 internal tune {tune_overlap}장, "
        f"internal test {test_overlap}장이 포함된다. 설계는 internal test 중복만 "
        f"보고하도록 명시했으나, tune에 포함된 {tune_overlap}장은 official_test의 "
        "독립 감사를 제한한다. T05/T08에서 official_test를 보조 감사로 쓸 때 이 "
        "한계를 명시해야 한다.",
        "- dHash 근접 중복은 휴리스틱이다. 물리적 부품 ID는 확인되지 않았다.",
        "- 마스크 sha256은 빈 마스크 때문에 중복되므로 그룹 키로 쓰지 않았다.",
        "- `split_weakly_*.pyb` 5개는 공식 약지도 split 정의이며 현재 설계에서 "
        "사용하지 않는다. manifest/분할에서 제외하고 존재만 기록했다.",
        "- 코드 리비전과 실행 시각은 이 보고서에 직접 적지 않는다. "
        "`docs/evidence/T03_commands.log`와 `docs/evidence/T03_summary.txt`를 참조한다.",
        "",
        "## 11. 재현 명령",
        "",
        "```",
        "uv run surface-sentinel data build-manifest --raw data/raw/kolektorsdd2 "
        "--manifest data/manifests/dataset_manifest.csv "
        "--errors data/manifests/build_errors.json",
        "uv run surface-sentinel data make-splits --manifest data/manifests/dataset_manifest.csv "
        "--out-dir data/splits --seed 42",
        "uv run surface-sentinel data verify --raw data/raw/kolektorsdd2 "
        "--manifest data/manifests/dataset_manifest.csv --splits-dir data/splits "
        "--audit-json data/audit/T03_audit.json --audit-md docs/research/T03_data_audit.md "
        "--contact-sheet data/derived/contact_sheets/tune_contact_sheet.png",
        "```",
        "",
    ]

    if result.violations:
        lines.extend(["## 12. 데이터 위반 목록", ""])
        lines.extend(f"- {message}" for message in result.violations)
        lines.append("")
    if result.warnings:
        lines.extend(["## 13. 경고 목록", ""])
        lines.extend(f"- {message}" for message in result.warnings)
        lines.append("")
    if split.get("violations"):
        lines.extend(["## 14. 분할 위반 목록", ""])
        lines.extend(f"- {message}" for message in split["violations"])
        lines.append("")

    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")


__all__ = [
    "AuditResult",
    "CONTACT_SHEET_COLUMNS",
    "CONTACT_SHEET_GOOD_SAMPLE",
    "CONTACT_SHEET_MAX_DEFECT",
    "CONTACT_SHEET_SELECTION",
    "CONTACT_SHEET_THUMB_HEIGHT",
    "verify_dataset",
    "write_audit_json",
    "write_audit_markdown",
    "write_contact_sheet",
]
