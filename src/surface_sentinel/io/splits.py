"""중복 그룹 단위 tune/test 분할, 동결 해시, 누출 검사(L1-L8).

분할은 이미지 단위가 아니라 ``group_id`` 단위로 수행한다. 정본
(``sample_id == original_id``) 행만 CSV에 기록하고, 같은 그룹이 양쪽에
들어가지 않도록 검증한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import (
    DHASH_THRESHOLD,
    GROUP_SIZE_CAP_RATIO,
    ManifestError,
    ManifestRow,
    canonical_rows,
    counts,
    load_manifest,
    write_manifest_csv,
)

DESIGN_VERSION = "SS-DESIGN-1.1"
DEFAULT_SEED = 42
TEST_RATIO_TARGET = 0.4
P_TEST_MIN = 100
TEST_RATIO_MIN = 0.38
TEST_RATIO_MAX = 0.42

EXPECTED_CANONICAL_ROWS = 3335
EXPECTED_DEFECT_ROWS = 356
EXPECTED_GOOD_ROWS = 2979
EXPECTED_OFFICIAL_TRAIN_ROWS = 2331
EXPECTED_OFFICIAL_TRAIN_DEFECT = 246
EXPECTED_OFFICIAL_TRAIN_GOOD = 2085
EXPECTED_OFFICIAL_TEST_ROWS = 1004
EXPECTED_OFFICIAL_TEST_DEFECT = 110
EXPECTED_OFFICIAL_TEST_GOOD = 894

EXPECTED_DATASET_COUNTS: dict[str, int] = {
    "canonical_rows": EXPECTED_CANONICAL_ROWS,
    "defect": EXPECTED_DEFECT_ROWS,
    "good": EXPECTED_GOOD_ROWS,
    "official_train": EXPECTED_OFFICIAL_TRAIN_ROWS,
    "official_train_defect": EXPECTED_OFFICIAL_TRAIN_DEFECT,
    "official_train_good": EXPECTED_OFFICIAL_TRAIN_GOOD,
    "official_test": EXPECTED_OFFICIAL_TEST_ROWS,
    "official_test_defect": EXPECTED_OFFICIAL_TEST_DEFECT,
    "official_test_good": EXPECTED_OFFICIAL_TEST_GOOD,
}

SPLIT_TUNE = "tune"
SPLIT_TEST = "test"


class SplitError(ValueError):
    """분할을 만들 수 없거나 분할 입력이 계약을 위반할 때 발생한다."""


@dataclass(frozen=True)
class Group:
    """중복 그룹. ``members``는 정본 행만 담는다."""

    group_id: str
    label: str
    members: tuple[ManifestRow, ...]
    size: int
    canonical_size: int


@dataclass(frozen=True)
class GroupAssignment:
    """그룹 배정 결과."""

    assignment: dict[str, str]
    adjustments: tuple[dict[str, Any], ...]
    targets: dict[str, dict[str, int]]


@dataclass(frozen=True)
class SplitVerifyResult:
    """분할 검증 결과."""

    ok: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, Any]
    p_test: int
    test_ratio: float
    overlap: dict[str, Any]
    l_checks: dict[str, Any]
    report: dict[str, Any] | None = None


def build_groups(
    rows: Sequence[ManifestRow],
    warnings: list[str] | None = None,
) -> list[Group]:
    """정본 행을 ``group_id``로 묶고 그룹 라벨을 정한다.

    라벨이 섞인 그룹은 ``warnings``에 기록하고 defect로 취급한다.
    """

    warning_list = warnings if warnings is not None else []
    all_by_group: dict[str, list[ManifestRow]] = defaultdict(list)
    canonical_by_group: dict[str, list[ManifestRow]] = defaultdict(list)

    for row in rows:
        all_by_group[row.group_id].append(row)
        if row.sample_id == row.original_id:
            canonical_by_group[row.group_id].append(row)

    groups: list[Group] = []
    for group_id in sorted(all_by_group):
        members = sorted(canonical_by_group.get(group_id, []), key=lambda row: row.sample_id)
        if not members:
            warning_list.append(
                f"그룹 {group_id}에 정본 행이 없어 분할에서 제외합니다."
            )
            continue
        labels = {row.label for row in members}
        if len(labels) > 1:
            warning_list.append(
                f"그룹 {group_id}에 defect와 good 라벨이 섞여 있습니다: {sorted(labels)}. "
                "그룹 라벨은 defect로 둡니다."
            )
        label = "defect" if "defect" in labels else "good"
        groups.append(
            Group(
                group_id=group_id,
                label=label,
                members=tuple(members),
                size=len(all_by_group[group_id]),
                canonical_size=len(members),
            )
        )
    return groups


def _half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def assign_groups(
    groups: Sequence[Group],
    seed: int = DEFAULT_SEED,
    test_ratio: float = TEST_RATIO_TARGET,
    p_test_min: int = P_TEST_MIN,
) -> GroupAssignment:
    """시드로 섞은 그룹을 라벨별 크기 내림차순으로 탐욕 배정한다."""

    if not (0.0 < test_ratio < 1.0):
        raise SplitError(f"test_ratio는 (0, 1) 범위여야 합니다: {test_ratio}")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)

    by_label: dict[str, list[Group]] = {"defect": [], "good": []}
    for group in shuffled:
        if group.label not in by_label:
            raise SplitError(f"알 수 없는 그룹 라벨입니다: {group.label}")
        by_label[group.label].append(group)
    for label in by_label:
        by_label[label].sort(key=lambda group: -group.canonical_size)

    totals = {
        label: sum(group.canonical_size for group in members)
        for label, members in by_label.items()
    }
    target_test = {label: _half_up(test_ratio * totals[label]) for label in by_label}
    target_tune = {label: totals[label] - target_test[label] for label in by_label}

    assignment: dict[str, str] = {}
    test_count = {"defect": 0, "good": 0}
    tune_count = {"defect": 0, "good": 0}

    for label in ("defect", "good"):
        for group in by_label[label]:
            size = group.canonical_size
            deviation_if_test = abs(test_count[label] + size - target_test[label])
            deviation_if_tune = abs(test_count[label] - target_test[label])
            if deviation_if_test < deviation_if_tune:
                side = SPLIT_TEST
            elif deviation_if_tune < deviation_if_test:
                side = SPLIT_TUNE
            else:
                side = SPLIT_TEST if test_count[label] <= tune_count[label] else SPLIT_TUNE
            assignment[group.group_id] = side
            if side == SPLIT_TEST:
                test_count[label] += size
            else:
                tune_count[label] += size

    adjustments: list[dict[str, Any]] = []
    defect_test = test_count["defect"]
    if defect_test < p_test_min:
        candidates = sorted(
            (
                group
                for group in by_label["defect"]
                if assignment[group.group_id] == SPLIT_TUNE
            ),
            key=lambda group: (group.canonical_size, group.group_id),
        )
        for group in candidates:
            if defect_test >= p_test_min:
                break
            before = defect_test
            assignment[group.group_id] = SPLIT_TEST
            defect_test += group.canonical_size
            adjustments.append(
                {
                    "group_id": group.group_id,
                    "label": "defect",
                    "size": group.canonical_size,
                    "from": SPLIT_TUNE,
                    "to": SPLIT_TEST,
                    "reason": "P_test_guard",
                    "defect_test_before": before,
                    "defect_test_after": defect_test,
                }
            )
        if defect_test < p_test_min:
            raise SplitError(
                f"P_test >= {p_test_min}을 만족할 수 없습니다: "
                f"전체 결함 정본 {totals['defect']}장, 현재 test 결함 {defect_test}장"
            )
        tune_count["defect"] -= defect_test - test_count["defect"]
        test_count["defect"] = defect_test

    targets = {
        label: {
            "total": totals[label],
            "test_target": target_test[label],
            "tune_target": target_tune[label],
        }
        for label in ("defect", "good")
    }
    return GroupAssignment(assignment=assignment, adjustments=tuple(adjustments), targets=targets)


def _sort_key(row: ManifestRow) -> tuple[str, str]:
    return (row.source_split, row.sample_id)


def _split_summary(rows: Sequence[ManifestRow], sha256: str) -> dict[str, Any]:
    summary = counts(rows)
    return {
        "rows": summary["total_rows"],
        "defect": summary["defect"],
        "good": summary["good"],
        "sha256": sha256,
    }


def write_splits(
    rows: Sequence[ManifestRow],
    assignment: Mapping[str, str],
    out_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    test_ratio: float = TEST_RATIO_TARGET,
    p_test_min: int = P_TEST_MIN,
    dhash_threshold: int = DHASH_THRESHOLD,
    group_rebuild_applied: bool = False,
    group_size_cap_ratio: float = GROUP_SIZE_CAP_RATIO,
    adjustments: Sequence[Mapping[str, Any]] = (),
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """tune/test/official_test CSV, test.sha256, split_report.json을 쓴다."""

    canonical = canonical_rows(rows)
    unknown_groups = sorted(
        {row.group_id for row in canonical if row.group_id not in assignment}
    )
    if unknown_groups:
        raise SplitError(f"배정이 없는 그룹이 있습니다: {unknown_groups[:10]}")
    invalid_sides = sorted(
        {side for side in assignment.values() if side not in (SPLIT_TUNE, SPLIT_TEST)}
    )
    if invalid_sides:
        raise SplitError(f"알 수 없는 분할 값이 있습니다: {invalid_sides}")

    tune_rows = sorted(
        [row for row in canonical if assignment[row.group_id] == SPLIT_TUNE],
        key=_sort_key,
    )
    test_rows = sorted(
        [row for row in canonical if assignment[row.group_id] == SPLIT_TEST],
        key=_sort_key,
    )
    official_test_rows = sorted(
        [row for row in canonical if row.source_split == "test"],
        key=_sort_key,
    )

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    tune_sha256 = write_manifest_csv(tune_rows, target / "tune.csv")
    test_sha256 = write_manifest_csv(test_rows, target / "test.csv")
    official_sha256 = write_manifest_csv(official_test_rows, target / "official_test.csv")
    (target / "test.sha256").write_text(
        f"{test_sha256}  test.csv\n", encoding="utf-8", newline="\n"
    )

    official_ids = {row.sample_id for row in official_test_rows}
    official_groups = {row.group_id for row in official_test_rows}

    def overlap(split_rows: Sequence[ManifestRow]) -> dict[str, int]:
        return {
            "samples": len(official_ids & {row.sample_id for row in split_rows}),
            "groups": len(official_groups & {row.group_id for row in split_rows}),
        }

    overlap_payload = {
        "internal_test": overlap(test_rows),
        "internal_tune": overlap(tune_rows),
    }

    hash_payload = {
        "seed": seed,
        "test_ratio_target": test_ratio,
        "p_test_min": p_test_min,
        "dhash_threshold": dhash_threshold,
        "tune_sha256": tune_sha256,
        "test_sha256": test_sha256,
        "official_test_sha256": official_sha256,
    }
    canonical_json = json.dumps(
        hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    split_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    split_hash_recipe = (
        "sha256(canonical_json({seed,test_ratio_target,p_test_min,dhash_threshold,"
        "tune_sha256,test_sha256,official_test_sha256}))"
    )

    manifest_summary = counts(rows)
    report = {
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "test_ratio_target": test_ratio,
        "p_test_min": p_test_min,
        "dhash_threshold": dhash_threshold,
        "group_size_cap_ratio": group_size_cap_ratio,
        "group_rebuild_applied": group_rebuild_applied,
        "manifest": {
            **manifest_summary,
            "canonical": counts(canonical),
        },
        "splits": {
            "tune": _split_summary(tune_rows, tune_sha256),
            "test": _split_summary(test_rows, test_sha256),
            "official_test": _split_summary(official_test_rows, official_sha256),
        },
        "official_test_overlap": overlap_payload,
        "split_hash": split_hash,
        "split_hash_recipe": split_hash_recipe,
        "adjustments": [dict(item) for item in adjustments],
        "warnings": list(warnings),
    }

    with (target / "split_report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return report


def _official_count_payload(rows: Sequence[ManifestRow]) -> dict[str, int]:
    train = [row for row in rows if row.source_split == "train"]
    test = [row for row in rows if row.source_split == "test"]
    return {
        "canonical_rows": len(rows),
        "defect": sum(1 for row in rows if row.label == "defect"),
        "good": sum(1 for row in rows if row.label == "good"),
        "official_train": len(train),
        "official_train_defect": sum(1 for row in train if row.label == "defect"),
        "official_train_good": sum(1 for row in train if row.label == "good"),
        "official_test": len(test),
        "official_test_defect": sum(1 for row in test if row.label == "defect"),
        "official_test_good": sum(1 for row in test if row.label == "good"),
    }


def _dedupe(messages: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(messages))


def verify_splits(
    rows: Sequence[ManifestRow],
    split_dir: str | Path,
    *,
    enforce_dataset_counts: bool = True,
    p_test_min: int = P_TEST_MIN,
) -> SplitVerifyResult:
    """분할 파일을 다시 읽어 L1-L5/L7/L8과 수량 계약을 검사한다."""

    target = Path(split_dir)
    violations: list[str] = []
    warnings: list[str] = []
    loaded: dict[str, list[ManifestRow]] = {}

    for name, filename in (
        ("tune", "tune.csv"),
        ("test", "test.csv"),
        ("official_test", "official_test.csv"),
    ):
        path = target / filename
        if not path.is_file():
            violations.append(f"{filename} 파일이 없습니다: {path}")
            loaded[name] = []
            continue
        try:
            loaded[name] = load_manifest(path)
        except ManifestError as exc:
            violations.append(f"{filename} 로드 실패: {exc}")
            loaded[name] = []

    tune = loaded["tune"]
    test = loaded["test"]
    official = loaded["official_test"]

    for name, split_rows in loaded.items():
        for row in split_rows:
            if row.sample_id != row.original_id:
                violations.append(
                    f"{name}.csv에 정본이 아닌 행이 있습니다: "
                    f"sample_id={row.sample_id}, original_id={row.original_id}"
                )

    combined = [*tune, *test]
    combined_ids = [row.sample_id for row in combined]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(combined_ids).items() if count > 1
    )
    if duplicate_ids:
        violations.append(
            "tune/test에 같은 sample_id가 두 번 이상 있습니다: " + ", ".join(duplicate_ids[:10])
        )

    canonical = canonical_rows(rows)
    canonical_ids = {row.sample_id for row in canonical}
    missing = sorted(canonical_ids - set(combined_ids))
    extra = sorted(set(combined_ids) - canonical_ids)
    if missing:
        violations.append(
            f"tune/test에 포함되지 않은 정본 행이 {len(missing)}개 있습니다: "
            + ", ".join(missing[:10])
        )
    if extra:
        violations.append(
            f"tune/test에 manifest 정본에 없는 행이 {len(extra)}개 있습니다: "
            + ", ".join(extra[:10])
        )

    tune_originals = {row.original_id for row in tune}
    test_originals = {row.original_id for row in test}
    tune_shas = {row.image_sha256 for row in tune}
    test_shas = {row.image_sha256 for row in test}
    tune_groups = {row.group_id for row in tune}
    test_groups = {row.group_id for row in test}

    l1_intersection = sorted(tune_originals & test_originals)
    l2_intersection = sorted(tune_shas & test_shas)
    l3_intersection = sorted(tune_groups & test_groups)
    if l1_intersection:
        violations.append(
            f"L1 위반: tune과 test에 같은 original_id가 있습니다: {l1_intersection[:10]}"
        )
    if l2_intersection:
        violations.append(
            f"L2 위반: tune과 test에 같은 image_sha256이 있습니다: {l2_intersection[:10]}"
        )
    if l3_intersection:
        violations.append(
            f"L3 위반: tune과 test에 같은 group_id가 있습니다: {l3_intersection[:10]}"
        )

    manifest_by_sample = {row.sample_id: row for row in rows}
    original_to_group: dict[str, str] = {}
    l4_messages: list[str] = []
    for row in rows:
        existing = original_to_group.get(row.original_id)
        if existing is None:
            original_to_group[row.original_id] = row.group_id
        elif existing != row.group_id:
            l4_messages.append(
                f"L4 위반: original_id {row.original_id}가 둘 이상 group_id에 속합니다: "
                f"{existing}, {row.group_id}"
            )
        canonical_row = manifest_by_sample.get(row.original_id)
        if canonical_row is not None and canonical_row.group_id != row.group_id:
            l4_messages.append(
                f"L4 위반: {row.sample_id}가 정본 {row.original_id}의 "
                "group_id를 상속하지 않았습니다."
            )
    for name, split_rows in loaded.items():
        for row in split_rows:
            manifest_row = manifest_by_sample.get(row.sample_id)
            if manifest_row is None:
                l4_messages.append(f"L4 위반: {name}.csv 행이 manifest에 없습니다: {row.sample_id}")
                continue
            if row.group_id != manifest_row.group_id:
                l4_messages.append(
                    f"L4 위반: {name}.csv {row.sample_id}의 group_id가 manifest와 다릅니다."
                )
            if row.original_id != manifest_row.original_id:
                l4_messages.append(
                    f"L4 위반: {name}.csv {row.sample_id}의 original_id가 manifest와 다릅니다."
                )
    l4_messages = _dedupe(l4_messages)
    violations.extend(l4_messages)

    l5_ok = False
    test_csv = target / "test.csv"
    test_hash_file = target / "test.sha256"
    if test_csv.is_file() and test_hash_file.is_file():
        actual_sha = hashlib.sha256(test_csv.read_bytes()).hexdigest()
        expected_text = f"{actual_sha}  test.csv\n"
        recorded_text = test_hash_file.read_text(encoding="utf-8")
        l5_ok = recorded_text == expected_text
        if not l5_ok:
            violations.append(
                "L5 위반: test.sha256 내용이 test.csv 해시와 일치하지 않습니다."
            )
    else:
        violations.append("L5 위반: test.csv 또는 test.sha256 파일이 없습니다.")

    official_ids = {row.sample_id for row in official}
    official_groups = {row.group_id for row in official}
    overlap_payload = {
        "internal_test": {
            "samples": len(official_ids & {row.sample_id for row in test}),
            "groups": len(official_groups & test_groups),
        },
        "internal_tune": {
            "samples": len(official_ids & {row.sample_id for row in tune}),
            "groups": len(official_groups & tune_groups),
        },
    }

    l8_messages: list[str] = []
    for name, split_rows in loaded.items():
        for row in split_rows:
            expected_label = "defect" if row.defect_pixels > 0 else "good"
            if row.label != expected_label:
                l8_messages.append(
                    f"L8 위반: {name}.csv {row.sample_id}의 label과 defect_pixels가 "
                    f"일치하지 않습니다: label={row.label}, defect_pixels={row.defect_pixels}"
                )
            if row.label == "defect" and not row.mask_path:
                l8_messages.append(
                    f"L8 위반: {name}.csv 결함 행 {row.sample_id}에 mask_path가 없습니다."
                )
            manifest_row = manifest_by_sample.get(row.sample_id)
            if manifest_row is not None and (
                row.label != manifest_row.label
                or row.defect_pixels != manifest_row.defect_pixels
            ):
                l8_messages.append(
                    f"L8 위반: {name}.csv {row.sample_id}의 라벨/마스크 수치가 "
                    "manifest와 다릅니다."
                )
    l8_messages = _dedupe(l8_messages)
    violations.extend(l8_messages)

    expected_official = {
        row.sample_id: row for row in canonical if row.source_split == "test"
    }
    actual_official = {row.sample_id: row for row in official}
    missing_official = sorted(set(expected_official) - set(actual_official))
    extra_official = sorted(set(actual_official) - set(expected_official))
    official_mismatches: list[str] = []
    for sample_id in sorted(set(expected_official) & set(actual_official)):
        expected_row = expected_official[sample_id]
        actual_row = actual_official[sample_id]
        for field_name in ("label", "defect_pixels", "group_id", "original_id"):
            expected_value = getattr(expected_row, field_name)
            actual_value = getattr(actual_row, field_name)
            if actual_value != expected_value:
                official_mismatches.append(
                    f"official_test.csv {sample_id}의 {field_name}가 canonical 값과 "
                    f"다릅니다: 기대={expected_value}, 실제={actual_value}"
                )
    if missing_official:
        violations.append(
            f"official_test.csv 누락: canonical source_split=test 행 "
            f"{len(missing_official)}개가 없습니다: "
            + ", ".join(missing_official[:10])
        )
    if extra_official:
        violations.append(
            f"official_test.csv 추가: canonical source_split=test에 없는 행 "
            f"{len(extra_official)}개가 있습니다: "
            + ", ".join(extra_official[:10])
        )
    violations.extend(official_mismatches)

    report_path = target / "split_report.json"
    report_payload: dict[str, Any] | None = None
    report_violations: list[str] = []
    if not report_path.is_file():
        report_violations.append(f"split_report.json이 없습니다: {report_path}")
    else:
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report_violations.append(f"split_report.json을 읽지 못했습니다: {exc}")

    if isinstance(report_payload, dict):
        report_splits = report_payload.get("splits")
        actual_hashes: dict[str, str] = {}
        for split_name, filename in (
            ("tune", "tune.csv"),
            ("test", "test.csv"),
            ("official_test", "official_test.csv"),
        ):
            split_path = target / filename
            if not split_path.is_file():
                continue
            actual_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
            actual_hashes[split_name] = actual_sha
            recorded_sha = None
            if isinstance(report_splits, Mapping):
                section = report_splits.get(split_name)
                if isinstance(section, Mapping):
                    recorded_sha = section.get("sha256")
            if recorded_sha != actual_sha:
                report_violations.append(
                    f"split_report.json {split_name} sha256가 실제 {filename} "
                    "해시와 다릅니다."
                )
        required_keys = ("seed", "test_ratio_target", "p_test_min", "dhash_threshold")
        missing_keys = [key for key in required_keys if key not in report_payload]
        if missing_keys:
            report_violations.append(
                f"split_report.json 필수 키가 없습니다: {missing_keys}"
            )
        if len(actual_hashes) == 3 and not missing_keys:
            hash_payload = {
                "seed": report_payload["seed"],
                "test_ratio_target": report_payload["test_ratio_target"],
                "p_test_min": report_payload["p_test_min"],
                "dhash_threshold": report_payload["dhash_threshold"],
                "tune_sha256": actual_hashes["tune"],
                "test_sha256": actual_hashes["test"],
                "official_test_sha256": actual_hashes["official_test"],
            }
            canonical_json = json.dumps(
                hash_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            recomputed_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
            if report_payload.get("split_hash") != recomputed_hash:
                report_violations.append(
                    "split_report.json split_hash가 recipe 재계산 값과 다릅니다."
                )
        if report_payload.get("split_hash_recipe") is None:
            report_violations.append(
                "split_report.json에 split_hash_recipe가 없습니다."
            )
    violations.extend(report_violations)

    official_counts = _official_count_payload(canonical)
    if enforce_dataset_counts:
        for key, expected in EXPECTED_DATASET_COUNTS.items():
            actual = official_counts[key]
            if actual != expected:
                violations.append(
                    f"데이터 수량 불일치: {key} 기대 {expected}, 실제 {actual}"
                )

    p_test = sum(1 for row in test if row.label == "defect")
    test_ratio = len(test) / len(canonical) if canonical else 0.0
    if p_test < p_test_min:
        violations.append(f"P_test가 {p_test_min} 미만입니다: {p_test}")
    if not (TEST_RATIO_MIN <= test_ratio <= TEST_RATIO_MAX):
        violations.append(
            f"test 비율이 {TEST_RATIO_MIN}-{TEST_RATIO_MAX} 범위를 벗어났습니다: "
            f"{test_ratio:.4f}"
        )

    l_checks: dict[str, Any] = {
        "L1": {
            "status": "PASS" if not l1_intersection else "FAIL",
            "value": len(l1_intersection),
            "detail": "tune/test original_id 교집합",
        },
        "L2": {
            "status": "PASS" if not l2_intersection else "FAIL",
            "value": len(l2_intersection),
            "detail": "tune/test image_sha256 교집합",
        },
        "L3": {
            "status": "PASS" if not l3_intersection else "FAIL",
            "value": len(l3_intersection),
            "detail": "tune/test group_id 교집합",
        },
        "L4": {
            "status": "PASS" if not l4_messages else "FAIL",
            "value": len(l4_messages),
            "detail": "그룹 상속/일관성 위반",
        },
        "L5": {
            "status": "PASS" if l5_ok else "FAIL",
            "value": 1 if l5_ok else 0,
            "detail": "test.csv 해시와 test.sha256 일치",
        },
        "L6": {
            "status": "NOT_APPLICABLE",
            "value": 0,
            "detail": "T03 범위에서는 run 로그를 만들지 않습니다.",
        },
        "L7": {
            "status": "REPORTED",
            "value": overlap_payload["internal_test"]["samples"]
            + overlap_payload["internal_tune"]["samples"],
            "detail": "공식 test와 내부 test/tune 중복(보고만, 튜닝 없음)",
            "overlap": overlap_payload,
        },
        "L8": {
            "status": "PASS" if not l8_messages else "FAIL",
            "value": len(l8_messages),
            "detail": "라벨/마스크 일치",
        },
        "official_test_identity": {
            "status": (
                "PASS"
                if not (missing_official or extra_official or official_mismatches)
                else "FAIL"
            ),
            "value": len(missing_official)
            + len(extra_official)
            + len(official_mismatches),
            "detail": (
                "official_test.csv 행 집합/라벨/defect_pixels/group_id/original_id 일치"
            ),
            "missing": len(missing_official),
            "extra": len(extra_official),
            "mismatch": len(official_mismatches),
        },
        "split_report": {
            "status": "PASS" if not report_violations else "FAIL",
            "value": len(report_violations),
            "detail": "split_report.json 분할 sha256과 split_hash 재계산 검증",
        },
    }

    counts_payload = {
        "manifest": counts(rows),
        "canonical": counts(canonical),
        "tune": counts(tune),
        "test": counts(test),
        "official_test": counts(official),
        "official_test_identity": {
            "expected_rows": len(expected_official),
            "actual_rows": len(actual_official),
            "missing": len(missing_official),
            "extra": len(extra_official),
            "mismatch": len(official_mismatches),
        },
    }

    violations = _dedupe(violations)
    return SplitVerifyResult(
        ok=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
        counts=counts_payload,
        p_test=p_test,
        test_ratio=test_ratio,
        overlap=overlap_payload,
        l_checks=l_checks,
        report=report_payload,
    )


__all__ = [
    "DEFAULT_SEED",
    "DESIGN_VERSION",
    "EXPECTED_CANONICAL_ROWS",
    "EXPECTED_DATASET_COUNTS",
    "EXPECTED_DEFECT_ROWS",
    "EXPECTED_GOOD_ROWS",
    "EXPECTED_OFFICIAL_TEST_DEFECT",
    "EXPECTED_OFFICIAL_TEST_GOOD",
    "EXPECTED_OFFICIAL_TEST_ROWS",
    "EXPECTED_OFFICIAL_TRAIN_DEFECT",
    "EXPECTED_OFFICIAL_TRAIN_GOOD",
    "EXPECTED_OFFICIAL_TRAIN_ROWS",
    "Group",
    "GroupAssignment",
    "P_TEST_MIN",
    "SPLIT_TEST",
    "SPLIT_TUNE",
    "SplitError",
    "SplitVerifyResult",
    "TEST_RATIO_MAX",
    "TEST_RATIO_MIN",
    "TEST_RATIO_TARGET",
    "assign_groups",
    "build_groups",
    "verify_splits",
    "write_splits",
]
