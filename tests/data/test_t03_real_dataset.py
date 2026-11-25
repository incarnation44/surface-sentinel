"""실제 KSDD2 반입본(`data/`)의 manifest 수량·그룹 분할 누출·해시 일치 검증(`@pytest.mark.data`).

원본 전체 재디코드는 ``surface-sentinel data verify``가 담당하므로 이
테스트는 manifest/split CSV만 읽어 수량과 누출을 확인한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from surface_sentinel.io.manifest import canonical_rows, counts, load_manifest
from surface_sentinel.io.splits import verify_splits

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "kolektorsdd2"
MANIFEST = ROOT / "data" / "manifests" / "dataset_manifest.csv"
SPLITS = ROOT / "data" / "splits"


def _require_artifacts() -> None:
    missing = [
        str(path)
        for path in (RAW_ROOT, MANIFEST, SPLITS / "tune.csv", SPLITS / "test.csv")
        if not path.exists()
    ]
    if missing:
        pytest.skip("T03 반입본/manifest/split이 없습니다: " + ", ".join(missing))


def test_real_manifest_official_counts() -> None:
    _require_artifacts()
    rows = load_manifest(MANIFEST)

    assert len(rows) == 3336
    summary = counts(rows)
    assert summary == {
        "total_rows": 3336,
        "canonical_rows": 3335,
        "defect": 357,
        "good": 2979,
        "official_train": 2332,
        "official_test": 1004,
        "duplicate_rows": 1,
    }

    canonical = canonical_rows(rows)
    assert len(canonical) == 3335
    canonical_counts = counts(canonical)
    assert canonical_counts["defect"] == 356
    assert canonical_counts["good"] == 2979
    assert canonical_counts["official_train"] == 2331
    assert canonical_counts["official_test"] == 1004

    train_defect = sum(
        1 for row in canonical if row.source_split == "train" and row.label == "defect"
    )
    train_good = sum(
        1 for row in canonical if row.source_split == "train" and row.label == "good"
    )
    test_defect = sum(
        1 for row in canonical if row.source_split == "test" and row.label == "defect"
    )
    test_good = sum(
        1 for row in canonical if row.source_split == "test" and row.label == "good"
    )
    assert (train_defect, train_good) == (246, 2085)
    assert (test_defect, test_good) == (110, 894)


def test_real_splits_pass_leakage_and_quantity_contract() -> None:
    _require_artifacts()
    rows = load_manifest(MANIFEST)

    result = verify_splits(rows, SPLITS, enforce_dataset_counts=True)

    assert result.violations == ()
    assert result.ok is True
    assert result.p_test >= 100
    assert 0.38 <= result.test_ratio <= 0.42
    assert result.l_checks["L1"]["value"] == 0
    assert result.l_checks["L2"]["value"] == 0
    assert result.l_checks["L3"]["value"] == 0
    assert result.l_checks["L4"]["status"] == "PASS"
    assert result.l_checks["L5"]["status"] == "PASS"
    assert result.l_checks["L8"]["status"] == "PASS"
    assert result.l_checks["official_test_identity"]["status"] == "PASS"
    assert result.l_checks["official_test_identity"]["missing"] == 0
    assert result.l_checks["official_test_identity"]["extra"] == 0
    assert result.l_checks["official_test_identity"]["mismatch"] == 0
    assert result.counts["official_test_identity"]["expected_rows"] == 1004
    assert result.counts["official_test_identity"]["actual_rows"] == 1004
    assert result.l_checks["split_report"]["status"] == "PASS"
    assert result.counts["canonical"]["total_rows"] == 3335
    assert result.report is not None
    assert result.report["seed"] == 42

    expected_test_sha = hashlib.sha256((SPLITS / "test.csv").read_bytes()).hexdigest()
    assert (SPLITS / "test.sha256").read_text(encoding="utf-8") == (
        f"{expected_test_sha}  test.csv\n"
    )
