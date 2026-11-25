"""후보 생성 재보정(`evaluation/recalibration.py`) 단위 테스트.

후보 TOML 로딩, 5개 하드 게이트 동시 판정, JSONL·보고서 출력 및 봉인 분할 차단을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from surface_sentinel.evaluation.grid import GridRow
from surface_sentinel.evaluation.metrics import Ratio
from surface_sentinel.evaluation.recalibration import (
    TARGET_ESCAPE_CI_HIGH,
    RecalibrationError,
    RecalibrationRow,
    load_candidate_configs,
    run_recalibration,
    write_jsonl,
    write_report,
)


class _StubReport:
    """``GridRow``가 요구하는 표면만 흉내 내는 보고서 대역."""

    def __init__(self, values: dict[str, tuple[float | None, float | None]]) -> None:
        self._values = values
        self.counts: dict[str, Any] = {"samples": 714, "defect": 214, "good": 500}
        self.outcomes: tuple[Any, ...] = ()

    def ratio(self, name: str) -> Ratio:
        value, ci_high = self._values.get(name, (0.0, 0.10))
        return Ratio(
            definition=name,
            numerator=None,
            denominator=None,
            value=value,
            ci_low=None if value is None else 0.0,
            ci_high=ci_high,
        )


@pytest.fixture()
def workdir() -> Path:
    """임시 디렉터리 대신 저장소 안의 무시된 경로를 쓴다(정리 단계 회피)."""

    path = Path(".work/unit_recalibration")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _row(
    escape: float | None = 0.10,
    escape_high: float | None = 0.15,
    false_ng: float | None = 0.10,
    review: float | None = 0.20,
    recall: float | None = 0.60,
    candidate_id: str = "r03",
) -> RecalibrationRow:
    report = _StubReport(
        {
            "defect_escape_rate": (escape, escape_high),
            "good_false_ng_rate": (false_ng, 0.15),
            "review_rate": (review, 0.25),
            "localization_recall_iou0.10": (recall, 0.70),
        }
    )
    grid_row = GridRow(
        candidate_id=candidate_id,
        overrides={"candidates.z_threshold": 6.0},
        config_hash="deadbeef",
        report=report,  # type: ignore[arg-type]
    )
    return RecalibrationRow(
        candidate_id=candidate_id,
        config_path=f"configs/candidates/recalib/{candidate_id}.toml",
        config_hash="deadbeef",
        grid_row=grid_row,
        candidates_params={"z_threshold": 6.0},
        scoring_params={"ng_area_px": 800.0},
        truncated=3,
        samples=714,
        components_total=1428,
        elapsed_seconds=1.5,
        cache_hit=False,
        examples={"escape": ["a"], "false_ng": [], "review": ["c"]},
    )


# ------------------------------------------------------------------ 후보 로딩


def test_load_candidate_configs_missing_dir(workdir: Path) -> None:
    with pytest.raises(RecalibrationError):
        load_candidate_configs(workdir / "missing")


def test_load_candidate_configs_empty_dir(workdir: Path) -> None:
    empty = workdir / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RecalibrationError):
        load_candidate_configs(empty)


def test_load_candidate_configs_sorted(workdir: Path) -> None:
    directory = workdir / "sorted"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("b_two", "a_one"):
        (directory / f"{name}.toml").write_text(
            'schema_version = "1.0"\n', encoding="utf-8"
        )
    (directory / "notes.txt").write_text("ignore me", encoding="utf-8")
    loaded = load_candidate_configs(directory)
    assert [candidate_id for candidate_id, _ in loaded] == ["a_one", "b_two"]


# ---------------------------------------------------------------- 통과 기준


def test_passes_when_all_five_criteria_met() -> None:
    row = _row()
    assert row.passes() is True
    assert all(value is True for value in row.criteria().values())


def test_fails_when_escape_ci_high_exceeds_limit() -> None:
    row = _row(escape=0.12, escape_high=TARGET_ESCAPE_CI_HIGH + 0.01)
    assert row.criteria()["escape_ci95_high_le_0.20"] is False
    assert row.passes() is False


def test_fails_when_recall_below_limit() -> None:
    row = _row(recall=0.49)
    assert row.criteria()["localization_recall_ge_0.50"] is False
    assert row.passes() is False


def test_fails_when_any_metric_is_undefined() -> None:
    row = _row(false_ng=None)
    assert row.criteria()["false_ng_le_0.20"] is None
    assert row.passes() is False


def test_avg_components_uses_sample_count() -> None:
    row = _row()
    assert row.avg_components == pytest.approx(1428 / 714)


# ------------------------------------------------------------------- 산출물


def test_write_jsonl_round_trips_one_row(workdir: Path) -> None:
    target = write_jsonl([_row()], workdir / "tune_grid2.jsonl")
    payload = json.loads(target.read_text(encoding="utf-8").strip())
    assert payload["candidate_id"] == "r03"
    assert payload["truncated"]["count"] == 3
    assert payload["criteria"]["escape_le_0.15"] is True
    assert payload["passes"] is True
    assert payload["config_path"].startswith("configs/candidates/recalib/")


def test_write_report_contains_sections_and_criteria(workdir: Path) -> None:
    path = write_report(
        [_row()],
        workdir / "report.md",
        jsonl_path="docs/evidence/tune_grid2.jsonl",
        code_revision="c618ca4",
    )
    text = path.read_text(encoding="utf-8")
    assert "# T04 후보 생성 재보정 보고서" in text
    assert "## 4. 통과 기준 대조" in text
    assert "c618ca4" in text
    assert "r03" in text
    assert "base.toml" in text


def test_write_report_honors_title_candidate_dir_and_cap_note(workdir: Path) -> None:
    path = write_report(
        [_row()],
        workdir / "report_fullverify.md",
        jsonl_path="docs/evidence/tune_fullverify.jsonl",
        title="T04 전수 tune 검증 보고서",
        candidate_dir="configs/candidates/fullverify",
        good_target=1787,
        cap_note="(전수 검증)",
    )
    text = path.read_text(encoding="utf-8")
    assert "# T04 전수 tune 검증 보고서" in text
    assert "(층화 1787)" in text
    assert "후보 수: 1 (전수 검증)" in text
    assert "--candidate-dir configs/candidates/fullverify" in text
    assert "configs/candidates/fullverify/*.toml" in text


# ------------------------------------------------------------------ 봉인 차단


def test_run_recalibration_rejects_sealed_split() -> None:
    with pytest.raises(RecalibrationError):
        run_recalibration(split="test")
