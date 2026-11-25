"""배치 검토 저장소(`review_decisions.jsonl`) 단위 테스트 — append-only, REVIEW 제약, 사유 검증.

합성 입력만 사용한다. 모든 파일은 ``tmp_path`` 아래에 만들어지고
저장소의 ``runs/``는 건드리지 않는다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ui_run
from surface_sentinel.review.store import (
    DECISION_FIELDS,
    REVIEW_DECISIONS_JSONL,
    ReviewDecision,
    ReviewError,
    ReviewStorageError,
    ReviewStore,
)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / ui_run.RUN_ID
    path.mkdir()
    return path


def _append(store: ReviewStore, **overrides: str) -> ReviewDecision:
    kwargs: dict[str, str] = {
        "sample_id": "70002",
        "machine_status": "REVIEW",
        "final_status": "NG",
        "reason_code": "CONFIRMED_DEFECT",
        "reviewer": "local",
        "config_hash": ui_run.CONFIG_HASH,
        "overlay_sha256": "a" * 64,
    }
    kwargs.update(overrides)
    return store.append(**kwargs)


def test_append_writes_contract_field_order(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    decision = _append(store)
    assert store.path.name == REVIEW_DECISIONS_JSONL
    assert decision.run_id == run_dir.name
    assert decision.decided_at.endswith("Z")
    assert len(decision.decision_id) == 32
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert list(payload.keys()) == list(DECISION_FIELDS)
    assert payload["machine_status"] == "REVIEW"
    assert payload["final_status"] == "NG"


def test_append_strips_note_and_reviewer(run_dir: Path) -> None:
    decision = _append(ReviewStore(run_dir), note="  메모  ", reviewer="  local  ")
    assert decision.note == "메모"
    assert decision.reviewer == "local"


@pytest.mark.parametrize("machine_status", ["OK", "NG", "ERROR"])
def test_non_review_machine_status_rejected(run_dir: Path, machine_status: str) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, machine_status=machine_status)
    assert not store.path.exists()


@pytest.mark.parametrize("final_status", ["MAYBE", "ok", ""])
def test_invalid_final_status_rejected(run_dir: Path, final_status: str) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, final_status=final_status)
    assert not store.path.exists()


def test_invalid_reason_code_rejected(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, reason_code="NO_SUCH_REASON")
    assert not store.path.exists()


@pytest.mark.parametrize("note", ["", "   "])
def test_other_reason_requires_note(run_dir: Path, note: str) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, reason_code="OTHER", note=note)
    assert not store.path.exists()


def test_other_reason_with_note_allowed(run_dir: Path) -> None:
    decision = _append(ReviewStore(run_dir), reason_code="OTHER", note="경계 조명 반사")
    assert decision.reason_code == "OTHER"


@pytest.mark.parametrize("field", ["reviewer", "sample_id"])
def test_blank_required_text_rejected(run_dir: Path, field: str) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, **{field: "   "})
    assert not store.path.exists()


@pytest.mark.parametrize("field", ["config_hash", "overlay_sha256"])
def test_invalid_hash_rejected(run_dir: Path, field: str) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, **{field: "xyz"})
    assert not store.path.exists()


def test_decided_at_must_be_utc(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    with pytest.raises(ReviewError):
        _append(store, decided_at="2026-11-12T00:00:00")
    assert not store.path.exists()


def test_last_decision_wins_and_history_preserved(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    first = _append(store, final_status="NG")
    first_bytes = store.path.read_bytes()
    second = _append(store, final_status="OK", reason_code="FALSE_ALARM")
    # append-only: 두 번째 append 뒤에도 첫 줄 바이트는 그대로여야 한다.
    assert store.path.read_bytes().startswith(first_bytes)
    decisions = store.read_all()
    assert [d.decision_id for d in decisions] == [first.decision_id, second.decision_id]
    assert store.latest()["70002"].final_status == "OK"


def test_read_all_missing_file_returns_empty(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    assert store.read_all() == []
    assert store.latest() == {}


def test_read_all_skips_blank_lines(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    _append(store)
    body = store.path.read_text(encoding="utf-8")
    store.path.write_text("\n" + body + "\n\n", encoding="utf-8")
    assert len(store.read_all()) == 1


def test_read_all_corrupted_line_raises(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    _append(store)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("{broken json\n")
    with pytest.raises(ReviewError, match="손상"):
        store.read_all()


def test_read_all_schema_violation_raises(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    payload = _append(store).to_dict()
    del payload["note"]
    store.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="스키마"):
        store.read_all()


def test_read_all_non_string_value_raises(run_dir: Path) -> None:
    store = ReviewStore(run_dir)
    payload = _append(store).to_dict()
    payload["note"] = 3
    store.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="문자열"):
        store.read_all()


def test_from_dict_roundtrip(run_dir: Path) -> None:
    decision = _append(ReviewStore(run_dir))
    assert ReviewDecision.from_dict(decision.to_dict()) == decision


def test_append_missing_run_dir_raises(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "no-such-run")
    with pytest.raises(ReviewError):
        _append(store)


def test_fsync_failure_raises_storage_error(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(ReviewStorageError):
        _append(ReviewStore(run_dir))