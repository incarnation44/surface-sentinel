"""git_code_identity 진입점 헬퍼 단위 테스트 (P0-2 code_sha 주입).

code_sha/code_dirty 는 진입점(API 생성·워커 시작·평가 CLI)에서 1회 계산한다.
subprocess 는 스텁으로 대체해 git 없이 결정론적으로 검증한다.
"""

from __future__ import annotations

import types

import pytest

from surface_sentinel.evaluation import recalibration


def _completed(stdout: str, returncode: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(stdout=stdout, returncode=returncode)


def test_clean_tree_returns_sha_and_not_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        calls.append(cmd)
        if cmd[:2] == ["git", "rev-parse"]:
            return _completed("a" * 40 + "\n")
        return _completed("")

    monkeypatch.setattr(recalibration.subprocess, "run", fake_run)
    sha, dirty = recalibration.git_code_identity()
    assert sha == "a" * 40
    assert dirty is False
    assert len(calls) == 2  # rev-parse 1회 + status 1회(진입점 1회 계산)


def test_dirty_tree_returns_dirty_true(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        if cmd[:2] == ["git", "rev-parse"]:
            return _completed("b" * 40)
        return _completed(" M src/x.py\n?? new.py\n")

    monkeypatch.setattr(recalibration.subprocess, "run", fake_run)
    assert recalibration.git_code_identity() == ("b" * 40, True)


def test_git_missing_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        raise OSError("git not found")

    monkeypatch.setattr(recalibration.subprocess, "run", fake_run)
    assert recalibration.git_code_identity() == ("", False)


def test_status_failure_keeps_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        if cmd[:2] == ["git", "rev-parse"]:
            return _completed("c" * 40)
        return _completed("", returncode=128)

    monkeypatch.setattr(recalibration.subprocess, "run", fake_run)
    assert recalibration.git_code_identity() == ("c" * 40, False)
