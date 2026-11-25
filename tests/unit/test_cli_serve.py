"""CLI `serve` 웹 서버 기동 인자 단위 테스트 — 실제 서버 없이 127.0.0.1 바인딩과 포트 검증만 확인.

uvicorn.run을 가로채서 host/port 인자만 확인한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from surface_sentinel.cli import main


def test_serve_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--help"])
    assert excinfo.value.code == 0
    assert "127.0.0.1" in capsys.readouterr().out


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_serve_port_out_of_range(
    tmp_path: Path, port: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    assert main(["serve", "--runs-dir", str(runs), "--port", port]) == 2
    assert "포트 범위" in capsys.readouterr().err


def test_serve_missing_runs_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope"
    assert main(["serve", "--runs-dir", str(missing)]) == 3
    assert "runs 디렉터리" in capsys.readouterr().err


def test_serve_binds_loopback_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["serve", "--runs-dir", str(runs), "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["app"].state.runs_dir == runs
    assert "http://127.0.0.1:8123" in capsys.readouterr().out


def test_serve_keyboard_interrupt_returns_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()

    def fake_run(app: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["serve", "--runs-dir", str(runs)]) == 0