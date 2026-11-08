"""CLI skeleton tests."""

from __future__ import annotations

import json

import pytest

from surface_sentinel.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "surface-sentinel 0.1.0" in capsys.readouterr().out


def test_doctor_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["package_version"] == "0.1.0"
    assert payload["config_schema"] == "1.0"
    assert "python" in payload
    assert "opencv" in payload
    assert "numpy" in payload


def test_no_command_returns_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["unknown"])
    assert excinfo.value.code == 2
