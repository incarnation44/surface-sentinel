"""CLI ``inspect`` 서브커맨드 종료 코드와 한국어 도움말 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surface_sentinel.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.toml"
SYNTHETIC_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"
DARK_IMAGE = SYNTHETIC_DIR / "dark_defect.png"
DARK_MASK = SYNTHETIC_DIR / "dark_defect_mask.png"
GOOD_IMAGE = SYNTHETIC_DIR / "good.png"


def _inspect_args(tmp_path: Path, image: Path, *extra: str) -> list[str]:
    return [
        "inspect",
        "--image",
        str(image),
        "--config",
        str(BASE_CONFIG_PATH),
        "--output",
        str(tmp_path / "out"),
        *extra,
    ]


def test_inspect_help_is_korean(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["inspect", "--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "검사" in output
    assert "봉인" in output
    assert "--image" in output
    assert "--mask" in output
    assert "--split" in output


def test_inspect_good_returns_exit_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(_inspect_args(tmp_path, GOOD_IMAGE))

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "OK"
    assert payload["reasons"] == ["NO_DEFECT_EVIDENCE"]
    assert "판정: OK" in captured.err
    assert "오버레이:" in captured.err


def test_inspect_dark_defect_returns_exit_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        _inspect_args(
            tmp_path,
            DARK_IMAGE,
            "--mask",
            str(DARK_MASK),
            "--label",
            "defect",
        )
    )

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "NG"
    assert payload["reasons"] == ["DEFECT_EVIDENCE_STRONG"]


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_inspect_sealed_split_returns_exit_six_without_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    split: str,
) -> None:
    code = main(_inspect_args(tmp_path, DARK_IMAGE, "--split", split))

    captured = capsys.readouterr()
    assert code == 6
    assert captured.out == ""
    assert "봉인" in captured.err


def test_inspect_missing_image_returns_exit_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(_inspect_args(tmp_path, tmp_path / "missing.png"))

    captured = capsys.readouterr()
    assert code == 3
    payload = json.loads(captured.out)
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "E_INPUT_MISSING"
    assert "검사 오류" in captured.err


def test_inspect_missing_config_returns_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "inspect",
            "--image",
            str(GOOD_IMAGE),
            "--config",
            str(tmp_path / "missing.toml"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "설정" in captured.err


def test_inspect_invalid_config_returns_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = BASE_CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("use_dark_channel = true", "use_dark_channel = false")
    text = text.replace("use_bright_channel = true", "use_bright_channel = false")
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(text, encoding="utf-8")

    code = main(
        [
            "inspect",
            "--image",
            str(GOOD_IMAGE),
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "설정" in captured.err


def test_inspect_storage_failure_returns_exit_five(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(source: object, destination: object) -> None:
        raise OSError("의도된 저장 실패")

    monkeypatch.setattr(
        "surface_sentinel.pipeline.evidence.os.replace",
        fail_replace,
    )

    code = main(_inspect_args(tmp_path, DARK_IMAGE))

    captured = capsys.readouterr()
    assert code == 5
    payload = json.loads(captured.out)
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "E_STORAGE_WRITE"
    assert payload["error_stage"] == "save_evidence"


def test_inspect_other_error_returns_exit_four(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("의도된 내부 오류")

    monkeypatch.setattr(
        "surface_sentinel.pipeline.inspect.calculate_score",
        explode,
    )

    code = main(_inspect_args(tmp_path, DARK_IMAGE))

    captured = capsys.readouterr()
    assert code == 4
    payload = json.loads(captured.out)
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "E_PROCESS_INTERNAL"
    assert payload["error_stage"] == "calculate_score"


def test_inspect_requires_image_argument(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["inspect"])

    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err
