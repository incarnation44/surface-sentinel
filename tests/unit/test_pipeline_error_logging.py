"""예상 못한 예외의 내부 진단 로깅 테스트(감사 5.8).

ERROR 결과로 변환되는 실패는 그대로 ERROR + ``E_PROCESS_INTERNAL`` 을 유지하되,
원인 추적용 컨텍스트(stage·예외 타입·traceback·run_id·image SHA·config_hash)를
내부 로거에 남겨야 한다. 결과 객체에는 내부 정보가 새지 않아야 한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

import synthetic as syn
from surface_sentinel.config import load_config
from surface_sentinel.contracts import ErrorCode, Stage, Status
from surface_sentinel.pipeline import inspect as inspect_module
from surface_sentinel.pipeline import inspect_image
from surface_sentinel.pipeline.schema import SchemaError

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "base.toml")
LOGGER_NAME = "surface_sentinel.pipeline.inspect"


def _write_image(tmp_path: Path) -> Path:
    return syn.write_png(tmp_path / "inputs" / "sample.png", syn.good_image())


def _error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno >= logging.ERROR
    ]


def _formatted(record: logging.LogRecord) -> str:
    """traceback까지 포함한 최종 로그 문장을 만든다."""

    return logging.Formatter().format(record)


def _assert_context(
    record: logging.LogRecord,
    *,
    exc_type: str,
    stage: Stage,
    run_id: str,
    sample_id: str,
    image_sha256: str,
    config_hash: str,
) -> None:
    message = record.getMessage()
    assert exc_type in message
    assert f"stage={stage.value}" in message
    assert f"run_id={run_id}" in message
    assert f"sample_id={sample_id}" in message
    assert f"image_sha256={image_sha256}" in message
    assert f"config_hash={config_hash}" in message

    assert record.exc_info is not None
    formatted = _formatted(record)
    assert "Traceback (most recent call last)" in formatted
    assert exc_type in formatted


def test_unexpected_exception_logs_internal_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _write_image(tmp_path)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("의도된 내부 오류")

    monkeypatch.setattr(inspect_module, "preprocess", boom)

    outcome = inspect_image(image_path, BASE_CONFIG, output_dir=tmp_path / "out")
    result = outcome.result

    # 판정 동작은 그대로: ERROR + E_PROCESS_INTERNAL + 실패한 stage.
    assert result.status is Status.ERROR
    assert result.error_code is ErrorCode.E_PROCESS_INTERNAL
    assert result.error_stage is Stage.PREPROCESS
    assert result.score == 0.0
    assert result.reasons == ()
    assert outcome.debug is None

    records = _error_records(caplog)
    assert len(records) == 1
    assert result.input_sha256  # metadata가 확보된 뒤의 실패임을 보장한다.
    _assert_context(
        records[0],
        exc_type="RuntimeError",
        stage=Stage.PREPROCESS,
        run_id=result.run_id,
        sample_id=result.sample_id,
        image_sha256=result.input_sha256,
        config_hash=BASE_CONFIG.config_hash,
    )

    # 결과 객체에는 traceback이나 내부 경로가 노출되지 않는다.
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "Traceback" not in payload
    assert "RuntimeError" not in payload
    assert str(tmp_path.name) not in payload
    assert result.overlay_path == ""


def test_schema_failure_logs_internal_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _write_image(tmp_path)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    real_validate = inspect_module.validate_sample_result
    state = {"raised": False}

    def fake_validate(result: Any, **kwargs: Any) -> Any:
        if not state["raised"]:
            state["raised"] = True
            raise SchemaError("의도된 스키마 실패")
        return real_validate(result, **kwargs)

    monkeypatch.setattr(inspect_module, "validate_sample_result", fake_validate)

    outcome = inspect_image(image_path, BASE_CONFIG, output_dir=tmp_path / "out")
    result = outcome.result

    assert state["raised"] is True  # 폴백 검증까지 실제 검증기로 통과했음
    assert result.status is Status.ERROR
    assert result.error_code is ErrorCode.E_PROCESS_INTERNAL
    # 폴백 시점의 current_stage는 저장 단계다(기존 동작 유지).
    assert result.error_stage is Stage.SAVE_EVIDENCE
    assert outcome.debug is None

    records = _error_records(caplog)
    assert len(records) == 1
    assert result.input_sha256
    _assert_context(
        records[0],
        exc_type="SchemaError",
        stage=Stage.SAVE_EVIDENCE,
        run_id=result.run_id,
        sample_id=result.sample_id,
        image_sha256=result.input_sha256,
        config_hash=BASE_CONFIG.config_hash,
    )


def test_expected_pipeline_error_and_success_do_not_log_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    missing = inspect_image(
        tmp_path / "missing.png",
        BASE_CONFIG,
        output_dir=tmp_path / "missing_out",
    )
    assert missing.result.status is Status.ERROR
    assert missing.result.error_code is ErrorCode.E_INPUT_MISSING

    ok = inspect_image(_write_image(tmp_path), BASE_CONFIG, output_dir=tmp_path / "ok_out")
    assert ok.result.status is Status.OK

    assert _error_records(caplog) == []