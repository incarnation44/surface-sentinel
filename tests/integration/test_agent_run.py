"""T09 에이전트 실행 통합 테스트. 합성 fixture와 실제 파이프라인만 사용한다.

봉인 분할은 열지 않으며, 판정 소유권이 파이프라인에 있음을 확인한다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from surface_sentinel.agent.config import load_agent_config
from surface_sentinel.agent.run import (
    AgentRunError,
    load_profiles,
    run_agent_episode,
)
from surface_sentinel.agent.trace import audit_trace, load_trace
from surface_sentinel.cli import main
from surface_sentinel.config import load_config
from surface_sentinel.contracts import Status

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.toml"
AGENT_CONFIG_PATH = REPO_ROOT / "configs" / "agent.toml"
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"
SYNTHETIC_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"

BASE_CONFIG = load_config(BASE_CONFIG_PATH)
AGENT_CONFIG = load_agent_config(AGENT_CONFIG_PATH)
PROFILES = load_profiles(BASE_CONFIG_PATH, PROFILES_DIR)


def _run(tmp_path: Path, image: Path, *, profiles=None, **kwargs):
    return run_agent_episode(
        image,
        agent_config=AGENT_CONFIG,
        profiles=profiles if profiles is not None else PROFILES,
        output_dir=tmp_path / "runs",
        **kwargs,
    )


def test_good_fixture_done_ok_with_clean_audit(tmp_path: Path) -> None:
    outcome = _run(tmp_path, SYNTHETIC_DIR / "good.png")
    episode = outcome.episode
    assert episode.terminal_state.value == "DONE"
    assert episode.final_status is Status.OK
    assert episode.profiles_used == 1
    assert outcome.trace_path.is_file()
    assert audit_trace(load_trace(outcome.trace_path)) == []
    payload = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "agent-run-v1"
    assert payload["final_status"] == "OK"
    assert payload["config_hash"] == BASE_CONFIG.config_hash


def test_dark_defect_fixture_done_ng(tmp_path: Path) -> None:
    outcome = _run(
        tmp_path,
        SYNTHETIC_DIR / "dark_defect.png",
        mask_path=SYNTHETIC_DIR / "dark_defect_mask.png",
        label="defect",
    )
    assert outcome.episode.terminal_state.value == "DONE"
    assert outcome.episode.final_status is Status.NG
    assert outcome.episode.actions_used == 1


def test_quality_low_runs_second_profile_and_never_optimistic_ok(tmp_path: Path) -> None:
    profiles = dict(PROFILES)
    profiles["base"] = replace(
        BASE_CONFIG,
        quality=replace(BASE_CONFIG.quality, dark_mean_min=200.0),
    )
    outcome = _run(tmp_path, SYNTHETIC_DIR / "good.png", profiles=profiles)
    episode = outcome.episode
    assert episode.profiles_used == 2
    assert [observation.profile_id for observation in episode.results] == [
        "base",
        "denoise_strong",
    ]
    assert episode.final_status is not Status.OK
    assert episode.terminal_state.value == "REVIEW"
    assert audit_trace(load_trace(outcome.trace_path)) == []


def test_roi_uncertain_runs_roi_fixed_profile(tmp_path: Path) -> None:
    profiles = dict(PROFILES)
    profiles["base"] = replace(
        BASE_CONFIG,
        roi=replace(BASE_CONFIG.roi, min_area_ratio=1.0),
    )
    outcome = _run(tmp_path, SYNTHETIC_DIR / "good.png", profiles=profiles)
    episode = outcome.episode
    assert episode.profiles_used == 2
    assert [observation.profile_id for observation in episode.results] == [
        "base",
        "roi_fixed",
    ]
    assert episode.terminal_state.value == "REVIEW"
    assert audit_trace(load_trace(outcome.trace_path)) == []


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_sealed_splits_are_never_opened(tmp_path: Path, split: str) -> None:
    with pytest.raises(AgentRunError) as excinfo:
        _run(tmp_path, SYNTHETIC_DIR / "good.png", split=split)
    assert excinfo.value.code == "E_SEALED_DENIED"


def test_missing_image_rejected(tmp_path: Path) -> None:
    with pytest.raises(AgentRunError) as excinfo:
        _run(tmp_path, tmp_path / "missing.png")
    assert excinfo.value.code == "E_INPUT_MISSING"


def test_run_id_collision_gets_suffix(tmp_path: Path) -> None:
    first = _run(tmp_path, SYNTHETIC_DIR / "good.png", run_id="fixed-run")
    second = _run(tmp_path, SYNTHETIC_DIR / "good.png", run_id="fixed-run")
    assert first.run_id == "fixed-run"
    assert second.run_id == "fixed-run-01"


def _cli_run(tmp_path: Path, image: Path, *extra: str) -> int:
    return main(
        [
            "agent",
            "run",
            "--image",
            str(image),
            "--output-dir",
            str(tmp_path / "runs"),
            "--config",
            str(BASE_CONFIG_PATH),
            "--agent-config",
            str(AGENT_CONFIG_PATH),
            "--profiles-dir",
            str(PROFILES_DIR),
            *extra,
        ]
    )


def test_cli_run_replay_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert _cli_run(tmp_path, SYNTHETIC_DIR / "good.png") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_status"] == "OK"
    trace_path = Path(payload["trace_path"])
    assert trace_path.is_file()

    assert main(["agent", "replay", "--trace", str(trace_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["deterministic"] is True
    assert report["audit_valid"] is True
    assert report["final_status"] == "OK"


def test_cli_sealed_split_exit_6(tmp_path: Path) -> None:
    assert _cli_run(tmp_path, SYNTHETIC_DIR / "good.png", "--split", "test") == 6


def test_cli_missing_image_exit_3(tmp_path: Path) -> None:
    assert _cli_run(tmp_path, tmp_path / "missing.png") == 3


def test_cli_replay_tampered_trace_exit_7(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert _cli_run(tmp_path, SYNTHETIC_DIR / "good.png") == 0
    payload = json.loads(capsys.readouterr().out)
    trace_path = Path(payload["trace_path"])
    events = load_trace(trace_path)
    for event in events:
        if event["event"] == "act" and event["tool"] == "run_profile":
            event["result"]["status"] = "NG"
    trace_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    assert main(["agent", "replay", "--trace", str(trace_path)]) == 7
    report = json.loads(capsys.readouterr().out)
    assert report["deterministic"] is False


def test_cli_replay_missing_trace_exit_3(tmp_path: Path) -> None:
    assert main(["agent", "replay", "--trace", str(tmp_path / "missing.jsonl")]) == 3
