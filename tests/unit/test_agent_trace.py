"""T09 감사 로그·replay 단위 테스트 (설계 §6: append-only, 결정론)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_fakes import make_engine, make_result
from surface_sentinel.agent.trace import (
    TraceWriter,
    audit_trace,
    canonical_json,
    hash_action,
    hash_args,
    hash_result,
    load_trace,
    rebuild_result,
    replay_trace,
    result_summary,
)
from surface_sentinel.contracts import Status


def test_writer_assigns_monotonic_seq_and_appends_only(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append({"event": "a"})
    writer.append({"event": "b"})
    writer.append({"event": "c"})
    writer.close()
    assert [event["seq"] for event in writer.events] == [0, 1, 2]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line) for line in lines] == list(writer.events)


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_hash_helpers_are_deterministic() -> None:
    args = {"profile_id": "base"}
    assert hash_args(args) == hash_args({"profile_id": "base"})
    assert hash_action("run_profile", hash_args(args)) == hash_action(
        "run_profile", hash_args(args)
    )
    assert hash_action("compare", hash_args(args)) != hash_action(
        "run_profile", hash_args(args)
    )
    assert hash_result({"status": "OK"}) == hash_result({"status": "OK"})


def test_summary_roundtrip_for_all_statuses() -> None:
    variants = [
        make_result(Status.OK),
        make_result(Status.NG, reasons=["DEFECT_EVIDENCE_STRONG"], score=0.9),
        make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"],
                    quality_flags=["INPUT_QUALITY_LOW"]),
        make_result(Status.REVIEW, reasons=["ROI_UNCERTAIN"], roi_uncertain=True),
        make_result(Status.ERROR, error_code="E_INPUT_DECODE"),
    ]
    for result in variants:
        summary = result_summary(result)
        assert result_summary(rebuild_result(summary)) == summary


def test_two_runs_produce_identical_traces() -> None:
    results = {
        "base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
        "denoise_strong": make_result(Status.OK),
    }
    first, _ = make_engine(results)
    second, _ = make_engine(results)
    assert canonical_json(list(first.run().events)) == canonical_json(
        list(second.run().events)
    )


def test_replay_from_file_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    engine, _ = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["ROI_UNCERTAIN"]),
            "roi_fixed": make_result(Status.OK),
        },
        trace=writer,
    )
    episode = engine.run()
    writer.close()
    events = load_trace(path)
    assert audit_trace(events) == []
    report = replay_trace(events)
    assert report.deterministic is True
    assert report.audit_valid is True
    assert report.mismatches == ()
    assert report.terminal_state == episode.terminal_state.value
    assert report.final_status == episode.final_status.value
    assert report.events_compared == len(events)


def test_replay_reproduces_latency_budget_review(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    times = iter([0.0, 0.0, 0.0, 6.0])
    engine, _ = make_engine(
        {
            "base": make_result(Status.REVIEW, reasons=["INPUT_QUALITY_LOW"]),
            "denoise_strong": make_result(Status.OK),
        },
        clock=lambda: next(times),
        trace=writer,
    )
    episode = engine.run()
    writer.close()
    assert episode.terminal_state.value == "REVIEW"
    report = replay_trace(load_trace(path))
    assert report.deterministic is True
    assert report.terminal_state == "REVIEW"
    assert report.final_status == "REVIEW"


def test_tampered_result_breaks_audit_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    engine, _ = make_engine({"base": make_result(Status.OK)}, trace=writer)
    engine.run()
    writer.close()
    events = load_trace(path)
    for event in events:
        if event["event"] == "act" and event["tool"] == "run_profile":
            event["result"]["status"] = "NG"
    errors = audit_trace(events)
    assert any("result_hash" in error for error in errors)
    report = replay_trace(events)
    assert report.deterministic is False


def test_seq_gap_breaks_audit() -> None:
    writer = TraceWriter(None)
    engine, _ = make_engine({"base": make_result(Status.OK)}, trace=writer)
    events = list(engine.run().events)
    del events[2]
    errors = audit_trace(events)
    assert any("seq gap" in error for error in errors)


def test_inconsistent_final_status_breaks_audit() -> None:
    writer = TraceWriter(None)
    engine, _ = make_engine({"base": make_result(Status.OK)}, trace=writer)
    events = list(engine.run().events)
    events[-1]["final_status"] = "REVIEW"
    errors = audit_trace(events)
    assert any("final_status" in error for error in errors)


def test_audit_empty_trace() -> None:
    assert audit_trace([]) == ["empty trace"]


def test_replay_missing_start_is_not_deterministic() -> None:
    report = replay_trace([])
    assert report.deterministic is False
    assert report.audit_valid is False


def test_audit_and_replay_trace_malformed_events_and_fixed_roi_clamp(tmp_path: Path) -> None:
    """손상 이벤트는 미포착 예외 대신 오류 보고/비결정론 판정으로 수렴한다."""

    from dataclasses import replace as dc_replace

    import numpy as np

    from surface_sentinel.config import default_config
    from surface_sentinel.pipeline.stages import prepare_roi

    writer = TraceWriter(None)
    engine, _ = make_engine({"base": make_result(Status.OK)}, trace=writer)
    events = list(engine.run().events)

    # 1) 비정형 budget(int 변환 불가)도 예외 없이 오류 목록으로 수렴.
    malformed_budget = [dict(event) for event in events]
    malformed_budget[1]["budget"] = {"actions_used": "abc", "profiles_used": None}
    errors = audit_trace(malformed_budget)
    assert any("budget" in error for error in errors)

    # 2) 비-dict args/result: 해시 계산 전 형태 검사 + replay 비결정론 보고.
    malformed_args = [dict(event) for event in events]
    for event in malformed_args:
        if event.get("event") == "act":
            event["args"] = ["not", "a", "dict"]
            event["result"] = "not-a-dict"
    errors = audit_trace(malformed_args)
    assert any("args" in error or "result" in error for error in errors)
    report = replay_trace(malformed_args)
    assert report.deterministic is False

    # 3) 컨텍스트 매니저 + 고정 줄바꿈(Windows CRLF 혼입 방지).
    trace_path = tmp_path / "ctx_trace.jsonl"
    with TraceWriter(trace_path) as ctx_writer:
        ctx_writer.append({"event": "episode_start"})
    assert b"\r\n" not in trace_path.read_bytes()

    # 4) fixed ROI: 음수 너비/높이 대신 (0, 0, 0, 0) 클램프.
    config = default_config()
    fixed_roi = dc_replace(config.roi, mode="fixed", fixed_margin_px=50)
    config = dc_replace(config, roi=fixed_roi)
    roi = prepare_roi(np.zeros((10, 10), np.uint8), config)
    assert roi.bbox == (0, 0, 0, 0)
    assert roi.uncertain is True
