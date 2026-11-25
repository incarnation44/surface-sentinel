"""CLI `gate`, `repro check`, `agent` 분석 서브커맨드 테스트.

저장소 무결성 게이트, 재현성 비교, 감사·드리프트·트리아지·반사실 해석 CLI 동작을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_fakes import make_component, make_engine, make_result
from surface_sentinel.agent.trace import TraceWriter
from surface_sentinel.cli import EXIT_GATE, EXIT_OK, EXIT_USAGE, main
from surface_sentinel.config import default_config
from surface_sentinel.contracts import Status

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_sample_run(
    run_dir: Path,
    *,
    run_id: str,
    records: list[dict],
    config_hash: str,
) -> None:
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    for rec in records:
        rel = rec.get("overlay_path")
        if rel:
            (run_dir / rel).write_bytes(b"png")
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    ok_n = sum(1 for r in records if r["status"] == "OK")
    ng_n = sum(1 for r in records if r["status"] == "NG")
    rev_n = sum(1 for r in records if r["status"] == "REVIEW")
    err_n = sum(1 for r in records if r["status"] == "ERROR")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "COMPLETE",
                "config_hash": config_hash,
                "counts": {
                    "samples": len(records),
                    "ok": ok_n,
                    "ng": ng_n,
                    "review": rev_n,
                    "error": err_n,
                },
            }
        ),
        encoding="utf-8",
    )


def test_cli_gate_passes_on_repo_and_fails_on_tampered_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_json = tmp_path / "gate.json"
    rc = main(["gate", "--repo-root", str(REPO_ROOT), "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "[PASS]" in out
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["overall_verdict"] == "PASS"

    # 빈 디렉터리는 configs/base.toml 미존재로 REJECT -> 종료 코드 7
    rc_fail = main(["gate", "--repo-root", str(tmp_path)])
    assert rc_fail == EXIT_GATE


def test_cli_repro_check_usage_trace_and_run_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["repro", "check"]) == EXIT_USAGE
    assert main(["repro", "check", "--run-a", str(tmp_path)]) == EXIT_USAGE

    trace_path = tmp_path / "agent_trace.jsonl"
    writer = TraceWriter(trace_path)
    engine, _ = make_engine({"base": make_result(Status.OK)}, trace=writer)
    engine.run()
    writer.close()

    assert main(["repro", "check", "--trace", str(trace_path)]) == EXIT_OK
    assert "트레이스 재현성 확인 완료" in capsys.readouterr().out

    # 트레이스 변조 시 EXIT_GATE (7)
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    tampered_ev = json.loads(lines[-1])
    tampered_ev["final_status"] = "NG"
    trace_path.write_text(
        "\n".join(lines[:-1] + [json.dumps(tampered_ev)]) + "\n", encoding="utf-8"
    )
    assert main(["repro", "check", "--trace", str(trace_path)]) == EXIT_GATE

    # 두 run 디렉터리 비교 (일치 -> 0, 불일치 -> 7)
    cfg_hash = default_config().config_hash
    rec = {
        "sample_id": "s1",
        "status": "OK",
        "score": 0.123456,
        "config_hash": cfg_hash,
        "overlay_path": "overlays/s1.png",
    }
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_sample_run(run_a, run_id="r_a", records=[rec], config_hash=cfg_hash)
    _write_sample_run(run_b, run_id="r_b", records=[rec], config_hash=cfg_hash)
    assert main(["repro", "check", "--run-a", str(run_a), "--run-b", str(run_b)]) == EXIT_OK

    rec_bad = dict(rec, score=0.456789)
    _write_sample_run(run_b, run_id="r_b", records=[rec_bad], config_hash=cfg_hash)
    rc_mismatch = main(["repro", "check", "--run-a", str(run_a), "--run-b", str(run_b)])
    assert rc_mismatch == EXIT_GATE
    assert "score 불일치" in capsys.readouterr().err


def test_cli_agent_audit_drift_and_triage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_hash = default_config().config_hash
    run_dir = tmp_path / "run_ok"
    _write_sample_run(
        run_dir,
        run_id="r1",
        records=[
            {
                "sample_id": "s1",
                "status": "OK",
                "score": 0.05,
                "config_hash": cfg_hash,
                "overlay_path": "overlays/s1.png",
            }
        ],
        config_hash=cfg_hash,
    )
    assert main(["agent", "audit", "--run-dir", str(run_dir)]) == EXIT_OK

    # agent drift
    base_jsonl = tmp_path / "base.jsonl"
    target_jsonl = tmp_path / "target.jsonl"
    base_rows = [
        {"status": "OK", "score": 0.02, "quality": {"mean_gray": 40.0, "blur_variance": 200.0}}
        for _ in range(25)
    ]
    target_rows = [
        {"status": "NG", "score": 0.75, "quality": {"mean_gray": 150.0, "blur_variance": 200.0}}
        for _ in range(25)
    ]
    base_jsonl.write_text("\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
    target_jsonl.write_text("\n".join(json.dumps(r) for r in target_rows) + "\n", encoding="utf-8")
    drift_out = tmp_path / "drift.json"
    assert (
        main(
            [
                "agent",
                "drift",
                "--baseline",
                str(base_jsonl),
                "--target",
                str(target_jsonl),
                "--json",
                str(drift_out),
            ]
        )
        == EXIT_OK
    )
    drift_payload = json.loads(drift_out.read_text(encoding="utf-8"))
    assert drift_payload["status"] == "DRIFT_ALERT"

    # agent triage
    triage_jsonl = tmp_path / "results.jsonl"
    comp_hi = make_component(0, (40, 100, 14, 14), area_px=120, evidence=0.45).to_dict()
    comp_hi["max_contrast_gray"] = 68.0
    comp_hi["mean_contrast_gray"] = 30.0
    comp_hi["integrated_contrast"] = 3600.0
    comp_lo = make_component(0, (1, 10, 8, 8), area_px=42, evidence=0.25).to_dict()
    comp_lo["touches_roi_border"] = True
    comp_lo["border_distance_px"] = 1.0
    triage_rows = [
        {"sample_id": "weak_border", "status": "REVIEW", "score": 0.25, "components": [comp_lo]},
        {"sample_id": "strong_cand", "status": "REVIEW", "score": 0.45, "components": [comp_hi]},
        {"sample_id": "ok_skip", "status": "OK", "score": 0.0, "components": []},
    ]
    triage_jsonl.write_text(
        "\n".join(json.dumps(r) for r in triage_rows) + "\n", encoding="utf-8"
    )
    triage_out = tmp_path / "triage.json"
    assert (
        main(
            [
                "agent",
                "triage",
                "--results",
                str(triage_jsonl),
                "--top-k",
                "5",
                "--json",
                str(triage_out),
            ]
        )
        == EXIT_OK
    )
    triage_payload = json.loads(triage_out.read_text(encoding="utf-8"))
    assert triage_payload["triaged_count"] == 2
    assert [item["sample_id"] for item in triage_payload["items"]] == [
        "strong_cand",
        "weak_border",
    ]
    _ = capsys.readouterr()


def test_cli_agent_explain_and_calibrate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from surface_sentinel.cli import EXIT_INPUT
    from surface_sentinel.review.store import ReviewStore

    cfg_hash = default_config().config_hash
    run_dir = tmp_path / "run_wave3b"
    comp_review = make_component(0, (40, 100, 10, 10), area_px=96, evidence=0.31).to_dict()
    comp_review["mean_contrast_gray"] = 17.8
    records = [
        {
            "sample_id": "s_rev_1",
            "status": "REVIEW",
            "score": 0.31,
            "reasons": ["EVIDENCE_IN_REVIEW_BAND"],
            "components": [comp_review],
            "config_hash": cfg_hash,
            "overlay_path": "overlays/s_rev_1.png",
        },
        {
            "sample_id": "s_ok_1",
            "status": "OK",
            "score": 0.0,
            "reasons": ["NO_DEFECT_EVIDENCE"],
            "components": [],
            "config_hash": cfg_hash,
            "overlay_path": "overlays/s_ok_1.png",
        },
    ]
    _write_sample_run(run_dir, run_id="r_w3b", records=records, config_hash=cfg_hash)

    # agent explain
    explain_out = tmp_path / "explain.json"
    rc = main([
        "agent", "explain",
        "--results", str(run_dir / "results.jsonl"),
        "--sample-id", "s_rev_1",
        "--json", str(explain_out),
    ])
    assert rc == EXIT_OK
    out_txt = capsys.readouterr().out
    assert "반사실적 경계 해석" in out_txt
    exp_payload = json.loads(explain_out.read_text(encoding="utf-8"))
    assert exp_payload["sample_id"] == "s_rev_1"
    assert exp_payload["components"][0]["target_contrast_for_ng"] is None
    assert exp_payload["components"][0]["target_area_for_ng"] is not None

    # 없는 sample_id -> EXIT_INPUT
    assert (
        main(
            [
                "agent",
                "explain",
                "--results",
                str(run_dir / "results.jsonl"),
                "--sample-id",
                "missing",
            ]
        )
        == EXIT_INPUT
    )

    # agent calibrate (결정 전 & 결정 후)
    ReviewStore(run_dir).append(
        sample_id="s_rev_1",
        machine_status="REVIEW",
        final_status="NG",
        reason_code="CONFIRMED_DEFECT",
        reviewer="qa",
        config_hash=cfg_hash,
        overlay_sha256="a" * 64,
    )
    cal_out = tmp_path / "calibrate.json"
    rc_cal = main(["agent", "calibrate", "--run-dir", str(run_dir), "--json", str(cal_out)])
    assert rc_cal == EXIT_OK
    cal_txt = capsys.readouterr().out
    assert "Phase 4 캘리브레이션" in cal_txt
    cal_payload = json.loads(cal_out.read_text(encoding="utf-8"))
    assert cal_payload["total_review_samples"] == 1
    assert cal_payload["decided_count"] == 1
    assert cal_payload["confirmed_ng_ratio"]["numerator"] == 1
    _ = capsys.readouterr()
