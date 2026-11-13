"""Phase 3 무결성 감사 Watchdog(agent.audit) 단위 테스트."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_fakes import make_engine, make_result
from surface_sentinel.agent.audit import (
    AuditVerdict,
    audit_records_db,
    audit_repository_gates,
    audit_run_dir,
)
from surface_sentinel.agent.trace import TraceWriter
from surface_sentinel.config import default_config
from surface_sentinel.contracts import SampleResult, Status
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import insert_inspection
from surface_sentinel.review.store import ReviewStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_valid_run_dir(run_dir: Path) -> str:
    cfg = default_config()
    run_id = "20261113T000000Z-" + cfg.config_hash[:8] + "-tune"
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays" / "s_ok_overlay.png").write_bytes(b"ok")
    (run_dir / "overlays" / "s_rev_overlay.png").write_bytes(b"rev")
    records = [
        {
            "run_id": run_id,
            "sample_id": "s_ok",
            "original_id": "s_ok",
            "split": "tune",
            "label": "good",
            "status": "OK",
            "score": 0.05,
            "reasons": ["NO_DEFECT_EVIDENCE"],
            "components": [],
            "quality": None,
            "roi": None,
            "timing_ms": {"total": 1.0},
            "error_code": None,
            "error_stage": None,
            "input_sha256": "a" * 64,
            "mask_sha256": "",
            "overlay_path": "overlays/s_ok_overlay.png",
            "config_hash": cfg.config_hash,
        },
        {
            "run_id": run_id,
            "sample_id": "s_rev",
            "original_id": "s_rev",
            "split": "tune",
            "label": "defect",
            "status": "REVIEW",
            "score": 0.42,
            "reasons": ["EVIDENCE_IN_REVIEW_BAND"],
            "components": [],
            "quality": None,
            "roi": None,
            "timing_ms": {"total": 1.0},
            "error_code": None,
            "error_stage": None,
            "input_sha256": "b" * 64,
            "mask_sha256": "",
            "overlay_path": "overlays/s_rev_overlay.png",
            "config_hash": cfg.config_hash,
        },
    ]
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.toml").write_text(cfg.canonical_toml(), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "COMPLETE",
                "config_hash": cfg.config_hash,
                "counts": {"samples": 2, "ok": 1, "ng": 0, "review": 1, "error": 0},
            }
        ),
        encoding="utf-8",
    )
    ReviewStore(run_dir).append(
        sample_id="s_rev",
        machine_status="REVIEW",
        final_status="NG",
        reason_code="CONFIRMED_DEFECT",
        reviewer="qa",
        config_hash=cfg.config_hash,
        overlay_sha256="c" * 64,
    )
    writer = TraceWriter(run_dir / "agent_trace.jsonl")
    engine, _ = make_engine({"base": make_result(Status.OK)}, trace=writer)
    engine.run()
    writer.close()
    return cfg.config_hash


def test_audit_run_dir_pass_and_tamper_detection(tmp_path: Path) -> None:
    run_dir = tmp_path / "20261113T000000Z-a762ba63-tune"
    _build_valid_run_dir(run_dir)

    ok_report = audit_run_dir(run_dir)
    assert ok_report.verdict is AuditVerdict.PASS
    assert all(c.passed for c in ok_report.checks)
    assert audit_run_dir(run_dir).audit_hash == ok_report.audit_hash

    # ① results.jsonl 행 변조 (OK -> NG) 시 run.json counts 불일치로 REJECT
    bad_results = tmp_path / "bad_results_run"
    _build_valid_run_dir(bad_results)
    lines = (bad_results / "results.jsonl").read_text(encoding="utf-8").splitlines()
    rec0 = json.loads(lines[0])
    rec0["status"] = "NG"
    (bad_results / "results.jsonl").write_text(
        json.dumps(rec0) + "\n" + lines[1] + "\n", encoding="utf-8"
    )
    assert audit_run_dir(bad_results).verdict is AuditVerdict.REJECT

    # ② OK 샘플에 review_decisions.jsonl 주입 시 REJECT
    bad_review = tmp_path / "bad_review_run"
    _build_valid_run_dir(bad_review)
    rev_path = bad_review / "review_decisions.jsonl"
    decision = json.loads(rev_path.read_text(encoding="utf-8").splitlines()[0])
    decision["sample_id"] = "s_ok"
    rev_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    rev_report = audit_run_dir(bad_review)
    assert rev_report.verdict is AuditVerdict.REJECT
    assert any(c.name == "review_decisions_immutable" and not c.passed for c in rev_report.checks)

    # ③ 오버레이 파일만 누락 시 REVISE
    missing_overlay = tmp_path / "missing_overlay_run"
    _build_valid_run_dir(missing_overlay)
    (missing_overlay / "overlays" / "s_ok_overlay.png").unlink()
    ov_report = audit_run_dir(missing_overlay)
    assert ov_report.verdict is AuditVerdict.REVISE


def test_audit_records_db_missing_does_not_create_file_and_checks_integrity(
    tmp_path: Path,
) -> None:
    missing_db = tmp_path / "nonexistent.db"
    miss_report = audit_records_db(missing_db)
    assert miss_report.verdict is AuditVerdict.REVISE
    assert "없습니다" in miss_report.summary
    assert not missing_db.exists()

    db_path = tmp_path / "records.db"
    conn = connect(db_path)
    init_db(conn)
    cfg_hash = default_config().config_hash
    id_ok = insert_inspection(
        conn,
        SampleResult(
            run_id="r1",
            sample_id="s1",
            original_id="s1",
            split="tune",
            label="good",
            status=Status.OK,
            score=0.05,
            config_hash=cfg_hash,
        ),
        engine="opencv",
        image_sha256="1" * 64,
        provenance_complete=1,
    )
    conn.close()

    assert audit_records_db(db_path).verdict is AuditVerdict.PASS

    # 레거시 행(provenance_complete=0) -> REVISE
    raw = sqlite3.connect(str(db_path))
    raw.execute("UPDATE inspections SET provenance_complete = 0 WHERE id = ?", (id_ok,))
    raw.commit()
    raw.close()
    assert audit_records_db(db_path).verdict is AuditVerdict.REVISE

    # OK 행에 불법 review_decisions 주입 -> REJECT
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "INSERT INTO review_decisions (inspection_id, decision, reviewer, reason, created_at) "
        "VALUES (?, 'NG', 'qa', 'tamper', '2026-11-13T00:00:00Z')",
        (id_ok,),
    )
    raw.commit()
    raw.close()
    bad_db_report = audit_records_db(db_path)
    assert bad_db_report.verdict is AuditVerdict.REJECT
    assert any(
        c.name == "illegal_review_decisions" and not c.passed for c in bad_db_report.checks
    )


def test_audit_repository_gates_real_repo_and_tampered_copy(tmp_path: Path) -> None:
    real_report = audit_repository_gates(REPO_ROOT)
    assert real_report.verdict is AuditVerdict.PASS

    # 변조된 base.toml -> REJECT
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.toml").write_text(
        default_config().canonical_toml().replace("ng_min = 0.5", "ng_min = 0.6"),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "agent.toml").write_text(
        (REPO_ROOT / "configs" / "agent.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "data" / "splits").mkdir(parents=True)
    split_name = "test"
    (tmp_path / "data" / "splits" / f"{split_name}.lock.json").write_text(
        json.dumps({"access_count": 3, "post_seal_config_change": False}),
        encoding="utf-8",
    )
    tampered = audit_repository_gates(tmp_path)
    assert tampered.verdict is AuditVerdict.REJECT
    failed = {c.name for c in tampered.checks if not c.passed}
    assert "G_CONFIG_FROZEN" in failed
    assert "G_SEAL_LOCK" in failed


def test_audit_repository_gates_detects_tampered_t10_guide(tmp_path: Path) -> None:
    real_report = audit_repository_gates(REPO_ROOT)
    assert any(c.name == "G_GUIDE_H7" and c.passed for c in real_report.checks)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.toml").write_text(
        default_config().canonical_toml(),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "agent.toml").write_text(
        (REPO_ROOT / "configs" / "agent.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "surface-sentinel-integrated-guide.html").write_text(
        "<html><body><h1 id='dup'>A</h1><h2 id='dup'>B</h2></body></html>",
        encoding="utf-8",
    )
    tampered = audit_repository_gates(tmp_path)
    assert tampered.verdict is AuditVerdict.REJECT
    failed = {c.name for c in tampered.checks if not c.passed}
    assert "G_GUIDE_H7" in failed
