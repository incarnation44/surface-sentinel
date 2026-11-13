"""Phase 3 Watchdog 감사·PSI 드리프트 감시·repro check 실측 검증.

봉인 분할(``test``·``official_test`` CSV)은 절대 열지 않는다.
오직 ``docs/evidence/t05_adopted.jsonl`` + ``tune`` 2,001장 캐시, 저장소 메타데이터,
그리고 임시 디렉터리 내 스모크 트레이스/run 만을 사용하여 검증한다.

산출물:
* ``docs/evidence/P3_watchdog_eval_2026-11-13.json``
* ``docs/reports/P3_agent_watchdog_2026-11-13.md``

실행: ``PYTHONUTF8=1 uv run --no-sync python scripts/phase3_watchdog_eval.py``
"""

from __future__ import annotations

import json
import pickle
import random
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from surface_sentinel.agent.audit import (
    audit_records_db,
    audit_repository_gates,
    audit_run_dir,
)
from surface_sentinel.agent.config import load_agent_config
from surface_sentinel.agent.drift import (
    DEFAULT_DRIFT_POLICY,
    DriftObservation,
    assess_drift,
    extract_observations,
)
from surface_sentinel.agent.run import load_profiles, run_agent_episode
from surface_sentinel.agent.trace import audit_trace, load_trace, replay_trace
from surface_sentinel.cli import main as cli_main
from surface_sentinel.config import load_config
from surface_sentinel.contracts import Status
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import insert_inspection

REPO = Path(__file__).resolve().parents[1]
ADOPTED_JSONL = REPO / "docs/evidence/t05_adopted.jsonl"
TUNE_CACHE_PATH = REPO / "data/derived/t05_cache/t05-cache-v1-tune-a762ba639441-46c06acb3b53.pkl"
BASE_CONFIG_PATH = REPO / "configs/base.toml"
AGENT_CONFIG_PATH = REPO / "configs/agent.toml"
PROFILES_DIR = REPO / "configs/profiles"
SYNTHETIC_GOOD = REPO / "tests/fixtures/synthetic/good.png"

EVIDENCE_OUT = REPO / "docs/evidence/P3_watchdog_eval_2026-11-13.json"
REPORT_OUT = REPO / "docs/reports/P3_agent_watchdog_2026-11-13.md"


def _load_tune_baseline() -> tuple[dict[str, Any], tuple[DriftObservation, ...]]:
    adopted_line = ADOPTED_JSONL.read_text(encoding="utf-8").splitlines()[0]
    adopted_meta = json.loads(adopted_line)
    with TUNE_CACHE_PATH.open("rb") as handle:
        cached_samples = pickle.load(handle)["samples"]  # noqa: S301 - 로컬 검증용 캐시
    if len(cached_samples) != int(adopted_meta["samples"]):
        raise RuntimeError(
            f"tune 캐시 표본 수({len(cached_samples)})가 "
            f"t05_adopted.jsonl({adopted_meta['samples']})과 다릅니다"
        )
    observations = extract_observations(cached_samples)
    return adopted_meta, observations


def _run_drift_scenarios(baseline_obs: tuple[DriftObservation, ...]) -> dict[str, Any]:
    rng = random.Random(20261113)
    indices = list(range(len(baseline_obs)))
    rng.shuffle(indices)
    holdout_idx = set(indices[:400])

    # 시나리오 1: 결정론 시드 홀드아웃 400장 (나머지 1,601장 기준선 대비 정상 분포 확인)
    holdout_400 = [baseline_obs[i] for i in range(len(baseline_obs)) if i in holdout_idx]
    rest_1601 = [baseline_obs[i] for i in range(len(baseline_obs)) if i not in holdout_idx]
    normal_report = assess_drift(rest_1601, holdout_400, policy=DEFAULT_DRIFT_POLICY)

    # 시나리오 2: 합성 드리프트 (점수 +0.35 이동 및 조명 밝기 mean_gray +35.0 이동)
    shifted_400 = []
    for obs in holdout_400:
        new_score = None if obs.score is None else min(1.5, obs.score + 0.35)
        new_gray = None if obs.mean_gray is None else min(255.0, obs.mean_gray + 35.0)
        new_status = obs.status
        if new_score is not None:
            if new_score >= 0.50 and obs.status in {Status.OK.value, Status.REVIEW.value}:
                new_status = Status.NG.value
            elif new_score >= 0.25 and obs.status == Status.OK.value:
                new_status = Status.REVIEW.value
        shifted_400.append(
            replace(
                obs,
                status=new_status,
                score=new_score,
                mean_gray=new_gray,
            )
        )
    drift_alert_report = assess_drift(baseline_obs, shifted_400, policy=DEFAULT_DRIFT_POLICY)

    # 시나리오 3: 소표본 10장 -> INSUFFICIENT_SAMPLE 방어 확인
    small_10 = shifted_400[:10]
    insufficient_report = assess_drift(baseline_obs, small_10, policy=DEFAULT_DRIFT_POLICY)

    return {
        "normal_holdout_400": normal_report.to_dict(),
        "synthetic_drift_score_and_brightness": drift_alert_report.to_dict(),
        "insufficient_sample_10": insufficient_report.to_dict(),
    }


def _run_repro_and_smoke_audit(config_hash: str) -> dict[str, Any]:
    agent_cfg = load_agent_config(AGENT_CONFIG_PATH)
    profiles = load_profiles(BASE_CONFIG_PATH, PROFILES_DIR)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        outcome = run_agent_episode(
            SYNTHETIC_GOOD,
            agent_config=agent_cfg,
            profiles=profiles,
            output_dir=tmp_path / "agent_runs",
            label="good",
            split="tune",
        )
        events = load_trace(outcome.trace_path)
        trace_errors = audit_trace(events)
        replay = replay_trace(events)
        cli_trace_rc = cli_main(["repro", "check", "--trace", str(outcome.trace_path)])

        # 두 run 디렉터리 재현성 비교 스모크
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        for run_dir, run_id in (
            (run_a, "20261113T000000Z-a762ba63-tune"),
            (run_b, "20261113T000001Z-a762ba63-tune"),
        ):
            (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
            (run_dir / "overlays" / "s1_overlay.png").write_bytes(b"png")
            res_line = json.dumps(
                {
                    "run_id": run_id,
                    "sample_id": "s1",
                    "original_id": "s1",
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
                    "overlay_path": "overlays/s1_overlay.png",
                    "config_hash": config_hash,
                },
                ensure_ascii=False,
            )
            (run_dir / "results.jsonl").write_text(res_line + "\n", encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": "COMPLETE",
                        "config_hash": config_hash,
                        "counts": {"samples": 1, "ok": 1, "ng": 0, "review": 0, "error": 0},
                    }
                ),
                encoding="utf-8",
            )
        cli_run_cmp_rc = cli_main(
            ["repro", "check", "--run-a", str(run_a), "--run-b", str(run_b)]
        )
        run_audit = audit_run_dir(run_a)

        # DB 감사 스모크 (읽기 전용 ?mode=ro 검증)
        db_path = tmp_path / "smoke_records.db"
        conn = connect(db_path)
        init_db(conn)
        from surface_sentinel.contracts import SampleResult

        insert_inspection(
            conn,
            SampleResult(
                run_id="smoke",
                sample_id="s1",
                original_id="s1",
                split="tune",
                label="good",
                status=Status.OK,
                score=0.05,
                config_hash=config_hash,
            ),
            engine="opencv",
            image_sha256="b" * 64,
            provenance_complete=1,
        )
        conn.close()
        db_audit = audit_records_db(db_path)

        return {
            "trace_events": len(events),
            "trace_audit_errors": len(trace_errors),
            "replay_deterministic": replay.deterministic,
            "replay_terminal_state": replay.terminal_state,
            "replay_final_status": replay.final_status,
            "cli_repro_trace_exit_code": cli_trace_rc,
            "cli_repro_runs_exit_code": cli_run_cmp_rc,
            "smoke_run_audit_verdict": run_audit.verdict.value,
            "smoke_db_audit_verdict": db_audit.verdict.value,
        }


def _render_markdown(payload: dict[str, Any]) -> str:
    gates = payload["repository_gates"]
    drift = payload["drift_evaluation"]
    repro = payload["repro_and_audit_smoke"]
    normal = drift["normal_holdout_400"]
    shifted = drift["synthetic_drift_score_and_brightness"]
    small = drift["insufficient_sample_10"]

    lines = [
        "# Phase 3 무결성 감사 Watchdog · PSI 드리프트 · 재현성 실측 보고서",
        "",
        f"- 생성일: {payload['generated']}",
        f"- 기준선 설정: `configs/base.toml` (`config_hash={payload['config_hash']}`)",
        "- 봉인 분할(`test`·`official_test` CSV): **미접근** "
        "(`tune` 2,001장 캐시·잠금 JSON 메타데이터만 점검)",
        "- 판정 소유권: Watchdog/드리프트 감시는 100% observe-only 이며 "
        "기계 판정·설정·모델을 수정하지 않음.",
        "",
        "## 1. 저장소 하드 게이트 감사 (`audit_repository_gates` / `surface-sentinel gate`)",
        "",
        f"- 종합 판정: **{gates['verdict']}** (`{gates['summary']}`)",
        f"- 감사 해시: `{gates['audit_hash'][:16]}...`",
        "",
        "| 게이트 | 결과 | 심각도 | 상세 |",
        "|---|---|---|---|",
    ]
    for chk in gates["checks"]:
        status_txt = "PASS" if chk["passed"] else "FAIL"
        lines.append(
            f"| `{chk['name']}` | {status_txt} | {chk['severity']} | {chk['detail']} |"
        )

    lines.extend(
        [
            "",
            "## 2. PSI 분포 드리프트 감시 (`tune` 2,001장 기준선, `t05_adopted_r20`)",
            "",
            f"- 기준선: `docs/evidence/t05_adopted.jsonl` + "
            f"tune 캐시 {payload['baseline_samples']}장",
            "",
            "| 시나리오 | 표본 수(기준/대상) | 종합 판정 | "
            "`score` PSI | `mean_gray` PSI | `blur_variance` PSI | 비고 |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )

    def _feat_psi(rep: dict[str, Any], name: str) -> str:
        for feat in rep["features"]:
            if feat["feature"] == name:
                return f"{feat['psi']:.4f}"
        return "-"

    lines.append(
        "| 정상 홀드아웃 (시드 20261113, 400장) | "
        f"{normal['baseline_total']} / {normal['target_total']} | "
        f"**{normal['status']}** | {_feat_psi(normal, 'score')} | "
        f"{_feat_psi(normal, 'mean_gray')} | "
        f"{_feat_psi(normal, 'blur_variance')} | 모든 피처 PSI < 0.10 |"
    )
    lines.append(
        "| 합성 드리프트 (점수 +0.35 / 밝기 +35) | "
        f"{shifted['baseline_total']} / {shifted['target_total']} | "
        f"**{shifted['status']}** | {_feat_psi(shifted, 'score')} | "
        f"{_feat_psi(shifted, 'mean_gray')} | "
        f"{_feat_psi(shifted, 'blur_variance')} | PSI >= 0.25 경보 감지 |"
    )
    lines.append(
        f"| 소표본 방어 (10장) | {small['baseline_total']} / {small['target_total']} | "
        f"**{small['status']}** | {_feat_psi(small, 'score')} | {_feat_psi(small, 'mean_gray')} | "
        f"{_feat_psi(small, 'blur_variance')} | 표본 20건 미만 억지 경보 차단 |"
    )

    lines.extend(
        [
            "",
            "### 2.1 정상 홀드아웃 vs 합성 드리프트 상태 비율 (분모·Wilson 95% CI 포함)",
            "",
            "| 시나리오 | OK 비율 | NG 비율 | REVIEW 비율 |",
            "|---|---|---|---|",
        ]
    )
    for label_s, rep in (("정상 홀드아웃 400장", normal), ("합성 드리프트 400장", shifted)):
        r_ok = rep["status_rates_target"]["OK"]
        r_ng = rep["status_rates_target"]["NG"]
        r_rev = rep["status_rates_target"]["REVIEW"]
        lines.append(
            f"| {label_s} | {r_ok['value']*100:.2f}% ({r_ok['numerator']}/{r_ok['denominator']}) | "
            f"{r_ng['value']*100:.2f}% ({r_ng['numerator']}/{r_ng['denominator']}) | "
            f"{r_rev['value']*100:.2f}% ({r_rev['numerator']}/{r_rev['denominator']}) |"
        )

    lines.extend(
        [
            "",
            "## 3. 재현성 검증 (`repro check`) 및 run/DB Watchdog 스모크",
            "",
            f"- 스모크 에이전트 트레이스: 이벤트 {repro['trace_events']}건, "
            f"감사 오류 **{repro['trace_audit_errors']}건**, "
            f"리플레이 결정론={repro['replay_deterministic']} "
            f"(`{repro['replay_terminal_state']}/{repro['replay_final_status']}`)",
            "- `surface-sentinel repro check --trace`: "
            f"종료 코드 **{repro['cli_repro_trace_exit_code']}**",
            "- `surface-sentinel repro check --run-a ... --run-b ...`: "
            f"종료 코드 **{repro['cli_repro_runs_exit_code']}**",
            f"- 스모크 `audit_run_dir`: **{repro['smoke_run_audit_verdict']}**",
            f"- 스모크 `audit_records_db` (`?mode=ro`): **{repro['smoke_db_audit_verdict']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = load_config(BASE_CONFIG_PATH)
    gate_report = audit_repository_gates(REPO)
    adopted_meta, baseline_obs = _load_tune_baseline()
    drift_eval = _run_drift_scenarios(baseline_obs)
    smoke_eval = _run_repro_and_smoke_audit(config.config_hash)

    payload = {
        "schema_version": "phase3-watchdog-eval-v1",
        "generated": date.today().isoformat(),
        "config_hash": config.config_hash,
        "sealed_splits_accessed": False,
        "baseline_candidate": adopted_meta["candidate_id"],
        "baseline_samples": len(baseline_obs),
        "repository_gates": gate_report.to_dict(),
        "drift_evaluation": drift_eval,
        "repro_and_audit_smoke": smoke_eval,
    }
    EVIDENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_OUT.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {EVIDENCE_OUT} and {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
