"""평가 경로와 API·워커 서빙 경로가 같은 ONNX·정책에서 동일 판정을 내는지 400장 실측 스크립트.

평가 경로(``evaluate_rows`` + ``YoloOnnxEngine``)와 서빙 경로
(``inspect_with_yolo`` + ``YoloEngineRegistry``)를 **같은 이미지·같은 모델 바이트**로
돌려 이미지별 status·score 와 검사 신원(fingerprint) 일치를 실측한다.

규칙:
* 봉인 분할은 열지 않는다(val 은 tune CSV 내부 group-aware 부분집합).
* 판정 로직은 바꾸지 않는다(관찰만). 결합(combined)은 이번 실측 대상이 아니다.
* code_sha/code_dirty 는 진입점 1회 계산(git_code_identity)으로 양쪽에 주입.
* 서빙은 워커 관례대로 split="tune" 메타데이터를 쓴다(판정은 split·label 무관).

산출물:
* ``docs/evidence/P0_parity_<날짜>.json`` — 구조화 증거(git 추적)

실행(PowerShell): ``uv run python scripts/p0_parity_measure.py``
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from surface_sentinel.config import load_config
from surface_sentinel.evaluation.recalibration import git_code_identity
from surface_sentinel.evaluation.runner import load_split_rows
from surface_sentinel.records.fingerprint import file_sha256, inspection_fingerprint, policy_hash
from surface_sentinel.yolo.evaluate import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RAW_ROOT,
    evaluate_rows,
    resolve_split_rows,
)
from surface_sentinel.yolo.inference import YoloOnnxEngine
from surface_sentinel.yolo.inspect import inspect_with_yolo, inspection_identity, resolve_onnx_path
from surface_sentinel.yolo.runtime import YoloEngineRegistry

OUT_DIR = Path("docs/evidence")
SCORE_TOLERANCE = 1e-9


def _status_counts(statuses: list[str]) -> dict[str, int]:
    return {s: statuses.count(s) for s in ("OK", "NG", "REVIEW", "ERROR")}


def measure_task(task: str, rows: list[Any], config: Any, code_sha: str, code_dirty: bool) -> dict:
    """한 task(det/seg)의 평가·서빙 짝 실측."""

    name = f"yolo-{task}"
    onnx = resolve_onnx_path(name)
    model_sha = file_sha256(onnx)
    spec = inspection_identity(name, config, code_sha=code_sha, code_dirty=code_dirty)
    raw_root = Path(DEFAULT_RAW_ROOT)

    started = time.perf_counter()
    report = evaluate_rows(
        rows,
        engine=YoloOnnxEngine(onnx, task=task),
        task=task,
        split="val",
        onnx=onnx,
        config=config,
        model_sha256=model_sha,
        code_sha=code_sha,
        code_dirty=code_dirty,
    )
    eval_elapsed = time.perf_counter() - started

    registry = YoloEngineRegistry()
    serving_status: list[str] = []
    serving_score: list[float] = []
    infer_ms: list[float] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        outcome = inspect_with_yolo(
            raw_root / row.image_path,
            config,
            engine=name,
            split="tune",
            label=row.label,
            registry=registry,
        )
        serving_status.append(outcome.result.status.value)
        serving_score.append(float(outcome.result.score))
        infer_ms.append(float(outcome.result.timing_ms.get("infer_ms", 0.0)))
        if index % 100 == 0:
            print(f"  {name} 서빙 {index}/{len(rows)}", file=sys.stderr, flush=True)
    serving_elapsed = time.perf_counter() - started

    eval_status = [sample.status.value for sample in report.samples]
    eval_score = [float(sample.score) for sample in report.samples]
    status_mismatch = [
        row.sample_id
        for row, left, right in zip(rows, eval_status, serving_status, strict=True)
        if left != right
    ]
    deltas = [abs(left - right) for left, right in zip(eval_score, serving_score, strict=True)]
    score_mismatch = [
        row.sample_id
        for row, delta in zip(rows, deltas, strict=True)
        if delta > SCORE_TOLERANCE
    ]

    # 평가 보고서 필드로 재구성한 fingerprint == 서빙 EngineSpec fingerprint.
    eval_fingerprint = inspection_fingerprint(
        engine=report.engine,
        task=report.task,
        model_sha256=report.model_sha256,
        config_hash=report.config_hash,
        yolo_policy_hash=policy_hash(report.policy["ng_conf"], report.policy["review_lo"]),
        combine_policy_hash="",
        code_sha=report.code_sha,
        code_dirty=report.code_dirty,
    )
    identity_fields = {
        "model_sha256_match": report.model_sha256 == spec.model_sha256 == model_sha,
        "config_hash_match": report.config_hash == spec.config_hash,
        "policy_match": dict(report.policy) == {"ng_conf": 0.50, "review_lo": 0.25},
        "code_sha_match": report.code_sha == spec.code_sha,
        "code_dirty_match": report.code_dirty == spec.code_dirty,
        "fingerprint_match": eval_fingerprint == spec.fingerprint,
    }
    return {
        "engine": name,
        "onnx": str(onnx),
        "model_bytes": onnx.stat().st_size,
        "model_sha256": model_sha,
        "serving_fingerprint": spec.fingerprint,
        "eval_fingerprint": eval_fingerprint,
        "identity_fields": identity_fields,
        "identity_match": all(identity_fields.values()),
        "samples": len(rows),
        "status_mismatch": status_mismatch,
        "score_max_abs_delta": max(deltas) if deltas else 0.0,
        "score_mismatch": score_mismatch,
        "status_counts_eval": _status_counts(eval_status),
        "status_counts_serving": _status_counts(serving_status),
        "eval_elapsed_sec": round(eval_elapsed, 2),
        "serving_elapsed_sec": round(serving_elapsed, 2),
        "serving_cold_starts": registry.cold_starts,
        "serving_sessions": registry.size,
        "serving_mean_infer_ms": round(sum(infer_ms) / len(infer_ms), 2) if infer_ms else 0.0,
    }


def main() -> int:
    started = time.perf_counter()
    import onnxruntime  # noqa: PLC0415 - 실측 런타임 버전 기록용

    config = load_config(DEFAULT_CONFIG_PATH)
    rows = resolve_split_rows(load_split_rows("tune"), "val")
    code_sha, code_dirty = git_code_identity()  # 진입점 1회 계산

    print(f"표본: val {len(rows)}장 / code_sha={code_sha[:12]} dirty={code_dirty}")
    tasks = [measure_task(task, rows, config, code_sha, code_dirty) for task in ("det", "seg")]

    ok = all(
        task["identity_match"] and not task["status_mismatch"] and not task["score_mismatch"]
        for task in tasks
    )
    elapsed = time.perf_counter() - started
    payload = {
        "generated": date.today().isoformat(),
        "purpose": "P0-3 평가·서빙 짝 실측(같은 val 이미지·같은 ONNX 바이트·같은 정책)",
        "split": "val",
        "samples": len(rows),
        "defect": sum(1 for row in rows if row.label == "defect"),
        "good": sum(1 for row in rows if row.label == "good"),
        "groups": len({row.group_id for row in rows}),
        "config_hash": config.config_hash,
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "onnxruntime_version": str(onnxruntime.__version__),
        "score_tolerance": SCORE_TOLERANCE,
        "serving_split_metadata": "tune(워커 관례 — 판정은 split·label 무관)",
        "tasks": tasks,
        "verdict": "짝 일치 확인" if ok else "불일치 발견 — 원인 조사 필요",
        "elapsed_sec": round(elapsed, 2),
    }
    out_path = OUT_DIR / f"P0_parity_{payload['generated']}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for task in tasks:
        print(
            f"{task['engine']}: status 불일치 {len(task['status_mismatch'])}건 / "
            f"score 최대차 {task['score_max_abs_delta']:.2e} / "
            f"신원 일치 {task['identity_match']} / "
            f"평가 {task['eval_elapsed_sec']}s · 서빙 {task['serving_elapsed_sec']}s"
        )
    print(f"판정: {payload['verdict']}")
    print(f"증거 JSON: {out_path}")
    print(f"총 소요: {elapsed:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
