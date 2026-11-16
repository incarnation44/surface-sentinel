"""Phase 1 교차 증거 실제 ONNX 스모크 (val 2장: defect 1 + good 1).

결합 검사(실제 ONNX) → 엔진 관찰 2개 → 교차 증거 에피소드 → 판정 불변 확인.
봉인 분할은 열지 않는다. val 은 tune CSV 내부 부분집합이므로 서빙 split 은
워커 관례대로 ``tune`` 을 쓴다(판정은 split·label 무관).

산출물(gitignore): ``runs/evidence-smoke/`` 아래
* ``combined_<sample>/`` — 결합 검사 오버레이
* ``evidence/<run_id>/`` — dual overlay·결과 JSON·트레이스
* ``smoke_summary.json`` — 요약(판정 불변·플래그·해시)

실행(PowerShell): ``uv run python scripts/phase1_evidence_smoke.py``
전제: ``data/derived/yolo/models/{det,seg}/weights/best.onnx`` 존재,
``SURFACE_SENTINEL_YOLO_COMBINED_TASK`` 는 스크립트가 ``det`` 로 설정한다.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("SURFACE_SENTINEL_YOLO_COMBINED_TASK", "det")

from surface_sentinel.agent.config import load_agent_config
from surface_sentinel.agent.run import load_profiles, run_evidence_episode
from surface_sentinel.config import load_config
from surface_sentinel.evaluation.runner import load_split_rows
from surface_sentinel.yolo.evaluate import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RAW_ROOT,
    resolve_split_rows,
)
from surface_sentinel.yolo.inspect import inspect_with_yolo
from surface_sentinel.yolo.runtime import YoloEngineRegistry

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "runs" / "evidence-smoke"


def main() -> int:
    import onnxruntime  # noqa: PLC0415 - 실측 런타임 버전 기록용

    config = load_config(DEFAULT_CONFIG_PATH)
    rows = resolve_split_rows(load_split_rows("tune"), "val")
    defect = next(row for row in rows if row.label == "defect")
    good = next(row for row in rows if row.label == "good")
    agent_config = load_agent_config(REPO / "configs" / "agent.toml")
    profiles = load_profiles(DEFAULT_CONFIG_PATH, REPO / "configs" / "profiles")
    registry = YoloEngineRegistry()

    samples = []
    for row in (defect, good):
        image = Path(DEFAULT_RAW_ROOT) / row.image_path
        outcome = inspect_with_yolo(
            image,
            config,
            engine="combined",
            output_dir=OUT_ROOT / f"combined_{row.sample_id}",
            split="tune",
            label=row.label,
            registry=registry,
        )
        observations = outcome.engine_observations
        if len(observations) != 2:
            print(f"관찰 2개 계약 위반: {len(observations)}", file=sys.stderr)
            return 1
        episode_out = run_evidence_episode(
            image,
            combined_result=outcome.result,
            observations=observations,
            agent_config=agent_config,
            profiles=profiles,
            output_dir=OUT_ROOT / "evidence",
            label=row.label,
            split="tune",
        )
        verdict_unchanged = episode_out.episode.final_status is outcome.result.status
        if not verdict_unchanged:
            print("판정 불변 위반", file=sys.stderr)
            return 1
        assessment = episode_out.assessment
        samples.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "combined_status": outcome.result.status.value,
                "episode_final_status": episode_out.episode.final_status.value,
                "verdict_unchanged": verdict_unchanged,
                "engines": [obs.engine for obs in observations],
                "flags": [flag.value for flag in assessment.flags],
                "recommended_action": assessment.recommended_action,
                "matches": len(assessment.matches),
                "assessment_hash": assessment.assessment_hash,
                "overlay_bytes": episode_out.overlay_path.stat().st_size,
                "run_dir": str(episode_out.run_dir),
            }
        )

    summary = {
        "generated": date.today().isoformat(),
        "purpose": "Phase 1 교차 증거 실제 ONNX 스모크(판정 불변·별도 증거 기록)",
        "split": "val(tune 부분집합) / 서빙 split=tune",
        "onnxruntime_version": str(onnxruntime.__version__),
        "config_hash": config.config_hash,
        "combined_task": os.environ["SURFACE_SENTINEL_YOLO_COMBINED_TASK"],
        "samples": samples,
        "verdict": "판정 불변 확인" if all(s["verdict_unchanged"] for s in samples) else "위반",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
