"""OpenCV r20 vs YOLO seg vs 결합 엔진 `tune-val` 400장 짝비교 채택 게이트 분석 스크립트.

결합 감사 JSONL의 ``opencv_status`` 열은 **같은 이미지**에서 돌린 OpenCV r20
판정이다. 이 스크립트는 그 열로 r20 / YOLO seg / 결합 세 엔진을 같은 표본에서
비교하고, 사전 등록한(2026-11-13, 실측 전) G1~G6 채택 기준을 판정한다.

산출물:
* ``docs/evidence/E_val_paired_analysis.json`` — 구조화 증거(git 추적)
* ``docs/reports/E_val_gate_2026-11-13.md`` — 게이트 보고서

실행(PowerShell): ``uv run python scripts/e_val_paired.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from surface_sentinel.evaluation.bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_BOOTSTRAP_SEED,
    group_bootstrap_ci,
    paired_bootstrap_delta,
)
from surface_sentinel.evaluation.metrics import ratio
from surface_sentinel.evaluation.runner import load_split_rows
from surface_sentinel.yolo.evaluate import resolve_split_rows

EVAL_DIR = Path("data/derived/yolo/eval")
COMBINED_JSONL = EVAL_DIR / "yolo-seg_opencv-val.jsonl"
SEG_JSONL = EVAL_DIR / "yolo-seg-val.jsonl"
COMBINED_JSON = EVAL_DIR / "yolo-seg_opencv-val.json"
OUT_JSON = Path("docs/evidence/E_val_paired_analysis.json")
OUT_REPORT = Path("docs/reports/E_val_gate_2026-11-13.md")

#: 사전 등록 채택 기준(실측 전 고정). 목표 값은 동결 상수와 같다.
GATE_TARGETS = {
    "escape": 0.15,
    "escape_ci_high": 0.20,
    "false_ng": 0.20,
    "review": 0.40,
    "recall_010": 0.50,
}


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(row["sample_id"]): row for row in rows}


def _indicator(statuses: list[str], labels: list[str], want: str, subset: str | None) -> list[int]:
    return [
        1 if status == want and (subset is None or label == subset) else 0
        for status, label in zip(statuses, labels, strict=True)
    ]


def _denominator(labels: list[str], subset: str | None) -> list[int]:
    return [1 if subset is None or label == subset else 0 for label in labels]


def main() -> int:
    started = time.perf_counter()

    rows = resolve_split_rows(load_split_rows("tune"), "val")
    combined = _load_jsonl(COMBINED_JSONL)
    seg = _load_jsonl(SEG_JSONL)
    combined_meta = json.loads(COMBINED_JSON.read_text(encoding="utf-8"))

    sample_ids = [row.sample_id for row in rows]
    labels = [row.label for row in rows]
    groups = [row.group_id for row in rows]
    if set(sample_ids) != set(combined) or set(sample_ids) != set(seg):
        raise SystemExit("표본 불일치: val 행과 감사 JSONL의 sample_id 집합이 다르다.")

    # 같은 모델·같은 이미지 추론 동일성 검증(결합 run의 YOLO 부분 = 단독 run).
    inference_mismatch = [
        sid
        for sid in sample_ids
        if combined[sid]["detections"] != seg[sid]["detections"]
        or combined[sid]["matches"] != seg[sid]["matches"]
    ]
    label_mismatch = [
        sid for sid in sample_ids if combined[sid]["label"] != seg[sid]["label"]
    ]

    statuses = {
        "opencv_r20": [str(combined[sid]["opencv_status"]) for sid in sample_ids],
        "yolo_seg": [str(seg[sid]["status"]) for sid in sample_ids],
        "combined": [str(combined[sid]["status"]) for sid in sample_ids],
    }
    if any(s is None or s == "None" for s in statuses["opencv_r20"]):
        raise SystemExit("opencv_status가 비어 있다 — 결합 run이 아니다.")

    defect_n = labels.count("defect")
    good_n = labels.count("good")
    total_n = len(labels)

    engines: dict[str, Any] = {}
    for name, sts in statuses.items():
        esc = _indicator(sts, labels, "OK", "defect")
        fng = _indicator(sts, labels, "NG", "good")
        rev = _indicator(sts, labels, "REVIEW", None)
        den_def = _denominator(labels, "defect")
        den_good = _denominator(labels, "good")
        den_all = _denominator(labels, None)
        engines[name] = {
            "escape": ratio("OK_on_defect / P", sum(esc), defect_n).as_dict(),
            "false_ng": ratio("NG_on_good / N", sum(fng), good_n).as_dict(),
            "review": ratio("REVIEW / T", sum(rev), total_n).as_dict(),
            "error_count": sts.count("ERROR"),
            "status_counts": {s: sts.count(s) for s in ("OK", "NG", "REVIEW", "ERROR")},
            "bootstrap": {
                "escape": group_bootstrap_ci(esc, den_def),
                "false_ng": group_bootstrap_ci(fng, den_good),
                "review": group_bootstrap_ci(rev, den_all),
            },
            "_arrays": {"esc": esc, "fng": fng, "rev": rev},
        }

    # 짝 차이: combined − r20 (같은 이미지 순서).
    paired: dict[str, Any] = {}
    for key, arr, den in (
        ("escape", "esc", _denominator(labels, "defect")),
        ("false_ng", "fng", _denominator(labels, "good")),
        ("review", "rev", _denominator(labels, None)),
    ):
        paired[key] = paired_bootstrap_delta(
            engines["combined"]["_arrays"][arr],
            den,
            engines["opencv_r20"]["_arrays"][arr],
            den,
        )

    degraded = sum(
        1
        for sid in sample_ids
        if combined[sid]["opencv_status"] == "ERROR" or seg[sid]["status"] == "ERROR"
    )
    recall_010 = float(combined_meta["recall_at"]["iou0.10"]["value"])
    recall_050 = float(combined_meta["recall_at"]["iou0.50"]["value"])

    c = engines["combined"]
    gates = {
        "G1_escape": {
            "rule": (
                f"escape <= {GATE_TARGETS['escape']} AND "
                f"Wilson 상한 <= {GATE_TARGETS['escape_ci_high']}"
            ),
            "value": c["escape"]["value"],
            "ci_high": c["escape"]["ci95_high"],
            "pass": c["escape"]["value"] <= GATE_TARGETS["escape"]
            and c["escape"]["ci95_high"] <= GATE_TARGETS["escape_ci_high"],
        },
        "G2_false_ng": {
            "rule": f"false-NG <= {GATE_TARGETS['false_ng']}",
            "value": c["false_ng"]["value"],
            "pass": c["false_ng"]["value"] <= GATE_TARGETS["false_ng"],
        },
        "G3_review": {
            "rule": f"review <= {GATE_TARGETS['review']}",
            "value": c["review"]["value"],
            "pass": c["review"]["value"] <= GATE_TARGETS["review"],
        },
        "G4_recall_010": {
            "rule": f"recall@0.10 >= {GATE_TARGETS['recall_010']}",
            "value": recall_010,
            "pass": recall_010 >= GATE_TARGETS["recall_010"],
        },
        "G5_paired_escape": {
            "rule": "짝 차이(combined-r20) 점추정 < 0 AND 부트스트랩 CI 상한 <= 0",
            "delta": paired["escape"]["delta"],
            "ci_low": paired["escape"]["ci_low"],
            "ci_high": paired["escape"]["ci_high"],
            "pass": paired["escape"]["delta"] < 0 and paired["escape"]["ci_high"] <= 0,
        },
        "G6_error": {
            "rule": "combined ERROR 0건",
            "value": c["error_count"],
            "degraded": degraded,
            "pass": c["error_count"] == 0,
        },
    }
    all_pass = all(g["pass"] for g in gates.values())
    verdict = (
        "채택 근거 확보 (G1~G6 전부 통과)"
        if all_pass
        else "채택 보류 — r20 유지 (미통과 게이트 존재)"
    )

    elapsed = time.perf_counter() - started
    payload = {
        "generated": "2026-11-13",
        "split": "val",
        "samples": total_n,
        "defect": defect_n,
        "good": good_n,
        "groups": len(set(groups)),
        "config_hash": combined_meta["config_hash"],
        "onnx": combined_meta["onnx"],
        "policy": combined_meta["policy"],
        "bootstrap": {
            "b": DEFAULT_BOOTSTRAP_B,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "alpha": DEFAULT_ALPHA,
        },
        "consistency": {
            "inference_mismatch": inference_mismatch,
            "label_mismatch": label_mismatch,
        },
        "engines": {
            name: {k: v for k, v in data.items() if k != "_arrays"}
            for name, data in engines.items()
        },
        "paired_delta_combined_minus_r20": paired,
        "recall_at": {"iou0.10": recall_010, "iou0.50": recall_050},
        "degraded": degraded,
        "gates": gates,
        "verdict": verdict,
        "analysis_elapsed_sec": round(elapsed, 2),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def fmt(metric: dict[str, Any]) -> str:
        return (
            f"{metric['value']:.4f} [{metric['ci95_low']:.3f}, {metric['ci95_high']:.3f}] "
            f"({metric['numerator']:g}/{metric['denominator']:g})"
        )

    lines = [
        "# E_val_gate — E트랙 채택 게이트 (val 400 짝비교, P0-1)",
        "",
        "- 생성일: 2026-11-13",
        f"- 표본: val {total_n} (defect {defect_n} + good {good_n}, 그룹 {len(set(groups))}개)",
        f"- config: {combined_meta['config_hash']}",
        f"- ONNX: {combined_meta['onnx']} / "
        f"정책 ng_conf={combined_meta['policy']['ng_conf']}, "
        f"review_lo={combined_meta['policy']['review_lo']}",
        f"- 부트스트랩: b={DEFAULT_BOOTSTRAP_B}, seed={DEFAULT_BOOTSTRAP_SEED}, "
        f"alpha={DEFAULT_ALPHA} (F6 이미지 단위 복원추출)",
        "- 채택 기준은 실측 전에 사전 등록했다(G1~G6, 동결 목표 값). 결과 후 기준 변경 없음.",
        "",
        "## 1. 짝비교 표 (같은 이미지·같은 정책)",
        "",
        "| 엔진 | escape(/43) | false-NG(/357) | review(/400) | ERROR |",
        "|---|---|---|---|---|",
    ]
    for name, label in (
        ("opencv_r20", "OpenCV r20"),
        ("yolo_seg", "YOLO seg"),
        ("combined", "YOLO seg + OpenCV(결합)"),
    ):
        e = engines[name]
        lines.append(
            f"| {label} | {fmt(e['escape'])} | {fmt(e['false_ng'])} | "
            f"{fmt(e['review'])} | {e['error_count']} |"
        )
    lines += [
        "",
        f"- recall@0.10(결합, YOLO 위치): {recall_010:.4f} / "
        f"recall@0.50: {recall_050:.4f} (분모 50 GT)",
        f"- degraded(한쪽 엔진 ERROR): {degraded}건",
        f"- 추론 동일성 검증: 불일치 {len(inference_mismatch)}건 / "
        f"라벨 불일치 {len(label_mismatch)}건",
        "",
        "## 2. 짝 차이 (combined − r20, 부트스트랩 95%)",
        "",
        "| 지표 | 점추정 | CI 하한 | CI 상한 | n |",
        "|---|---|---|---|---|",
    ]
    for key in ("escape", "false_ng", "review"):
        p = paired[key]
        lines.append(
            f"| {key} | {p['delta']:+.4f} | {p['ci_low']:+.4f} | {p['ci_high']:+.4f} | {p['n']} |"
        )
    lines += [
        "",
        "## 3. 게이트 판정 (사전 등록 기준)",
        "",
        "| 게이트 | 기준 | 실측 | 판정 |",
        "|---|---|---|---|",
    ]
    for name, g in gates.items():
        measured = g.get("value", g.get("delta"))
        if name == "G1_escape":
            measured = f"{g['value']:.4f} (상한 {g['ci_high']:.3f})"
        elif name == "G5_paired_escape":
            measured = f"{g['delta']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]"
        elif name == "G6_error":
            measured = f"ERROR {g['value']}건, degraded {g['degraded']}건"
        else:
            measured = f"{measured:.4f}"
        lines.append(f"| {name} | {g['rule']} | {measured} | {'PASS' if g['pass'] else 'FAIL'} |")
    lines += [
        "",
        "## 4. 결론",
        "",
        f"- **{verdict}**",
        "- 이 보고서는 scripts/e_val_paired.py가 자동 생성한다. 수치를 손으로 고치지 않는다.",
        f"- 분석 소요: {elapsed:.1f}s (val 결합 실측 82.2s는 별도 기록)",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"표본: val {total_n} (defect {defect_n}, good {good_n}, 그룹 {len(set(groups))})")
    for name in ("opencv_r20", "yolo_seg", "combined"):
        e = engines[name]
        print(
            f"{name}: escape {fmt(e['escape'])} / false-NG {fmt(e['false_ng'])} / "
            f"review {fmt(e['review'])} / ERROR {e['error_count']}"
        )
    for key in ("escape", "false_ng", "review"):
        p = paired[key]
        print(f"짝차이 {key}: {p['delta']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]")
    for name, g in gates.items():
        print(f"{name}: {'PASS' if g['pass'] else 'FAIL'}")
    print(f"판정: {verdict}")
    print(f"분석 소요: {elapsed:.1f}s")
    print(f"증거 JSON: {OUT_JSON}")
    print(f"게이트 보고서: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
