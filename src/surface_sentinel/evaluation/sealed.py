"""T08 봉인 test 일회성 측정 (설계 §2.5, §4.1, §4.4).

봉인 분할(``test``/``official_test``)은 설정 동결 후 ``--allow-sealed`` 플래그로만
연다. 첫 접근은 ``data/splits/<split>.lock.json`` 에 split 해시·config 해시·코드
SHA·시각·목적을 기록하고, 이후 접근은 ``access_count`` 를 올리며 ``reuse=true`` 로
둔다. ``official_test`` 는 별도 잠금을 쓴다. 첫 봉인 실행 이후의 config 변경은
``post_seal_config_change`` 로 기록한다(막지 않되 숨기지 않는다).

측정 범위(설계 §4.1):

* ``test``: B0/B1/P 전수 1회. B2·ablation·튜닝 그리드·파일럿은 실행하지 않는다.
* ``official_test``: 동결 config 로 P 전수 보조 1회.

모든 비율에 Wilson 95% CI 와 분모를 붙이고, 그룹 부트스트랩 CI(F6: b=2000,
seed=20261111, alpha=0.05)를 함께 남긴다. **test 를 본 뒤 임계값은 바꾸지 않는다.**
목표 미달도 실제 값 그대로 보고한다.

봉인 CSV 파일명은 ``f"{split}.csv"`` 로 조립한다(리터럴 금지 계약 보존).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import load_config
from .adopted import METRIC_NAMES, metric_arrays
from .baselines import B0_NAME, B1_NAME, evaluate_baselines
from .bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_BOOTSTRAP_SEED,
    group_bootstrap_ci,
)
from .cache import CACHE_DIR, DEFAULT_RAW_ROOT
from .failures import (
    build_failure_rows,
    reason_distribution,
    select_cases,
    write_failure_csv,
)
from .grid import (
    TARGET_ESCAPE,
    TARGET_FALSE_NG,
    TARGET_LOCALIZATION_RECALL,
    TARGET_REVIEW,
)
from .measure import VariantReport, evaluate_cached
from .recalibration import TARGET_ESCAPE_CI_HIGH, git_head
from .report import _fmt, _table
from .runner import (
    SEALED_SPLITS,
    EvalError,
    SealedSplitAccess,
    cached_samples,
    load_split_rows,
    split_sha256,
)
from .sampling import build_sample_plan

DEFAULT_CONFIG_PATH = Path("configs/base.toml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_EVIDENCE_DIR = Path("docs/evidence")
DEFAULT_REPORT_DIR = Path("docs/reports")
DEFAULT_PURPOSE = "T08 봉인 test 일회성 측정"

#: 보고서에 적는 목표(설계 §7). 실제 값과 함께 PASS/FAIL 을 숨기지 않는다.
GOAL_ROWS = (
    ("escape 점추정", "defect_escape_rate", "<=", TARGET_ESCAPE),
    ("escape Wilson 상한", "defect_escape_rate_ci_high", "<=", TARGET_ESCAPE_CI_HIGH),
    ("false-NG", "good_false_ng_rate", "<=", TARGET_FALSE_NG),
    ("review", "review_rate", "<=", TARGET_REVIEW),
    ("recall@IoU0.10", "localization_recall_iou0.10", ">=", TARGET_LOCALIZATION_RECALL),
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def ensure_sealed_access(
    split: str,
    *,
    splits_dir: str | Path,
    allow_sealed: bool,
    config_hash: str,
    code_sha: str,
    purpose: str,
) -> dict[str, Any] | None:
    """봉인 접근을 기록한다. tune 등 비봉인은 None 을 돌려준다(잠금 없음).

    ``allow_sealed`` 가 False 면 파일을 열기 전에 ``SealedSplitAccess``(exit 6)로
    거부한다. 첫 접근은 잠금을 만들고, 이후 접근은 ``access_count`` 를 올린다.
    """

    if split not in SEALED_SPLITS:
        return None
    if not allow_sealed:
        raise SealedSplitAccess(
            f"봉인된 split은 --allow-sealed 없이 열 수 없습니다: {split}. "
            "T08 일회성 측정만 봉인을 해제합니다."
        )
    splits = Path(splits_dir)
    split_path = splits / f"{split}.csv"
    if not split_path.is_file():
        raise EvalError(f"분할 CSV를 찾을 수 없습니다: {split_path}")
    digest = split_sha256(split_path)
    lock_path = splits / f"{split}.lock.json"
    now = _utc_now_iso()
    record = {
        "timestamp": now,
        "config_hash": config_hash,
        "code_sha": code_sha,
        "purpose": purpose,
    }
    if lock_path.is_file():
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        accesses = list(payload.get("accesses", []))
        accesses.append(record)
        payload["accesses"] = accesses
        payload["access_count"] = len(accesses)
        payload["reuse"] = len(accesses) > 1
        first_config = accesses[0].get("config_hash", config_hash)
        payload["post_seal_config_change"] = config_hash != first_config
        if payload.get("split_sha256") != digest:
            payload["split_hash_changed"] = True
            payload["split_sha256_current"] = digest
    else:
        payload = {
            "split": split,
            "split_sha256": digest,
            "config_hash": config_hash,
            "code_sha": code_sha,
            "created": now,
            "purpose": purpose,
            "access_count": 1,
            "reuse": False,
            "accesses": [record],
            "post_seal_config_change": False,
        }
    _atomic_write_json(lock_path, payload)
    return payload


def _goal_value(report: VariantReport, key: str) -> float | None:
    if key.endswith("_ci_high"):
        return report.ratio(key[: -len("_ci_high")]).ci_high
    return report.ratio(key).value


def _goals_block(report: VariantReport) -> dict[str, Any]:
    """목표 대비 실제 값과 PASS/FAIL. 분모 0(값 None)은 판정하지 않는다."""

    goals: dict[str, Any] = {}
    for label, key, op, target in GOAL_ROWS:
        value = _goal_value(report, key)
        if value is None:
            passed: bool | None = None
        elif op == "<=":
            passed = value <= target
        else:
            passed = value >= target
        goals[label] = {
            "metric": key,
            "op": op,
            "target": target,
            "value": value,
            "pass": passed,
        }
    return goals


def _variant_payload(
    samples: Any,
    report: VariantReport,
    *,
    b: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    arrays = metric_arrays(samples, report)
    ci = {
        name: group_bootstrap_ci(*arrays.pair(name), b=b, seed=seed, alpha=alpha)
        for name in METRIC_NAMES
    }
    return {"report": report.as_dict(), "bootstrap_ci": ci}


def _render_report(
    payload: dict[str, Any],
    reports: dict[str, VariantReport],
    *,
    split: str,
    date: str,
) -> str:
    plan = payload["sample_plan"]
    lock = payload["lock"]
    order = [name for name in (B0_NAME, B1_NAME, "P") if name in reports]
    lines = [
        f"# T08 봉인 측정 보고서 ({split}, {date})",
        "",
        f"- 분할: `{split}` (봉인, `--allow-sealed` 일회성 측정)",
        f"- 표본: defect {plan['defect_taken']} (전수) + good {plan['good_taken']} (전수) "
        f"= {plan['total']}",
        f"- 동결 config: `{payload['config_path']}` (hash `{payload['config_hash'][:12]}`)",
        f"- 코드 SHA: `{payload['code_sha'] or '(없음)'}`",
        f"- 봉인 접근: access_count={lock['access_count']}, reuse={str(lock['reuse']).lower()}, "
        f"post_seal_config_change={str(lock['post_seal_config_change']).lower()}",
        f"- 목적: {payload['purpose']}",
        "",
        "모든 비율은 `값 [Wilson 95% CI] (분모 n)`으로 적는다. 분모가 0이면 값을 쓰지 않는다.",
        "**test를 본 뒤 임계값·config는 바꾸지 않는다**(설계 §4.4). 목표 미달도 숨기지 않는다.",
        "",
    ]
    if split == "official_test":
        lines.extend(
            [
                "> official_test는 내부 tune/test와 중복이 있어 **독립 감사가 아니다**(보조 측정).",
                "> 중복 수량은 `docs/evidence/T08_data_verify.md` 감사 증거를 참조한다.",
                "",
            ]
        )

    lines.append("## 요약 판정 (목표 대비 실제 값)")
    lines.append("")
    goal_rows: list[list[str]] = []
    any_fail = False
    for label, goal in payload["goals"].items():
        value = goal["value"]
        verdict = "판정 불가(분모 0)" if goal["pass"] is None else (
            "PASS" if goal["pass"] else "FAIL"
        )
        if goal["pass"] is False:
            any_fail = True
        shown = "-" if value is None else f"{value:.4f}"
        goal_rows.append(
            [label, f"{goal['op']} {goal['target']:.2f}", shown, verdict]
        )
    lines.extend(_table(["목표", "기준", "실제 값", "판정"], goal_rows))
    lines.append("")
    if any_fail:
        lines.append("**목표 미달 항목이 있다. 위 실제 값 그대로 보고하며 은폐하지 않는다.**")
        lines.append("")

    lines.append("## 1. 기준선·제안 파이프라인 (설계 §4.1·§4.2)")
    lines.append("")
    metric_rows = (
        "defect_escape_rate",
        "defect_ng_rate",
        "defect_review_rate",
        "good_false_ng_rate",
        "good_review_rate",
        "review_rate",
        "localization_recall_iou0.10",
        "localization_recall_iou0.30",
    )
    variant_rows = [
        [metric] + [_fmt(reports[name].ratio(metric)) for name in order]
        for metric in metric_rows
    ]
    lines.extend(_table(["지표", *order], variant_rows))
    lines.append("")
    lines.append("B0/B1은 REVIEW를 내지 않는 이진 기준선이다(review 지표 0이 정상).")
    lines.append("")

    lines.append("## 2. 그룹 부트스트랩 CI (F6)")
    lines.append("")
    boot = payload["bootstrap"]
    lines.append(
        f"b={boot['b']}, seed={boot['seed']}, alpha={boot['alpha']}. "
        "이미지 단위 복원추출, 분모 0 이미지는 제외."
    )
    lines.append("")
    p_ci = payload["variants"]["P"]["bootstrap_ci"]
    ci_rows = []
    for name in METRIC_NAMES:
        entry = p_ci[name]
        value = "-" if entry["value"] is None else f"{entry['value']:.4f}"
        low = "-" if entry["ci_low"] is None else f"{entry['ci_low']:.4f}"
        high = "-" if entry["ci_high"] is None else f"{entry['ci_high']:.4f}"
        ci_rows.append([name, value, low, high, str(entry["n"])])
    lines.extend(_table(["지표(P)", "값", "CI low", "CI high", "n"], ci_rows))
    lines.append("")

    lines.append("## 3. 실패 사례 (설계 §4.5 규칙 선택)")
    lines.append("")
    selection = payload["failure_selection"]
    counts = selection.get("category_counts", {})
    lines.append(
        "카테고리별 실패 수: "
        + (", ".join(f"{key} {value}" for key, value in sorted(counts.items())) or "없음")
    )
    lines.append("")
    for key in ("high_score_escape_top2", "high_score_false_ng_top2"):
        entries = selection.get(key, [])
        lines.append(f"### {key}")
        lines.append("")
        if not entries:
            lines.append("해당 사례 없음.")
            lines.append("")
            continue
        lines.extend(
            _table(
                ["sample_id", "label", "status", "score", "best_iou"],
                [
                    [
                        entry["sample_id"],
                        entry["label"],
                        entry["status"],
                        f"{entry['score']:.4f}",
                        f"{entry['best_iou']:.4f}",
                    ]
                    for entry in entries
                ],
            )
        )
        lines.append("")
    lines.append(
        f"전체 실패 색인: `{payload['paths']['failure_index']}` / "
        f"대표 사례: `{payload['paths']['failure_selection']}`"
    )
    lines.append("")

    lines.append("## 4. 재현")
    lines.append("")
    lines.append("```powershell")
    lines.append(f"uv run surface-sentinel eval --split {split} --allow-sealed")
    lines.append("```")
    lines.append("")
    lines.append("산출물:")
    lines.append("")
    for key, value in payload["paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def run_sealed_measurement(
    *,
    split: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    cache_dir: str | Path = CACHE_DIR,
    purpose: str = DEFAULT_PURPOSE,
    allow_sealed: bool = True,
    use_cache: bool = True,
    overlays: bool = True,
    b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """봉인 분할을 전수 측정하고 증거·보고서·잠금을 남긴다.

    ``test`` 는 B0/B1/P, ``official_test`` 는 P 만 실행한다. ablation·그리드·
    파일럿은 실행하지 않는다(설계 §4.1). 동결 config 를 그대로 쓴다.
    """

    if split not in SEALED_SPLITS:
        raise EvalError(f"봉인 측정은 {sorted(SEALED_SPLITS)}만 실행합니다: {split!r}")
    config = load_config(config_path)
    code_sha = git_head()
    lock = ensure_sealed_access(
        split,
        splits_dir=splits_dir,
        allow_sealed=allow_sealed,
        config_hash=config.config_hash,
        code_sha=code_sha,
        purpose=purpose,
    )
    rows = load_split_rows(split, splits_dir=splits_dir, allow_sealed=True)
    good_count = sum(1 for row in rows if row.label == "good")
    plan = build_sample_plan(
        rows, split=split, good_target=good_count, allow_sealed=True
    )
    split_hash = split_sha256(Path(splits_dir) / f"{split}.csv")
    print(
        f"봉인 표본({split}): defect {plan.defect_taken}/{plan.defect_population} (전수), "
        f"good {plan.good_taken}/{plan.good_population} (전수)",
        flush=True,
    )
    samples = cached_samples(
        plan,
        config,
        raw_root=raw_root,
        split_hash=split_hash,
        cache_dir=cache_dir,
        overlays=overlays,
        use_cache=use_cache,
        label="P",
        allow_sealed=True,
    )

    reports: dict[str, VariantReport] = {
        "P": evaluate_cached(
            samples,
            config,
            name="P",
            notes=("제안 파이프라인. 동결 config, 캐시된 컴포넌트 위에서 재채점.",),
        )
    }
    if split == "test":
        for baseline in evaluate_baselines(
            samples, config, raw_root=raw_root, which=(B0_NAME, B1_NAME)
        ):
            reports[baseline.name] = baseline

    variants = {
        name: _variant_payload(samples, report, b=b, seed=bootstrap_seed, alpha=alpha)
        for name, report in reports.items()
    }
    goals = _goals_block(reports["P"])

    failure_rows = build_failure_rows(reports["P"], samples)
    selection = select_cases(failure_rows, reports["P"], samples, config)
    selection["reason_distribution"] = reason_distribution(failure_rows)

    evidence_path = Path(evidence_dir)
    report_path_dir = Path(report_dir)
    date = datetime.now(UTC).date().isoformat()
    failure_index = evidence_path / "failure_cases" / f"T08_{split}_failure_index.csv"
    failure_selection = evidence_path / "failure_cases" / f"T08_{split}_failure_selection.json"
    evidence_json = evidence_path / f"T08_sealed_{split}.json"
    report_md = report_path_dir / f"T08_sealed_{split}_{date}.md"

    write_failure_csv(failure_rows, failure_index)
    _atomic_write_json(failure_selection, selection)

    payload: dict[str, Any] = {
        "split": split,
        "generated_at": _utc_now_iso(),
        "purpose": purpose,
        "config_path": Path(config_path).as_posix(),
        "config_hash": config.config_hash,
        "code_sha": code_sha,
        "lock": lock,
        "sample_plan": plan.as_dict(),
        "bootstrap": {"b": b, "seed": bootstrap_seed, "alpha": alpha},
        "variants": variants,
        "goals": goals,
        "failure_selection": selection,
        "paths": {
            "evidence": evidence_json.as_posix(),
            "report": report_md.as_posix(),
            "failure_index": failure_index.as_posix(),
            "failure_selection": failure_selection.as_posix(),
            "lock": (Path(splits_dir) / f"{split}.lock.json").as_posix(),
        },
    }
    _atomic_write_json(evidence_json, payload)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        _render_report(payload, reports, split=split, date=date), encoding="utf-8"
    )
    print(f"봉인 증거: {evidence_json.as_posix()}", flush=True)
    print(f"봉인 보고서: {report_md.as_posix()}", flush=True)
    return payload


__all__ = [
    "DEFAULT_PURPOSE",
    "GOAL_ROWS",
    "ensure_sealed_access",
    "run_sealed_measurement",
]
