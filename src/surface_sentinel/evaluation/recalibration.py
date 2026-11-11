"""T04 후보 생성 재보정 (2026-11-11).

T05가 확인한 병목(714/714장에 컴포넌트가 존재 → ``OK`` 경로가 구조적으로 닫힘)을
**설정 우선**으로 재보정한다. 파이프라인 코드도 캐시 스키마도 바꾸지 않는다.

원칙
----
* 후보는 ``configs/candidates/recalib/*.toml`` 한 파일이 한 후보다(전부 측정용·미채택).
* 스코어링 축은 T05 파일럿의 **S4**로 고정하고 후보 생성 축
  (``z_threshold``·``min_contrast_gray``·``open_kernel``·``close_kernel``·
  ``merge_kernel``·``min_area_px``)만 바꾼다.
* ``max_candidates``는 안전 상한으로 건드리지 않는다. 현재 코드는
  ``total_components > max_candidates``면 ``CANDIDATE_TRUNCATED``이므로 상한을
  낮추면 절단 가능성만 커진다.
* 표본은 T05와 동일 규칙(tune 분할, defect 전수 + good 층화 500, seed 고정)을 쓴다.
* 봉인 분할은 이름 단계에서 거부한다.

산출물
------
* ``docs/evidence/tune_grid2.jsonl`` — 후보 1줄씩(지표·분모·CI·절단 통계)
* ``docs/reports/T04_recalib_<날짜>.md`` — 결정 보고서
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_config
from .cache import CACHE_DIR, DEFAULT_RAW_ROOT, cache_meta, cache_path
from .grid import (
    TARGET_ESCAPE,
    TARGET_FALSE_NG,
    TARGET_LOCALIZATION_RECALL,
    TARGET_REVIEW,
    GridRow,
)
from .measure import evaluate_cached
from .runner import cached_samples, load_split_rows, split_sha256
from .sampling import EVAL_SPLITS, build_sample_plan

RECALIBRATION_CANDIDATE_DIR = Path("configs/candidates/recalib")
DEFAULT_JSONL = Path("docs/evidence/tune_grid2.jsonl")
DEFAULT_REPORT = Path("docs/reports/T04_recalib_2026-11-11.md")
DEFAULT_SPLITS_DIR = Path("data/splits")

#: T05 파일럿과 같은 표본 규칙. 바꾸면 비교가 성립하지 않는다.
SEED = "t05-2026-11-10"
GOOD_TARGET = 500

#: 통과 기준(§7.6). 승인되지 않은 새 하드 게이트를 임의로 추가하지 않는다.
TARGET_ESCAPE_CI_HIGH = 0.20


class RecalibrationError(RuntimeError):
    """재보정을 시작할 수 없을 때의 오류."""


def git_head() -> str:
    """현재 Git HEAD. 실패하면 빈 문자열."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


@dataclass(frozen=True)
class RecalibrationRow:
    """재보정 후보 하나의 결과."""

    candidate_id: str
    config_path: str
    config_hash: str
    grid_row: GridRow
    candidates_params: Mapping[str, Any]
    scoring_params: Mapping[str, Any]
    truncated: int
    samples: int
    components_total: int
    elapsed_seconds: float
    cache_hit: bool
    examples: Mapping[str, Sequence[str]]

    @property
    def escape(self) -> float | None:
        return self.grid_row.escape

    @property
    def escape_ci_high(self) -> float | None:
        return self.grid_row.metric("defect_escape_rate").get("ci95_high")

    @property
    def false_ng(self) -> float | None:
        return self.grid_row.false_ng

    @property
    def review(self) -> float | None:
        return self.grid_row.review

    @property
    def recall(self) -> float | None:
        return self.grid_row.localization_recall

    @property
    def avg_components(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.components_total / self.samples

    def criteria(self) -> dict[str, bool | None]:
        """통과 기준 5개 대조 결과."""
        escape = self.escape
        escape_high = self.escape_ci_high
        false_ng = self.false_ng
        review = self.review
        recall = self.recall
        return {
            "escape_le_0.15": None if escape is None else escape <= TARGET_ESCAPE,
            "escape_ci95_high_le_0.20": (
                None if escape_high is None else escape_high <= TARGET_ESCAPE_CI_HIGH
            ),
            "false_ng_le_0.20": None if false_ng is None else false_ng <= TARGET_FALSE_NG,
            "review_le_0.40": None if review is None else review <= TARGET_REVIEW,
            "localization_recall_ge_0.50": (
                None if recall is None else recall >= TARGET_LOCALIZATION_RECALL
            ),
        }

    def passes(self) -> bool:
        return all(value is True for value in self.criteria().values())

    def as_dict(self) -> dict[str, Any]:
        payload = self.grid_row.as_dict()
        payload.update(
            {
                "config_path": self.config_path,
                "candidates_params": dict(self.candidates_params),
                "scoring_params": dict(self.scoring_params),
                "truncated": {
                    "count": self.truncated,
                    "rate": (self.truncated / self.samples) if self.samples else None,
                    "samples": self.samples,
                },
                "components": {
                    "total": self.components_total,
                    "avg_per_image": self.avg_components,
                },
                "criteria": self.criteria(),
                "passes": self.passes(),
                "elapsed_seconds": self.elapsed_seconds,
                "cache_hit": self.cache_hit,
                "examples": {key: list(value) for key, value in self.examples.items()},
            }
        )
        return payload


def load_candidate_configs(
    candidate_dir: str | Path = RECALIBRATION_CANDIDATE_DIR,
) -> list[tuple[str, Path]]:
    """후보 TOML 목록. 파일명 순(결정론)이다."""
    directory = Path(candidate_dir)
    if not directory.is_dir():
        raise RecalibrationError(f"후보 디렉터리가 없습니다: {directory}")
    paths = sorted(directory.glob("*.toml"))
    if not paths:
        raise RecalibrationError(f"후보 TOML이 없습니다: {directory}")
    return [(path.stem, path) for path in paths]


def _examples(report: Any, limit: int = 3) -> dict[str, list[str]]:
    """escape·false-NG·REVIEW 대표 샘플 id."""
    buckets: dict[str, list[str]] = {"escape": [], "false_ng": [], "review": []}
    for outcome in report.outcomes:
        status = outcome.status.value
        if status == "OK" and outcome.label == "defect":
            key = "escape"
        elif status == "NG" and outcome.label == "good":
            key = "false_ng"
        elif status == "REVIEW":
            key = "review"
        else:
            continue
        if len(buckets[key]) < limit:
            buckets[key].append(outcome.sample_id)
    return buckets


def evaluate_candidate(
    candidate_id: str,
    config_path: str | Path,
    plan: Any,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    split_hash: str = "",
    cache_dir: str | Path = CACHE_DIR,
) -> RecalibrationRow:
    """후보 하나로 파이프라인을 돌리고 지표를 낸다."""
    config = load_config(config_path)
    meta = cache_meta(plan, config, split_sha256=split_hash)
    cache_file = cache_path(meta, cache_dir)
    cache_hit = cache_file.is_file()
    started = time.perf_counter()
    samples = cached_samples(
        plan,
        config,
        raw_root=raw_root,
        split_hash=split_hash,
        cache_dir=cache_dir,
        overlays=False,
        label=candidate_id,
    )
    elapsed = time.perf_counter() - started
    report = evaluate_cached(samples, config, name=candidate_id)
    values = config.to_dict()
    candidates_params = dict(values.get("candidates", {}))
    scoring_params = dict(values.get("scoring", {}))
    overrides = {
        f"candidates.{key}": float(value) for key, value in candidates_params.items()
    }
    grid_row = GridRow(
        candidate_id=candidate_id,
        overrides=overrides,
        config_hash=config.config_hash,
        report=report,
    )
    return RecalibrationRow(
        candidate_id=candidate_id,
        config_path=str(config_path).replace("\\", "/"),
        config_hash=config.config_hash,
        grid_row=grid_row,
        candidates_params=candidates_params,
        scoring_params=scoring_params,
        truncated=sum(1 for sample in samples if sample.truncated),
        samples=len(samples),
        components_total=sum(len(sample.components) for sample in samples),
        elapsed_seconds=elapsed,
        cache_hit=cache_hit,
        examples=_examples(report),
    )


def run_recalibration(
    candidate_dir: str | Path = RECALIBRATION_CANDIDATE_DIR,
    *,
    split: str = "tune",
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    cache_dir: str | Path = CACHE_DIR,
    good_target: int = GOOD_TARGET,
    seed: str = SEED,
) -> tuple[RecalibrationRow, ...]:
    """후보 전체를 고정 표본에서 측정한다."""
    if split not in EVAL_SPLITS:
        raise RecalibrationError(f"봉인·미지원 분할입니다: {split}")
    rows = load_split_rows(split, splits_dir=splits_dir)
    split_path = Path(splits_dir) / f"{split}.csv"
    split_hash = split_sha256(split_path)
    plan = build_sample_plan(rows, split=split, good_target=good_target, seed=seed)
    print(
        f"표본: defect {plan.defect_taken}/{plan.defect_population} (전수), "
        f"good {plan.good_taken}/{plan.good_population} (층화), seed={seed}",
        flush=True,
    )
    results: list[RecalibrationRow] = []
    for candidate_id, path in load_candidate_configs(candidate_dir):
        print(f"[{candidate_id}] 측정 시작", flush=True)
        row = evaluate_candidate(
            candidate_id,
            path,
            plan,
            raw_root=raw_root,
            split_hash=split_hash,
            cache_dir=cache_dir,
        )
        print(
            f"[{candidate_id}] escape {row.escape} (CI 상한 {row.escape_ci_high}) · "
            f"false-NG {row.false_ng} · review {row.review} · recall {row.recall} · "
            f"절단 {row.truncated}/{row.samples} · {row.elapsed_seconds:.1f}s · "
            f"통과={row.passes()}",
            flush=True,
        )
        results.append(row)
    return tuple(results)


def write_jsonl(rows: Sequence[RecalibrationRow], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) for row in rows]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _ci(ratio: Mapping[str, Any]) -> str:
    value = ratio.get("value")
    low = ratio.get("ci95_low")
    high = ratio.get("ci95_high")
    numerator = ratio.get("numerator")
    denominator = ratio.get("denominator")
    if value is None:
        return "—"
    return (
        f"{_pct(value)} [{_pct(low)}, {_pct(high)}] "
        f"({numerator}/{denominator})"
    )


def write_report(
    rows: Sequence[RecalibrationRow],
    path: str | Path,
    *,
    plan: Any = None,
    jsonl_path: str | Path = DEFAULT_JSONL,
    seed: str = SEED,
    code_revision: str = "",
) -> Path:
    """결정 보고서를 쓴다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    passed = [row for row in rows if row.passes()]
    lines: list[str] = []
    lines.append("# T04 후보 생성 재보정 보고서")
    lines.append("")
    lines.append(f"- 작성: {time.strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append(f"- 코드 리비전: {code_revision or '기록 없음'}")
    lines.append(
        f"- 표본: split=tune, defect {getattr(plan, 'defect_taken', '?')}/"
        f"{getattr(plan, 'defect_population', '?')} (전수), "
        f"good {getattr(plan, 'good_taken', '?')}/"
        f"{getattr(plan, 'good_population', '?')} (층화 {GOOD_TARGET}), seed={seed}"
    )
    lines.append(f"- 후보 수: {len(rows)} (상한 24)")
    lines.append(
        "- 봉인 접근: 없음. `test`/`official_test`는 읽지 않았고 `--allow-sealed`도 쓰지 않았다."
    )
    lines.append(
        "- `configs/base.toml`은 **수정하지 않았다**. 후보 파일은 전부 측정용·미채택이다."
    )
    lines.append("")
    lines.append("## 1. 결론")
    lines.append("")
    if passed:
        lines.append(
            f"통과 후보 {len(passed)}건: " + ", ".join(row.candidate_id for row in passed)
        )
        lines.append("")
        lines.append(
            "채택은 아직 하지 않는다. 아래 §4에서 여유와 리스크를 함께 보고한 뒤 "
            "사용자 승인을 받아야 한다."
        )
    else:
        lines.append("**통과 후보 0건.** 기준을 모두 동시에 만족한 후보가 없다.")
    lines.append("")
    lines.append("## 2. 후보 목록과 config_hash")
    lines.append("")
    lines.append(
        "| 후보 | z_threshold | min_contrast_gray | open | close | merge | "
        "min_area | config_hash |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        params = row.candidates_params
        lines.append(
            "| {id} | {z} | {c} | {o} | {cl} | {m} | {a} | `{h}` |".format(
                id=row.candidate_id,
                z=params.get("z_threshold"),
                c=params.get("min_contrast_gray"),
                o=params.get("open_kernel"),
                cl=params.get("close_kernel"),
                m=params.get("merge_kernel"),
                a=params.get("min_area_px"),
                h=row.config_hash,
            )
        )
    lines.append("")
    lines.append(
        "스코어링 축은 전 후보 S4 고정(ng_area_px 800, ng_contrast_gray 22, "
        "ok_max 0.40, ng_min 0.50). max_candidates는 20 고정(안전 상한)."
    )
    lines.append("")
    lines.append("## 3. 후보별 지표 (분자·분모·Wilson 95% CI)")
    lines.append("")
    lines.append(
        "| 후보 | escape | escape CI 상한 | false-NG | review | recall@IoU0.10 | "
        "절단 | 평균 컴포넌트 | 시간 | 캐시 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        metrics = row.grid_row.as_dict()["metrics"]
        lines.append(
            "| {id} | {esc} | {esc_hi} | {fng} | {rev} | {rec} | {tr}/{n} | "
            "{avg:.2f} | {sec:.1f}s | {cache} |".format(
                id=row.candidate_id,
                esc=_ci(metrics["escape"]),
                esc_hi=_pct(row.escape_ci_high),
                fng=_ci(metrics["false_ng"]),
                rev=_ci(metrics["review"]),
                rec=_ci(metrics["localization_recall_iou0.10"]),
                tr=row.truncated,
                n=row.samples,
                avg=row.avg_components,
                sec=row.elapsed_seconds,
                cache="재사용" if row.cache_hit else "신규",
            )
        )
    lines.append("")
    lines.append("## 4. 통과 기준 대조")
    lines.append("")
    lines.append(
        "기준: escape ≤ 0.15 · escape CI 상한 ≤ 0.20 · false-NG ≤ 0.20 · "
        "review ≤ 0.40 · recall@IoU0.10 ≥ 0.50 (모두 동시에 충족)"
    )
    lines.append("")
    lines.append("| 후보 | escape | escape CI | false-NG | review | recall | 결과 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        criteria = row.criteria()
        mark = lambda flag: "✅" if flag else ("—" if flag is None else "❌")  # noqa: E731
        lines.append(
            "| {id} | {a} | {b} | {c} | {d} | {e} | {result} |".format(
                id=row.candidate_id,
                a=mark(criteria["escape_le_0.15"]),
                b=mark(criteria["escape_ci95_high_le_0.20"]),
                c=mark(criteria["false_ng_le_0.20"]),
                d=mark(criteria["review_le_0.40"]),
                e=mark(criteria["localization_recall_ge_0.50"]),
                result="**통과**" if row.passes() else "미통과",
            )
        )
    lines.append("")
    lines.append("## 5. Pareto 표")
    lines.append("")
    lines.append("| 후보 | escape | false-NG | review | recall |")
    lines.append("|---|---|---|---|---|")
    for row in sorted(
        rows,
        key=lambda item: (
            item.escape if item.escape is not None else 1.0,
            item.false_ng if item.false_ng is not None else 1.0,
            item.review if item.review is not None else 1.0,
            -(item.recall if item.recall is not None else 0.0),
            item.candidate_id,
        ),
    ):
        lines.append(
            f"| {row.candidate_id} | {_pct(row.escape)} | {_pct(row.false_ng)} | "
            f"{_pct(row.review)} | {_pct(row.recall)} |"
        )
    lines.append("")
    lines.append("## 6. 대표 사례")
    lines.append("")
    for row in rows:
        examples = row.examples
        lines.append(
            f"- **{row.candidate_id}** — escape: {', '.join(examples['escape']) or '없음'}"
            f" · false-NG: {', '.join(examples['false_ng']) or '없음'}"
            f" · REVIEW: {', '.join(examples['review']) or '없음'}"
        )
    lines.append("")
    lines.append("## 7. 재현 명령")
    lines.append("")
    lines.append("```text")
    lines.append(
        "uv run surface-sentinel recalibrate --split tune "
        "--candidate-dir configs/candidates/recalib"
    )
    lines.append(f"  --jsonl {jsonl_path} --report {target.as_posix()}")
    lines.append("```")
    lines.append("")
    lines.append("## 8. 산출물")
    lines.append("")
    lines.append(f"- JSONL: `{jsonl_path}`")
    lines.append(f"- 보고서: `{target.as_posix()}`")
    lines.append("- 후보 설정: `configs/candidates/recalib/*.toml` (전부 측정용·미채택)")
    lines.append("")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = [
    "DEFAULT_JSONL",
    "DEFAULT_REPORT",
    "GOOD_TARGET",
    "RECALIBRATION_CANDIDATE_DIR",
    "SEED",
    "TARGET_ESCAPE_CI_HIGH",
    "RecalibrationError",
    "RecalibrationRow",
    "evaluate_candidate",
    "git_head",
    "load_candidate_configs",
    "run_recalibration",
    "write_jsonl",
    "write_report",
]
