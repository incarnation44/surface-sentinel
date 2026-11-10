"""T05 실행 진입점: 표본 → 캐시 → P/B0·B1·B2 → A1–A6 → 그리드 → 보고서.

한 번 돌리면 아래를 남긴다.

| 산출물 | 경로 |
|---|---|
| 튜닝 그리드 | ``docs/evidence/tune_grid.jsonl`` |
| 변형별 지표 | ``docs/evidence/T05_variants.json`` |
| 표본 근거 | ``docs/evidence/T05_sample_plan.json`` |
| 실패 색인 | ``docs/evidence/failure_cases/T05_failure_index.csv`` |
| 실패 대표 사례 | ``docs/evidence/failure_cases/T05_failure_selection.json`` |
| 후보 config | ``configs/candidates/*.toml`` (base는 건드리지 않는다) |
| 보고서 | ``docs/reports/T05_eval_2026-11-10.md`` |

봉인 분할은 **이름 단계에서** 거부한다. ``data/splits/test.csv``와
``official_test.csv``는 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig, load_config
from ..io.manifest import ManifestRow, load_manifest
from . import report as report_module
from .ablation import ABLATIONS, apply_overrides, evaluate_ablation, spec_config
from .baselines import B0_NAME, B1_NAME, B2_NAME, evaluate_baselines
from .cache import (
    CACHE_DIR,
    OVERLAY_DIR,
    CachedSample,
    build_cache,
    cache_meta,
    cache_path,
    cache_summary,
    load_cache,
    save_cache,
)
from .failures import (
    build_failure_rows,
    reason_distribution,
    select_cases,
    write_failure_csv,
)
from .grid import (
    TARGET_ESCAPE,
    TARGET_FALSE_NG,
    TARGET_REVIEW,
    GridRow,
    Selection,
    build_candidates,
    evaluate_grid,
    minimax_key,
    select,
    write_jsonl,
)
from .measure import VariantReport, evaluate_cached
from .pilot import (
    PILOT_CANDIDATES,
    PilotRow,
    evaluate_pilot_scoring,
    select_pilot,
)
from .sampling import SamplePlan, build_sample_plan

SEALED_SPLITS = frozenset({"test", "official_test"})
EVAL_SPLITS = ("tune",)

DEFAULT_CONFIG_PATH = Path("configs/base.toml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_RAW_ROOT = Path("data/raw/kolektorsdd2")
EVIDENCE_DIR = Path("docs/evidence")
FAILURE_DIR = EVIDENCE_DIR / "failure_cases"
REPORT_DIR = Path("docs/reports")
CANDIDATE_DIR = Path("configs/candidates")
REPORT_NAME = "T05_eval_2026-11-10.md"
GRID_NAME = "tune_grid.jsonl"


class EvalError(RuntimeError):
    """평가를 계속할 수 없을 때의 오류. ``exit_code``를 갖는다."""

    exit_code = 3


class SealedSplitAccess(EvalError):
    """봉인 분할 접근 거부 (종료 코드 6)."""

    exit_code = 6


def load_split_rows(
    split: str = "tune",
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> list[ManifestRow]:
    """평가용 분할 CSV를 읽는다. 봉인 분할은 파일을 열기 전에 거부한다."""

    if split in SEALED_SPLITS:
        raise SealedSplitAccess(
            f"봉인된 split은 평가에서 열 수 없습니다: {split}. "
            "T05는 tune에서만 실행합니다."
        )
    if split not in EVAL_SPLITS:
        raise EvalError(f"T05는 {list(EVAL_SPLITS)}만 실행합니다: {split!r}")
    path = Path(splits_dir) / f"{split}.csv"
    if path.name in {f"{name}.csv" for name in SEALED_SPLITS}:
        raise SealedSplitAccess(f"봉인된 분할 파일은 열지 않습니다: {path}")
    if not path.is_file():
        raise EvalError(f"분할 CSV를 찾을 수 없습니다: {path}")
    return load_manifest(path)


def split_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(label: str) -> Callable[[int, int, str], None]:
    def report(done: int, total: int, sample_id: str) -> None:
        if done == total or done % 100 == 0:
            print(f"  {label}: {done}/{total} ({sample_id})", flush=True)

    return report


def cached_samples(
    plan: SamplePlan,
    config: AppConfig,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    split_hash: str = "",
    cache_dir: str | Path = CACHE_DIR,
    overlays: bool = False,
    use_cache: bool = True,
    label: str = "cache",
) -> tuple[CachedSample, ...]:
    """샘플 캐시를 읽거나 만든다. 키는 (버전, config hash, 분할, 표본 id)다."""

    meta = cache_meta(plan, config, split_sha256=split_hash)
    path = cache_path(meta, cache_dir)
    if use_cache and path.is_file():
        try:
            samples = load_cache(path, meta)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  캐시를 읽지 못해 다시 만듭니다: {exc}", flush=True)
        else:
            print(f"  캐시 재사용: {path}", flush=True)
            return samples
    started = time.perf_counter()
    samples = build_cache(
        plan,
        config,
        raw_root=raw_root,
        overlay_dir=OVERLAY_DIR if overlays else None,
        progress=_progress(label),
    )
    elapsed = time.perf_counter() - started
    save_cache(samples, meta, path)
    print(f"  캐시 생성: {path} ({elapsed:.1f}s)", flush=True)
    return samples


@dataclass(frozen=True)
class EvalResult:
    """실행 결과 묶음. CLI가 요약을 찍고 종료 코드를 정한다."""

    plan: SamplePlan
    base_config: AppConfig
    samples: tuple[CachedSample, ...]
    reports: tuple[VariantReport, ...]
    grid: tuple[GridRow, ...]
    selection: Selection
    pilots: tuple[PilotRow, ...]
    pilot_best: PilotRow | None
    adopted_label: str
    adopted_basis: str
    failures: tuple[dict[str, Any], ...]
    case_selection: dict[str, Any]
    paths: dict[str, str]
    elapsed_seconds: float

    def by_name(self, name: str) -> VariantReport:
        for report in self.reports:
            if report.name == name:
                return report
        raise KeyError(name)

    def summary(self) -> dict[str, Any]:
        headline = self.by_name("P")
        pilot = self.pilot_best
        payload: dict[str, Any] = {
            "split": self.plan.split,
            "samples": len(self.plan.rows),
            "defect": self.plan.defect_taken,
            "good": self.plan.good_taken,
            "proposal_escape": headline.ratio("defect_escape_rate").as_dict(),
            "proposal_false_ng": headline.ratio("good_false_ng_rate").as_dict(),
            "proposal_review": headline.ratio("review_rate").as_dict(),
            "grid_candidates": len(self.grid),
            "selection": self.selection.reason,
            "pilot_best": None if pilot is None else pilot.as_dict(),
            "adopted": self.adopted_label,
            "adopted_basis": self.adopted_basis,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "paths": self.paths,
        }
        return payload


def _posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _write_json(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _write_candidate_toml(config: AppConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config.canonical_toml(), encoding="utf-8")
    return target


def run_evaluation(
    *,
    split: str = "tune",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    evidence_dir: str | Path = EVIDENCE_DIR,
    report_dir: str | Path = REPORT_DIR,
    candidate_dir: str | Path = CANDIDATE_DIR,
    cache_dir: str | Path = CACHE_DIR,
    good_target: int | None = None,
    ablations: bool = True,
    baselines: bool = True,
    pilot: bool = True,
    use_cache: bool = True,
    overlays: bool = True,
) -> EvalResult:
    """T05 전체를 실행한다.

    파이프라인 실제 실행 = 기본 1 + ablation 6 + 후보 축 파일럿 3 = 10회.
    그리드 24후보는 캐시 재계산이므로 "실행"에 포함되지 않는다(설계 §4.4 상한 내).
    """

    started = time.perf_counter()
    rows = load_split_rows(split, splits_dir=splits_dir)
    split_path = Path(splits_dir) / f"{split}.csv"
    split_hash = split_sha256(split_path)

    plan = (
        build_sample_plan(rows, split=split)
        if good_target is None
        else build_sample_plan(rows, split=split, good_target=good_target)
    )
    base_config = load_config(config_path)
    print(
        f"표본: defect {plan.defect_taken}/{plan.defect_population} (전수), "
        f"good {plan.good_taken}/{plan.good_population} (층화)",
        flush=True,
    )

    samples = cached_samples(
        plan,
        base_config,
        raw_root=raw_root,
        split_hash=split_hash,
        cache_dir=cache_dir,
        overlays=overlays,
        use_cache=use_cache,
        label="P",
    )
    print(f"  캐시 요약: {cache_summary(samples)}", flush=True)

    reports: list[VariantReport] = [
        evaluate_cached(
            samples,
            base_config,
            name="P",
            notes=("제안 파이프라인. 캐시된 컴포넌트 위에서 evidence·score·판정만 재계산.",),
        )
    ]
    if baselines:
        reports.extend(
            evaluate_baselines(
                samples,
                base_config,
                raw_root=raw_root,
                which=(B0_NAME, B1_NAME, B2_NAME),
            )
        )

    if ablations:
        for spec in ABLATIONS:
            variant_config = spec_config(spec, base_config)
            variant_samples = cached_samples(
                plan,
                variant_config,
                raw_root=raw_root,
                split_hash=split_hash,
                cache_dir=cache_dir,
                overlays=False,
                use_cache=use_cache,
                label=spec.id,
            )
            reports.append(evaluate_ablation(spec, variant_samples, base_config))

    print(f"그리드: 최대 {len(build_candidates())}후보 평가", flush=True)
    grid_rows = evaluate_grid(samples, base_config)
    selection = select(grid_rows)
    print(f"  선택: {selection.reason}", flush=True)

    pilot_rows: list[PilotRow] = []
    pilot_best: PilotRow | None = None
    if pilot:
        for axis in PILOT_CANDIDATES:
            axis_config = apply_overrides(base_config, axis.overrides)
            axis_samples = cached_samples(
                plan,
                axis_config,
                raw_root=raw_root,
                split_hash=split_hash,
                cache_dir=cache_dir,
                overlays=False,
                use_cache=use_cache,
                label=f"pilot-{axis.id}",
            )
            pilot_rows.extend(
                evaluate_pilot_scoring(axis_samples, axis, base_config)
            )
        pilot_best = select_pilot(
            pilot_rows,
            escape_max=TARGET_ESCAPE,
            false_ng_max=TARGET_FALSE_NG,
            review_max=TARGET_REVIEW,
        )
        label = (
            "없음"
            if pilot_best is None
            else f"{pilot_best.candidate_id}+{pilot_best.scoring_id}"
        )
        print(
            f"  파일럿: {len(pilot_rows)}칸, 목표 동시 충족 {label}",
            flush=True,
        )

    evidence_path = Path(evidence_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)
    grid_path = write_jsonl(grid_rows, evidence_path / GRID_NAME)
    pilot_path = _write_json(
        [row.as_dict() for row in pilot_rows],
        evidence_path / "T05_candidates_pilot.json",
    )
    variants_path = _write_json(
        [report.as_dict() for report in reports],
        evidence_path / "T05_variants.json",
    )
    plan_path = _write_json(plan.as_dict(), evidence_path / "T05_sample_plan.json")

    if pilot_best is not None:
        adopted_label = f"{pilot_best.candidate_id}+{pilot_best.scoring_id}"
        adopted_config = apply_overrides(base_config, pilot_best.overrides)
        adopted_report = pilot_best.report
        basis = (
            "후보 축 파일럿에서 세 목표(escape·false-NG·review)를 동시에 만족한 "
            f"유일/최선 칸: {adopted_label}"
        )
    else:
        adopted = selection.best
        adopted_config = base_config if adopted is None else _config_for(adopted, base_config)
        adopted_report = reports[0] if adopted is None else adopted.report
        adopted_label = "base" if adopted is None else adopted.candidate_id
        basis = selection.reason

    failure_rows = build_failure_rows(adopted_report, samples)
    failure_dir = evidence_path / "failure_cases"
    failure_csv = write_failure_csv(failure_rows, failure_dir / "T05_failure_index.csv")
    case_selection = select_cases(failure_rows, adopted_report, samples, adopted_config)
    case_selection["reason_distribution"] = reason_distribution(failure_rows)
    case_json = _write_json(case_selection, failure_dir / "T05_failure_selection.json")

    candidate_paths: list[Path] = []
    candidate_paths.append(
        _write_candidate_toml(
            adopted_config,
            Path(candidate_dir) / f"t05_{adopted_label}_recommended.toml",
        )
    )
    pareto_rest = sorted(
        (point for point in selection.pareto if point is not selection.best),
        key=minimax_key,
    )
    for row in pareto_rest[:3]:
        candidate_paths.append(
            _write_candidate_toml(
                _config_for(row, base_config),
                Path(candidate_dir) / f"t05_{row.candidate_id}_pareto.toml",
            )
        )

    paths = {
        "grid": _posix(grid_path),
        "pilot": _posix(pilot_path),
        "variants": _posix(variants_path),
        "sample_plan": _posix(plan_path),
        "failure_index": _posix(failure_csv),
        "failure_selection": _posix(case_json),
        "candidates": ", ".join(_posix(path) for path in candidate_paths),
    }

    elapsed = time.perf_counter() - started
    result = EvalResult(
        plan=plan,
        base_config=base_config,
        samples=samples,
        reports=tuple(reports),
        grid=tuple(grid_rows),
        selection=selection,
        pilots=tuple(pilot_rows),
        pilot_best=pilot_best,
        adopted_label=adopted_label,
        adopted_basis=basis,
        failures=tuple(row.as_dict() for row in failure_rows),
        case_selection=case_selection,
        paths=paths,
        elapsed_seconds=elapsed,
    )

    report_path = Path(report_dir) / REPORT_NAME
    report_module.write_report(result, report_path, adopted_config=adopted_config)
    paths["report"] = _posix(report_path)
    return result


def _config_for(row: GridRow, base_config: AppConfig) -> AppConfig:
    sectioned: dict[str, dict[str, float]] = {}
    for key, value in row.overrides.items():
        section, name = key.split(".", 1)
        sectioned.setdefault(section, {})[name] = value
    return apply_overrides(base_config, sectioned)


__all__ = [
    "CANDIDATE_DIR",
    "EVIDENCE_DIR",
    "EvalError",
    "EvalResult",
    "FAILURE_DIR",
    "GRID_NAME",
    "REPORT_DIR",
    "REPORT_NAME",
    "SEALED_SPLITS",
    "SealedSplitAccess",
    "cached_samples",
    "load_split_rows",
    "run_evaluation",
    "split_sha256",
]
