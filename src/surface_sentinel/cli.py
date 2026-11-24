"""Surface Sentinel CLI.

제공 서브커맨드: ``doctor``, ``data``(``build-manifest``, ``make-splits``, ``verify``),
``inspect``, ``batch``, ``eval``, ``recalibrate``, ``fullverify-details``,
``adopted-verify``, ``defect-map``, ``serve``, ``yolo``(``export``, ``evaluate``),
``records``(``init``, ``list``, ``stats``, ``verify``), ``worker``(``run``),
``agent``(``run``, ``replay``, ``audit``, ``drift``, ``triage``, ``explain``, ``calibrate``),
``gate``, ``repro``(``check``). 사람이 읽는 도움말과 오류 메시지는 한국어다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .analysis.defect_map import (
    DEFAULT_GRID,
    DefectMapError,
    accumulate,
    load_excluded_ids,
    render_heatmap,
    write_stats,
)
from .batch import (
    BatchError,
    SealedSplitError,
    run_batch,
)
from .config import CONFIG_SCHEMA_VERSION, ConfigError, load_config
from .contracts import ErrorCode, Status
from .io.audit import (
    verify_dataset,
    write_audit_json,
    write_audit_markdown,
    write_contact_sheet,
)
from .io.manifest import (
    DHASH_THRESHOLD,
    ManifestError,
    build_manifest,
    load_manifest,
)
from .io.splits import (
    SplitError,
    assign_groups,
    build_groups,
    verify_splits,
    write_splits,
)
from .pipeline import inspect_image
from .version import __version__

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_ERROR = 4
EXIT_STORAGE = 5
EXIT_SEALED = 6
EXIT_GATE = 7

INPUT_ERROR_CODES = frozenset(
    {
        ErrorCode.E_INPUT_MISSING,
        ErrorCode.E_INPUT_DECODE,
        ErrorCode.E_INPUT_SIZE,
        ErrorCode.E_INPUT_UNIFORM,
        ErrorCode.E_MASK_SHAPE,
        ErrorCode.E_MASK_MISMATCH,
    }
)


def _doctor() -> dict[str, Any]:
    info: dict[str, Any] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_schema": CONFIG_SCHEMA_VERSION,
        "cwd": str(Path.cwd()),
        "opencv": None,
        "numpy": None,
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except Exception:
        info["opencv"] = None
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:
        info["numpy"] = None
    return info


def _add_data_parser(subparsers: Any) -> None:
    data_parser = subparsers.add_parser(
        "data",
        help="데이터 manifest, tune/test 분할, 감사를 관리합니다.",
        description=(
            "KolektorSDD2 반입본에서 manifest를 만들고, 중복 그룹 단위 "
            "tune/test 분할과 누출/무결성 감사를 수행합니다."
        ),
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    build = data_subparsers.add_parser(
        "build-manifest",
        help="원본 이미지/마스크를 디코드해 dataset_manifest.csv를 만듭니다.",
        description=(
            "raw_root 아래 train/test PNG를 이미지/마스크로 페어링하고, "
            "sha256·dHash 기반 중복 그룹과 manifest를 결정론적으로 생성합니다."
        ),
    )
    build.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="원본 데이터 루트 (예: data/raw/kolektorsdd2).",
    )
    build.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="생성할 manifest CSV 경로.",
    )
    build.add_argument(
        "--errors",
        type=Path,
        default=None,
        help="빌드 오류/경고/그룹 통계 JSON 경로. 생략하면 쓰지 않습니다.",
    )

    make = data_subparsers.add_parser(
        "make-splits",
        help="manifest에서 중복 그룹 단위 tune/test 분할을 만듭니다.",
        description=(
            "시드 42로 그룹을 섞고 라벨별 60/40 목표로 탐욕 배정한 뒤 "
            "P_test >= 100을 보장하고 test.csv를 동결합니다. 같은 디렉터리에 "
            "build_errors.json이 있으면 그룹 통계를 보고서에 반영합니다."
        ),
    )
    make.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="입력 manifest CSV 경로.",
    )
    make.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="분할 CSV와 split_report.json을 쓸 디렉터리 (예: data/splits).",
    )
    make.add_argument(
        "--seed",
        type=int,
        default=42,
        help="그룹 셔플 시드 (기본 42).",
    )

    verify = data_subparsers.add_parser(
        "verify",
        help="분할 누출(L1–L8)과 전체 데이터 무결성을 검사합니다.",
        description=(
            "분할 CSV를 다시 읽어 누출을 검사하고, 모든 manifest 행의 "
            "이미지/마스크를 재디코드해 라벨·해시·크기를 검증합니다. "
            "감사 JSON/Markdown과 tune contact sheet를 선택적으로 저장합니다."
        ),
    )
    verify.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="원본 데이터 루트 (예: data/raw/kolektorsdd2).",
    )
    verify.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="검증할 manifest CSV 경로.",
    )
    verify.add_argument(
        "--splits-dir",
        required=True,
        type=Path,
        help="tune.csv/test.csv/official_test.csv가 있는 디렉터리.",
    )
    verify.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="감사 결과 JSON 저장 경로 (선택).",
    )
    verify.add_argument(
        "--audit-md",
        type=Path,
        default=None,
        help="한국어 감사 Markdown 저장 경로 (선택).",
    )
    verify.add_argument(
        "--contact-sheet",
        type=Path,
        default=None,
        help="tune contact sheet PNG 저장 경로 (선택).",
    )


def _add_eval_parser(subparsers: Any) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="tune 표본에서 P/B0·B1·B2, A1–A6, 튜닝 그리드를 실행합니다.",
        description=(
            "tune 분할에서 defect 전수 + good 층화 표본을 골라 기준선(B0/B1/B2), "
            "제안 파이프라인(P), ablation(A1–A6), 튜닝 그리드(최대 24후보)를 "
            "실행하고 지표·실패 색인·보고서를 남깁니다. 모든 비율에 분모와 "
            "Wilson 95% 구간을 붙입니다. test/official_test는 봉인되어 거부합니다."
        ),
    )
    eval_parser.add_argument(
        "--split",
        default="tune",
        help="평가 분할 (tune만 허용, test/official_test는 --allow-sealed 시 T08 측정).",
    )
    eval_parser.add_argument(
        "--allow-sealed",
        action="store_true",
        default=False,
        help="봉인 분할(test/official_test) 일회성 측정을 허용합니다(T08). 미지정 시 거부.",
    )
    eval_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="기본 설정 TOML 경로 (기본 configs/base.toml). 수정하지 않는다.",
    )
    eval_parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="tune.csv가 있는 디렉터리 (기본 data/splits).",
    )
    eval_parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    eval_parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("docs/evidence"),
        help="증거 산출물 디렉터리 (기본 docs/evidence).",
    )
    eval_parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("docs/reports"),
        help="보고서 디렉터리 (기본 docs/reports).",
    )
    eval_parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("configs/candidates"),
        help="추천 후보 config를 쓸 디렉터리 (기본 configs/candidates).",
    )
    eval_parser.add_argument(
        "--good-target",
        type=int,
        default=None,
        help="good 층화 표본 수 (기본 500).",
    )
    eval_parser.add_argument(
        "--no-baselines",
        dest="baselines",
        action="store_false",
        help="B0/B1/B2를 실행하지 않습니다.",
    )
    eval_parser.add_argument(
        "--no-ablations",
        dest="ablations",
        action="store_false",
        help="A1–A6를 실행하지 않습니다.",
    )
    eval_parser.add_argument(
        "--no-pilot",
        dest="pilot",
        action="store_false",
        help="후보(candidates) 축 파일럿을 실행하지 않습니다.",
    )
    eval_parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="캐시를 읽지 않고 새로 만듭니다(같은 키 파일은 덮어쓴다).",
    )
    eval_parser.add_argument(
        "--no-overlays",
        dest="overlays",
        action="store_false",
        help="실패 색인용 오버레이를 만들지 않습니다.",
    )


def _add_recalibrate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "recalibrate",
        help="T04 후보 생성 재보정을 실행합니다(설정 우선, tune 전용).",
        description=(
            "T05가 확인한 후보 생성 병목을 설정 우선으로 재보정합니다. "
            "configs/candidates/recalib/*.toml 후보를 T05와 같은 tune 표본에서 "
            "측정하고 지표·분모·Wilson 95% 구간·절단 통계를 남깁니다. "
            "T05 산출물(tune_grid.jsonl 등)은 덮어쓰지 않습니다. "
            "test/official_test는 봉인되어 거부합니다."
        ),
    )
    parser.add_argument("--split", default="tune", help="평가 분할 (tune만 허용).")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("configs/candidates/recalib"),
        help="후보 TOML 디렉터리 (기본 configs/candidates/recalib).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="tune.csv가 있는 디렉터리 (기본 data/splits).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/derived/t05_cache"),
        help="캐시 디렉터리 (기본 data/derived/t05_cache).",
    )
    parser.add_argument(
        "--good-target",
        type=int,
        default=500,
        help="good 층화 표본 수 (기본 500, T05와 같은 값).",
    )
    parser.add_argument(
        "--seed",
        default="t05-2026-11-10",
        help="표본 추출 seed (기본 t05-2026-11-10).",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("docs/evidence/tune_grid2.jsonl"),
        help="후보별 결과 JSONL (기본 docs/evidence/tune_grid2.jsonl).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/reports/T04_recalib_2026-11-11.md"),
        help="보고서 경로 (기본 docs/reports/T04_recalib_2026-11-11.md).",
    )
    parser.add_argument(
        "--report-title",
        default="T04 후보 생성 재보정 보고서",
        help="보고서 제목 (기본 T04 후보 생성 재보정 보고서).",
    )
    parser.add_argument(
        "--report-cap-note",
        default="(상한 24)",
        help='보고서의 후보 수 꼬리표 (기본 "(상한 24)").',
    )


def _cmd_recalibrate(args: argparse.Namespace) -> int:
    from .evaluation.recalibration import (
        RecalibrationError,
        git_head,
        run_recalibration,
        write_jsonl,
        write_report,
    )
    from .evaluation.runner import load_split_rows
    from .evaluation.sampling import build_sample_plan

    if args.split != "tune":
        _print_error(f"봉인·미지원 분할입니다: {args.split} (tune만 허용)")
        return EXIT_SEALED
    try:
        rows = run_recalibration(
            args.candidate_dir,
            split=args.split,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            cache_dir=args.cache_dir,
            good_target=args.good_target,
            seed=args.seed,
        )
    except RecalibrationError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except FileNotFoundError as exc:
        _print_error(f"산출물을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    plan = build_sample_plan(
        load_split_rows(args.split, splits_dir=args.splits_dir),
        split=args.split,
        good_target=args.good_target,
        seed=args.seed,
    )
    jsonl_path = write_jsonl(rows, args.jsonl)
    report_path = write_report(
        rows,
        args.report,
        plan=plan,
        jsonl_path=args.jsonl.as_posix(),
        seed=args.seed,
        code_revision=git_head(),
        title=args.report_title,
        candidate_dir=args.candidate_dir,
        good_target=args.good_target,
        cap_note=args.report_cap_note,
    )
    passed = [row for row in rows if row.passes()]
    print("")
    print(f"후보 {len(rows)}건 · 통과 {len(passed)}건")
    for row in rows:
        print(
            "{id}: escape {esc} / false-NG {fng} / review {rev} / recall {rec} / {result}".format(
                id=row.candidate_id,
                esc=_ratio_text(row.escape),
                fng=_ratio_text(row.false_ng),
                rev=_ratio_text(row.review),
                rec=_ratio_text(row.recall),
                result="통과" if row.passes() else "미통과",
            )
        )
    print(f"JSONL: {jsonl_path}")
    print(f"보고서: {report_path}")
    _print_sealed_notice()
    return EXIT_OK


def _ratio_text(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _print_sealed_notice() -> None:
    print("sealed test 미실행 (--allow-sealed 미사용)", file=sys.stderr)


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation.runner import (
        SEALED_SPLITS,
        EvalError,
        SealedSplitAccess,
        run_evaluation,
    )

    if args.split in SEALED_SPLITS:
        return _cmd_eval_sealed(args)

    try:
        result = run_evaluation(
            split=args.split,
            config_path=args.config,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            evidence_dir=args.evidence_dir,
            report_dir=args.report_dir,
            candidate_dir=args.candidate_dir,
            good_target=args.good_target,
            ablations=args.ablations,
            baselines=args.baselines,
            pilot=args.pilot,
            use_cache=args.use_cache,
            overlays=args.overlays,
        )
    except SealedSplitAccess as exc:
        _print_error(str(exc))
        return EXIT_SEALED
    except EvalError as exc:
        _print_error(str(exc))
        return int(exc.exit_code)
    except FileNotFoundError as exc:
        _print_error(f"산출물을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    proposal = result.by_name("P")
    print(f"표본: defect {result.plan.defect_taken} (전수) + good {result.plan.good_taken} (층화)")
    print(
        "P: escape {} / false-NG {} / review {}".format(
            proposal.image["defect_escape_rate"].formatted(),
            proposal.image["good_false_ng_rate"].formatted(),
            proposal.image["review_rate"].formatted(),
        )
    )
    for report in result.reports:
        if report.name == "P":
            continue
        print(
            "{}: escape {} / false-NG {}".format(
                report.name,
                report.image["defect_escape_rate"].formatted(),
                report.image["good_false_ng_rate"].formatted(),
            )
        )
    best = result.pilot_best
    if best is not None:
        print(
            f"추천 후보: {result.adopted_label} "
            f"(escape {_pct_text(best.escape)} / false-NG {_pct_text(best.false_ng)}"
            f" / review {_pct_text(best.review)})"
        )
    else:
        grid_best = result.selection.best
        if grid_best is not None:
            print(
                f"그리드 최선: {grid_best.candidate_id} "
                f"(escape {_pct_text(grid_best.escape)} / false-NG "
                f"{_pct_text(grid_best.false_ng)} / review {_pct_text(grid_best.review)})"
            )
    for key, value in result.paths.items():
        print(f"{key}: {value}")
    return EXIT_OK


def _pct_text(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _cmd_eval_sealed(args: argparse.Namespace) -> int:
    """T08 봉인 분할 일회성 측정. --allow-sealed 가 없으면 exit 6 으로 거부한다."""

    from .evaluation.runner import EvalError, SealedSplitAccess
    from .evaluation.sealed import run_sealed_measurement

    if not args.allow_sealed:
        _print_error(
            f"봉인된 split은 --allow-sealed 없이 열 수 없습니다: {args.split}."
        )
        return EXIT_SEALED
    try:
        payload = run_sealed_measurement(
            split=args.split,
            config_path=args.config,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            evidence_dir=args.evidence_dir,
            report_dir=args.report_dir,
            use_cache=args.use_cache,
            overlays=args.overlays,
        )
    except SealedSplitAccess as exc:
        _print_error(str(exc))
        return EXIT_SEALED
    except EvalError as exc:
        _print_error(str(exc))
        return int(exc.exit_code)
    except FileNotFoundError as exc:
        _print_error(f"산출물을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    plan = payload["sample_plan"]
    lock = payload["lock"]
    image = payload["variants"]["P"]["report"]["image_metrics"]
    print(
        f"봉인 분할: {payload['split']} "
        f"(access_count={lock['access_count']}, reuse={str(lock['reuse']).lower()})"
    )
    print(
        f"표본: defect {plan['defect_taken']} (전수) + good {plan['good_taken']} (전수)"
    )
    for key in ("defect_escape_rate", "good_false_ng_rate", "review_rate"):
        ratio = image[key]
        print(f"P {key}: {_pct_text(ratio['value'])} (분모 {ratio['denominator']})")
    for label, goal in payload["goals"].items():
        verdict = (
            "판정 불가" if goal["pass"] is None else ("PASS" if goal["pass"] else "FAIL")
        )
        print(f"목표 {label}: {verdict}")
    for key, value in payload["paths"].items():
        print(f"{key}: {value}")
    return EXIT_OK


def _add_adopted_verify_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "adopted-verify",
        help="채택 캐시의 그룹 부트스트랩 CI와 r18 짝 delta를 계산합니다.",
        description=(
            "채택 config의 tune 캐시를 읽어 escape·false-NG·review·recall의 "
            "이미지 단위 그룹 CI를 내고, r18 캐시를 새 해시로 재빌드해 짝 "
            "부트스트랩 delta를 계산합니다. 봉인 분할은 거부합니다."
        ),
    )
    parser.add_argument("--split", default="tune", help="분석 분할 (tune만 허용).")
    parser.add_argument(
        "--adopted-config",
        type=Path,
        default=Path("configs/candidates/adopted/t05_adopted_r20.toml"),
        help="채택 config TOML.",
    )
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/candidates/fullverify/r18_z7_c16_m5_a45_o3.toml"),
        help="비교 config TOML (기본 r18).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/derived/t05_cache"),
        help="캐시 디렉터리.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="tune.csv가 있는 디렉터리.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트.",
    )
    parser.add_argument(
        "--good-target",
        type=int,
        default=1787,
        help="good 층화 표본 수 (기본 1787, 전수).",
    )
    parser.add_argument(
        "--seed",
        default="t05-2026-11-10",
        help="표본 추출 seed.",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=2000,
        help="부트스트랩 반복 수 (기본 2000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20261111,
        help="부트스트랩 RNG seed (기본 20261111).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/t05_adopted_verify.json"),
        help="결과 JSON 경로.",
    )


def _cmd_adopted_verify(args: argparse.Namespace) -> int:
    from .evaluation.adopted import (
        AdoptedVerifyError,
        build_adopted_verify,
        write_adopted_verify,
    )

    if args.split in {"test", "official_test"}:
        _print_error(f"봉인된 split은 열 수 없습니다: {args.split}")
        return EXIT_SEALED
    if args.split != "tune":
        _print_error(f"지원하지 않는 split입니다: {args.split} (tune만 허용)")
        return EXIT_INPUT
    try:
        payload = build_adopted_verify(
            adopted_config_path=args.adopted_config,
            comparison_config_path=args.comparison_config,
            cache_dir=args.cache_dir,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            good_target=args.good_target,
            seed=args.seed,
            b=args.b,
            bootstrap_seed=args.bootstrap_seed,
        )
        out = write_adopted_verify(payload, args.out)
    except AdoptedVerifyError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except FileNotFoundError as exc:
        _print_error(f"캐시·입력 파일을 찾지 못했습니다: {exc}")
        return EXIT_INPUT
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        _print_error(f"검증 JSON을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    for name, value in payload["group_ci"].items():
        print(
            f"{name}: {value['value']} "
            f"[{value['ci_low']}, {value['ci_high']}] (n={value['n']})"
        )
    for name, value in payload["paired_delta"].items():
        print(
            f"delta {name} (채택-r18): {value['delta']} "
            f"[{value['ci_low']}, {value['ci_high']}] (n={value['n']})"
        )
    print(f"검증 JSON: {out}")
    return EXIT_OK


def _add_fullverify_details_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "fullverify-details",
        help="전수 tune 검증의 상세 분석(escape 원인·REVIEW 사유·IoU 분포)을 만듭니다.",
        description=(
            "전수 측정 JSONL과 후보별 캐시를 읽어 escape 원인, REVIEW 사유, "
            "IoU 분포·recall@0.50, 혼동행렬, 714 표본 대비 delta를 남깁니다. "
            "파이프라인은 escape 샘플만 재실행합니다. 봉인 분할은 거부합니다."
        ),
    )
    parser.add_argument(
        "--split",
        default="tune",
        help="분석 분할 (tune만 허용, 기본 tune).",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("docs/evidence/tune_fullverify.jsonl"),
        help="전수 측정 JSONL (기본 docs/evidence/tune_fullverify.jsonl).",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("configs/candidates/fullverify"),
        help="후보 TOML 디렉터리 (기본 configs/candidates/fullverify).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/derived/t05_cache"),
        help="캐시 디렉터리 (기본 data/derived/t05_cache).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="tune.csv가 있는 디렉터리 (기본 data/splits).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    parser.add_argument(
        "--good-target",
        type=int,
        default=1787,
        help="good 층화 표본 수 (기본 1787, 전수).",
    )
    parser.add_argument(
        "--seed",
        default="t05-2026-11-10",
        help="표본 추출 seed (기본 t05-2026-11-10).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/tune_fullverify_details.json"),
        help="상세 JSON 출력 (기본 docs/evidence/tune_fullverify_details.json).",
    )


def _cmd_fullverify_details(args: argparse.Namespace) -> int:
    from .evaluation.fullverify import FullVerifyError, build_details, write_details

    if args.split in {"test", "official_test"}:
        _print_error(f"봉인된 split은 열 수 없습니다: {args.split}")
        return EXIT_SEALED
    if args.split != "tune":
        _print_error(f"지원하지 않는 split입니다: {args.split} (tune만 허용)")
        return EXIT_INPUT
    try:
        details = build_details(
            jsonl_path=args.jsonl,
            candidate_dir=args.candidate_dir,
            cache_dir=args.cache_dir,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            good_target=args.good_target,
            seed=args.seed,
        )
        out = write_details(details, args.out)
    except FullVerifyError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except FileNotFoundError as exc:
        _print_error(f"캐시·입력 파일을 찾지 못했습니다: {exc}")
        return EXIT_INPUT
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        _print_error(f"상세 JSON을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    print(
        f"후보 {details['totals']['candidates']}건 · "
        f"표본 {details['totals']['samples']}장 "
        f"(defect {details['totals']['defect']} + good {details['totals']['good']})"
    )
    for item in details["candidates"]:
        causes = ", ".join(
            f"{name} {count}" for name, count in item["escape_causes"].items()
        ) or "없음"
        print(
            f"{item['candidate_id']}: escape {len(item['escapes'])} "
            f"({causes}) · REVIEW {item['review_breakdown']['samples']['total']} · "
            f"recall@0.50 {item['metrics']['recall_iou0.50']['value']}"
        )
    print(f"상세 JSON: {out}")
    return EXIT_OK


def _add_inspect_parser(subparsers: Any) -> None:
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="이미지 한 장을 검사하고 오버레이와 결과 JSON을 저장합니다.",
        description=(
            "고정 10단계 파이프라인으로 이미지 한 장을 OK/NG/REVIEW로 "
            "판정하고, 오버레이 PNG와 결과 JSON을 저장합니다. "
            "test/official_test split은 봉인되어 거부합니다."
        ),
    )
    inspect_parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="검사할 이미지 파일 경로.",
    )
    inspect_parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="선택 마스크 PNG 경로. 크기와 라벨 일관성을 검사합니다.",
    )
    inspect_parser.add_argument(
        "--label",
        choices=("unknown", "defect", "good"),
        default="unknown",
        help="입력 라벨 (기본 unknown).",
    )
    inspect_parser.add_argument(
        "--split",
        choices=("tune", "test", "official_test"),
        default="tune",
        help="데이터 분할 이름 (기본 tune). test/official_test는 거부됩니다.",
    )
    inspect_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="설정 TOML 경로 (기본 configs/base.toml).",
    )
    inspect_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/inspect"),
        help="오버레이와 결과 JSON을 저장할 디렉터리 (기본 runs/inspect).",
    )


def _add_batch_parser(subparsers: Any) -> None:
    batch_parser = subparsers.add_parser(
        "batch",
        help="tune 분할을 배치 실행하고 run 디렉터리를 원자적으로 만듭니다.",
        description=(
            "tune 분할을 순차 실행해 runs/<run_id>/ 아래에 run.json, "
            "results.jsonl, summary.csv, summary.json, 설정/manifest 스냅샷, "
            "로그, 오버레이를 남깁니다. 저장은 임시 디렉터리 → fsync → "
            "os.replace 순서이며, 실패하면 최종 디렉터리를 남기지 않습니다. "
            "test/official_test는 봉인되어 거부합니다."
        ),
    )
    batch_parser.add_argument(
        "--split",
        default="tune",
        help="데이터 분할 이름 (tune만 허용, test/official_test는 --allow-sealed 시 허용).",
    )
    batch_parser.add_argument(
        "--allow-sealed",
        action="store_true",
        default=False,
        help="봉인 분할(test/official_test) 일회성 실행을 허용합니다(T08). 미지정 시 거부.",
    )
    batch_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="설정 TOML 경로 (기본 configs/base.toml).",
    )
    batch_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_manifest.csv"),
        help="manifest CSV 경로 (기본 data/manifests/dataset_manifest.csv).",
    )
    batch_parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="tune.csv가 있는 디렉터리 (기본 data/splits).",
    )
    batch_parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="run 디렉터리를 만들 상위 경로 (기본 runs).",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="검증용 샘플 수 제한 (기본 전체).",
    )
    batch_parser.add_argument(
        "--run-id",
        default=None,
        help="run_id 직접 지정 (형식 YYYYMMDDTHHMMSSZ-<config_hash8>-<split>[-NN]).",
    )


def _cmd_batch(args: argparse.Namespace) -> int:
    try:
        outcome = run_batch(
            split=args.split,
            config_path=args.config,
            manifest_path=args.manifest,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            output_dir=args.output_dir,
            limit=args.limit,
            run_id=args.run_id,
            allow_sealed=args.allow_sealed,
        )
    except SealedSplitError as exc:
        _print_error(str(exc))
        return EXIT_SEALED
    except BatchError as exc:
        _print_error(str(exc))
        return int(exc.exit_code)
    except OSError as exc:
        _print_error(f"run 디렉터리를 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE
    except Exception as exc:  # 미완료 run은 저장 실패와 같은 종료 코드를 쓴다
        _print_error(f"배치를 완료하지 못했습니다: {exc}")
        return EXIT_STORAGE

    counts = outcome.counts
    timing = outcome.timing
    print(f"run: {outcome.run_id}")
    print(f"경로: {outcome.run_dir}")
    print(f"상태: {outcome.status.value}")
    print(
        f"샘플 {counts['samples']:,} / OK {counts['ok']:,} / NG {counts['ng']:,} "
        f"/ REVIEW {counts['review']:,} / ERROR {counts['error']:,}"
    )
    print(
        f"샘플 처리 p50 {timing['p50_ms']:.3f} ms / p95 {timing['p95_ms']:.3f} ms "
        f"/ wall {timing['wall_ms']:.3f} ms"
    )
    print(f"오버레이: {counts['overlays']:,}")
    return outcome.exit_code


def _add_defect_map_parser(subparsers: Any) -> None:
    map_parser = subparsers.add_parser(
        "defect-map",
        help="defect 마스크를 누적해 위치 히트맵 PNG와 통계 JSON을 만듭니다.",
        description=(
            "manifest의 defect 행 전체를 읽어 마스크 좌표를 정규화 그리드에 "
            "누적하고 단일 히트맵과 통계를 저장합니다. 판정 파이프라인을 "
            "변경하지 않는 읽기 전용 분석입니다."
        ),
    )
    map_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_manifest.csv"),
        help="manifest CSV 경로 (기본 data/manifests/dataset_manifest.csv).",
    )
    map_parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    map_parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/audit/T03_audit.json"),
        help="비정본 목록이 있는 감사 JSON (기본 data/audit/T03_audit.json).",
    )
    map_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/derived/defect_map"),
        help="산출물 디렉터리 (기본 data/derived/defect_map).",
    )
    map_parser.add_argument(
        "--grid",
        type=int,
        default=DEFAULT_GRID,
        help=f"누적 그리드 크기 (기본 {DEFAULT_GRID}).",
    )


def _cmd_defect_map(args: argparse.Namespace) -> int:
    try:
        rows = load_manifest(args.manifest)
    except ManifestError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    try:
        excluded = load_excluded_ids(args.audit)
        result = accumulate(rows, args.raw_root, excluded, grid=args.grid)
        heatmap_path = args.out_dir / "heatmap.png"
        stats_path = args.out_dir / "stats.json"
        render_heatmap(result.pixel_hits, heatmap_path)
        payload = write_stats(result, args.grid, stats_path)
    except DefectMapError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        _print_error(f"산출물을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE
    top = payload["top_bins"][0] if payload["top_bins"] else None
    top_text = f"y={top['y_pct']}% x={top['x_pct']}%" if top else "-"
    coverage = top["sample_coverage"] if top else 0.0
    print(f"결함 샘플: {result.samples:,} / 결함 픽셀: {result.total_defect_pixels:,}")
    print(f"그리드: {args.grid}x{args.grid} / 채워진 bin: {result.nonzero_bins:,}")
    print(f"최밀집 구역: {top_text} (샘플 커버리지 {coverage})")
    print(f"히트맵: {heatmap_path}")
    print(f"통계: {stats_path}")
    return EXIT_OK


def _add_serve_parser(subparsers: Any) -> None:
    serve_parser = subparsers.add_parser(
        "serve",
        help="로컬 검토 UI를 127.0.0.1에서 실행합니다 (T07).",
        description=(
            "runs/ 디렉터리의 run을 읽는 로컬 UI를 띄웁니다. run 목록, run 요약, "
            "샘플 상세, REVIEW 검토 큐(키보드 J/K), 지표, /healthz를 제공합니다. "
            "127.0.0.1 전용이고 업로드·설정 편집·삭제는 없습니다. 쓰기는 REVIEW "
            "샘플의 검토 결정 append(review_decisions.jsonl) 하나뿐이며 OK/NG "
            "기계 판정은 불변입니다. 외부 CDN·외부 요청을 사용하지 않습니다."
        ),
    )
    serve_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="run 디렉터리가 있는 상위 경로 (기본 runs).",
    )
    serve_parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 이미지 루트 (기본 data/raw/kolektorsdd2). 없으면 원본 없이 표시.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="포트 (기본 8000). 호스트는 127.0.0.1로 고정입니다.",
    )
    serve_parser.add_argument(
        "--with-api",
        action="store_true",
        default=False,
        help="로컬 검사 API(/api/*)와 실시간 검사 워크벤치를 함께 활성화합니다.",
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        _print_error(f"포트 범위가 아닙니다: {args.port} (1–65535)")
        return EXIT_USAGE
    if not args.runs_dir.is_dir():
        _print_error(f"runs 디렉터리가 없습니다: {args.runs_dir}")
        return EXIT_INPUT

    # 서버 의존성은 serve에서만 불러온다 (다른 명령의 시작 비용을 늘리지 않는다).
    import uvicorn

    from .ui import create_app

    app = create_app(runs_dir=args.runs_dir, raw_root=args.raw_root, enable_api=args.with_api)
    print(f"UI: http://127.0.0.1:{args.port} (종료: Ctrl+C)")
    print(f"runs: {args.runs_dir} / 원본: {args.raw_root}")
    if args.with_api:
        print("API: /api/* 및 실시간 검사 워크벤치 활성화")
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    return EXIT_OK


def _add_yolo_parser(subparsers: Any) -> None:
    yolo_parser = subparsers.add_parser(
        "yolo",
        help="YOLO 비교 트랙(ADR 0007): 라벨 export 등.",
        description=(
            "YOLO detection/segmentation 비교를 위한 데이터셋을 준비합니다. "
            "학습은 격리 환경(.venv-train)에서 수행하고 추론은 ONNX로 수행합니다."
        ),
    )
    yolo_subparsers = yolo_parser.add_subparsers(dest="yolo_command")
    export = yolo_subparsers.add_parser(
        "export",
        help="분할 CSV를 YOLO 학습 데이터셋(det/seg)으로 export합니다.",
        description=(
            "tune 분할을 group_id 단위 group-aware train/val로 나누고, GT 마스크에서 "
            "detection bbox와 segmentation 폴리곤 라벨을 만들어 ultralytics 형식으로 "
            "저장합니다. 같은 그룹이 양쪽에 섞이지 않으며 시드로 결정론적입니다."
        ),
    )
    export.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/splits/tune.csv"),
        help="분할 CSV 경로 (기본 data/splits/tune.csv).",
    )
    export.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    export.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/derived/yolo"),
        help="산출물 디렉터리 (기본 data/derived/yolo).",
    )
    export.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="val 비율 (기본 0.2).",
    )
    export.add_argument(
        "--seed",
        type=int,
        default=20261112,
        help="분할 시드 (기본 20261112).",
    )
    export.add_argument(
        "--tasks",
        default="det,seg",
        help="export할 task 목록, 쉼표 구분 (기본 det,seg).",
    )
    export.add_argument(
        "--min-area-px",
        type=int,
        default=1,
        help="라벨로 남길 최소 컴포넌트 면적(px) (기본 1).",
    )
    _add_yolo_evaluate_parser(yolo_subparsers)


def _add_yolo_evaluate_parser(subparsers: Any) -> None:
    evaluate = subparsers.add_parser(
        "evaluate",
        help="학습된 ONNX(det/seg)를 tune-val/tune에서 평가해 r20과 비교합니다.",
        description=(
            "학습된 ONNX 모델로 tune 내부 val(기본) 또는 전량 tune을 평가하고 "
            "r20 기준선과 분모를 붙여 비교합니다. 상태는 decide_from_detections, "
            "영역 IoU는 evaluation.measure의 component_iou를 그대로 씁니다. "
            "산출물은 엔진별 지표 JSON과 이미지별 감사 JSONL, 그리고 "
            "docs/reports/T07_yolo_eval_<date>.md입니다. "
            "test/official_test는 봉인되어 거부합니다."
        ),
    )
    evaluate.add_argument(
        "--task",
        choices=("det", "seg"),
        required=True,
        help="평가할 모델 종류 (det 또는 seg).",
    )
    evaluate.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="ONNX 모델 경로 (예: data/derived/yolo/models/det/weights/best.onnx).",
    )
    evaluate.add_argument(
        "--split",
        default="val",
        help="평가 분할 (val 또는 tune만 허용, 기본 val. test/official_test는 거부).",
    )
    evaluate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/candidates/adopted/t05_adopted_r20.toml"),
        help="기준선 config TOML 경로 (기본 r20 채택본). 수정하지 않는다.",
    )
    evaluate.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="tune.csv가 있는 디렉터리 (기본 data/splits).",
    )
    evaluate.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/kolektorsdd2"),
        help="원본 데이터 루트 (기본 data/raw/kolektorsdd2).",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        default=Path("data/derived/yolo/eval"),
        help="평가 산출물 디렉터리 (기본 data/derived/yolo/eval).",
    )
    evaluate.add_argument(
        "--report",
        type=Path,
        default=None,
        help="보고서 경로 (기본 docs/reports/T07_yolo_eval_<date>.md).",
    )
    evaluate.add_argument(
        "--combine",
        action="store_true",
        help="OpenCV 판정을 함께 돌려 결합(inspect_image + YOLO → combine)을 평가합니다.",
    )
    evaluate.add_argument(
        "--note",
        default="",
        help="보고서 '측정 조건'에 남길 임계값 조정 메모(조정 전/후 수치 등).",
    )


def _cmd_yolo_evaluate(args: argparse.Namespace) -> int:
    from .yolo.evaluate import (
        ALLOWED_SPLITS,
        SEALED_SPLITS,
        YoloEvalError,
        run_yolo_evaluation,
    )

    split = str(args.split)
    if split in SEALED_SPLITS:
        _print_error(f"봉인된 split은 열 수 없습니다: {split} (허용: {list(ALLOWED_SPLITS)})")
        return EXIT_SEALED
    if split not in ALLOWED_SPLITS:
        _print_error(f"허용하지 않는 split입니다: {split} (허용: {list(ALLOWED_SPLITS)})")
        return EXIT_USAGE

    try:
        run = run_yolo_evaluation(
            task=args.task,
            onnx=args.onnx,
            split=split,
            config_path=args.config,
            splits_dir=args.splits_dir,
            raw_root=args.raw_root,
            out_dir=args.out,
            combine_opencv=args.combine,
            report_path=args.report,
            adjustment_note=args.note,
        )
    except YoloEvalError as exc:
        _print_error(str(exc))
        return int(exc.exit_code)
    except (ConfigError, ManifestError) as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        _print_error(f"평가 산출물을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    report = run.report
    print(f"평가: {report.engine} / split {report.split} / 이미지 {run.rows:,}장")
    print(
        f"escape {report.image['defect_escape_rate'].formatted()} / "
        f"false-NG {report.image['good_false_ng_rate'].formatted()} / "
        f"review {report.image['review_rate'].formatted()}"
    )
    print(
        f"recall@0.10 {report.recall_010.formatted()} / "
        f"recall@0.50 {report.recall_050.formatted()}"
    )
    print(f"지표 JSON: {run.json_path}")
    print(f"감사 JSONL: {run.jsonl_path}")
    print(f"보고서: {run.report_path}")
    print("sealed test 미실행 (test/official_test 미사용)", file=sys.stderr)
    return EXIT_OK


def _cmd_yolo_export(args: argparse.Namespace) -> int:
    from .yolo.dataset import TASKS, export_yolo_dataset

    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        _print_error(f"알 수 없는 task입니다: {unknown} (가능: {TASKS})")
        return EXIT_USAGE
    try:
        report = export_yolo_dataset(
            split_csv=args.split_csv,
            raw_root=args.raw_root,
            out_dir=args.out_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            tasks=tasks,
            min_area_px=args.min_area_px,
        )
    except (ValueError, ManifestError) as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        _print_error(f"산출물을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE
    counts = report.split.counts
    print(f"YOLO export 완료: {report.out_dir}")
    print(f"task: {', '.join(report.tasks)} / 시드: {report.split.seed}")
    print(
        f"train {counts['train']['rows']:,}장 (결함 {counts['train']['defect']:,}) "
        f"/ val {counts['val']['rows']:,}장 (결함 {counts['val']['defect']:,})"
    )
    stats = report.label_stats
    print(
        f"컴포넌트: {stats['total_components']:,} "
        f"(면적 최소 {stats['area_px_min']} / 중앙 {stats['area_px_median']} "
        f"/ 최대 {stats['area_px_max']}, bbox 대체 폴리곤 {stats['bbox_fallback_polygons']})"
    )
    print(f"분할 기록: {report.split_json_path}")
    return EXIT_OK


def _run_yolo(args: argparse.Namespace) -> int:
    if args.yolo_command == "export":
        return _cmd_yolo_export(args)
    if args.yolo_command == "evaluate":
        return _cmd_yolo_evaluate(args)
    print("yolo 하위 명령이 필요합니다: export, evaluate", file=sys.stderr)
    return EXIT_USAGE


def _add_records_parser(subparsers: Any) -> None:
    records_parser = subparsers.add_parser(
        "records",
        help="검사 기록 DB(SQLite)를 초기화하고 조회합니다 (E5).",
        description=(
            "검사 기록 DB를 다룹니다. 기계 판정(inspections.status)은 불변이고 "
            "사람 판정은 review_decisions에 append-only로만 쌓입니다. 같은 "
            "이미지(sha256)는 한 행만 갖습니다. 모든 비율은 분모를 함께 출력합니다."
        ),
    )
    records_subparsers = records_parser.add_subparsers(dest="records_command")

    init = records_subparsers.add_parser(
        "init",
        help="DB 파일과 스키마를 만듭니다(여러 번 실행해도 안전).",
        description="schema.sql을 실행해 inspections/review_decisions와 인덱스를 만듭니다.",
    )
    init.add_argument(
        "--db",
        type=Path,
        default=Path("data/derived/records.db"),
        help="DB 파일 경로 (기본 data/derived/records.db).",
    )

    listing = records_subparsers.add_parser(
        "list",
        help="검사 기록을 id 오름차순으로 조회합니다.",
        description="상태로 거를 수 있고 limit/offset으로 페이지를 넘깁니다.",
    )
    listing.add_argument(
        "--db",
        type=Path,
        default=Path("data/derived/records.db"),
        help="DB 파일 경로 (기본 data/derived/records.db).",
    )
    listing.add_argument(
        "--status",
        choices=("OK", "NG", "REVIEW", "ERROR"),
        default=None,
        help="기계 판정 필터 (기본 전체).",
    )
    listing.add_argument("--limit", type=int, default=20, help="최대 행 수 (기본 20).")
    listing.add_argument("--offset", type=int, default=0, help="건너뛸 행 수 (기본 0).")
    listing.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="행 전체를 JSON으로 출력합니다.",
    )

    stats_parser = records_subparsers.add_parser(
        "stats",
        help="상태 분포·리뷰·오버레이 통계를 분모와 함께 출력합니다.",
        description=(
            "상태별 건수와 비율, 사람 판정 수, 기계 판정 일치율, 오버레이 보관율을 "
            "JSON으로 출력합니다. 비율은 분모(n)를 함께 담고 분모가 0이면 값이 null입니다."
        ),
    )
    stats_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/derived/records.db"),
        help="DB 파일 경로 (기본 data/derived/records.db).",
    )

    verify_parser = records_subparsers.add_parser(
        "verify",
        help="기록 DB 무결성과 append-only 트리거를 검사합니다.",
        description=(
            "PRAGMA integrity_check·PRAGMA foreign_key_check·필수 append-only "
            "트리거 8개 존재를 검사합니다. 정상이면 PASS(exit 0), "
            "무결성/외래키/트리거 위반이면 FAIL(exit 5)입니다. 읽기 전용입니다."
        ),
    )
    verify_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/derived/records.db"),
        help="DB 파일 경로 (기본 data/derived/records.db).",
    )


def _require_db_file(db_path: Path) -> bool:
    if not db_path.is_file():
        _print_error(f"기록 DB가 없습니다: {db_path} (먼저 records init)")
        return False
    return True


def _cmd_records_init(args: argparse.Namespace) -> int:
    from .records.db import DatabaseError, connect, init_db

    connection = None
    try:
        connection = connect(args.db)
        init_db(connection)
    except DatabaseError as exc:
        _print_error(f"기록 DB를 만들지 못했습니다: {exc}")
        return EXIT_STORAGE
    except OSError as exc:
        _print_error(f"기록 DB를 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE
    finally:
        if connection is not None:
            connection.close()
    print(f"기록 DB 준비 완료: {args.db}")
    print("테이블: inspections, review_decisions")
    return EXIT_OK


def _cmd_records_list(args: argparse.Namespace) -> int:
    from .records.db import DatabaseError, connect_readonly
    from .records.store import RecordsError, list_inspections

    if not _require_db_file(args.db):
        return EXIT_INPUT
    connection = None
    try:
        connection = connect_readonly(args.db)
        rows = list_inspections(
            connection,
            status=args.status,
            limit=args.limit,
            offset=args.offset,
        )
    except RecordsError as exc:
        _print_error(str(exc))
        return EXIT_USAGE
    except DatabaseError as exc:
        _print_error(f"기록 DB 오류: {exc}")
        return EXIT_STORAGE
    finally:
        if connection is not None:
            connection.close()

    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return EXIT_OK
    print(f"검사 기록 {len(rows)}건 (limit {args.limit} / offset {args.offset})")
    for row in rows:
        print(
            f"{row['id']:>4} {row['sample_id']:<24} {row['status']:<7} "
            f"score {float(row['score']):.4f} {row['engine']:<9} "
            f"{row['overlay_path'] or '-'}"
        )
    return EXIT_OK


def _cmd_records_stats(args: argparse.Namespace) -> int:
    from .records.db import DatabaseError, connect_readonly
    from .records.store import stats_payload

    if not _require_db_file(args.db):
        return EXIT_INPUT
    connection = None
    try:
        connection = connect_readonly(args.db)
        payload = stats_payload(connection)
    except DatabaseError as exc:
        _print_error(f"기록 DB 오류: {exc}")
        return EXIT_STORAGE
    finally:
        if connection is not None:
            connection.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_records_verify(args: argparse.Namespace) -> int:
    from .records.db import DatabaseError, connect_readonly, verify_records_integrity

    if not _require_db_file(args.db):
        return EXIT_INPUT
    connection = None
    try:
        connection = connect_readonly(args.db)
        report = verify_records_integrity(connection)
    except DatabaseError as exc:
        _print_error(f"기록 DB 오류: {exc}")
        return EXIT_STORAGE
    finally:
        if connection is not None:
            connection.close()

    print(f"대상: {args.db}")
    print(f"integrity_check: {report.integrity_check}")
    print(f"foreign_key_check: 위반 {report.foreign_key_violations}건")
    trigger_total = 8
    missing = report.missing_triggers
    print(f"append-only 트리거: {trigger_total - len(missing)}/{trigger_total} 존재")
    for name in missing:
        print(f"  누락: {name}")
    if report.ok:
        print("PASS: 기록 DB 무결성·append-only 트리거 정상")
        return EXIT_OK
    _print_error("FAIL: 기록 DB 무결성 검사에 실패했습니다 (records init 으로 트리거 재설치)")
    return EXIT_STORAGE


def _run_records(args: argparse.Namespace) -> int:
    if args.records_command == "init":
        return _cmd_records_init(args)
    if args.records_command == "list":
        return _cmd_records_list(args)
    if args.records_command == "stats":
        return _cmd_records_stats(args)
    if args.records_command == "verify":
        return _cmd_records_verify(args)
    print(
        "records 하위 명령이 필요합니다: init, list, stats, verify",
        file=sys.stderr,
    )
    return EXIT_USAGE


def _add_worker_parser(subparsers: Any) -> None:
    worker_parser = subparsers.add_parser(
        "worker",
        help="incoming 디렉터리를 감시해 검사하고 기록합니다 (E5, ADR D6).",
        description=(
            "incoming의 이미지를 이름 정렬 순으로 검사하고 DB에 기록합니다. 같은 "
            "sha256은 한 번만 검사하고, 실패는 재시도 후 ERROR로 기록하며, 처리/스킵/"
            "에러를 .work/worker_audit.jsonl에 append-only로 남깁니다. 네트워크·"
            "셸을 쓰지 않고 판정을 바꾸지 않습니다. 봉인 분할은 열지 않습니다."
        ),
    )
    worker_subparsers = worker_parser.add_subparsers(dest="worker_command")
    run = worker_subparsers.add_parser(
        "run",
        help="워커를 실행합니다(--once면 한 바퀴만).",
        description="incoming을 감시해 검사·기록합니다.",
    )
    run.add_argument(
        "--incoming",
        type=Path,
        default=Path("data/incoming"),
        help="감시할 디렉터리 (기본 data/incoming, 없으면 만든다).",
    )
    run.add_argument(
        "--db",
        type=Path,
        default=Path("data/derived/records.db"),
        help="기록 DB 경로 (기본 data/derived/records.db).",
    )
    run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="설정 TOML 경로 (기본 configs/base.toml).",
    )
    run.add_argument(
        "--engine",
        default="opencv",
        help="엔진 (opencv/yolo-det/yolo-seg/combined, 기본 opencv).",
    )
    run.add_argument(
        "--once",
        action="store_true",
        help="현재 파일을 한 바퀴 처리하고 종료합니다.",
    )
    run.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="검사 ERROR 시 재시도 횟수 (기본 3, 최소 1회는 시도).",
    )
    run.add_argument(
        "--ok-sample-rate",
        type=int,
        default=1,
        help="OK 오버레이 보관 비율 1/N (기본 1=전량, 0=보관 안 함).",
    )
    run.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="감시 주기 초 (기본 2.0, --once면 무시).",
    )
    recheck = worker_subparsers.add_parser(
        "recheck",
        help="ERROR로만 기록된 이미지를 새 provenance 태그로 재검사합니다(append-only).",
        description=(
            "기존 ERROR 행을 수정·삭제하지 않고 recheck 태그가 반영된 새 행을 추가합니다. "
            "이미 OK/NG/REVIEW 행이 있는 이미지는 건너뜁니다."
        ),
    )
    recheck.add_argument(
        "--incoming",
        type=Path,
        default=Path("data/incoming"),
        help="재검사 대상 이미지 디렉터리 (기본 data/incoming).",
    )
    recheck.add_argument(
        "--db",
        type=Path,
        default=Path("data/derived/records.db"),
        help="기록 DB 경로 (기본 data/derived/records.db).",
    )
    recheck.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="설정 TOML 경로 (기본 configs/base.toml).",
    )
    recheck.add_argument(
        "--engine",
        default="opencv",
        help="엔진 (opencv/yolo-det/yolo-seg/combined, 기본 opencv).",
    )
    recheck.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="재검사 ERROR 시 최대 시도 횟수 (기본 3, 최소 1회는 시도).",
    )
    recheck.add_argument(
        "--ok-sample-rate",
        type=int,
        default=1,
        help="OK 오버레이 보관 비율 1/N (기본 1=전량, 0=보관 안 함).",
    )
    recheck.add_argument(
        "--tag",
        default=None,
        help="재검사 provenance 태그(생략 시 ERROR 건수 기반 결정론적 자동 부여).",
    )


def _cmd_worker_run(args: argparse.Namespace) -> int:
    from .agent.worker import WorkerError, audit_log_path, run_worker
    from .config import ConfigError, load_config
    from .records.db import DatabaseError, connect
    from .records.store import stats_payload

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        _print_error(f"설정을 불러오지 못했습니다: {exc}")
        return EXIT_USAGE

    before = 0
    connection = None
    try:
        if args.db.is_file():
            connection = connect(args.db)
            before = int(stats_payload(connection)["total"])
            connection.close()
            connection = None
        run_worker(
            incoming=args.incoming,
            db_path=args.db,
            config=config,
            engine=args.engine,
            max_retries=args.max_retries,
            ok_sample_rate=args.ok_sample_rate,
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
        connection = connect(args.db)
        payload = stats_payload(connection)
    except WorkerError as exc:
        _print_error(str(exc))
        return EXIT_USAGE
    except DatabaseError as exc:
        _print_error(f"기록 DB 오류: {exc}")
        return EXIT_STORAGE
    except OSError as exc:
        _print_error(f"기록을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE
    finally:
        if connection is not None:
            connection.close()

    print(
        "처리: 신규 {}건 / 총 {}건".format(payload["total"] - before, payload["total"])
    )
    print("상태: " + ", ".join(f"{name} {count}" for name, count in payload["by_status"].items()))
    print(f"감사 로그: {audit_log_path(args.db)}")
    print(f"기록: {args.db}")
    return EXIT_OK


def _cmd_worker_recheck(args: argparse.Namespace) -> int:
    from .agent.worker import WorkerError, audit_log_path, recheck_errors
    from .config import ConfigError, load_config
    from .records.db import DatabaseError

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        _print_error(f"설정을 불러오지 못했습니다: {exc}")
        return EXIT_USAGE

    try:
        counts = recheck_errors(
            incoming=args.incoming,
            db_path=args.db,
            config=config,
            engine=args.engine,
            max_retries=args.max_retries,
            ok_sample_rate=args.ok_sample_rate,
            tag=args.tag,
        )
    except WorkerError as exc:
        _print_error(str(exc))
        return EXIT_USAGE
    except DatabaseError as exc:
        _print_error(f"기록 DB 오류: {exc}")
        return EXIT_STORAGE
    except OSError as exc:
        _print_error(f"기록을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    print(
        f"재검사: {counts['rechecked']}건 "
        f"(복구 {counts['resolved']} · 재오류 {counts['still_error']} · "
        f"건너뜀 {counts['skipped']} · 읽기오류 {counts['error']})"
    )
    print(f"감사 로그: {audit_log_path(args.db)}")
    print(f"기록: {args.db}")
    return EXIT_OK


def _run_worker_command(args: argparse.Namespace) -> int:
    if args.worker_command == "run":
        return _cmd_worker_run(args)
    if args.worker_command == "recheck":
        return _cmd_worker_recheck(args)
    print("worker 하위 명령이 필요합니다: run, recheck", file=sys.stderr)
    return EXIT_USAGE


def _add_agent_parser(subparsers: Any) -> None:
    agent_parser = subparsers.add_parser(
        "agent",
        help="제한된 결정론 에이전트를 실행/리플레이합니다 (T09, 설계 §6).",
        description=(
            "네트워크·셸·파일 쓰기·설정 변경 도구 없이 결정론적 정책 엔진으로 "
            "검사를 보조합니다. 판정은 항상 파이프라인이 소유하며 예산(행동 3회, "
            "프로파일 2개, 추가 지연 5,000ms) 소진은 REVIEW로 끝납니다. 봉인 분할은 "
            "열지 않습니다."
        ),
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")
    run = agent_subparsers.add_parser(
        "run",
        help="이미지 하나에 에이전트 에피소드를 실행합니다.",
        description="에이전트 에피소드를 실행하고 agent_trace.jsonl을 남깁니다.",
    )
    run.add_argument("--image", type=Path, required=True, help="검사할 이미지 경로.")
    run.add_argument("--mask", type=Path, default=None, help="GT 마스크 경로(선택).")
    run.add_argument("--label", default="unknown", help="라벨 (기본 unknown).")
    run.add_argument("--split", default="tune", help="분할 (기본 tune, 봉인 분할 거부).")
    run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="base 설정 TOML 경로 (기본 configs/base.toml).",
    )
    run.add_argument(
        "--agent-config",
        type=Path,
        default=Path("configs/agent.toml"),
        help="에이전트 설정 TOML 경로 (기본 configs/agent.toml).",
    )
    run.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("configs/profiles"),
        help="프로파일 디렉터리 (기본 configs/profiles).",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/agent"),
        help="run 디렉터리 부모 (기본 runs/agent).",
    )
    replay = agent_subparsers.add_parser(
        "replay",
        help="agent_trace.jsonl을 결정론 리플레이합니다.",
        description="기록된 결과만으로 정책을 재실행해 동일 행동/판정을 검증합니다.",
    )
    replay.add_argument("--trace", type=Path, required=True, help="agent_trace.jsonl 경로.")
    audit_cmd = agent_subparsers.add_parser(
        "audit",
        help="run 디렉터리·기록 DB·저장소 하드 게이트를 감사합니다.",
        description="Phase 3 Watchdog 감사기로 산출물·DB·하드 게이트 무결성을 점검합니다.",
    )
    audit_cmd.add_argument("--run-dir", type=Path, default=None, help="감사할 run 디렉터리.")
    audit_cmd.add_argument("--db", type=Path, default=None, help="감사할 검사 기록 DB.")
    audit_cmd.add_argument(
        "--gates", action="store_true", help="저장소 하드 게이트도 점검합니다."
    )
    audit_cmd.add_argument(
        "--repo-root", type=Path, default=Path("."), help="저장소 루트 (기본 .)."
    )
    audit_cmd.add_argument(
        "--json", dest="json_out", type=Path, default=None, help="결과 JSON 경로."
    )

    drift_cmd = agent_subparsers.add_parser(
        "drift",
        help="기준선 대비 대상 배치의 PSI 분포 이동을 감지합니다.",
        description=(
            "점수·화질·상태 분포의 PSI를 계산해 "
            "NORMAL/WATCH/DRIFT_ALERT/INSUFFICIENT_SAMPLE을 보고합니다."
        ),
    )
    drift_cmd.add_argument("--baseline", type=Path, required=True, help="기준선 JSONL.")
    drift_cmd.add_argument("--target", type=Path, required=True, help="대상 JSONL.")
    drift_cmd.add_argument(
        "--min-samples", type=int, default=20, help="최소 표본 수 (기본 20)."
    )
    drift_cmd.add_argument(
        "--json", dest="json_out", type=Path, default=None, help="결과 JSON 경로."
    )

    triage_cmd = agent_subparsers.add_parser(
        "triage",
        help="results.jsonl의 REVIEW 샘플을 기하+점수 기반으로 우선순위 정렬합니다.",
        description=(
            "기존 results.jsonl에서 약후보 기하·물리 신호를 종합해 "
            "P0~P3 검토 큐를 생성합니다."
        ),
    )
    triage_cmd.add_argument("--results", type=Path, required=True, help="results.jsonl 경로.")
    triage_cmd.add_argument("--top-k", type=int, default=20, help="상위 출력 건수 (기본 20).")
    triage_cmd.add_argument(
        "--all",
        dest="include_all",
        action="store_true",
        help="REVIEW 외 전체 상태를 포함해 정렬합니다.",
    )
    triage_cmd.add_argument(
        "--json", dest="json_out", type=Path, default=None, help="결과 JSON 경로."
    )

    explain_cmd = agent_subparsers.add_parser(
        "explain",
        help="특정 샘플의 닫힌 형태 반사실적 경계와 Leave-One-Out 귀속도를 계산합니다.",
        description=(
            "results.jsonl 에서 sample_id 를 찾아 stages.py 점수 공식을 역산한 "
            "닫힌 형태 반사실 경계(필요 면적/대비, 도달 불능 증명)와 Leave-One-Out "
            "귀속도를 출력합니다. 기계 판정과 설정은 변경하지 않습니다."
        ),
    )
    explain_cmd.add_argument("--results", type=Path, required=True, help="results.jsonl 경로.")
    explain_cmd.add_argument("--sample-id", required=True, help="분석할 sample_id.")
    explain_cmd.add_argument(
        "--config",
        type=Path,
        default=None,
        help="설정 TOML 경로 (생략 시 기본 설정 사용).",
    )
    explain_cmd.add_argument(
        "--json", dest="json_out", type=Path, default=None, help="결과 JSON 경로."
    )

    calibrate_cmd = agent_subparsers.add_parser(
        "calibrate",
        help="run 디렉터리의 사람 검토 이력과 에이전트 우선순위를 캘리브레이션합니다.",
        description=(
            "Phase 4 observe-only 분석으로 review_decisions.jsonl 과 에이전트 P0~P3 "
            "우선순위를 대조해 정합성·하드 샘플 큐·엔지니어링 권고를 출력합니다."
        ),
    )
    calibrate_cmd.add_argument("--run-dir", type=Path, required=True, help="분석할 run 디렉터리.")
    calibrate_cmd.add_argument(
        "--json", dest="json_out", type=Path, default=None, help="결과 JSON 경로."
    )


def _cmd_agent_run(args: argparse.Namespace) -> int:
    from .agent.config import load_agent_config
    from .agent.run import AgentRunError, load_profiles, run_agent_episode

    try:
        agent_config = load_agent_config(args.agent_config)
        profiles = load_profiles(args.config, args.profiles_dir)
    except ConfigError as exc:
        _print_error(f"설정을 불러오지 못했습니다: {exc}")
        return EXIT_USAGE

    try:
        outcome = run_agent_episode(
            args.image,
            agent_config=agent_config,
            profiles=profiles,
            output_dir=args.output_dir,
            label=args.label,
            split=args.split,
            mask_path=args.mask,
        )
    except AgentRunError as exc:
        _print_error(str(exc))
        if exc.code == "E_SEALED_DENIED":
            return EXIT_SEALED
        if exc.code == "E_INPUT_MISSING":
            return EXIT_INPUT
        return EXIT_USAGE
    except OSError as exc:
        _print_error(f"run 디렉터리를 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    episode = outcome.episode
    print(
        json.dumps(
            {
                "run_id": outcome.run_id,
                "terminal_state": episode.terminal_state.value,
                "final_status": episode.final_status.value,
                "reason": episode.reason,
                "actions_used": episode.actions_used,
                "profiles_used": episode.profiles_used,
                "base_latency_ms": episode.base_latency_ms,
                "extra_latency_ms": episode.extra_latency_ms,
                "rejections": len(episode.rejections),
                "trace_path": str(outcome.trace_path),
                "result_path": str(outcome.result_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return EXIT_ERROR if episode.final_status is Status.ERROR else EXIT_OK


def _cmd_agent_replay(args: argparse.Namespace) -> int:
    from .agent.trace import TraceError, load_trace, replay_trace

    try:
        events = load_trace(args.trace)
    except TraceError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    report = replay_trace(events)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return EXIT_OK if report.deterministic else EXIT_GATE


def _add_gate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "gate",
        help="설계 계약 하드 게이트와 run/DB 무결성을 감사하고 종료 코드로 판정합니다.",
        description=(
            "동결 설정·에이전트 설정·데이터 누출(T03)·봉인 잠금·증거 분모를 "
            "점검하는 Watchdog 게이트입니다. 모든 감사가 PASS이면 0, REVISE/REJECT가 "
            "하나라도 있으면 종료 코드 7을 반환합니다."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="저장소 루트 경로 (기본 .).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="함께 감사할 배치 run 디렉터리 경로 (선택).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="함께 감사할 검사 기록 DB 경로 (선택).",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="감사 결과 JSON을 저장할 경로 (선택).",
    )


def _print_audit_report(report: Any) -> None:
    print(f"[{report.verdict.value}] {report.summary}")
    for check in report.checks:
        tag = "PASS" if check.passed else f"FAIL/{check.severity}"
        print(f"  - [{tag}] {check.name}: {check.detail}")


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cmd_gate(args: argparse.Namespace) -> int:
    from .agent.audit import (
        AuditVerdict,
        audit_records_db,
        audit_repository_gates,
        audit_run_dir,
    )

    reports = [audit_repository_gates(args.repo_root)]
    if args.run_dir is not None:
        reports.append(audit_run_dir(args.run_dir))
    if args.db is not None:
        reports.append(audit_records_db(args.db))

    for report in reports:
        _print_audit_report(report)

    verdicts = {report.verdict for report in reports}
    if AuditVerdict.REJECT in verdicts:
        overall = AuditVerdict.REJECT
    elif AuditVerdict.REVISE in verdicts:
        overall = AuditVerdict.REVISE
    else:
        overall = AuditVerdict.PASS

    if args.json_out is not None:
        try:
            _write_json_file(
                args.json_out,
                {
                    "overall_verdict": overall.value,
                    "reports": [report.to_dict() for report in reports],
                },
            )
        except OSError as exc:
            _print_error(f"감사 결과 JSON을 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE
        print(f"감사 JSON: {args.json_out}")

    return EXIT_OK if overall is AuditVerdict.PASS else EXIT_GATE


def _add_repro_parser(subparsers: Any) -> None:
    repro_parser = subparsers.add_parser(
        "repro",
        help="에이전트 트레이스와 배치 실행의 결정론적 재현성을 검증합니다.",
        description=(
            "agent_trace.jsonl 의 감사·리플레이 결정론성 또는 두 run 디렉터리의 "
            "판정·점수·설정 해시 동일성을 검증합니다."
        ),
    )
    repro_subparsers = repro_parser.add_subparsers(dest="repro_command")
    check = repro_subparsers.add_parser(
        "check",
        help="트레이스 리플레이 또는 두 run 결과를 대조 검증합니다.",
        description=(
            "--trace 또는 --run-a/--run-b 중 최소 하나를 지정해야 합니다. "
            "일치하면 0, 불일치·오류가 있으면 상세 원인을 출력하고 7을 반환합니다."
        ),
    )
    check.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="검증할 agent_trace.jsonl 경로 (선택).",
    )
    check.add_argument(
        "--run-a",
        type=Path,
        default=None,
        help="비교할 첫 번째 run 디렉터리 경로 (선택, --run-b 와 함께 사용).",
    )
    check.add_argument(
        "--run-b",
        type=Path,
        default=None,
        help="비교할 두 번째 run 디렉터리 경로 (선택, --run-a 와 함께 사용).",
    )


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL 파일을 찾을 수 없습니다: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"JSONL 파일을 읽지 못했습니다({path}): {exc}") from exc
    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} {line_no}번째 줄 JSON 파싱 오류: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path.name} {line_no}번째 줄이 JSON 객체가 아닙니다")
        items.append(obj)
    return items


def _compare_run_dirs(run_a: Path, run_b: Path) -> tuple[bool, list[str], int]:
    from .io.artifacts import RESULTS_JSONL, RUN_JSON

    errors: list[str] = []
    try:
        rec_a = _load_jsonl_dicts(run_a / RESULTS_JSONL)
        rec_b = _load_jsonl_dicts(run_b / RESULTS_JSONL)
    except ValueError as exc:
        return False, [str(exc)], 0

    run_json_a = run_a / RUN_JSON
    run_json_b = run_b / RUN_JSON
    if run_json_a.is_file() and run_json_b.is_file():
        try:
            meta_a = json.loads(run_json_a.read_text(encoding="utf-8"))
            meta_b = json.loads(run_json_b.read_text(encoding="utf-8"))
            if (
                isinstance(meta_a, dict)
                and isinstance(meta_b, dict)
                and meta_a.get("config_hash") != meta_b.get("config_hash")
            ):
                ha = meta_a.get("config_hash")
                hb = meta_b.get("config_hash")
                errors.append(f"run.json config_hash 불일치: run_a={ha!r} != run_b={hb!r}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"run.json 읽기 오류: {exc}")

    map_a: dict[str, dict[str, Any]] = {}
    map_b: dict[str, dict[str, Any]] = {}
    for label_run, recs, target in (("run-a", rec_a, map_a), ("run-b", rec_b, map_b)):
        for idx, rec in enumerate(recs, start=1):
            sid = rec.get("sample_id")
            if not isinstance(sid, str) or not sid:
                errors.append(f"{label_run} {idx}번째 행에 sample_id가 없습니다")
                continue
            if sid in target:
                errors.append(f"{label_run}에 중복 sample_id가 있습니다: {sid}")
                continue
            target[sid] = rec

    missing_in_b = sorted(set(map_a) - set(map_b))
    extra_in_b = sorted(set(map_b) - set(map_a))
    if missing_in_b:
        errors.append(
            f"run-b에 누락된 샘플 {len(missing_in_b)}건: {', '.join(missing_in_b[:5])}"
        )
    if extra_in_b:
        errors.append(
            f"run-b에 추가된 샘플 {len(extra_in_b)}건: {', '.join(extra_in_b[:5])}"
        )

    for sid in sorted(set(map_a) & set(map_b)):
        a = map_a[sid]
        b = map_b[sid]
        if str(a.get("status")) != str(b.get("status")):
            errors.append(
                f"{sid}: status 불일치 (run-a={a.get('status')!r} != run-b={b.get('status')!r})"
            )
        try:
            score_a = round(float(a.get("score", 0.0)), 6)
            score_b = round(float(b.get("score", 0.0)), 6)
            if score_a != score_b:
                errors.append(f"{sid}: score 불일치 (run-a={score_a:.6f} != run-b={score_b:.6f})")
        except (TypeError, ValueError):
            errors.append(f"{sid}: score 숫자 변환 실패")
        if str(a.get("config_hash", "")) != str(b.get("config_hash", "")):
            errors.append(
                f"{sid}: config_hash 불일치 "
                f"(run-a={a.get('config_hash')!r} != run-b={b.get('config_hash')!r})"
            )

    return not errors, errors, len(map_a)


def _cmd_repro_check(args: argparse.Namespace) -> int:
    from .agent.trace import TraceError, load_trace, replay_trace

    if (args.run_a is None) != (args.run_b is None):
        _print_error("--run-a 와 --run-b 는 함께 지정해야 합니다.")
        return EXIT_USAGE
    if args.trace is None and args.run_a is None:
        _print_error("--trace 또는 --run-a/--run-b 중 최소 하나를 지정해야 합니다.")
        return EXIT_USAGE

    all_ok = True
    if args.trace is not None:
        try:
            events = load_trace(args.trace)
        except TraceError as exc:
            _print_error(f"트레이스 검증 실패({args.trace}): {exc}")
            return EXIT_GATE
        report = replay_trace(events)
        end_event = events[-1] if events else {}
        end_match = (
            report.terminal_state == str(end_event.get("terminal_state", ""))
            and report.final_status == str(end_event.get("final_status", ""))
        )
        if report.audit_valid and report.deterministic and end_match:
            print(
                f"트레이스 재현성 확인 완료: {args.trace} "
                f"(이벤트 {report.events_compared}건, "
                f"종료 {report.terminal_state}/{report.final_status})"
            )
        else:
            all_ok = False
            _print_error(f"트레이스 재현성 검증 실패: {args.trace}")
            for err in report.audit_errors:
                print(f"  - [감사 오류] {err}", file=sys.stderr)
            for mm in report.mismatches:
                print(f"  - [불일치] {mm}", file=sys.stderr)
            if not end_match:
                print(
                    f"  - [종료 상태 불일치] 기록={end_event.get('terminal_state')}/"
                    f"{end_event.get('final_status')} != "
                    f"재생={report.terminal_state}/{report.final_status}",
                    file=sys.stderr,
                )

    if args.run_a is not None and args.run_b is not None:
        matched, errors, compared = _compare_run_dirs(args.run_a, args.run_b)
        if matched:
            print(
                f"배치 재현성 확인 완료: {args.run_a} == {args.run_b} (샘플 {compared}건 일치)"
            )
        else:
            all_ok = False
            _print_error(f"배치 재현성 검증 실패: {args.run_a} vs {args.run_b}")
            for err in errors[:20]:
                print(f"  - {err}", file=sys.stderr)

    return EXIT_OK if all_ok else EXIT_GATE


def _run_repro(args: argparse.Namespace) -> int:
    if args.repro_command == "check":
        return _cmd_repro_check(args)
    print("repro 하위 명령이 필요합니다: check", file=sys.stderr)
    return EXIT_USAGE


def _sample_result_from_record(rec: dict[str, Any]) -> Any:
    from .contracts import (
        ComponentFeature,
        ErrorCode,
        QualityInfo,
        ReasonCode,
        RoiInfo,
        SampleResult,
        Stage,
    )

    raw_comps = rec.get("components") or []
    components = tuple(
        ComponentFeature(
            index=int(c.get("index", idx)),
            area_px=int(c.get("area_px", 0)),
            bbox=tuple(int(x) for x in c.get("bbox", (0, 0, 0, 0))),
            area_ratio=float(c.get("area_ratio", 0.0)),
            aspect_ratio=float(c.get("aspect_ratio", 1.0)),
            elongation=float(c.get("elongation", 1.0)),
            solidity=float(c.get("solidity", 1.0)),
            mean_contrast_gray=float(c.get("mean_contrast_gray", 0.0)),
            max_contrast_gray=float(c.get("max_contrast_gray", 0.0)),
            integrated_contrast=float(c.get("integrated_contrast", 0.0)),
            border_distance_px=float(c.get("border_distance_px", 100.0)),
            touches_roi_border=bool(c.get("touches_roi_border", False)),
            evidence=float(c.get("evidence", 0.0)),
        )
        for idx, c in enumerate(raw_comps)
    )
    q = rec.get("quality")
    quality = (
        QualityInfo(
            blur_variance=float(q.get("blur_variance", 0.0)),
            mean_gray=float(q.get("mean_gray", 0.0)),
            saturated_ratio=float(q.get("saturated_ratio", 0.0)),
            flags=tuple(ReasonCode(val) for val in q.get("flags", [])),
        )
        if isinstance(q, dict)
        else None
    )
    r = rec.get("roi")
    roi = (
        RoiInfo(
            mode=str(r.get("mode", "auto")),
            bbox=tuple(int(x) for x in r.get("bbox", (0, 0, 0, 0))),
            area_px=int(r.get("area_px", 0)),
            area_ratio=float(r.get("area_ratio", 0.0)),
            uncertain=bool(r.get("uncertain", False)),
        )
        if isinstance(r, dict)
        else None
    )
    return SampleResult(
        run_id=str(rec.get("run_id", "")),
        sample_id=str(rec.get("sample_id", "")),
        original_id=str(rec.get("original_id", rec.get("sample_id", ""))),
        split=str(rec.get("split", "tune")),
        label=str(rec.get("label", "unknown")),
        status=Status(str(rec.get("status", "REVIEW"))),
        score=float(rec.get("score", 0.0)),
        reasons=tuple(ReasonCode(val) for val in rec.get("reasons", [])),
        components=components,
        quality=quality,
        roi=roi,
        timing_ms={str(k): float(v) for k, v in (rec.get("timing_ms") or {}).items()},
        error_code=ErrorCode(rec["error_code"]) if rec.get("error_code") else None,
        error_stage=Stage(rec["error_stage"]) if rec.get("error_stage") else None,
        input_sha256=str(rec.get("input_sha256", "")),
        mask_sha256=str(rec.get("mask_sha256", "")),
        overlay_path=str(rec.get("overlay_path", "")),
        config_hash=str(rec.get("config_hash", "")),
    )


def _cmd_agent_audit(args: argparse.Namespace) -> int:
    from .agent.audit import (
        AuditVerdict,
        audit_records_db,
        audit_repository_gates,
        audit_run_dir,
    )

    if args.run_dir is None and args.db is None and not args.gates:
        _print_error("--run-dir, --db, --gates 중 최소 하나를 지정해야 합니다.")
        return EXIT_USAGE

    reports = []
    if args.gates:
        reports.append(audit_repository_gates(args.repo_root))
    if args.run_dir is not None:
        reports.append(audit_run_dir(args.run_dir))
    if args.db is not None:
        reports.append(audit_records_db(args.db))

    for report in reports:
        _print_audit_report(report)

    verdicts = {report.verdict for report in reports}
    if AuditVerdict.REJECT in verdicts:
        overall = AuditVerdict.REJECT
    elif AuditVerdict.REVISE in verdicts:
        overall = AuditVerdict.REVISE
    else:
        overall = AuditVerdict.PASS

    if args.json_out is not None:
        try:
            _write_json_file(
                args.json_out,
                {
                    "overall_verdict": overall.value,
                    "reports": [report.to_dict() for report in reports],
                },
            )
        except OSError as exc:
            _print_error(f"감사 결과 JSON을 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE
        print(f"감사 JSON: {args.json_out}")

    return EXIT_OK if overall is AuditVerdict.PASS else EXIT_GATE


def _cmd_agent_drift(args: argparse.Namespace) -> int:
    from .agent.drift import DriftPolicy, assess_drift

    if args.min_samples < 1:
        _print_error(f"--min-samples 는 1 이상이어야 합니다: {args.min_samples}")
        return EXIT_USAGE
    try:
        base_records = _load_jsonl_dicts(args.baseline)
        target_records = _load_jsonl_dicts(args.target)
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    report = assess_drift(
        base_records,
        target_records,
        policy=DriftPolicy(min_samples=args.min_samples),
    )
    print(
        f"드리프트 상태: {report.status.value} "
        f"(기준선 {report.baseline_total}건 / 대상 {report.target_total}건)"
    )
    for feat in report.features:
        print(
            f"  - {feat.feature}: PSI={feat.psi:.4f} [{feat.status.value}] "
            f"(평균 {feat.baseline_mean:.4f} -> {feat.target_mean:.4f}, "
            f"변화 {feat.delta_mean:+.4f})"
        )
    for warning in report.warnings:
        print(f"  * {warning}")

    if args.json_out is not None:
        try:
            _write_json_file(args.json_out, report.to_dict())
        except OSError as exc:
            _print_error(f"드리프트 JSON을 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE
        print(f"드리프트 JSON: {args.json_out}")

    return EXIT_OK


def _cmd_agent_triage(args: argparse.Namespace) -> int:
    from .agent.review import assess_review_triage, rank_review_queue
    from .agent.stability import analyze_weak_geometry
    from .config import default_config

    if args.top_k < 1:
        _print_error(f"--top-k 는 1 이상이어야 합니다: {args.top_k}")
        return EXIT_USAGE
    try:
        records = _load_jsonl_dicts(args.results)
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    ng_min = default_config().scoring.ng_min
    assessments = []
    for rec in records:
        if not args.include_all and str(rec.get("status", "")) != Status.REVIEW.value:
            continue
        try:
            res = _sample_result_from_record(rec)
        except (KeyError, TypeError, ValueError) as exc:
            _print_error(f"결과 레코드 해석 실패({rec.get('sample_id', '?')}): {exc}")
            return EXIT_INPUT
        geom = analyze_weak_geometry(res.components, res.roi)
        assessments.append(assess_review_triage(res, geometry=geom, ng_min=ng_min))

    ranked = rank_review_queue(assessments)
    priority_counts = {
        band: sum(1 for item in ranked if item.priority.value == band)
        for band in ("P0", "P1", "P2", "P3")
    }
    shown = ranked[: args.top_k]

    print(
        f"트리아지 요약: 전체 {len(records)}건 중 대상 {len(ranked)}건 "
        f"(P0={priority_counts['P0']}, P1={priority_counts['P1']}, "
        f"P2={priority_counts['P2']}, P3={priority_counts['P3']})"
    )
    for idx, item in enumerate(shown, start=1):
        print(
            f"  {idx:>2}. {item.sample_id:<16} {item.priority.value} "
            f"score={item.priority_score:.3f} pattern={item.weak_pattern.value} "
            f"| {item.argumentation.operator_guidance}"
        )

    if args.json_out is not None:
        try:
            _write_json_file(
                args.json_out,
                {
                    "total_records": len(records),
                    "triaged_count": len(ranked),
                    "top_k": args.top_k,
                    "priority_counts": priority_counts,
                    "items": [item.to_dict() for item in shown],
                },
            )
        except OSError as exc:
            _print_error(f"트리아지 JSON을 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE
        print(f"트리아지 JSON: {args.json_out}")

    return EXIT_OK


def _cmd_agent_explain(args: argparse.Namespace) -> int:
    from .agent.counterfactual import explain_counterfactual
    from .config import default_config

    if args.config is not None:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            _print_error(f"설정을 불러오지 못했습니다: {exc}")
            return EXIT_USAGE
    else:
        cfg = default_config()

    try:
        records = _load_jsonl_dicts(args.results)
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    target_rec = next((r for r in records if str(r.get("sample_id", "")) == args.sample_id), None)
    if target_rec is None:
        _print_error(f"샘플을 찾을 수 없습니다: {args.sample_id}")
        return EXIT_INPUT

    try:
        res = _sample_result_from_record(target_rec)
    except (KeyError, TypeError, ValueError) as exc:
        _print_error(f"결과 레코드 해석 실패({args.sample_id}): {exc}")
        return EXIT_INPUT

    report = explain_counterfactual(res, cfg)
    print(
        f"반사실적 경계 해석: {report.sample_id} "
        f"[{report.machine_status.value}] score={report.score:.4f}"
    )
    print(f"요약: {report.headline_ko}")
    print(
        f"여유: NG({cfg.scoring.ng_min:.2f})까지 {report.margin_to_ng:+.4f} / "
        f"REVIEW 상한까지 {report.margin_to_review_band:+.4f}"
    )
    for attr in report.components:
        print(f"  - {attr.explanation_ko}")

    if args.json_out is not None:
        try:
            _write_json_file(args.json_out, report.to_dict())
        except OSError as exc:
            _print_error(f"설명 JSON을 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE
        print(f"설명 JSON: {args.json_out}")
    return EXIT_OK


def _cmd_agent_calibrate(args: argparse.Namespace) -> int:
    from .agent.calibration import analyze_human_feedback
    from .config import REVIEW_BAND_MAX, default_config
    from .io.artifacts import CONFIG_SNAPSHOT, RESULTS_JSONL
    from .review import ReviewError, ReviewStorageError, ReviewStore

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        _print_error(f"run 디렉터리가 없습니다: {run_dir}")
        return EXIT_INPUT
    try:
        records = _load_jsonl_dicts(run_dir / RESULTS_JSONL)
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    try:
        decisions = ReviewStore(run_dir).read_all()
    except (ReviewError, ReviewStorageError) as exc:
        _print_error(f"검토 결정을 읽지 못했습니다: {exc}")
        return EXIT_INPUT

    snap = run_dir / CONFIG_SNAPSHOT
    cfg = default_config()
    if snap.is_file():
        with contextlib.suppress(ConfigError):
            cfg = load_config(snap)

    summary = analyze_human_feedback(
        records,
        decisions,
        ng_min=float(cfg.scoring.ng_min),
        review_band_max=REVIEW_BAND_MAX,
    )
    print(
        f"Phase 4 캘리브레이션: REVIEW {summary.total_review_samples}건 중 "
        f"결정 완료 {summary.decided_count}건 (미결정 {summary.undecided_count}건)"
    )
    for band in ("P0", "P1", "P2", "P3"):
        pb = summary.by_priority[band]
        ng_r = pb["confirmed_ng_ratio"]
        pct = "-" if ng_r["value"] is None else f"{ng_r['value'] * 100:.1f}%"
        print(
            f"  - {band}: REVIEW {pb['review_count']}건 / 결정 {pb['decided_count']}건 / "
            f"NG {pb['confirmed_ng_count']}건 (확정률 {pct}, 분모 {int(ng_r['denominator'])})"
        )
    print(f"하드 샘플 큐: {len(summary.hard_examples)}건")
    for h in summary.hard_examples[:10]:
        print(
            f"  * [{h['kind']}] {h['sample_id']} "
            f"({h['priority']}, score={h['score']:.4f}, human={h['human_status']})"
        )
    for rec in summary.recommendations_ko:
        print(f"권고: {rec}")

    if args.json_out is not None:
        try:
            _write_json_file(args.json_out, summary.to_dict())
        except OSError as exc:
            _print_error(f"캘리브레이션 JSON을 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE
        print(f"캘리브레이션 JSON: {args.json_out}")
    return EXIT_OK


def _run_agent_command(args: argparse.Namespace) -> int:
    if args.agent_command == "run":
        return _cmd_agent_run(args)
    if args.agent_command == "replay":
        return _cmd_agent_replay(args)
    if args.agent_command == "audit":
        return _cmd_agent_audit(args)
    if args.agent_command == "drift":
        return _cmd_agent_drift(args)
    if args.agent_command == "triage":
        return _cmd_agent_triage(args)
    if args.agent_command == "explain":
        return _cmd_agent_explain(args)
    if args.agent_command == "calibrate":
        return _cmd_agent_calibrate(args)
    print(
        "agent 하위 명령이 필요합니다: run, replay, audit, drift, triage, explain, calibrate",
        file=sys.stderr,
    )
    return EXIT_USAGE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-sentinel",
        description="결정론적 표면 결함 검사 및 검증 도구.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="환경 진단 정보를 JSON으로 출력합니다.")
    _add_data_parser(subparsers)
    _add_inspect_parser(subparsers)
    _add_batch_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_recalibrate_parser(subparsers)
    _add_fullverify_details_parser(subparsers)
    _add_adopted_verify_parser(subparsers)
    _add_defect_map_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_yolo_parser(subparsers)
    _add_records_parser(subparsers)
    _add_worker_parser(subparsers)
    _add_agent_parser(subparsers)
    _add_gate_parser(subparsers)
    _add_repro_parser(subparsers)
    return parser


def _print_error(message: str) -> None:
    print(f"오류: {message}", file=sys.stderr)


def _sibling_build_stats(manifest_path: Path) -> dict[str, Any]:
    report_path = manifest_path.parent / "build_errors.json"
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    group_stats = payload.get("group_stats")
    return dict(group_stats) if isinstance(group_stats, dict) else {}


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    try:
        result = build_manifest(args.raw, args.manifest, args.errors)
    except OSError as exc:
        _print_error(f"manifest를 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE
    except ManifestError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    if not result.success:
        _print_error("manifest 생성에 실패했습니다. manifest를 쓰지 않았습니다.")
        for error in result.errors:
            print(f"- [{error.get('kind', 'error')}] {error.get('message', '')}")
        return EXIT_INPUT

    summary = result.counts
    group_stats = result.group_stats
    canonical = [row for row in result.rows if row.sample_id == row.original_id]
    canonical_defect = sum(1 for row in canonical if row.label == "defect")
    canonical_good = sum(1 for row in canonical if row.label == "good")
    print(f"manifest 생성 완료: {result.manifest_path}")
    print(f"sha256: {result.manifest_sha256}")
    print(
        f"전체 {summary['total_rows']:,}행 (결함 {summary['defect']:,} 포함) "
        f"/ 정본 {summary['canonical_rows']:,} "
        f"(결함 {canonical_defect:,} + 정상 {canonical_good:,})"
    )
    print(
        f"공식 train {summary['official_train']:,} / test {summary['official_test']:,} "
        f"/ 중복 {summary['duplicate_rows']:,} / 그룹 {summary['groups']:,} "
        f"/ 최대 그룹 {summary['max_group_rows']:,}행"
    )
    print(
        f"dHash threshold: {group_stats['dhash_threshold']} "
        f"/ 재그룹핑: {'예' if group_stats['group_rebuild_applied'] else '아니오'} "
        f"/ 2% 초과 그룹: {group_stats['groups_over_cap']} "
        f"/ pyb 파일: {summary['pyb_files']}"
    )
    print(f"경고: {len(result.warnings)}건")
    for warning in result.warnings:
        print(f"- [{warning.get('kind', 'warning')}] {warning.get('message', '')}")
    return EXIT_OK


def _cmd_make_splits(args: argparse.Namespace) -> int:
    try:
        rows = load_manifest(args.manifest)
    except ManifestError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    warnings: list[str] = []
    groups = build_groups(rows, warnings)
    try:
        assignment = assign_groups(groups, seed=args.seed)
    except SplitError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    stats = _sibling_build_stats(args.manifest)
    try:
        report = write_splits(
            rows,
            assignment.assignment,
            args.out_dir,
            seed=args.seed,
            dhash_threshold=int(stats.get("dhash_threshold", DHASH_THRESHOLD)),
            group_rebuild_applied=bool(stats.get("group_rebuild_applied", False)),
            adjustments=assignment.adjustments,
            warnings=warnings,
        )
    except SplitError as exc:
        _print_error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        _print_error(f"분할 파일을 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    tune = report["splits"]["tune"]
    test = report["splits"]["test"]
    official = report["splits"]["official_test"]
    canonical_total = report["manifest"]["canonical"]["total_rows"]
    ratio = test["rows"] / canonical_total if canonical_total else 0.0
    print(f"분할 생성 완료: {args.out_dir}")
    print(
        f"tune {tune['rows']:,} / test {test['rows']:,} ({ratio:.4f}) "
        f"/ official_test {official['rows']:,}"
    )
    print(f"P_test: {test['defect']:,} (하한 100)")
    print(f"test sha256: {test['sha256']}")
    print(f"split_hash: {report['split_hash']}")
    print(f"조정: {len(report['adjustments'])}건 / 경고: {len(warnings)}건")
    for warning in warnings:
        print(f"- {warning}")
    return EXIT_OK


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        rows = load_manifest(args.manifest)
    except ManifestError as exc:
        _print_error(str(exc))
        return EXIT_INPUT

    split_result = verify_splits(rows, args.splits_dir)

    contact_sheet_info: dict[str, Any] | None = None
    if args.contact_sheet is not None:
        try:
            tune_rows = load_manifest(args.splits_dir / "tune.csv")
            test_rows = load_manifest(args.splits_dir / "test.csv")
            contact_sheet_info = write_contact_sheet(
                tune_rows,
                args.raw,
                args.contact_sheet,
                test_sample_ids={row.sample_id for row in test_rows},
            )
        except (ManifestError, SplitError) as exc:
            _print_error(str(exc))
            return EXIT_INPUT
        except ValueError as exc:
            _print_error(str(exc))
            return EXIT_INPUT
        except OSError as exc:
            _print_error(f"contact sheet를 저장하지 못했습니다: {exc}")
            return EXIT_STORAGE

    try:
        audit = verify_dataset(
            args.raw,
            rows,
            split_result,
            contact_sheet_path=args.contact_sheet,
            contact_sheet_info=contact_sheet_info,
        )
    except (ManifestError, OSError) as exc:
        _print_error(f"데이터 감사에 실패했습니다: {exc}")
        return EXIT_INPUT

    try:
        if args.audit_json is not None:
            write_audit_json(audit, args.audit_json)
        if args.audit_md is not None:
            write_audit_markdown(audit, args.audit_md)
    except OSError as exc:
        _print_error(f"감사 보고서를 저장하지 못했습니다: {exc}")
        return EXIT_STORAGE

    print(f"분할 검증: {'PASS' if split_result.ok else 'FAIL'} "
          f"(위반 {len(split_result.violations)}건, 경고 {len(split_result.warnings)}건)")
    print(f"데이터 감사: {'PASS' if audit.ok else 'FAIL'} "
          f"(위반 {len(audit.violations)}건, 경고 {len(audit.warnings)}건)")
    print(
        f"tune {split_result.counts['tune']['total_rows']:,} "
        f"/ test {split_result.counts['test']['total_rows']:,} "
        f"/ official_test {split_result.counts['official_test']['total_rows']:,}"
    )
    print(
        f"P_test: {split_result.p_test} / test 비율: {split_result.test_ratio:.4f} "
        f"/ 공식 test 중복: test {split_result.overlap['internal_test']['samples']}장, "
        f"tune {split_result.overlap['internal_tune']['samples']}장"
    )
    if contact_sheet_info is not None:
        print(f"contact sheet: {contact_sheet_info['path']} "
              f"({contact_sheet_info['total_rows']}장)")
    if args.audit_json is not None:
        print(f"감사 JSON: {args.audit_json}")
    if args.audit_md is not None:
        print(f"감사 Markdown: {args.audit_md}")

    for message in split_result.violations:
        print(f"[분할 위반] {message}")
    for message in audit.violations:
        print(f"[데이터 위반] {message}")
    for message in split_result.warnings:
        print(f"[분할 경고] {message}")
    for message in audit.warnings:
        print(f"[감사 경고] {message}")

    if split_result.ok and audit.ok:
        return EXIT_OK
    return EXIT_INPUT


def _error_exit_code(error_code: ErrorCode | None) -> int:
    if error_code is ErrorCode.E_SEALED_DENIED:
        return EXIT_SEALED
    if error_code in INPUT_ERROR_CODES:
        return EXIT_INPUT
    if error_code is ErrorCode.E_STORAGE_WRITE:
        return EXIT_STORAGE
    return EXIT_ERROR


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        _print_error(f"설정을 불러오지 못했습니다: {exc}")
        return EXIT_USAGE

    outcome = inspect_image(
        args.image,
        config,
        mask_path=args.mask,
        label=args.label,
        split=args.split,
        output_dir=args.output,
    )
    result = outcome.result

    if result.status is Status.ERROR:
        code = result.error_code
        if code is ErrorCode.E_SEALED_DENIED:
            _print_error(
                "봉인된 split은 inspect에서 열 수 없습니다. "
                "test/official_test 접근은 거부됩니다."
            )
            return EXIT_SEALED
        stage = result.error_stage.value if result.error_stage is not None else "-"
        _print_error(f"검사 오류: {code.value if code else 'E_PROCESS_INTERNAL'} ({stage})")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return _error_exit_code(code)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    reasons = ", ".join(reason.value for reason in result.reasons) or "-"
    print(
        f"판정: {result.status.value} / 점수: {result.score:.4f} / 사유: {reasons}",
        file=sys.stderr,
    )
    if result.overlay_path:
        print(f"오버레이: {result.overlay_path}", file=sys.stderr)
    return EXIT_OK


def _run_data(args: argparse.Namespace) -> int:
    if args.data_command == "build-manifest":
        return _cmd_build_manifest(args)
    if args.data_command == "make-splits":
        return _cmd_make_splits(args)
    if args.data_command == "verify":
        return _cmd_verify(args)
    print("data 하위 명령이 필요합니다: build-manifest, make-splits, verify", file=sys.stderr)
    return EXIT_USAGE


def _ensure_utf8_stdio() -> None:
    """비-UTF-8 콘솔·파이프(예: 영문 Windows cp1252)에서 한글 출력 오류를 막는다."""

    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("_", "-")
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding != "utf-8" and callable(reconfigure):
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(_doctor(), ensure_ascii=False, indent=2))
        return EXIT_OK
    if args.command == "data":
        return _run_data(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "recalibrate":
        return _cmd_recalibrate(args)
    if args.command == "fullverify-details":
        return _cmd_fullverify_details(args)
    if args.command == "adopted-verify":
        return _cmd_adopted_verify(args)
    if args.command == "defect-map":
        return _cmd_defect_map(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "yolo":
        return _run_yolo(args)
    if args.command == "records":
        return _run_records(args)
    if args.command == "worker":
        return _run_worker_command(args)
    if args.command == "agent":
        return _run_agent_command(args)
    if args.command == "gate":
        return _cmd_gate(args)
    if args.command == "repro":
        return _run_repro(args)
    parser.print_help(sys.stdout)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
