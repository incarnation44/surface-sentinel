"""Surface Sentinel CLI.

T03의 ``data`` 서브커맨드(``build-manifest``, ``make-splits``, ``verify``)와
T04의 ``inspect`` 서브커맨드를 제공한다. 사람이 읽는 도움말과 오류 메시지는
한국어다.
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
        help="평가 분할 (tune만 허용, test/official_test는 거부).",
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
    from .evaluation.runner import EvalError, SealedSplitAccess, run_evaluation

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
        help="데이터 분할 이름 (tune만 허용, test/official_test는 거부).",
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

    app = create_app(runs_dir=args.runs_dir, raw_root=args.raw_root)
    print(f"UI: http://127.0.0.1:{args.port} (종료: Ctrl+C)")
    print(f"runs: {args.runs_dir} / 원본: {args.raw_root}")
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    return EXIT_OK


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


def main(argv: Sequence[str] | None = None) -> int:
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
    parser.print_help(sys.stdout)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
