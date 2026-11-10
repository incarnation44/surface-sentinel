"""Surface Sentinel CLI.

T03의 ``data`` 서브커맨드(``build-manifest``, ``make-splits``, ``verify``)와
T04의 ``inspect`` 서브커맨드를 제공한다. 사람이 읽는 도움말과 오류 메시지는
한국어다.
"""

from __future__ import annotations

import argparse
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
    _add_defect_map_parser(subparsers)
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
    if args.command == "defect-map":
        return _cmd_defect_map(args)
    parser.print_help(sys.stdout)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
