"""배치 실행과 run 디렉터리 (설계 §5.1–§5.4, T06).

한 번의 배치 실행은 ``runs/<run_id>/`` 디렉터리 하나를 만든다. 모든 파일은
``runs/.tmp-<run_id>-<pid>/``에 먼저 쓰고 fsync한 뒤 ``os.replace``로 최종
이름으로 교체한다. ``run.json``은 항상 마지막에 쓴다.

봉인 규칙: ``test``/``official_test``는 파일을 열기 전에 거부한다. 이 모듈에는
봉인 분할을 읽는 경로가 없다.
"""

from __future__ import annotations

import csv
import hashlib
import math
import platform
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import CONFIG_SCHEMA_VERSION, ConfigError, load_config
from ..contracts import RunStatus, Status
from ..io import artifacts
from ..io.atomic import AtomicWriteError, atomic_directory
from ..io.manifest import MANIFEST_COLUMNS, ManifestError, ManifestRow, load_manifest
from ..pipeline import inspect_image
from ..pipeline.evidence import sanitize_sample_id
from ..version import __version__

BATCH_MODE = "batch"

DEFAULT_CONFIG_PATH = Path("configs/base.toml")
DEFAULT_MANIFEST_PATH = Path("data/manifests/dataset_manifest.csv")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_RAW_ROOT = Path("data/raw/kolektorsdd2")
DEFAULT_OUTPUT_DIR = Path("runs")

#: batch가 실제로 실행할 수 있는 분할. 나머지는 봉인이다.
BATCH_SPLITS: tuple[str, ...] = ("tune",)
SEALED_SPLITS = frozenset({"test", "official_test"})

#: `YYYYMMDDTHHMMSSZ-<config_hash8>-<split>[-NN]`
RUN_ID_PATTERN = re.compile(
    r"^\d{8}T\d{6}Z-[0-9a-f]{8}-(?:tune|test|official_test)(?:-\d{2})?$"
)
MAX_RUN_ID_ATTEMPTS = 100


class BatchError(RuntimeError):
    """배치를 계속할 수 없을 때의 공통 상위 예외. ``exit_code``를 갖는다."""

    exit_code = 5


class BatchUsageError(BatchError):
    """사용법/설정 오류 (종료 코드 2)."""

    exit_code = 2


class BatchInputError(BatchError):
    """입력 검증 오류 (종료 코드 3)."""

    exit_code = 3


class BatchStorageError(BatchError):
    """저장 실패 또는 완료하지 못한 run (종료 코드 5)."""

    exit_code = 5


class SealedSplitError(BatchError):
    """봉인된 분할 접근 거부 (종료 코드 6)."""

    exit_code = 6


@dataclass(frozen=True)
class BatchOutcome:
    """배치 실행 결과 요약. CLI가 종료 코드와 보고에 쓴다."""

    exit_code: int
    run_id: str
    run_dir: Path
    status: RunStatus
    counts: dict[str, int]
    timing: dict[str, Any]
    error_count: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _iso_utc(moment: datetime) -> str:
    return (
        moment.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], percent: int) -> float:
    """nearest-rank 백분위수. 값이 비면 0.0, 단일 값이면 그 값이다."""

    if not 1 <= percent <= 100:
        raise ValueError(f"percent는 1..100 범위여야 합니다: {percent}")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percent / 100.0 * len(ordered)) - 1)
    return ordered[index]


def _timing_payload(durations: Sequence[float], wall_ms: float) -> dict[str, Any]:
    count = len(durations)
    total = float(sum(durations))
    return {
        "samples": count,
        "wall_ms": round(wall_ms, 3),
        "total_ms": round(total, 3),
        "mean_ms": round(total / count, 3) if count else 0.0,
        "min_ms": round(min(durations), 3) if count else 0.0,
        "max_ms": round(max(durations), 3) if count else 0.0,
        "p50_ms": round(percentile(durations, 50), 3),
        "p95_ms": round(percentile(durations, 95), 3),
    }


def _env_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": None,
        "numpy": None,
    }
    try:
        import cv2

        payload["opencv"] = cv2.__version__
    except Exception:  # pragma: no cover - opencv는 필수 의존성이다
        payload["opencv"] = None
    try:
        import numpy

        payload["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover - numpy는 필수 의존성이다
        payload["numpy"] = None
    return payload


def allocate_run_id(
    output_dir: str | Path,
    *,
    split: str,
    config_hash: str,
    now: datetime | None = None,
) -> str:
    """충돌하지 않는 run_id를 만든다. 기존 디렉터리는 절대 덮어쓰지 않는다."""

    base = Path(output_dir)
    stamp = _utc_stamp(now if now is not None else _utc_now())
    candidate = f"{stamp}-{config_hash[:8]}-{split}"
    if not (base / candidate).exists():
        return candidate
    for index in range(1, MAX_RUN_ID_ATTEMPTS):
        numbered = f"{candidate}-{index:02d}"
        if not (base / numbered).exists():
            return numbered
    raise BatchStorageError(
        f"run_id를 {MAX_RUN_ID_ATTEMPTS}회까지 늘렸지만 빈 이름을 찾지 못했습니다: {candidate}"
    )


def _validate_request(
    *,
    split: str,
    limit: int | None,
    run_id: str | None,
) -> None:
    if split in SEALED_SPLITS:
        raise SealedSplitError(
            f"봉인된 split은 batch에서 열 수 없습니다: {split}. "
            "test/official_test 접근은 거부됩니다."
        )
    if split not in BATCH_SPLITS:
        raise BatchUsageError(
            f"batch는 {', '.join(BATCH_SPLITS)} split만 실행합니다: {split!r}"
        )
    if limit is not None and limit < 1:
        raise BatchUsageError(f"--limit은 1 이상이어야 합니다: {limit}")
    if run_id is not None and not RUN_ID_PATTERN.match(run_id):
        raise BatchUsageError(
            "run-id 형식이 계약과 다릅니다: YYYYMMDDTHHMMSSZ-<config_hash8>-"
            f"<split>[-NN] (받은 값 {run_id!r})"
        )


def _load_split_rows(splits_dir: Path, split: str, limit: int | None) -> list[ManifestRow]:
    split_path = splits_dir / f"{split}.csv"
    if not split_path.is_file():
        raise BatchInputError(f"분할 CSV를 찾을 수 없습니다: {split_path}")
    try:
        rows = load_manifest(split_path)
    except ManifestError as exc:
        raise BatchInputError(f"분할 CSV를 읽지 못했습니다: {exc}") from exc
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise BatchInputError(f"분할 CSV에 처리할 행이 없습니다: {split_path}")
    return rows


def _check_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise BatchInputError(f"manifest를 찾을 수 없습니다: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = tuple(next(csv.reader(handle)))
    except (OSError, StopIteration) as exc:
        raise BatchInputError(f"manifest를 읽지 못했습니다: {manifest_path}") from exc
    if header != MANIFEST_COLUMNS:
        raise BatchInputError(
            f"manifest 헤더가 계약과 다릅니다: {manifest_path} "
            f"기대={MANIFEST_COLUMNS} 실제={header}"
        )


def _resolve_sample_paths(
    raw_root: Path,
    row: ManifestRow,
) -> tuple[Path, Path | None]:
    image_path = raw_root / row.image_path
    mask_path = raw_root / row.mask_path if row.mask_path else None
    return image_path, mask_path


def _count_statuses(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    status_counts = Counter(str(record["status"]) for record in records)
    overlays = sum(1 for record in records if record.get("overlay_path"))
    return {
        "samples": len(records),
        "results": len(records),
        "summary_rows": len(records),
        "ok": status_counts.get(Status.OK.value, 0),
        "ng": status_counts.get(Status.NG.value, 0),
        "review": status_counts.get(Status.REVIEW.value, 0),
        "error": status_counts.get(Status.ERROR.value, 0),
        "overlays": overlays,
    }


def _verify_counts(
    run_dir: Path,
    *,
    records_written: int,
    rows_written: int,
    counts: dict[str, int],
) -> None:
    results_on_disk = artifacts.count_results(run_dir / artifacts.RESULTS_JSONL)
    summary_on_disk = artifacts.count_summary_rows(run_dir / artifacts.SUMMARY_CSV)
    expected = counts["samples"]
    observed = {
        "run.json.counts.samples": expected,
        "run.json.counts.results": counts["results"],
        "run.json.counts.summary_rows": counts["summary_rows"],
        "records_written": records_written,
        "rows_written": rows_written,
        "results.jsonl": results_on_disk,
        "summary.csv": summary_on_disk,
    }
    if len(set(observed.values())) != 1:
        raise BatchStorageError(
            "결과/요약 수가 일치하지 않아 run을 완료로 표시하지 않습니다: "
            f"{observed}"
        )


def run_batch(
    *,
    split: str = "tune",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    run_id: str | None = None,
) -> BatchOutcome:
    """tune split을 순차 실행하고 run 디렉터리를 원자적으로 만든다."""

    _validate_request(split=split, limit=limit, run_id=run_id)

    config_file = Path(config_path)
    try:
        config = load_config(config_file)
    except ConfigError as exc:
        raise BatchUsageError(f"설정을 불러오지 못했습니다: {exc}") from exc

    manifest_file = Path(manifest_path)
    splits_path = Path(splits_dir)
    raw_path = Path(raw_root)
    output_path = Path(output_dir)

    _check_manifest(manifest_file)
    rows = _load_split_rows(splits_path, split, limit)
    split_path = splits_path / f"{split}.csv"

    if run_id is None:
        run_id = allocate_run_id(
            output_path, split=split, config_hash=config.config_hash
        )
    elif (output_path / run_id).exists():
        raise BatchStorageError(
            f"이미 있는 run 디렉터리는 덮어쓰지 않습니다: {output_path / run_id}"
        )

    manifest_sha256 = _sha256_file(manifest_file)
    split_sha256 = _sha256_file(split_path)
    started_moment = _utc_now()
    started_at = _iso_utc(started_moment)
    wall_started = time.perf_counter()

    records: list[dict[str, Any]] = []
    log_lines: list[dict[str, Any]] = [
        {
            "event": "run_start",
            "ts": started_at,
            "run_id": run_id,
            "split": split,
            "samples": len(rows),
        }
    ]
    durations: list[float] = []

    final_dir = output_path / run_id
    try:
        with atomic_directory(final_dir) as temporary:
            overlays_dir = temporary / artifacts.OVERLAYS_DIR
            for row in rows:
                image_path, mask_path = _resolve_sample_paths(raw_path, row)
                outcome = inspect_image(
                    image_path,
                    config,
                    mask_path=mask_path,
                    label=row.label,
                    split=split,
                    output_dir=overlays_dir,
                    save_stages=False,
                )
                result = outcome.result
                safe_id = sanitize_sample_id(row.sample_id)

                # inspect는 `<id>_result.json`도 함께 쓴다. run 디렉터리 계약
                # (§5.2)에는 오버레이만 있으므로 방금 만든 그 파일만 지운다.
                stray = overlays_dir / f"{safe_id}_result.json"
                if stray.is_file():
                    stray.unlink()

                overlay_path = (
                    f"{artifacts.OVERLAYS_DIR}/{safe_id}_overlay.png"
                    if result.overlay_path
                    else ""
                )
                record = artifacts.build_result_record(
                    result,
                    run_id=run_id,
                    sample_id=row.sample_id,
                    original_id=row.original_id,
                    split=split,
                    label=row.label,
                    overlay_path=overlay_path,
                )
                records.append(record)
                duration = float((record.get("timing_ms") or {}).get("total", 0.0))
                durations.append(duration)
                log_lines.append(
                    {
                        "event": "sample",
                        "sample_id": row.sample_id,
                        "status": record["status"],
                        "duration_ms": round(duration, 3),
                        "error_code": record["error_code"],
                    }
                )

            counts = _count_statuses(records)
            status = (
                RunStatus.COMPLETE if counts["error"] == 0 else RunStatus.COMPLETE_WITH_ERRORS
            )

            summary_rows = [artifacts.build_summary_row(record) for record in records]
            records_written = artifacts.write_results_jsonl(
                temporary / artifacts.RESULTS_JSONL, records
            )
            rows_written = artifacts.write_summary_csv(
                temporary / artifacts.SUMMARY_CSV, summary_rows
            )
            artifacts.write_config_snapshot(
                temporary / artifacts.CONFIG_SNAPSHOT, config.canonical_toml()
            )
            snapshot_sha256 = artifacts.write_manifest_snapshot(
                temporary / artifacts.MANIFEST_SNAPSHOT, manifest_file
            )
            if snapshot_sha256 != manifest_sha256:
                raise BatchStorageError(
                    "manifest 스냅샷 해시가 원본과 다릅니다: "
                    f"{snapshot_sha256} != {manifest_sha256}"
                )

            wall_ms = (time.perf_counter() - wall_started) * 1000.0
            timing = _timing_payload(durations, wall_ms)
            finished_at = _iso_utc(_utc_now())

            _verify_counts(
                temporary,
                records_written=records_written,
                rows_written=rows_written,
                counts=counts,
            )

            log_lines.append(
                {
                    "event": "run_end",
                    "ts": finished_at,
                    "run_id": run_id,
                    "status": status.value,
                    "samples": counts["samples"],
                    "errors": counts["error"],
                }
            )
            artifacts.write_log_lines(temporary / artifacts.LOG_JSONL, log_lines)
            artifacts.write_summary_json(
                temporary / artifacts.SUMMARY_JSON,
                {
                    "run_id": run_id,
                    "split": split,
                    "schema_version": artifacts.SUMMARY_SCHEMA_VERSION,
                    "status": status.value,
                    "counts": counts,
                    "timing": timing,
                },
            )

            # run.json은 마지막이다. 이 파일이 있으면 run이 완료된 것이다.
            artifacts.write_run_json(
                temporary / artifacts.RUN_JSON,
                {
                    "run_id": run_id,
                    "mode": BATCH_MODE,
                    "split": split,
                    "config_path": str(config_file),
                    "config_hash": config.config_hash,
                    "code": {
                        "package_version": __version__,
                        "config_schema": CONFIG_SCHEMA_VERSION,
                    },
                    "env": _env_payload(),
                    "manifest_sha256": manifest_sha256,
                    "split_sha256": split_sha256,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": status.value,
                    "counts": counts,
                    "timing": timing,
                    "artifacts": {
                        "run": artifacts.RUN_JSON,
                        "results": artifacts.RESULTS_JSONL,
                        "summary_csv": artifacts.SUMMARY_CSV,
                        "summary_json": artifacts.SUMMARY_JSON,
                        "config_snapshot": artifacts.CONFIG_SNAPSHOT,
                        "manifest_snapshot": artifacts.MANIFEST_SNAPSHOT,
                        "log": artifacts.LOG_JSONL,
                        "overlays_dir": artifacts.OVERLAYS_DIR,
                        "overlay_count": counts["overlays"],
                    },
                },
            )
    except AtomicWriteError as exc:
        raise BatchStorageError(f"run 디렉터리를 원자적으로 저장하지 못했습니다: {exc}") from exc
    except OSError as exc:
        raise BatchStorageError(f"run 디렉터리를 저장하지 못했습니다: {exc}") from exc

    exit_code = 0 if status is RunStatus.COMPLETE else 4
    return BatchOutcome(
        exit_code=exit_code,
        run_id=run_id,
        run_dir=final_dir,
        status=status,
        counts=counts,
        timing=timing,
        error_count=counts["error"],
    )


__all__ = [
    "BATCH_MODE",
    "BATCH_SPLITS",
    "BatchError",
    "BatchInputError",
    "BatchOutcome",
    "BatchStorageError",
    "BatchUsageError",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RAW_ROOT",
    "DEFAULT_SPLITS_DIR",
    "MAX_RUN_ID_ATTEMPTS",
    "RUN_ID_PATTERN",
    "SEALED_SPLITS",
    "SealedSplitError",
    "allocate_run_id",
    "percentile",
    "run_batch",
]
