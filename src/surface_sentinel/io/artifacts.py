"""run 디렉터리 산출물 스키마와 직렬화 (설계 §5.2–§5.4).

필드·컬럼 목록과 순서는 설계 문서와 1:1로 고정한다. 순서를 바꾸면 스키마
계약 위반이며, ``result-v1``/``summary.csv``/``run-v1`` 소비자는 이 순서를
그대로 신뢰한다.

직렬화만 담당하고 디렉터리 원자성은 ``io.atomic``이 맡는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contracts import RESULT_SCHEMA_VERSION, RUN_SCHEMA_VERSION, SampleResult

SUMMARY_SCHEMA_VERSION = "summary-v1"

RESULTS_JSONL = "results.jsonl"
SUMMARY_CSV = "summary.csv"
SUMMARY_JSON = "summary.json"
RUN_JSON = "run.json"
CONFIG_SNAPSHOT = "config_snapshot.toml"
MANIFEST_SNAPSHOT = "manifest_snapshot.csv"
LOG_JSONL = "logs/run.log.jsonl"
OVERLAYS_DIR = "overlays"

#: ``results.jsonl`` (``result-v1``) 필드 순서.
RESULT_FIELDS: tuple[str, ...] = (
    "run_id",
    "sample_id",
    "original_id",
    "split",
    "label",
    "status",
    "score",
    "reasons",
    "components",
    "quality",
    "roi",
    "timing_ms",
    "error_code",
    "error_stage",
    "input_sha256",
    "mask_sha256",
    "overlay_path",
    "config_hash",
)

#: ``summary.csv`` 컬럼 순서.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "sample_id",
    "original_id",
    "split",
    "label",
    "status",
    "score",
    "top_area_px",
    "top_contrast_gray",
    "top_evidence",
    "reason_codes",
    "component_count",
    "duration_ms",
    "error_code",
    "overlay_path",
)

#: ``run.json`` (``run-v1``) 필드 순서.
RUN_FIELDS: tuple[str, ...] = (
    "run_id",
    "mode",
    "split",
    "config_path",
    "config_hash",
    "code",
    "env",
    "manifest_sha256",
    "split_sha256",
    "started_at",
    "finished_at",
    "status",
    "counts",
    "timing",
    "artifacts",
)

#: ``summary.csv``의 ``reason_codes`` 구분자. 사유 코드에는 나타나지 않는 문자다.
REASON_SEPARATOR = "|"


class ArtifactError(ValueError):
    """산출물 스키마를 위반하는 값이 들어왔을 때 발생한다."""


def _ordered(payload: Mapping[str, Any], fields: Sequence[str], kind: str) -> dict[str, Any]:
    missing = [name for name in fields if name not in payload]
    extra = [name for name in payload if name not in fields]
    if missing or extra:
        raise ArtifactError(
            f"{kind} 스키마가 계약과 다릅니다: 누락={missing} 초과={extra}"
        )
    return {name: payload[name] for name in fields}


def _write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: str | Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def build_result_record(
    result: SampleResult,
    *,
    run_id: str,
    sample_id: str,
    original_id: str,
    split: str,
    label: str,
    overlay_path: str,
) -> dict[str, Any]:
    """``SampleResult``를 ``result-v1`` 레코드로 바꾼다.

    ``run_id``/``sample_id``/``original_id``/``split``/``label``/
    ``overlay_path``는 run 문맥의 값을 쓰고(manifest·분할 CSV가 정본),
    나머지는 파이프라인 결과를 그대로 옮긴다.
    """

    payload = result.to_dict()
    record = {
        "run_id": run_id,
        "sample_id": sample_id,
        "original_id": original_id,
        "split": split,
        "label": label,
        "status": payload["status"],
        "score": payload["score"],
        "reasons": payload["reasons"],
        "components": payload["components"],
        "quality": payload["quality"],
        "roi": payload["roi"],
        "timing_ms": payload["timing_ms"],
        "error_code": payload["error_code"],
        "error_stage": payload["error_stage"],
        "input_sha256": payload["input_sha256"],
        "mask_sha256": payload["mask_sha256"],
        "overlay_path": overlay_path,
        "config_hash": payload["config_hash"],
    }
    return _ordered(record, RESULT_FIELDS, "result-v1")


def _top_component(components: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """증거가 가장 큰 컴포넌트. 동률이면 index가 작은 쪽을 고른다."""

    if not components:
        return None
    return max(
        components,
        key=lambda item: (float(item["evidence"]), -int(item["index"])),
    )


def build_summary_row(record: Mapping[str, Any]) -> dict[str, str]:
    """``result-v1`` 레코드 하나를 ``summary.csv`` 행으로 요약한다."""

    components = record.get("components") or []
    top = _top_component(components)
    timing = record.get("timing_ms") or {}
    duration = float(timing.get("total", 0.0))
    row = {
        "run_id": str(record["run_id"]),
        "sample_id": str(record["sample_id"]),
        "original_id": str(record["original_id"]),
        "split": str(record["split"]),
        "label": str(record["label"]),
        "status": str(record["status"]),
        "score": f"{float(record['score']):.6f}",
        "top_area_px": "" if top is None else str(int(top["area_px"])),
        "top_contrast_gray": (
            "" if top is None else f"{float(top['mean_contrast_gray']):.6f}"
        ),
        "top_evidence": "" if top is None else f"{float(top['evidence']):.6f}",
        "reason_codes": REASON_SEPARATOR.join(str(code) for code in record.get("reasons") or []),
        "component_count": str(len(components)),
        "duration_ms": f"{duration:.3f}",
        "error_code": "" if record.get("error_code") is None else str(record["error_code"]),
        "overlay_path": str(record.get("overlay_path") or ""),
    }
    return _ordered(row, SUMMARY_COLUMNS, "summary.csv")


def write_results_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> int:
    """``results.jsonl``을 쓰고 쓴 레코드 수를 돌려준다."""

    lines: list[str] = []
    for record in records:
        ordered = _ordered(record, RESULT_FIELDS, "result-v1")
        lines.append(json.dumps(ordered, ensure_ascii=False) + "\n")
    _write_text(path, "".join(lines))
    return len(lines)


def write_summary_csv(path: str | Path, rows: Iterable[Mapping[str, str]]) -> int:
    """``summary.csv``를 쓰고 쓴 데이터 행 수를 돌려준다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_ordered(row, SUMMARY_COLUMNS, "summary.csv"))
            written += 1
        handle.flush()
    return written


def write_summary_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """``summary.json``을 결정론적 JSON으로 쓴다."""

    _write_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_run_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """``run.json``을 ``run-v1`` 필드 순서로 쓴다. 반드시 마지막에 호출한다."""

    ordered = _ordered(payload, RUN_FIELDS, "run-v1")
    _write_text(path, json.dumps(ordered, ensure_ascii=False, indent=2) + "\n")


def write_config_snapshot(path: str | Path, canonical_toml: str) -> None:
    """설정 스냅샷을 config_hash 계산 입력과 같은 바이트로 쓴다."""

    text = canonical_toml if canonical_toml.endswith("\n") else canonical_toml + "\n"
    _write_text(path, text)


def write_manifest_snapshot(path: str | Path, source: str | Path) -> str:
    """manifest 원본 바이트를 그대로 복사하고 sha256을 돌려준다."""

    payload = Path(source).read_bytes()
    _write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def write_log_lines(path: str | Path, lines: Iterable[Mapping[str, Any]]) -> int:
    """``logs/run.log.jsonl``을 쓰고 쓴 줄 수를 돌려준다."""

    written = 0
    payload: list[str] = []
    for line in lines:
        payload.append(json.dumps(dict(line), ensure_ascii=False) + "\n")
        written += 1
    _write_text(path, "".join(payload))
    return written


def count_results(path: str | Path) -> int:
    """``results.jsonl``의 비어 있지 않은 줄 수를 센다."""

    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def count_summary_rows(path: str | Path) -> int:
    """``summary.csv``의 데이터 행 수(헤더 제외)를 센다."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for row in reader if row)


__all__ = [
    "ArtifactError",
    "CONFIG_SNAPSHOT",
    "LOG_JSONL",
    "MANIFEST_SNAPSHOT",
    "OVERLAYS_DIR",
    "REASON_SEPARATOR",
    "RESULT_FIELDS",
    "RESULTS_JSONL",
    "RESULT_SCHEMA_VERSION",
    "RUN_FIELDS",
    "RUN_JSON",
    "RUN_SCHEMA_VERSION",
    "SUMMARY_COLUMNS",
    "SUMMARY_CSV",
    "SUMMARY_JSON",
    "SUMMARY_SCHEMA_VERSION",
    "build_result_record",
    "build_summary_row",
    "count_results",
    "count_summary_rows",
    "write_config_snapshot",
    "write_log_lines",
    "write_manifest_snapshot",
    "write_results_jsonl",
    "write_run_json",
    "write_summary_csv",
    "write_summary_json",
]
