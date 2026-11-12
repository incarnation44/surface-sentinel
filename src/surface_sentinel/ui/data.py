"""UI용 run 디렉터리 읽기 전용 로딩 (설계 §5.2·§5.5, T07).

이 모듈은 파일을 만들거나 고치지 않는다. 쓰기는 ``review.ReviewStore``의
검토 결정 append 하나뿐이다.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..batch import RUN_ID_PATTERN
from ..io.artifacts import MANIFEST_SNAPSHOT, RESULTS_JSONL, RUN_JSON


class UiInputError(ValueError):
    """없는 run/샘플 등 잘못된 요청. HTTP 404로 매핑한다."""


@dataclass(frozen=True)
class RunEntry:
    """run 디렉터리 하나와 ``run.json`` 내용."""

    run_id: str
    run_dir: Path
    payload: dict[str, Any]


def discover_runs(runs_dir: Path) -> list[RunEntry]:
    """run_id 형식 디렉터리만 최신순으로 모은다. 손상된 run은 목록에서 뺀다."""

    if not runs_dir.is_dir():
        return []
    entries: list[RunEntry] = []
    for child in sorted(runs_dir.iterdir(), key=lambda path: path.name, reverse=True):
        if not child.is_dir() or RUN_ID_PATTERN.fullmatch(child.name) is None:
            continue
        try:
            payload = json.loads((child / RUN_JSON).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            entries.append(RunEntry(run_id=child.name, run_dir=child, payload=payload))
    return entries


def load_run(runs_dir: Path, run_id: str) -> RunEntry:
    """run_id 형식·존재 여부를 검증하고 run.json을 읽는다."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise UiInputError(f"run_id 형식이 아닙니다: {run_id!r}")
    run_dir = runs_dir / run_id
    if not run_dir.is_dir():
        raise UiInputError(f"run을 찾을 수 없습니다: {run_id}")
    try:
        payload = json.loads((run_dir / RUN_JSON).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UiInputError(f"run.json을 읽을 수 없습니다: {run_id}") from exc
    if not isinstance(payload, dict):
        raise UiInputError(f"run.json이 객체가 아닙니다: {run_id}")
    return RunEntry(run_id=run_id, run_dir=run_dir, payload=payload)


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    """``results.jsonl``을 기록 순서대로 읽는다."""

    try:
        text = (run_dir / RESULTS_JSONL).read_text(encoding="utf-8")
    except OSError as exc:
        raise UiInputError(f"results.jsonl을 읽을 수 없습니다: {run_dir.name}") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UiInputError(f"results.jsonl {number}번째 줄이 손상됐습니다.") from exc
        if not isinstance(payload, dict):
            raise UiInputError(f"results.jsonl {number}번째 줄이 객체가 아닙니다.")
        records.append(payload)
    return records


def find_sample(records: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    for record in records:
        if record.get("sample_id") == sample_id:
            return record
    raise UiInputError(f"샘플을 찾을 수 없습니다: {sample_id}")


def load_manifest_map(run_dir: Path) -> dict[str, dict[str, str]]:
    """manifest 스냅샷을 sample_id → 행으로 읽는다. 없으면 빈 dict."""

    path = run_dir / MANIFEST_SNAPSHOT
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {row["sample_id"]: row for row in csv.DictReader(handle)}
    except (OSError, csv.Error, KeyError) as exc:
        raise UiInputError(f"manifest 스냅샷을 읽을 수 없습니다: {run_dir.name}") from exc


def safe_child(base: Path, relative: str) -> Path:
    """base 밖으로 나가는 경로는 거부한다 (경로 이탈 방지)."""

    if not relative or Path(relative).is_absolute():
        raise UiInputError("상대 경로만 허용합니다.")
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise UiInputError("허용된 디렉터리 밖 경로입니다.")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RunEntry",
    "UiInputError",
    "discover_runs",
    "find_sample",
    "load_manifest_map",
    "load_results",
    "load_run",
    "safe_child",
    "sha256_file",
]