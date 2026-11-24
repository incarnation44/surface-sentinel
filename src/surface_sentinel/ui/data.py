"""UI용 run 디렉터리 및 records.db 읽기 전용 로딩 (설계 §5.2·§5.5, T07).

이 모듈은 파일을 만들거나 고치지 않는다. 쓰기는 ``review.ReviewStore``의
검토 결정 append 하나뿐이다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..batch import RUN_ID_PATTERN
from ..contracts import (
    ComponentFeature,
    ErrorCode,
    QualityInfo,
    ReasonCode,
    RoiInfo,
    SampleResult,
    Stage,
    Status,
)
from ..io.artifacts import MANIFEST_SNAPSHOT, RESULTS_JSONL, RUN_JSON
from ..records.db import DatabaseError, connect_readonly
from ..records.store import (
    STATUS_ORDER,
    get_inspection,
    list_agent_assessments,
    list_engine_observations,
)


class UiInputError(ValueError):
    """없는 run/샘플/검사 기록 등 잘못된 요청. HTTP 404로 매핑한다."""


@dataclass(frozen=True)
class RunEntry:
    """run 디렉터리 하나와 ``run.json`` 내용."""

    run_id: str
    run_dir: Path
    payload: dict[str, Any]


def read_only_connection(path: Path) -> sqlite3.Connection:
    """검사 DB를 읽기 전용(``?mode=ro``, busy_timeout·query_only)으로 연다."""

    return connect_readonly(path)


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
            if not isinstance(payload.get("counts"), dict):
                payload["counts"] = {}
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
    if "counts" in payload and not isinstance(payload["counts"], dict):
        raise UiInputError(f"run.json counts 필드가 객체가 아닙니다: {run_id}")
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

    if not relative or "\x00" in relative:
        raise UiInputError("유효하지 않은 경로입니다.")
    rel = Path(relative)
    if rel.is_absolute() or rel.anchor or rel.drive:
        raise UiInputError("상대 경로만 허용합니다.")
    try:
        candidate = (base / rel).resolve()
        if not candidate.is_relative_to(base.resolve()):
            raise UiInputError("허용된 디렉터리 밖 경로입니다.")
    except (OSError, ValueError) as exc:
        raise UiInputError("유효하지 않은 경로입니다.") from exc
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bbox(raw: Any, *, field_name: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise UiInputError(f"손상된 {field_name} 형식입니다: {raw!r}")
    try:
        return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
    except (TypeError, ValueError) as exc:
        raise UiInputError(f"손상된 {field_name} 좌표입니다: {raw!r}") from exc


def sample_result_from_record(rec: dict[str, Any]) -> SampleResult:
    """``results.jsonl`` 딕셔너리를 ``SampleResult``로 복원한다(읽기 전용 분석용)."""

    if not isinstance(rec, dict):
        raise UiInputError("손상된 results.jsonl 레코드입니다.")
    try:
        raw_comps = rec.get("components") or []
        if not isinstance(raw_comps, list):
            raise UiInputError("results.jsonl components 필드가 배열이 아닙니다.")
        components = tuple(
            ComponentFeature(
                index=int(c.get("index", idx)),
                area_px=int(c.get("area_px", 0)),
                bbox=_parse_bbox(c.get("bbox", (0, 0, 0, 0)), field_name="component.bbox"),
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
            if isinstance(c, dict)
        )
        q = rec.get("quality")
        quality = None
        if isinstance(q, dict):
            q_flags: list[ReasonCode] = []
            for val in q.get("flags") or []:
                try:
                    q_flags.append(ReasonCode(val))
                except ValueError:
                    continue
            quality = QualityInfo(
                blur_variance=float(q.get("blur_variance", 0.0)),
                mean_gray=float(q.get("mean_gray", 0.0)),
                saturated_ratio=float(q.get("saturated_ratio", 0.0)),
                flags=tuple(q_flags),
            )
        r = rec.get("roi")
        roi = (
            RoiInfo(
                mode=str(r.get("mode", "auto")),
                bbox=_parse_bbox(r.get("bbox", (0, 0, 0, 0)), field_name="roi.bbox"),
                area_px=int(r.get("area_px", 0)),
                area_ratio=float(r.get("area_ratio", 0.0)),
                uncertain=bool(r.get("uncertain", False)),
            )
            if isinstance(r, dict)
            else None
        )
        reasons: list[ReasonCode] = []
        for val in rec.get("reasons") or []:
            try:
                reasons.append(ReasonCode(val))
            except ValueError:
                continue
        error_code = None
        if rec.get("error_code"):
            try:
                error_code = ErrorCode(rec["error_code"])
            except ValueError:
                error_code = None
        error_stage = None
        if rec.get("error_stage"):
            try:
                error_stage = Stage(rec["error_stage"])
            except ValueError:
                error_stage = None
        raw_status = str(rec.get("status", "REVIEW"))
        try:
            parsed_status = Status(raw_status)
        except ValueError as exc:
            raise UiInputError(
                f"results.jsonl의 status 값이 유효하지 않습니다: {raw_status!r}"
            ) from exc
        raw_timing = rec.get("timing_ms") or {}
        if not isinstance(raw_timing, dict):
            raise UiInputError("results.jsonl timing_ms 필드가 객체가 아닙니다.")
        return SampleResult(
            run_id=str(rec.get("run_id", "")),
            sample_id=str(rec.get("sample_id", "")),
            original_id=str(rec.get("original_id", rec.get("sample_id", ""))),
            split=str(rec.get("split", "tune")),
            label=str(rec.get("label", "unknown")),
            status=parsed_status,
            score=float(rec.get("score", 0.0)),
            reasons=tuple(reasons),
            components=components,
            quality=quality,
            roi=roi,
            timing_ms={str(k): float(v) for k, v in raw_timing.items()},
            error_code=error_code,
            error_stage=error_stage,
            input_sha256=str(rec.get("input_sha256", "")),
            mask_sha256=str(rec.get("mask_sha256", "")),
            overlay_path=str(rec.get("overlay_path", "")),
            config_hash=str(rec.get("config_hash", "")),
        )
    except UiInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise UiInputError(f"results.jsonl 레코드 복원에 실패했습니다: {exc}") from exc


def row_overlay_path(row: dict[str, Any], runs_dir: Path) -> Path | None:
    """DB 행의 오버레이 경로가 ``runs_dir`` 안의 실제 PNG 파일일 때만 돌려준다."""

    raw = str(row.get("overlay_path") or "").strip()
    if not raw or "\x00" in raw:
        return None
    normalized = raw.replace("\\", "/")
    try:
        resolved = Path(normalized).resolve()
        if not resolved.is_file():
            return None
        if resolved.suffix.lower() != ".png":
            return None
        if not resolved.is_relative_to(runs_dir.resolve()):
            return None
    except (OSError, ValueError):
        return None
    return resolved


def load_inspections_view(
    db_path: Path,
    runs_dir: Path,
    *,
    status: str | None = None,
    engine: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """검사 DB 목록 + 교차 증거 요약(읽기 전용, 미존재 시 파일 미생성)."""

    base: dict[str, Any] = {
        "available": False,
        "path": str(db_path),
        "message": "",
        "items": [],
        "total": 0,
        "all_total": 0,
        "engines": [],
        "status_counts": dict.fromkeys(STATUS_ORDER, 0),
        "status": status or "",
        "engine": engine or "",
    }
    if not db_path.is_file():
        return {
            **base,
            "message": (
                "검사 기록 DB가 없습니다. 검사 API(--with-api) 또는 워커로 기록을 남기면 "
                "교차증거와 검사 이력이 여기에 표시됩니다."
            ),
        }

    try:
        connection = read_only_connection(db_path)
        try:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "inspections" not in tables:
                return {**base, "message": "inspections 테이블이 없는 DB입니다."}

            status_counts = dict.fromkeys(STATUS_ORDER, 0)
            for r in connection.execute(
                "SELECT status, COUNT(*) AS n FROM inspections GROUP BY status"
            ).fetchall():
                status_counts[str(r["status"])] = int(r["n"])
            all_total = sum(status_counts.values())
            engines = [
                str(r["engine"])
                for r in connection.execute(
                    "SELECT DISTINCT engine FROM inspections ORDER BY engine ASC"
                ).fetchall()
            ]

            where: list[str] = []
            params: list[Any] = []
            if status and status in STATUS_ORDER:
                where.append("status = ?")
                params.append(status)
            if engine:
                where.append("engine = ?")
                params.append(engine)
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""

            total = int(
                connection.execute(
                    f"SELECT COUNT(*) AS n FROM inspections{where_sql}", params
                ).fetchone()["n"]
            )
            rows = [
                dict(r)
                for r in connection.execute(
                    f"SELECT * FROM inspections{where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
                    [*params, int(limit), int(offset)],
                ).fetchall()
            ]

            has_obs = "engine_observations" in tables
            has_assess = "agent_assessments" in tables
            has_reviews = "review_decisions" in tables

            items: list[dict[str, Any]] = []
            for row in rows:
                iid = int(row["id"])
                reasons = [str(x) for x in json.loads(str(row.get("reasons") or "[]"))]
                obs_count = (
                    len(list_engine_observations(connection, iid)) if has_obs else 0
                )
                assessments = list_agent_assessments(connection, iid) if has_assess else []
                latest_assess = assessments[-1] if assessments else None
                flags: list[str] = []
                priority: str | None = None
                recommended_action: str | None = None
                if latest_assess is not None:
                    flags = [
                        str(f)
                        for f in json.loads(str(latest_assess.get("flags_json") or "[]"))
                    ]
                    priority = latest_assess.get("review_priority")
                    recommended_action = str(latest_assess.get("recommended_action") or "")
                review_count = 0
                if has_reviews:
                    rc = connection.execute(
                        "SELECT COUNT(*) AS n FROM review_decisions WHERE inspection_id = ?",
                        (iid,),
                    ).fetchone()
                    review_count = int(rc["n"]) if rc is not None else 0
                items.append(
                    {
                        "id": iid,
                        "run_id": str(row["run_id"]),
                        "sample_id": str(row["sample_id"]),
                        "status": str(row["status"]),
                        "score": float(row["score"]),
                        "engine": str(row["engine"]),
                        "task": str(row.get("task") or ""),
                        "label": str(row["label"]),
                        "split": str(row["split"]),
                        "reasons": reasons,
                        "fingerprint": str(row.get("inspection_fingerprint") or ""),
                        "provenance_complete": bool(row.get("provenance_complete", 0)),
                        "has_overlay": row_overlay_path(row, runs_dir) is not None,
                        "obs_count": obs_count,
                        "assessment_count": len(assessments),
                        "flags": flags,
                        "review_priority": priority,
                        "recommended_action": recommended_action,
                        "review_count": review_count,
                        "created_at": str(row["created_at"]),
                    }
                )
        finally:
            connection.close()
    except (
        sqlite3.Error,
        DatabaseError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return {**base, "message": f"검사 기록 DB를 읽지 못했습니다: {exc}"}

    return {
        **base,
        "available": True,
        "items": items,
        "total": total,
        "all_total": all_total,
        "engines": engines,
        "status_counts": status_counts,
    }


def load_inspection_detail_view(
    db_path: Path,
    runs_dir: Path,
    inspection_id: int,
) -> dict[str, Any]:
    """단일 검사 DB 행 + 엔진 관찰 + 교차 증거·패치 중재·논증 + 리뷰 이력(읽기 전용)."""

    if not db_path.is_file():
        raise UiInputError("검사 기록 DB가 없습니다.")
    try:
        connection = read_only_connection(db_path)
        try:
            row = get_inspection(connection, inspection_id)
            if row is None:
                raise UiInputError(f"검사 기록을 찾을 수 없습니다: id={inspection_id}")

            tables = {
                str(r[0])
                for r in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            observations_raw = (
                list_engine_observations(connection, inspection_id)
                if "engine_observations" in tables
                else []
            )
            assessments_raw = (
                list_agent_assessments(connection, inspection_id)
                if "agent_assessments" in tables
                else []
            )
            reviews_raw = (
                [
                    dict(r)
                    for r in connection.execute(
                        "SELECT id, decision, reviewer, reason, created_at FROM review_decisions "
                        "WHERE inspection_id = ? ORDER BY id ASC",
                        (int(inspection_id),),
                    ).fetchall()
                ]
                if "review_decisions" in tables
                else []
            )
        finally:
            connection.close()

        observations: list[dict[str, Any]] = []
        for obs in observations_raw:
            obs_reasons = json.loads(str(obs.get("reasons") or "[]"))
            obs_comps = json.loads(str(obs.get("components") or "[]"))
            obs_timing = json.loads(str(obs.get("timing_ms") or "{}"))
            if (
                not isinstance(obs_reasons, list)
                or not isinstance(obs_comps, list)
                or not isinstance(obs_timing, dict)
            ):
                raise UiInputError("관찰 JSON 컬럼 형식이 손상됐습니다.")
            observations.append(
                {
                    "id": int(obs["id"]),
                    "engine": str(obs["engine"]),
                    "task": str(obs.get("task") or ""),
                    "status": str(obs["status"]),
                    "score": float(obs["score"]),
                    "reasons": [str(x) for x in obs_reasons],
                    "components": obs_comps,
                    "model_sha256": str(obs.get("model_sha256") or ""),
                    "error_code": str(obs.get("error_code") or ""),
                    "timing_ms": obs_timing,
                    "created_at": str(obs["created_at"]),
                }
            )

        assessments: list[dict[str, Any]] = []
        for item in assessments_raw:
            diag = json.loads(str(item.get("diagnostics_json") or "{}"))
            if not isinstance(diag, dict):
                diag = {}
            flags_raw = json.loads(str(item.get("flags_json") or "[]"))
            if not isinstance(flags_raw, list):
                raise UiInputError("평가 flags_json 컬럼 형식이 손상됐습니다.")
            assessments.append(
                {
                    "id": int(item["id"]),
                    "assessment_hash": str(item["assessment_hash"]),
                    "policy_hash": str(item["policy_hash"]),
                    "flags": [str(f) for f in flags_raw],
                    "review_priority": item.get("review_priority"),
                    "stability": item.get("stability"),
                    "recommended_action": str(item["recommended_action"]),
                    "diagnostics": diag,
                    "trace_path": str(item.get("trace_path") or ""),
                    "created_at": str(item["created_at"]),
                }
            )

        reasons_list_raw = json.loads(str(row.get("reasons") or "[]"))
        timing_dict_raw = json.loads(str(row.get("timing_ms") or "{}"))
        if not isinstance(reasons_list_raw, list) or not isinstance(timing_dict_raw, dict):
            raise UiInputError("검사 기록 JSON 컬럼 형식이 손상됐습니다.")

        return {
            "inspection": {
                **row,
                "reasons_list": [str(x) for x in reasons_list_raw],
                "timing_dict": timing_dict_raw,
                "has_overlay": row_overlay_path(row, runs_dir) is not None,
            },
            "observations": observations,
            "assessments": assessments,
            "reviews": reviews_raw,
        }
    except UiInputError:
        raise
    except (
        sqlite3.Error,
        DatabaseError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise UiInputError(f"검사 기록 DB를 읽지 못했습니다: {exc}") from exc


__all__ = [
    "RunEntry",
    "UiInputError",
    "discover_runs",
    "find_sample",
    "load_inspection_detail_view",
    "load_inspections_view",
    "load_manifest_map",
    "load_results",
    "load_run",
    "read_only_connection",
    "row_overlay_path",
    "safe_child",
    "sample_result_from_record",
    "sha256_file",
]
