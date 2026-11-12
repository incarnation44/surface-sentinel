"""검사 기록 저장소 (E5, ADR 0007).

규칙:

* ``inspections.status``는 **기계 판정이며 불변**이다. 사람 판정은
  ``review_decisions``에 append-only로만 쌓고 기계 판정을 UPDATE 하지 않는다.
* ``image_sha256`` UNIQUE가 중복제거 키다. 같은 이미지를 다시 넣으면 기존
  행의 ``id``를 그대로 돌려주고 새 행을 만들지 않는다.
* 모든 쓰기는 트랜잭션 하나로 끝낸다(``with connection:``).
* 통계의 모든 비율은 :class:`~surface_sentinel.evaluation.metrics.Ratio`로
  분모를 함께 보고한다. 분모가 0이면 값은 ``None``이다.

파일 산출물(오버레이 PNG, 결과 JSON)의 원자 저장은 파이프라인이 이미
수행한다(``pipeline.evidence`` / ``io.atomic``). 이 모듈은 DB만 다룬다.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ..contracts import SampleResult, Status
from ..evaluation.metrics import Ratio, ratio

#: 허용 엔진 이름. DB CHECK 제약 대신 애플리케이션에서 검증한다(PG 이식 대비).
ENGINES: tuple[str, ...] = ("opencv", "yolo-det", "yolo-seg", "combined")

#: 사람 판정 값. 기계 판정의 REVIEW/ERROR는 사람 판정 값이 아니다.
DECISIONS: tuple[str, ...] = ("OK", "NG")

STATUS_ORDER: tuple[str, ...] = ("OK", "NG", "REVIEW", "ERROR")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RecordsError(ValueError):
    """기록 저장소 계약을 위반하는 입력이 들어왔을 때 발생한다."""


def utc_now_iso() -> str:
    """현재 시각을 ISO8601 UTC(``YYYY-MM-DDTHH:MM:SSZ``)로 만든다."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_sha256(value: str) -> str:
    """sha256 문자열을 검증하고 소문자로 정규화한다."""

    text = str(value).strip().lower()
    if not _SHA256_RE.match(text):
        raise RecordsError(f"sha256 형식이 아닙니다: {value!r}")
    return text


def _status_value(value: Status | str) -> str:
    text = value.value if isinstance(value, Status) else str(value)
    if text not in STATUS_ORDER:
        raise RecordsError(f"알 수 없는 기계 판정입니다: {value!r} (가능: {STATUS_ORDER})")
    return text


def _decision_value(value: str) -> str:
    text = str(value).strip()
    if text not in DECISIONS:
        raise RecordsError(f"사람 판정은 OK/NG만 허용합니다: {value!r}")
    return text


def _reasons_json(result: SampleResult) -> str:
    return json.dumps([reason.value for reason in result.reasons], ensure_ascii=False)


def _timing_json(result: SampleResult) -> str:
    return json.dumps(dict(result.timing_ms), ensure_ascii=False, sort_keys=True)


def insert_inspection(
    connection: sqlite3.Connection,
    result: SampleResult,
    *,
    engine: str,
    image_sha256: str,
    model_version: str = "",
) -> int:
    """검사 결과 한 건을 기록하고 행 ``id``를 돌려준다.

    ``image_sha256``이 이미 있으면 새 행을 만들지 않고 **기존 행의 id**를
    돌려준다(중복제거). 기존 행은 덮어쓰지 않는다.
    """

    sha = normalize_sha256(image_sha256)
    engine_name = str(engine).strip()
    if engine_name not in ENGINES:
        raise RecordsError(f"알 수 없는 엔진입니다: {engine!r} (가능: {ENGINES})")
    status = _status_value(result.status)

    existing = connection.execute(
        "SELECT id FROM inspections WHERE image_sha256 = ?",
        (sha,),
    ).fetchone()
    if existing is not None:
        return int(existing["id"])

    params = (
        str(result.run_id),
        str(result.sample_id),
        sha,
        str(result.label),
        str(result.split),
        status,
        float(result.score),
        _reasons_json(result),
        engine_name,
        str(result.config_hash),
        str(model_version),
        str(result.overlay_path),
        _timing_json(result),
        utc_now_iso(),
    )
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO inspections ("
                "run_id, sample_id, image_sha256, label, split, status, score, reasons,"
                " engine, config_hash, model_version, overlay_path, timing_ms, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )
    except sqlite3.IntegrityError:
        # 동시 삽입으로 UNIQUE가 걸린 경우에도 중복제거 계약을 지킨다.
        row = connection.execute(
            "SELECT id FROM inspections WHERE image_sha256 = ?",
            (sha,),
        ).fetchone()
        if row is None:
            raise
        return int(row["id"])
    return int(cursor.lastrowid or 0)


def append_review(
    connection: sqlite3.Connection,
    inspection_id: int,
    decision: str,
    reviewer: str = "",
    reason: str = "",
) -> int:
    """사람 판정을 append-only로 남기고 ``review_decisions.id``를 돌려준다.

    기계 판정(``inspections.status``)은 절대 바꾸지 않는다.
    """

    decision_value = _decision_value(decision)
    row = connection.execute(
        "SELECT id FROM inspections WHERE id = ?",
        (int(inspection_id),),
    ).fetchone()
    if row is None:
        raise RecordsError(f"검사 기록이 없습니다: inspection_id={inspection_id}")

    with connection:
        cursor = connection.execute(
            "INSERT INTO review_decisions "
            "(inspection_id, decision, reviewer, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                int(inspection_id),
                decision_value,
                str(reviewer).strip(),
                str(reason).strip(),
                utc_now_iso(),
            ),
        )
    return int(cursor.lastrowid or 0)


def find_inspection_id(
    connection: sqlite3.Connection,
    image_sha256: str,
) -> int | None:
    """sha256으로 기존 행 ``id``를 찾는다. 없으면 ``None``."""

    row = connection.execute(
        "SELECT id FROM inspections WHERE image_sha256 = ?",
        (normalize_sha256(image_sha256),),
    ).fetchone()
    return None if row is None else int(row["id"])


def list_inspections(
    connection: sqlite3.Connection,
    *,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """검사 기록을 ``id`` 오름차순으로 조회한다(결정론적 순서)."""

    if limit < 0 or offset < 0:
        raise RecordsError(f"limit/offset은 0 이상이어야 합니다: limit={limit} offset={offset}")
    sql = "SELECT * FROM inspections"
    params: list[Any] = []
    if status is not None:
        params.append(_status_value(status))
        sql += " WHERE status = ?"
    sql += " ORDER BY id ASC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    return [dict(row) for row in connection.execute(sql, params).fetchall()]


def get_inspection(
    connection: sqlite3.Connection,
    inspection_id: int,
) -> dict[str, Any] | None:
    """행 하나를 돌려준다. 없으면 ``None``."""

    row = connection.execute(
        "SELECT * FROM inspections WHERE id = ?",
        (int(inspection_id),),
    ).fetchone()
    return None if row is None else dict(row)


def _scalar(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = connection.execute(sql, params).fetchone()
    return 0 if row is None else int(row[0])


def stats(connection: sqlite3.Connection) -> dict[str, Any]:
    """상태 분포·리뷰·오버레이 통계를 **분모와 함께** 돌려준다."""

    counts = {
        str(row["status"]): int(row["n"])
        for row in connection.execute(
            "SELECT status, COUNT(*) AS n FROM inspections GROUP BY status"
        ).fetchall()
    }
    by_status = {name: counts.get(name, 0) for name in STATUS_ORDER}
    total = sum(by_status.values())

    decisions = _scalar(connection, "SELECT COUNT(*) FROM review_decisions")
    reviewed = _scalar(
        connection,
        "SELECT COUNT(DISTINCT inspection_id) FROM review_decisions",
    )
    overlays = _scalar(
        connection,
        "SELECT COUNT(*) FROM inspections WHERE overlay_path <> ''",
    )
    agree = _scalar(
        connection,
        "SELECT COUNT(*) FROM ("
        "  SELECT r.decision AS decision, i.status AS status"
        "  FROM review_decisions r"
        "  JOIN inspections i ON i.id = r.inspection_id"
        "  WHERE r.id IN (SELECT MAX(id) FROM review_decisions GROUP BY inspection_id)"
        ") WHERE decision = status",
    )

    ratios: dict[str, Ratio] = {
        name.lower(): ratio(f"검사 전체 중 {name} 비율", by_status[name], total)
        for name in STATUS_ORDER
    }
    ratios["reviewed"] = ratio("검사 전체 중 사람 판정이 있는 비율", reviewed, total)
    ratios["agreement"] = ratio(
        "사람 판정 중 기계 판정과 같은 비율(REVIEW/ERROR는 불일치로 센다)",
        agree,
        reviewed,
    )
    ratios["overlay"] = ratio("검사 전체 중 오버레이를 남긴 비율", overlays, total)

    return {
        "total": total,
        "by_status": by_status,
        "reviews": {"decisions": decisions, "reviewed_inspections": reviewed},
        "overlays": overlays,
        "ratios": ratios,
    }


def stats_payload(connection: sqlite3.Connection) -> dict[str, Any]:
    """``stats`` 결과를 JSON 직렬화 가능한 dict로 바꾼다(CLI·보고서용)."""

    payload = stats(connection)
    ratios = payload["ratios"]
    return {
        "total": payload["total"],
        "by_status": payload["by_status"],
        "reviews": payload["reviews"],
        "overlays": payload["overlays"],
        "ratios": {name: value.as_dict() for name, value in ratios.items()},
    }


__all__ = [
    "DECISIONS",
    "ENGINES",
    "RecordsError",
    "STATUS_ORDER",
    "append_review",
    "find_inspection_id",
    "get_inspection",
    "insert_inspection",
    "list_inspections",
    "normalize_sha256",
    "stats",
    "stats_payload",
    "utc_now_iso",
]
