"""SQLite 연결과 스키마 초기화 (E5, ADR 0007).

스키마 정본은 ``records/schema.sql`` 한 파일이다. 파이썬 코드에 DDL을 중복해
두지 않으며, ``importlib.resources``로 읽어 wheel 설치본에서도 같은 파일을
쓴다(패키지 데이터).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .fingerprint import inspection_fingerprint

SCHEMA_PACKAGE = "surface_sentinel.records"
SCHEMA_RESOURCE = "schema.sql"

# append-only 트리거 정본은 schema.sql 의 마커 구간이다(아래 두 상수로 추출).
TRIGGER_SECTION_BEGIN = "-- APPEND-ONLY TRIGGERS (SQLite 전용) BEGIN"
TRIGGER_SECTION_END = "-- END APPEND-ONLY TRIGGERS"

# 필수 트리거 8개(테이블 4개 × UPDATE/DELETE). schema.sql 과 이름이 같아야 한다.
APPEND_ONLY_TRIGGER_NAMES = (
    "trg_inspections_no_update",
    "trg_inspections_no_delete",
    "trg_review_decisions_no_update",
    "trg_review_decisions_no_delete",
    "trg_engine_observations_no_update",
    "trg_engine_observations_no_delete",
    "trg_agent_assessments_no_update",
    "trg_agent_assessments_no_delete",
)


class DatabaseError(RuntimeError):
    """DB 연결·스키마 초기화를 완료하지 못했을 때 발생한다."""


def load_schema_sql() -> str:
    """패키지에 포함된 ``schema.sql`` 원문을 돌려준다."""

    resource = resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_RESOURCE)
    return resource.read_text(encoding="utf-8")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """DB에 연결하고 WAL·외래키를 켠다.

    부모 디렉터리가 없으면 만든다. 반환된 연결은 ``row_factory``가
    :class:`sqlite3.Row`라 컬럼 이름으로 값을 읽는다.
    """

    path = Path(db_path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(str(path))
    except sqlite3.Error as exc:
        raise DatabaseError(f"DB를 열지 못했습니다: {path} ({exc})") from exc
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _schema_table_block(table: str) -> str:
    """``schema.sql`` 에서 테이블 하나의 CREATE 블록을 추출한다(DDL 정본 단일화)."""

    sql = load_schema_sql()
    marker = f"CREATE TABLE IF NOT EXISTS {table}"
    start = sql.index(marker)
    end = sql.index(");", start) + 2
    return sql[start:end]


def append_only_trigger_sql() -> str:
    """``schema.sql`` 의 append-only 트리거 구간만 추출한다(DDL 정본 단일화).

    트리거 DDL 도 스키마 정본 한 곳에만 둔다. 마이그레이션으로 테이블을
    DROP/RENAME 하면 그 테이블의 트리거가 함께 사라지므로, 여기서 꺼내 다시 건다.
    """

    sql = load_schema_sql()
    start = sql.index(TRIGGER_SECTION_BEGIN)
    end = sql.index(TRIGGER_SECTION_END, start)
    return sql[start:end]


def ensure_append_only_triggers(connection: sqlite3.Connection) -> None:
    """필수 append-only 트리거를 (재)설치한다(멱등).

    ``executescript`` 는 실행 전에 열린 트랜잭션을 커밋하므로, 마이그레이션
    트랜잭션 **안에서** 호출하지 않는다. 마이그레이션이 끝난 뒤 호출한다.
    """

    connection.executescript(append_only_trigger_sql())


@dataclass(frozen=True)
class RecordsIntegrityReport:
    """``verify_records_integrity`` 결과(불변 값 객체)."""

    integrity_check: str
    foreign_key_violations: int
    missing_triggers: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """무결성·외래키·트리거가 모두 정상인가."""

        return (
            self.integrity_check == "ok"
            and self.foreign_key_violations == 0
            and not self.missing_triggers
        )


def verify_records_integrity(connection: sqlite3.Connection) -> RecordsIntegrityReport:
    """기록 DB 의 무결성·외래키·append-only 트리거를 검사한다.

    ``PRAGMA integrity_check``, ``PRAGMA foreign_key_check``, 필수 트리거 8개
    존재를 확인해 :class:`RecordsIntegrityReport` 로 돌려준다(쓰기 없음).
    """

    integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
    integrity = str(integrity_row[0]) if integrity_row is not None else "unknown"
    violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
    present = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
    }
    missing = tuple(name for name in APPEND_ONLY_TRIGGER_NAMES if name not in present)
    return RecordsIntegrityReport(
        integrity_check=integrity,
        foreign_key_violations=violations,
        missing_triggers=missing,
    )


def _inspections_v2_ddl() -> str:
    """``schema.sql``의 inspections 정의를 inspections_v2 로 이름만 바꿔 돌려준다.

    DDL 을 파이썬에 중복하지 않고 스키마 정본에서 추출한다(마이그레이션용).
    """

    return _schema_table_block("inspections").replace(
        "CREATE TABLE IF NOT EXISTS inspections", "CREATE TABLE inspections_v2", 1
    )


def migrate_to_fingerprint(connection: sqlite3.Connection) -> bool:
    """v1(이미지 단독 UNIQUE) → v2(이미지+fingerprint UNIQUE) 마이그레이션 (P0-2).

    기존 ``inspections``·``review_decisions`` 의 id 를 그대로 보존한다. 과거 행은
    실제 모델 SHA 를 알 수 없으므로 ``provenance_complete=0`` 으로 backfill 하고
    모델 SHA 를 추측해 채우지 않는다. 이미 v2 이거나 테이블이 없으면 no-op.
    무결성(``integrity_check``)·외래키(``foreign_key_check``) 검증 실패 시 예외.
    """

    columns = {row[1] for row in connection.execute("PRAGMA table_info(inspections)")}
    if not columns or "inspection_fingerprint" in columns:
        return False

    connection.execute("PRAGMA foreign_keys=OFF")
    try:
        with connection:
            connection.execute(_inspections_v2_ddl())
            connection.execute(
                "INSERT INTO inspections_v2 ("
                "id, run_id, sample_id, image_sha256, inspection_fingerprint, label, split,"
                " status, score, reasons, engine, task, config_hash, model_sha256,"
                " model_version, provenance_complete, overlay_path, timing_ms, created_at"
                ") SELECT id, run_id, sample_id, image_sha256, '', label, split,"
                " status, score, reasons, engine, '', config_hash, '',"
                " model_version, 0, overlay_path, timing_ms, created_at FROM inspections"
            )
            for row in connection.execute(
                "SELECT id, engine, config_hash FROM inspections_v2"
            ).fetchall():
                fingerprint = inspection_fingerprint(
                    engine=str(row["engine"]), config_hash=str(row["config_hash"])
                )
                connection.execute(
                    "UPDATE inspections_v2 SET inspection_fingerprint = ? WHERE id = ?",
                    (fingerprint, int(row["id"])),
                )
            connection.execute("DROP TABLE inspections")
            connection.execute("ALTER TABLE inspections_v2 RENAME TO inspections")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(status)"
            )
            # 엔진별 관찰 테이블도 보장한다(P0-3, 과거 행은 관찰 없음 → 빈 테이블).
            connection.execute(_schema_table_block("engine_observations"))
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_inspection "
                "ON engine_observations(inspection_id)"
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or violations:
            raise DatabaseError(
                "마이그레이션 검증 실패: "
                f"integrity={integrity}, foreign_key_violations={len(violations)}"
            )
    finally:
        connection.execute("PRAGMA foreign_keys=ON")
    return True


def init_db(connection: sqlite3.Connection) -> None:
    """``schema.sql``을 실행하고 v1→v2 마이그레이션을 적용한다(여러 번 호출 안전).

    마이그레이션은 ``inspections`` 를 DROP/RENAME 하므로 그 테이블의 트리거가
    함께 사라진다. 그래서 마이그레이션 **뒤에** append-only 트리거를 다시 건다.
    """

    connection.executescript(load_schema_sql())
    migrate_to_fingerprint(connection)
    ensure_append_only_triggers(connection)
    connection.commit()


__all__ = [
    "APPEND_ONLY_TRIGGER_NAMES",
    "DatabaseError",
    "RecordsIntegrityReport",
    "SCHEMA_PACKAGE",
    "SCHEMA_RESOURCE",
    "TRIGGER_SECTION_BEGIN",
    "TRIGGER_SECTION_END",
    "append_only_trigger_sql",
    "connect",
    "ensure_append_only_triggers",
    "init_db",
    "load_schema_sql",
    "migrate_to_fingerprint",
    "verify_records_integrity",
]
