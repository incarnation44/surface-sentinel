"""SQLite 연결과 스키마 초기화 (E5, ADR 0007).

스키마 정본은 ``records/schema.sql`` 한 파일이다. 파이썬 코드에 DDL을 중복해
두지 않으며, ``importlib.resources``로 읽어 wheel 설치본에서도 같은 파일을
쓴다(패키지 데이터).
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

SCHEMA_PACKAGE = "surface_sentinel.records"
SCHEMA_RESOURCE = "schema.sql"


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


def init_db(connection: sqlite3.Connection) -> None:
    """``schema.sql``을 실행한다. 여러 번 호출해도 안전하다(IF NOT EXISTS)."""

    connection.executescript(load_schema_sql())
    connection.commit()


__all__ = [
    "DatabaseError",
    "SCHEMA_PACKAGE",
    "SCHEMA_RESOURCE",
    "connect",
    "init_db",
    "load_schema_sql",
]
