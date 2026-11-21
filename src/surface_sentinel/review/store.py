"""append-only 검토 결정 저장소 (설계 §5.3, T07).

``review_decisions.jsonl``은 한 줄에 결정 하나를 추가만 한다. 기존 줄은
수정·삭제하지 않으며, 같은 샘플의 마지막 결정이 최종 상태다. 기계 판정이
``REVIEW``인 샘플만 사람이 덮어쓸 수 있고 ``OK``/``NG``는 불변이다.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_DECISIONS_JSONL = "review_decisions.jsonl"

#: 메모(note) 최대 길이(문자 수). 로그 비대화·남용을 막는다.
MAX_NOTE_LENGTH = 2000

#: 검토자 이름(reviewer) 최대 길이(문자 수).
MAX_REVIEWER_LENGTH = 128

_APPEND_LOCKS_GUARD = threading.Lock()
_APPEND_LOCKS: dict[Path, threading.Lock] = {}


def _append_lock_for(path: Path) -> threading.Lock:
    key = path.resolve()
    with _APPEND_LOCKS_GUARD:
        lock = _APPEND_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _APPEND_LOCKS[key] = lock
        return lock

#: ``review_decisions.jsonl`` 필드 순서. 설계 §5.3과 1:1로 고정한다.
DECISION_FIELDS: tuple[str, ...] = (
    "decision_id",
    "run_id",
    "sample_id",
    "machine_status",
    "final_status",
    "reason_code",
    "note",
    "reviewer",
    "decided_at",
    "config_hash",
    "overlay_sha256",
)

#: 사람이 고를 수 있는 최종 상태. 보류는 ``REVIEW``로 기록한다.
FINAL_STATUSES: tuple[str, ...] = ("OK", "NG", "REVIEW")

#: 검토 사유 어휘. 설계 §5.3은 필드만 고정하므로 어휘는 T07에서 정의한다.
#: ``OTHER``는 메모가 필수다.
REVIEW_REASON_CODES: tuple[str, ...] = (
    "CONFIRMED_DEFECT",
    "FALSE_ALARM",
    "INSUFFICIENT_EVIDENCE",
    "IMAGE_QUALITY",
    "OTHER",
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class ReviewError(ValueError):
    """잘못된 상태 전이·스키마 위반 등 거부된 결정 (HTTP 400)."""


class ReviewStorageError(RuntimeError):
    """감사 로그 append 실패. T07 중단 조건이므로 숨기지 않는다 (HTTP 500)."""


def _iso_utc(moment: datetime) -> str:
    return (
        moment.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class ReviewDecision:
    """검토 결정 한 건. 필드 순서는 ``DECISION_FIELDS``와 같다."""

    decision_id: str
    run_id: str
    sample_id: str
    machine_status: str
    final_status: str
    reason_code: str
    note: str
    reviewer: str
    decided_at: str
    config_hash: str
    overlay_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in DECISION_FIELDS}

    def validate(self) -> None:
        """계약 위반이면 ``ReviewError``. append 전에 반드시 통과해야 한다."""

        if self.machine_status != "REVIEW":
            raise ReviewError(
                f"기계 판정 REVIEW만 검토로 덮어쓸 수 있습니다 "
                f"(OK/NG 불변, 받은 값: {self.machine_status!r})."
            )
        if self.final_status not in FINAL_STATUSES:
            raise ReviewError(
                f"final_status는 {FINAL_STATUSES} 중 하나여야 합니다: {self.final_status!r}"
            )
        if self.reason_code not in REVIEW_REASON_CODES:
            raise ReviewError(
                f"reason_code는 {REVIEW_REASON_CODES} 중 하나여야 합니다: {self.reason_code!r}"
            )
        if self.reason_code == "OTHER" and not self.note.strip():
            raise ReviewError("OTHER 사유에는 메모(note)가 필요합니다.")
        if not self.sample_id.strip():
            raise ReviewError("sample_id가 비어 있습니다.")
        if not self.decision_id.strip():
            raise ReviewError("decision_id가 비어 있습니다.")
        if not self.run_id.strip():
            raise ReviewError("run_id가 비어 있습니다.")
        if not self.reviewer.strip():
            raise ReviewError("검토자(reviewer)가 비어 있습니다.")
        if len(self.reviewer) > MAX_REVIEWER_LENGTH:
            raise ReviewError(
                f"검토자(reviewer)는 최대 {MAX_REVIEWER_LENGTH}자까지 입력할 수 있습니다."
            )
        if len(self.note) > MAX_NOTE_LENGTH:
            raise ReviewError(
                f"메모(note)는 최대 {MAX_NOTE_LENGTH}자까지 입력할 수 있습니다."
            )
        if not _SHA256_HEX.fullmatch(self.config_hash):
            raise ReviewError("config_hash는 sha256 hex 64자여야 합니다.")
        if not _SHA256_HEX.fullmatch(self.overlay_sha256):
            raise ReviewError("overlay_sha256는 sha256 hex 64자여야 합니다.")
        if not self.decided_at.endswith("Z"):
            raise ReviewError("decided_at은 UTC(Z) 표기여야 합니다.")

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> ReviewDecision:
        missing = [name for name in DECISION_FIELDS if name not in payload]
        extra = [name for name in payload if name not in DECISION_FIELDS]
        if missing or extra:
            raise ReviewError(
                f"결정 스키마가 계약과 다릅니다: 누락={missing} 초과={extra}"
            )
        values = {name: payload[name] for name in DECISION_FIELDS}
        if not all(isinstance(value, str) for value in values.values()):
            raise ReviewError("결정 스키마의 모든 값은 문자열이어야 합니다.")
        decision = ReviewDecision(**values)
        decision.validate()
        return decision


class ReviewStore:
    """run 디렉터리 하나의 검토 결정 로그. 쓰기는 append뿐이다."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    @property
    def path(self) -> Path:
        return self.run_dir / REVIEW_DECISIONS_JSONL

    def read_all(self) -> list[ReviewDecision]:
        """저장된 결정을 기록 순서대로 읽는다. 파일이 없으면 빈 목록."""

        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReviewStorageError(f"검토 로그를 읽지 못했습니다: {exc}") from exc
        decisions: list[ReviewDecision] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"검토 로그 {number}번째 줄이 손상됐습니다.") from exc
            if not isinstance(payload, dict):
                raise ReviewError(f"검토 로그 {number}번째 줄이 객체가 아닙니다.")
            decisions.append(ReviewDecision.from_dict(payload))
        return decisions

    def latest(self) -> dict[str, ReviewDecision]:
        """샘플별 마지막 결정. 마지막 결정이 최종 상태다 (설계 §5.3)."""

        latest: dict[str, ReviewDecision] = {}
        for decision in self.read_all():
            latest[decision.sample_id] = decision
        return latest

    def append(
        self,
        *,
        sample_id: str,
        machine_status: str,
        final_status: str,
        reason_code: str,
        reviewer: str,
        config_hash: str,
        overlay_sha256: str,
        note: str = "",
        decision_id: str | None = None,
        decided_at: str | None = None,
    ) -> ReviewDecision:
        """결정 한 건을 검증한 뒤 파일 끝에 추가하고 fsync한다.

        검증 실패는 ``ReviewError``로 거부되고 파일은 변하지 않는다.
        쓰기 실패는 ``ReviewStorageError``로 드러난다 (감사 append 실패 은폐 금지).
        """

        if not self.run_dir.is_dir():
            raise ReviewError(f"run 디렉터리가 없습니다: {self.run_dir}")
        decision = ReviewDecision(
            decision_id=decision_id or uuid.uuid4().hex,
            run_id=self.run_dir.name,
            sample_id=sample_id,
            machine_status=machine_status,
            final_status=final_status,
            reason_code=reason_code,
            note=note.strip(),
            reviewer=reviewer.strip(),
            decided_at=decided_at or _iso_utc(datetime.now(UTC)),
            config_hash=config_hash,
            overlay_sha256=overlay_sha256,
        )
        decision.validate()
        line = json.dumps(decision.to_dict(), ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        try:
            with _append_lock_for(self.path), self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ReviewStorageError(f"검토 결정을 추가하지 못했습니다: {exc}") from exc
        return decision


__all__ = [
    "DECISION_FIELDS",
    "FINAL_STATUSES",
    "MAX_NOTE_LENGTH",
    "MAX_REVIEWER_LENGTH",
    "REVIEW_DECISIONS_JSONL",
    "REVIEW_REASON_CODES",
    "ReviewDecision",
    "ReviewError",
    "ReviewStorageError",
    "ReviewStore",
]