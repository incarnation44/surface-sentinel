"""결정론 워커 (E5, ADR 0007 / D6).

``incoming`` 디렉터리를 감시해 이미지를 검사하고 결과를 SQLite에 기록한다.

결정론 규칙:

* 파일은 **이름 정렬 순**으로 처리한다(파일 시스템 순서에 의존하지 않는다).
* 같은 ``(image_sha256, inspection_fingerprint)`` 쌍은 한 번만 검사한다(DB UNIQUE 중복제거).
* 검사가 ``ERROR``로 끝나면 ``max_retries``회까지 다시 시도하고 마지막 결과를
  ``ERROR``로 기록한다.
* 처리/스킵/에러는 ``.work/worker_audit.jsonl``에 append-only로 남긴다.

금지: 네트워크/셸/핫패스 subprocess. 판정 소유권은 파이프라인에 있으며 이
모듈은 판정을 바꾸지 않는다. 코드 신원(git)은 ``run_worker`` 시작 시 1회만
계산한다(이미지별 재계산 금지). 봉인 분할(test/official_test)은 절대 열지 않는다
(항상 ``split="tune"``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..contracts import SampleResult, Status
from ..pipeline import InspectionOutcome, inspect_image
from ..pipeline.evidence import sanitize_sample_id
from ..records.db import connect, init_db
from ..records.identity import EngineSpec, opencv_spec
from ..records.store import (
    ENGINES,
    append_agent_assessment,
    find_inspection_id,
    insert_engine_observation,
    insert_inspection,
    utc_now_iso,
)
from .evidence import assess_cross_evidence

#: 감시 대상 확장자(소문자 비교).
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: 감사 로그 위치: ``<db 부모>/.work/worker_audit.jsonl``.
AUDIT_DIRNAME = ".work"
AUDIT_FILENAME = "worker_audit.jsonl"

#: 오버레이 보관 위치: ``<db 부모>/worker_overlays/<run_id>/``.
OVERLAY_DIRNAME = "worker_overlays"

#: 워커가 쓰는 분할. 봉인 분할은 열지 않는다.
WORKER_SPLIT = "tune"

YOLO_ENGINES: tuple[str, ...] = tuple(name for name in ENGINES if name != "opencv")


class WorkerError(RuntimeError):
    """워커 인자나 실행 환경이 계약을 어겼을 때 발생한다."""


def audit_log_path(db_path: str | Path) -> Path:
    """감사 로그 경로(``.work/worker_audit.jsonl``)를 돌려준다."""

    return Path(db_path).parent / AUDIT_DIRNAME / AUDIT_FILENAME


def overlay_dir(db_path: str | Path, run_id: str) -> Path:
    """run 하나의 오버레이 디렉터리를 돌려준다."""

    return Path(db_path).parent / OVERLAY_DIRNAME / run_id


def new_run_id(config: AppConfig) -> str:
    """실행 단위 ``run_id``를 결정론적 형식으로 만든다(``YYYYMMDDTHHMMSSZ-<hash8>``)."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config.config_hash[:8]}"


def file_sha256(path: str | Path) -> str:
    """파일 바이트의 sha256을 스트리밍으로 계산한다."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keep_overlay(status: Status, sha256: str, ok_sample_rate: int) -> bool:
    """D7: NG/REVIEW는 전량, OK는 해시 기반 1/ok_sample_rate 표본만 남긴다.

    ``ok_sample_rate``가 0이면 OK는 남기지 않고, 1이면 전량 남긴다.
    """

    if status is not Status.OK:
        return True
    if ok_sample_rate <= 0:
        return False
    if ok_sample_rate == 1:
        return True
    return int(sha256[:8], 16) % ok_sample_rate == 0


def _iter_images(incoming: Path) -> Iterator[Path]:
    """이미지 파일을 이름 정렬 순으로 돌려준다(결정론)."""

    candidates = [
        item
        for item in incoming.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    yield from sorted(candidates, key=lambda item: item.name)


def _append_audit(path: Path, event: dict[str, Any]) -> None:
    """감사 이벤트 한 줄을 append-only로 남긴다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": utc_now_iso(), **event}, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_yolo_entry(engine: str) -> Callable[..., InspectionOutcome]:
    """YOLO 엔진 진입점을 **지연 import**로 찾는다.

    ``yolo/inspect.py``(E5b 통합 계약)에서 ``inspect_with_yolo`` 를 가져온다.
    import 는 엔진이 요청될 때만 수행하므로, opencv 경로는 ONNX Runtime extra 가
    없어도 동작한다. 진입점이 없으면 ``WorkerError``로 멈춘다(조용히 opencv로
    폴백하지 않는다).
    """

    try:
        module = importlib.import_module("surface_sentinel.yolo.inspect")
    except ImportError as exc:  # pragma: no cover - yolo 레인 의존
        raise WorkerError(
            f"{engine} 엔진 경로가 아직 없습니다(surface_sentinel.yolo.inspect 미구현). "
            "opencv 엔진으로 실행하세요."
        ) from exc
    entry = getattr(module, "inspect_with_yolo", None)
    if not callable(entry):  # pragma: no cover - yolo 레인 의존
        raise WorkerError(
            f"{engine} 엔진 진입점(inspect_with_yolo)을 찾지 못했습니다."
        )
    return entry


def _model_version(engine: str) -> str:
    """기록에 남길 모델/엔진 버전 문자열."""

    if engine == "opencv":
        import cv2

        return f"opencv-{cv2.__version__}"
    return engine


def _resolve_spec(
    engine: str,
    config: AppConfig,
    *,
    code_sha: str = "",
    code_dirty: bool = False,
) -> EngineSpec:
    """검사 신원 EngineSpec 을 진입점에서 1회 계산한다(P0-2/P0-3).

    opencv 는 기준선 신원. YOLO 계열은 ``yolo.inspect.inspection_identity`` 로
    실제 모델 SHA·정책 해시를 계산한다(API·평가와 같은 정의). 워커 핫패스가 아니라
    ``run_worker`` 시작 시 1회만 호출한다. ``code_sha``/``code_dirty`` 는
    ``run_worker`` 가 시작 시 1회 계산해 주입한다(핫패스 subprocess 금지).
    모델 부재·combined task 미설정은 명확한 오류로 멈춘다.
    """

    if engine == "opencv":
        return opencv_spec(config.config_hash, code_sha=code_sha, code_dirty=code_dirty)
    try:
        module = importlib.import_module("surface_sentinel.yolo.inspect")
    except ImportError as exc:  # pragma: no cover - yolo 레인 의존
        raise WorkerError(
            f"{engine} 엔진 경로가 없습니다(surface_sentinel.yolo.inspect 미구현). "
            "opencv 엔진으로 실행하세요."
        ) from exc
    identity = getattr(module, "inspection_identity", None)
    if not callable(identity):  # pragma: no cover - yolo 레인 의존
        raise WorkerError(f"{engine} 엔진 신원 진입점(inspection_identity)을 찾지 못했습니다.")
    try:
        return identity(engine, config, code_sha=code_sha, code_dirty=code_dirty)
    except FileNotFoundError as exc:
        raise WorkerError(f"{engine} 엔진 모델 파일이 없습니다: {exc}") from exc
    except ValueError as exc:
        raise WorkerError(f"{engine} 엔진 설정 오류: {exc}") from exc


def _inspect(
    image_path: Path,
    config: AppConfig,
    *,
    engine: str,
    output_dir: Path,
    yolo_entry: Callable[..., InspectionOutcome] | None,
    registry: Any = None,
) -> InspectionOutcome:
    """엔진에 따라 한 장을 검사한다(opencv는 실제 구현, YOLO는 지연 분기)."""

    if engine == "opencv":
        return inspect_image(
            image_path,
            config,
            label="unknown",
            split=WORKER_SPLIT,
            output_dir=output_dir,
        )
    assert yolo_entry is not None  # run_worker가 미리 확인한다
    return yolo_entry(
        image_path,
        config,
        engine=engine,
        output_dir=output_dir,
        split=WORKER_SPLIT,
        registry=registry,
    )


#: 워커 검사에서 생성될 수 있는 샘플 산출물 접미사(접두사 글롭 오삭제 방지).
_ARTIFACT_SUFFIXES: tuple[str, ...] = (
    "_overlay.png",
    "_result.json",
    "_response.png",
    "_candidates.png",
    "_yolo.png",
)


def _discard_artifacts(directory: Path, sample_id: str) -> None:
    """표본에서 빠진 OK 샘플의 오버레이·결과 JSON 등 지정 산출물만 지운다."""

    if not directory.is_dir():
        return
    safe_id = sanitize_sample_id(sample_id)
    for suffix in _ARTIFACT_SUFFIXES:
        (directory / f"{safe_id}{suffix}").unlink(missing_ok=True)


def _validate_options(
    *,
    incoming: Path,
    engine: str,
    max_retries: int,
    ok_sample_rate: int,
    poll_seconds: float,
    once: bool,
) -> None:
    if engine not in ENGINES:
        raise WorkerError(f"알 수 없는 엔진입니다: {engine!r} (가능: {ENGINES})")
    if max_retries < 0:
        raise WorkerError(f"max_retries는 0 이상이어야 합니다: {max_retries}")
    if ok_sample_rate < 0:
        raise WorkerError(f"ok_sample_rate는 0 이상이어야 합니다: {ok_sample_rate}")
    if not once and poll_seconds <= 0:
        raise WorkerError(f"poll_seconds는 0보다 커야 합니다: {poll_seconds}")
    if incoming.exists() and not incoming.is_dir():
        raise WorkerError(f"incoming이 디렉터리가 아닙니다: {incoming}")


def _process_one(
    connection: sqlite3.Connection,
    path: Path,
    *,
    run_id: str,
    config: AppConfig,
    engine: str,
    db_path: Path,
    audit_path: Path,
    max_retries: int,
    ok_sample_rate: int,
    yolo_entry: Callable[..., InspectionOutcome] | None,
    model_version: str,
    spec: EngineSpec,
    registry: Any = None,
) -> str:
    """파일 한 장을 처리한다. 반환값은 ``processed``/``skip``/``error``."""

    sample_id = path.stem or path.name
    try:
        sha256 = file_sha256(path)
    except OSError as exc:
        _append_audit(
            audit_path,
            {
                "event": "error",
                "run_id": run_id,
                "sample_id": sample_id,
                "path": str(path),
                "attempt": 1,
                "message": f"파일을 읽지 못했습니다: {exc}",
            },
        )
        return "error"

    if find_inspection_id(connection, sha256, spec.fingerprint) is not None:
        _append_audit(
            audit_path,
            {
                "event": "skip",
                "run_id": run_id,
                "sample_id": sample_id,
                "sha256": sha256,
                "path": str(path),
                "reason": "duplicate_sha256",
            },
        )
        return "skip"

    attempts = max(1, max_retries)
    target = overlay_dir(db_path, run_id)
    result: SampleResult | None = None
    observations: tuple[Any, ...] = ()
    for attempt in range(1, attempts + 1):
        outcome = _inspect(
            path,
            config,
            engine=engine,
            output_dir=target,
            yolo_entry=yolo_entry,
            registry=registry,
        )
        result = outcome.result
        observations = outcome.engine_observations
        if result.status is not Status.ERROR:
            break
        code = result.error_code.value if result.error_code is not None else "E_PROCESS_INTERNAL"
        _append_audit(
            audit_path,
            {
                "event": "error",
                "run_id": run_id,
                "sample_id": sample_id,
                "sha256": sha256,
                "path": str(path),
                "attempt": attempt,
                "message": code,
            },
        )

    assert result is not None
    overlay_path = result.overlay_path
    if not keep_overlay(result.status, sha256, ok_sample_rate):
        _discard_artifacts(target, sample_id)
        overlay_path = ""
    record = replace(result, run_id=run_id, split=WORKER_SPLIT, overlay_path=overlay_path)

    inspection_id = insert_inspection(
        connection,
        record,
        engine=engine,
        image_sha256=sha256,
        model_version=model_version,
        fingerprint=spec.fingerprint,
        task=spec.task,
        model_sha256=spec.model_sha256,
        provenance_complete=1 if spec.provenance_complete else 0,
    )
    for observation in observations:
        insert_engine_observation(
            connection,
            inspection_id,
            observation.result,
            engine=observation.engine,
            task=observation.task,
            model_sha256="" if observation.engine == "opencv" else spec.model_sha256,
        )
    if len(observations) >= 2:
        # 교차 증거(Phase 1, observe-only): 엔진 관찰 둘을 비교해 별도 증거를
        # append 한다. 기계 판정은 건드리지 않는다. 증거 실패는 검사 기록을
        # 막지 않도록 격리하고 감사 로그로만 남긴다.
        try:
            assessment = assess_cross_evidence(observations[0], observations[1])
            append_agent_assessment(
                connection,
                inspection_id,
                assessment_hash=assessment.assessment_hash,
                policy_hash=assessment.policy_hash,
                flags=[flag.value for flag in assessment.flags],
                recommended_action=assessment.recommended_action,
                diagnostics=assessment.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 - 증거 실패는 검사 기록과 격리
            _append_audit(
                audit_path,
                {
                    "event": "evidence_error",
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "sha256": sha256,
                    "path": str(path),
                    "inspection_id": inspection_id,
                    "message": str(exc),
                },
            )
    _append_audit(
        audit_path,
        {
            "event": "processed",
            "run_id": run_id,
            "sample_id": sample_id,
            "sha256": sha256,
            "path": str(path),
            "status": record.status.value,
            "inspection_id": inspection_id,
            "overlay": overlay_path,
        },
    )
    return "processed"


def run_worker(
    *,
    incoming: Path,
    db_path: Path,
    config: AppConfig,
    engine: str = "opencv",
    max_retries: int = 3,
    ok_sample_rate: int = 1,
    poll_seconds: float = 2.0,
    once: bool = False,
) -> None:
    """``incoming``을 감시해 검사하고 DB에 기록한다.

    ``once=True``면 현재 있는 파일을 한 바퀴 처리하고 종료한다(테스트·배치용).
    ``once=False``면 ``poll_seconds`` 간격으로 무한 감시한다. 처리 중인 파일은
    ``(이름, 크기, mtime)`` 지문으로 기억해 같은 프로세스에서 다시 검사하지 않는다.

    네트워크/셸을 쓰지 않는다. 코드 신원(git)은 시작 시 1회만 계산한다
    (핫패스 subprocess 금지). 판정은 파이프라인이 소유한다.
    """

    incoming_path = Path(incoming)
    db_file = Path(db_path)
    _validate_options(
        incoming=incoming_path,
        engine=engine,
        max_retries=max_retries,
        ok_sample_rate=ok_sample_rate,
        poll_seconds=poll_seconds,
        once=once,
    )
    incoming_path.mkdir(parents=True, exist_ok=True)

    # 코드 신원은 진입점 1회 계산(이미지별 재계산·핫패스 subprocess 금지).
    from ..evaluation import recalibration  # noqa: PLC0415 - 진입점 1회 계산

    code_sha, code_dirty = recalibration.git_code_identity()

    yolo_entry = _resolve_yolo_entry(engine) if engine != "opencv" else None
    model_version = _model_version(engine)
    spec = _resolve_spec(engine, config, code_sha=code_sha, code_dirty=code_dirty)
    registry: Any = None
    if engine != "opencv":
        from ..yolo.runtime import YoloEngineRegistry  # noqa: PLC0415 - yolo 레인 지연 import

        registry = YoloEngineRegistry()
    run_id = new_run_id(config)
    audit_path = audit_log_path(db_file)
    seen: set[tuple[str, int, int]] = set()
    oserror_counts: dict[tuple[str, int, int], int] = {}
    oserror_limit = max(1, max_retries)

    connection = connect(db_file)
    try:
        init_db(connection)
        while True:
            handled = 0
            for path in _iter_images(incoming_path):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                fingerprint = (path.name, int(stat.st_size), int(stat.st_mtime_ns))
                if not once and fingerprint in seen:
                    continue
                outcome = _process_one(
                    connection,
                    path,
                    run_id=run_id,
                    config=config,
                    engine=engine,
                    db_path=db_file,
                    audit_path=audit_path,
                    max_retries=max_retries,
                    ok_sample_rate=ok_sample_rate,
                    yolo_entry=yolo_entry,
                    model_version=model_version,
                    spec=spec,
                    registry=registry,
                )
                if outcome != "error":
                    seen.add(fingerprint)
                    oserror_counts.pop(fingerprint, None)
                else:
                    failures = oserror_counts.get(fingerprint, 0) + 1
                    if failures >= oserror_limit:
                        seen.add(fingerprint)
                        oserror_counts.pop(fingerprint, None)
                    else:
                        oserror_counts[fingerprint] = failures
                handled += 1
            if once:
                return
            if handled == 0:
                time.sleep(poll_seconds)
    finally:
        connection.close()


__all__ = [
    "AUDIT_DIRNAME",
    "AUDIT_FILENAME",
    "IMAGE_SUFFIXES",
    "OVERLAY_DIRNAME",
    "WORKER_SPLIT",
    "WorkerError",
    "audit_log_path",
    "file_sha256",
    "keep_overlay",
    "new_run_id",
    "overlay_dir",
    "run_worker",
]
