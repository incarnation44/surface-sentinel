"""검사 API (E4, ADR 0007).

127.0.0.1 전용 로컬 API다. 외부 요청·클라우드 호출은 없다.

* opencv 엔진은 :func:`surface_sentinel.pipeline.inspect_image`를 그대로 쓴다.
* YOLO 엔진은 ``surface_sentinel.yolo.inspect.inspect_with_yolo``를 **지연
  import**한다. 모듈·진입점·모델 파일이 없으면 503과 한국어 안내를 돌려주며,
  조용히 opencv로 폴백하지 않는다.
* 검사 결과는 :mod:`surface_sentinel.records.store`로 DB에 남긴다.
  ``(image_sha256, inspection_fingerprint)`` 쌍이 같으면 기존 행을 재사용한다
  (중복제거).
* ``inspections.status``는 기계 판정이며 불변이다. 사람 판정은
  ``POST /api/inspections/{id}/review``가 append-only로만 쌓는다.

```python
app = create_inspection_app(runs_dir="runs")
```
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from ..agent.evidence import assess_cross_evidence
from ..agent.review import assess_review_triage
from ..agent.stability import analyze_weak_geometry, arbitrate_patches
from ..config import REVIEW_BAND_MAX, AppConfig, default_config, load_config
from ..contracts import Status
from ..pipeline import InspectionOutcome, inspect_image
from ..pipeline.stages import VALID_LABELS, VALID_SPLITS
from ..records.db import DatabaseError, connect, init_db
from ..records.identity import EngineSpec, opencv_spec
from ..records.store import (
    ENGINES,
    STATUS_ORDER,
    RecordsError,
    append_agent_assessment,
    append_review,
    get_inspection,
    insert_engine_observation,
    insert_inspection_with_dedup,
    list_agent_assessments,
    list_engine_observations,
    list_inspections,
    stats_payload,
)
from ..version import __version__
from .schemas import (
    AgentAssessmentOut,
    ComponentOut,
    EngineName,
    EngineObservationOut,
    InspectionDetail,
    InspectionItem,
    InspectionList,
    InspectResponse,
    RatioOut,
    ReviewItem,
    ReviewRequest,
    ReviewResponse,
    ReviewStats,
    StatsResponse,
)

#: 기본 DB 파일 이름(``runs_dir`` 아래). UI도 같은 이름을 쓴다.
DEFAULT_DB_NAME = "records.db"

_logger = logging.getLogger(__name__)

#: 업로드 원본 보관 위치(``runs_dir`` 아래).
UPLOAD_DIRNAME = "api_uploads"

#: 오버레이·결과 JSON 보관 위치(``runs_dir`` 아래).
OUTPUT_DIRNAME = "api_inspect"

#: 업로드 상한(32 MiB). 로컬 API라도 메모리 폭주는 막는다.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

#: 업로드 청크 크기. 전체를 한 번에 읽지 않고 이 크기로 나눠 읽는다(메모리 상한).
UPLOAD_CHUNK_BYTES = 64 * 1024

#: 허용 이미지 확장자(워커와 같은 집합).
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def engine_status_counts(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """엔진별 상태 건수를 돌려준다(모든 상태 키를 0으로 채운다).

    UI 통계 페이지도 같은 함수를 쓴다(정의 한 곳).
    """

    counts: dict[str, dict[str, int]] = {}
    rows = connection.execute(
        "SELECT engine, status, COUNT(*) AS n FROM inspections GROUP BY engine, status"
    ).fetchall()
    for row in rows:
        engine = str(row["engine"])
        bucket = counts.setdefault(engine, dict.fromkeys(STATUS_ORDER, 0))
        bucket[str(row["status"])] = int(row["n"])
    return dict(sorted(counts.items()))


def _row_overlay_path(row: dict[str, Any], runs_dir: Path) -> Path | None:
    """DB 행의 오버레이 경로가 ``runs_dir`` 안의 실제 파일일 때만 돌려준다."""

    raw = str(row.get("overlay_path") or "")
    if not raw:
        return None
    candidate = Path(raw)
    try:
        if not candidate.is_file():
            return None
        if not candidate.resolve().is_relative_to(runs_dir.resolve()):
            return None
    except OSError:
        return None
    return candidate


def _inspection_item(row: dict[str, Any], runs_dir: Path) -> InspectionItem:
    """DB 행을 목록 아이템으로 바꾼다. 오버레이 URL은 행 경로에서 계산한다."""

    inspection_id = int(row["id"])
    has_overlay = _row_overlay_path(row, runs_dir) is not None
    return InspectionItem(
        inspection_id=inspection_id,
        run_id=str(row["run_id"]),
        sample_id=str(row["sample_id"]),
        label=str(row["label"]),
        split=str(row["split"]),
        status=Status(str(row["status"])),
        score=float(row["score"]),
        engine=EngineName(str(row["engine"])),
        model_version=str(row["model_version"]),
        reasons=[str(reason) for reason in json.loads(str(row["reasons"]) or "[]")],
        overlay_url=(
            f"/api/inspections/{inspection_id}/overlay.png" if has_overlay else None
        ),
        created_at=str(row["created_at"]),
    )


def _reviews(connection: sqlite3.Connection, inspection_id: int) -> list[ReviewItem]:
    """검사 한 건의 사람 판정 이력을 append 순서로 읽는다."""

    rows = connection.execute(
        "SELECT id, decision, reviewer, reason, created_at FROM review_decisions "
        "WHERE inspection_id = ? ORDER BY id ASC",
        (int(inspection_id),),
    ).fetchall()
    return [
        ReviewItem(
            review_id=int(row["id"]),
            decision=str(row["decision"]),
            reviewer=str(row["reviewer"]),
            reason=str(row["reason"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]



def _observations(
    connection: sqlite3.Connection, inspection_id: int
) -> list[EngineObservationOut]:
    """검사 한 건의 엔진별 원본 관찰 목록을 id 오름차순으로 읽는다."""

    rows = list_engine_observations(connection, inspection_id)
    items: list[EngineObservationOut] = []
    for row in rows:
        components_raw = json.loads(str(row.get("components") or "[]"))
        timing_raw = json.loads(str(row.get("timing_ms") or "{}"))
        items.append(
            EngineObservationOut(
                observation_id=int(row["id"]),
                engine=str(row["engine"]),
                task=str(row.get("task") or ""),
                status=Status(str(row["status"])),
                score=float(row["score"]),
                reasons=[str(r) for r in json.loads(str(row.get("reasons") or "[]"))],
                components=[ComponentOut(**c) for c in components_raw],
                model_sha256=str(row.get("model_sha256") or ""),
                overlay_path=str(row.get("overlay_path") or ""),
                error_code=str(row.get("error_code") or ""),
                timing_ms={str(k): float(v) for k, v in timing_raw.items()},
                created_at=str(row["created_at"]),
            )
        )
    return items


def _assessments(
    connection: sqlite3.Connection, inspection_id: int
) -> list[AgentAssessmentOut]:
    """검사 한 건의 교차 증거·트리아지 평가 목록을 id 오름차순으로 읽는다."""

    rows = list_agent_assessments(connection, inspection_id)
    items: list[AgentAssessmentOut] = []
    for row in rows:
        diag_raw = json.loads(str(row.get("diagnostics_json") or "{}"))
        items.append(
            AgentAssessmentOut(
                assessment_id=int(row["id"]),
                assessment_hash=str(row["assessment_hash"]),
                policy_hash=str(row["policy_hash"]),
                flags=[str(f) for f in json.loads(str(row.get("flags_json") or "[]"))],
                review_priority=(
                    str(row["review_priority"])
                    if row.get("review_priority") is not None
                    else None
                ),
                stability=(
                    str(row["stability"]) if row.get("stability") is not None else None
                ),
                recommended_action=str(row["recommended_action"]),
                diagnostics=diag_raw if isinstance(diag_raw, dict) else {},
                trace_path=str(row.get("trace_path") or ""),
                created_at=str(row["created_at"]),
            )
        )
    return items


def _load_yolo_entry(engine: str) -> Callable[..., InspectionOutcome]:
    """YOLO 엔진 진입점을 지연 import한다. 없으면 503(한국어 안내)."""

    try:
        module = importlib.import_module("surface_sentinel.yolo.inspect")
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{engine} 엔진을 사용할 수 없습니다: surface_sentinel.yolo.inspect 를 "
                "불러오지 못했습니다. ONNX Runtime extra 설치와 모델 파일 경로를 확인하세요."
            ),
        ) from exc
    entry = getattr(module, "inspect_with_yolo", None)
    if not callable(entry):
        raise HTTPException(
            status_code=503,
            detail=f"{engine} 엔진 진입점(inspect_with_yolo)을 찾지 못했습니다.",
        )
    return entry


def _model_version(engine: str) -> str:
    """기록에 남길 모델/엔진 버전 문자열."""

    if engine == EngineName.OPENCV.value:
        import cv2

        return f"opencv-{cv2.__version__}"
    return engine


def _inspection_spec(
    engine: str,
    config: AppConfig,
    *,
    code_sha: str = "",
    code_dirty: bool = False,
    allow_minimal_fallback: bool = False,
) -> EngineSpec:
    """검사 신원 EngineSpec 을 계산한다(P0-2/P0-3, 평가·워커와 같은 정의).

    opencv 는 모델 파일이 없는 기준선 신원. YOLO 계열은 ``yolo.inspect`` 의
    ``inspection_identity`` 로 실제 모델 SHA·정책 해시를 계산한다. ``code_sha``/
    ``code_dirty`` 는 앱 생성 시 1회 계산한 값을 주입한다(요청별 재계산 금지).
    봉인 조기 거부(``E_SEALED_DENIED``)처럼 모델 로드 전에 ``ERROR`` 로 끝난
    경우(``allow_minimal_fallback=True``)에만 최소 spec 으로 폴백하고, 정상
    판정 경로에서는 모델/설정 오류를 숨기지 않고 그대로 전파한다.
    """

    if engine == EngineName.OPENCV.value:
        return opencv_spec(config.config_hash, code_sha=code_sha, code_dirty=code_dirty)
    module = importlib.import_module("surface_sentinel.yolo.inspect")
    identity = getattr(module, "inspection_identity", None)
    if callable(identity):
        try:
            return identity(engine, config, code_sha=code_sha, code_dirty=code_dirty)
        except (FileNotFoundError, ValueError):
            if not allow_minimal_fallback:
                raise
    return EngineSpec(
        engine=engine,
        config_hash=config.config_hash,
        code_sha=code_sha,
        code_dirty=code_dirty,
    )


def _inspect(
    image_path: Path,
    config: AppConfig,
    *,
    engine: EngineName,
    output_dir: Path,
    label: str,
    split: str,
    registry: Any = None,
) -> InspectionOutcome:
    """엔진별 검사 실행. YOLO 실행 환경·모델 문제는 503으로 바꾼다."""

    if engine is EngineName.OPENCV:
        return inspect_image(
            image_path,
            config,
            label=label,
            split=split,
            output_dir=output_dir,
        )
    entry = _load_yolo_entry(engine.value)
    try:
        return entry(
            image_path,
            config,
            engine=engine.value,
            output_dir=output_dir,
            split=split,
            label=label,
            registry=registry,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{engine.value} 엔진 실행 환경이 없습니다(ONNX Runtime). "
                "`pip install 'surface-sentinel[yolo]'` 또는 ONNX Runtime 설치를 확인하세요."
            ),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{engine.value} 엔진 모델 파일이 없습니다: {exc}",
        ) from exc
    except ValueError as exc:
        # combined task 미설정 등 설정 오류(P0-3): 조용히 폴백하지 않고 명시 오류.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_label_split(label: str, split: str) -> None:
    """``label``/``split`` 이 허용 집합 안인지 경계에서 검증한다.

    허용 값의 단일 정본은 :data:`surface_sentinel.pipeline.stages.VALID_LABELS`
    / ``VALID_SPLITS`` 다(사본을 만들지 않는다). 허용 집합 밖 값은 파일 읽기·
    검사 실행·DB 기록 **전에** 422로 끝낸다 — 잘못된 값이 ERROR 판정과 검사
    행을 만들지 않게 하기 위함이다.

    ``split`` 의 ``test``/``official_test`` 는 허용 값이지만 봉인 정책상
    파이프라인이 ERROR(``E_SEALED_DENIED``)로 거부한다(여기서 우회하지 않는다).
    """

    if label not in VALID_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"label은 {', '.join(VALID_LABELS)} 중 하나여야 합니다: {label!r}",
        )
    if split not in VALID_SPLITS:
        raise HTTPException(
            status_code=422,
            detail=f"split은 {', '.join(VALID_SPLITS)} 중 하나여야 합니다: {split!r}",
        )


def _upload_too_large_detail() -> str:
    """업로드 상한 초과 시 한국어 안내(상한 값은 호출 시점의 전역을 읽는다)."""

    return f"업로드는 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB까지 허용합니다."


async def _read_upload_bounded(upload: UploadFile) -> bytes:
    """업로드를 ``MAX_UPLOAD_BYTES`` + 청크 1개까지만 읽는다(메모리 상한).

    전체를 ``await file.read()`` 로 한 번에 받으면 상한 검사 전에 메모리가
    폭주한다. 여기서는 ``UPLOAD_CHUNK_BYTES`` 단위로 읽으며 누적량이 상한을
    넘는 순간 413으로 중단한다 — 읽기 총량은 상한 + 청크 1개를 넘지 않는다.
    """

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=_upload_too_large_detail())
        chunks.append(chunk)
    return b"".join(chunks)


def _store_upload(upload_dir: Path, payload: bytes, sha256: str, suffix: str) -> Path:
    """업로드 바이트를 ``runs_dir`` 아래에 원자적으로 보관한다(같은 해시는 같은 경로)."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{sha256[:16]}{suffix}"
    if not path.is_file():
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise
    return path


def create_inspection_app(
    *,
    runs_dir: str | Path,
    db_path: str | Path | None = None,
    raw_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """검사 API 앱을 만든다.

    ``db_path``가 없으면 ``runs_dir/records.db``를 만들고 스키마를 초기화한다.
    ``raw_root``는 UI 앱과 시그니처를 맞추기 위한 값이며 API는 원본을 읽지
    않는다(업로드 이미지만 검사한다).
    """

    app = FastAPI(
        title="Surface Sentinel 검사 API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    db_file = Path(db_path) if db_path is not None else directory / DEFAULT_DB_NAME
    config = load_config(config_path) if config_path is not None else default_config()

    # 코드 신원은 앱 생성 시 1회 계산(요청별 재계산 금지 — 워커와 같은 규칙).
    from ..evaluation import recalibration  # noqa: PLC0415 - 진입점 1회 계산

    code_sha, code_dirty = recalibration.git_code_identity()

    try:
        connection = connect(db_file)
        try:
            init_db(connection)
        finally:
            connection.close()
    except DatabaseError as exc:
        raise RuntimeError(f"검사 기록 DB를 초기화하지 못했습니다: {db_file} ({exc})") from exc

    app.state.runs_dir = directory
    app.state.db_path = db_file
    app.state.raw_root = Path(raw_root) if raw_root is not None else None
    app.state.config = config
    try:
        runtime = importlib.import_module("surface_sentinel.yolo.runtime")
        app.state.yolo_registry = runtime.YoloEngineRegistry()
    except ImportError:
        app.state.yolo_registry = None

    @contextmanager
    def open_db() -> Iterator[sqlite3.Connection]:
        connection = connect(db_file)
        try:
            yield connection
        finally:
            connection.close()

    @app.get("/api/healthz")
    @app.get("/healthz")
    def healthz() -> JSONResponse:
        with open_db() as connection:
            total = int(stats_payload(connection)["total"])
        return JSONResponse(
            {
                "status": "ok",
                "service": "surface-sentinel-api",
                "version": __version__,
                "inspections": total,
                "engines": list(ENGINES),
            }
        )

    @app.post("/api/inspect", response_model=InspectResponse)
    async def inspect(
        file: Annotated[UploadFile, File()],
        engine: Annotated[EngineName, Query()] = EngineName.OPENCV,
        label: Annotated[str, Form()] = "unknown",
        split: Annotated[str, Form()] = "tune",
    ) -> InspectResponse:
        """이미지 한 장을 검사하고 기록한다(기계 판정은 파이프라인이 소유).

        경계 검증: ``label``/``split`` 이 허용 집합(:mod:`pipeline.stages` 정본)
        밖이면 파일을 읽기 전에 422 로 끝낸다(ERROR 행을 만들지 않는다).
        업로드는 상한까지만 청크로 읽고, 넘으면 413 이다.

        중복제거(``deduplicated=True``)일 때 응답의 저장 필드(``run_id``,
        ``sample_id``, ``status``, ``score``, ``reasons``, ``overlay_url``)는
        방금 계산한 결과가 아니라 **DB에 저장된 canonical 행**에서 만든다.
        ``components`` 는 해당 엔진의 저장된 ``engine_observations`` 행이 있으면
        그 ``components`` JSON 을, 없으면(단일 엔진 검사는 관찰 미기록) 이번
        결과의 값을 쓴다 — 같은 fingerprint 면 파이프라인이 결정론적이라 일치한다.
        ``error_code`` 는 저장된 관찰 값을 쓰고, 저장하지 않는 ``error_stage`` 는
        빈 문자열이다. 저장 관찰이 없으면(단일 엔진) 이번 결과의
        ``error_code``/``error_stage`` 를 유지한다(같은 fingerprint = 결정론적
        = 원래 실행과 같은 진단 값). 내부 경로·stack trace 는 응답에 넣지 않는다.
        """

        _validate_label_split(label, split)

        try:
            payload = await _read_upload_bounded(file)
        finally:
            await file.close()
        if not payload:
            raise HTTPException(status_code=400, detail="빈 파일은 검사할 수 없습니다.")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"이미지 확장자만 업로드할 수 있습니다: {suffix or '(없음)'}",
            )

        sha256 = hashlib.sha256(payload).hexdigest()
        image_path = _store_upload(directory / UPLOAD_DIRNAME, payload, sha256, suffix)
        output_dir = directory / OUTPUT_DIRNAME / f"{sha256[:16]}-{engine.value}"

        outcome = await run_in_threadpool(
            _inspect,
            image_path,
            config,
            engine=engine,
            output_dir=output_dir,
            label=label,
            split=split,
            registry=getattr(app.state, "yolo_registry", None),
        )
        result = outcome.result
        model_version = _model_version(engine.value)
        spec = await run_in_threadpool(
            _inspection_spec,
            engine.value,
            config,
            code_sha=code_sha,
            code_dirty=code_dirty,
            allow_minimal_fallback=(result.status is Status.ERROR),
        )

        with open_db() as connection:
            inspection_id, deduplicated = insert_inspection_with_dedup(
                connection,
                result,
                engine=engine.value,
                image_sha256=sha256,
                model_version=model_version,
                fingerprint=spec.fingerprint,
                task=spec.task,
                model_sha256=spec.model_sha256,
                provenance_complete=1 if spec.provenance_complete else 0,
            )
            for observation in outcome.engine_observations:
                insert_engine_observation(
                    connection,
                    inspection_id,
                    observation.result,
                    engine=observation.engine,
                    task=observation.task,
                    model_sha256="" if observation.engine == "opencv" else spec.model_sha256,
                )
            if len(outcome.engine_observations) >= 2:
                # 교차 증거(Phase 1/2, observe-only): 별도 증거·트리아지만 append 한다.
                # 기계 판정은 건드리지 않으며, 증거 실패는 검사 응답을 막지
                # 않도록 격리한다(워커와 같은 계약).
                try:
                    obs_a = outcome.engine_observations[0]
                    obs_b = outcome.engine_observations[1]
                    assessment = assess_cross_evidence(obs_a, obs_b)
                    arbitration = None
                    try:
                        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            arbitration = arbitrate_patches(bgr, obs_a, obs_b)
                    except Exception:  # noqa: BLE001 - 패치 중재 실패 시 교차 증거는 유지
                        arbitration = None
                    geom = analyze_weak_geometry(result.components, result.roi)
                    triage = assess_review_triage(
                        result,
                        geometry=geom,
                        cross_assessment=assessment,
                        arbitration=arbitration,
                        ng_min=config.scoring.ng_min,
                        review_band_max=REVIEW_BAND_MAX,
                    )
                    diagnostics = {
                        **assessment.to_dict(),
                        "triage": triage.to_dict(),
                        "arbitration": (
                            arbitration.to_dict() if arbitration is not None else None
                        ),
                    }
                    append_agent_assessment(
                        connection,
                        inspection_id,
                        assessment_hash=assessment.assessment_hash,
                        policy_hash=assessment.policy_hash,
                        flags=[flag.value for flag in assessment.flags],
                        recommended_action=assessment.recommended_action,
                        diagnostics=diagnostics,
                        review_priority=triage.priority.value,
                    )
                except Exception:  # noqa: BLE001 - 증거 실패는 검사 기록과 격리
                    _logger.warning(
                        "교차 증거 기록 실패(검사 기록은 유지): inspection_id=%s",
                        inspection_id,
                        exc_info=True,
                    )
            row = get_inspection(connection, inspection_id)
            observation_rows = (
                list_engine_observations(connection, inspection_id) if deduplicated else []
            )

        overlay_url = None
        if row is not None and _row_overlay_path(row, directory) is not None:
            overlay_url = f"/api/inspections/{inspection_id}/overlay.png"

        # 기본값: 이번 요청의 계산 결과(새 행 경로).
        run_id = str(result.run_id)
        sample_id = str(result.sample_id)
        status = result.status
        score = float(result.score)
        reasons: list[Any] = list(result.reasons)
        components_out = [ComponentOut.from_feature(feature) for feature in result.components]
        error_code = "" if result.error_code is None else result.error_code.value
        error_stage = "" if result.error_stage is None else result.error_stage.value

        if deduplicated and row is not None:
            # 중복 적중: 응답은 저장된 canonical 행이 정본이다(새 계산값으로 덮지 않는다).
            run_id = str(row["run_id"])
            sample_id = str(row["sample_id"])
            status = Status(str(row["status"]))
            score = float(row["score"])
            reasons = [str(reason) for reason in json.loads(str(row["reasons"]) or "[]")]
            stored = next(
                (obs for obs in observation_rows if str(obs["engine"]) == engine.value),
                None,
            )
            if stored is not None:
                stored_components = json.loads(str(stored.get("components") or "[]"))
                components_out = [ComponentOut(**component) for component in stored_components]
                error_code = str(stored.get("error_code") or "")
                # error_stage 는 관찰 표에 저장하지 않는다 → 미저장이므로 빈 문자열.
                error_stage = ""
            elif status is not Status.ERROR:
                # 저장 관찰이 없는 단일 엔진 dedup 에서 canonical 행이 정상/결함/검토
                # 상태라면 이번 요청의 오류 진단 필드가 응답에 혼입되지 않게 비운다.
                error_code = ""
                error_stage = ""
            # 저장 관찰이 없고 canonical 행도 ERROR 면 이번 결과의
            # error_code/error_stage 를 유지한다 — 같은 fingerprint 면 파이프라인이
            # 결정론적이라 원래 실행과 같은 진단 값이다(ERROR 진단 가능성 보존).

        return InspectResponse(
            inspection_id=inspection_id,
            run_id=run_id,
            sample_id=sample_id,
            status=status,
            score=score,
            reasons=reasons,
            components=components_out,
            engine=engine,
            model_version=model_version,
            overlay_url=overlay_url,
            deduplicated=deduplicated,
            error_code=error_code,
            error_stage=error_stage,
            task=spec.task,
            model_sha256=spec.model_sha256,
            inspection_fingerprint=spec.fingerprint,
            provenance_complete=spec.provenance_complete,
            code_sha=spec.code_sha,
            code_dirty=spec.code_dirty,
        )

    @app.get("/api/inspections", response_model=InspectionList)
    def inspections(
        limit: Annotated[int, Query(ge=0, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        status: Annotated[Status | None, Query()] = None,
    ) -> InspectionList:
        """검사 기록 목록(``id`` 오름차순, 페이지네이션·상태 필터)."""

        with open_db() as connection:
            payload = stats_payload(connection)
            total = (
                int(payload["total"])
                if status is None
                else int(payload["by_status"].get(status.value, 0))
            )
            rows = list_inspections(
                connection,
                status=None if status is None else status.value,
                limit=limit,
                offset=offset,
            )
            items = [_inspection_item(row, directory) for row in rows]
        return InspectionList(total=total, limit=limit, offset=offset, items=items)

    @app.get("/api/inspections/{inspection_id}", response_model=InspectionDetail)
    def inspection_detail(inspection_id: int) -> InspectionDetail:
        """검사 한 건 + 사람 판정 이력."""

        with open_db() as connection:
            row = get_inspection(connection, inspection_id)
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"검사 기록이 없습니다: id={inspection_id}",
                )
            item = _inspection_item(row, directory)
            return InspectionDetail(
                **item.model_dump(),
                image_sha256=str(row["image_sha256"]),
                config_hash=str(row["config_hash"]),
                overlay_path=str(row["overlay_path"]),
                timing_ms={
                    str(key): float(value)
                    for key, value in json.loads(str(row["timing_ms"]) or "{}").items()
                },
                reviews=_reviews(connection, inspection_id),
                task=str(row["task"]),
                model_sha256=str(row["model_sha256"]),
                inspection_fingerprint=str(row["inspection_fingerprint"]),
                provenance_complete=bool(row["provenance_complete"]),
                observations=_observations(connection, inspection_id),
                assessments=_assessments(connection, inspection_id),
            )

    @app.post(
        "/api/inspections/{inspection_id}/review",
        response_model=ReviewResponse,
    )
    def review(inspection_id: int, body: ReviewRequest) -> ReviewResponse:
        """사람 판정을 append한다. 기계 판정(``status``)은 바꾸지 않는다."""

        with open_db() as connection:
            row = get_inspection(connection, inspection_id)
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"검사 기록이 없습니다: id={inspection_id}",
                )
            machine_status = str(row["status"])
            if machine_status != Status.REVIEW.value:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "기계 판정 OK/NG는 불변입니다. REVIEW 기록만 사람 판정을 "
                        "남길 수 있습니다."
                    ),
                )
            try:
                review_id = append_review(
                    connection,
                    inspection_id,
                    body.decision.value,
                    body.reviewer,
                    body.reason,
                )
            except RecordsError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            appended = _reviews(connection, inspection_id)
        return ReviewResponse(
            inspection_id=inspection_id,
            review_id=review_id,
            decision=body.decision,
            machine_status=Status(machine_status),
            reviews=len(appended),
        )

    @app.get("/api/inspections/{inspection_id}/overlay.png")
    def overlay_image(inspection_id: int) -> FileResponse:
        """기록에 연결된 오버레이 PNG(``runs_dir`` 안 파일만)."""

        with open_db() as connection:
            row = get_inspection(connection, inspection_id)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"검사 기록이 없습니다: id={inspection_id}",
            )
        path = _row_overlay_path(row, directory)
        if path is None:
            raise HTTPException(status_code=404, detail="오버레이 파일이 없습니다.")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        """상태별 집계(분모 포함)와 엔진별 비교."""

        with open_db() as connection:
            payload = stats_payload(connection)
            by_engine = engine_status_counts(connection)
        return StatsResponse(
            total=int(payload["total"]),
            by_status={str(name): int(value) for name, value in payload["by_status"].items()},
            reviews=ReviewStats(**payload["reviews"]),
            overlays=int(payload["overlays"]),
            ratios={name: RatioOut(**value) for name, value in payload["ratios"].items()},
            by_engine=by_engine,
        )

    return app


__all__ = [
    "DEFAULT_DB_NAME",
    "IMAGE_SUFFIXES",
    "MAX_UPLOAD_BYTES",
    "OUTPUT_DIRNAME",
    "UPLOAD_DIRNAME",
    "create_inspection_app",
    "engine_status_counts",
]
