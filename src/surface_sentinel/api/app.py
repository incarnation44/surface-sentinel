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

import hashlib
import importlib
import json
import logging
import sqlite3
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
from ..records.db import DatabaseError, connect, init_db
from ..records.identity import EngineSpec, opencv_spec
from ..records.store import (
    ENGINES,
    STATUS_ORDER,
    RecordsError,
    append_agent_assessment,
    append_review,
    find_inspection_id,
    get_inspection,
    insert_engine_observation,
    insert_inspection,
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
) -> EngineSpec:
    """검사 신원 EngineSpec 을 계산한다(P0-2/P0-3, 평가·워커와 같은 정의).

    opencv 는 모델 파일이 없는 기준선 신원. YOLO 계열은 ``yolo.inspect`` 의
    ``inspection_identity`` 로 실제 모델 SHA·정책 해시를 계산한다. ``code_sha``/
    ``code_dirty`` 는 앱 생성 시 1회 계산한 값을 주입한다(요청별 재계산 금지).
    진입점이 없는 스텁 환경(테스트)에서는 신원 불완전 최소 spec 으로 폴백한다
    (조용히 opencv 로 판정을 바꾸지는 않는다 — 판정은 이미 파이프라인이 확정했다).
    """

    if engine == EngineName.OPENCV.value:
        return opencv_spec(config.config_hash, code_sha=code_sha, code_dirty=code_dirty)
    module = importlib.import_module("surface_sentinel.yolo.inspect")
    identity = getattr(module, "inspection_identity", None)
    if callable(identity):
        return identity(engine, config, code_sha=code_sha, code_dirty=code_dirty)
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


def _store_upload(upload_dir: Path, payload: bytes, sha256: str, suffix: str) -> Path:
    """업로드 바이트를 ``runs_dir`` 아래에 보관한다(같은 해시는 같은 경로)."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{sha256[:16]}{suffix}"
    if not path.is_file():
        path.write_bytes(payload)
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
        """이미지 한 장을 검사하고 기록한다(기계 판정은 파이프라인이 소유)."""

        payload = await file.read()
        await file.close()
        if not payload:
            raise HTTPException(status_code=400, detail="빈 파일은 검사할 수 없습니다.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"업로드는 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB까지 허용합니다.",
            )
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
            _inspection_spec, engine.value, config, code_sha=code_sha, code_dirty=code_dirty
        )

        with open_db() as connection:
            deduplicated = find_inspection_id(connection, sha256, spec.fingerprint) is not None
            inspection_id = insert_inspection(
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

        overlay_url = None
        if row is not None and _row_overlay_path(row, directory) is not None:
            overlay_url = f"/api/inspections/{inspection_id}/overlay.png"

        return InspectResponse(
            inspection_id=inspection_id,
            run_id=str(result.run_id),
            sample_id=str(result.sample_id),
            status=result.status,
            score=float(result.score),
            reasons=list(result.reasons),
            components=[ComponentOut.from_feature(feature) for feature in result.components],
            engine=engine,
            model_version=model_version,
            overlay_url=overlay_url,
            deduplicated=deduplicated,
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
