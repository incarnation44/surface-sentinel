"""로컬 웹 검토·관제 콘솔 FastAPI 라우터 — run 탐색, 리뷰 큐, 지표, 트리아지, Watchdog, 3D 시네마.

127.0.0.1 전용. 외부 CDN·외부 요청 없음.
기본 모드(``enable_api=False``)에서는 읽기 전용 + REVIEW 검토 결정 append
하나뿐이며 ``OK``/``NG``는 불변이다. ``enable_api=True``(``--with-api``)일 때만
``/api/*`` 로컬 검사 API와 실시간 검사 워크벤치를 함께 마운트한다.
한국어 문구와 영어 상태 배지를 함께 쓰고, 색상만으로 의미를 전달하지 않는다.
"""

from __future__ import annotations

import csv
import io
import random
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..agent.audit import (
    audit_records_db,
    audit_repository_gates,
    audit_run_dir,
)
from ..agent.calibration import analyze_human_feedback
from ..agent.counterfactual import explain_counterfactual
from ..agent.drift import DEFAULT_DRIFT_POLICY, assess_drift
from ..agent.review import (
    assess_review_triage,
    evaluate_triage_lift,
    rank_review_queue,
)
from ..agent.stability import analyze_weak_geometry
from ..api.app import create_inspection_app, engine_status_counts
from ..config import REVIEW_BAND_MAX, ConfigError, default_config, load_config
from ..evaluation.metrics import Confusion, image_metrics, ratio
from ..io.artifacts import CONFIG_SNAPSHOT
from ..io.image_guard import ImageHeaderError, ImagePixelLimitError, check_image_pixel_budget
from ..io.limits import (
    CsrfOriginError,
    UploadTooLargeError,
    async_pipeline_slot,
    configure_opencv_threads,
    pipeline_slot,
    read_upload_bounded,
    validate_same_origin,
)
from ..pipeline.evidence import sanitize_sample_id
from ..records.db import DatabaseError
from ..records.store import STATUS_ORDER, get_inspection, stats
from ..review import ReviewError, ReviewStorageError, ReviewStore
from ..version import __version__
from . import cinema as _cinema_mod
from .cinema import (
    ALLOWED_CINEMA_EXTENSIONS,
    CINEMA_CHAPTERS,
    CINEMA_UPLOAD_CHUNK_BYTES,
    MAX_CINEMA_UPLOAD_BYTES,
    discover_cinema_presets,
    load_cinema_config,
    run_cinema_preset,
    run_cinema_trace,
)
from .data import (
    RunEntry,
    UiInputError,
    discover_runs,
    find_sample,
    load_inspection_detail_view,
    load_inspections_view,
    load_manifest_map,
    load_results,
    load_run,
    read_only_connection,
    row_overlay_path,
    safe_child,
    sample_result_from_record,
    sha256_file,
)
from .verdict_explain import explain_verdict

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: UI가 읽는 검사 DB 파일 이름(``runs_dir`` 아래). API의 ``DEFAULT_DB_NAME``과 같다.
DEFAULT_DB_NAME = "records.db"

#: 샘플 상세·마스크 뷰어가 찾는 단계 이미지 종류(``stages/<id>_<kind>.png``).
STAGE_KINDS: tuple[str, ...] = ("response", "candidates", "roi", "refined")

#: 마스크 뷰어 레이어 순서: (키, 한국어 라벨).
MASK_LAYERS: tuple[tuple[str, str], ...] = (
    ("original", "원본"),
    ("roi", "ROI"),
    ("candidate", "candidate (후보 마스크)"),
    ("refined", "refined (정제 마스크)"),
    ("overlay", "오버레이"),
)

#: 상태 배지 → 한국어 뜻. 배지 텍스트(영어)와 나란히 표시한다.
STATUS_LABELS = {
    "OK": "정상",
    "NG": "불량",
    "REVIEW": "보류",
    "ERROR": "오류(판정 아님)",
    "COMPLETE": "완료",
    "DONE": "완료",
    "PARTIAL": "부분 완료",
}

#: 평가 지표 코드명 → 한국어 설명.
METRIC_LABELS = {
    "defect_escape_rate": "결함 누출률 (Escape)",
    "good_false_ng_rate": "정상 오판률 (False-NG)",
    "review_rate": "검토 보류율 (Review)",
    "error_rate": "오류율 (Error)",
    "decided_rate": "자동 판정률 (Decided)",
    "defect_recall_on_decided": "판정 내 결함 재현율 (Recall)",
    "ng_precision": "NG 정밀도 (Precision)",
}

#: 검토 사유 코드 → 한국어 설명. 어휘 정의는 ``review.store``가 정본이다.
REASON_LABELS = {
    "CONFIRMED_DEFECT": "결함 확인",
    "FALSE_ALARM": "오탐 확인(정상)",
    "INSUFFICIENT_EVIDENCE": "증거 부족",
    "IMAGE_QUALITY": "이미지 품질 문제",
    "OTHER": "기타(메모 필수)",
}

#: 기계 판정 사유 코드 → 쉬운 한국어 설명(상세 화면 판정 근거 카드용).
VERDICT_REASON_LABELS = {
    "DEFECT_EVIDENCE_STRONG": (
        "강한 결함 증거 확인 — 유효 후보(면적 ≥ 40px, 대비 ≥ 16) 최고 점수가 "
        "NG 기준(0.50) 이상"
    ),
    "NO_DEFECT_EVIDENCE": "결함 증거 없음 — 유효 결함 후보 0개 및 품질·경계 이상 없음",
    "EVIDENCE_IN_REVIEW_BAND": (
        "보류 구간 결함 후보 검출 — 유효 후보가 있으나 최고 점수가 NG 기준(0.50) 미만"
    ),
    "MULTIPLE_WEAK_CANDIDATES": (
        "다수 약한 후보 동시 검출 — 보류 대역(0.25 ≤ 점수 < 0.50) 후보가 2개 이상 잡혀 "
        "미세 결함·표면 노이즈 확인 필요"
    ),
    "BORDER_TOUCHING_CANDIDATE": (
        "ROI 가장자리 접촉 후보 — 경계 인접 후보(점수 ≥ 0.40)로 "
        "치구·절단면 그림자 오검 확인 필요"
    ),
    "CANDIDATE_TRUNCATED": (
        "후보 수 상한(20개) 초과 절단 — 전면 텍스처·조명 이상 가능성으로 사람 확인 필요"
    ),
    "INPUT_QUALITY_LOW": "입력 영상 품질 저하 — 흐림·과암부·과노출·포화 비율 임계 초과",
    "ROI_UNCERTAIN": "검사 영역(ROI) 불확실 — 유효 표면 면적 비율(20%) 미달",
    "MANUAL_OVERRIDE": "작업자 수동 판정 반영",
    "MANUAL_DEFERRED": "작업자 수동 보류",
}

#: 낮을수록 좋은 지표(run 비교 Δ 표시에 사용).
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"defect_escape_rate", "good_false_ng_rate", "review_rate", "error_rate"}
)

#: 이미지 파일이 없을 때 안내 문구.
NO_IMAGE_TEXT = "이미지 파일이 없습니다."


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "base.toml").is_file():
            return parent
    return Path.cwd()


def _read_only_connection(path: Path) -> sqlite3.Connection:
    """검사 DB를 읽기 전용으로 연다(파일을 새로 만들지 않는다)."""

    return read_only_connection(path)


def load_stats_view(db_path: Path) -> dict[str, Any]:
    """검사 DB 통계를 UI 표시용으로 만든다.

    DB 파일이 없으면 **만들지 않고** 안내 문구만 돌려준다(UI는 읽기 전용).
    """

    base: dict[str, Any] = {
        "available": False,
        "path": str(db_path),
        "message": "",
        "payload": None,
        "engines": [],
        "status_order": STATUS_ORDER,
    }
    if not db_path.is_file():
        return {
            **base,
            "message": (
                "검사 기록 DB가 없습니다. 검사 API로 기록을 만들면 여기에 표시됩니다."
            ),
        }
    try:
        connection = _read_only_connection(db_path)
        try:
            payload = stats(connection)
            by_engine = engine_status_counts(connection)
        finally:
            connection.close()
    except (sqlite3.Error, DatabaseError) as exc:
        return {**base, "message": f"검사 기록 DB를 읽지 못했습니다: {exc}"}

    engines = []
    for engine in sorted(by_engine):
        bucket = by_engine[engine]
        total = sum(bucket.values())
        engines.append(
            {
                "engine": engine,
                "total": total,
                "counts": {name: bucket.get(name, 0) for name in STATUS_ORDER},
                "ok_ratio": ratio(f"{engine} 검사 중 OK 비율", bucket.get("OK", 0), total),
            }
        )
    return {**base, "available": True, "payload": payload, "engines": engines}


def _run_app_config(entry: RunEntry):
    snapshot = entry.run_dir / CONFIG_SNAPSHOT
    if snapshot.is_file():
        try:
            return load_config(snapshot)
        except ConfigError:
            pass
    return default_config()


def _run_scoring_bounds(entry: RunEntry) -> tuple[float, float]:
    cfg = _run_app_config(entry)
    return float(cfg.scoring.ng_min), REVIEW_BAND_MAX


def _build_gain_svg(
    ranked_sample_ids: list[str],
    labels_by_id: dict[str, str],
) -> dict[str, Any]:
    total = len(ranked_sample_ids)
    total_defects = sum(1 for sid in ranked_sample_ids if labels_by_id.get(sid) == "defect")
    if total == 0 or total_defects == 0:
        return {"available": False, "curve_points": "", "area_points": "", "markers": []}

    x0, x1 = 46.0, 404.0
    y0, y1 = 18.0, 168.0
    w = x1 - x0
    h = y1 - y0
    pts: list[tuple[float, float]] = [(x0, y1)]
    captured = 0
    for idx, sid in enumerate(ranked_sample_ids, start=1):
        if labels_by_id.get(sid) == "defect":
            captured += 1
        px = x0 + (idx / float(total)) * w
        py = y1 - (captured / float(total_defects)) * h
        pts.append((px, py))

    curve_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    area_str = f"{x0:.1f},{y1:.1f} {curve_str} {x1:.1f},{y1:.1f}"
    markers = []
    for frac in (0.10, 0.20, 0.30):
        k = min(total, max(1, int((total * frac) + 0.999999)))
        cap_k = sum(1 for sid in ranked_sample_ids[:k] if labels_by_id.get(sid) == "defect")
        mx = x0 + (k / float(total)) * w
        my = y1 - (cap_k / float(total_defects)) * h
        markers.append(
            {
                "label": f"Top {int(frac * 100)}%",
                "x": round(mx, 1),
                "y": round(my, 1),
                "recall_pct": round((cap_k / float(total_defects)) * 100.0, 1),
            }
        )
    return {
        "available": True,
        "curve_points": curve_str,
        "area_points": area_str,
        "markers": markers,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }


def _heat_fill(count: int, max_count: int, palette: str) -> str:
    if count <= 0 or max_count <= 0:
        return "#f8fafc"
    t = min(max(float(count) / float(max_count), 0.15), 1.0)
    if palette == "ng":
        r = int(254 - t * (254 - 185))
        g = int(226 - t * (226 - 28))
        b = int(226 - t * (226 - 28))
    else:
        r = int(254 - t * (254 - 180))
        g = int(243 - t * (243 - 83))
        b = int(199 - t * (199 - 9))
    return f"rgb({r},{g},{b})"


def _build_strip_hotspot_heatmap(
    records: list[dict[str, Any]],
    manifest_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    cols, rows = 6, 14
    cell = 24.0
    pad = 12.0
    ng_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    rev_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    ng_total = 0
    ng_edge = 0
    rev_total = 0
    rev_edge = 0

    for rec in records:
        status = str(rec.get("status", ""))
        if status not in {"NG", "REVIEW"}:
            continue
        comps = rec.get("components") or []
        if not comps:
            continue
        if not isinstance(comps, list):
            raise UiInputError("results.jsonl components 필드가 배열이 아닙니다.")
        sid = str(rec.get("sample_id", ""))
        mrow = manifest_map.get(sid) or {}
        try:
            img_w = int(mrow.get("width") or 0)
            img_h = int(mrow.get("height") or 0)
        except (TypeError, ValueError):
            img_w, img_h = 0, 0
        if img_w <= 0 or img_h <= 0:
            roi = rec.get("roi") or {}
            rb = roi.get("bbox") if isinstance(roi, dict) else None
            if isinstance(rb, (list, tuple)) and len(rb) == 4:
                try:
                    img_w = max(int(rb[0]) + int(rb[2]), 1)
                    img_h = max(int(rb[1]) + int(rb[3]), 1)
                except (TypeError, ValueError) as exc:
                    raise UiInputError("results.jsonl roi.bbox 좌표가 손상됐습니다.") from exc
        if img_w <= 0 or img_h <= 0:
            try:
                max_cx = max(
                    (
                        int(c["bbox"][0]) + int(c["bbox"][2])
                        for c in comps
                        if isinstance(c, dict)
                        and isinstance(c.get("bbox"), (list, tuple))
                        and len(c["bbox"]) == 4
                    ),
                    default=230,
                )
                max_cy = max(
                    (
                        int(c["bbox"][1]) + int(c["bbox"][3])
                        for c in comps
                        if isinstance(c, dict)
                        and isinstance(c.get("bbox"), (list, tuple))
                        and len(c["bbox"]) == 4
                    ),
                    default=630,
                )
            except (TypeError, ValueError) as exc:
                raise UiInputError("results.jsonl component.bbox 좌표가 손상됐습니다.") from exc
            img_w = max(max_cx, 1)
            img_h = max(max_cy, 1)

        for c in comps:
            if not isinstance(c, dict):
                raise UiInputError("results.jsonl component 항목이 객체가 아닙니다.")
            bbox = c.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise UiInputError(f"results.jsonl component.bbox 형식이 손상됐습니다: {bbox!r}")
            try:
                cx = float(bbox[0]) + float(bbox[2]) / 2.0
                cy = float(bbox[1]) + float(bbox[3]) / 2.0
            except (TypeError, ValueError) as exc:
                raise UiInputError("results.jsonl component.bbox 좌표가 손상됐습니다.") from exc
            nx = min(max(cx / float(img_w), 0.0), 0.999999)
            ny = min(max(cy / float(img_h), 0.0), 0.999999)
            gx = int(nx * cols)
            gy = int(ny * rows)
            is_edge = nx <= 0.15 or nx >= 0.85
            if status == "NG":
                ng_grid[gy][gx] += 1
                ng_total += 1
                if is_edge:
                    ng_edge += 1
            else:
                rev_grid[gy][gx] += 1
                rev_total += 1
                if is_edge:
                    rev_edge += 1

    ng_max = max((v for row in ng_grid for v in row), default=0)
    rev_max = max((v for row in rev_grid for v in row), default=0)

    def _cells(grid: list[list[int]], max_val: int, palette: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ry in range(rows):
            for cx_idx in range(cols):
                cnt = grid[ry][cx_idx]
                out.append(
                    {
                        "row": ry,
                        "col": cx_idx,
                        "x": round(pad + cx_idx * cell, 1),
                        "y": round(pad + ry * cell, 1),
                        "w": cell,
                        "h": cell,
                        "count": cnt,
                        "fill": _heat_fill(cnt, max_val, palette),
                    }
                )
        return out

    grid_w = cols * cell
    grid_h = rows * cell
    return {
        "cols": cols,
        "rows": rows,
        "view_box": f"0 0 {int(pad * 2 + grid_w)} {int(pad * 2 + grid_h)}",
        "pad": pad,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "left_15_x": round(pad + 0.15 * grid_w, 1),
        "right_15_x": round(pad + 0.85 * grid_w, 1),
        "ng_cells": _cells(ng_grid, ng_max, "ng"),
        "review_cells": _cells(rev_grid, rev_max, "review"),
        "ng_total_hits": ng_total,
        "review_total_hits": rev_total,
        "ng_max_cell": ng_max,
        "review_max_cell": rev_max,
        "ng_edge_ratio": ratio("NG 후보 중 좌·우 가장자리(15%) 집중 비율", ng_edge, ng_total),
        "review_edge_ratio": ratio(
            "REVIEW 후보 중 좌·우 가장자리(15%) 집중 비율", rev_edge, rev_total
        ),
    }


def create_app(
    *,
    runs_dir: str | Path,
    raw_root: str | Path,
    db_path: str | Path | None = None,
    enable_api: bool = False,
    repo_root: str | Path | None = None,
) -> FastAPI:
    """run 디렉터리와 원본 데이터 루트를 읽는 UI 앱을 만든다.

    ``db_path``가 없으면 ``runs_dir/records.db``를 **읽기 전용**으로 본다.
    ``enable_api=False``일 때는 DB 파일을 절대 새로 만들지 않는다.
    ``enable_api=True``일 때는 ``/api/*`` 검사 API와 실시간 검사 워크벤치를 함께 연다.
    """

    app = FastAPI(
        title="Surface Sentinel 로컬 UI",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runs_dir = Path(runs_dir)
    app.state.raw_root = Path(raw_root)
    app.state.db_path = (
        Path(db_path) if db_path is not None else Path(runs_dir) / DEFAULT_DB_NAME
    )
    app.state.api_enabled = bool(enable_api)
    app.state.repo_root = (
        Path(repo_root) if repo_root is not None else _resolve_repo_root()
    )
    configure_opencv_threads()

    def _check_origin(request: Request) -> None:
        host = request.headers.get("host") or request.url.netloc
        try:
            validate_same_origin(request.headers, host)
        except CsrfOriginError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    if app.state.api_enabled:
        api_app = create_inspection_app(
            runs_dir=app.state.runs_dir,
            db_path=app.state.db_path,
            raw_root=app.state.raw_root,
        )
        for route in api_app.router.routes:
            route_path = getattr(route, "path", "")
            if isinstance(route_path, str) and route_path.startswith("/api"):
                app.router.routes.append(route)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["version"] = __version__
    templates.env.globals["status_labels"] = STATUS_LABELS
    templates.env.globals["reason_labels"] = REASON_LABELS
    templates.env.globals["verdict_reason_labels"] = VERDICT_REASON_LABELS
    templates.env.globals["metric_labels"] = METRIC_LABELS

    def render(
        request: Request,
        name: str,
        context: dict[str, Any],
        status_code: int = 200,
    ) -> HTMLResponse:
        payload: dict[str, Any] = {
            "request": request,
            "api_enabled": app.state.api_enabled,
            "current_path": request.url.path,
        }
        payload.update(context)
        return templates.TemplateResponse(request, name, payload, status_code=status_code)

    def sample_view(
        entry: RunEntry,
        record: dict[str, Any],
        latest: dict[str, Any],
        *,
        ng_min: float = 0.50,
        review_band_max: float = REVIEW_BAND_MAX,
        sort_mode: str = "",
    ) -> dict[str, Any]:
        if "sample_id" not in record or "status" not in record:
            raise UiInputError("results.jsonl 레코드에 필수 키(sample_id/status)가 없습니다.")
        sample_id = str(record["sample_id"])
        decision = latest.get(sample_id)
        parsed = sample_result_from_record(record)
        geom = analyze_weak_geometry(parsed.components, parsed.roi)
        triage = assess_review_triage(
            parsed,
            geometry=geom,
            ng_min=ng_min,
            review_band_max=review_band_max,
        )
        suffix = "&sort=priority" if sort_mode == "priority" else ""
        return {
            "record": record,
            "sample_id": sample_id,
            "machine_status": str(record["status"]),
            "effective_status": decision.final_status if decision else str(record["status"]),
            "decision": decision,
            "geometry": geom,
            "triage": triage,
            "priority": triage.priority.value,
            "priority_score": triage.priority_score,
            "url": f"/runs/{entry.run_id}/samples/{quote(sample_id)}",
            "review_url": f"/runs/{entry.run_id}/review?sample={quote(sample_id)}{suffix}",
            "overlay_url": f"/runs/{entry.run_id}/samples/{quote(sample_id)}/overlay.png",
            "original_url": f"/runs/{entry.run_id}/samples/{quote(sample_id)}/original.png",
        }

    def stage_paths(entry: RunEntry, sample_id: str) -> dict[str, str | None]:
        safe_id = sanitize_sample_id(sample_id)
        urls: dict[str, str | None] = {}
        for kind in STAGE_KINDS:
            path = entry.run_dir / "stages" / f"{safe_id}_{kind}.png"
            base = f"/runs/{entry.run_id}/samples/{quote(sample_id)}"
            urls[kind] = f"{base}/{kind}.png" if path.is_file() else None
        return urls

    def original_available(entry: RunEntry, sample_id: str) -> bool:
        row = load_manifest_map(entry.run_dir).get(sample_id)
        if row is None or not row.get("image_path"):
            return False
        try:
            path = safe_child(app.state.raw_root, row["image_path"])
        except UiInputError:
            return False
        return path.is_file()

    def run_metrics(entry: RunEntry) -> dict[str, Any]:
        records = load_results(entry.run_dir)
        try:
            confusion = Confusion.from_pairs(
                (str(record["label"]), str(record["status"])) for record in records
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UiInputError(str(exc)) from exc
        return {
            "run": entry,
            "confusion": confusion,
            "ratios": image_metrics(confusion),
            "total": len(records),
        }

    def mask_layers(entry: RunEntry, record: dict[str, Any]) -> list[dict[str, Any]]:
        if "sample_id" not in record:
            raise UiInputError("results.jsonl 레코드에 sample_id가 없습니다.")
        sample_id = str(record["sample_id"])
        stages = stage_paths(entry, sample_id)
        base = f"/runs/{entry.run_id}/samples/{quote(sample_id)}"
        urls: dict[str, str | None] = {
            "original": f"{base}/original.png" if original_available(entry, sample_id) else None,
            "roi": stages.get("roi"),
            "candidate": stages.get("candidates"),
            "refined": stages.get("refined"),
            "overlay": f"{base}/overlay.png" if record.get("overlay_path") else None,
        }
        return [
            {"key": key, "label": label, "url": urls.get(key)}
            for key, label in MASK_LAYERS
        ]

    @app.exception_handler(UiInputError)
    def handle_input_error(request: Request, exc: UiInputError) -> HTMLResponse:
        return render(request, "error.html", {"code": 404, "message": str(exc)}, 404)

    @app.exception_handler(ReviewError)
    def handle_review_error(request: Request, exc: ReviewError) -> HTMLResponse:
        return render(request, "error.html", {"code": 400, "message": str(exc)}, 400)

    @app.exception_handler(ReviewStorageError)
    def handle_storage_error(request: Request, exc: ReviewStorageError) -> HTMLResponse:
        return render(request, "error.html", {"code": 500, "message": str(exc)}, 500)

    @app.exception_handler(StarletteHTTPException)
    def handle_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> HTMLResponse | JSONResponse:
        if request.url.path.startswith(("/api/", "/cinema/preset/", "/cinema/inspect")):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        message = (
            "요청한 페이지가 없습니다."
            if exc.status_code == 404
            else str(exc.detail or "요청을 처리할 수 없습니다.")
        )
        return render(
            request,
            "error.html",
            {"code": exc.status_code, "message": message},
            exc.status_code,
        )

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(DatabaseError)
    def handle_db_error(
        request: Request, exc: sqlite3.Error | DatabaseError
    ) -> HTMLResponse | JSONResponse:
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {"detail": "검사 기록 데이터베이스를 처리할 수 없습니다."},
                status_code=503,
            )
        return render(
            request,
            "error.html",
            {"code": 404, "message": "검사 기록 데이터베이스를 읽을 수 없습니다."},
            404,
        )

    @app.get("/", response_class=HTMLResponse)
    def runs_index(request: Request) -> HTMLResponse:
        entries = discover_runs(app.state.runs_dir)
        total_samples = 0
        for e in entries:
            counts_val = e.payload.get("counts")
            if isinstance(counts_val, dict):
                try:
                    total_samples += int(counts_val.get("samples") or 0)
                except (TypeError, ValueError):
                    continue
        preview_samples: list[dict[str, Any]] = []
        if entries:
            latest_entry = entries[0]
            try:
                records = load_results(latest_entry.run_dir)
                latest = ReviewStore(latest_entry.run_dir).latest()
                ng_min, review_band_max = _run_scoring_bounds(latest_entry)
                priority_order = {"REVIEW": 0, "NG": 1, "OK": 2, "ERROR": 3}
                with_overlay = [r for r in records if r.get("overlay_path")]
                with_overlay.sort(
                    key=lambda rec: (
                        priority_order.get(str(rec.get("status", "")), 9),
                        -float(rec.get("score", 0.0)),
                    )
                )
                for record in with_overlay[:6]:
                    preview_samples.append(
                        sample_view(
                            latest_entry,
                            record,
                            latest,
                            ng_min=ng_min,
                            review_band_max=review_band_max,
                        )
                    )
            except (UiInputError, ReviewStorageError, KeyError, TypeError, ValueError):
                preview_samples = []
        return render(
            request,
            "runs.html",
            {
                "runs": entries,
                "total_samples": total_samples,
                "preview_samples": preview_samples,
            },
        )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "surface-sentinel-ui",
                "version": __version__,
                "runs": len(discover_runs(app.state.runs_dir)),
                "api_enabled": app.state.api_enabled,
            }
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        records = load_results(entry.run_dir)
        latest = ReviewStore(entry.run_dir).latest()
        ng_min, review_band_max = _run_scoring_bounds(entry)
        samples = [
            sample_view(
                entry,
                record,
                latest,
                ng_min=ng_min,
                review_band_max=review_band_max,
            )
            for record in records
        ]
        review_samples = [s for s in samples if s["machine_status"] == "REVIEW"]
        review_done = sum(1 for s in review_samples if s["decision"] is not None)
        return render(
            request,
            "run_detail.html",
            {
                "run": entry,
                "samples": samples,
                "review_total": len(review_samples),
                "review_done": review_done,
            },
        )

    @app.get("/runs/{run_id}/samples/{sample_id}", response_class=HTMLResponse)
    def sample_detail(request: Request, run_id: str, sample_id: str) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        record = find_sample(load_results(entry.run_dir), sample_id)
        store = ReviewStore(entry.run_dir)
        latest = store.latest()
        history = [d for d in store.read_all() if d.sample_id == sample_id]
        run_cfg = _run_app_config(entry)
        ng_min, review_band_max = float(run_cfg.scoring.ng_min), REVIEW_BAND_MAX
        counterfactual = explain_counterfactual(sample_result_from_record(record), run_cfg)
        timing_raw = record.get("timing_ms") or {}
        if not isinstance(timing_raw, dict):
            raise UiInputError("results.jsonl timing_ms 필드가 객체가 아닙니다.")
        try:
            stage_items = [(str(k), float(v)) for k, v in timing_raw.items() if str(k) != "total"]
            max_stage = max((v for _, v in stage_items), default=0.0)
            if max_stage <= 0.0:
                max_stage = max((float(v) for v in timing_raw.values()), default=1.0) or 1.0
            timing_bars = [
                {
                    "name": str(k),
                    "ms": float(v),
                    "pct": 100.0
                    if str(k) == "total"
                    else min(100.0, round((float(v) / max_stage) * 100.0, 1)),
                }
                for k, v in timing_raw.items()
            ]
        except (TypeError, ValueError) as exc:
            raise UiInputError("results.jsonl timing_ms 값이 손상됐습니다.") from exc
        return render(
            request,
            "sample.html",
            {
                "run": entry,
                "sample": sample_view(
                    entry,
                    record,
                    latest,
                    ng_min=ng_min,
                    review_band_max=review_band_max,
                ),
                "history": history,
                "stages": stage_paths(entry, sample_id),
                "has_original": original_available(entry, sample_id),
                "saved": request.query_params.get("saved") == "1",
                "ng_min": ng_min,
                "review_band_max": review_band_max,
                "timing_bars": timing_bars,
                "counterfactual": counterfactual,
            },
        )

    def _media_file(run_id: str, sample_id: str, kind: str) -> FileResponse:
        entry = load_run(app.state.runs_dir, run_id)
        record = find_sample(load_results(entry.run_dir), sample_id)
        if kind == "overlay":
            relative = str(record.get("overlay_path") or "")
            if not relative:
                raise UiInputError(NO_IMAGE_TEXT)
            path = safe_child(entry.run_dir, relative)
        elif kind == "original":
            row = load_manifest_map(entry.run_dir).get(sample_id)
            if row is None or not row.get("image_path"):
                raise UiInputError(NO_IMAGE_TEXT)
            path = safe_child(app.state.raw_root, row["image_path"])
        else:
            safe_id = sanitize_sample_id(sample_id)
            path = safe_child(entry.run_dir, f"stages/{safe_id}_{kind}.png")
        if not path.is_file():
            raise UiInputError(NO_IMAGE_TEXT)
        return FileResponse(path, media_type="image/png")

    @app.get("/runs/{run_id}/samples/{sample_id}/overlay.png")
    def overlay_image(run_id: str, sample_id: str) -> FileResponse:
        return _media_file(run_id, sample_id, "overlay")

    @app.get("/runs/{run_id}/samples/{sample_id}/original.png")
    def original_image(run_id: str, sample_id: str) -> FileResponse:
        return _media_file(run_id, sample_id, "original")

    @app.get("/runs/{run_id}/samples/{sample_id}/response.png")
    def response_image(run_id: str, sample_id: str) -> FileResponse:
        return _media_file(run_id, sample_id, "response")

    @app.get("/runs/{run_id}/samples/{sample_id}/candidates.png")
    def candidates_image(run_id: str, sample_id: str) -> FileResponse:
        return _media_file(run_id, sample_id, "candidates")

    @app.get("/runs/{run_id}/samples/{sample_id}/roi.png")
    def roi_image(run_id: str, sample_id: str) -> FileResponse:
        return _media_file(run_id, sample_id, "roi")

    @app.get("/runs/{run_id}/samples/{sample_id}/refined.png")
    def refined_image(run_id: str, sample_id: str) -> FileResponse:
        return _media_file(run_id, sample_id, "refined")

    @app.get("/runs/{run_id}/samples/{sample_id}/mask", response_class=HTMLResponse)
    def mask_viewer(request: Request, run_id: str, sample_id: str) -> HTMLResponse:
        """원본·ROI·candidate·refined·오버레이를 토글 및 중첩 슬라이더로 비교한다."""

        entry = load_run(app.state.runs_dir, run_id)
        record = find_sample(load_results(entry.run_dir), sample_id)
        latest = ReviewStore(entry.run_dir).latest()
        ng_min, review_band_max = _run_scoring_bounds(entry)
        layers = mask_layers(entry, record)
        available_layers = [layer for layer in layers if layer["url"]]
        return render(
            request,
            "mask_viewer.html",
            {
                "run": entry,
                "sample": sample_view(
                    entry,
                    record,
                    latest,
                    ng_min=ng_min,
                    review_band_max=review_band_max,
                ),
                "layers": layers,
                "available_layers": available_layers,
            },
        )

    @app.get("/runs/{run_id}/review", response_class=HTMLResponse)
    def review_page(
        request: Request,
        run_id: str,
        sample: str | None = None,
        sort: str = "",
    ) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        records = load_results(entry.run_dir)
        store = ReviewStore(entry.run_dir)
        latest = store.latest()
        ng_min, review_band_max = _run_scoring_bounds(entry)
        sort_mode = "priority" if sort == "priority" else ""
        queue = [r for r in records if str(r.get("status", "")) == "REVIEW"]
        views = [
            sample_view(
                entry,
                record,
                latest,
                ng_min=ng_min,
                review_band_max=review_band_max,
                sort_mode=sort_mode,
            )
            for record in queue
        ]
        if sort_mode == "priority":
            views.sort(key=lambda item: (-float(item["priority_score"]), item["sample_id"]))
        current = None
        if views:
            if sample is not None:
                current = next((v for v in views if v["sample_id"] == sample), None)
                if current is None:
                    raise UiInputError(f"검토 큐에 없는 샘플입니다: {sample}")
            else:
                current = next((v for v in views if v["decision"] is None), views[0])
        index = views.index(current) if current is not None else -1
        history = (
            [d for d in store.read_all() if d.sample_id == current["sample_id"]]
            if current is not None
            else []
        )
        return render(
            request,
            "review.html",
            {
                "run": entry,
                "queue": views,
                "current": current,
                "history": history,
                "sort_mode": sort_mode,
                "stages": stage_paths(entry, current["sample_id"]) if current else {},
                "has_original": (
                    original_available(entry, current["sample_id"]) if current else False
                ),
                "prev_url": views[index - 1]["review_url"] if index > 0 else None,
                "next_url": (
                    views[index + 1]["review_url"] if 0 <= index < len(views) - 1 else None
                ),
                "saved": request.query_params.get("saved") == "1",
            },
        )

    @app.post("/runs/{run_id}/review/{sample_id}")
    def review_submit(
        request: Request,
        run_id: str,
        sample_id: str,
        final_status: str = Form(...),
        reason_code: str = Form(...),
        note: str = Form(""),
        reviewer: str = Form("local"),
        return_to: str = Form("review"),
        sort: str = Form(""),
    ) -> RedirectResponse:
        _check_origin(request)
        entry = load_run(app.state.runs_dir, run_id)
        record = find_sample(load_results(entry.run_dir), sample_id)
        if str(record["status"]) != "REVIEW":
            raise ReviewError(
                f"기계 판정 REVIEW만 검토할 수 있습니다 (OK/NG 불변, 현재: {record['status']})."
            )
        relative = str(record.get("overlay_path") or "")
        if not relative:
            raise ReviewError("오버레이가 없어 검토 결정을 기록할 수 없습니다.")
        overlay_file = safe_child(entry.run_dir, relative)
        if not overlay_file.is_file():
            raise ReviewError("오버레이 파일이 없어 검토 결정을 기록할 수 없습니다.")
        ReviewStore(entry.run_dir).append(
            sample_id=sample_id,
            machine_status=str(record["status"]),
            final_status=final_status,
            reason_code=reason_code,
            note=note,
            reviewer=reviewer,
            config_hash=str(record.get("config_hash") or ""),
            overlay_sha256=sha256_file(overlay_file),
        )
        if return_to == "sample":
            target = f"/runs/{run_id}/samples/{quote(sample_id)}?saved=1"
        else:
            sort_q = "&sort=priority" if sort == "priority" else ""
            target = f"/runs/{run_id}/review?sample={quote(sample_id)}&saved=1{sort_q}"
        return RedirectResponse(target, status_code=303)

    @app.get("/runs/{run_id}/metrics", response_class=HTMLResponse)
    def metrics_page(request: Request, run_id: str) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        records = load_results(entry.run_dir)
        try:
            confusion = Confusion.from_pairs(
                (str(r["label"]), str(r["status"])) for r in records
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UiInputError(str(exc)) from exc
        ratios = image_metrics(confusion)
        latest = ReviewStore(entry.run_dir).latest()
        ng_min, review_band_max = _run_scoring_bounds(entry)
        review_records = [r for r in records if str(r.get("status", "")) == "REVIEW"]
        review_done = sum(1 for r in review_records if str(r.get("sample_id", "")) in latest)
        priority_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for r in review_records:
            parsed = sample_result_from_record(r)
            triage = assess_review_triage(
                parsed, ng_min=ng_min, review_band_max=review_band_max
            )
            priority_counts[triage.priority.value] += 1
        escapes = [r for r in records if r.get("label") == "defect" and r.get("status") == "OK"]
        false_ng = [r for r in records if r.get("label") == "good" and r.get("status") == "NG"]
        hotspot = _build_strip_hotspot_heatmap(records, load_manifest_map(entry.run_dir))
        return render(
            request,
            "metrics.html",
            {
                "run": entry,
                "confusion": confusion,
                "ratios": ratios,
                "review_total": len(review_records),
                "review_done": review_done,
                "priority_counts": priority_counts,
                "escapes": escapes,
                "false_ng": false_ng,
                "hotspot": hotspot,
            },
        )

    @app.get("/runs/{run_id}/triage", response_class=HTMLResponse)
    def triage_page(request: Request, run_id: str) -> HTMLResponse:
        """Phase 2 에이전트 트리아지 & Recall@Top-K Lift 분석실."""

        entry = load_run(app.state.runs_dir, run_id)
        records = load_results(entry.run_dir)
        latest = ReviewStore(entry.run_dir).latest()
        ng_min, review_band_max = _run_scoring_bounds(entry)
        review_records = [r for r in records if str(r.get("status", "")) == "REVIEW"]
        try:
            by_id = {str(r["sample_id"]): r for r in review_records}
        except KeyError as exc:
            raise UiInputError("results.jsonl 레코드에 sample_id가 없습니다.") from exc
        assessments = []
        geoms_by_id = {}
        for r in review_records:
            parsed = sample_result_from_record(r)
            geom = analyze_weak_geometry(parsed.components, parsed.roi)
            geoms_by_id[parsed.sample_id] = geom
            assessments.append(
                assess_review_triage(
                    parsed,
                    geometry=geom,
                    ng_min=ng_min,
                    review_band_max=review_band_max,
                )
            )
        ranked = rank_review_queue(assessments)
        ranked_ids = [item.sample_id for item in ranked]
        labels_by_id = {
            sid: str(rec.get("label", "unknown")) for sid, rec in by_id.items()
        }
        lift = evaluate_triage_lift(
            ranked_ids, labels_by_id, top_fractions=(0.10, 0.20, 0.30)
        )
        gain_svg = _build_gain_svg(ranked_ids, labels_by_id)

        priority_summary = []
        for band in ("P0", "P1", "P2", "P3"):
            band_items = [item for item in ranked if item.priority.value == band]
            count = len(band_items)
            decided = sum(1 for item in band_items if item.sample_id in latest)
            defects = sum(
                1 for item in band_items if labels_by_id.get(item.sample_id) == "defect"
            )
            mean_score = (
                sum(item.priority_score for item in band_items) / float(count)
                if count
                else 0.0
            )
            priority_summary.append(
                {
                    "band": band,
                    "count": count,
                    "ratio": ratio(f"{band} 비율", count, len(ranked)),
                    "decided": decided,
                    "defects": defects,
                    "mean_score": mean_score,
                }
            )

        pattern_counts: dict[str, int] = {}
        for item in ranked:
            pname = item.weak_pattern.value
            pattern_counts[pname] = pattern_counts.get(pname, 0) + 1

        rows = []
        for rank_idx, item in enumerate(ranked, start=1):
            rec = by_id[item.sample_id]
            decision = latest.get(item.sample_id)
            rows.append(
                {
                    "rank": rank_idx,
                    "triage": item,
                    "geometry": geoms_by_id[item.sample_id],
                    "record": rec,
                    "decision": decision,
                    "url": f"/runs/{entry.run_id}/samples/{quote(item.sample_id)}",
                    "review_url": (
                        f"/runs/{entry.run_id}/review?sample={quote(item.sample_id)}&sort=priority"
                    ),
                }
            )

        return render(
            request,
            "triage.html",
            {
                "run": entry,
                "total_records": len(records),
                "ranked": ranked,
                "rows": rows,
                "lift": lift,
                "gain_svg": gain_svg,
                "priority_summary": priority_summary,
                "pattern_counts": sorted(
                    pattern_counts.items(), key=lambda kv: (-kv[1], kv[0])
                ),
            },
        )

    @app.get("/stats", response_class=HTMLResponse)
    def stats_page(request: Request) -> HTMLResponse:
        """검사 기록 DB 통계 + run별 요약."""

        return render(
            request,
            "stats.html",
            {
                "stats": load_stats_view(app.state.db_path),
                "runs": discover_runs(app.state.runs_dir),
            },
        )

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(
        request: Request,
        base: str | None = None,
        target: str | None = None,
    ) -> HTMLResponse:
        """두 run의 지표를 나란히 비교한다(Δ = target - base 포함)."""

        base_view = run_metrics(load_run(app.state.runs_dir, base)) if base else None
        target_view = run_metrics(load_run(app.state.runs_dir, target)) if target else None
        rows: list[dict[str, Any]] = []
        if base_view is not None and target_view is not None:
            for name, base_ratio in base_view["ratios"].items():
                target_ratio = target_view["ratios"][name]
                delta_val: float | None = None
                delta_cls = "neutral"
                if base_ratio.value is not None and target_ratio.value is not None:
                    delta_val = float(target_ratio.value) - float(base_ratio.value)
                    if abs(delta_val) < 1e-9:
                        delta_cls = "neutral"
                    elif name in LOWER_IS_BETTER:
                        delta_cls = "good" if delta_val < 0 else "bad"
                    else:
                        delta_cls = "good" if delta_val > 0 else "bad"
                rows.append(
                    {
                        "name": name,
                        "definition": base_ratio.definition,
                        "base": base_ratio,
                        "target": target_ratio,
                        "delta": delta_val,
                        "delta_cls": delta_cls,
                    }
                )
        return render(
            request,
            "compare.html",
            {
                "runs": discover_runs(app.state.runs_dir),
                "base": base or "",
                "target": target or "",
                "base_view": base_view,
                "target_view": target_view,
                "rows": rows,
            },
        )

    @app.get("/watchdog", response_class=HTMLResponse)
    def watchdog_page(
        request: Request,
        run_id: str | None = None,
        baseline: str | None = None,
        target: str | None = None,
    ) -> HTMLResponse:
        """Phase 3 Watchdog 무결성 감사 & PSI 드리프트 관제."""

        runs = discover_runs(app.state.runs_dir)
        gate_report = audit_repository_gates(app.state.repo_root)

        selected_run_id = run_id or (runs[0].run_id if runs else "")
        run_audit = None
        calibration = None
        if selected_run_id:
            entry = load_run(app.state.runs_dir, selected_run_id)
            run_audit = audit_run_dir(entry.run_dir)
            wd_records = load_results(entry.run_dir)
            wd_decisions = ReviewStore(entry.run_dir).read_all()
            wd_ng_min, wd_review_max = _run_scoring_bounds(entry)
            try:
                calibration = analyze_human_feedback(
                    wd_records,
                    wd_decisions,
                    ng_min=wd_ng_min,
                    review_band_max=wd_review_max,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise UiInputError(f"손상된 run 결과를 분석할 수 없습니다: {exc}") from exc

        db_exists = app.state.db_path.is_file()
        db_audit = audit_records_db(app.state.db_path)

        default_base = runs[-1].run_id if len(runs) >= 2 else (runs[0].run_id if runs else "")
        base_id = baseline or default_base
        target_id = target or (runs[0].run_id if runs else "")
        drift_report = None
        drift_gauges: list[dict[str, Any]] = []
        if base_id and target_id:
            base_entry = load_run(app.state.runs_dir, base_id)
            target_entry = load_run(app.state.runs_dir, target_id)
            try:
                drift_report = assess_drift(
                    load_results(base_entry.run_dir),
                    load_results(target_entry.run_dir),
                    policy=DEFAULT_DRIFT_POLICY,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise UiInputError(
                    f"손상된 run 결과로 드리프트를 계산할 수 없습니다: {exc}"
                ) from exc
            scale_max = 0.35
            for feat in drift_report.features:
                drift_gauges.append(
                    {
                        "feature": feat,
                        "psi_pct": min(100.0, round((feat.psi / scale_max) * 100.0, 1)),
                        "watch_pct": round((DEFAULT_DRIFT_POLICY.watch_psi / scale_max) * 100.0, 1),
                        "alert_pct": round((DEFAULT_DRIFT_POLICY.alert_psi / scale_max) * 100.0, 1),
                    }
                )

        return render(
            request,
            "watchdog.html",
            {
                "runs": runs,
                "gate_report": gate_report,
                "selected_run_id": selected_run_id,
                "run_audit": run_audit,
                "db_exists": db_exists,
                "db_path": str(app.state.db_path),
                "db_audit": db_audit,
                "base_id": base_id,
                "target_id": target_id,
                "drift_report": drift_report,
                "drift_gauges": drift_gauges,
                "drift_policy": DEFAULT_DRIFT_POLICY,
                "calibration": calibration,
            },
        )

    @app.get("/inspections", response_class=HTMLResponse)
    def inspections_page(
        request: Request,
        status: str | None = None,
        engine: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> HTMLResponse:
        """검사 기록 DB · 교차증거 탐색기 & 실시간 검사 워크벤치."""

        view = load_inspections_view(
            app.state.db_path,
            app.state.runs_dir,
            status=status,
            engine=engine,
            limit=max(1, min(limit, 200)),
            offset=max(0, offset),
        )
        return render(request, "inspections.html", {"db": view})

    def _resolve_demo_train_dir(raw_root: Path) -> Path | None:
        for candidate in (raw_root / "train", raw_root / "kolektorsdd2" / "train"):
            if candidate.is_dir():
                return candidate
        return None

    def _even_pick(items: list[str], count: int) -> list[str]:
        if count <= 0 or not items:
            return []
        if len(items) <= count:
            return list(items)
        return [items[(i * len(items)) // count] for i in range(count)]

    def _load_tune_train_pool(
        train_dir: Path, repo_root: Path
    ) -> tuple[list[str], list[str], set[str]]:
        tune_csv = repo_root / "data" / "splits" / "tune.csv"
        if not tune_csv.is_file():
            return [], [], set()
        defects: list[str] = []
        goods: list[str] = []
        with tune_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_split") != "train":
                    continue
                name = Path(row.get("image_path", "")).name
                if (
                    not name
                    or not name.lower().endswith(".png")
                    or Path(name).stem.endswith("_GT")
                    or not (train_dir / name).is_file()
                ):
                    continue
                if row.get("label") == "defect":
                    defects.append(name)
                else:
                    goods.append(name)
        defects_sorted = sorted(set(defects))
        goods_sorted = sorted(set(goods))
        allowed = set(defects_sorted) | set(goods_sorted)
        if not allowed:
            return [], [], set()
        return defects_sorted, goods_sorted, allowed

    demo_lock = threading.Lock()
    demo_queue: dict[str, Any] = {}

    @app.post("/inspections/demo-samples/reset")
    def inspection_demo_samples_reset() -> JSONResponse:
        """무작위 샘플 대기열을 비워 처음부터 다시 뽑게 한다(검사 기록은 그대로)."""

        with demo_lock:
            demo_queue.clear()
        return JSONResponse({"reset": True})

    @app.get("/inspections/demo-samples")
    def inspection_demo_samples(limit: int = 20) -> JSONResponse:
        """시연용 실제 공장 사진(train/tune) 목록을 반환한다 (봉인 test 미접근)."""

        capped = max(1, min(limit, 50))
        train_dir = _resolve_demo_train_dir(app.state.raw_root)
        if train_dir is None:
            return JSONResponse({"samples": [], "detail": "샘플 폴더를 찾을 수 없습니다."})
        defects, goods, allowed = _load_tune_train_pool(train_dir, app.state.repo_root)
        if allowed:
            # 서버가 켜져 있는 동안 중복 없이 소진: 섞어 둔 불량·정상 대기열에서
            # 매번 불량 약 25% + 정상 나머지를 꺼낸다. 다 쓰면 빈 목록 + 안내.
            rng = random.SystemRandom()
            with demo_lock:
                if demo_queue.get("pool") != (len(defects), len(goods)):
                    q_def, q_good = list(defects), list(goods)
                    rng.shuffle(q_def)
                    rng.shuffle(q_good)
                    demo_queue.update(
                        pool=(len(defects), len(goods)), defects=q_def, goods=q_good
                    )
                q_def, q_good = demo_queue["defects"], demo_queue["goods"]
                n_def = min(len(q_def), max(1, round(capped * 0.25)))
                n_good = min(len(q_good), capped - n_def)
                n_def = min(len(q_def), capped - n_good)
                chosen_def = [q_def.pop() for _ in range(n_def)]
                chosen_good = [q_good.pop() for _ in range(n_good)]
                remaining = len(q_def) + len(q_good)
            total = len(defects) + len(goods)
            if not chosen_def and not chosen_good:
                return JSONResponse({
                    "samples": [], "labels": {}, "remaining": 0, "total": total,
                    "detail": f"샘플 {total}장을 모두 검사했습니다. 더 남은 사진이 없습니다.",
                })
            picked = chosen_def + chosen_good
            rng.shuffle(picked)
            labels = {name: "defect" for name in chosen_def}
            labels.update({name: "good" for name in chosen_good})
            return JSONResponse(
                {"samples": picked, "labels": labels, "remaining": remaining, "total": total}
            )
        else:
            names = sorted(
                p.name
                for p in train_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png" and not p.stem.endswith("_GT")
            )
            picked = _even_pick(names, capped)
        return JSONResponse({"samples": picked})

    @app.get("/inspections/demo-samples/{filename}")
    def inspection_demo_sample_file(filename: str) -> FileResponse:
        """시연용 train 샘플 1장을 안전하게 반환한다."""

        train_dir = _resolve_demo_train_dir(app.state.raw_root)
        if train_dir is None:
            raise UiInputError(NO_IMAGE_TEXT)
        _, _, allowed = _load_tune_train_pool(train_dir, app.state.repo_root)
        target = safe_child(train_dir, filename)
        if not target.is_file() or target.suffix.lower() != ".png" or target.stem.endswith("_GT"):
            raise UiInputError(NO_IMAGE_TEXT)
        if allowed and target.name not in allowed:
            raise UiInputError(NO_IMAGE_TEXT)
        return FileResponse(target, media_type="image/png")

    @app.get("/inspections/export.csv")
    def inspections_export_csv(
        status: str | None = None,
        engine: str | None = None,
    ) -> Response:
        """검사 기록 DB(`inspections`)를 UTF-8 BOM CSV로 내려보낸다."""

        columns = (
            "id",
            "sample_id",
            "engine",
            "status",
            "score",
            "reasons",
            "label",
            "split",
            "config_hash",
            "created_at",
        )
        rows_out: list[tuple[Any, ...]] = []
        if app.state.db_path.is_file():
            connection = _read_only_connection(app.state.db_path)
            try:
                tables = {
                    str(r[0])
                    for r in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "inspections" in tables:
                    where: list[str] = []
                    params: list[Any] = []
                    if status:
                        where.append("status = ?")
                        params.append(status)
                    if engine:
                        where.append("engine = ?")
                        params.append(engine)
                    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
                    sql = (
                        f"SELECT {', '.join(columns)} FROM inspections"
                        f"{where_sql} ORDER BY id DESC"
                    )
                    for r in connection.execute(sql, params).fetchall():
                        rows_out.append(tuple(r[col] for col in columns))
            finally:
                connection.close()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        writer.writerows(rows_out)
        payload = buf.getvalue().encode("utf-8-sig")
        suffix = f"_{status.lower()}" if status and status in STATUS_ORDER else ""
        filename = f"inspections{suffix}.csv"
        return Response(
            content=payload,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/inspections/{inspection_id}", response_class=HTMLResponse)
    def inspection_detail_page(request: Request, inspection_id: int) -> HTMLResponse:
        """단일 검사 DB 행 · 엔진 비교 · 패치 중재 · 에이전트 논증 영수증."""

        detail = load_inspection_detail_view(
            app.state.db_path, app.state.runs_dir, inspection_id
        )
        result_json = detail.pop("result_json", None)
        explain = None
        if result_json is not None:
            insp = detail["inspection"]
            merged = {
                **result_json,
                "status": insp.get("status"),
                "score": insp.get("score"),
                "reasons": insp.get("reasons_list"),
            }
            explain = explain_verdict(merged, load_cinema_config(app.state.repo_root))
        return render(request, "inspection_detail.html", {**detail, "explain": explain})

    @app.get("/inspections/{inspection_id}/overlay.png")
    def inspection_overlay_image(inspection_id: int) -> FileResponse:
        if not app.state.db_path.is_file():
            raise UiInputError(NO_IMAGE_TEXT)
        try:
            connection = _read_only_connection(app.state.db_path)
            try:
                row = get_inspection(connection, inspection_id)
            finally:
                connection.close()
        except (sqlite3.Error, DatabaseError) as exc:
            raise UiInputError(f"검사 기록 DB를 읽지 못했습니다: {exc}") from exc
        if row is None:
            raise UiInputError(f"검사 기록을 찾을 수 없습니다: id={inspection_id}")
        path = row_overlay_path(row, app.state.runs_dir)
        if path is None:
            raise UiInputError(NO_IMAGE_TEXT)
        return FileResponse(path, media_type="image/png")

    @app.get("/cinema", response_class=HTMLResponse)
    def cinema_page(request: Request) -> HTMLResponse:
        """3D 레이어 분해 & 영상식 10단계 검사 시네마 시연 뷰어."""

        cfg = load_cinema_config(app.state.repo_root)
        presets = discover_cinema_presets(
            app.state.runs_dir,
            app.state.raw_root,
            app.state.repo_root,
            config_hash=cfg.config_hash,
        )
        return render(
            request,
            "cinema.html",
            {
                "presets": [p.to_public_dict() for p in presets],
                "chapters": list(CINEMA_CHAPTERS),
                "config_hash": cfg.config_hash,
                "ng_min": float(cfg.scoring.ng_min),
            },
        )

    @app.get("/cinema/preset/{preset_id}")
    async def cinema_preset_payload(preset_id: str) -> JSONResponse:
        """사전 등록된 tune/합성 프리셋의 실제 10단계 프레임·텔레메트리를 반환한다."""

        def _load_preset_and_config() -> tuple[Any, Any]:
            cfg_loaded = load_cinema_config(app.state.repo_root)
            presets = {
                p.preset_id: p
                for p in discover_cinema_presets(
                    app.state.runs_dir,
                    app.state.raw_root,
                    app.state.repo_root,
                    config_hash=cfg_loaded.config_hash,
                )
            }
            preset_found = presets.get(preset_id)
            if preset_found is None:
                return None, None
            return preset_found, cfg_loaded

        preset, cfg = await run_in_threadpool(_load_preset_and_config)
        if preset is None:
            return JSONResponse(
                {"detail": f"알 수 없는 시네마 프리셋입니다: {preset_id}"},
                status_code=404,
            )

        def _run_preset() -> dict[str, Any]:
            with pipeline_slot():
                return run_cinema_preset(preset, config=cfg)

        try:
            async with async_pipeline_slot():
                payload = await run_in_threadpool(_run_preset)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse(payload)

    @app.post("/cinema/inspect")
    async def cinema_inspect_upload(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> JSONResponse:
        """업로드 이미지의 실제 10단계 OpenCV 파이프라인을 임시 실행해 시네마 프레임을 반환한다.

        DB 및 runs 디렉터리에는 아무 파일도 남기지 않으며 항상 ``split="tune"`` 으로만 실행한다.
        """

        try:
            _check_origin(request)
            filename = Path(file.filename or "upload.png").name
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_CINEMA_EXTENSIONS:
                return JSONResponse(
                    {"detail": f"지원하지 않는 이미지 확장자입니다: {suffix or '(없음)'}"},
                    status_code=400,
                )

            try:
                payload_bytes = await read_upload_bounded(
                    file,
                    max_bytes=MAX_CINEMA_UPLOAD_BYTES,
                    chunk_bytes=CINEMA_UPLOAD_CHUNK_BYTES,
                )
            except UploadTooLargeError:
                return JSONResponse(
                    {
                        "detail": (
                            f"업로드 용량이 제한({MAX_CINEMA_UPLOAD_BYTES // (1024 * 1024)}MiB)을 "
                            "초과했습니다."
                        )
                    },
                    status_code=413,
                )
        finally:
            await file.close()

        if not payload_bytes:
            return JSONResponse({"detail": "빈 파일은 검사할 수 없습니다."}, status_code=400)

        try:
            check_image_pixel_budget(
                payload_bytes,
                max_pixels=_cinema_mod.MAX_CINEMA_IMAGE_PIXELS,
                require_dimensions=True,
            )
        except (ImagePixelLimitError, ImageHeaderError) as exc:
            return JSONResponse({"detail": f"이미지 검사 실패: {exc}"}, status_code=400)

        cfg = load_cinema_config(app.state.repo_root)
        sample_id = sanitize_sample_id(Path(filename).stem or "upload")

        def _run() -> dict[str, Any]:
            with (
                pipeline_slot(),
                tempfile.TemporaryDirectory(prefix="ss-cinema-up-") as tmp_dir,
            ):
                target = Path(tmp_dir) / f"{sample_id}{suffix}"
                target.write_bytes(payload_bytes)
                return run_cinema_trace(
                    target,
                    sample_id=sample_id,
                    config=cfg,
                    label="unknown",
                    source="upload",
                )

        try:
            async with async_pipeline_slot():
                data = await run_in_threadpool(_run)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse({"detail": f"이미지 검사 실패: {exc}"}, status_code=400)
        return JSONResponse(data)

    return app


__all__ = ["REASON_LABELS", "STATUS_LABELS", "create_app", "load_stats_view"]
