"""로컬 검토 UI (설계 §5.5, T07 — 산업용 정밀 계측 콘솔).

127.0.0.1 전용. 외부 CDN·외부 요청 없음.
기본 모드(``enable_api=False``)에서는 읽기 전용 + REVIEW 검토 결정 append
하나뿐이며 ``OK``/``NG``는 불변이다. ``enable_api=True``(``--with-api``)일 때만
``/api/*`` 로컬 검사 API와 실시간 검사 워크벤치를 함께 마운트한다.
한국어 문구와 영어 상태 배지를 함께 쓰고, 색상만으로 의미를 전달하지 않는다.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
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
from ..pipeline.evidence import sanitize_sample_id
from ..records.store import STATUS_ORDER, get_inspection, stats
from ..review import ReviewError, ReviewStorageError, ReviewStore
from ..version import __version__
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
    except sqlite3.Error as exc:
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
                img_w = max(int(rb[0]) + int(rb[2]), 1)
                img_h = max(int(rb[1]) + int(rb[3]), 1)
        if img_w <= 0 or img_h <= 0:
            max_cx = max(
                (int(c["bbox"][0]) + int(c["bbox"][2]) for c in comps if "bbox" in c),
                default=230,
            )
            max_cy = max(
                (int(c["bbox"][1]) + int(c["bbox"][3]) for c in comps if "bbox" in c),
                default=630,
            )
            img_w = max(max_cx, 1)
            img_h = max(max_cy, 1)

        for c in comps:
            bbox = c.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            cx = float(bbox[0]) + float(bbox[2]) / 2.0
            cy = float(bbox[1]) + float(bbox[3]) / 2.0
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
        except ValueError as exc:
            raise UiInputError(str(exc)) from exc
        return {
            "run": entry,
            "confusion": confusion,
            "ratios": image_metrics(confusion),
            "total": len(records),
        }

    def mask_layers(entry: RunEntry, record: dict[str, Any]) -> list[dict[str, Any]]:
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
        return render(
            request,
            "error.html",
            {"code": exc.status_code, "message": "요청한 페이지가 없습니다."},
            exc.status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    def runs_index(request: Request) -> HTMLResponse:
        entries = discover_runs(app.state.runs_dir)
        total_samples = sum(
            int((e.payload.get("counts") or {}).get("samples") or 0) for e in entries
        )
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
        queue = [r for r in records if str(r["status"]) == "REVIEW"]
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
        run_id: str,
        sample_id: str,
        final_status: str = Form(...),
        reason_code: str = Form(...),
        note: str = Form(""),
        reviewer: str = Form("local"),
        return_to: str = Form("review"),
        sort: str = Form(""),
    ) -> RedirectResponse:
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
        except ValueError as exc:
            raise UiInputError(str(exc)) from exc
        ratios = image_metrics(confusion)
        latest = ReviewStore(entry.run_dir).latest()
        ng_min, review_band_max = _run_scoring_bounds(entry)
        review_records = [r for r in records if str(r["status"]) == "REVIEW"]
        review_done = sum(1 for r in review_records if str(r["sample_id"]) in latest)
        priority_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for r in review_records:
            parsed = sample_result_from_record(r)
            triage = assess_review_triage(
                parsed, ng_min=ng_min, review_band_max=review_band_max
            )
            priority_counts[triage.priority.value] += 1
        escapes = [r for r in records if r["label"] == "defect" and r["status"] == "OK"]
        false_ng = [r for r in records if r["label"] == "good" and r["status"] == "NG"]
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
        review_records = [r for r in records if str(r["status"]) == "REVIEW"]
        by_id = {str(r["sample_id"]): r for r in review_records}
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
            calibration = analyze_human_feedback(
                wd_records,
                wd_decisions,
                ng_min=wd_ng_min,
                review_band_max=wd_review_max,
            )

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
            drift_report = assess_drift(
                load_results(base_entry.run_dir),
                load_results(target_entry.run_dir),
                policy=DEFAULT_DRIFT_POLICY,
            )
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

    @app.get("/inspections/{inspection_id}", response_class=HTMLResponse)
    def inspection_detail_page(request: Request, inspection_id: int) -> HTMLResponse:
        """단일 검사 DB 행 · 엔진 비교 · 패치 중재 · 에이전트 논증 영수증."""

        detail = load_inspection_detail_view(
            app.state.db_path, app.state.runs_dir, inspection_id
        )
        return render(request, "inspection_detail.html", detail)

    @app.get("/inspections/{inspection_id}/overlay.png")
    def inspection_overlay_image(inspection_id: int) -> FileResponse:
        if not app.state.db_path.is_file():
            raise UiInputError(NO_IMAGE_TEXT)
        connection = _read_only_connection(app.state.db_path)
        try:
            row = get_inspection(connection, inspection_id)
        finally:
            connection.close()
        if row is None:
            raise UiInputError(f"검사 기록을 찾을 수 없습니다: id={inspection_id}")
        path = row_overlay_path(row, app.state.runs_dir)
        if path is None:
            raise UiInputError(NO_IMAGE_TEXT)
        return FileResponse(path, media_type="image/png")

    @app.get("/cinema", response_class=HTMLResponse)
    def cinema_page(request: Request) -> HTMLResponse:
        """3D 레이어 분해 & 영상식 10단계 검사 시네마 시연 뷰어."""

        presets = discover_cinema_presets(
            app.state.runs_dir, app.state.raw_root, app.state.repo_root
        )
        cfg = load_cinema_config(app.state.repo_root)
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
    def cinema_preset_payload(preset_id: str) -> JSONResponse:
        """사전 등록된 tune/합성 프리셋의 실제 10단계 프레임·텔레메트리를 반환한다."""

        presets = {
            p.preset_id: p
            for p in discover_cinema_presets(
                app.state.runs_dir, app.state.raw_root, app.state.repo_root
            )
        }
        preset = presets.get(preset_id)
        if preset is None:
            return JSONResponse(
                {"detail": f"알 수 없는 시네마 프리셋입니다: {preset_id}"},
                status_code=404,
            )
        cfg = load_cinema_config(app.state.repo_root)
        try:
            payload = run_cinema_preset(preset, config=cfg)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse(payload)

    @app.post("/cinema/inspect")
    async def cinema_inspect_upload(file: Annotated[UploadFile, File(...)]) -> JSONResponse:
        """업로드 이미지의 실제 10단계 OpenCV 파이프라인을 임시 실행해 시네마 프레임을 반환한다.

        DB 및 runs 디렉터리에는 아무 파일도 남기지 않으며 항상 ``split="tune"`` 으로만 실행한다.
        """

        filename = Path(file.filename or "upload.png").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_CINEMA_EXTENSIONS:
            return JSONResponse(
                {"detail": f"지원하지 않는 이미지 확장자입니다: {suffix or '(없음)'}"},
                status_code=400,
            )

        chunks: list[bytes] = []
        total_read = 0
        while True:
            chunk = await file.read(CINEMA_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > MAX_CINEMA_UPLOAD_BYTES:
                return JSONResponse(
                    {
                        "detail": (
                            f"업로드 용량이 제한({MAX_CINEMA_UPLOAD_BYTES // (1024 * 1024)}MiB)을 "
                            "초과했습니다."
                        )
                    },
                    status_code=413,
                )
            chunks.append(chunk)

        payload_bytes = b"".join(chunks)
        if not payload_bytes:
            return JSONResponse({"detail": "빈 파일은 검사할 수 없습니다."}, status_code=400)

        cfg = load_cinema_config(app.state.repo_root)
        sample_id = sanitize_sample_id(Path(filename).stem or "upload")

        def _run() -> dict[str, Any]:
            with tempfile.TemporaryDirectory(prefix="ss-cinema-up-") as tmp_dir:
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
            data = await run_in_threadpool(_run)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse({"detail": f"이미지 검사 실패: {exc}"}, status_code=400)
        return JSONResponse(data)

    return app


__all__ = ["REASON_LABELS", "STATUS_LABELS", "create_app", "load_stats_view"]
