"""로컬 검토 UI (설계 §5.5, T07).

127.0.0.1 전용. 업로드·설정 편집·삭제 없음. 외부 CDN·외부 요청 없음.
쓰기는 REVIEW 샘플의 검토 결정 append 하나뿐이며 ``OK``/``NG``는 불변이다.
한국어 문구와 영어 상태 배지를 함께 쓰고, 색상만으로 의미를 전달하지 않는다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..api.app import engine_status_counts
from ..evaluation.metrics import Confusion, image_metrics, ratio
from ..pipeline.evidence import sanitize_sample_id
from ..records.store import STATUS_ORDER, stats
from ..review import ReviewError, ReviewStorageError, ReviewStore
from ..version import __version__
from .data import (
    RunEntry,
    UiInputError,
    discover_runs,
    find_sample,
    load_manifest_map,
    load_results,
    load_run,
    safe_child,
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
STATUS_LABELS = {"OK": "정상", "NG": "불량", "REVIEW": "보류", "ERROR": "오류(판정 아님)"}

#: 검토 사유 코드 → 한국어 설명. 어휘 정의는 ``review.store``가 정본이다.
REASON_LABELS = {
    "CONFIRMED_DEFECT": "결함 확인",
    "FALSE_ALARM": "오탐 확인(정상)",
    "INSUFFICIENT_EVIDENCE": "증거 부족",
    "IMAGE_QUALITY": "이미지 품질 문제",
    "OTHER": "기타(메모 필수)",
}

#: 이미지 파일이 없을 때 안내 문구.
NO_IMAGE_TEXT = "이미지 파일이 없습니다."


def _read_only_connection(path: Path) -> sqlite3.Connection:
    """검사 DB를 읽기 전용으로 연다(파일을 새로 만들지 않는다)."""

    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_stats_view(db_path: Path) -> dict[str, Any]:
    """검사 DB 통계를 UI 표시용으로 만든다.

    DB 파일이 없으면 **만들지 않고** 안내 문구만 돌려준다(UI는 읽기 전용).
    모든 비율은 :class:`~surface_sentinel.evaluation.metrics.Ratio`라 분모가
    함께 표시된다.
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


def create_app(
    *,
    runs_dir: str | Path,
    raw_root: str | Path,
    db_path: str | Path | None = None,
) -> FastAPI:
    """run 디렉터리와 원본 데이터 루트를 읽는 UI 앱을 만든다.

    ``db_path``가 없으면 ``runs_dir/records.db``를 **읽기 전용**으로 본다.
    통계 페이지는 DB가 없을 때 파일을 만들지 않는다.
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
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["version"] = __version__
    templates.env.globals["status_labels"] = STATUS_LABELS
    templates.env.globals["reason_labels"] = REASON_LABELS

    def render(
        request: Request,
        name: str,
        context: dict[str, Any],
        status_code: int = 200,
    ) -> HTMLResponse:
        payload: dict[str, Any] = {"request": request}
        payload.update(context)
        return templates.TemplateResponse(request, name, payload, status_code=status_code)

    def sample_view(entry: RunEntry, record: dict[str, Any], latest: dict) -> dict[str, Any]:
        sample_id = str(record["sample_id"])
        decision = latest.get(sample_id)
        return {
            "record": record,
            "sample_id": sample_id,
            "machine_status": str(record["status"]),
            "effective_status": decision.final_status if decision else str(record["status"]),
            "decision": decision,
            "url": f"/runs/{entry.run_id}/samples/{quote(sample_id)}",
            "review_url": f"/runs/{entry.run_id}/review?sample={quote(sample_id)}",
            "overlay_url": f"/runs/{entry.run_id}/samples/{quote(sample_id)}/overlay.png",
            "original_url": f"/runs/{entry.run_id}/samples/{quote(sample_id)}/original.png",
        }

    def stage_paths(entry: RunEntry, sample_id: str) -> dict[str, str | None]:
        """선택 저장된 단계 이미지(response/candidates/roi/refined)의 URL. 없으면 None."""

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
        """run 하나의 혼동행렬·이미지 지표(``/runs/{id}/metrics``와 같은 정의)."""

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
        """마스크 뷰어 레이어 목록. 산출물이 없으면 ``url``이 None이다."""

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
    def handle_http_error(request: Request, exc: StarletteHTTPException) -> HTMLResponse:
        return render(
            request,
            "error.html",
            {"code": exc.status_code, "message": "요청한 페이지가 없습니다."},
            exc.status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    def runs_index(request: Request) -> HTMLResponse:
        entries = discover_runs(app.state.runs_dir)
        return render(request, "runs.html", {"runs": entries})

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "surface-sentinel-ui",
                "version": __version__,
                "runs": len(discover_runs(app.state.runs_dir)),
            }
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        records = load_results(entry.run_dir)
        latest = ReviewStore(entry.run_dir).latest()
        samples = [sample_view(entry, record, latest) for record in records]
        return render(request, "run_detail.html", {"run": entry, "samples": samples})

    @app.get("/runs/{run_id}/samples/{sample_id}", response_class=HTMLResponse)
    def sample_detail(request: Request, run_id: str, sample_id: str) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        record = find_sample(load_results(entry.run_dir), sample_id)
        store = ReviewStore(entry.run_dir)
        latest = store.latest()
        history = [d for d in store.read_all() if d.sample_id == sample_id]
        return render(
            request,
            "sample.html",
            {
                "run": entry,
                "sample": sample_view(entry, record, latest),
                "history": history,
                "stages": stage_paths(entry, sample_id),
                "has_original": original_available(entry, sample_id),
                "saved": request.query_params.get("saved") == "1",
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
        """원본·ROI·candidate·refined·오버레이를 토글로 비교한다."""

        entry = load_run(app.state.runs_dir, run_id)
        record = find_sample(load_results(entry.run_dir), sample_id)
        latest = ReviewStore(entry.run_dir).latest()
        return render(
            request,
            "mask_viewer.html",
            {
                "run": entry,
                "sample": sample_view(entry, record, latest),
                "layers": mask_layers(entry, record),
            },
        )

    @app.get("/runs/{run_id}/review", response_class=HTMLResponse)
    def review_page(
        request: Request,
        run_id: str,
        sample: str | None = None,
    ) -> HTMLResponse:
        entry = load_run(app.state.runs_dir, run_id)
        records = load_results(entry.run_dir)
        store = ReviewStore(entry.run_dir)
        latest = store.latest()
        queue = [r for r in records if str(r["status"]) == "REVIEW"]
        views = [sample_view(entry, record, latest) for record in queue]
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
            target = f"/runs/{run_id}/review?sample={quote(sample_id)}&saved=1"
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
        review_records = [r for r in records if str(r["status"]) == "REVIEW"]
        review_done = sum(1 for r in review_records if str(r["sample_id"]) in latest)
        escapes = [r for r in records if r["label"] == "defect" and r["status"] == "OK"]
        false_ng = [r for r in records if r["label"] == "good" and r["status"] == "NG"]
        return render(
            request,
            "metrics.html",
            {
                "run": entry,
                "confusion": confusion,
                "ratios": ratios,
                "review_total": len(review_records),
                "review_done": review_done,
                "escapes": escapes,
                "false_ng": false_ng,
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
        """두 run의 지표를 나란히 비교한다(기존 metrics와 같은 정의)."""

        base_view = run_metrics(load_run(app.state.runs_dir, base)) if base else None
        target_view = run_metrics(load_run(app.state.runs_dir, target)) if target else None
        rows: list[dict[str, Any]] = []
        if base_view is not None and target_view is not None:
            for name, base_ratio in base_view["ratios"].items():
                rows.append(
                    {
                        "name": name,
                        "definition": base_ratio.definition,
                        "base": base_ratio,
                        "target": target_view["ratios"][name],
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

    return app


__all__ = ["REASON_LABELS", "STATUS_LABELS", "create_app"]