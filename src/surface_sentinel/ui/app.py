"""로컬 검토 UI (설계 §5.5, T07).

127.0.0.1 전용. 업로드·설정 편집·삭제 없음. 외부 CDN·외부 요청 없음.
쓰기는 REVIEW 샘플의 검토 결정 append 하나뿐이며 ``OK``/``NG``는 불변이다.
한국어 문구와 영어 상태 배지를 함께 쓰고, 색상만으로 의미를 전달하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..evaluation.metrics import Confusion, image_metrics
from ..pipeline.evidence import sanitize_sample_id
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


def create_app(*, runs_dir: str | Path, raw_root: str | Path) -> FastAPI:
    """run 디렉터리와 원본 데이터 루트를 읽는 UI 앱을 만든다."""

    app = FastAPI(
        title="Surface Sentinel 로컬 UI",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runs_dir = Path(runs_dir)
    app.state.raw_root = Path(raw_root)
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
        """선택 저장된 단계 이미지(response/candidates)의 URL. 없으면 None."""

        safe_id = sanitize_sample_id(sample_id)
        urls: dict[str, str | None] = {}
        for kind in ("response", "candidates"):
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

    return app


__all__ = ["REASON_LABELS", "STATUS_LABELS", "create_app"]