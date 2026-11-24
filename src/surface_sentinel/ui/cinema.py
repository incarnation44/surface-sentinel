"""3D 레이어 분해 & 영상식 10단계 검사 시네마 시각화 엔진.

읽기 전용 시각화 계층이다.
- 항상 ``split="tune"``·``allow_sealed=False`` 로 실행하며 봉인 분할(test/official_test)에
  접근하지 않는다.
- ``TemporaryDirectory`` 내부에서만 실행하고 즉시 삭제하므로 ``runs/`` 나 SQLite DB 에
  아무 파일도 남기지 않는다.
- 외부 네트워크·CDN 없이 ``data:image/png;base64,...`` 인라인 프레임과 10단계 실측
  텔레메트리(타이밍·ROI·컴포넌트·점수·판정)를 돌려준다.
"""

from __future__ import annotations

import base64
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import REVIEW_BAND_MAX, AppConfig, ConfigError, default_config, load_config
from ..contracts import STAGE_ORDER
from ..io.image_guard import (
    MAX_IMAGE_PIXELS,
    check_decoded_pixel_budget,
    check_image_pixel_budget,
)
from ..pipeline.inspect import inspect_image
from .data import (
    UiInputError,
    discover_runs,
    load_manifest_map,
    load_results,
    safe_child,
)

#: 시네마 업로드 최대 크기 (10 MiB, API 와 동일).
MAX_CINEMA_UPLOAD_BYTES = 10 * 1024 * 1024

#: 시네마 업로드 최대 해상도 픽셀 수 (공통 가드 상한과 동일 — 디코드 메모리 폭주 방어).
MAX_CINEMA_IMAGE_PIXELS = MAX_IMAGE_PIXELS

#: 업로드 청크 크기 (64 KiB).
CINEMA_UPLOAD_CHUNK_BYTES = 64 * 1024

#: 허용 이미지 확장자.
ALLOWED_CINEMA_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

#: 8개 시네마 챕터 메타데이터 (10단계 파이프라인을 7개 실제 영상 프레임 + 최종 판정으로 매핑).
CINEMA_CHAPTERS: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "key": "original",
        "stage_range": "STAGE 01–02",
        "stages": ["validate_input", "load_image"],
        "title": "원본 표면 스캔 및 입력 검증",
        "subtitle": "SHA-256 무결성·크기 검증 후 3채널 BGR + 8비트 그레이스케일 로드",
        "accent": "#38bdf8",
    },
    {
        "index": 1,
        "key": "roi",
        "stage_range": "STAGE 03",
        "stages": ["prepare_roi"],
        "title": "유효 검사 영역(ROI) 분리",
        "subtitle": "배경 암부(dark_floor)를 제거하고 스트립 유효 면적과 경계 링을 확정",
        "accent": "#10b981",
    },
    {
        "index": 2,
        "key": "response",
        "stage_range": "STAGE 04-A",
        "stages": ["preprocess"],
        "title": "배경 보정 대비 응답 (Response)",
        "subtitle": "대형 커널 배경 추정으로 조명 편차를 제거해 미세 밝기/어두움 단차를 증폭",
        "accent": "#f59e0b",
    },
    {
        "index": 3,
        "key": "z_map",
        "stage_range": "STAGE 04-B",
        "stages": ["preprocess"],
        "title": "강건 Z-Score 이상치 지도",
        "subtitle": "중앙값·MAD 기반 표준화로 정상 텍스처를 억제하고 z_threshold 초과 픽셀 강조",
        "accent": "#a855f7",
    },
    {
        "index": 4,
        "key": "candidates",
        "stage_range": "STAGE 05",
        "stages": ["generate_candidates"],
        "title": "결함 후보 픽셀 추출",
        "subtitle": "ROI 내부에서 강건 z 문턱과 최소 대비 조건을 동시에 통과한 초기 후보 마스크",
        "accent": "#06b6d4",
    },
    {
        "index": 5,
        "key": "refined",
        "stage_range": "STAGE 06",
        "stages": ["refine_mask"],
        "title": "모폴로지 정제 & 연결 요소",
        "subtitle": "열림/닫힘·병합 필터로 점 노이즈를 제거하고 min_area_px 이상 결함 후보 확정",
        "accent": "#ec4899",
    },
    {
        "index": 6,
        "key": "overlay",
        "stage_range": "STAGE 07–10",
        "stages": ["analyze_components", "calculate_score", "decide_status", "save_evidence"],
        "title": "특징 계측 · 점수화 · 최종 판정",
        "subtitle": "면적·대비 기반 점수 계산 후 고정 임계값(REVIEW 0.40 / NG 0.50)으로 판정 확정",
        "accent": "#f43f5e",
    },
)

_STATUS_LABELS_KO: dict[str, str] = {
    "OK": "정상",
    "NG": "불량",
    "REVIEW": "보류",
    "ERROR": "오류(판정 아님)",
}


@dataclass(frozen=True)
class CinemaPreset:
    """시네마 화면에서 원클릭 재생 가능한 사전 샘플."""

    preset_id: str
    title: str
    subtitle: str
    badge: str
    source: str
    image_path: Path | None
    label: str = "unknown"
    synthetic_kind: str = ""

    def to_public_dict(self) -> dict[str, str]:
        return {
            "id": self.preset_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "badge": self.badge,
            "source": self.source,
        }


def load_cinema_config(repo_root: Path) -> AppConfig:
    """동결된 ``configs/base.toml`` 을 우선 로드하고 없으면 ``default_config()`` 를 쓴다."""

    base_toml = repo_root / "configs" / "base.toml"
    if base_toml.is_file():
        try:
            return load_config(base_toml)
        except ConfigError:
            pass
    return default_config()


def _encode_data_uri(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("시네마 프레임 PNG 인코딩에 실패했습니다.")
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_roi_frame(bgr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    canvas = (bgr.astype(np.float32) * 0.25).astype(np.uint8)
    inside = roi_mask > 0
    if np.any(inside):
        canvas[inside] = bgr[inside]
        tint = np.zeros_like(canvas)
        tint[:, :] = (129, 185, 16)  # BGR emerald
        canvas[inside] = cv2.addWeighted(canvas[inside], 0.88, tint[inside], 0.12, 0.0)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, -1, (140, 235, 60), 2)
    return canvas


def _render_response_frame(response: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    resp_clean = np.where(np.isfinite(response), np.maximum(response, 0.0), 0.0)
    max_val = float(np.max(resp_clean)) if resp_clean.size else 0.0
    if max_val <= 0.0:
        norm = np.zeros(response.shape, dtype=np.uint8)
    else:
        scale_max = max(max_val, 24.0)
        norm = np.clip(resp_clean.astype(np.float32) * (255.0 / scale_max), 0.0, 255.0).astype(
            np.uint8
        )
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored[roi_mask == 0] = (colored[roi_mask == 0].astype(np.float32) * 0.25).astype(np.uint8)
    return colored


def _render_zmap_frame(z_map: np.ndarray, roi_mask: np.ndarray, z_threshold: float) -> np.ndarray:
    z_clean = np.where(np.isfinite(z_map), np.maximum(z_map, 0.0), 0.0)
    ceil_val = max(float(z_threshold) * 2.0, 12.0)
    norm = np.clip(z_clean * (255.0 / ceil_val), 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    hot = (z_clean >= float(z_threshold)) & (roi_mask > 0)
    colored[hot] = (255, 240, 80)  # BGR neon cyan highlight for threshold exceedance
    colored[roi_mask == 0] = (14, 18, 28)
    return colored


def _render_mask_glow(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    fill_bgr: tuple[int, int, int],
    edge_bgr: tuple[int, int, int],
) -> np.ndarray:
    base = (bgr.astype(np.float32) * 0.40).astype(np.uint8)
    active = mask > 0
    if np.any(active):
        glow = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.5)
        glow_f = (glow.astype(np.float32) / 255.0)[..., None]
        color_arr = np.array(fill_bgr, dtype=np.float32)[None, None, :]
        blended = np.clip(base.astype(np.float32) + glow_f * color_arr * 0.9, 0.0, 255.0)
        base = blended.astype(np.uint8)
        base[active] = fill_bgr
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(base, contours, -1, edge_bgr, 2)
    return base


def _make_fallback_synthetic(kind: str) -> np.ndarray:
    """외부 fixture 파일이 없는 테스트 환경에서도 동작하는 결정론적 합성 금속 스트립."""

    h, w = 320, 120
    rng = np.random.default_rng(20261121 if kind == "defect" else 20261122)
    base = np.full((h, w), 132.0, dtype=np.float32)
    # 미세 세로 브러시드 메탈 텍스처
    col_bias = rng.normal(0.0, 2.2, size=(1, w)).astype(np.float32)
    noise = rng.normal(0.0, 1.5, size=(h, w)).astype(np.float32)
    gray = np.clip(base + col_bias + noise, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if kind == "defect":
        # r20 기준 확실한 결함(깊은 다크 스크래치 + 크레이터)
        cv2.line(bgr, (44, 108), (74, 198), (22, 22, 22), thickness=5)
        cv2.circle(bgr, (60, 152), 9, (18, 18, 18), thickness=-1)
    return bgr


def _safe_score(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def discover_cinema_presets(
    runs_dir: Path,
    raw_root: Path,
    repo_root: Path,
) -> list[CinemaPreset]:
    """실제 tune run 샘플(존재 시) + 합성 fixture 프리셋을 안전하게 수집한다.

    봉인 분할(test/official_test)은 절대 포함하지 않는다.
    """

    presets: list[CinemaPreset] = []
    seen_ids: set[str] = set()

    # 1) 실제 tune run 들에서 NG / REVIEW / OK 대표 샘플 최대 3건 추출 (봉인 분할 제외)
    entries = discover_runs(runs_dir)
    picked: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if str(entry.payload.get("split", "")) != "tune":
            continue
        try:
            records = load_results(entry.run_dir)
            manifest = load_manifest_map(entry.run_dir)
        except UiInputError:
            continue
        for status_want in ("NG", "REVIEW", "OK"):
            if status_want in picked:
                continue
            candidates = [
                r
                for r in records
                if str(r.get("status", "")) == status_want
                and str(r.get("split", "tune")) == "tune"
            ]
            if not candidates:
                continue
            reverse = status_want != "OK"
            candidates.sort(key=lambda rec: _safe_score(rec.get("score")), reverse=reverse)
            for rec in candidates:
                sid = str(rec.get("sample_id", ""))
                mrow = manifest.get(sid)
                if not mrow or not mrow.get("image_path"):
                    continue
                try:
                    img_path = safe_child(raw_root, mrow["image_path"])
                except UiInputError:
                    continue
                if not img_path.is_file():
                    continue
                try:
                    mw = int(mrow.get("width") or 0)
                    mh = int(mrow.get("height") or 0)
                except (TypeError, ValueError):
                    mw, mh = 0, 0
                if (mw > 0 and mw < 64) or (mh > 0 and mh < 64):
                    continue
                picked[status_want] = {"record": rec, "path": img_path}
                break
        if len(picked) == 3:
            break

    for status_want, label_ko in (
        ("NG", "실제 불량"),
        ("REVIEW", "실제 보류"),
        ("OK", "실제 정상"),
    ):
        info = picked.get(status_want)
        if info is None:
            continue
        rec = info["record"]
        sid = str(rec["sample_id"])
        pid = f"run-{status_want.lower()}-{sid}"
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        score = float(rec.get("score", 0.0))
        presets.append(
            CinemaPreset(
                preset_id=pid,
                title=f"{label_ko} #{sid}",
                subtitle=f"KSDD2 tune · score {score:.3f}",
                badge=status_want,
                source="tune-run",
                image_path=info["path"],
                label=str(rec.get("label") or "unknown"),
            )
        )

    # 2) 저장소 합성 fixture (오프라인/CI/클린 클론에서도 항상 재생 가능)
    synth_dir = repo_root / "tests" / "fixtures" / "synthetic"
    synth_specs = (
        ("synth-dark", "합성 다크 결함", "dark_defect.png", "NG", "defect"),
        ("synth-bright", "합성 브라이트 결함", "bright_defect.png", "NG", "defect"),
        ("synth-good", "합성 정상 표면", "good.png", "OK", "good"),
    )
    for pid, title, filename, badge, label in synth_specs:
        candidate = synth_dir / filename
        if candidate.is_file() and pid not in seen_ids:
            seen_ids.add(pid)
            presets.append(
                CinemaPreset(
                    preset_id=pid,
                    title=title,
                    subtitle=f"합성 fixture · {filename}",
                    badge=badge,
                    source="synthetic",
                    image_path=candidate,
                    label=label,
                )
            )

    # 3) 최후 폴백: 빈 tmp 환경에서도 즉시 재생 가능하도록 메모리 합성 프리셋 제공
    if not presets:
        presets.append(
            CinemaPreset(
                preset_id="builtin-defect",
                title="내장 합성 결함",
                subtitle="오프라인 결정론적 테스트 스트립",
                badge="NG",
                source="builtin",
                image_path=None,
                label="defect",
                synthetic_kind="defect",
            )
        )
        presets.append(
            CinemaPreset(
                preset_id="builtin-good",
                title="내장 합성 정상",
                subtitle="오프라인 정상 금속 표면",
                badge="OK",
                source="builtin",
                image_path=None,
                label="good",
                synthetic_kind="good",
            )
        )

    return presets


def run_cinema_trace(
    image_path: Path,
    *,
    sample_id: str,
    config: AppConfig,
    label: str = "unknown",
    source: str = "upload",
) -> dict[str, Any]:
    """실제 10단계 OpenCV 파이프라인을 임시 디렉터리에서 실행해 시네마 JSON을 생성한다."""

    check_image_pixel_budget(image_path, max_pixels=MAX_CINEMA_IMAGE_PIXELS)

    with tempfile.TemporaryDirectory(prefix="ss-cinema-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        outcome = inspect_image(
            image_path,
            config,
            label=label,
            split="tune",
            output_dir=tmp_path,
            save_stages=False,
            allow_sealed=False,
        )
        result = outcome.result
        debug = outcome.debug

        from ..io.imageio import imread_unicode  # noqa: PLC0415

        if debug is not None and debug.bgr is not None:
            bgr = debug.bgr
        else:
            bgr = imread_unicode(image_path)
            if bgr is None:
                raise ValueError("이미지를 디코드하지 못했습니다.")
        height, width = int(bgr.shape[0]), int(bgr.shape[1])
        check_decoded_pixel_budget(width, height, max_pixels=MAX_CINEMA_IMAGE_PIXELS)

        overlay_bgr = bgr
        if result.overlay_path:
            ov_path = Path(result.overlay_path)
            if ov_path.is_file():
                loaded_ov = imread_unicode(ov_path)
                if loaded_ov is not None:
                    overlay_bgr = loaded_ov

        if debug is not None:
            roi_img = _render_roi_frame(bgr, debug.roi_mask)
            resp_img = _render_response_frame(debug.response, debug.roi_mask)
            z_img = _render_zmap_frame(
                debug.z_map, debug.roi_mask, float(config.candidates.z_threshold)
            )
            cand_img = _render_mask_glow(
                bgr,
                debug.candidate_mask,
                fill_bgr=(240, 210, 20),
                edge_bgr=(255, 245, 110),
            )
            ref_img = _render_mask_glow(
                bgr,
                debug.refined_mask,
                fill_bgr=(165, 75, 240),
                edge_bgr=(215, 140, 255),
            )
            candidate_px = int(np.count_nonzero(debug.candidate_mask))
            refined_px = int(np.count_nonzero(debug.refined_mask))
            finite_z = debug.z_map[np.isfinite(debug.z_map) & (debug.roi_mask > 0)]
            max_z = float(np.max(finite_z)) if finite_z.size else 0.0
            finite_resp = debug.response[np.isfinite(debug.response) & (debug.roi_mask > 0)]
            max_resp = float(np.max(finite_resp)) if finite_resp.size else 0.0
        else:
            roi_img = bgr
            resp_img = bgr
            z_img = bgr
            cand_img = bgr
            ref_img = bgr
            candidate_px = 0
            refined_px = 0
            max_z = 0.0
            max_resp = 0.0

        frames = {
            "original": _encode_data_uri(bgr),
            "roi": _encode_data_uri(roi_img),
            "response": _encode_data_uri(resp_img),
            "z_map": _encode_data_uri(z_img),
            "candidates": _encode_data_uri(cand_img),
            "refined": _encode_data_uri(ref_img),
            "overlay": _encode_data_uri(overlay_bgr),
        }

        timings = {
            stage.value: round(float(result.timing_ms.get(stage.value, 0.0)), 3)
            for stage in STAGE_ORDER
        }
        total_ms = round(float(result.timing_ms.get("total", sum(timings.values()))), 3)

        chapters_payload: list[dict[str, Any]] = []
        cumulative_ms = 0.0
        seen_stages: set[str] = set()
        for ch in CINEMA_CHAPTERS:
            stage_ms = 0.0
            for st in ch["stages"]:
                if st not in seen_stages:
                    stage_ms += timings.get(st, 0.0)
                    seen_stages.add(st)
            cumulative_ms += stage_ms
            chapters_payload.append(
                {
                    "index": ch["index"],
                    "key": ch["key"],
                    "stage_range": ch["stage_range"],
                    "stages": list(ch["stages"]),
                    "title": ch["title"],
                    "subtitle": ch["subtitle"],
                    "accent": ch["accent"],
                    "duration_ms": round(stage_ms, 3),
                    "cumulative_ms": round(cumulative_ms, 3),
                }
            )

        components_payload = [
            {
                "index": comp.index,
                "area_px": int(comp.area_px),
                "bbox": [int(v) for v in comp.bbox],
                "mean_contrast_gray": round(float(comp.mean_contrast_gray), 2),
                "max_contrast_gray": round(float(comp.max_contrast_gray), 2),
                "evidence": round(float(comp.evidence), 4),
                "touches_roi_border": bool(comp.touches_roi_border),
            }
            for comp in result.components[:8]
        ]

        roi_info = (
            {
                "mode": result.roi.mode,
                "bbox": [int(v) for v in result.roi.bbox],
                "area_px": int(result.roi.area_px),
                "area_ratio": round(float(result.roi.area_ratio), 4),
                "uncertain": bool(result.roi.uncertain),
            }
            if result.roi is not None
            else None
        )
        quality_info = (
            {
                "mean_gray": round(float(result.quality.mean_gray), 2),
                "blur_variance": round(float(result.quality.blur_variance), 2),
                "saturated_ratio": round(float(result.quality.saturated_ratio), 4),
            }
            if result.quality is not None
            else None
        )

        status_str = result.status.value
        return {
            "sample_id": sample_id,
            "source": source,
            "split": "tune",
            "label": label,
            "status": status_str,
            "status_label": _STATUS_LABELS_KO.get(status_str, status_str),
            "score": round(float(result.score), 4),
            "ng_min": float(config.scoring.ng_min),
            "review_band_max": float(REVIEW_BAND_MAX),
            "z_threshold": float(config.candidates.z_threshold),
            "min_area_px": int(config.candidates.min_area_px),
            "config_hash": result.config_hash,
            "reasons": [r.value for r in result.reasons],
            "error_code": result.error_code.value if result.error_code is not None else "",
            "image_size": {"width": width, "height": height},
            "roi": roi_info,
            "quality": quality_info,
            "telemetry": {
                "candidate_px": candidate_px,
                "refined_px": refined_px,
                "component_count": len(result.components),
                "max_z": round(max_z, 2),
                "max_response": round(max_resp, 2),
                "max_area_px": max((c.area_px for c in result.components), default=0),
                "max_contrast_gray": round(
                    max((float(c.max_contrast_gray) for c in result.components), default=0.0), 2
                ),
            },
            "components": components_payload,
            "timing_ms": timings,
            "total_ms": total_ms,
            "chapters": chapters_payload,
            "frames": frames,
        }


def run_cinema_preset(preset: CinemaPreset, *, config: AppConfig) -> dict[str, Any]:
    """사전 등록된 프리셋에 대해 시네마 10단계 페이로드를 생성한다."""

    if preset.image_path is not None:
        return run_cinema_trace(
            preset.image_path,
            sample_id=preset.preset_id,
            config=config,
            label=preset.label,
            source=preset.source,
        )
    with tempfile.TemporaryDirectory(prefix="ss-cinema-synth-") as tmp_dir:
        synth_path = Path(tmp_dir) / f"{preset.preset_id}.png"
        bgr = _make_fallback_synthetic(preset.synthetic_kind or "defect")
        from ..io.imageio import imwrite_png_atomic  # noqa: PLC0415

        if not imwrite_png_atomic(synth_path, bgr):
            raise OSError(f"합성 이미지 저장 실패: {synth_path}")
        return run_cinema_trace(
            synth_path,
            sample_id=preset.preset_id,
            config=config,
            label=preset.label,
            source=preset.source,
        )


__all__ = [
    "ALLOWED_CINEMA_EXTENSIONS",
    "CINEMA_CHAPTERS",
    "CINEMA_UPLOAD_CHUNK_BYTES",
    "MAX_CINEMA_IMAGE_PIXELS",
    "MAX_CINEMA_UPLOAD_BYTES",
    "CinemaPreset",
    "discover_cinema_presets",
    "load_cinema_config",
    "run_cinema_preset",
    "run_cinema_trace",
]
