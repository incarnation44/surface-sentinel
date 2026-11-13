"""YOLO vs OpenCV 비교 평가(E2c).

학습된 ONNX(det/seg)를 tune-val에서 평가하고, r20 기준선과 **분모를 붙여**
비교한다. 이 모듈이 만드는 것:

* 엔진별 지표 JSON — ``metrics_to_dict`` 형식(모든 비율에 분자·분모·Wilson 95%)
* 이미지별 감사 JSONL — sample_id·label·status·score·탐지 수·GT별 best IoU
* 마크다운 보고서 — 기준선 행 + 측정 엔진 행, 미측정 행은 ``[NOT_MEASURED]``

판정과 매칭은 새로 만들지 않는다. 상태는 ``decide_from_detections``(결합은
``combine``), 영역 IoU는 ``evaluation.measure.component_iou``를 그대로 쓴다.
GT 컴포넌트는 ``evaluation.cache``와 같은 정의(8-연결, 원본 픽셀 마스크)의
``CachedComponent``로 만들어 기준선 분모와 비교 가능하게 한다.

**봉인 분할**: ``test``·``official_test``는 어떤 경로로도 열지 않는다.
허용 분할은 ``val``(tune 내부 group-aware 20%)과 ``tune``(전량 2001)뿐이다.

**측정 태그**: 실제로 돌린 행만 ``[MEASURED, <split>]``이다. 기준선 행은
``[MEASURED, tune 2001]``(이전 단계 실측)이고, 돌리지 않은 엔진은
``[NOT_MEASURED]``로 남긴다. 수치를 추정해 채우지 않는다.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig, load_config
from ..contracts import Status
from ..evaluation.cache import CachedComponent
from ..evaluation.measure import component_intersection, component_iou
from ..evaluation.metrics import (
    IOU_MATCH_FULL,
    IOU_MATCH_RECALL,
    Confusion,
    GtMatch,
    PixelCounts,
    Ratio,
    image_metrics,
    localization_metrics,
    metrics_to_dict,
    ratio,
    recall_at,
)
from ..io.manifest import ManifestError, ManifestRow, read_png
from ..pipeline import inspect_image
from ..records.fingerprint import file_sha256
from .adapter import YoloDecisionPolicy, decide_from_detections
from .combine import combine
from .inference import Detection, YoloOnnxEngine
from .split import DEFAULT_SEED, DEFAULT_VAL_FRACTION, SIDE_VAL, group_train_val_split

TASKS = ("det", "seg")

#: 평가를 허용하는 분할. 이 둘 밖은 전부 거부한다.
ALLOWED_SPLITS = ("val", "tune")
#: 절대 열지 않는 분할(파일 접근 전에 거부).
SEALED_SPLITS = frozenset({"test", "official_test"})

ENGINE_DET = "yolo-det"
ENGINE_SEG = "yolo-seg"
ENGINE_DET_COMBINED = "yolo-det+opencv"
ENGINE_SEG_COMBINED = "yolo-seg+opencv"

ENGINE_LABELS: dict[str, str] = {
    ENGINE_DET: "YOLO det",
    ENGINE_SEG: "YOLO seg",
    ENGINE_DET_COMBINED: "YOLO det + OpenCV(결합)",
    ENGINE_SEG_COMBINED: "YOLO seg + OpenCV(결합)",
}

DEFAULT_CONFIG_PATH = Path("configs/candidates/adopted/t05_adopted_r20.toml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_RAW_ROOT = Path("data/raw/kolektorsdd2")
DEFAULT_OUT_DIR = Path("data/derived/yolo/eval")
DEFAULT_REPORT_DIR = Path("docs/reports")
REPORT_PREFIX = "T07_yolo_eval"

#: 채택 기준선 config(r20)의 해시. ``config_hash`` 계산 결과와 대조해 쓴다.
BASELINE_CONFIG_HASH = "a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098"

#: 자동 생성 보고서의 고정 문구(수치 해석 규칙).
CONCLUSION_NOTE = (
    "결론은 위 표의 수치로만 판단한다. 측정하지 않은 칸은 추정으로 채우지 않는다. "
    "test/official_test는 SEALED이므로 이 비교에 쓰지 않는다."
)


class YoloEvalError(RuntimeError):
    """평가를 계속할 수 없을 때의 오류. ``exit_code``를 갖는다."""

    exit_code = 3


class YoloSplitError(YoloEvalError):
    """허용하지 않는 분할(val/tune 밖)."""

    exit_code = 2


class YoloSealedSplitError(YoloEvalError):
    """봉인 분할 접근 거부 (CLI 종료 코드 6과 맞춘다)."""

    exit_code = 6


# --------------------------------------------------------------- 기준선 상수


@dataclass(frozen=True)
class BaselineReference:
    """r20 기준선 실측값(하드코딩 금지 대상이 아니라 '출처가 붙은 상수').

    출처: T05 채택 단계의 전량 tune 측정(tune 2001 = defect 214 + good 1787,
    GT 컴포넌트 242). 값은 ``[MEASURED, tune 2001]``이며, 이 모듈은 이 값을
    **다시 측정하지 않는다** — 비교표의 기준 행으로만 쓴다.
    """

    label: str
    config_path: str
    config_hash: str
    split_note: str
    escape: Ratio
    false_ng: Ratio
    review: Ratio
    recall_010: Ratio
    recall_050: Ratio


BASELINE = BaselineReference(
    label="OpenCV r20(기준선)",
    config_path=DEFAULT_CONFIG_PATH.as_posix(),
    config_hash=BASELINE_CONFIG_HASH,
    split_note="tune 2001 (defect 214 + good 1787)",
    escape=ratio("OK_on_defect / P", 25, 214),
    false_ng=ratio("NG_on_good / N", 104, 1787),
    review=ratio("(REVIEW_defect+REVIEW_good) / T", 779, 2001),
    recall_010=ratio("matched_GT(IoU>=0.10) / 전체 GT", 145, 242),
    recall_050=ratio("matched_GT(IoU>=0.50) / 전체 GT", 50, 242),
)


# --------------------------------------------------------------- 데이터 구조


@dataclass(frozen=True)
class EngineSample:
    """이미지 한 장의 판정·영역 결과. 감사 JSONL 1행과 같다."""

    sample_id: str
    label: str
    status: Status
    score: float
    detection_count: int = 0
    matches: tuple[float, ...] = ()
    gt_components: int = 0
    gt_pixels: int = 0
    pred_pixels: int = 0
    tp_pixels: int = 0
    opencv_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "status": self.status.value,
            "score": round(float(self.score), 6),
            "detections": int(self.detection_count),
            "matches": [round(float(value), 6) for value in self.matches],
            "gt_components": int(self.gt_components),
            "gt_pixels": int(self.gt_pixels),
            "pred_pixels": int(self.pred_pixels),
            "tp_pixels": int(self.tp_pixels),
            "opencv_status": self.opencv_status,
        }


@dataclass(frozen=True)
class EngineReport:
    """엔진 하나의 지표 묶음. 모든 비율이 분모를 갖는다."""

    engine: str
    task: str
    split: str
    onnx: str
    config_hash: str
    combine: bool
    policy: Mapping[str, float]
    samples: tuple[EngineSample, ...]
    confusion: Confusion
    image: Mapping[str, Ratio]
    localization: Mapping[str, Ratio]
    recall_010: Ratio
    recall_050: Ratio
    pixels: PixelCounts
    notes: tuple[str, ...] = ()
    #: 실제 ONNX 모델 바이트 SHA(P0-3 서빙 동일성 — 서빙 EngineSpec 과 같은 신원).
    model_sha256: str = ""
    #: 진입점 주입 코드 신원(P0-2 — 서빙 EngineSpec 과 짝).
    code_sha: str = ""
    code_dirty: bool = False

    @property
    def measured(self) -> bool:
        """실제로 이미지를 돌린 결과인가(0장이면 미측정으로 본다)."""

        return bool(self.samples)

    @property
    def label(self) -> str:
        return engine_label(self.engine)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "label": self.label,
            "task": self.task,
            "split": self.split,
            "onnx": self.onnx,
            "config_hash": self.config_hash,
            "model_sha256": self.model_sha256,
            "code_sha": self.code_sha,
            "code_dirty": self.code_dirty,
            "combine": self.combine,
            "policy": dict(self.policy),
            "measured": self.measured,
            "tag": _tag(self.measured, self.split),
            "counts": _counts(self),
            "confusion": self.confusion.as_dict(),
            "image_metrics": metrics_to_dict(self.image),
            "localization_metrics": metrics_to_dict(self.localization),
            "recall_at": {
                "iou0.10": self.recall_010.as_dict(),
                "iou0.50": self.recall_050.as_dict(),
            },
            "pixels": {
                "gt_pixels": self.pixels.gt_pixels,
                "pred_pixels": self.pixels.pred_pixels,
                "tp_pixels": self.pixels.tp_pixels,
            },
            "notes": list(self.notes),
        }


def engine_label(engine: str) -> str:
    return ENGINE_LABELS.get(engine, engine)


def _tag(measured: bool, split: str) -> str:
    return f"[MEASURED, {split}]" if measured else "[NOT_MEASURED]"


# --------------------------------------------------------------- 분할 해석


def resolve_split_rows(
    rows: Sequence[ManifestRow],
    split: str,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> list[ManifestRow]:
    """tune 행을 val/tune으로 해석한다. 봉인·미지원 분할은 즉시 거부한다."""

    if split in SEALED_SPLITS:
        raise YoloSealedSplitError(
            f"봉인된 split은 열 수 없습니다: {split}. 허용: {list(ALLOWED_SPLITS)}"
        )
    if split not in ALLOWED_SPLITS:
        raise YoloSplitError(f"허용하지 않는 split입니다: {split!r} (허용: {list(ALLOWED_SPLITS)})")
    if split == "tune":
        return list(rows)
    if not rows:
        raise YoloSplitError("val 분할을 만들 tune 행이 없습니다.")

    result = group_train_val_split(rows, val_fraction=val_fraction, seed=seed)
    return [row for row in rows if result.assignment.get(row.group_id) == SIDE_VAL]


# --------------------------------------------------------------- 컴포넌트 변환


def gt_components_from_mask(mask: np.ndarray) -> tuple[CachedComponent, ...]:
    """GT 마스크 → ``CachedComponent`` 목록(기준선과 같은 GT 정의).

    ``evaluation.cache._gt_components``와 동일하게 8-연결 컴포넌트의 **원본
    픽셀 마스크**를 그대로 담는다. ``yolo.labels.mask_to_components``와 같은
    연결 요소 정의(8-연결, 최소 면적 1)지만, 폴리곤을 다시 그리면 면적이 수
    픽셀 달라져 기준선 분모(tune GT 242)와의 비교 가능성이 깨진다. 그래서
    여기서는 폴리곤 근사 없이 픽셀 마스크를 직접 쓴다.
    """

    if mask.ndim != 2:
        raise YoloEvalError(f"GT 마스크는 2차원이어야 합니다: shape={mask.shape}")
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[CachedComponent] = []
    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        patch = np.ascontiguousarray(
            (labels[y : y + height, x : x + width] == label).astype(np.uint8) * 255
        )
        components.append(
            CachedComponent(
                area_px=area_px,
                bbox=(x, y, width, height),
                mean_contrast_gray=0.0,
                max_contrast_gray=0.0,
                integrated_contrast=0.0,
                border_distance_px=0.0,
                touches_roi_border=False,
                mask=patch,
            )
        )
    return tuple(components)


def predicted_components(
    detections: Sequence[Detection],
    *,
    image_h: int,
    image_w: int,
) -> tuple[CachedComponent, ...]:
    """예측 Detection → ``CachedComponent`` 목록.

    seg 마스크가 있으면 그 픽셀을, 없으면(det) bbox 전체를 영역으로 본다
    (``adapter``의 ``solidity=1.0`` 규칙과 같다). 퇴화 bbox는 버린다.
    """

    if image_h <= 0 or image_w <= 0:
        raise YoloEvalError(f"이미지 크기가 0입니다: image_h={image_h} image_w={image_w}")

    components: list[CachedComponent] = []
    for detection in detections:
        x1, y1, x2, y2 = (int(round(float(value))) for value in detection.bbox_xyxy)
        x1 = min(max(x1, 0), image_w)
        x2 = min(max(x2, 0), image_w)
        y1 = min(max(y1, 0), image_h)
        y2 = min(max(y2, 0), image_h)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        mask = detection.mask
        if mask is None:
            patch = np.full((height, width), 255, np.uint8)
        else:
            array = np.asarray(mask)
            if array.shape[:2] != (image_h, image_w):
                raise YoloEvalError(
                    "예측 마스크 크기가 이미지와 다릅니다: "
                    f"mask={array.shape[:2]} image=({image_h}, {image_w})"
                )
            patch = np.ascontiguousarray((array[y1:y2, x1:x2] > 0).astype(np.uint8) * 255)
        area = int(np.count_nonzero(patch))
        if area <= 0:
            continue
        components.append(
            CachedComponent(
                area_px=area,
                bbox=(x1, y1, width, height),
                mean_contrast_gray=0.0,
                max_contrast_gray=0.0,
                integrated_contrast=0.0,
                border_distance_px=0.0,
                touches_roi_border=False,
                mask=patch,
            )
        )
    return tuple(components)


def match_components(
    gt: Sequence[CachedComponent],
    predicted: Sequence[CachedComponent],
) -> tuple[float, ...]:
    """GT마다 예측과의 **최대** IoU 하나(``evaluation.measure`` 방식 그대로)."""

    return tuple(
        max((component_iou(component, other) for other in predicted), default=0.0)
        for component in gt
    )


def _pixel_counts(
    gt: Sequence[CachedComponent],
    predicted: Sequence[CachedComponent],
) -> PixelCounts:
    return PixelCounts(
        gt_pixels=sum(component.area_px for component in gt),
        pred_pixels=sum(component.area_px for component in predicted),
        tp_pixels=sum(
            component_intersection(other, component)
            for other in predicted
            for component in gt
        ),
    )


# --------------------------------------------------------------- 집계


def summarise_engine(
    engine: str,
    samples: Sequence[EngineSample],
    *,
    task: str,
    split: str,
    onnx: str,
    config_hash: str,
    combine: bool = False,
    policy: YoloDecisionPolicy | None = None,
    notes: Sequence[str] = (),
    model_sha256: str = "",
    code_sha: str = "",
    code_dirty: bool = False,
) -> EngineReport:
    """이미지별 결과를 이미지 지표·영역 지표로 집계한다."""

    if task not in TASKS:
        raise YoloEvalError(f"task는 {list(TASKS)} 중 하나입니다: {task!r}")
    effective = policy if policy is not None else YoloDecisionPolicy()

    pairs = [(sample.label, sample.status.value) for sample in samples]
    confusion = Confusion.from_pairs(pairs)

    matches: list[GtMatch] = [
        GtMatch(iou=iou, image_status=sample.status.value)
        for sample in samples
        for iou in sample.matches
    ]
    pixels = PixelCounts()
    for sample in samples:
        pixels = pixels.merged(
            PixelCounts(
                gt_pixels=sample.gt_pixels,
                pred_pixels=sample.pred_pixels,
                tp_pixels=sample.tp_pixels,
            )
        )

    return EngineReport(
        engine=engine,
        task=task,
        split=split,
        onnx=str(onnx),
        config_hash=config_hash,
        combine=combine,
        policy={"ng_conf": effective.ng_conf, "review_lo": effective.review_lo},
        samples=tuple(samples),
        confusion=confusion,
        image=image_metrics(confusion),
        localization=localization_metrics(matches, pixels),
        recall_010=recall_at(matches, IOU_MATCH_RECALL),
        recall_050=recall_at(matches, IOU_MATCH_FULL),
        pixels=pixels,
        notes=tuple(notes),
        model_sha256=model_sha256,
        code_sha=code_sha,
        code_dirty=code_dirty,
    )


def _counts(report: EngineReport) -> dict[str, Any]:
    status: dict[str, int] = {}
    for sample in report.samples:
        status[sample.status.value] = status.get(sample.status.value, 0) + 1
    return {
        "samples": len(report.samples),
        "defect": report.confusion.total("defect"),
        "good": report.confusion.total("good"),
        "defect_evaluated": report.confusion.evaluated("defect"),
        "good_evaluated": report.confusion.evaluated("good"),
        "gt_components": len(_all_matches(report)),
        "predicted_components": sum(sample.detection_count for sample in report.samples),
        "status": status,
    }


def _all_matches(report: EngineReport) -> list[float]:
    return [iou for sample in report.samples for iou in sample.matches]


# --------------------------------------------------------------- 평가 루프


def _read_bgr(path: Path) -> np.ndarray:
    """이미지 경로 → BGR 3채널 배열(한글 경로 안전: ``read_png`` 재사용)."""

    try:
        image, _digest = read_png(path)
    except (ManifestError, OSError) as exc:
        raise YoloEvalError(f"이미지를 읽을 수 없습니다: {path} ({exc})") from exc
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = int(image.shape[2])
    if channels == 3:
        return image
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise YoloEvalError(f"지원하지 않는 채널 수입니다: {path} shape={image.shape}")


def _read_mask(path: Path) -> np.ndarray:
    try:
        mask, _digest = read_png(path)
    except (ManifestError, OSError) as exc:
        raise YoloEvalError(f"마스크를 읽을 수 없습니다: {path} ({exc})") from exc
    if mask.ndim == 3:
        return (np.any(mask != 0, axis=2)).astype(np.uint8) * 255
    if mask.ndim == 2:
        return (mask > 0).astype(np.uint8) * 255
    raise YoloEvalError(f"지원하지 않는 마스크 차원입니다: {path} shape={mask.shape}")


def evaluate_row(
    row: ManifestRow,
    *,
    engine: Any,
    config: AppConfig,
    raw_root: str | Path,
    policy: YoloDecisionPolicy | None = None,
    combine_opencv: bool = False,
    evidence_dir: str | Path | None = None,
) -> EngineSample:
    """이미지 한 장을 평가한다(추론 1회, 결합이면 OpenCV 판정까지)."""

    root = Path(raw_root)
    image_path = root / row.image_path
    mask_path = root / row.mask_path if row.mask_path else None

    bgr = _read_bgr(image_path)
    height, width = int(bgr.shape[0]), int(bgr.shape[1])
    detections = tuple(engine.infer(bgr))

    yolo_status, _reasons, score = decide_from_detections(detections, policy)
    status = yolo_status
    opencv_status: str | None = None
    if combine_opencv:
        # val은 tune의 부분집합이다. inspect_image는 tune 이름만 받으므로
        # 증거 파일 표기는 tune으로 두고 데이터는 val 행만 쓴다.
        outcome = inspect_image(
            image_path,
            config,
            mask_path=mask_path,
            label=row.label,
            split="tune",
            output_dir=evidence_dir,
        )
        opencv_status = outcome.result.status.value
        status = combine(outcome.result.status, yolo_status)

    gt: tuple[CachedComponent, ...] = ()
    if row.label == "defect" and mask_path is not None:
        mask = _read_mask(mask_path)
        if mask.shape[:2] != (height, width):
            raise YoloEvalError(
                f"GT 마스크 크기가 이미지와 다릅니다: {row.sample_id} "
                f"mask={mask.shape[:2]} image={(height, width)}"
            )
        gt = gt_components_from_mask(mask)

    predicted = predicted_components(detections, image_h=height, image_w=width)
    pixels = _pixel_counts(gt, predicted)
    return EngineSample(
        sample_id=row.sample_id,
        label=row.label,
        status=status,
        score=float(score),
        detection_count=len(detections),
        matches=match_components(gt, predicted),
        gt_components=len(gt),
        gt_pixels=pixels.gt_pixels,
        pred_pixels=pixels.pred_pixels,
        tp_pixels=pixels.tp_pixels,
        opencv_status=opencv_status,
    )


def evaluate_rows(
    rows: Sequence[ManifestRow],
    *,
    engine: Any,
    task: str,
    split: str,
    onnx: str | Path,
    config: AppConfig,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    combine_opencv: bool = False,
    policy: YoloDecisionPolicy | None = None,
    evidence_dir: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    model_sha256: str = "",
    code_sha: str = "",
    code_dirty: bool = False,
) -> EngineReport:
    """행 목록을 순서대로 평가한다(결정론: ``ManifestRow`` 순서, RNG 없음)."""

    if task not in TASKS:
        raise YoloEvalError(f"task는 {list(TASKS)} 중 하나입니다: {task!r}")
    if split in SEALED_SPLITS:
        raise YoloSealedSplitError(f"봉인된 split은 평가할 수 없습니다: {split}")
    if split not in ALLOWED_SPLITS:
        raise YoloSplitError(f"허용하지 않는 split입니다: {split!r}")

    samples: list[EngineSample] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        samples.append(
            evaluate_row(
                row,
                engine=engine,
                config=config,
                raw_root=raw_root,
                policy=policy,
                combine_opencv=combine_opencv,
                evidence_dir=evidence_dir,
            )
        )
        if progress is not None:
            progress(index, total, row.sample_id)

    engine_name = f"yolo-{task}" + ("+opencv" if combine_opencv else "")
    notes: tuple[str, ...] = ()
    if split != "tune":
        notes = (f"{split}은 tune 내부 group-aware 부분집합이다(전량 tune 지표와 직접 비교 금지).",)
    return summarise_engine(
        engine_name,
        samples,
        task=task,
        split=split,
        onnx=str(onnx),
        config_hash=config.config_hash,
        combine=combine_opencv,
        policy=policy,
        notes=notes,
        model_sha256=model_sha256,
        code_sha=code_sha,
        code_dirty=code_dirty,
    )


# --------------------------------------------------------------- 산출물 쓰기


def _atomic_write_text(path: Path, text: str) -> None:
    """같은 디렉터리 임시 파일 → fsync → ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def artifact_stem(report: EngineReport) -> str:
    """파일 이름 조각. ``+``는 파일 시스템에서 피한다."""

    return f"{report.engine.replace('+', '_')}-{report.split}"


def write_engine_artifacts(
    report: EngineReport,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> tuple[Path, Path]:
    """엔진 JSON(지표 전체)과 이미지별 감사 JSONL을 원자적으로 쓴다."""

    directory = Path(out_dir)
    stem = artifact_stem(report)
    json_path = directory / f"{stem}.json"
    jsonl_path = directory / f"{stem}.jsonl"

    _atomic_write_text(
        json_path,
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        json.dumps(sample.as_dict(), ensure_ascii=False, sort_keys=True)
        for sample in report.samples
    ]
    _atomic_write_text(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))
    return json_path, jsonl_path


def _load_previous_payloads(out_dir: str | Path, *, split: str) -> list[dict[str, Any]]:
    """이전 실행이 남긴 엔진 JSON을 읽는다(같은 split만). 실패는 무시한다."""

    payloads: list[dict[str, Any]] = []
    for path in sorted(Path(out_dir).glob("*.json"), key=lambda item: item.name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("split") == split and payload.get("engine"):
            payloads.append(payload)
    return payloads


# --------------------------------------------------------------- 보고서


def _number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:g}"


def _pct_cell(
    value: float | None,
    numerator: float | None,
    denominator: float | None,
) -> str:
    """``11.68% (25/214)`` — 값과 분자/분모를 한 칸에 붙인다."""

    if value is None:
        return f"- (/{_number(denominator)})"
    return f"{value * 100:.2f}% ({_number(numerator)}/{_number(denominator)})"


def _cell_from_ratio(value: Ratio) -> str:
    return _pct_cell(value.value, value.numerator, value.denominator)


def _cell_from_payload(payload: Mapping[str, Any]) -> str:
    return _pct_cell(
        payload.get("value"),
        payload.get("numerator"),
        payload.get("denominator"),
    )


def _detail_from_ratio(name: str, value: Ratio) -> str:
    return f"- {name}: {value.formatted()}"


def _detail_from_payload(name: str, payload: Mapping[str, Any]) -> str:
    value = payload.get("value")
    if value is None:
        return f"- {name}: - (분모 {_number(payload.get('denominator'))})"
    low = payload.get("ci95_low")
    high = payload.get("ci95_high")
    interval = "" if low is None or high is None else f" [{low:.3f}, {high:.3f}]"
    return f"- {name}: {value:.4f}{interval} (분모 {_number(payload.get('denominator'))})"


def _engine_row_label(engine: str) -> str:
    label = engine_label(engine)
    return label if label == engine else f"{label} ({engine})"


def _table_row(engine_cell: str, cells: Sequence[str]) -> str:
    """마크다운 표 한 줄. 셀 수를 세는 대신 넘겨받은 그대로 붙인다."""

    return "| " + " | ".join([engine_cell, *cells]) + " |"


def render_report(
    reports: Sequence[EngineReport],
    *,
    split: str,
    planned: Sequence[str] = (),
    previous: Sequence[Mapping[str, Any]] = (),
    baseline: BaselineReference = BASELINE,
    generated: str | None = None,
    adjustment_note: str = "",
) -> str:
    """§4.3 표(분모 포함) + 지표 상세 + 측정 조건을 마크다운으로 만든다.

    ``planned``에 있으나 결과가 없는 엔진은 ``[NOT_MEASURED]`` 행으로 남긴다
    (수치를 추정해 채우지 않는다). ``previous``는 같은 split의 이전 실행 JSON이며,
    실행을 나눠 돌렸을 때(det → seg) 표에 둘 다 실측으로 남기기 위해 쓴다.
    """

    fresh = {report.engine: report for report in reports}
    older: dict[str, Mapping[str, Any]] = {}
    for payload in previous:
        engine = str(payload.get("engine", ""))
        if engine and engine not in fresh and engine not in older:
            older[engine] = payload

    order: list[str] = []
    for engine in [*planned, *fresh, *older]:
        if engine not in order:
            order.append(engine)

    lines: list[str] = [
        f"# {REPORT_PREFIX} — YOLO vs OpenCV 비교 (split={split})",
        "",
        f"- 생성일: {generated or date.today().isoformat()}",
        f"- 기준선 config: {baseline.config_path} (hash {baseline.config_hash})",
        f"- 기준선 측정 분모: {baseline.split_note}",
        (
            "- 표의 괄호는 그 행의 분자/분모다. 기준선 열 이름의 분모(/214·/1787·"
            "/2001·/242)는 기준선 분모이고, val 측정 행은 자기 분모를 칸에 적는다."
        ),
        "- test/official_test: SEALED — 이 보고서는 열지 않는다.",
        "",
        "## 1. 요약",
        "",
        (
            "| 엔진 | escape(/214) | false-NG(/1787) | review(/2001) "
            "| recall@0.10(/242) | recall@0.50 |"
        ),
        "|---|---|---|---|---|---|",
        _table_row(
            baseline.label,
            [
                _cell_from_ratio(baseline.escape),
                _cell_from_ratio(baseline.false_ng),
                _cell_from_ratio(baseline.review),
                _cell_from_ratio(baseline.recall_010),
                _cell_from_ratio(baseline.recall_050),
            ],
        ),
    ]

    for engine in order:
        report = fresh.get(engine)
        if report is not None:
            lines.append(
                _table_row(
                    f"{_engine_row_label(engine)} {_tag(report.measured, report.split)}",
                    [
                        _cell_from_ratio(report.image["defect_escape_rate"]),
                        _cell_from_ratio(report.image["good_false_ng_rate"]),
                        _cell_from_ratio(report.image["review_rate"]),
                        _cell_from_ratio(report.recall_010),
                        _cell_from_ratio(report.recall_050),
                    ],
                )
            )
            continue
        payload = older.get(engine)
        if payload is not None:
            image = payload.get("image_metrics", {})
            recall = payload.get("recall_at", {})
            lines.append(
                _table_row(
                    f"{_engine_row_label(engine)} {_tag(True, str(payload.get('split')))}",
                    [
                        _cell_from_payload(image.get("defect_escape_rate", {})),
                        _cell_from_payload(image.get("good_false_ng_rate", {})),
                        _cell_from_payload(image.get("review_rate", {})),
                        _cell_from_payload(recall.get("iou0.10", {})),
                        _cell_from_payload(recall.get("iou0.50", {})),
                    ],
                )
            )
            continue
        lines.append(_table_row(f"{_engine_row_label(engine)} [NOT_MEASURED]", ["-"] * 5))

    lines.extend(["", "## 2. 지표 상세 (분모 · Wilson 95%)", ""])

    lines.append(f"### {baseline.label} — [MEASURED, tune 2001]")
    lines.extend(
        _detail_from_ratio(name, value)
        for name, value in (
            ("defect_escape_rate", baseline.escape),
            ("good_false_ng_rate", baseline.false_ng),
            ("review_rate", baseline.review),
            ("recall_iou0.10", baseline.recall_010),
            ("recall_iou0.50", baseline.recall_050),
        )
    )
    lines.append("")

    for engine in order:
        report = fresh.get(engine)
        if report is not None:
            tag = _tag(report.measured, report.split)
            lines.append(f"### {_engine_row_label(engine)} — {tag}")
            lines.extend(
                _detail_from_ratio(name, value)
                for name, value in [
                    *report.image.items(),
                    *report.localization.items(),
                ]
            )
            lines.extend(
                [
                    _detail_from_ratio("recall_iou0.10", report.recall_010),
                    _detail_from_ratio("recall_iou0.50", report.recall_050),
                ]
            )
            lines.append("")
            continue
        payload = older.get(engine)
        if payload is not None:
            tag = _tag(True, str(payload.get("split")))
            lines.append(f"### {_engine_row_label(engine)} — {tag}")
            for name, value in payload.get("image_metrics", {}).items():
                lines.append(_detail_from_payload(name, value))
            for name, value in payload.get("localization_metrics", {}).items():
                lines.append(_detail_from_payload(name, value))
            for name, value in payload.get("recall_at", {}).items():
                lines.append(_detail_from_payload(f"recall_{name}", value))
            lines.append("")

    lines.extend(["## 3. 측정 조건", ""])
    lines.append(f"- 기준선: {baseline.label} / {baseline.config_path}")
    lines.append("- 이 보고서는 자동 생성물이며 수치를 손으로 고치지 않는다.")
    for engine in order:
        report = fresh.get(engine)
        if report is None:
            payload = older.get(engine)
            if payload is None:
                lines.append(f"- {engine}: 미실행 [NOT_MEASURED]")
                continue
            policy = payload.get("policy", {})
            counts = payload.get("counts", {})
            lines.append(
                f"- {engine}: task={payload.get('task')} / split={payload.get('split')} / "
                f"onnx={payload.get('onnx')} / config_hash={payload.get('config_hash')} / "
                f"combine={'예' if payload.get('combine') else '아니오'} / "
                f"정책 ng_conf={float(policy.get('ng_conf', 0.0)):.2f}, "
                f"review_lo={float(policy.get('review_lo', 0.0)):.2f} / "
                f"이미지 {int(counts.get('samples', 0))}장 (이전 실행 JSON)"
            )
            continue
        lines.append(
            f"- {engine}: task={report.task} / split={report.split} / "
            f"onnx={report.onnx} / config_hash={report.config_hash} / "
            f"combine={'예' if report.combine else '아니오'} / "
            f"정책 ng_conf={float(report.policy.get('ng_conf', 0.0)):.2f}, "
            f"review_lo={float(report.policy.get('review_lo', 0.0)):.2f} / "
            f"이미지 {len(report.samples)}장"
        )
    if adjustment_note:
        lines.append(f"- 임계값 조정: {adjustment_note}")
    lines.append("- test/official_test: SEALED — 미사용")

    lines.extend(["", "## 4. 결론", "", f"- {CONCLUSION_NOTE}", ""])
    return "\n".join(lines)


# --------------------------------------------------------------- 실행 진입점


@dataclass(frozen=True)
class EvalRun:
    """한 번의 평가 실행 산출물."""

    report: EngineReport
    rows: int
    json_path: Path
    jsonl_path: Path
    report_path: Path


def _stderr_progress(label: str) -> Callable[[int, int, str], None]:
    def report(done: int, total: int, sample_id: str) -> None:
        if done == total or done % 100 == 0:
            print(f"  {label}: {done}/{total} ({sample_id})", file=sys.stderr, flush=True)

    return report


def run_yolo_evaluation(
    *,
    task: str,
    onnx: str | Path,
    split: str = "val",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    combine_opencv: bool = False,
    report_path: str | Path | None = None,
    policy: YoloDecisionPolicy | None = None,
    engine: Any = None,
    adjustment_note: str = "",
) -> EvalRun:
    """tune CSV → (val|tune) 행 → ONNX 평가 → JSON·JSONL·보고서."""

    if split in SEALED_SPLITS:
        raise YoloSealedSplitError(f"봉인된 split은 열 수 없습니다: {split}")
    if split not in ALLOWED_SPLITS:
        raise YoloSplitError(f"허용하지 않는 split입니다: {split!r} (허용: {list(ALLOWED_SPLITS)})")
    if task not in TASKS:
        raise YoloEvalError(f"task는 {list(TASKS)} 중 하나입니다: {task!r}")

    # 봉인 CSV를 열지 않기 위해 tune만 읽고 val은 그 안에서 만든다.
    from ..evaluation import recalibration  # noqa: PLC0415 - 진입점 1회 계산
    from ..evaluation.runner import load_split_rows

    # 코드 신원은 평가 진입점 1회 계산(서빙 API·워커와 같은 헬퍼·같은 규칙).
    code_sha, code_dirty = recalibration.git_code_identity()

    tune_rows = load_split_rows("tune", splits_dir=splits_dir)
    rows = resolve_split_rows(tune_rows, split)
    if not rows:
        raise YoloEvalError(f"평가할 행이 없습니다: split={split}")

    config = load_config(config_path)
    onnx_path = Path(onnx)
    if not onnx_path.is_file():
        raise YoloEvalError(f"ONNX 모델을 찾을 수 없습니다: {onnx_path}")

    directory = Path(out_dir)
    evidence_dir = directory / f"opencv_evidence_{split}" if combine_opencv else None
    active_engine = engine if engine is not None else YoloOnnxEngine(onnx_path, task=task)

    report = evaluate_rows(
        rows,
        engine=active_engine,
        task=task,
        split=split,
        onnx=onnx_path,
        config=config,
        raw_root=raw_root,
        combine_opencv=combine_opencv,
        policy=policy,
        evidence_dir=evidence_dir,
        progress=_stderr_progress(f"{task}/{split}"),
        model_sha256=file_sha256(onnx_path),
        code_sha=code_sha,
        code_dirty=code_dirty,
    )
    json_path, jsonl_path = write_engine_artifacts(report, directory)

    destination = (
        Path(report_path)
        if report_path is not None
        else DEFAULT_REPORT_DIR / f"{REPORT_PREFIX}_{date.today().isoformat()}.md"
    )
    planned = [ENGINE_DET, ENGINE_SEG]
    if combine_opencv:
        planned = [ENGINE_DET_COMBINED, ENGINE_SEG_COMBINED]
    text = render_report(
        [report],
        split=split,
        planned=tuple(planned),
        previous=_load_previous_payloads(directory, split=split),
        baseline=BASELINE,
        adjustment_note=adjustment_note,
    )
    _atomic_write_text(destination, text)

    return EvalRun(
        report=report,
        rows=len(rows),
        json_path=json_path,
        jsonl_path=jsonl_path,
        report_path=destination,
    )


__all__ = [
    "ALLOWED_SPLITS",
    "BASELINE",
    "BASELINE_CONFIG_HASH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUT_DIR",
    "DEFAULT_RAW_ROOT",
    "DEFAULT_REPORT_DIR",
    "DEFAULT_SPLITS_DIR",
    "ENGINE_DET",
    "ENGINE_DET_COMBINED",
    "ENGINE_LABELS",
    "ENGINE_SEG",
    "ENGINE_SEG_COMBINED",
    "EvalRun",
    "BaselineReference",
    "EngineReport",
    "EngineSample",
    "REPORT_PREFIX",
    "SEALED_SPLITS",
    "TASKS",
    "YoloEvalError",
    "YoloSealedSplitError",
    "YoloSplitError",
    "artifact_stem",
    "engine_label",
    "evaluate_row",
    "evaluate_rows",
    "gt_components_from_mask",
    "match_components",
    "predicted_components",
    "render_report",
    "resolve_split_rows",
    "run_yolo_evaluation",
    "summarise_engine",
    "write_engine_artifacts",
]
