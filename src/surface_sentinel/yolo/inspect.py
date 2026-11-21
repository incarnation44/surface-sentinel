"""YOLO 검사 진입점(det / seg / combined) — 워커·API 공용 계약 (E5b 통합).

``inspect_with_yolo`` 하나로 YOLO 추론 → 어댑터 변환 → 오버레이 저장을 끝낸다.
판정 규칙은 이 모듈이 아니라 ``yolo.adapter``(점수 → 상태)와
``yolo.combine``(opencv + YOLO 결합)이 소유한다. 이 모듈은 조립만 한다.

모델 경로 해석 순서:

1. 인자 ``onnx_path`` (명시적 override)
2. 환경변수 ``SURFACE_SENTINEL_YOLO_ONNX_ROOT`` 아래 ``<task>/weights/best.onnx``
3. 기본 ``data/derived/yolo/models/<task>/weights/best.onnx``

``combined`` 는 task 를 **명시 설정**으로 요구한다(P0-3). 숨은 기본값은 없으며
``SURFACE_SENTINEL_YOLO_COMBINED_TASK=det|seg`` 가 없으면 설정 오류로 멈춘다.
모델 파일이 없으면 ``FileNotFoundError`` 로 멈춘다(조용히 opencv 로 폴백하지 않는다).
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import cv2

from ..config import AppConfig
from ..contracts import EngineObservation, ErrorCode, ReasonCode, SampleResult, Stage, Status
from ..pipeline import InspectionOutcome, inspect_image
from ..pipeline.evidence import sanitize_sample_id
from ..pipeline.stages import SEALED_SPLITS
from ..records.fingerprint import (
    COMBINE_POLICY_VERSION,
    canonical_json_sha256,
    file_sha256,
    policy_hash,
)
from ..records.identity import EngineSpec
from .adapter import YoloDecisionPolicy, to_sample_result
from .combine import combine
from .inference import YoloOnnxEngine
from .overlay import save_overlay
from .runtime import YoloEngineRegistry

#: 모델 루트를 바꾸는 환경변수(아래에 ``<task>/weights/best.onnx`` 가 있어야 한다).
ONNX_ROOT_ENV = "SURFACE_SENTINEL_YOLO_ONNX_ROOT"
#: ``combined`` 가 쓸 모델 task 를 바꾸는 환경변수(``det``/``seg``).
COMBINED_TASK_ENV = "SURFACE_SENTINEL_YOLO_COMBINED_TASK"
#: 학습 산출물 기본 루트(``yolo/train.py`` 의 ``project`` 기본값과 같다).
DEFAULT_ONNX_ROOT = Path("data/derived/yolo/models")
#: 오버레이 파일 접미사. 워커가 ``{safe_id}_*`` 글롭으로 정리하므로 접두사를 지킨다.
OVERLAY_SUFFIX = "_yolo.png"

_TASKS: dict[str, str] = {"yolo-det": "det", "yolo-seg": "seg"}
_ENGINES: tuple[str, ...] = (*_TASKS, "combined")


def engine_task(engine: str) -> str:
    """엔진 이름을 ONNX task(``det``/``seg``)로 바꾼다.

    ``combined`` 는 ``SURFACE_SENTINEL_YOLO_COMBINED_TASK`` 가 **반드시** 있어야
    한다(P0-3). 숨은 det 기본값을 제거했다 — 평가는 seg 로 했는데 서빙은 det 로
    돌아가는 모델 불일치를 막는다. 없으면 설정 오류(``ValueError``).
    """

    name = str(engine).strip()
    if name in _TASKS:
        return _TASKS[name]
    if name == "combined":
        override = os.environ.get(COMBINED_TASK_ENV, "").strip().lower()
        if not override:
            raise ValueError(
                f"combined 엔진은 task 를 명시해야 합니다: {COMBINED_TASK_ENV} 를 "
                "'det' 또는 'seg' 로 설정하세요(숨은 기본값 없음)."
            )
        if override not in ("det", "seg"):
            raise ValueError(
                f"{COMBINED_TASK_ENV} 는 'det' 또는 'seg' 여야 합니다: {override!r}"
            )
        return override
    raise ValueError(f"알 수 없는 YOLO 엔진입니다: {engine!r} (가능: {_ENGINES})")


def resolve_onnx_path(engine: str, onnx_path: str | Path | None = None) -> Path:
    """엔진 이름과 override 로 실제 ONNX 파일 경로를 정한다.

    파일이 없으면 :class:`FileNotFoundError` 를 던진다(어떤 설정을 봤는지
    메시지에 남긴다).
    """

    if onnx_path is not None:
        candidate = Path(onnx_path)
        source = "인자 onnx_path"
    else:
        task = engine_task(engine)
        root_value = os.environ.get(ONNX_ROOT_ENV, "").strip()
        root = Path(root_value) if root_value else DEFAULT_ONNX_ROOT
        candidate = root / task / "weights" / "best.onnx"
        source = f"환경변수 {ONNX_ROOT_ENV}" if root_value else "기본 경로"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"YOLO ONNX 모델이 없습니다({source}): {candidate} — "
            f"`.venv-train` 학습·export 로 생성하거나 {ONNX_ROOT_ENV} 로 루트를 지정하세요."
        )
    return candidate


def _overlay_target(output_dir: Path, sample_id: str) -> Path:
    """오버레이 저장 경로 ``<output_dir>/<safe_id>_yolo.png``."""

    return Path(output_dir) / f"{sanitize_sample_id(sample_id)}{OVERLAY_SUFFIX}"


#: 모델 바이트 SHA 캐시. 키는 (경로, mtime_ns, size) — 파일이 바뀌면 다시 해시한다.
#: 모델 ONNX 는 크므로 이미지마다 재해시하지 않는다(결정론·subprocess 없음).
_MODEL_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def _cached_model_sha(path: Path) -> str:
    """모델 파일 바이트 SHA256(경로·mtime·size 키로 캐시)."""

    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _MODEL_SHA_CACHE.get(key)
    if cached is None:
        cached = file_sha256(path)
        _MODEL_SHA_CACHE[key] = cached
    return cached


def inspection_identity(
    engine: str,
    config: AppConfig,
    *,
    onnx_path: Path | None = None,
    policy: YoloDecisionPolicy | None = None,
    code_sha: str = "",
    code_dirty: bool = False,
) -> EngineSpec:
    """YOLO 계열 엔진의 검사 신원 EngineSpec 을 만든다(P0-2/P0-3).

    평가·API·워커가 **같은** 이 헬퍼로 신원을 계산해 짝 일치시킨다. 모델 신원은
    경로가 아니라 바이트 SHA 로 고정하고, 정책 해시(ng_conf·review_lo)와 결합 규칙
    버전을 포함한다. ``code_sha``/``code_dirty`` 는 진입점(API 생성·워커 시작·
    평가 CLI)에서 1회 계산해 주입한다. 이 헬퍼는 subprocess 를 호출하지 않는다
    (워커 핫패스 금지).
    """

    name = str(engine).strip()
    task = engine_task(name)
    model_path = resolve_onnx_path(name, onnx_path)
    effective = policy if policy is not None else YoloDecisionPolicy()
    combine_hash = (
        canonical_json_sha256({"version": COMBINE_POLICY_VERSION}) if name == "combined" else ""
    )
    return EngineSpec(
        engine=name,
        task=task,
        model_sha256=_cached_model_sha(model_path),
        config_hash=config.config_hash,
        yolo_policy_hash=policy_hash(effective.ng_conf, effective.review_lo),
        combine_policy_hash=combine_hash,
        code_sha=code_sha,
        code_dirty=code_dirty,
        onnx_path=str(model_path),
    )


def _error_result(
    *,
    run_id: str,
    sample_id: str,
    split: str,
    label: str,
    config: AppConfig,
    timings: dict[str, float],
    error_code: ErrorCode,
    error_stage: Stage,
) -> SampleResult:
    """검사 자체가 성립하지 않았을 때의 ``ERROR`` 결과(파이프라인과 같은 모양)."""

    return SampleResult(
        run_id=run_id,
        sample_id=sample_id,
        original_id=sample_id,
        split=split,
        label=label,
        status=Status.ERROR,
        score=0.0,
        reasons=(),
        timing_ms=dict(timings),
        error_code=error_code,
        error_stage=error_stage,
        overlay_path="",
        config_hash=config.config_hash,
    )


def _union_reasons(
    opencv_reasons: tuple[ReasonCode, ...],
    yolo_reasons: tuple[ReasonCode, ...],
) -> tuple[ReasonCode, ...]:
    """양쪽 이유의 합집합(순서 보존·중복 제거)."""

    merged: list[ReasonCode] = []
    for reason in (*opencv_reasons, *yolo_reasons):
        if reason not in merged:
            merged.append(reason)
    return tuple(merged)


def _merge_combined(opencv_result: SampleResult, yolo_result: SampleResult) -> SampleResult:
    """opencv 결과와 YOLO 결과를 하나로 합친다.

    상태는 ``combine`` 우선순위(NG > REVIEW > ERROR > OK)를 **그대로** 따른다.
    한쪽의 NG/REVIEW 가 ERROR 로 묻히면 결함 놓침(escape)이 늘어날 수 있으므로,
    ERROR 는 양쪽 모두 판정(OK/NG/REVIEW)이 없을 때만 결합 상태가 된다.
    결합 상태가 ERROR 일 때만 오류 코드/단계를 보존한다. 나머지 필드(성분·타이밍)는
    YOLO 어댑터 결과를 그대로 두고 상태·이유·점수만 교체한다.
    ``score`` 는 양쪽 최댓값이며 **보정 안 된 순위용**이다(임계값 판정에 쓰지 않는다).
    엔진별 원본 점수·성분·타이밍은 ``InspectionOutcome.engine_observations`` 로
    보존해 소실되지 않는다(P0-3).
    """

    status = combine(opencv_result.status, yolo_result.status)

    error_source: SampleResult | None = None
    if status is Status.ERROR:
        if opencv_result.status is Status.ERROR:
            error_source = opencv_result
        elif yolo_result.status is Status.ERROR:
            error_source = yolo_result

    return replace(
        yolo_result,
        status=status,
        reasons=_union_reasons(opencv_result.reasons, yolo_result.reasons),
        score=max(float(opencv_result.score), float(yolo_result.score)),
        error_code=error_source.error_code if error_source is not None else None,
        error_stage=error_source.error_stage if error_source is not None else None,
    )


def inspect_with_yolo(
    image_path: Path,
    config: AppConfig,
    *,
    engine: str,
    output_dir: Path | None = None,
    split: str = "tune",
    label: str = "unknown",
    onnx_path: Path | None = None,
    policy: YoloDecisionPolicy | None = None,
    registry: YoloEngineRegistry | None = None,
    allow_sealed: bool = False,
) -> InspectionOutcome:
    """한 이미지를 YOLO(det/seg/combined)로 검사한다.

    ``engine`` 은 ``yolo-det``/``yolo-seg``/``combined`` 중 하나다. ``combined``
    는 OpenCV 파이프라인(:func:`~surface_sentinel.pipeline.inspect_image`)과
    YOLO 를 **둘 다** 실행해 :func:`~surface_sentinel.yolo.combine.combine` 로
    상태를 결합한다. ``output_dir`` 가 있으면 ``<safe_id>_yolo.png`` 오버레이를
    남기고 ``SampleResult.overlay_path`` 에 채운다.

    ``registry`` 를 주면 ONNX 세션을 재사용한다(P0-3, cold init 비용 절감).
    없으면 이미지마다 세션을 만든다(하위 호환). ``combined`` 는 엔진별 원본 결과를
    ``InspectionOutcome.engine_observations`` 에 보존한다(판정 불변·별도 증거).

    실패 처리: 모델 파일 부재·onnxruntime 미설치처럼 **환경** 문제는 예외로
    멈추고, 이미지 디코드·추론·오버레이 쓰기 실패는 ``ERROR`` 결과로 돌려준다
    (워커가 재시도 후 기록한다). ``debug`` 는 YOLO 경로에서 쓰지 않는다(None).
    """

    path = Path(image_path)
    sample_id = path.stem or "sample"
    run_id = f"inspect-{config.config_hash[:8]}-{sanitize_sample_id(sample_id)}"
    timings: dict[str, float] = {"load_ms": 0.0, "infer_ms": 0.0, "total": 0.0}

    def fail(error_code: ErrorCode, stage: Stage) -> InspectionOutcome:
        timings["total"] = sum(value for key, value in timings.items() if key != "total")
        return InspectionOutcome(
            result=_error_result(
                run_id=run_id,
                sample_id=sample_id,
                split=split,
                label=label,
                config=config,
                timings=timings,
                error_code=error_code,
                error_stage=stage,
            )
        )

    if split in SEALED_SPLITS and not allow_sealed:
        return fail(ErrorCode.E_SEALED_DENIED, Stage.VALIDATE_INPUT)

    task = engine_task(engine)
    model_path = resolve_onnx_path(engine, onnx_path)

    opencv_result: SampleResult | None = None
    if engine == "combined":
        started = time.perf_counter()
        opencv_result = inspect_image(
            path,
            config,
            label=label,
            split=split,
            output_dir=output_dir,
            allow_sealed=allow_sealed,
        ).result
        timings["opencv_total_ms"] = (time.perf_counter() - started) * 1000.0

    if not path.is_file():
        return fail(ErrorCode.E_INPUT_MISSING, Stage.VALIDATE_INPUT)
    from ..io.image_guard import (  # noqa: PLC0415
        ImageHeaderError,
        ImagePixelLimitError,
        check_decoded_pixel_budget,
        check_image_pixel_budget,
    )

    try:
        check_image_pixel_budget(path)
    except ImagePixelLimitError:
        return fail(ErrorCode.E_INPUT_SIZE, Stage.VALIDATE_INPUT)
    except (ImageHeaderError, OSError):
        return fail(ErrorCode.E_INPUT_DECODE, Stage.LOAD_IMAGE)
    started = time.perf_counter()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    timings["load_ms"] = (time.perf_counter() - started) * 1000.0
    if bgr is None:
        return fail(ErrorCode.E_INPUT_DECODE, Stage.LOAD_IMAGE)
    try:
        check_decoded_pixel_budget(int(bgr.shape[1]), int(bgr.shape[0]))
    except ImagePixelLimitError:
        return fail(ErrorCode.E_INPUT_SIZE, Stage.VALIDATE_INPUT)

    started = time.perf_counter()
    try:
        if registry is not None:
            detections, cold_ms = registry.infer(model_path, task=task, bgr=bgr)
            timings["session_create_ms"] = cold_ms
        else:
            yolo_engine = YoloOnnxEngine(model_path, task=task)
            detections = yolo_engine.infer(bgr)
    except Exception:  # 표본 단위 실패는 ERROR 결과로 남긴다(워커 재시도 대상)
        timings["infer_ms"] = (time.perf_counter() - started) * 1000.0
        return fail(ErrorCode.E_PROCESS_INTERNAL, Stage.ANALYZE_COMPONENTS)
    timings["infer_ms"] = (time.perf_counter() - started) * 1000.0

    timings["total"] = sum(value for key, value in timings.items() if key != "total")
    result = to_sample_result(
        detections,
        run_id=run_id,
        sample_id=sample_id,
        image_size=(int(bgr.shape[1]), int(bgr.shape[0])),
        split=split,
        label=label,
        config_hash=config.config_hash,
        policy=policy,
        timing_ms=timings,
    )

    if output_dir is not None:
        started = time.perf_counter()
        try:
            saved = save_overlay(bgr, detections, _overlay_target(Path(output_dir), sample_id))
        except OSError:
            timings["overlay_ms"] = (time.perf_counter() - started) * 1000.0
            return fail(ErrorCode.E_STORAGE_WRITE, Stage.SAVE_EVIDENCE)
        timings["overlay_ms"] = (time.perf_counter() - started) * 1000.0
        timings["total"] = sum(value for key, value in timings.items() if key != "total")
        result = replace(result, overlay_path=str(saved), timing_ms=dict(timings))

    if opencv_result is not None:
        yolo_observation = result
        merged = _merge_combined(opencv_result, result)
        observations = (
            EngineObservation(engine="opencv", result=opencv_result),
            EngineObservation(engine=f"yolo-{task}", result=yolo_observation, task=task),
        )
        return InspectionOutcome(result=merged, debug=None, engine_observations=observations)
    return InspectionOutcome(result=result, debug=None)


__all__ = [
    "COMBINED_TASK_ENV",
    "DEFAULT_ONNX_ROOT",
    "ONNX_ROOT_ENV",
    "OVERLAY_SUFFIX",
    "engine_task",
    "inspect_with_yolo",
    "inspection_identity",
    "resolve_onnx_path",
]
