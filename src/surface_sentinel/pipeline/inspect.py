"""고정 순서 오케스트레이터.

단계 실행, 타이밍, 오류 변환, 결과 스키마 검증을 한곳에서 처리한다.
제품 판정은 항상 ``decide_status``가 소유하며, 오류는 OK/NG로 바뀌지 않는다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from ..config import AppConfig
from ..contracts import (
    STAGE_ORDER,
    ErrorCode,
    SampleResult,
    Stage,
    Status,
)
from .errors import PipelineError
from .evidence import sanitize_sample_id, save_evidence
from .schema import SchemaError, validate_sample_result
from .stages import (
    analyze_components,
    calculate_score,
    decide_status,
    generate_candidates,
    load_image,
    prepare_roi,
    preprocess,
    refine_mask,
    validate_input,
)
from .types import (
    AnalyzeResult,
    CandidateResult,
    DecisionResult,
    InputMetadata,
    InspectionOutcome,
    LoadedImage,
    PipelineDebug,
    PreprocessResult,
    RefineResult,
    RoiResult,
    ScoreResult,
)

_T = TypeVar("_T")

DEFAULT_OUTPUT_DIR = Path("runs/inspect")


def _error_result(
    *,
    error_code: ErrorCode,
    error_stage: Stage,
    timings: dict[str, float],
    run_id: str,
    sample_id: str,
    split: str,
    label: str,
    config: AppConfig,
    metadata: InputMetadata | None,
    roi: RoiResult | None,
    pre: PreprocessResult | None,
    analyzed: AnalyzeResult | None,
) -> SampleResult:
    return SampleResult(
        run_id=run_id,
        sample_id=sample_id,
        original_id=sample_id,
        split=split,
        label=label,
        status=Status.ERROR,
        score=0.0,
        reasons=(),
        components=analyzed.components if analyzed is not None else (),
        quality=pre.quality if pre is not None else None,
        roi=roi.to_info() if roi is not None else None,
        timing_ms=timings,
        error_code=error_code,
        error_stage=error_stage,
        input_sha256=metadata.image_sha256 if metadata is not None else "",
        mask_sha256=metadata.mask_sha256 if metadata is not None else "",
        overlay_path="",
        config_hash=config.config_hash,
    )


def inspect_image(
    image_path: str | Path,
    config: AppConfig,
    *,
    mask_path: str | Path | None = None,
    label: str = "unknown",
    split: str = "tune",
    output_dir: str | Path | None = None,
    save_stages: bool | None = None,
) -> InspectionOutcome:
    """한 이미지를 고정 10단계로 검사하고 결과와 디버그 산출물을 돌려준다."""

    sample_id = Path(image_path).stem or "sample"
    run_id = f"inspect-{config.config_hash[:8]}-{sanitize_sample_id(sample_id)}"
    directory = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    effective_save_stages = (
        config.artifacts.save_stages if save_stages is None else bool(save_stages)
    )

    timings: dict[str, float] = {stage.value: 0.0 for stage in STAGE_ORDER}
    timings["total"] = 0.0

    metadata: InputMetadata | None = None
    loaded: LoadedImage | None = None
    roi: RoiResult | None = None
    pre: PreprocessResult | None = None
    candidate: CandidateResult | None = None
    refined: RefineResult | None = None
    analyzed: AnalyzeResult | None = None
    scored: ScoreResult | None = None
    decision: DecisionResult | None = None
    debug: PipelineDebug | None = None
    current_stage = Stage.VALIDATE_INPUT

    def execute(stage: Stage, operation: Callable[[], _T]) -> _T:
        nonlocal current_stage
        current_stage = stage
        started = time.perf_counter()
        try:
            return operation()
        finally:
            timings[stage.value] = (time.perf_counter() - started) * 1000.0

    try:
        metadata, decoded = execute(
            Stage.VALIDATE_INPUT,
            lambda: validate_input(
                image_path,
                mask_path=mask_path,
                label=label,
                split=split,
                config=config,
            ),
        )
        loaded = execute(Stage.LOAD_IMAGE, lambda: load_image(decoded, metadata))
        roi = execute(Stage.PREPARE_ROI, lambda: prepare_roi(loaded.gray, config))
        pre = execute(Stage.PREPROCESS, lambda: preprocess(loaded.gray, roi, config))
        candidate = execute(
            Stage.GENERATE_CANDIDATES,
            lambda: generate_candidates(pre.z_map, pre.response, roi.mask, config),
        )
        refined = execute(
            Stage.REFINE_MASK,
            lambda: refine_mask(
                candidate.mask,
                pre.z_map,
                pre.response,
                roi.mask,
                config,
            ),
        )
        analyzed = execute(
            Stage.ANALYZE_COMPONENTS,
            lambda: analyze_components(refined, roi, pre.response, config),
        )
        scored = execute(
            Stage.CALCULATE_SCORE,
            lambda: calculate_score(analyzed.components, config),
        )
        decision = execute(
            Stage.DECIDE_STATUS,
            lambda: decide_status(scored, pre.quality, roi, refined, config),
        )

        result = SampleResult(
            run_id=run_id,
            sample_id=sample_id,
            original_id=sample_id,
            split=split,
            label=label,
            status=decision.status,
            score=scored.score,
            reasons=decision.reasons,
            components=analyzed.components,
            quality=pre.quality,
            roi=roi.to_info(),
            timing_ms=timings,
            input_sha256=metadata.image_sha256,
            mask_sha256=metadata.mask_sha256,
            overlay_path="",
            config_hash=config.config_hash,
        )
        debug = PipelineDebug(
            gray=loaded.gray,
            roi_mask=roi.mask,
            response=pre.response,
            z_map=pre.z_map,
            candidate_mask=candidate.mask,
            refined_mask=refined.combined_mask,
        )

        current_stage = Stage.SAVE_EVIDENCE
        save_started = time.perf_counter()
        try:
            result = save_evidence(
                result,
                bgr=loaded.bgr,
                roi_mask=roi.mask,
                components=refined.components,
                response=pre.response,
                candidate_mask=candidate.mask,
                output_dir=directory,
                timing_ms=timings,
                config=config,
                save_stages=effective_save_stages,
                stage_start=save_started,
            )
        except BaseException:
            timings[Stage.SAVE_EVIDENCE.value] = (
                time.perf_counter() - save_started
            ) * 1000.0
            raise
        timings["total"] = float(
            sum(timings[stage.value] for stage in STAGE_ORDER)
        )
        result = replace(result, timing_ms=timings)
    except PipelineError as exc:
        timings["total"] = float(
            sum(timings[stage.value] for stage in STAGE_ORDER)
        )
        result = _error_result(
            error_code=exc.error_code,
            error_stage=exc.error_stage,
            timings=timings,
            run_id=run_id,
            sample_id=sample_id,
            split=split,
            label=label,
            config=config,
            metadata=metadata,
            roi=roi,
            pre=pre,
            analyzed=analyzed,
        )
        debug = None
    except Exception:
        timings["total"] = float(
            sum(timings[stage.value] for stage in STAGE_ORDER)
        )
        result = _error_result(
            error_code=ErrorCode.E_PROCESS_INTERNAL,
            error_stage=current_stage,
            timings=timings,
            run_id=run_id,
            sample_id=sample_id,
            split=split,
            label=label,
            config=config,
            metadata=metadata,
            roi=roi,
            pre=pre,
            analyzed=analyzed,
        )
        debug = None

    try:
        validate_sample_result(result)
    except SchemaError:
        fallback = _error_result(
            error_code=ErrorCode.E_PROCESS_INTERNAL,
            error_stage=current_stage,
            timings=timings,
            run_id=run_id,
            sample_id=sample_id,
            split=split,
            label=label,
            config=config,
            metadata=metadata,
            roi=roi,
            pre=pre,
            analyzed=analyzed,
        )
        validate_sample_result(fallback)
        result = fallback
        debug = None

    return InspectionOutcome(result=result, debug=debug)


__all__ = ["DEFAULT_OUTPUT_DIR", "inspect_image"]
