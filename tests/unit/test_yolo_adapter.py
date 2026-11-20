"""YOLO 판정 어댑터·결합 판정 단위 테스트(합성 detection, ONNX 불필요)."""

from __future__ import annotations

import numpy as np
import pytest

from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.yolo.adapter import (
    YoloDecisionPolicy,
    decide_from_detections,
    to_sample_result,
)
from surface_sentinel.yolo.combine import combine
from surface_sentinel.yolo.inference import Detection

POLICY = YoloDecisionPolicy()


def _det(
    score: float,
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 50.0),
) -> Detection:
    return Detection(bbox_xyxy=bbox, score=score)


# ---------------------------------------------------------------- 판정 정책 경계


def test_no_detections_is_ok() -> None:
    status, reasons, score = decide_from_detections([], POLICY)
    assert status is Status.OK
    assert reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)
    assert score == 0.0


def test_below_review_lo_is_ok() -> None:
    status, reasons, score = decide_from_detections([_det(0.249)], POLICY)
    assert status is Status.OK
    assert reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)
    assert score == pytest.approx(0.249)


def test_review_lo_boundary_is_review() -> None:
    status, reasons, _score = decide_from_detections([_det(0.25)], POLICY)
    assert status is Status.REVIEW
    assert reasons == (ReasonCode.EVIDENCE_IN_REVIEW_BAND,)


def test_inside_review_band_is_review() -> None:
    status, _reasons, _score = decide_from_detections([_det(0.499)], POLICY)
    assert status is Status.REVIEW


def test_ng_conf_boundary_is_ng() -> None:
    status, reasons, _score = decide_from_detections([_det(0.50)], POLICY)
    assert status is Status.NG
    assert reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)


def test_max_score_wins_among_multiple() -> None:
    dets = [_det(0.2, (0.0, 0.0, 5.0, 5.0)), _det(0.8, (20.0, 20.0, 40.0, 40.0))]
    status, _reasons, score = decide_from_detections(dets, POLICY)
    assert status is Status.NG
    assert score == pytest.approx(0.8)


def test_custom_policy_thresholds() -> None:
    policy = YoloDecisionPolicy(ng_conf=0.9, review_lo=0.5)
    assert decide_from_detections([_det(0.6)], policy)[0] is Status.REVIEW
    assert decide_from_detections([_det(0.9)], policy)[0] is Status.NG


# ---------------------------------------------------------------- 결합 판정


@pytest.mark.parametrize("opencv", list(Status))
@pytest.mark.parametrize("yolo", list(Status))
def test_combine_never_returns_ok_when_any_side_flags(opencv: Status, yolo: Status) -> None:
    result = combine(opencv, yolo)
    if opencv is Status.OK and yolo is Status.OK:
        assert result is Status.OK
    else:
        assert result is not Status.OK


def test_combine_priority_ng_then_review_then_error() -> None:
    assert combine(Status.NG, Status.REVIEW) is Status.NG
    assert combine(Status.REVIEW, Status.NG) is Status.NG
    assert combine(Status.REVIEW, Status.OK) is Status.REVIEW
    assert combine(Status.ERROR, Status.REVIEW) is Status.REVIEW
    assert combine(Status.ERROR, Status.OK) is Status.ERROR


# ---------------------------------------------------------------- SampleResult 계약


def test_to_sample_result_fills_contract_fields() -> None:
    mask = np.zeros((100, 200), np.uint8)
    mask[10:30, 40:60] = 1
    dets = [Detection(bbox_xyxy=(40.0, 10.0, 60.0, 30.0), score=0.8, mask=mask)]
    result = to_sample_result(
        dets,
        run_id="run-1",
        sample_id="s1",
        split="tune",
        label="defect",
        image_size=(200, 100),
        config_hash="cfg-abc",
    )
    assert isinstance(result, SampleResult)
    assert result.status is Status.NG
    assert result.score == pytest.approx(0.8)
    assert result.run_id == "run-1"
    assert result.sample_id == "s1"
    assert result.original_id == "s1"
    assert result.split == "tune"
    assert result.label == "defect"
    assert result.schema_version
    assert "cfg-abc" in result.config_hash
    feature = result.components[0]
    assert feature.index == 0
    assert feature.bbox == (40, 10, 20, 20)
    assert feature.area_px == 400
    assert feature.area_ratio == pytest.approx(400 / (200 * 100))
    assert feature.aspect_ratio == pytest.approx(1.0)
    assert 0.0 <= feature.solidity <= 1.0
    assert feature.elongation >= 1.0
    assert feature.mean_contrast_gray == 0.0  # [DESIGN] 대비 계열은 미계산
    assert feature.max_contrast_gray == 0.0
    assert feature.integrated_contrast == 0.0
    assert feature.evidence == pytest.approx(0.8)


def test_to_sample_result_without_detections_is_ok() -> None:
    result = to_sample_result(
        [],
        run_id="run-1",
        sample_id="s0",
        split="tune",
        label="good",
        image_size=(64, 64),
    )
    assert result.status is Status.OK
    assert result.components == ()
    assert result.reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)
    assert result.score == 0.0


def test_to_sample_result_border_touch_flag() -> None:
    dets = [Detection(bbox_xyxy=(0.0, 0.0, 20.0, 20.0), score=0.3)]
    result = to_sample_result(
        dets,
        run_id="run-2",
        sample_id="s2",
        image_size=(100, 100),
    )
    assert result.components[0].touches_roi_border is True
    assert result.components[0].border_distance_px == 0.0
    assert result.status is Status.REVIEW


def test_to_sample_result_reports_area_ratio_denominator() -> None:
    """비율은 분모(이미지 전체 면적)를 함께 갖는다. 분모 0은 거부한다."""
    with pytest.raises(ValueError):
        to_sample_result(
            [_det(0.6)],
            run_id="run-3",
            sample_id="s3",
            image_size=(0, 0),
        )


def test_to_sample_result_is_deterministic() -> None:
    dets = [_det(0.7), _det(0.4, (30.0, 30.0, 60.0, 60.0))]
    kwargs = {
        "run_id": "run-4",
        "sample_id": "s4",
        "image_size": (128, 128),
    }
    first = to_sample_result(dets, **kwargs)  # type: ignore[arg-type]
    second = to_sample_result(dets, **kwargs)  # type: ignore[arg-type]
    assert first.to_dict() == second.to_dict()


def test_adapter_bbox_xywh_contract_matches_evidence_and_stability() -> None:
    """adapter 의 ComponentFeature.bbox 는 (x, y, w, h) 계약으로 IoU·중재와 일치한다."""

    from surface_sentinel.agent.evidence import assess_cross_evidence, bbox_iou
    from surface_sentinel.agent.stability import ArbitrationVerdict, arbitrate_patches
    from surface_sentinel.contracts import ComponentFeature, EngineObservation

    yolo_res = to_sample_result(
        [Detection(bbox_xyxy=(100.0, 200.0, 120.0, 230.0), score=0.85)],
        run_id="run-xywh",
        sample_id="s-xywh",
        image_size=(240, 320),
    )
    yolo_comp = yolo_res.components[0]
    assert yolo_comp.bbox == (100, 200, 20, 30)
    assert bbox_iou(yolo_comp.bbox, (100, 200, 20, 30)) == pytest.approx(1.0)

    opencv_comp = ComponentFeature(
        index=0,
        area_px=600,
        bbox=(100, 200, 20, 30),
        area_ratio=600 / (240 * 320),
        aspect_ratio=20 / 30,
        elongation=1.5,
        solidity=1.0,
        mean_contrast_gray=40.0,
        max_contrast_gray=80.0,
        integrated_contrast=24000.0,
        border_distance_px=100.0,
        touches_roi_border=False,
        evidence=0.85,
    )
    opencv_res = SampleResult(
        run_id="run-xywh",
        sample_id="s-xywh",
        original_id="s-xywh",
        split="tune",
        label="defect",
        status=Status.NG,
        score=0.85,
        reasons=(ReasonCode.DEFECT_EVIDENCE_STRONG,),
        components=(opencv_comp,),
    )
    obs_cv = EngineObservation(engine="opencv", result=opencv_res)
    obs_yolo = EngineObservation(engine="yolo-det", result=yolo_res, task="det")

    assessment = assess_cross_evidence(obs_cv, obs_yolo)
    assert len(assessment.matches) == 1
    assert assessment.matches[0].iou == pytest.approx(1.0)

    image = np.full((320, 240, 3), 120, dtype=np.uint8)
    image[200:230, 100:120, :] = 20
    arbitration = arbitrate_patches(image, obs_cv, obs_yolo)
    assert arbitration.verdict is ArbitrationVerdict.ALIGNED
    assert arbitration.patch_b is not None
    assert arbitration.patch_b.bbox == (100, 200, 20, 30)
    assert arbitration.patch_b.local_contrast >= 80.0
