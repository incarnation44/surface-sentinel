"""YOLO vs OpenCV 평가(`yolo/evaluate.py`) 단위 테스트 — 봉인 차단, 분모·Wilson CI, val 분할 검증.

합성 GT 마스크·합성 Detection·stub 엔진만 쓴다. ONNX 가중치와 실제 평가
실행은 필요 없다(학습 완료 후 별도 단계). 검증 대상은 ① 지표와 분모,
② 영역 IoU/recall, ③ 보고서 렌더링, ④ split 거부, ⑤ CLI 파싱, ⑥ 감사 로그다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from surface_sentinel.cli import EXIT_SEALED, EXIT_USAGE, build_parser, main
from surface_sentinel.config import default_config
from surface_sentinel.contracts import Status
from surface_sentinel.io.manifest import ManifestRow
from surface_sentinel.yolo.adapter import YoloDecisionPolicy
from surface_sentinel.yolo.evaluate import (
    BASELINE,
    ENGINE_DET,
    ENGINE_DET_COMBINED,
    ENGINE_SEG,
    EngineSample,
    YoloSealedSplitError,
    YoloSplitError,
    evaluate_rows,
    gt_components_from_mask,
    match_components,
    predicted_components,
    render_report,
    resolve_split_rows,
    summarise_engine,
    write_engine_artifacts,
)
from surface_sentinel.yolo.inference import Detection

SPLIT_VAL = "val"
SPLIT_TUNE = "tune"


# --------------------------------------------------------------- 합성 도우미


def _row(
    sample_id: str,
    *,
    label: str = "good",
    group_id: str | None = None,
    image_path: str | None = None,
    mask_path: str = "",
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        original_id=sample_id,
        group_id=group_id or f"g{sample_id}",
        image_path=image_path or f"train/{sample_id}.png",
        mask_path=mask_path,
        label=label,
        source_split="train",
        width=64,
        height=64,
        channels=3,
        format="PNG",
        image_sha256="a" * 64,
        mask_sha256="b" * 64 if mask_path else "",
        defect_pixels=400 if mask_path else 0,
    )


def _mask(size: int, squares: list[tuple[int, int, int, int]], value: int = 255) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    for y0, y1, x0, x1 in squares:
        mask[y0:y1, x0:x1] = value
    return mask


def _synthetic_samples() -> tuple[EngineSample, ...]:
    """defect 2장 + good 3장. 분모가 서로 다른 지표를 한 번에 검증한다."""

    return (
        EngineSample(
            sample_id="d1",
            label="defect",
            status=Status.OK,
            score=0.10,
            matches=(0.05,),
            gt_components=1,
            gt_pixels=100,
        ),
        EngineSample(
            sample_id="d2",
            label="defect",
            status=Status.NG,
            score=0.90,
            detection_count=1,
            matches=(0.60,),
            gt_components=1,
            gt_pixels=100,
            pred_pixels=80,
            tp_pixels=50,
        ),
        EngineSample(
            sample_id="g1",
            label="good",
            status=Status.NG,
            score=0.70,
            detection_count=2,
        ),
        EngineSample(
            sample_id="g2",
            label="good",
            status=Status.REVIEW,
            score=0.30,
            detection_count=1,
        ),
        EngineSample(
            sample_id="g3",
            label="good",
            status=Status.OK,
            score=0.00,
        ),
    )


def _report(split: str = SPLIT_VAL):
    return summarise_engine(
        ENGINE_DET,
        _synthetic_samples(),
        task="det",
        split=split,
        onnx="data/derived/yolo/models/det/weights/best.onnx",
        config_hash="deadbeef",
        combine=False,
        policy=YoloDecisionPolicy(),
    )


class _StubEngine:
    """ONNX 없이 추론을 흉내내는 대역. 호출 순서·횟수를 기록한다."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.calls = 0

    def infer(self, bgr: np.ndarray, **_kwargs: object) -> list[Detection]:
        self.calls += 1
        return list(self._detections)


# ------------------------------------------------------------- split 계약


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_resolve_split_rows_rejects_sealed_split(split: str) -> None:
    """봉인 분할은 행이 비어 있어도 파일 접근 전에 거부한다."""

    with pytest.raises(YoloSealedSplitError):
        resolve_split_rows([], split)


@pytest.mark.parametrize("split", ["", "train", "TUNE", "all"])
def test_resolve_split_rows_rejects_unknown_split(split: str) -> None:
    with pytest.raises(YoloSplitError):
        resolve_split_rows([], split)


def test_resolve_split_rows_tune_returns_every_row() -> None:
    rows = [_row(str(index)) for index in range(6)]
    assert resolve_split_rows(rows, SPLIT_TUNE) == rows


def test_resolve_split_rows_val_is_group_aware_and_deterministic() -> None:
    rows = [
        _row(f"{group}-{member}", label="defect" if group % 2 else "good", group_id=f"g{group}")
        for group in range(20)
        for member in range(2)
    ]
    first = resolve_split_rows(rows, SPLIT_VAL)
    second = resolve_split_rows(rows, SPLIT_VAL)
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert 0 < len(first) < len(rows)
    # 같은 그룹의 행은 train/val 양쪽에 섞이지 않는다.
    val_groups = {row.group_id for row in first}
    train_groups = {row.group_id for row in rows} - val_groups
    assert val_groups and train_groups
    assert not (val_groups & train_groups)


# ------------------------------------------------------------- 이미지 지표


def test_image_metrics_carry_denominators() -> None:
    report = _report()
    assert report.confusion.evaluated("defect") == 2
    assert report.confusion.evaluated("good") == 3

    escape = report.image["defect_escape_rate"]
    assert escape.numerator == 1
    assert escape.denominator == 2  # 분모 = 결함 이미지 수
    assert escape.value == pytest.approx(0.5)

    false_ng = report.image["good_false_ng_rate"]
    assert false_ng.numerator == 1
    assert false_ng.denominator == 3  # 분모 = 정상 이미지 수
    assert false_ng.value == pytest.approx(1 / 3)

    review = report.image["review_rate"]
    assert review.numerator == 1
    assert review.denominator == 5  # 분모 = 전체 이미지 수
    assert review.value == pytest.approx(0.2)


def test_image_metrics_include_wilson_interval() -> None:
    escape = _report().image["defect_escape_rate"]
    assert escape.ci_low == pytest.approx(0.0945, abs=1e-3)
    assert escape.ci_high == pytest.approx(0.9055, abs=1e-3)


def test_zero_denominator_keeps_none_value() -> None:
    report = summarise_engine(
        ENGINE_SEG,
        (
            EngineSample(sample_id="g1", label="good", status=Status.OK, score=0.0),
        ),
        task="seg",
        split=SPLIT_VAL,
        onnx="m.onnx",
        config_hash="x",
    )
    escape = report.image["defect_escape_rate"]
    assert escape.denominator == 0
    assert escape.value is None
    assert report.measured is True


def test_as_dict_keeps_denominator_and_interval() -> None:
    payload = _report().as_dict()
    escape = payload["image_metrics"]["defect_escape_rate"]
    assert escape["numerator"] == 1
    assert escape["denominator"] == 2
    assert escape["ci95_low"] is not None
    assert payload["counts"]["defect_evaluated"] == 2
    assert payload["counts"]["good_evaluated"] == 3


# ------------------------------------------------------------- 영역 지표


def test_gt_components_from_mask_uses_connected_components() -> None:
    components = gt_components_from_mask(_mask(100, [(10, 30, 10, 30), (60, 80, 60, 80)]))
    assert len(components) == 2
    assert [component.area_px for component in components] == [400, 400]
    assert components[0].bbox == (10, 10, 20, 20)
    assert components[1].bbox == (60, 60, 20, 20)
    assert components[0].mask.shape == (20, 20)
    assert int(np.count_nonzero(components[0].mask)) == 400


def test_seg_prediction_is_converted_to_cached_component() -> None:
    prediction_mask = _mask(100, [(10, 30, 10, 30)], value=1)
    detections = [
        Detection(bbox_xyxy=(10.0, 10.0, 30.0, 30.0), score=0.9, mask=prediction_mask)
    ]
    components = predicted_components(detections, image_h=100, image_w=100)
    assert len(components) == 1
    assert components[0].area_px == 400
    assert components[0].bbox == (10, 10, 20, 20)


def test_det_prediction_without_mask_uses_bbox_area() -> None:
    detections = [Detection(bbox_xyxy=(10.0, 10.0, 30.0, 40.0), score=0.5)]
    components = predicted_components(detections, image_h=100, image_w=100)
    assert components[0].bbox == (10, 10, 20, 30)
    assert components[0].area_px == 20 * 30


def test_degenerate_prediction_is_dropped() -> None:
    detections = [Detection(bbox_xyxy=(10.0, 10.0, 10.0, 40.0), score=0.5)]
    assert predicted_components(detections, image_h=100, image_w=100) == ()


def test_match_components_reports_best_iou_per_gt() -> None:
    gt = gt_components_from_mask(_mask(100, [(10, 30, 10, 30), (60, 80, 60, 80)]))
    prediction_mask = _mask(100, [(10, 30, 10, 30)], value=1)
    predictions = predicted_components(
        [Detection(bbox_xyxy=(10.0, 10.0, 30.0, 30.0), score=0.9, mask=prediction_mask)],
        image_h=100,
        image_w=100,
    )
    ious = match_components(gt, predictions)
    assert ious == pytest.approx((1.0, 0.0))


def test_localization_recall_uses_gt_denominator() -> None:
    report = _report()
    assert report.recall_010.numerator == 1
    assert report.recall_010.denominator == 2
    assert report.recall_010.value == pytest.approx(0.5)
    assert report.recall_050.numerator == 1
    assert report.recall_050.denominator == 2

    pixels = report.localization["pixel_precision"]
    assert pixels.numerator == 50
    assert pixels.denominator == 80
    recall = report.localization["pixel_recall"]
    assert recall.numerator == 50
    assert recall.denominator == 200


# ------------------------------------------------------------- 보고서 렌더링


def test_report_renders_baseline_with_denominator_and_tag() -> None:
    text = render_report([_report()], split=SPLIT_VAL, planned=(ENGINE_DET, ENGINE_SEG))
    assert "| 엔진 |" in text
    assert "escape(/214)" in text
    assert "false-NG(/1787)" in text
    assert "review(/2001)" in text
    assert "recall@0.10(/242)" in text
    assert "recall@0.50" in text

    assert BASELINE.label in text
    assert "11.68% (25/214)" in text
    assert "5.82% (104/1787)" in text
    assert "38.93% (779/2001)" in text
    assert "59.92% (145/242)" in text
    assert "20.66% (50/242)" in text
    assert "[MEASURED, tune 2001]" in text


def test_report_marks_measured_engine_with_its_own_split() -> None:
    text = render_report([_report()], split=SPLIT_VAL, planned=(ENGINE_DET, ENGINE_SEG))
    assert f"[MEASURED, {SPLIT_VAL}]" in text
    assert "50.00% (1/2)" in text  # escape
    assert "33.33% (1/3)" in text  # false-NG
    assert "20.00% (1/5)" in text  # review


def test_report_marks_planned_engine_as_not_measured() -> None:
    text = render_report([_report()], split=SPLIT_VAL, planned=(ENGINE_DET, ENGINE_SEG))
    seg_line = next(
        line for line in text.splitlines() if ENGINE_SEG in line and line.startswith("|")
    )
    assert "[NOT_MEASURED]" in seg_line
    assert "[MEASURED" not in seg_line
    assert seg_line.count("-") >= 5  # 수치를 추정해 채우지 않는다


def test_report_merges_previous_run_payload_as_measured() -> None:
    """det 실행 후 seg를 실행해도 표에 둘 다 실측으로 남는다."""

    payload = dict(_report().as_dict())
    payload["engine"] = ENGINE_SEG
    payload["split"] = SPLIT_VAL
    text = render_report(
        [_report()],
        split=SPLIT_VAL,
        planned=(ENGINE_DET, ENGINE_SEG),
        previous=[payload],
    )
    seg_line = next(
        line for line in text.splitlines() if ENGINE_SEG in line and line.startswith("|")
    )
    assert f"[MEASURED, {SPLIT_VAL}]" in seg_line
    assert "[NOT_MEASURED]" not in seg_line
    assert "50.00% (1/2)" in seg_line


def test_report_conditions_include_previous_run_engine() -> None:
    """이전 실행 JSON 엔진도 측정 조건에 실측으로 남아야 한다('미실행' 금지)."""

    payload = dict(_report().as_dict())
    payload["engine"] = ENGINE_SEG
    payload["split"] = SPLIT_VAL
    text = render_report(
        [_report()],
        split=SPLIT_VAL,
        planned=(ENGINE_DET, ENGINE_SEG),
        previous=[payload],
    )
    conditions = text.split("## 3. 측정 조건")[1]
    seg_line = next(
        line for line in conditions.splitlines() if line.startswith(f"- {ENGINE_SEG}:")
    )
    assert "미실행" not in seg_line
    assert "[NOT_MEASURED]" not in seg_line
    assert "task=" in seg_line
    assert "config_hash=" in seg_line


def test_report_includes_wilson_interval_and_sealed_notice() -> None:
    text = render_report([_report()], split=SPLIT_VAL, planned=(ENGINE_DET,))
    assert "0.095" in text  # Wilson 95% 하한 (1/2)
    assert "0.905" in text  # Wilson 95% 상한 (1/2)
    assert "Wilson 95%" in text
    assert "SEALED" in text


# ------------------------------------------------------------- 평가 루프·산출물


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def test_evaluate_rows_uses_one_inference_per_image(tmp_path: Path) -> None:
    _write_png(tmp_path / "train" / "1.png", np.zeros((64, 64, 3), np.uint8))
    _write_png(tmp_path / "train" / "1_GT.png", _mask(64, [(10, 30, 10, 30)]))
    _write_png(tmp_path / "train" / "2.png", np.zeros((64, 64, 3), np.uint8))

    rows = [
        _row("1", label="defect", mask_path="train/1_GT.png"),
        _row("2"),
    ]
    stub = _StubEngine([Detection(bbox_xyxy=(10.0, 10.0, 30.0, 30.0), score=0.9)])
    report = evaluate_rows(
        rows,
        engine=stub,
        task="det",
        split=SPLIT_VAL,
        onnx="m.onnx",
        config=default_config(),
        raw_root=tmp_path,
        policy=YoloDecisionPolicy(),
    )

    assert stub.calls == 2
    assert report.engine == ENGINE_DET
    assert report.split == SPLIT_VAL
    assert report.config_hash == default_config().config_hash
    assert report.confusion.count("defect", "NG") == 1
    assert report.confusion.count("good", "NG") == 1
    assert report.recall_050.numerator == 1
    assert report.recall_050.denominator == 1


def test_evaluate_rows_without_detections_is_ok(tmp_path: Path) -> None:
    _write_png(tmp_path / "train" / "1.png", np.zeros((64, 64, 3), np.uint8))
    report = evaluate_rows(
        [_row("1")],
        engine=_StubEngine([]),
        task="seg",
        split=SPLIT_VAL,
        onnx="m.onnx",
        config=default_config(),
        raw_root=tmp_path,
    )
    assert report.engine == ENGINE_SEG
    assert report.confusion.count("good", "OK") == 1
    assert report.samples[0].detection_count == 0


def test_write_engine_artifacts_writes_json_and_audit_jsonl(tmp_path: Path) -> None:
    report = _report()
    json_path, jsonl_path = write_engine_artifacts(report, tmp_path / "eval")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["engine"] == ENGINE_DET
    assert payload["split"] == SPLIT_VAL
    assert payload["image_metrics"]["defect_escape_rate"]["denominator"] == 2

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5  # 이미지 1장 = 1행
    first = json.loads(lines[0])
    assert first["sample_id"] == "d1"
    assert first["label"] == "defect"
    assert first["status"] == "OK"
    assert first["detections"] == 0
    assert first["matches"] == [0.05]


def test_audit_jsonl_is_deterministic(tmp_path: Path) -> None:
    report = _report()
    first = write_engine_artifacts(report, tmp_path / "a")
    second = write_engine_artifacts(report, tmp_path / "b")
    assert first[0].read_text(encoding="utf-8") == second[0].read_text(encoding="utf-8")
    assert first[1].read_text(encoding="utf-8") == second[1].read_text(encoding="utf-8")


def test_engine_artifact_names_are_filesystem_safe(tmp_path: Path) -> None:
    report = summarise_engine(
        ENGINE_DET_COMBINED,
        _synthetic_samples(),
        task="det",
        split=SPLIT_TUNE,
        onnx="m.onnx",
        config_hash="deadbeef",
        combine=True,
    )
    json_path, jsonl_path = write_engine_artifacts(report, tmp_path)
    assert report.engine == ENGINE_DET_COMBINED
    assert json_path.name == "yolo-det_opencv-tune.json"
    assert jsonl_path.name == "yolo-det_opencv-tune.jsonl"
    assert "+" not in json_path.name


# ------------------------------------------------------------- CLI 계약


def test_cli_parses_yolo_evaluate_defaults() -> None:
    args = build_parser().parse_args(
        ["yolo", "evaluate", "--task", "seg", "--onnx", "m.onnx"]
    )
    assert args.command == "yolo"
    assert args.yolo_command == "evaluate"
    assert args.task == "seg"
    assert Path(args.onnx) == Path("m.onnx")
    assert args.split == "val"
    assert args.combine is False
    assert Path(args.config) == Path("configs/candidates/adopted/t05_adopted_r20.toml")


def test_cli_parses_yolo_evaluate_options() -> None:
    args = build_parser().parse_args(
        [
            "yolo",
            "evaluate",
            "--task",
            "det",
            "--onnx",
            "det.onnx",
            "--split",
            "tune",
            "--out",
            "data/derived/yolo/eval",
            "--combine",
        ]
    )
    assert args.split == "tune"
    assert Path(args.out) == Path("data/derived/yolo/eval")
    assert args.combine is True


def test_cli_rejects_sealed_split_before_reading_files() -> None:
    code = main(["yolo", "evaluate", "--task", "det", "--onnx", "m.onnx", "--split", "test"])
    assert code == EXIT_SEALED


def test_cli_rejects_unknown_split() -> None:
    code = main(["yolo", "evaluate", "--task", "det", "--onnx", "m.onnx", "--split", "train"])
    assert code == EXIT_USAGE
