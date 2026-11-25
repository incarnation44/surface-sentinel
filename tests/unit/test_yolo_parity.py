"""평가 경로와 서빙 경로의 신원·판정 동일성(`yolo` + `records/identity.py`) 짝 일치 단위 테스트.

같은 이미지·모델·정책에서 평가 경로(``evaluate_row``/``evaluate_rows``)와
서빙 경로(``inspect_with_yolo``)가 같은 결합 상태·모델 SHA·정책을 쓰는지 검증한다.
실제 ONNX 없이 스텁 엔진으로 결합 규칙·신원 일치를 확인한다(실측 ONNX 짝 평가는
통합 테스트에서 별도 수행). 합성 이미지만 사용하며 저장소 데이터는 건드리지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import synthetic
from surface_sentinel.config import load_config
from surface_sentinel.contracts import Status
from surface_sentinel.io.manifest import ManifestRow
from surface_sentinel.records.fingerprint import file_sha256, policy_hash
from surface_sentinel.yolo import inspect as yolo_inspect
from surface_sentinel.yolo.evaluate import evaluate_row, evaluate_rows
from surface_sentinel.yolo.inference import Detection
from surface_sentinel.yolo.inspect import (
    COMBINED_TASK_ENV,
    inspect_with_yolo,
    inspection_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "base.toml")


def _det(score: float = 0.8) -> Detection:
    return Detection(bbox_xyxy=(100.0, 40.0, 130.0, 70.0), score=score)


class _StubEngine:
    def __init__(self, model_path: str | Path, *, task: str = "det", **_kw: Any) -> None:
        self.task = task

    def infer(self, bgr: Any, **_kw: Any) -> list[Detection]:
        return [_det()]


def _row(image_path: str, label: str = "good") -> ManifestRow:
    return ManifestRow(
        sample_id="s1",
        original_id="s1",
        group_id="g1",
        image_path=image_path,
        mask_path="",
        label=label,
        source_split="tune",
        width=280,
        height=210,
        channels=3,
        format="png",
        image_sha256="a" * 64,
        mask_sha256="",
        defect_pixels=0,
    )


def test_serving_identity_uses_file_sha_and_policy(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"model-bytes")
    spec = inspection_identity("yolo-det", BASE_CONFIG, onnx_path=model)
    assert spec.model_sha256 == file_sha256(model)
    assert spec.yolo_policy_hash == policy_hash(0.5, 0.25)


def test_eval_report_records_same_identity_as_serving(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"model-bytes")
    synthetic.write_png(tmp_path / "good.png", synthetic.good_image())
    report = evaluate_rows(
        [_row("good.png")],
        engine=_StubEngine(model),
        task="det",
        split="tune",
        onnx=model,
        config=BASE_CONFIG,
        raw_root=tmp_path,
        model_sha256=file_sha256(model),
    )
    spec = inspection_identity("yolo-det", BASE_CONFIG, onnx_path=model)
    assert report.model_sha256 == spec.model_sha256
    assert report.policy == {"ng_conf": 0.5, "review_lo": 0.25}
    assert report.as_dict()["model_sha256"] == spec.model_sha256


def test_combined_status_parity_eval_vs_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    image = synthetic.write_png(tmp_path / "good.png", synthetic.good_image())
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    eval_sample = evaluate_row(
        _row("good.png"),
        engine=_StubEngine(model),
        config=BASE_CONFIG,
        raw_root=tmp_path,
        combine_opencv=True,
    )

    monkeypatch.setattr(yolo_inspect, "YoloOnnxEngine", _StubEngine)
    serving = inspect_with_yolo(image, BASE_CONFIG, engine="combined", onnx_path=model)

    # 같은 결합 규칙(combine: NG > REVIEW > ERROR > OK)을 공유한다.
    assert eval_sample.status is serving.result.status
    assert eval_sample.status is Status.NG  # opencv OK + YOLO NG → NG
    assert eval_sample.opencv_status == "OK"
    # 서빙은 엔진별 원본 관찰도 보존한다(평가의 opencv_status 와 짝).
    assert serving.engine_observations[0].result.status is Status.OK
    assert serving.engine_observations[1].result.status is Status.NG


def test_eval_report_carries_injected_code_identity(tmp_path: Path) -> None:
    """평가 보고서도 진입점 주입 code_sha/code_dirty 를 기록한다(서빙 신원과 짝)."""

    model = tmp_path / "m.onnx"
    model.write_bytes(b"model-bytes")
    synthetic.write_png(tmp_path / "good.png", synthetic.good_image())
    report = evaluate_rows(
        [_row("good.png")],
        engine=_StubEngine(model),
        task="det",
        split="tune",
        onnx=model,
        config=BASE_CONFIG,
        raw_root=tmp_path,
        model_sha256=file_sha256(model),
        code_sha="f" * 40,
        code_dirty=True,
    )
    assert report.code_sha == "f" * 40
    assert report.code_dirty is True
    payload = report.as_dict()
    assert payload["code_sha"] == "f" * 40
    assert payload["code_dirty"] is True
    spec = inspection_identity(
        "yolo-det", BASE_CONFIG, onnx_path=model, code_sha="f" * 40, code_dirty=True
    )
    assert report.model_sha256 == spec.model_sha256
    assert report.code_sha == spec.code_sha
