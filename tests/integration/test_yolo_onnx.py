"""실제 ONNX 모델 통합 테스트(TDD: 가중치 존재 시에만 실행).

현재 학습이 진행 중이라 ONNX 파일이 없으면 skip 된다. 기본 경로는 학습
스크립트(yolo/train.py)의 실제 산출물 위치인
``data/derived/yolo/models/{task}/weights/best.onnx`` 다.
루트는 환경변수 ``SURFACE_SENTINEL_YOLO_ONNX_ROOT`` 로 덮어쓸 수 있다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from surface_sentinel.yolo.inference import YoloOnnxEngine, to_input_tensor

pytestmark = [pytest.mark.data, pytest.mark.slow]

DEFAULT_ONNX_ROOT = Path("data/derived/yolo/models")
ONNX_ROOT = Path(os.environ.get("SURFACE_SENTINEL_YOLO_ONNX_ROOT", DEFAULT_ONNX_ROOT))
DET_ONNX = ONNX_ROOT / "det" / "weights" / "best.onnx"
SEG_ONNX = ONNX_ROOT / "seg" / "weights" / "best.onnx"

try:  # optional extra `yolo` 미설치 환경에서도 모듈 수집이 깨지지 않게 한다
    import onnxruntime as _ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:  # pragma: no cover - extra 설치 환경에서는 실행되지 않음
    _ort = None
    ONNXRUNTIME_AVAILABLE = False

SKIP_NO_ORT = pytest.mark.skipif(
    not ONNXRUNTIME_AVAILABLE, reason="optional extra `yolo`(onnxruntime) 미설치"
)
SKIP_NO_DET = pytest.mark.skipif(
    not (ONNXRUNTIME_AVAILABLE and DET_ONNX.is_file()), reason="det ONNX 학습 완료 후 실행"
)
SKIP_NO_SEG = pytest.mark.skipif(
    not (ONNXRUNTIME_AVAILABLE and SEG_ONNX.is_file()), reason="seg ONNX 학습 완료 후 실행"
)


@SKIP_NO_DET
def test_det_onnx_output_shape() -> None:
    engine = YoloOnnxEngine(DET_ONNX, task="det", input_size=416)
    outputs = engine._session.run(  # noqa: SLF001  (shape 실측 전용)
        engine._output_names,  # noqa: SLF001
        {engine._input_name: np.zeros((1, 3, 416, 416), np.float32)},  # noqa: SLF001
    )
    raw = outputs[0]
    assert raw.shape[0] == 1
    # det: 4 bbox + nc(1) → 5. 클래스 수는 shape 에서 유도한다.
    assert raw.shape[1] == 5


@SKIP_NO_SEG
def test_seg_onnx_has_proto_and_masks() -> None:
    engine = YoloOnnxEngine(SEG_ONNX, task="seg", input_size=416)
    bgr = np.zeros((637, 229, 3), np.uint8)
    detections = engine.infer(bgr, conf_threshold=0.0)
    # proto 존재 여부는 엔진 내부 output1 사용으로 보장된다. 여기서는 후처리 계약만 본다.
    for detection in detections:
        assert detection.mask is not None
        assert detection.mask.shape == bgr.shape[:2]
        assert set(np.unique(detection.mask).tolist()) <= {0, 1}
        x1, y1, x2, y2 = detection.bbox_xyxy
        assert 0 <= x1 <= x2 <= bgr.shape[1]
        assert 0 <= y1 <= y2 <= bgr.shape[0]


@SKIP_NO_DET
def test_det_onnx_infer_is_deterministic() -> None:
    engine = YoloOnnxEngine(DET_ONNX, task="det", input_size=416)
    bgr = np.full((320, 320, 3), 128, np.uint8)
    first = engine.infer(bgr)
    second = engine.infer(bgr)
    assert [(d.bbox_xyxy, d.score) for d in first] == [(d.bbox_xyxy, d.score) for d in second]


def test_to_input_tensor_shape_contract() -> None:
    """ONNX 입력 계약(NCHW float32, batch 1)은 학습 완료와 무관하게 고정이다."""

    tensor = to_input_tensor(np.zeros((416, 416, 3), np.uint8))
    assert tensor.shape == (1, 3, 416, 416)
    assert tensor.dtype == np.float32


@SKIP_NO_DET
def test_api_registry_reuses_session_across_requests(tmp_path: Path) -> None:
    """같은 det 모델 2회 요청 → 세션 1회만 생성 (감사 P1-1 완료 기준)."""
    import cv2
    from fastapi.testclient import TestClient

    from surface_sentinel.api import create_inspection_app

    app = create_inspection_app(runs_dir=tmp_path / "runs")
    client = TestClient(app)
    image = np.tile(np.linspace(40, 200, 96, dtype=np.uint8), (96, 1))
    success, buffer = cv2.imencode(".png", image)
    assert success
    payload = buffer.tobytes()
    for _ in range(2):
        response = client.post(
            "/api/inspect",
            files={"file": ("sample.png", payload, "image/png")},
            data={"label": "unknown", "split": "tune"},
            params={"engine": "yolo-det"},
        )
        assert response.status_code == 200
    registry = app.state.yolo_registry
    assert registry is not None
    assert registry.cold_starts == 1
    assert registry.size == 1
