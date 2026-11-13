"""ONNX 세션 재사용 레지스트리 테스트 (P0-3).

스텁 engine_factory 를 주입해 onnxruntime 없이 캐시 키·재사용·cold/warm 을 검증한다.
모델 파일은 tmp_path 아래 작은 바이트 파일로 만든다(저장소 데이터 미접촉).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from surface_sentinel.yolo.runtime import YoloEngineRegistry


class _StubEngine:
    def __init__(self, model_path: str | Path, *, task: str = "det", **kwargs: Any) -> None:
        self.model_path = Path(model_path)
        self.task = task
        self.calls = 0

    def infer(self, bgr: Any) -> list[Any]:
        self.calls += 1
        return [("det", self.task, self.calls)]


def _model(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _bgr() -> np.ndarray:
    return np.zeros((8, 8, 3), np.uint8)


def test_same_model_and_task_reuses_session(tmp_path: Path) -> None:
    model = _model(tmp_path, "a.onnx", b"model-a")
    created: list[_StubEngine] = []

    def factory(path: str | Path, **kwargs: Any) -> _StubEngine:
        engine = _StubEngine(path, **kwargs)
        created.append(engine)
        return engine

    registry = YoloEngineRegistry(engine_factory=factory)
    first, cold_first = registry.infer(model, task="det", bgr=_bgr())
    second, cold_second = registry.infer(model, task="det", bgr=_bgr())

    assert len(created) == 1
    assert registry.size == 1
    assert registry.cold_starts == 1
    assert cold_first > 0.0 or cold_first == 0.0  # 생성 시간은 측정값(0 이상)
    assert cold_first >= 0.0
    assert cold_second == 0.0
    assert first[0][2] == 1
    assert second[0][2] == 2  # 같은 세션에서 호출 누적


def test_different_task_creates_separate_session(tmp_path: Path) -> None:
    model = _model(tmp_path, "a.onnx", b"model-a")
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(model, task="det", bgr=_bgr())
    registry.infer(model, task="seg", bgr=_bgr())
    assert registry.size == 2
    assert registry.cold_starts == 2


def test_different_model_bytes_creates_separate_session(tmp_path: Path) -> None:
    a = _model(tmp_path, "a.onnx", b"model-a")
    b = _model(tmp_path, "b.onnx", b"model-b")
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(a, task="det", bgr=_bgr())
    registry.infer(b, task="det", bgr=_bgr())
    assert registry.size == 2


def test_same_bytes_different_path_reuses_session(tmp_path: Path) -> None:
    a = _model(tmp_path, "a.onnx", b"same-bytes")
    b = _model(tmp_path, "b.onnx", b"same-bytes")
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(a, task="det", bgr=_bgr())
    _, cold = registry.infer(b, task="det", bgr=_bgr())
    assert registry.size == 1  # 경로는 키가 아니다(바이트 SHA 로 신원 고정)
    assert cold == 0.0


def test_model_bytes_change_invalidates_session(tmp_path: Path) -> None:
    model = _model(tmp_path, "a.onnx", b"v1")
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(model, task="det", bgr=_bgr())
    model.write_bytes(b"v2")
    _, cold = registry.infer(model, task="det", bgr=_bgr())
    assert registry.size == 2
    assert cold >= 0.0
    assert registry.cold_starts == 2
