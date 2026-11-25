"""ONNX 세션 재사용 레지스트리(`yolo/runtime.py`) 단위 테스트 — 바이트 해시 캐시 적중·무효화 검증.

스텁 engine_factory 를 주입해 onnxruntime 없이 캐시 키·재사용·cold/warm 을 검증한다.
모델 파일은 tmp_path 아래 작은 바이트 파일로 만든다(저장소 데이터 미접촉).

감사 5.7(SHA 재계산 제거) 검증: ``surface_sentinel.yolo.runtime.file_sha256`` 를 카운터로
감싸 전체 바이트 해시가 lifecycle 당 1회인지, stat 시그니처 변화·``invalidate()`` 가
재해시를 트리거하는지 확인한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from surface_sentinel.yolo import runtime as runtime_module
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


def _touch(path: Path, *, seconds: float = 10.0) -> None:
    """mtime 을 명시적으로 올린다(Windows NTFS mtime 해상도 이슈 회피)."""

    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1_000_000_000)))


def _count_sha(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """``runtime.file_sha256`` 를 카운터로 감싸 호출된 경로 목록을 돌려준다."""

    calls: list[Path] = []
    real = runtime_module.file_sha256

    def wrapper(path: str | Path) -> str:
        calls.append(Path(path))
        return real(path)

    monkeypatch.setattr(runtime_module, "file_sha256", wrapper)
    return calls


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
    _touch(model)
    _, cold = registry.infer(model, task="det", bgr=_bgr())
    assert registry.size == 2
    assert cold >= 0.0
    assert registry.cold_starts == 2


def test_repeated_infer_hashes_model_bytes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """동일 모델 N회 추론에서 전체 바이트 해시는 lifecycle 당 1회(감사 5.7)."""

    model = _model(tmp_path, "a.onnx", b"model-a")
    calls = _count_sha(monkeypatch)
    registry = YoloEngineRegistry(engine_factory=_StubEngine)

    for _ in range(5):
        registry.infer(model, task="det", bgr=_bgr())

    assert len(calls) == 1
    assert registry.size == 1
    assert registry.cold_starts == 1


def test_same_size_rewrite_with_mtime_bump_invalidates_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """크기가 같은 b"v1"→b"v2" 재작성도 재해시 후 새 세션으로 간다."""

    model = _model(tmp_path, "a.onnx", b"v1")
    calls = _count_sha(monkeypatch)
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    first, _ = registry.infer(model, task="det", bgr=_bgr())

    model.write_bytes(b"v2")  # 크기 동일(2 bytes), 바이트만 다름
    _touch(model)  # stat 시그니처 변화를 보장

    second, cold = registry.infer(model, task="det", bgr=_bgr())

    assert len(calls) == 2  # 바이트가 바뀌었으니 전체 재해시 1회
    assert registry.size == 2
    assert registry.cold_starts == 2
    assert cold >= 0.0
    assert first[0][2] == 1
    assert second[0][2] == 1  # 새 세션이라 호출 카운터 1부터


def test_mtime_change_with_same_bytes_rehashes_but_reuses_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stat 변화는 재해시 트리거일 뿐 — 같은 SHA 면 기존 세션을 유지한다."""

    model = _model(tmp_path, "a.onnx", b"same-bytes")
    calls = _count_sha(monkeypatch)
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(model, task="det", bgr=_bgr())

    _touch(model)  # 바이트는 그대로, stat 만 변경

    _, cold = registry.infer(model, task="det", bgr=_bgr())

    assert len(calls) == 2  # 시그니처가 다르면 반드시 재해시
    assert registry.size == 1  # 신원은 바이트 SHA → 세션 그대로
    assert registry.cold_starts == 1
    assert cold == 0.0


def test_invalidate_forces_rehash_and_keeps_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``invalidate()`` 후 다음 infer 는 재해시하되 같은 SHA 면 세션을 재사용한다."""

    model = _model(tmp_path, "a.onnx", b"model-a")
    calls = _count_sha(monkeypatch)
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(model, task="det", bgr=_bgr())
    registry.infer(model, task="det", bgr=_bgr())
    assert len(calls) == 1

    registry.invalidate(model)
    _, cold = registry.infer(model, task="det", bgr=_bgr())

    assert len(calls) == 2  # 무효화 후 강제 재해시
    assert registry.size == 1
    assert registry.cold_starts == 1
    assert cold == 0.0


def test_invalidate_only_affects_given_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """다른 경로의 SHA 캐시는 유지된다."""

    a = _model(tmp_path, "a.onnx", b"model-a")
    b = _model(tmp_path, "b.onnx", b"model-b")
    calls = _count_sha(monkeypatch)
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(a, task="det", bgr=_bgr())
    registry.infer(b, task="det", bgr=_bgr())
    assert len(calls) == 2

    registry.invalidate(a)
    registry.infer(b, task="det", bgr=_bgr())  # b 는 캐시 유지 → 해시 없음

    assert len(calls) == 2


def test_invalidate_resolves_paths_and_syncs_inspect_sha_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """상대 경로 무효화도 resolve 키로 동작하고 inspect SHA 캐시와 연동된다."""

    from surface_sentinel.yolo import inspect as yolo_inspect

    model = _model(tmp_path, "a.onnx", b"model-a")
    calls = _count_sha(monkeypatch)
    registry = YoloEngineRegistry(engine_factory=_StubEngine)
    registry.infer(model, task="det", bgr=_bgr())
    assert len(calls) == 1

    # inspect 진입점 캐시도 채운 뒤, 상대 경로로 무효화한다.
    yolo_inspect._cached_model_sha(model)
    resolved = str(model.resolve())
    assert any(key[0] == resolved for key in yolo_inspect._MODEL_SHA_CACHE)

    # CI처럼 저장소와 tmp_path가 다른 드라이브에 있어도 relpath가 성립하도록 cwd를 옮긴다.
    monkeypatch.chdir(tmp_path)
    relative = Path(os.path.relpath(model, Path.cwd()))
    registry.invalidate(relative)

    # inspect 캐시가 함께 비워졌는지(연동), 재해시가 트리거되는지(resolve 키).
    assert not any(key[0] == resolved for key in yolo_inspect._MODEL_SHA_CACHE)
    registry.infer(model, task="det", bgr=_bgr())
    assert len(calls) == 2