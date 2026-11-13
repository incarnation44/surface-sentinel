"""ONNX 세션 재사용 레지스트리 (P0-3, 서빙 동일성).

이미지마다 ``YoloOnnxEngine`` 을 새로 만들면 세션 cold init 비용이 매 이미지
발생한다. 이 레지스트리는 ``(model_sha256, task, onnxruntime_version, providers)``
키로 세션을 캐시해 재사용한다. 모델 신원은 경로가 아니라 **바이트 SHA** 로 고정하므로
같은 파일이 다른 경로에 있어도 같은 세션을 쓴다.

동시성: 검증 전까지 세션을 여러 스레드에 무제한 공유하지 않는다. 세션별 lock 으로
``infer`` 를 직렬화한다(생성은 전역 lock 으로 1회). 전역 싱글턴이 아니라 API lifespan·
워커 실행 범위·평가 실행 범위에 **주입**해 쓴다.

``engine_factory`` 는 테스트에서 스텁을 주입할 수 있다(onnxruntime 불필요).
이 모듈은 subprocess/네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..records.fingerprint import file_sha256
from .inference import YoloOnnxEngine


#: 현재 onnxruntime 버전(미설치면 ""). 세션 키에 포함해 런타임 교체를 구분한다.
def _runtime_version() -> str:
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError:
        return ""
    return str(getattr(onnxruntime, "__version__", ""))


class YoloEngineRegistry:
    """모델 바이트 SHA + task 키로 ONNX 세션을 재사용한다."""

    def __init__(self, *, engine_factory: Callable[..., Any] = YoloOnnxEngine) -> None:
        self._factory = engine_factory
        self._entries: dict[tuple[str, str, str, tuple[str, ...]], tuple[Any, threading.Lock]] = {}
        self._guard = threading.Lock()
        self._cold_starts = 0

    @property
    def size(self) -> int:
        """캐시된 세션 수."""

        return len(self._entries)

    @property
    def cold_starts(self) -> int:
        """새 세션을 만든 횟수(재사용이면 늘지 않는다)."""

        return self._cold_starts

    def infer(
        self,
        model_path: str | Path,
        *,
        task: str,
        bgr: Any,
        providers: list[str] | None = None,
    ) -> tuple[list[Any], float]:
        """세션을 재사용해 추론한다. 반환값은 ``(detections, cold_ms)``.

        ``cold_ms`` 는 이번 호출에서 세션을 새로 만든 시간(ms)이며 재사용이면 0.0.
        """

        key = (
            file_sha256(model_path),
            str(task),
            _runtime_version(),
            tuple(providers or ("CPUExecutionProvider",)),
        )
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                started = time.perf_counter()
                kwargs: dict[str, Any] = {"task": task}
                if providers is not None:
                    kwargs["providers"] = providers
                engine = self._factory(model_path, **kwargs)
                cold_ms = (time.perf_counter() - started) * 1000.0
                entry = (engine, threading.Lock())
                self._entries[key] = entry
                self._cold_starts += 1
            else:
                cold_ms = 0.0
        engine, lock = entry
        with lock:
            detections = engine.infer(bgr)
        return detections, cold_ms


__all__ = ["YoloEngineRegistry"]
