"""ONNX 세션 재사용 레지스트리 (P0-3, 서빙 동일성).

이미지마다 ``YoloOnnxEngine`` 을 새로 만들면 세션 cold init 비용이 매 이미지
발생한다. 이 레지스트리는 ``(model_sha256, task, onnxruntime_version, providers)``
키로 세션을 캐시해 재사용한다. 모델 신원은 경로가 아니라 **바이트 SHA** 로 고정하므로
같은 파일이 다른 경로에 있어도 같은 세션을 쓴다.

SHA 캐시(감사 5.7): ``infer`` 마다 모델 파일 전체(수 MB)를 다시 읽어 SHA 를 계산하면
이미지 핫패스에 불필요한 파일 I/O 가 남는다. 그래서 경로별로
``(stat 시그니처, sha256)`` 을 캐시하고, stat 시그니처가 캐시와 같으면 파일을 읽지 않고
캐시된 SHA 를 쓴다. 시그니처가 다르면 **전체 바이트를 다시 해시**해 신원을 확정한다
(같은 SHA 면 기존 세션 재사용, 다른 SHA 면 새 세션). stat 은 재해시 **트리거**일 뿐
신원이 아니다 — mtime/size 만으로 세션을 재사용하지 않는다.

잔여 위험(정직하게 기록): stat 시그니처 ``(st_size, st_mtime_ns, st_ctime_ns, st_ino)``
를 의도적으로 보존한 채 바이트만 바꾸는 변경은 자동 감지되지 않는다. 그런 경우
``invalidate()`` 로 캐시를 버려 다음 ``infer`` 에서 강제 재해시해야 한다. 운영 모델
파일은 불변 산출물로 관리하고, 교체 시 ``invalidate()`` 를 호출한다는 전제다.

동시성: 검증 전까지 세션을 여러 스레드에 무제한 공유하지 않는다. 세션별 lock 으로
``infer`` 를 직렬화한다(생성은 전역 lock 으로 1회). 전역 싱글턴이 아니라 API lifespan·
워커 실행 범위·평가 실행 범위에 **주입**해 쓴다. SHA 캐시도 같은 전역 lock 안에서
접근하므로 lifecycle 당 전체 해시는 1회로 수렴한다.

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

#: stat 시그니처(크기, mtime, ctime, inode). 재해시 필요 여부만 판단한다.
_StatSignature = tuple[int, int, int, int]


#: 현재 onnxruntime 버전(미설치면 ""). 세션 키에 포함해 런타임 교체를 구분한다.
def _runtime_version() -> str:
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError:
        return ""
    return str(getattr(onnxruntime, "__version__", ""))


def _stat_signature(path: Path) -> _StatSignature:
    """경로의 stat 시그니처. 값이 같으면 "파일이 그대로"라고 가정한다."""

    st = path.stat()
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


class YoloEngineRegistry:
    """모델 바이트 SHA + task 키로 ONNX 세션을 재사용한다."""

    def __init__(self, *, engine_factory: Callable[..., Any] = YoloOnnxEngine) -> None:
        self._factory = engine_factory
        self._entries: dict[tuple[str, str, str, tuple[str, ...]], tuple[Any, threading.Lock]] = {}
        self._guard = threading.Lock()
        self._cold_starts = 0
        self._sha_cache: dict[Path, tuple[_StatSignature, str]] = {}

    @property
    def size(self) -> int:
        """캐시된 세션 수."""

        return len(self._entries)

    @property
    def cold_starts(self) -> int:
        """새 세션을 만든 횟수(재사용이면 늘지 않는다)."""

        return self._cold_starts

    def invalidate(self, model_path: str | Path) -> None:
        """경로의 SHA 캐시를 버려 다음 ``infer`` 에서 강제 재해시한다.

        용도: 모델 파일을 교체했는데 stat 시그니처(크기·mtime·ctime·inode)가 그대로인
        경우(예: 배포 도구가 타임스탬프를 보존하는 교체, 같은 크기 재작성). 이때 자동
        감지가 불가능하므로 호출자가 모델 교체를 알린 직후 이 메서드를 부른다.
        세션 캐시는 건드리지 않는다 — 재해시 결과가 같은 SHA 면 세션은 그대로 재사용된다.
        """

        path = Path(model_path).resolve()
        with self._guard:
            self._sha_cache.pop(path, None)
        from .inspect import invalidate_model_sha_cache  # noqa: PLC0415 - 순환 import 회피

        invalidate_model_sha_cache(path)

    def _model_sha256(self, path: Path) -> str:
        """경로의 모델 바이트 SHA. **호출자가 ``self._guard`` 를 잡고 있어야 한다.**

        stat 시그니처가 캐시와 같으면 파일을 읽지 않고 캐시된 SHA 를 돌려준다.
        다르면 전체 바이트를 다시 해시해 신원을 확정한다(stat 은 트리거일 뿐).
        """

        signature = _stat_signature(path)
        cached = self._sha_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        digest = file_sha256(path)
        self._sha_cache[path] = (signature, digest)
        return digest

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

        path = Path(model_path).resolve()
        with self._guard:
            key = (
                self._model_sha256(path),
                str(task),
                _runtime_version(),
                tuple(providers or ("CPUExecutionProvider",)),
            )
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
