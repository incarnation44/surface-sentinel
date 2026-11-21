"""실행 자원 상한(동시성·OpenCV 스레드)과 로컬 웹 Origin/CSRF 검증."""

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import urlparse

#: API·시네마 동시 파이프라인 실행 기본 상한(2 OCPU / 12GB VM OOM·스레싱 방어).
DEFAULT_MAX_PIPELINE_CONCURRENCY = 3

#: OpenCV 내부 워커 스레드 기본 상한.
DEFAULT_OPENCV_THREADS = 2

#: 로컬 콘솔 기본 허용 호스트(DNS 리바인딩 방어; ``testserver`` 는 TestClient 기본값).
LOCAL_ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})

_LOCK = threading.Lock()
_SEMAPHORE: threading.BoundedSemaphore | None = None
_SEMAPHORE_LIMIT: int = 0
_ASYNC_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, value))


def max_pipeline_concurrency() -> int:
    """현재 프로세스의 파이프라인 동시 실행 상한을 돌려준다."""

    return _env_int(
        "SURFACE_SENTINEL_MAX_CONCURRENCY",
        DEFAULT_MAX_PIPELINE_CONCURRENCY,
        lo=1,
        hi=16,
    )


def _get_pipeline_semaphore() -> threading.BoundedSemaphore:
    global _SEMAPHORE, _SEMAPHORE_LIMIT  # noqa: PLW0603
    desired = max_pipeline_concurrency()
    with _LOCK:
        if _SEMAPHORE is None or desired != _SEMAPHORE_LIMIT:
            _SEMAPHORE = threading.BoundedSemaphore(desired)
            _SEMAPHORE_LIMIT = desired
        return _SEMAPHORE


@contextmanager
def pipeline_slot() -> Iterator[None]:
    """파이프라인·시네마 동시 실행 슬롯을 하나 점유한다(동기 경로용)."""

    sem = _get_pipeline_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


@asynccontextmanager
async def async_pipeline_slot() -> AsyncIterator[None]:
    """스레드풀 진입 전에 비동기로 동시 실행 슬롯을 대기한다(워커 스레드 고갈 방지)."""

    loop = asyncio.get_running_loop()
    desired = max_pipeline_concurrency()
    entry = _ASYNC_SEMAPHORES.get(loop)
    if entry is None or entry[0] != desired:
        sem = asyncio.Semaphore(desired)
        _ASYNC_SEMAPHORES[loop] = (desired, sem)
    else:
        sem = entry[1]
    async with sem:
        yield


def configure_opencv_threads(threads: int | None = None) -> int:
    """OpenCV 스레드 수를 상한으로 고정해 다중 요청 시 스레싱을 줄인다."""

    import cv2  # noqa: PLC0415

    target = (
        max(1, min(8, int(threads)))
        if threads is not None
        else _env_int("SURFACE_SENTINEL_CV_THREADS", DEFAULT_OPENCV_THREADS, lo=1, hi=8)
    )
    cv2.setNumThreads(target)
    return target


class CsrfOriginError(ValueError):
    """교차 출처(Cross-Site) 또는 비허용 Host POST 요청이 감지됐을 때 발생한다."""


def _allowed_hostnames() -> frozenset[str]:
    extra_raw = os.environ.get("SURFACE_SENTINEL_ALLOWED_HOSTS", "").strip()
    if not extra_raw:
        return LOCAL_ALLOWED_HOSTS
    extras = {item.strip().lower() for item in extra_raw.split(",") if item.strip()}
    return LOCAL_ALLOWED_HOSTS | extras


def _hostname_of_netloc(netloc: str) -> str:
    parsed = urlparse(f"http://{netloc.strip()}")
    return (parsed.hostname or "").lower()


def validate_same_origin(headers: Mapping[str, str], host: str) -> None:
    """로컬 웹 폼/API POST의 Host·Origin·Referer·Sec-Fetch-Site 헤더를 검증한다.

    CLI·curl·TestClient처럼 브라우저 출처 헤더가 없는 로컬 호출은 허용하되,
    로컬 허용 호스트(127.0.0.1/localhost/::1/testserver) 밖의 Host(DNS 리바인딩)와
    교차 출처 브라우저 요청(CSRF)을 모두 차단한다.
    """

    lowered = {k.lower(): v.strip() for k, v in headers.items()}
    sec_fetch_site = lowered.get("sec-fetch-site", "").lower()
    if sec_fetch_site == "cross-site":
        raise CsrfOriginError("외부 출처(cross-site) 요청은 허용되지 않습니다.")

    allowed = _allowed_hostnames()
    expected_host = host.strip().lower()
    host_name = _hostname_of_netloc(expected_host) if expected_host else ""
    if host_name and host_name not in allowed:
        raise CsrfOriginError(f"허용되지 않은 Host입니다: {expected_host!r}")

    origin = lowered.get("origin", "")
    if origin:
        if origin.lower() == "null":
            raise CsrfOriginError("Origin이 null인 요청은 허용되지 않습니다.")
        parsed = urlparse(origin)
        origin_host = (parsed.hostname or "").lower()
        if (
            not parsed.netloc
            or parsed.netloc.lower() != expected_host
            or origin_host not in allowed
        ):
            raise CsrfOriginError(
                f"허용되지 않은 Origin입니다: {origin!r} (기대 호스트: {expected_host})"
            )
        return

    referer = lowered.get("referer", "")
    if referer:
        parsed = urlparse(referer)
        referer_host = (parsed.hostname or "").lower()
        if (
            not parsed.netloc
            or parsed.netloc.lower() != expected_host
            or referer_host not in allowed
        ):
            raise CsrfOriginError(
                f"허용되지 않은 Referer입니다: {referer!r} (기대 호스트: {expected_host})"
            )


__all__ = [
    "DEFAULT_MAX_PIPELINE_CONCURRENCY",
    "DEFAULT_OPENCV_THREADS",
    "LOCAL_ALLOWED_HOSTS",
    "CsrfOriginError",
    "async_pipeline_slot",
    "configure_opencv_threads",
    "max_pipeline_concurrency",
    "pipeline_slot",
    "validate_same_origin",
]
