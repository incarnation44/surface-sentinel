"""실행 자원 상한(파이프라인 동시성·OpenCV 스레드)과 로컬 웹 Origin/CSRF·업로드 크기 검증.

과도한 동시 요청·대용량 업로드·교차 출처 요청으로부터 로컬 API와 UI 서버를 보호한다.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
import weakref
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import Any
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
    with _LOCK:
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


class UploadTooLargeError(ValueError):
    """업로드 바이트가 설정된 상한을 초과했을 때 발생한다."""


async def read_upload_bounded(
    upload: Any,
    *,
    max_bytes: int,
    chunk_bytes: int = 64 * 1024,
) -> bytes:
    """업로드를 청크 단위로 ``max_bytes`` 상한까지만 읽고 핸들을 닫는다."""

    close_fn = getattr(upload, "close", None)
    try:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError(f"max_bytes는 0 이상의 정수여야 합니다: {max_bytes!r}")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError(f"chunk_bytes는 1 이상의 정수여야 합니다: {chunk_bytes!r}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLargeError(f"업로드 용량 상한({max_bytes} bytes)을 초과했습니다.")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if callable(close_fn):
            closed = close_fn()
            if inspect.isawaitable(closed):
                await closed


def _allowed_hostnames() -> frozenset[str]:
    extra_raw = os.environ.get("SURFACE_SENTINEL_ALLOWED_HOSTS", "").strip()
    if not extra_raw:
        return LOCAL_ALLOWED_HOSTS
    extras = {item.strip().lower() for item in extra_raw.split(",") if item.strip()}
    return LOCAL_ALLOWED_HOSTS | extras


def _hostname_of_netloc(netloc: str) -> str:
    cleaned = netloc.strip()
    if (
        not cleaned
        or any(ch.isspace() for ch in cleaned)
        or any(ch in cleaned for ch in "@/\\")
    ):
        raise CsrfOriginError(f"유효하지 않은 Host 형식입니다: {netloc!r}")
    try:
        parsed = urlparse(f"http://{cleaned}")
        hostname = (parsed.hostname or "").lower()
        _ = parsed.port
    except ValueError as exc:
        raise CsrfOriginError(f"유효하지 않은 Host 형식입니다: {netloc!r}") from exc
    if (
        not hostname
        or "@" in parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.netloc.lower() != cleaned.lower()
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise CsrfOriginError(f"유효하지 않은 Host 형식입니다: {netloc!r}")
    return hostname


def _parse_origin_url(raw: str, *, label: str) -> tuple[str, str]:
    try:
        parsed = urlparse(raw)
        hostname = (parsed.hostname or "").lower()
        _ = parsed.port
    except ValueError as exc:
        raise CsrfOriginError(f"유효하지 않은 {label} 형식입니다: {raw!r}") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or "@" in parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or not hostname
    ):
        raise CsrfOriginError(f"허용되지 않은 {label}입니다: {raw!r}")
    return parsed.netloc.lower(), hostname


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
    if not expected_host:
        raise CsrfOriginError("Host 헤더가 비어 있습니다.")
    host_name = _hostname_of_netloc(expected_host)
    if host_name not in allowed:
        raise CsrfOriginError(f"허용되지 않은 Host입니다: {expected_host!r}")

    origin = lowered.get("origin", "")
    if origin:
        if origin.lower() == "null":
            raise CsrfOriginError("Origin이 null인 요청은 허용되지 않습니다.")
        origin_netloc, origin_host = _parse_origin_url(origin, label="Origin")
        if origin_netloc != expected_host or origin_host not in allowed:
            raise CsrfOriginError(
                f"허용되지 않은 Origin입니다: {origin!r} (기대 호스트: {expected_host})"
            )
        return

    referer = lowered.get("referer", "")
    if referer:
        referer_netloc, referer_host = _parse_origin_url(referer, label="Referer")
        if referer_netloc != expected_host or referer_host not in allowed:
            raise CsrfOriginError(
                f"허용되지 않은 Referer입니다: {referer!r} (기대 호스트: {expected_host})"
            )


__all__ = [
    "DEFAULT_MAX_PIPELINE_CONCURRENCY",
    "DEFAULT_OPENCV_THREADS",
    "LOCAL_ALLOWED_HOSTS",
    "CsrfOriginError",
    "UploadTooLargeError",
    "async_pipeline_slot",
    "configure_opencv_threads",
    "max_pipeline_concurrency",
    "pipeline_slot",
    "read_upload_bounded",
    "validate_same_origin",
]
