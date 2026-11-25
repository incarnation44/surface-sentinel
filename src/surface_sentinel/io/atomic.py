"""임시 디렉터리 작성·fsync·rename 기반 원자적 디렉터리 저장 유틸리티.

``runs/.tmp-<run_id>-<pid>/``에 모든 파일을 쓴 뒤 파일을 fsync하고 마지막에
``os.replace``로 최종 이름으로 바꾼다. replace가 실패하면 최종 디렉터리를
남기지 않고, 임시 디렉터리도 지운다.

Windows에서는 디렉터리 핸들 fsync가 지원되지 않으므로 디렉터리 fsync는
POSIX에서만 수행한다. 파일 fsync는 두 플랫폼 모두에서 수행한다.

보장 범위(감사 5.12): 이 구조가 보장하는 것은 **원자적 게시**다 — 최종
디렉터리는 완성본만 보이거나 아예 보이지 않는다(부분 run이 정상 결과로
노출되지 않는다). **전원 손실 후 완전한 crash durability**는 별개 주장이며
특히 Windows에서는 보장하지 않는다(일부 파일 fsync 오류 무시, 디렉터리
fsync 부재). durability가 필요한 운영 환경에서는 별도 검증이 필요하다.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class AtomicWriteError(OSError):
    """원자 저장을 완료할 수 없을 때 발생한다."""


def fsync_file(path: str | Path) -> None:
    """파일 하나를 fsync한다.

    Windows에서는 읽기 전용 핸들의 FlushFileBuffers가 EBADF로 실패하므로
    쓰기 가능 핸들로 연다. 그래도 실패하면 Windows에 한해 건너뛴다. 산출물
    파일은 기록 시점(``io.artifacts``)에 이미 fsync되므로 이중 안전장치다.
    """

    target = Path(path)
    handle = None
    try:
        handle = os.open(target, os.O_RDWR)
    except OSError:
        with contextlib.suppress(OSError):
            handle = os.open(target, os.O_RDONLY)
    if handle is None:
        return
    try:
        os.fsync(handle)
    except OSError:
        if os.name == "posix":
            raise
    finally:
        os.close(handle)


def fsync_directory(path: str | Path) -> None:
    """디렉터리 엔트리를 fsync한다. POSIX가 아니면 조용히 건너뛴다."""

    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        handle = os.open(Path(path), os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def fsync_tree(root: str | Path) -> None:
    """디렉터리 아래 모든 파일과 하위 디렉터리를 fsync한다."""

    base = Path(root)
    entries = sorted(base.rglob("*"), key=lambda item: item.as_posix())
    for entry in entries:
        if entry.is_file():
            fsync_file(entry)
    for entry in entries:
        if entry.is_dir():
            fsync_directory(entry)
    fsync_directory(base)


@contextmanager
def atomic_directory(target: str | Path) -> Iterator[Path]:
    """``target``을 원자적으로 만든다.

    블록 안에서 받은 임시 디렉터리에 파일을 쓰면, 블록이 정상 종료할 때
    fsync 후 ``os.replace``로 최종 이름으로 교체한다. 블록이 예외로 끝나거나
    replace가 실패하면 최종 디렉터리를 남기지 않고 임시 디렉터리를 지운다.
    """

    final = Path(target)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".tmp-{final.name}-{os.getpid()}"
    if temporary.exists():
        raise AtomicWriteError(f"임시 디렉터리가 이미 있습니다: {temporary}")
    if final.exists():
        raise AtomicWriteError(f"최종 디렉터리가 이미 있습니다: {final}")

    temporary.mkdir()
    committed = False
    try:
        yield temporary
        fsync_tree(temporary)
        os.replace(temporary, final)
        fsync_directory(final.parent)
        committed = True
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "AtomicWriteError",
    "atomic_directory",
    "fsync_directory",
    "fsync_file",
    "fsync_tree",
]
