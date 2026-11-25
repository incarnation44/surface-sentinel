"""Git 코드 리비전(`git_code_identity`) 헬퍼 단위 테스트 — clean/dirty 트리 해시 결정론 검증.

code_sha/code_dirty 는 진입점(API 생성·워커 시작·평가 CLI)에서 1회 계산한다.
subprocess 는 스텁으로 대체해 git 없이 결정론적으로 검증하고, 파싱 자체가 실제
git 출력과 맞는지는 별도 tmp 저장소 테스트로 확인한다.

계약(감사 5.4):
* clean 트리 → ``(HEAD, False)`` (기존과 동일, subprocess 2회)
* dirty 트리 → ``(f"{HEAD}-dirty-{16진수 16자}", True)`` — 미커밋 **내용**이
  다르면 code_sha 가 달라진다.
* git 실패 → ``("", False)``. diff 만 실패해도 clean 으로 오판하지 않는다.
* 상태 불명(status 실패·타임아웃) → ``("", False)`` — clean ``(HEAD, False)`` 와
  구분한다(감사 6.1 fail-open 제거).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from surface_sentinel.evaluation import recalibration
from surface_sentinel.records.fingerprint import inspection_fingerprint

HEAD = "a" * 40
OTHER_HEAD = "b" * 40
DIRTY_PREFIX = "-dirty-"


def _completed(stdout: str, returncode: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(stdout=stdout, returncode=returncode)


class _GitStub:
    """``git`` 하위 명령별로 고정 출력을 돌려주는 subprocess.run 대역."""

    def __init__(
        self,
        *,
        head: str = HEAD,
        status: str = "",
        diff: str = "",
        root: str = ".",
        untracked: tuple[str, ...] = (),
        status_code: int = 0,
        diff_ok: bool = True,
    ) -> None:
        self.head = head
        self.status = status
        self.diff = diff
        self.root = root
        self.untracked = untracked
        self.status_code = status_code
        self.diff_ok = diff_ok
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        self.calls.append(cmd)
        args = cmd[1:]
        if args[:2] == ["rev-parse", "HEAD"]:
            return _completed(self.head)
        if args[:1] == ["status"]:
            return _completed(self.status, self.status_code)
        if args[:1] == ["diff"]:
            return _completed(self.diff, 0 if self.diff_ok else 128)
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return _completed(self.root)
        if args[:1] == ["ls-files"]:
            return _completed("\0".join(self.untracked))
        return _completed("", returncode=1)


def _install(monkeypatch: pytest.MonkeyPatch, stub: _GitStub) -> _GitStub:
    monkeypatch.setattr(recalibration.subprocess, "run", stub)
    return stub


def _digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_clean_tree_returns_sha_and_not_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install(monkeypatch, _GitStub(head=HEAD + "\n"))

    sha, dirty = recalibration.git_code_identity()

    assert sha == HEAD
    assert dirty is False
    # clean 경로는 기존과 동일하게 2회(rev-parse 1회 + status 1회)만 호출한다.
    assert len(stub.calls) == 2
    assert stub.calls[0][:2] == ["git", "rev-parse"]
    assert stub.calls[1][:2] == ["git", "status"]


def test_clean_tree_fingerprint_is_backward_compatible() -> None:
    """clean 은 code_sha 자리에 HEAD 그대로 → 기존 fingerprint 와 같다."""

    spec_like = inspection_fingerprint(engine="opencv", config_hash="c" * 64, code_sha=HEAD)
    legacy = inspection_fingerprint(
        engine="opencv", config_hash="c" * 64, code_sha=HEAD, code_dirty=False
    )
    assert spec_like == legacy


def test_dirty_tree_returns_head_dirty_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    status = " M src/x.py\n?? new.py\n"
    stub = _install(monkeypatch, _GitStub(status=status, diff="@@ -1 +1 @@\n", untracked=()))

    sha, dirty = recalibration.git_code_identity()

    assert dirty is True
    assert sha.startswith(HEAD + DIRTY_PREFIX)
    digest = sha.split(DIRTY_PREFIX, 1)[1]
    assert len(digest) == recalibration.CODE_DIGEST_CHARS
    assert set(digest) <= set("0123456789abcdef")
    # dirty 경로에서만 추가 git 호출(diff·toplevel·untracked)이 붙는다.
    assert len(stub.calls) > 2


def test_dirty_same_content_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _install(monkeypatch, _GitStub(status=" M src/x.py\n", diff="diff-body\n"))
    sha_a, dirty_a = recalibration.git_code_identity()
    second = _install(monkeypatch, _GitStub(status=" M src/x.py\n", diff="diff-body\n"))
    sha_b, dirty_b = recalibration.git_code_identity()

    assert (sha_a, dirty_a) == (sha_b, dirty_b)
    assert sha_a != HEAD
    assert first.calls and second.calls  # 두 스텁 모두 실제로 쓰였다


def test_different_dirty_content_changes_code_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _GitStub(status=" M src/x.py\n", diff="@@ -1 +1 @@\n-a\n+b\n"))
    sha_a, _ = recalibration.git_code_identity()

    _install(monkeypatch, _GitStub(status=" M src/x.py\n", diff="@@ -1 +1 @@\n-a\n+c\n"))
    sha_b, _ = recalibration.git_code_identity()

    assert sha_a != sha_b
    assert sha_a.startswith(HEAD + DIRTY_PREFIX)
    assert sha_b.startswith(HEAD + DIRTY_PREFIX)
    # 감사 5.4: 같은 HEAD·같은 dirty 여부라도 fingerprint 가 달라진다.
    assert inspection_fingerprint(
        engine="opencv", code_sha=sha_a, code_dirty=True
    ) != inspection_fingerprint(engine="opencv", code_sha=sha_b, code_dirty=True)


def test_untracked_code_content_changes_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    code_file = src / "new.py"
    code_file.write_text("VALUE = 1\n", encoding="utf-8")

    _install(
        monkeypatch,
        _GitStub(status="?? src/new.py\n", root=str(tmp_path), untracked=("src/new.py",)),
    )
    sha_a, _ = recalibration.git_code_identity()

    code_file.write_text("VALUE = 2\n", encoding="utf-8")
    sha_b, _ = recalibration.git_code_identity()

    assert sha_a != sha_b


def test_untracked_outside_code_dirs_hashes_path_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``src/``·``configs/`` 밖 untracked 는 경로만 digest 에 넣는다(비용 회피)."""

    notes = tmp_path / "notes.md"
    notes.write_text("v1\n", encoding="utf-8")

    _install(
        monkeypatch,
        _GitStub(status="?? notes.md\n", root=str(tmp_path), untracked=("notes.md",)),
    )
    sha_a, _ = recalibration.git_code_identity()

    notes.write_text("v2\n", encoding="utf-8")
    sha_b, _ = recalibration.git_code_identity()

    # 경로는 status 출력에 이미 들어 있으므로 내용만 바뀐 경우 digest 는 같다.
    assert sha_a == sha_b


def test_git_missing_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        raise OSError("git not found")

    monkeypatch.setattr(recalibration.subprocess, "run", fake_run)
    assert recalibration.git_code_identity() == ("", False)


def test_status_unknown_is_not_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """status 실패·타임아웃(상태 불명)은 clean 이 아니라 ``("", False)`` (감사 6.1).

    rev-parse 가 성공했어도 dirty 여부를 모르면 ``(HEAD, False)`` 로 기록하지 않는다.
    """

    stub = _install(monkeypatch, _GitStub(head=OTHER_HEAD, status="", status_code=128))
    assert recalibration.git_code_identity() == ("", False)
    assert len(stub.calls) == 2

    def timeout_run(cmd: list[str], **_kw: object) -> types.SimpleNamespace:
        if cmd[1:2] == ["status"]:
            raise subprocess.TimeoutExpired(cmd, 10)
        return _completed(OTHER_HEAD)

    monkeypatch.setattr(recalibration.subprocess, "run", timeout_run)
    assert recalibration.git_code_identity() == ("", False)


def test_diff_failure_falls_back_and_stays_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    """diff 실패 폴백: porcelain 기반 digest 로 진행하되 clean 으로 오판하지 않는다."""

    status = " M src/x.py\n"
    _install(monkeypatch, _GitStub(status=status, diff="ignored", diff_ok=False))

    sha, dirty = recalibration.git_code_identity()

    assert dirty is True
    assert sha.startswith(HEAD + DIRTY_PREFIX)
    # status 문자열만으로 만든 digest 와 같아야 한다(diff 는 실패 표식으로 대체).
    expected = _digest_of("status\0" + status + "\0diff-unavailable\0")
    assert sha == f"{HEAD}{DIRTY_PREFIX}{expected[:16]}"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git 미설치")
def test_real_repository_content_sensitive_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """실제 tmp 저장소로 파싱·결정론·clean 하위 호환을 검증한다(실 저장소 상태 무관)."""

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    tracked = repo / "src" / "a.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "init",
    )
    head = _git(repo, "rev-parse", "HEAD").strip()
    monkeypatch.chdir(repo)

    assert recalibration.git_code_identity() == (head, False)

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    first, dirty = recalibration.git_code_identity()
    again, _ = recalibration.git_code_identity()
    assert dirty is True
    assert first.startswith(f"{head}{DIRTY_PREFIX}")
    assert len(first.split(DIRTY_PREFIX, 1)[1]) == recalibration.CODE_DIGEST_CHARS
    assert first == again  # 같은 내용 → 같은 code_sha

    tracked.write_text("VALUE = 3\n", encoding="utf-8")
    second, _ = recalibration.git_code_identity()
    assert second != first

    (repo / "src" / "new.py").write_text("EXTRA = 1\n", encoding="utf-8")
    third, _ = recalibration.git_code_identity()
    assert third != second
    assert inspection_fingerprint(
        engine="opencv", code_sha=third, code_dirty=True
    ) != inspection_fingerprint(engine="opencv", code_sha=second, code_dirty=True)