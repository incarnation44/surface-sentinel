"""3D 파이프라인 시네마 시연(`/cinema`) 라우트 및 보안·격리 테스트.

오프라인 렌더링, 프리셋/업로드 단계 프레임 생성, 경로 순회 차단 및 runs/DB 무변경을 검증한다.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import ui_run
from surface_sentinel.contracts import STAGE_ORDER
from surface_sentinel.ui import create_app
from surface_sentinel.ui.cinema import MAX_CINEMA_UPLOAD_BYTES


@pytest.fixture()
def env(tmp_path: Path) -> tuple[Path, Path, Path]:
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run_dir = ui_run.make_run(runs_dir, raw_root)
    return runs_dir, raw_root, run_dir


@pytest.fixture()
def client(env: tuple[Path, Path, Path]) -> TestClient:
    runs_dir, raw_root, _ = env
    return TestClient(create_app(runs_dir=runs_dir, raw_root=raw_root))


def _make_defect_png_bytes() -> bytes:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "dark_defect.png"
    if fixture.is_file():
        return fixture.read_bytes()
    img = np.full((320, 120, 3), 132, dtype=np.uint8)
    cv2.line(img, (44, 108), (74, 198), (22, 22, 22), thickness=5)
    cv2.circle(img, (60, 152), 9, (18, 18, 18), thickness=-1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_cinema_page_renders_offline_without_external_urls(client: TestClient) -> None:
    resp = client.get("/cinema")
    assert resp.status_code == 200
    text = resp.text
    assert "3D 파이프라인 시네마" in text
    assert "3D 분해 스택" in text
    assert "2D 시네마 와이프" in text
    assert "결함 줌인 콜아웃" in text
    assert "INSPECTION LOOP" in text
    assert "전체화면" in text
    assert "http://" not in text
    assert "https://" not in text


def test_cinema_preset_returns_all_frames_and_timings(client: TestClient) -> None:
    page = client.get("/cinema")
    assert page.status_code == 200
    resp = client.get("/cinema/preset/synth-dark")
    assert resp.status_code == 200
    data = resp.json()
    assert data["split"] == "tune"
    assert data["status"] == "NG"
    assert len(data["config_hash"]) == 64
    assert set(data["timing_ms"].keys()) == {st.value for st in STAGE_ORDER}
    assert len(data["chapters"]) == 7

    expected_frames = {"original", "roi", "response", "z_map", "candidates", "refined", "overlay"}
    assert set(data["frames"].keys()) == expected_frames
    for key, uri in data["frames"].items():
        assert uri.startswith("data:image/png;base64,"), f"{key} data URI 형식 오류"
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{key} PNG 시그니처 불일치"


def test_cinema_preset_empty_runs_uses_builtin_fallback(tmp_path: Path) -> None:
    empty_runs = tmp_path / "empty-runs"
    empty_runs.mkdir()
    # repo_root 를 빈 tmp_path 로 지정해 synthetic fixture 도 없는 환경 검증
    c = TestClient(create_app(runs_dir=empty_runs, raw_root=tmp_path, repo_root=tmp_path))
    assert c.get("/cinema").status_code == 200
    resp = c.get("/cinema/preset/builtin-defect")
    assert resp.status_code == 200
    assert resp.json()["status"] == "NG"
    assert not (empty_runs / "records.db").exists()


def test_cinema_preset_rejects_unknown_and_traversal(client: TestClient) -> None:
    assert client.get("/cinema/preset/nonexistent-id").status_code == 404
    assert client.get("/cinema/preset/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_cinema_upload_inspect_returns_frames_without_mutating_runs_or_db(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    before = sorted(p.relative_to(runs_dir).as_posix() for p in runs_dir.rglob("*"))

    png_bytes = _make_defect_png_bytes()
    resp = client.post(
        "/cinema/inspect",
        files={"file": ("demo_scratch.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sample_id"] == "demo_scratch"
    assert data["split"] == "tune"
    assert data["status"] == "NG"
    assert data["score"] >= data["ng_min"]
    assert data["telemetry"]["component_count"] >= 1
    assert len(data["frames"]) == 7

    # 읽기 전용 격리: runs_dir 에 파일/DB 가 전혀 생기지 않아야 한다.
    after = sorted(p.relative_to(runs_dir).as_posix() for p in runs_dir.rglob("*"))
    assert after == before
    assert not (runs_dir / "records.db").exists()


def test_cinema_upload_rejects_bad_ext_empty_corrupt_and_oversize(client: TestClient) -> None:
    r_ext = client.post(
        "/cinema/inspect",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r_ext.status_code == 400

    r_empty = client.post(
        "/cinema/inspect",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r_empty.status_code == 400

    r_corrupt = client.post(
        "/cinema/inspect",
        files={"file": ("broken.png", b"not-a-valid-png", "image/png")},
    )
    assert r_corrupt.status_code == 400

    r_big = client.post(
        "/cinema/inspect",
        files={"file": ("huge.png", b"0" * (MAX_CINEMA_UPLOAD_BYTES + 1), "image/png")},
    )
    assert r_big.status_code == 413


def test_cinema_rejects_oversize_pixel_dimensions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("surface_sentinel.ui.cinema.MAX_CINEMA_IMAGE_PIXELS", 1000)
    png_bytes = _make_defect_png_bytes()
    resp = client.post(
        "/cinema/inspect",
        files={"file": ("large_res.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 400
    assert "해상도" in resp.json()["detail"]


def test_cinema_preset_discovery_collects_across_multiple_tune_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run1 = ui_run.make_run(runs_dir, raw_root, run_id="20261110T010000Z-aaaa1111-tune")
    run2 = ui_run.make_run(runs_dir, raw_root, run_id="20261112T010000Z-bbbb2222-tune")
    # ui_run 기본 이미지는 8x8 이므로 manifest 해상도를 128x128 로 올려 64px 필터를 통과시킨다.
    for rdir in (run1, run2):
        mf = rdir / "manifest_snapshot.csv"
        updated = mf.read_text(encoding="utf-8").replace(",8,8,", ",128,128,")
        mf.write_text(updated, encoding="utf-8")
    # 최신 run 의 results.jsonl 에서 NG 행을 제거해도 이전 tune run 의 NG 를 보충해야 함
    res_path = run2 / "results.jsonl"
    kept = [
        line
        for line in res_path.read_text(encoding="utf-8").splitlines()
        if '"status": "NG"' not in line
    ]
    res_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    from surface_sentinel.ui.cinema import discover_cinema_presets

    presets = discover_cinema_presets(runs_dir, raw_root, tmp_path)
    badges = {p.badge for p in presets if p.source == "tune-run"}
    assert "NG" in badges
    assert "OK" in badges


def test_cinema_upload_closes_upload_file_on_all_paths(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from starlette.datastructures import UploadFile

    from surface_sentinel.ui import app as ui_app_mod

    close_calls = 0
    closed_before_trace = False
    orig_close = UploadFile.close
    orig_trace = ui_app_mod.run_cinema_trace

    async def tracked_close(self: UploadFile) -> None:
        nonlocal close_calls
        close_calls += 1
        await orig_close(self)

    def tracked_trace(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal closed_before_trace
        closed_before_trace = close_calls >= 1
        return orig_trace(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(UploadFile, "close", tracked_close)
    monkeypatch.setattr(ui_app_mod, "run_cinema_trace", tracked_trace)

    assert (
        client.post(
            "/cinema/inspect",
            files={"file": ("demo.png", _make_defect_png_bytes(), "image/png")},
        ).status_code
        == 200
    )
    assert closed_before_trace is True


def test_read_upload_bounded_validates_limits_and_always_closes() -> None:
    import asyncio

    from surface_sentinel.io.limits import UploadTooLargeError, read_upload_bounded

    class _DummyUpload:
        def __init__(self, chunks: list[bytes], *, fail_on_read: bool = False) -> None:
            self._chunks = list(chunks)
            self._fail = fail_on_read
            self.closed = False

        async def read(self, size: int) -> bytes:
            if self._fail:
                raise RuntimeError("read boom")
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

        async def close(self) -> None:
            self.closed = True

    async def _exercise() -> None:
        # 1) 경계값 거부
        u_bad = _DummyUpload([b"abc"])
        with pytest.raises(ValueError):
            await read_upload_bounded(u_bad, max_bytes=-1, chunk_bytes=8)
        with pytest.raises(ValueError):
            await read_upload_bounded(u_bad, max_bytes=10, chunk_bytes=0)
        with pytest.raises(ValueError):
            await read_upload_bounded(u_bad, max_bytes=10, chunk_bytes=-1)

        # 2) 정상 경로 close
        u_ok = _DummyUpload([b"ab", b"cd"])
        assert await read_upload_bounded(u_ok, max_bytes=4, chunk_bytes=2) == b"abcd"
        assert u_ok.closed is True

        # 3) 상한 초과 시 close
        u_big = _DummyUpload([b"abc", b"def"])
        with pytest.raises(UploadTooLargeError):
            await read_upload_bounded(u_big, max_bytes=4, chunk_bytes=3)
        assert u_big.closed is True

        # 4) 읽기 예외 시 close
        u_err = _DummyUpload([], fail_on_read=True)
        with pytest.raises(RuntimeError, match="read boom"):
            await read_upload_bounded(u_err, max_bytes=10, chunk_bytes=4)
        assert u_err.closed is True

    asyncio.run(_exercise())


def test_cinema_rejects_oversize_pixels_before_writing_temp_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    monkeypatch.setattr("surface_sentinel.ui.cinema.MAX_CINEMA_IMAGE_PIXELS", 1000)
    temp_created = False
    orig_tempdir = tempfile.TemporaryDirectory

    def tracked_tempdir(*args: object, **kwargs: object):  # noqa: ANN202
        nonlocal temp_created
        temp_created = True
        return orig_tempdir(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", tracked_tempdir)
    resp = client.post(
        "/cinema/inspect",
        files={"file": ("large_res.png", _make_defect_png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert "해상도" in resp.json()["detail"]
    assert temp_created is False


def test_cinema_preset_route_is_async_handler(env: tuple[Path, Path, Path]) -> None:
    import inspect

    runs_dir, raw_root, _ = env
    app = create_app(runs_dir=runs_dir, raw_root=raw_root)
    route = next(
        r for r in app.router.routes if getattr(r, "path", "") == "/cinema/preset/{preset_id}"
    )
    assert inspect.iscoroutinefunction(route.endpoint)


def test_cinema_rejects_unrecognized_header_before_temp_write(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile
    from contextlib import asynccontextmanager

    from starlette.datastructures import UploadFile

    from surface_sentinel.ui import app as ui_app_mod

    temp_created = False
    slot_entered = False
    orig_tempdir = tempfile.TemporaryDirectory

    def tracked_tempdir(*args: object, **kwargs: object):  # noqa: ANN202
        nonlocal temp_created
        temp_created = True
        return orig_tempdir(*args, **kwargs)

    @asynccontextmanager
    async def tracked_slot():  # noqa: ANN202
        nonlocal slot_entered
        slot_entered = True
        yield

    close_calls = 0
    orig_close = UploadFile.close

    async def tracked_close(self: UploadFile) -> None:
        nonlocal close_calls
        close_calls += 1
        await orig_close(self)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", tracked_tempdir)
    monkeypatch.setattr(ui_app_mod, "async_pipeline_slot", tracked_slot)
    monkeypatch.setattr(UploadFile, "close", tracked_close)

    # 1) 알 수 없는 이미지 헤더는 슬롯 획득·임시파일 쓰기 전에 400으로 차단한다.
    resp = client.post(
        "/cinema/inspect",
        files={"file": ("broken.png", b"not-a-valid-image-header", "image/png")},
    )
    assert resp.status_code == 400
    assert temp_created is False
    assert slot_entered is False
    assert close_calls >= 1

    # 2) 조기 403 Origin 거부 경로에서도 UploadFile.close()가 호출된다.
    before_403 = close_calls
    resp_403 = client.post(
        "/cinema/inspect",
        files={"file": ("broken.png", b"not-a-valid-image-header", "image/png")},
        headers={"Origin": "https://evil.example"},
    )
    assert resp_403.status_code == 403
    assert close_calls > before_403


def test_cinema_empty_roi_and_nonfinite_response_frame_safe(
    tmp_path: Path,
) -> None:
    import json
    import warnings

    from surface_sentinel.ui.cinema import (
        _render_response_frame,
        _render_roi_frame,
        discover_cinema_presets,
    )

    bgr = np.full((32, 32, 3), 120, dtype=np.uint8)
    empty_roi = np.zeros((32, 32), dtype=np.uint8)
    roi_frame = _render_roi_frame(bgr, empty_roi)
    assert roi_frame.shape == (32, 32, 3)
    assert roi_frame.dtype == np.uint8

    bad_resp = np.full((32, 32), 10.0, dtype=np.float32)
    bad_resp[0, 0] = np.nan
    bad_resp[0, 1] = np.inf
    bad_resp[0, 2] = -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        resp_frame = _render_response_frame(bad_resp, np.full((32, 32), 255, dtype=np.uint8))
    assert resp_frame.shape == (32, 32, 3)
    assert resp_frame.dtype == np.uint8

    # results.jsonl 레코드에 score=None 이 있어도 discover_cinema_presets 가 실패하지 않는다.
    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run_dir = ui_run.make_run(runs_dir, raw_root)
    res_path = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in res_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["score"] = None
    res_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    presets = discover_cinema_presets(runs_dir, raw_root, tmp_path)
    assert len(presets) >= 1



def test_cinema_presets_skip_runs_from_other_config_hash(tmp_path: Path) -> None:
    """동결 설정과 다른 해시의 run 은 프리셋 후보에서 빠진다 (재현되지 않는 '실제 불량' 방지)."""

    from surface_sentinel.ui.cinema import discover_cinema_presets

    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    run_dir = ui_run.make_run(runs_dir, raw_root)
    mf = run_dir / "manifest_snapshot.csv"
    mf.write_text(mf.read_text(encoding="utf-8").replace(",8,8,", ",128,128,"), encoding="utf-8")

    matched = discover_cinema_presets(runs_dir, raw_root, tmp_path, config_hash=ui_run.CONFIG_HASH)
    assert any(p.source == "tune-run" for p in matched)

    other = discover_cinema_presets(runs_dir, raw_root, tmp_path, config_hash="0" * 64)
    assert not any(p.source == "tune-run" for p in other)


def test_cinema_showcase_preset_comes_first_from_tune_csv(tmp_path: Path) -> None:
    """발표 대표 샘플(#20117)은 tune.csv 에 있고 이미지가 있으면 첫 프리셋이 된다."""

    from surface_sentinel.ui.cinema import discover_cinema_presets

    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    (raw_root / "test").mkdir(parents=True)
    img = np.full((128, 64, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(raw_root / "test" / "20117.png"), img)
    splits = tmp_path / "data" / "splits"
    splits.mkdir(parents=True)
    (splits / "tune.csv").write_text(
        "sample_id,image_path,label\n20117,test/20117.png,defect\n", encoding="utf-8"
    )

    presets = discover_cinema_presets(runs_dir, raw_root, tmp_path)
    assert presets[0].preset_id == "showcase-ng-20117"
    assert presets[0].title == "대표 불량 #20117"
    assert presets[0].label == "defect"

    # tune.csv 에 없으면(봉인 분할 등) 대표 프리셋을 만들지 않는다.
    (splits / "tune.csv").write_text("sample_id,image_path,label\n", encoding="utf-8")
    presets = discover_cinema_presets(runs_dir, raw_root, tmp_path)
    assert all(not p.preset_id.startswith("showcase-") for p in presets)
