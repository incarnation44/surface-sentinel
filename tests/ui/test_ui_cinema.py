"""3D 파이프라인 시네마 시연(/cinema) 라우트 및 보안·격리 테스트."""

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
    assert "발표 전체화면" in text
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
