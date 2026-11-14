"""T07 브라우저 테스트 (로컬 Chrome 필요).

CI에서는 제외된다(``-m "not data and not browser"``). 로컬 실행:
``uv run pytest -m browser``. Playwright 브라우저 바이너리 대신
시스템 Chrome(``channel="chrome"``)을 쓰고, 없으면 건너뛴다.
통과 기준: 콘솔 오류 0·외부 요청 0·키보드 J/K 흐름·폼 제출 흐름.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import ui_run
from surface_sentinel.ui import create_app

pytestmark = pytest.mark.browser


@pytest.fixture()
def base_url(tmp_path: Path) -> Iterator[str]:
    import uvicorn

    runs_dir = tmp_path / "runs"
    raw_root = tmp_path / "raw"
    runs_dir.mkdir()
    ui_run.make_run(runs_dir, raw_root)
    app = create_app(runs_dir=runs_dir, raw_root=raw_root)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError("UI 서버 시작 시간 초과")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5.0)


@pytest.fixture()
def browser_page(base_url: str) -> Iterator[tuple[Any, list[str], list[str]]]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
        except Exception:
            pytest.skip("로컬 Chrome을 찾을 수 없어 브라우저 테스트를 건너뜁니다.")
        page = browser.new_page()
        errors: list[str] = []
        external: list[str] = []
        page.on(
            "console",
            lambda msg: errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        def on_request(request: Any) -> None:
            if not request.url.startswith((base_url, "data:")):
                external.append(request.url)

        page.on("request", on_request)
        yield page, errors, external
        browser.close()


def test_pages_no_console_errors_no_external_requests(
    browser_page: tuple[Any, list[str], list[str]], base_url: str
) -> None:
    page, errors, external = browser_page
    paths = [
        "/",
        f"/runs/{ui_run.RUN_ID}",
        f"/runs/{ui_run.RUN_ID}/triage",
        f"/runs/{ui_run.RUN_ID}/samples/70002",
        f"/runs/{ui_run.RUN_ID}/samples/70002/mask",
        f"/runs/{ui_run.RUN_ID}/review",
        f"/runs/{ui_run.RUN_ID}/review?sort=priority",
        f"/runs/{ui_run.RUN_ID}/metrics",
        "/watchdog",
        "/inspections",
        "/stats",
        "/compare",
    ]
    for path in paths:
        page.goto(base_url + path, wait_until="networkidle")
    assert errors == []
    assert external == []


def test_keyboard_jk_navigation(
    browser_page: tuple[Any, list[str], list[str]], base_url: str
) -> None:
    page, errors, _ = browser_page
    page.goto(base_url + f"/runs/{ui_run.RUN_ID}/review", wait_until="networkidle")
    page.keyboard.press("j")
    page.wait_for_url(lambda url: "sample=80002" in url)
    page.keyboard.press("k")
    page.wait_for_url(lambda url: "sample=70002" in url)
    assert errors == []


def test_review_submit_flow(
    browser_page: tuple[Any, list[str], list[str]], base_url: str
) -> None:
    page, errors, external = browser_page
    page.goto(base_url + f"/runs/{ui_run.RUN_ID}/review", wait_until="networkidle")
    page.check('input[name="final_status"][value="NG"]')
    page.select_option("#reason_code", "CONFIRMED_DEFECT")
    page.fill("#note", "브라우저 테스트 결정")
    page.click('button[type="submit"]')
    page.wait_for_url(lambda url: "saved=1" in url)
    content = page.content()
    assert "검토 결정을 저장했습니다" in content
    assert "결정 완료" in content
    assert errors == []
    assert external == []

def test_browser_interactive_filters_and_loupe(
    browser_page: tuple[Any, list[str], list[str]], base_url: str
) -> None:
    page, errors, external = browser_page
    page.goto(base_url + f"/runs/{ui_run.RUN_ID}", wait_until="networkidle")
    page.click('button.filter-btn[data-filter="REVIEW"]')
    assert "표시 2 / 7장" in page.text_content("#filter-count")
    page.fill("#sample-search", "80002")
    assert "표시 1 / 7장" in page.text_content("#filter-count")
    page.fill("#sample-search", "")
    page.click('button.filter-btn[data-filter="ALL"]')
    assert "표시 7 / 7장" in page.text_content("#filter-count")

    page.goto(base_url + f"/runs/{ui_run.RUN_ID}/samples/70002/mask", wait_until="networkidle")
    page.click('button.alpha-preset[data-alpha="100"]')
    assert page.text_content("#loupe-alpha-val") == "100%"
    assert errors == []
    assert external == []


def test_browser_live_workbench_with_api_enabled(
    tmp_path: Path, browser_page: tuple[Any, list[str], list[str]]
) -> None:
    import cv2
    import numpy as np
    import uvicorn

    runs_dir = tmp_path / "live_runs"
    raw_root = tmp_path / "live_raw"
    runs_dir.mkdir()
    raw_root.mkdir()
    ui_run.make_run(runs_dir, raw_root)

    img_path = tmp_path / "probe.png"
    img = np.full((80, 80), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    img_path.write_bytes(buf.tobytes())

    app = create_app(runs_dir=runs_dir, raw_root=raw_root, enable_api=True)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("라이브 서버 시작 실패")
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        live_url = f"http://127.0.0.1:{port}"

        page, errors, external = browser_page
        page.goto(live_url + "/inspections", wait_until="networkidle")
        page.set_input_files("#live-file", str(img_path))
        page.click("#live-submit")
        page.wait_for_selector("#live-result .okbox a")
        page.click("#live-result .okbox a")
        page.wait_for_load_state("networkidle")
        assert "검사 상세 #1" in page.content()
        assert errors == []
        assert all(u.startswith((live_url, "data:")) for u in external)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_browser_wave3b_phase_diagram_hotspot_and_watchdog(
    browser_page: tuple[Any, list[str], list[str]], base_url: str
) -> None:
    page, errors, external = browser_page
    page.goto(base_url + f"/runs/{ui_run.RUN_ID}/samples/70002", wait_until="networkidle")
    assert page.locator("#phase-diagram-svg").count() == 1
    assert page.locator("#cf-table .cf-row").count() == 1
    page.click("#cf-table .cf-row[data-comp-idx='0']")
    assert page.get_attribute(".cf-dot[data-comp-idx='0']", "stroke") == "#2563eb"

    page.goto(base_url + f"/runs/{ui_run.RUN_ID}/metrics", wait_until="networkidle")
    assert "2D 라인스캔 스트립 공간 핫스팟 지도" in page.content()

    page.goto(base_url + f"/watchdog?run_id={ui_run.RUN_ID}", wait_until="networkidle")
    assert "4. Phase 4 사람-기계 피드백 캘리브레이션" in (page.text_content("main") or "")
    assert "Phase 4" not in page.title()
    assert errors == []
    assert external == []



def test_t10_integrated_guide_desktop_and_mobile(
    browser_page: tuple[Any, list[str], list[str]],
) -> None:
    guide_path = (
        Path(__file__).resolve().parents[2] / "docs" / "surface-sentinel-integrated-guide.html"
    )
    assert guide_path.is_file()
    guide_url = guide_path.resolve().as_uri()
    page, errors, external = browser_page

    # 1) 데스크톱 (1280x800)
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(guide_url, wait_until="load")
    assert page.locator("#sec-01-cover").count() == 1
    assert page.locator("#sec-30-changelog").count() == 1
    assert page.locator("nav#guide-toc a").count() == 30

    # 2) 모바일 (390x844)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(guide_url, wait_until="load")
    assert page.locator("#mobile-toc-details").count() == 1

    assert errors == []
    assert [u for u in external if u.startswith(("http://", "https://"))] == []
def test_browser_dark_telemetry_theme_toggle(
    browser_page: tuple[Any, list[str], list[str]], base_url: str
) -> None:
    page, errors, external = browser_page
    page.goto(base_url + "/", wait_until="networkidle")
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") is None
    assert page.get_attribute("#theme-toggle", "aria-pressed") == "false"

    page.click("#theme-toggle")
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "dark"
    assert page.evaluate("localStorage.getItem('ss-theme')") == "dark"
    assert page.get_attribute("#theme-toggle", "aria-pressed") == "true"

    # 다른 페이지 이동 시 localStorage 기반 다크 테마 유지 및 본문 렌더링 확인
    page.goto(base_url + f"/watchdog?run_id={ui_run.RUN_ID}", wait_until="networkidle")
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "dark"
    assert "4. Phase 4 사람-기계 피드백 캘리브레이션" in (page.text_content("main") or "")

    # 다시 클릭 시 라이트 모드 복귀 및 ?theme=dark 쿼리 파라미터 전환
    page.click("#theme-toggle")
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") is None
    page.goto(
        base_url + f"/runs/{ui_run.RUN_ID}/samples/70002?theme=dark",
        wait_until="networkidle",
    )
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "dark"
    assert errors == []
    assert external == []
