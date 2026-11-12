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
        f"/runs/{ui_run.RUN_ID}/samples/70002",
        f"/runs/{ui_run.RUN_ID}/review",
        f"/runs/{ui_run.RUN_ID}/metrics",
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