"""T07 UI 라우트 테스트 (TestClient).

합성 run(``ui_run.make_run``)만 사용한다. 저장소의 ``runs/``·``data/``는
건드리지 않는다. 통과 기준: 잘못된 전이 거부·append-only·외부 요청 0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ui_run
from surface_sentinel.review.store import DECISION_FIELDS, REVIEW_DECISIONS_JSONL
from surface_sentinel.ui import create_app


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


def _decision_form(**overrides: str) -> dict[str, str]:
    data = {
        "final_status": "NG",
        "reason_code": "CONFIRMED_DEFECT",
        "note": "스크래치 확인",
        "reviewer": "local",
        "return_to": "review",
    }
    data.update(overrides)
    return data


def test_index_lists_runs(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert ui_run.RUN_ID in response.text


def test_index_empty_runs_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty-runs"
    empty.mkdir()
    client = TestClient(create_app(runs_dir=empty, raw_root=tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "표시할 run이 없습니다" in response.text


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "surface-sentinel-ui"
    assert payload["runs"] == 1


def test_run_detail_grid(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}")
    assert response.status_code == 200
    for spec in ui_run.DEFAULT_SAMPLES:
        assert spec["sample_id"] in response.text
    # 한국어 뜻과 영어 배지를 함께 표시한다 (색만으로 의미 전달 금지).
    assert "정상" in response.text
    assert "보류" in response.text
    assert "b-REVIEW" in response.text


def test_sample_detail_shows_evidence(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002")
    assert response.status_code == 200
    text = response.text
    assert ui_run.CONFIG_HASH in text
    assert "0.300000" in text
    assert "response.png" in text
    assert "candidates.png" in text
    assert "검토 결정" in text


def test_sample_detail_ok_has_no_form(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70001")
    assert response.status_code == 200
    # 폼 요소 자체가 없어야 한다 ("검토 결정" 문구는 이력 빈 상태 안내에도 쓰인다).
    assert '<form class="decision"' not in response.text
    assert "불변" in response.text


def test_error_sample_page(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/60001")
    assert response.status_code == 200
    assert "E_INPUT_DECODE" in response.text
    assert "ERROR는 판정이 아니므로" in response.text


def test_overlay_bytes_match(client: TestClient, env: tuple[Path, Path, Path]) -> None:
    _, _, run_dir = env
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70001/overlay.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == (run_dir / "overlays" / "70001_overlay.png").read_bytes()


def test_original_bytes_match(client: TestClient, env: tuple[Path, Path, Path]) -> None:
    _, raw_root, _ = env
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70001/original.png")
    assert response.status_code == 200
    assert response.content == (raw_root / "tune" / "70001.png").read_bytes()


def test_stage_images(client: TestClient, env: tuple[Path, Path, Path]) -> None:
    _, _, run_dir = env
    for kind in ("response", "candidates"):
        response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002/{kind}.png")
        assert response.status_code == 200
        assert response.content == (run_dir / "stages" / f"70002_{kind}.png").read_bytes()


def test_missing_media_returns_404(client: TestClient) -> None:
    # ERROR 샘플은 오버레이가 없고, 70001은 단계 이미지가 없다.
    assert client.get(f"/runs/{ui_run.RUN_ID}/samples/60001/overlay.png").status_code == 404
    assert client.get(f"/runs/{ui_run.RUN_ID}/samples/70001/response.png").status_code == 404


def test_review_queue_defaults_to_first_undecided(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/review")
    assert response.status_code == 200
    text = response.text
    assert "검토 큐 (REVIEW 2건)" in text
    # 첫 미결정 샘플은 70002 → 폼 action과 다음(J) 링크로 확인한다.
    assert f'/runs/{ui_run.RUN_ID}/review/70002' in text
    assert "sample=80002" in text


def test_review_sample_param(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/review?sample=80002")
    assert response.status_code == 200
    assert f'/runs/{ui_run.RUN_ID}/review/80002' in response.text
    # REVIEW가 아닌 샘플은 큐에 없다.
    assert client.get(f"/runs/{ui_run.RUN_ID}/review?sample=70001").status_code == 404


def test_review_submit_appends_and_redirects(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    _, _, run_dir = env
    results_before = (run_dir / "results.jsonl").read_bytes()
    response = client.post(
        f"/runs/{ui_run.RUN_ID}/review/70002",
        data=_decision_form(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "saved=1" in response.headers["location"]
    log = run_dir / REVIEW_DECISIONS_JSONL
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert list(payload.keys()) == list(DECISION_FIELDS)
    assert payload["machine_status"] == "REVIEW"
    assert payload["final_status"] == "NG"
    assert payload["run_id"] == ui_run.RUN_ID
    assert payload["config_hash"] == ui_run.CONFIG_HASH
    overlay = run_dir / "overlays" / "70002_overlay.png"
    assert payload["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    # 기계 판정 파일은 바이트째 불변이어야 한다.
    assert (run_dir / "results.jsonl").read_bytes() == results_before


def test_review_submit_updates_effective_status(client: TestClient) -> None:
    client.post(f"/runs/{ui_run.RUN_ID}/review/70002", data=_decision_form())
    detail = client.get(f"/runs/{ui_run.RUN_ID}")
    assert "검토 반영" in detail.text
    sample = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002")
    assert "최종 상태" in sample.text
    # 마지막 결정이 최종: 두 번째 결정(OK)이 그리드에 반영된다.
    client.post(
        f"/runs/{ui_run.RUN_ID}/review/70002",
        data=_decision_form(final_status="OK", reason_code="FALSE_ALARM"),
    )
    metrics = client.get(f"/runs/{ui_run.RUN_ID}/metrics")
    assert "<th>결정 완료</th><td>1</td>" in metrics.text


@pytest.mark.parametrize("sample_id", ["70001", "80001", "60001"])
def test_review_submit_non_review_rejected(
    client: TestClient, env: tuple[Path, Path, Path], sample_id: str
) -> None:
    _, _, run_dir = env
    response = client.post(f"/runs/{ui_run.RUN_ID}/review/{sample_id}", data=_decision_form())
    assert response.status_code == 400
    assert not (run_dir / REVIEW_DECISIONS_JSONL).exists()


def test_review_submit_invalid_status_rejected(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    _, _, run_dir = env
    response = client.post(
        f"/runs/{ui_run.RUN_ID}/review/70002",
        data=_decision_form(final_status="MAYBE"),
    )
    assert response.status_code == 400
    assert not (run_dir / REVIEW_DECISIONS_JSONL).exists()


def test_review_submit_other_without_note_rejected(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    _, _, run_dir = env
    response = client.post(
        f"/runs/{ui_run.RUN_ID}/review/70002",
        data=_decision_form(reason_code="OTHER", note=""),
    )
    assert response.status_code == 400
    assert not (run_dir / REVIEW_DECISIONS_JSONL).exists()


def test_review_submit_missing_fields(client: TestClient) -> None:
    response = client.post(f"/runs/{ui_run.RUN_ID}/review/70002", data={})
    assert response.status_code == 422


def test_review_submit_unknown_sample(client: TestClient) -> None:
    response = client.post(f"/runs/{ui_run.RUN_ID}/review/99999", data=_decision_form())
    assert response.status_code == 404


def test_unknown_run_and_bad_run_id(client: TestClient) -> None:
    assert client.get("/runs/20261112T000000Z-deadbeef-tune").status_code == 404
    assert client.get("/runs/not-a-run-id").status_code == 404


def test_metrics_page(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/metrics")
    assert response.status_code == 200
    text = response.text
    assert "기계 판정 기준" in text
    # escape 1/3, 오탐 1/3, review 2/6 — 분모가 함께 표시된다.
    assert "0.3333" in text
    assert "(분모 3)" in text
    assert "(분모 6)" in text
    assert "/samples/80003" in text
    assert "/samples/70003" in text
    assert "<th>REVIEW 샘플</th><td>2</td>" in text


def test_note_is_html_escaped(client: TestClient) -> None:
    client.post(
        f"/runs/{ui_run.RUN_ID}/review/70002",
        data=_decision_form(reason_code="OTHER", note="<script>alert(1)</script>"),
    )
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002")
    assert "<script>alert" not in response.text
    assert "&lt;script&gt;" in response.text


def test_no_external_urls_and_no_docs(client: TestClient) -> None:
    paths = [
        "/",
        f"/runs/{ui_run.RUN_ID}",
        f"/runs/{ui_run.RUN_ID}/samples/70002",
        f"/runs/{ui_run.RUN_ID}/review",
        f"/runs/{ui_run.RUN_ID}/metrics",
    ]
    for path in paths:
        text = client.get(path).text
        assert "http://" not in text
        assert "https://" not in text
    # 설계 6경로 계약 유지: docs/openapi는 비활성화.
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_manifest_path_traversal_rejected(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    _, raw_root, run_dir = env
    manifest = run_dir / "manifest_snapshot.csv"
    original = manifest.read_text(encoding="utf-8")
    # 절대경로와 ../ 이탈을 각각 시도한다. 둘 다 원본 제공을 거부해야 한다.
    for bad in (str(raw_root / "tune" / "70001.png"), "../outside.png"):
        manifest.write_text(original.replace("tune/70001.png", bad), encoding="utf-8")
        response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70001/original.png")
        assert response.status_code == 404


def test_review_page_script_uses_tojson_without_html_entity_mangling(
    client: TestClient,
) -> None:
    import re

    response = client.get(f"/runs/{ui_run.RUN_ID}/review?sort=priority")
    assert response.status_code == 200
    scripts = re.findall(r"<script>(.*?)</script>", response.text, flags=re.DOTALL)
    nav_scripts = [s for s in scripts if "window.location.href" in s]
    assert nav_scripts
    script_body = nav_scripts[0]
    assert "&amp;" not in script_body
    href_match = re.search(r"window\.location\.href\s*=\s*(\".*?\");", script_body)
    assert href_match is not None
    parsed_url = json.loads(href_match.group(1))
    assert parsed_url.startswith(f"/runs/{ui_run.RUN_ID}/review?")
    assert "&sort=priority" in parsed_url
    assert "&amp;" not in parsed_url
