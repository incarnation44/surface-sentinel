"""로컬 웹 UI 고급 관제 라우트 테스트.

트리아지 Lift(`/triage`), 위험도순 리뷰 정렬, Watchdog(`/watchdog`), 검사 DB 탐색기 화면을 검증한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import ui_run
from agent_fakes import make_component, make_observation
from surface_sentinel.agent.evidence import assess_cross_evidence
from surface_sentinel.agent.review import assess_review_triage
from surface_sentinel.agent.stability import arbitrate_patches
from surface_sentinel.cli import main
from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import (
    append_agent_assessment,
    insert_engine_observation,
    insert_inspection,
)
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


def _seed_cross_evidence_db(runs_dir: Path) -> int:
    conn = connect(runs_dir / "records.db")
    try:
        init_db(conn)
        sha = hashlib.sha256(b"cross-evidence-seed").hexdigest()
        overlay_file = runs_dir / "seed_overlay.png"
        img = np.full((32, 32, 3), 120, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        overlay_file.write_bytes(buf.tobytes())

        comp_a = make_component(0, (4, 4, 10, 10), area_px=64, evidence=0.44)
        comp_b = make_component(0, (6, 6, 10, 10), area_px=72, evidence=0.81)
        res = SampleResult(
            run_id="seed-run",
            sample_id="cross_001",
            original_id="cross_001",
            split="tune",
            label="defect",
            status=Status.REVIEW,
            score=0.44,
            reasons=(ReasonCode.EVIDENCE_IN_REVIEW_BAND,),
            components=(comp_a,),
            overlay_path=str(overlay_file),
            config_hash="0" * 64,
        )
        iid = insert_inspection(conn, res, engine="combined", image_sha256=sha, task="det")
        obs_a = make_observation("opencv", Status.REVIEW, components=[comp_a], score=0.44)
        obs_b = make_observation("yolo-det", Status.NG, components=[comp_b], score=0.81, task="det")
        insert_engine_observation(conn, iid, obs_a.result, engine="opencv")
        insert_engine_observation(conn, iid, obs_b.result, engine="yolo-det", task="det")
        cross = assess_cross_evidence(obs_a, obs_b)
        arb = arbitrate_patches(img, obs_a, obs_b)
        triage = assess_review_triage(res, cross_assessment=cross, arbitration=arb)
        append_agent_assessment(
            conn,
            iid,
            assessment_hash=cross.assessment_hash,
            policy_hash=cross.policy_hash,
            flags=[f.value for f in cross.flags],
            recommended_action=cross.recommended_action,
            diagnostics={
                **cross.to_dict(),
                "arbitration": arb.to_dict(),
                "triage": triage.to_dict(),
            },
            review_priority=triage.priority.value,
        )
        return iid
    finally:
        conn.close()


def test_triage_page_renders_gain_curve_lift_and_ranked_queue(client: TestClient) -> None:
    resp = client.get(f"/runs/{ui_run.RUN_ID}/triage")
    assert resp.status_code == 200
    text = resp.text
    assert "에이전트 트리아지" in text
    assert "누적 결함 회수 곡선" in text
    assert "Top-20%" in text
    assert "80002" in text
    assert "70002" in text
    assert "P0" in text and "P3" in text
    # 80002 (score 0.35) 가 70002 (score 0.30) 보다 먼저 나와야 한다.
    assert text.index("80002") < text.index("70002")


def test_review_sort_priority_orders_highest_risk_first(client: TestClient) -> None:
    default_resp = client.get(f"/runs/{ui_run.RUN_ID}/review")
    assert default_resp.status_code == 200
    assert f"/runs/{ui_run.RUN_ID}/review/70002" in default_resp.text

    prio_resp = client.get(f"/runs/{ui_run.RUN_ID}/review", params={"sort": "priority"})
    assert prio_resp.status_code == 200
    text = prio_resp.text
    assert f"/runs/{ui_run.RUN_ID}/review/80002" in text
    assert "sort=priority" in text
    assert "에이전트 3단 논증" in text


def test_sample_page_has_metrology_ruler_argumentation_and_timing_bars(
    client: TestClient,
) -> None:
    resp = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002")
    assert resp.status_code == 200
    text = resp.text
    assert "점수 임계값 계측 눈금자" in text
    assert "에이전트 트리아지 &amp; 4단 논증 영수증" in text
    assert "1. 가설" in text
    assert "4. 작업자 지침" in text
    assert "단계별 시간 (ms)" in text


def test_mask_viewer_includes_superimpose_loupe(client: TestClient) -> None:
    resp = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002/mask")
    assert resp.status_code == 200
    text = resp.text
    assert "오버레이 중첩 비교기" in text
    assert "loupe-alpha" in text


def test_compare_page_shows_delta_column(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, raw_root, _ = env
    other_id = "20261112T000100Z-a762ba63-tune"
    ui_run.make_run(runs_dir, raw_root, run_id=other_id, samples=ui_run.DEFAULT_SAMPLES[:1])
    resp = client.get("/compare", params={"base": ui_run.RUN_ID, "target": other_id})
    assert resp.status_code == 200
    assert "Δ (target - base)" in resp.text


def test_watchdog_and_inspections_without_db_do_not_create_db(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    wd = client.get("/watchdog")
    assert wd.status_code == 200
    assert "Watchdog 무결성 감사" in wd.text
    assert "G_CONFIG_FROZEN" in wd.text
    assert "PSI 분포 드리프트 감시" in wd.text
    assert not (runs_dir / "records.db").exists()

    insp = client.get("/inspections")
    assert insp.status_code == 200
    assert "검사 기록 DB가 없습니다" in insp.text
    assert not (runs_dir / "records.db").exists()


def test_inspections_explorer_and_detail_with_seeded_db(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    iid = _seed_cross_evidence_db(runs_dir)

    listing = client.get("/inspections")
    assert listing.status_code == 200
    assert "cross_001" in listing.text
    assert "STATUS_MISMATCH" in listing.text

    detail = client.get(f"/inspections/{iid}")
    assert detail.status_code == 200
    text = detail.text
    assert "엔진별 원본 관찰 비교" in text
    assert "yolo-det" in text
    assert "능동 패치 중재 결론" in text
    assert "에이전트 4단 논증 영수증" in text

    ov = client.get(f"/inspections/{iid}/overlay.png")
    assert ov.status_code == 200
    assert ov.headers["content-type"] == "image/png"


def test_inspections_live_workbench_when_api_enabled(env: tuple[Path, Path, Path]) -> None:
    runs_dir, raw_root, _ = env
    app = create_app(runs_dir=runs_dir, raw_root=raw_root, enable_api=True)
    c = TestClient(app)
    page = c.get("/inspections")
    assert page.status_code == 200
    assert "실시간 이미지 검사 워크벤치" in page.text
    assert "live-inspect-form" in page.text


def test_cli_serve_with_api_flag_passes_enable_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["serve", "--runs-dir", str(runs), "--with-api"]) == 0
    assert captured["app"].state.api_enabled is True


def test_all_upgraded_pages_have_no_external_urls(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    iid = _seed_cross_evidence_db(runs_dir)
    paths = [
        "/",
        f"/runs/{ui_run.RUN_ID}",
        f"/runs/{ui_run.RUN_ID}/triage",
        f"/runs/{ui_run.RUN_ID}/samples/70002",
        f"/runs/{ui_run.RUN_ID}/samples/70002/mask",
        f"/runs/{ui_run.RUN_ID}/review",
        f"/runs/{ui_run.RUN_ID}/review?sort=priority",
        f"/runs/{ui_run.RUN_ID}/metrics",
        "/stats",
        "/compare",
        "/watchdog",
        "/inspections",
        f"/inspections/{iid}",
    ]
    for path in paths:
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "http://" not in resp.text, path
        assert "https://" not in resp.text, path


def test_wave3b_counterfactual_hotspot_and_watchdog_calibration(client: TestClient) -> None:
    # 1) sample.html 2D 등위선 위상도 & 반사실적 경계 해석
    resp_sample = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002")
    assert resp_sample.status_code == 200
    stxt = resp_sample.text
    assert "2D (면적 × 평균대비) 등위선 위상도" in stxt
    assert "id=\"phase-diagram-svg\"" in stxt
    assert "id=\"cf-table\"" in stxt
    assert "도달 불능(면적 병목)" in stxt
    assert "http://" not in stxt
    assert "https://" not in stxt

    # 2) metrics.html 2D 라인스캔 스트립 공간 핫스팟 (기존 결정 완료 표 유지 포함)
    resp_metrics = client.get(f"/runs/{ui_run.RUN_ID}/metrics")
    assert resp_metrics.status_code == 200
    mtxt = resp_metrics.text
    assert "2D 라인스캔 스트립 공간 핫스팟 지도" in mtxt
    assert "NG 샘플 후보 밀도" in mtxt
    assert "REVIEW 보류 후보 밀도" in mtxt
    assert "좌·우 가장자리(15%) 집중 비율" in mtxt
    assert "<th>결정 완료</th>" in mtxt
    assert "http://" not in mtxt

    # 3) watchdog.html Phase 4 캘리브레이션: <title>이 아닌 <main> 본문에 렌더링되는지 검증
    resp_wd_empty = client.get(f"/watchdog?run_id={ui_run.RUN_ID}")
    assert resp_wd_empty.status_code == 200
    empty_title = resp_wd_empty.text.split("<title>", 1)[1].split("</title>", 1)[0]
    empty_main = resp_wd_empty.text.split("<main>", 1)[1].split("</main>", 1)[0]
    assert "4. Phase 4 사람-기계 피드백 캘리브레이션" not in empty_title
    assert "4. Phase 4 사람-기계 피드백 캘리브레이션" in empty_main
    assert "아직 기록된 작업자 검토 결정이 없습니다" in empty_main

    post = client.post(
        f"/runs/{ui_run.RUN_ID}/review/70002",
        data={
            "final_status": "NG",
            "reason_code": "CONFIRMED_DEFECT",
            "reviewer": "qa",
            "note": "Wave 3B 캘리브레이션 테스트",
            "return_to": "review",
        },
        follow_redirects=False,
    )
    assert post.status_code == 303

    resp_wd_full = client.get(f"/watchdog?run_id={ui_run.RUN_ID}")
    assert resp_wd_full.status_code == 200
    wtxt = resp_wd_full.text
    full_title = wtxt.split("<title>", 1)[1].split("</title>", 1)[0]
    full_main = wtxt.split("<main>", 1)[1].split("</main>", 1)[0]
    assert "4. Phase 4 사람-기계 피드백 캘리브레이션" not in full_title
    assert "4. Phase 4 사람-기계 피드백 캘리브레이션" in full_main
    assert "결정 완료 <strong>1</strong>" in full_main
    assert "CONFIRMED_DEFECT" in full_main
    assert "id=\"theme-toggle\"" in wtxt
    assert "http://" not in wtxt
