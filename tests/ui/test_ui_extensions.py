"""로컬 웹 UI 확장 라우트 테스트 — DB 통계(`/stats`), run 비교(`/compare`), 마스크 뷰어 검증.

기존 ``tests/ui`` 패턴을 따른다(합성 run + 마크업 검증). DB는 ``tmp_path``
아래에만 만들고, 읽기 전용 페이지가 DB 파일을 새로 만들지 않는지도 고정한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ui_run
from surface_sentinel.contracts import ReasonCode, SampleResult, Status
from surface_sentinel.records.db import connect, init_db
from surface_sentinel.records.store import insert_inspection
from surface_sentinel.ui import create_app

OTHER_RUN_ID = "20261112T000100Z-a762ba63-tune"


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


def seed_stats(runs_dir: Path) -> None:
    """통계 페이지용 기록 3건(엔진 2종)을 DB에 넣는다."""

    specs = (
        ("stats-a", Status.OK, "opencv", 0.0),
        ("stats-b", Status.NG, "opencv", 1.2),
        ("stats-c", Status.REVIEW, "yolo-det", 0.3),
    )
    connection = connect(runs_dir / "records.db")
    try:
        init_db(connection)
        for text, status, engine, score in specs:
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            result = SampleResult(
                run_id="seed-run",
                sample_id=f"seed_{sha[:8]}",
                original_id="seed",
                split="tune",
                label="good",
                status=status,
                score=score,
                reasons=(ReasonCode.NO_DEFECT_EVIDENCE,),
                config_hash="0" * 64,
            )
            insert_inspection(connection, result, engine=engine, image_sha256=sha)
    finally:
        connection.close()


def test_sample_page_links_mask_viewer(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002")
    assert response.status_code == 200
    assert f"/runs/{ui_run.RUN_ID}/samples/70002/mask" in response.text


def test_mask_viewer_toggles(client: TestClient) -> None:
    response = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002/mask")
    assert response.status_code == 200
    text = response.text
    assert "마스크 뷰어" in text
    assert text.count('class="mask-toggle"') == 5
    for key in ("original", "roi", "candidate", "refined", "overlay"):
        assert f'data-layer="{key}"' in text
    # fixture는 response/candidates 단계 이미지만 저장한다.
    assert f"/runs/{ui_run.RUN_ID}/samples/70002/candidates.png" in text
    assert "ROI" in text
    assert text.count("산출물 없음</div>") >= 1


def test_mask_viewer_shows_roi_and_refined_when_present(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    _, _, run_dir = env
    for kind in ("roi", "refined"):
        (run_dir / "stages" / f"70002_{kind}.png").write_bytes(
            (run_dir / "stages" / "70002_candidates.png").read_bytes()
        )
    text = client.get(f"/runs/{ui_run.RUN_ID}/samples/70002/mask").text
    assert f"/runs/{ui_run.RUN_ID}/samples/70002/roi.png" in text
    assert f"/runs/{ui_run.RUN_ID}/samples/70002/refined.png" in text
    assert text.count("산출물 없음</div>") == 0


def test_mask_viewer_unknown_sample(client: TestClient) -> None:
    assert client.get(f"/runs/{ui_run.RUN_ID}/samples/99999/mask").status_code == 404


def test_stats_without_db_does_not_create_one(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    response = client.get("/stats")
    assert response.status_code == 200
    assert "검사 기록 DB가 없습니다" in response.text
    assert not (runs_dir / "records.db").exists()


def test_stats_reports_distribution_and_engines(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    seed_stats(runs_dir)

    response = client.get("/stats")
    assert response.status_code == 200
    text = response.text
    assert "검사 통계" in text
    assert "상태 분포" in text
    assert "분모 3" in text
    assert "0.3333" in text
    assert "엔진별 비교" in text
    assert "yolo-det" in text
    assert "run별 요약" in text
    # 기존 run 지표로 이어지는 링크를 유지한다.
    assert f"/runs/{ui_run.RUN_ID}/metrics" in text


def test_compare_page_lists_runs(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, raw_root, _ = env
    ui_run.make_run(runs_dir, raw_root, run_id=OTHER_RUN_ID, samples=ui_run.DEFAULT_SAMPLES[:1])
    response = client.get("/compare")
    assert response.status_code == 200
    assert ui_run.RUN_ID in response.text
    assert OTHER_RUN_ID in response.text


def test_compare_metrics_side_by_side(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, raw_root, _ = env
    ui_run.make_run(runs_dir, raw_root, run_id=OTHER_RUN_ID, samples=ui_run.DEFAULT_SAMPLES[:1])

    response = client.get("/compare", params={"base": ui_run.RUN_ID, "target": OTHER_RUN_ID})
    assert response.status_code == 200
    text = response.text
    assert "run 비교" in text
    assert "defect_escape_rate" in text
    # base run: escape 1/3, target run: 결함 이미지가 없어 분모 0.
    assert "(분모 3)" in text
    assert "(분모 0)" in text
    assert "혼동행렬" in text
    assert "good" in text and "defect" in text


def test_compare_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/compare", params={"base": "20261112T000000Z-deadbeef-tune"})
    assert response.status_code == 404


def test_compare_partial_selection_shows_form(client: TestClient) -> None:
    response = client.get("/compare", params={"base": ui_run.RUN_ID})
    assert response.status_code == 200
    assert "두 run을 선택하면" in response.text


def test_runs_index_links_stats_and_compare(client: TestClient) -> None:
    text = client.get("/").text
    assert 'href="/stats"' in text
    assert 'href="/compare"' in text


def test_inspection_overlay_handles_corrupted_db_without_500(
    env: tuple[Path, Path, Path],
) -> None:
    runs_dir, raw_root, _ = env
    (runs_dir / "records.db").write_bytes(b"not-a-valid-sqlite-database")
    client = TestClient(
        create_app(runs_dir=runs_dir, raw_root=raw_root),
        raise_server_exceptions=False,
    )
    response = client.get("/inspections/1/overlay.png")
    assert response.status_code == 404


def test_inspection_detail_and_readonly_connections_handle_corrupt_db_and_busy_timeout(
    env: tuple[Path, Path, Path],
) -> None:
    import json

    from surface_sentinel.agent.audit import AuditVerdict, audit_records_db
    from surface_sentinel.ui.data import (
        UiInputError,
        read_only_connection,
        row_overlay_path,
        safe_child,
    )

    runs_dir, raw_root, run_dir = env
    seed_stats(runs_dir)
    db_path = runs_dir / "records.db"

    # 1) read_only_connection이 connect_readonly(busy_timeout=5000, query_only=1)를 사용한다.
    ro = read_only_connection(db_path)
    try:
        assert int(ro.execute("PRAGMA busy_timeout").fetchone()[0]) == 5000
        assert int(ro.execute("PRAGMA query_only").fetchone()[0]) == 1
    finally:
        ro.close()

    # 2) safe_child NUL 바이트 거부 및 row_overlay_path 비-PNG/NUL 거부
    with pytest.raises(UiInputError):
        safe_child(runs_dir, "bad\x00path.png")
    assert row_overlay_path({"overlay_path": str(db_path)}, runs_dir) is None
    assert row_overlay_path({"overlay_path": "bad\x00.png"}, runs_dir) is None

    # 3) DB JSON 컬럼 손상 시 /inspections/{id}가 500 대신 404(UiInputError)를 반환한다.
    conn = connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO inspections ("
                "run_id, sample_id, image_sha256, inspection_fingerprint, label, split,"
                " status, score, reasons, engine, task, config_hash, model_sha256,"
                " model_version, provenance_complete, overlay_path, timing_ms, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "bad-json-run",
                    "s_bad",
                    "f" * 64,
                    "1" * 64,
                    "good",
                    "tune",
                    "OK",
                    0.1,
                    "{corrupt-json",
                    "opencv",
                    "",
                    "0" * 64,
                    "",
                    "opencv",
                    1,
                    "",
                    "{}",
                    "2026-01-01T00:00:00Z",
                ),
            )
            bad_id = int(cur.lastrowid or 0)
    finally:
        conn.close()

    client = TestClient(
        create_app(runs_dir=runs_dir, raw_root=raw_root),
        raise_server_exceptions=False,
    )
    assert client.get(f"/inspections/{bad_id}").status_code == 404

    # 4) 손상된 DB 파일에서도 /inspections/1 -> 404, audit_records_db -> REJECT
    db_path.write_bytes(b"corrupted-sqlite-bytes")
    assert client.get("/inspections/1").status_code == 404
    assert audit_records_db(db_path).verdict is AuditVerdict.REJECT

    # 5) 손상된 run.json(비-dict counts) 및 results.jsonl(키 누락)에서도 500 금지
    run_json = run_dir / "run.json"
    rj = json.loads(run_json.read_text(encoding="utf-8"))
    rj["counts"] = ["not-a-dict"]
    run_json.write_text(json.dumps(rj), encoding="utf-8")
    assert client.get("/").status_code == 200
    assert client.get(f"/runs/{ui_run.RUN_ID}").status_code == 404
    assert client.get("/watchdog").status_code == 404


def test_inspections_export_csv_without_db_does_not_create_db(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    runs_dir, _, _ = env
    resp = client.get("/inspections/export.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.content.startswith(b"\xef\xbb\xbf")
    text = resp.content.decode("utf-8-sig")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines == [
        "id,sample_id,engine,status,score,reasons,label,split,config_hash,created_at"
    ]
    assert not (runs_dir / "records.db").exists()


def test_inspections_export_csv_with_seeded_db_and_status_filter(
    client: TestClient, env: tuple[Path, Path, Path]
) -> None:
    import csv
    import io

    runs_dir, _, _ = env
    seed_stats(runs_dir)

    page = client.get("/inspections")
    assert page.status_code == 200
    assert 'href="/inspections/export.csv"' in page.text
    assert 'href="/inspections/export.csv?status=NG"' in page.text

    all_resp = client.get("/inspections/export.csv")
    assert all_resp.status_code == 200
    assert all_resp.content.startswith(b"\xef\xbb\xbf")
    all_rows = list(csv.DictReader(io.StringIO(all_resp.content.decode("utf-8-sig"))))
    assert len(all_rows) == 3
    assert [row["status"] for row in all_rows] == ["REVIEW", "NG", "OK"]

    ng_resp = client.get("/inspections/export.csv", params={"status": "NG"})
    assert ng_resp.status_code == 200
    assert "inspections_ng.csv" in ng_resp.headers.get("content-disposition", "")
    ng_rows = list(csv.DictReader(io.StringIO(ng_resp.content.decode("utf-8-sig"))))
    assert len(ng_rows) == 1
    assert ng_rows[0]["status"] == "NG"
    assert ng_rows[0]["engine"] == "opencv"


def test_inspection_demo_samples_direct_train_dir_and_gt_exclusion(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    raw_root = tmp_path / "kolektorsdd2"
    train_dir = raw_root / "train"
    train_dir.mkdir(parents=True)

    png_stub = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    (train_dir / "demo_01.png").write_bytes(png_stub)
    (train_dir / "demo_02.png").write_bytes(png_stub)
    (train_dir / "demo_01_GT.png").write_bytes(png_stub)

    client = TestClient(
        create_app(runs_dir=runs_dir, raw_root=raw_root, repo_root=tmp_path)
    )
    resp = client.get("/inspections/demo-samples", params={"limit": 20})
    assert resp.status_code == 200
    assert resp.json() == {"samples": ["demo_01.png", "demo_02.png"]}

    img_resp = client.get("/inspections/demo-samples/demo_01.png")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/png")

    gt_resp = client.get("/inspections/demo-samples/demo_01_GT.png")
    assert gt_resp.status_code == 404

    missing_client = TestClient(
        create_app(runs_dir=runs_dir, raw_root=tmp_path / "empty_raw", repo_root=tmp_path)
    )
    missing_resp = missing_client.get("/inspections/demo-samples", params={"limit": 20})
    assert missing_resp.status_code == 200
    assert missing_resp.json() == {"samples": [], "detail": "샘플 폴더를 찾을 수 없습니다."}


def test_inspection_detail_renders_korean_review_reason_summary(
    env: tuple[Path, Path, Path],
) -> None:
    runs_dir, raw_root, _ = env
    db_path = runs_dir / "records.db"
    connection = connect(db_path)
    try:
        init_db(connection)
        sha = hashlib.sha256(b"review-weak-sample").hexdigest()
        result = SampleResult(
            run_id="live-demo",
            sample_id="weak_01",
            original_id="weak_01",
            split="tune",
            label="unknown",
            status=Status.REVIEW,
            score=0.383909,
            reasons=(ReasonCode.MULTIPLE_WEAK_CANDIDATES,),
            config_hash="0" * 64,
        )
        iid = insert_inspection(connection, result, engine="opencv", image_sha256=sha)
    finally:
        connection.close()

    client = TestClient(create_app(runs_dir=runs_dir, raw_root=raw_root))
    resp = client.get(f"/inspections/{iid}")
    assert resp.status_code == 200
    text = resp.text
    assert "판정 근거 요약" in text
    assert "다수 약한 후보 동시 검출" in text
    assert "0.50" in text
    assert "MULTIPLE_WEAK_CANDIDATES" in text




def test_inspection_demo_samples_random_with_labels_from_tune_csv(tmp_path: Path) -> None:
    """tune.csv 가 있으면 20장씩 중복 없이 무작위로 소진하고, 다 쓰면 빈 목록 + 안내를 준다."""

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    raw_root = tmp_path / "kolektorsdd2"
    train_dir = raw_root / "train"
    train_dir.mkdir(parents=True)
    png_stub = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    rows = ["sample_id,image_path,label,source_split"]
    for i in range(60):
        name = f"{10000 + i}.png"
        (train_dir / name).write_bytes(png_stub)
        rows.append(f"{10000 + i},train/{name},{'defect' if i < 20 else 'good'},train")
    splits = tmp_path / "data" / "splits"
    splits.mkdir(parents=True)
    (splits / "tune.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    client = TestClient(create_app(runs_dir=runs_dir, raw_root=raw_root, repo_root=tmp_path))
    seen: list[str] = []
    first = client.get("/inspections/demo-samples", params={"limit": 20}).json()
    assert len(first["samples"]) == 20 and first["total"] == 60 and first["remaining"] == 40
    assert set(first["labels"]) == set(first["samples"])
    assert sum(1 for v in first["labels"].values() if v == "defect") == 5
    seen += first["samples"]
    for _ in range(2):
        body = client.get("/inspections/demo-samples", params={"limit": 20}).json()
        seen += body["samples"]
    assert len(seen) == 60 and len(set(seen)) == 60

    done = client.get("/inspections/demo-samples", params={"limit": 20}).json()
    assert done["samples"] == [] and done["remaining"] == 0
    assert "모두 검사했습니다" in done["detail"]

    assert client.post("/inspections/demo-samples/reset").json() == {"reset": True}
    again = client.get("/inspections/demo-samples", params={"limit": 20}).json()
    assert len(again["samples"]) == 20


def test_explain_verdict_reports_values_against_thresholds() -> None:
    """설명 표는 저장된 후보 수치를 기준값과 나란히 놓고, 판정을 다시 내리지 않는다."""

    from surface_sentinel.config import default_config
    from surface_sentinel.ui.verdict_explain import explain_verdict

    cfg = default_config()
    comp = {"area_px": 767, "mean_contrast_gray": 122.0, "evidence": 1.1992,
            "touches_roi_border": False}
    ng = explain_verdict({"status": "NG", "score": 1.1992, "reasons": ["DEFECT_EVIDENCE_STRONG"],
                          "components": [comp]}, cfg)
    assert "1.199" in ng["headline"] and "0.50" in ng["headline"]
    assert ng["rules"][0]["hit"] is True
    top = ng["candidates"][0]
    assert top["contrast_capped"] is True and abs(top["contrast_norm"] - 1.5) < 1e-9

    weak = [{"area_px": a, "mean_contrast_gray": 27.0, "evidence": e, "touches_roi_border": False}
            for a, e in ((96, 0.386), (73, 0.326))]
    rv = explain_verdict({"status": "REVIEW", "score": 0.386,
                          "reasons": ["MULTIPLE_WEAK_CANDIDATES"], "components": weak}, cfg)
    assert "2개" in rv["headline"] and "0.114" in rv["headline"]
    assert any(r["hit"] and r["effect"] == "REVIEW" for r in rv["rules"])
