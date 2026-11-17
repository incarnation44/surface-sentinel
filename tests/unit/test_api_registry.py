"""API 레지스트리 주입 검증 (감사 P1-1, ADR 0008)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import surface_sentinel.api.app as app_mod
from surface_sentinel.api.schemas import EngineName


@pytest.fixture(autouse=True)
def _stable_code_identity(monkeypatch):
    """앱 생성이 실제 저장소 git 상태를 읽지 않게 고정한다(dirty 무관)."""

    from surface_sentinel.evaluation import recalibration

    monkeypatch.setattr(recalibration, "git_code_identity", lambda: ("test-sha", False))


def _yolo_engine():
    return next(e for e in EngineName if e is not EngineName.OPENCV)


def test_inspect_forwards_registry(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_entry(image_path, config, **kw):
        seen.update(kw)
        return SimpleNamespace(status="OK")

    monkeypatch.setattr(app_mod, "_load_yolo_entry", lambda name: fake_entry)
    out = app_mod._inspect(
        Path("x.png"), None, engine=_yolo_engine(),
        output_dir=Path("out"), label="good", split="val", registry=sentinel,
    )
    assert seen["registry"] is sentinel
    assert out.status == "OK"


def test_create_app_sets_registry(tmp_path):
    app = app_mod.create_inspection_app(runs_dir=tmp_path / "runs")
    reg = getattr(app.state, "yolo_registry", None)
    if reg is not None:  # YOLO 실행 가능 환경
        assert reg.cold_starts == 0
        assert reg.size == 0


def test_create_app_registry_none_without_yolo(monkeypatch, tmp_path):
    import importlib

    real = importlib.import_module

    def blocked(name, *a, **kw):
        if name == "surface_sentinel.yolo.runtime":
            raise ImportError("no onnxruntime")
        return real(name, *a, **kw)

    monkeypatch.setattr(app_mod.importlib, "import_module", blocked)
    app = app_mod.create_inspection_app(runs_dir=tmp_path / "runs")
    assert app.state.yolo_registry is None
