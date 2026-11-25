"""검사 신원(`EngineSpec`) 단위 테스트 — 경로 독립 해시, provenance 완결성 판정, 불변성 검증.

fingerprint 가 신원 필드별로 분리되는지, provenance_complete 판정, opencv_spec,
그리고 onnx_path 가 fingerprint 에 영향을 주지 않는지(바이트 SHA 로 신원 고정) 검증.
"""

from __future__ import annotations

from surface_sentinel.records.fingerprint import inspection_fingerprint, policy_hash
from surface_sentinel.records.identity import EngineSpec, opencv_spec

CFG = "a" * 64
MODEL = "b" * 64


def test_spec_fingerprint_matches_direct_function() -> None:
    spec = EngineSpec(
        engine="yolo-seg",
        task="seg",
        model_sha256=MODEL,
        config_hash=CFG,
        yolo_policy_hash=policy_hash(0.5, 0.25),
    )
    assert spec.fingerprint == inspection_fingerprint(
        engine="yolo-seg",
        task="seg",
        model_sha256=MODEL,
        config_hash=CFG,
        yolo_policy_hash=policy_hash(0.5, 0.25),
    )


def test_onnx_path_does_not_change_fingerprint() -> None:
    a = EngineSpec(
        engine="yolo-det", task="det", model_sha256=MODEL, config_hash=CFG, onnx_path="/x/best.onnx"
    )
    b = EngineSpec(
        engine="yolo-det", task="det", model_sha256=MODEL, config_hash=CFG, onnx_path="/y/best.onnx"
    )
    assert a.fingerprint == b.fingerprint


def test_spec_separates_identity_fields() -> None:
    base = EngineSpec(engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG)
    variants = {
        base.fingerprint,
        EngineSpec(
            engine="yolo-det", task="det", model_sha256=MODEL, config_hash=CFG
        ).fingerprint,
        EngineSpec(
            engine="yolo-seg", task="seg", model_sha256="c" * 64, config_hash=CFG
        ).fingerprint,
        EngineSpec(
            engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash="d" * 64
        ).fingerprint,
        EngineSpec(
            engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG, code_sha="f" * 40
        ).fingerprint,
        EngineSpec(
            engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG, code_dirty=True
        ).fingerprint,
    }
    assert len(variants) == 6


def test_provenance_complete_rules() -> None:
    assert opencv_spec(CFG).provenance_complete is True
    assert EngineSpec(engine="yolo-seg", task="seg", config_hash=CFG).provenance_complete is False
    complete = EngineSpec(engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG)
    assert complete.provenance_complete is True


def test_opencv_spec_defaults() -> None:
    spec = opencv_spec(CFG)
    assert spec.engine == "opencv"
    assert spec.task == ""
    assert spec.model_sha256 == ""
    assert spec.yolo_policy_hash == ""
    assert spec.combine_policy_hash == ""
    assert spec.code_sha == ""
    assert spec.code_dirty is False


def test_spec_is_frozen() -> None:
    import dataclasses

    import pytest

    spec = opencv_spec(CFG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.engine = "yolo-det"  # type: ignore[misc]
