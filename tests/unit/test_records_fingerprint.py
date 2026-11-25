"""검사 지문(`inspection_fingerprint`) 단위 테스트 — 엔진·모델·설정·정책·코드별 해시 분리 검증.

결정론·신원 필드별 분리·파일/정책 해시를 검증한다. DB 없이 순수 함수만 다룬다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from surface_sentinel.records.fingerprint import (
    COMBINE_POLICY_VERSION,
    PREPROCESSING_VERSION,
    canonical_json_sha256,
    file_sha256,
    inspection_fingerprint,
    policy_hash,
)

CFG = "a" * 64
MODEL = "b" * 64


def test_fingerprint_is_deterministic_64hex() -> None:
    first = inspection_fingerprint(engine="opencv", config_hash=CFG)
    second = inspection_fingerprint(engine="opencv", config_hash=CFG)
    assert first == second
    assert len(first) == 64
    assert all(ch in "0123456789abcdef" for ch in first)


def test_fingerprint_separates_engine() -> None:
    opencv = inspection_fingerprint(engine="opencv", config_hash=CFG)
    yolo = inspection_fingerprint(
        engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG
    )
    assert opencv != yolo


def test_fingerprint_separates_model_sha() -> None:
    a = inspection_fingerprint(engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG)
    b = inspection_fingerprint(
        engine="yolo-seg", task="seg", model_sha256="c" * 64, config_hash=CFG
    )
    assert a != b


def test_fingerprint_separates_config_and_task() -> None:
    base = inspection_fingerprint(
        engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash=CFG
    )
    other_cfg = inspection_fingerprint(
        engine="yolo-seg", task="seg", model_sha256=MODEL, config_hash="d" * 64
    )
    other_task = inspection_fingerprint(
        engine="yolo-det", task="det", model_sha256=MODEL, config_hash=CFG
    )
    assert base != other_cfg
    assert base != other_task


def test_fingerprint_separates_policy_and_code() -> None:
    base = inspection_fingerprint(engine="opencv", config_hash=CFG)
    pol = inspection_fingerprint(
        engine="opencv", config_hash=CFG, yolo_policy_hash=policy_hash(0.5, 0.25)
    )
    code = inspection_fingerprint(engine="opencv", config_hash=CFG, code_sha="f" * 40)
    dirty = inspection_fingerprint(engine="opencv", config_hash=CFG, code_dirty=True)
    assert len({base, pol, code, dirty}) == 4


def test_canonical_json_key_order_independent() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})


def test_file_sha256_matches_bytes(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"surface-sentinel")
    assert file_sha256(path) == hashlib.sha256(b"surface-sentinel").hexdigest()


def test_policy_hash_deterministic_and_sensitive() -> None:
    assert policy_hash(0.5, 0.25) == policy_hash(0.5, 0.25)
    assert policy_hash(0.5, 0.25) != policy_hash(0.6, 0.25)
    assert policy_hash(0.5, 0.25) != policy_hash(0.5, 0.30)


def test_version_constants_are_nonempty() -> None:
    assert PREPROCESSING_VERSION
    assert COMBINE_POLICY_VERSION