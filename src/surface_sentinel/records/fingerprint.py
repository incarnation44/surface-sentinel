"""검사 조건 지문(`inspection_fingerprint`) SHA-256 계산 — 동일 이미지의 엔진·설정별 중복 구분.

중복제거 키를 ``image_sha256`` 단독에서 ``(image_sha256, inspection_fingerprint)``
쌍으로 바꾼다. 같은 이미지라도 엔진·task·모델·설정·정책·코드 버전이 다르면
다른 검사로 기록해야 엔진별 관찰이 서로 덮이거나 잘못 재사용되지 않는다.

fingerprint 는 이미지 SHA 를 **포함하지 않는다**(별도 컬럼으로 UNIQUE 쌍을 이루므로
중복). 나머지 검사 신원(identity) 필드를 canonical JSON 으로 직렬화해 SHA256 한다.

결정론: ``sort_keys`` + 고정 구분자 + ``ensure_ascii`` 로 직렬화하므로 입력 순서·
플랫폼과 무관하게 같은 값이 나온다. 이 모듈은 subprocess/네트워크를 쓰지 않는다
(``code_sha`` 는 진입점에서 1회 계산해 주입한다 — 워커 핫패스 subprocess 금지).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: 전처리(letterbox·NMS·입력 변환) 버전. ``yolo/inference.py`` 전처리 변경 시 bump.
PREPROCESSING_VERSION = "preprocess-v1"
#: 결합 규칙 버전. ``yolo/combine.py`` 의 NG > REVIEW > ERROR > OK 규칙이 바뀌면 bump.
COMBINE_POLICY_VERSION = "combine-v1"


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """canonical JSON(sort_keys·compact·ensure_ascii)의 SHA256 16진수."""

    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """파일 바이트의 SHA256. 모델 ONNX 의 실제 신원을 기록할 때 쓴다."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_hash(ng_conf: float, review_lo: float) -> str:
    """YOLO 판정 정책(ng_conf·review_lo)의 canonical 해시."""

    return canonical_json_sha256({"ng_conf": float(ng_conf), "review_lo": float(review_lo)})


def inspection_fingerprint(
    *,
    engine: str,
    task: str = "",
    model_sha256: str = "",
    config_hash: str = "",
    yolo_policy_hash: str = "",
    combine_policy_hash: str = "",
    preprocessing_version: str = PREPROCESSING_VERSION,
    code_sha: str = "",
    code_dirty: bool = False,
    recheck_tag: str = "",
) -> str:
    """검사 신원 필드의 canonical SHA256. 이미지 SHA 는 포함하지 않는다.

    ``recheck_tag`` 가 비어 있으면 기존 canonical 페이로드를 그대로 유지한다
    (하위 호환). ``ERROR`` 재검사처럼 별도 append-only 행이 필요할 때만
    비어 있지 않은 ``recheck_tag`` 를 페이로드에 포함한다.
    """

    payload: dict[str, Any] = {
        "engine": str(engine),
        "task": str(task),
        "model_sha256": str(model_sha256),
        "config_hash": str(config_hash),
        "yolo_policy_hash": str(yolo_policy_hash),
        "combine_policy_hash": str(combine_policy_hash),
        "preprocessing_version": str(preprocessing_version),
        "code_sha": str(code_sha),
        "code_dirty": bool(code_dirty),
    }
    tag = str(recheck_tag).strip()
    if tag:
        payload["recheck_tag"] = tag
    return canonical_json_sha256(payload)


__all__ = [
    "COMBINE_POLICY_VERSION",
    "PREPROCESSING_VERSION",
    "canonical_json_sha256",
    "file_sha256",
    "inspection_fingerprint",
    "policy_hash",
]