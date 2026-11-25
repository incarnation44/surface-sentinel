"""10단계 OpenCV 파이프라인(`inspect_image`) 통합 테스트.

정상·어두운/밝은 결함·경계 접촉·약후보 등 합성 fixture별 판정과 사유 코드를 검증한다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import synthetic as syn
from surface_sentinel.config import REVIEW_BAND_MAX, AppConfig, load_config
from surface_sentinel.contracts import (
    STAGE_ORDER,
    ErrorCode,
    ReasonCode,
    Stage,
    Status,
)
from surface_sentinel.pipeline import inspect_image
from surface_sentinel.pipeline.evidence import sanitize_sample_id

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.toml"
SYNTHETIC_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"

BASE_CONFIG = load_config(BASE_CONFIG_PATH)


def _run(
    tmp_path: Path,
    image_path: Path,
    *,
    config: AppConfig = BASE_CONFIG,
    mask_path: Path | None = None,
    label: str = "unknown",
    split: str = "tune",
    output_name: str = "out",
    **kwargs: object,
):
    return inspect_image(
        image_path,
        config,
        mask_path=mask_path,
        label=label,
        split=split,
        output_dir=tmp_path / output_name,
        **kwargs,
    )


def _write(tmp_path: Path, name: str, image: np.ndarray) -> Path:
    return syn.write_png(tmp_path / "inputs" / name, image)


def _dark_fixture() -> tuple[Path, Path]:
    return (
        SYNTHETIC_DIR / "dark_defect.png",
        SYNTHETIC_DIR / "dark_defect_mask.png",
    )


def test_persistent_fixtures_match_builders() -> None:
    def read(path: Path) -> np.ndarray:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    dark, dark_mask = syn.dark_defect_image()
    bright, bright_mask = syn.bright_defect_image()
    assert np.array_equal(read(SYNTHETIC_DIR / "good.png"), syn.good_image())
    assert np.array_equal(read(SYNTHETIC_DIR / "dark_defect.png"), dark)
    assert np.array_equal(read(SYNTHETIC_DIR / "dark_defect_mask.png"), dark_mask)
    assert np.array_equal(read(SYNTHETIC_DIR / "bright_defect.png"), bright)
    assert np.array_equal(read(SYNTHETIC_DIR / "bright_defect_mask.png"), bright_mask)


def test_good_fixture_is_ok(tmp_path: Path) -> None:
    outcome = _run(tmp_path, SYNTHETIC_DIR / "good.png")

    assert outcome.result.status is Status.OK
    assert outcome.result.score == 0.0
    assert outcome.result.reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)
    assert outcome.result.components == ()
    assert Path(outcome.result.overlay_path).is_file()
    assert outcome.debug is not None


def test_dark_defect_fixture_is_ng(tmp_path: Path) -> None:
    image, mask = _dark_fixture()
    outcome = _run(tmp_path, image, mask_path=mask, label="defect")

    assert outcome.result.status is Status.NG
    assert outcome.result.score >= BASE_CONFIG.scoring.ng_min
    assert outcome.result.reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)
    assert outcome.result.components
    assert outcome.result.components[0].mean_contrast_gray > 12.0


def test_bright_defect_fixture_is_ng(tmp_path: Path) -> None:
    outcome = _run(
        tmp_path,
        SYNTHETIC_DIR / "bright_defect.png",
        mask_path=SYNTHETIC_DIR / "bright_defect_mask.png",
        label="defect",
    )

    assert outcome.result.status is Status.NG
    assert outcome.result.reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)


def test_low_contrast_defect_is_ok(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=301))
    image = syn.add_dark_rect(image, 150, 50, 12, 12, 112)
    path = _write(tmp_path, "low_contrast.png", image)

    outcome = _run(tmp_path, path)

    assert outcome.result.status is Status.OK
    assert outcome.result.reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)
    assert outcome.result.components == ()


def test_border_touching_candidate_is_review(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=31))
    image = syn.add_dark_rect(image, 150, 8, 5, 20, 20)
    path = _write(tmp_path, "border_touch.png", image)

    outcome = _run(tmp_path, path)

    assert outcome.result.status is Status.REVIEW
    assert outcome.result.reasons == (ReasonCode.BORDER_TOUCHING_CANDIDATE,)
    assert outcome.result.score >= REVIEW_BAND_MAX


def test_truncated_candidates_are_review(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=61))
    for y in (25, 55, 85):
        for x in (25, 65, 105, 145, 185, 225, 265):
            image = syn.add_dark_rect(image, x, y, 8, 8, 40)
    path = _write(tmp_path, "truncated.png", image)

    outcome = _run(tmp_path, path)

    assert outcome.result.status is Status.REVIEW
    assert outcome.result.reasons == (ReasonCode.CANDIDATE_TRUNCATED,)
    assert len(outcome.result.components) == BASE_CONFIG.candidates.max_candidates


def test_multiple_weak_candidates_are_review(tmp_path: Path) -> None:
    image = syn.to_bgr(syn.surface_gray(seed=51))
    image = syn.add_dark_rect(image, 80, 40, 10, 10, 105)
    image = syn.add_dark_rect(image, 200, 70, 10, 10, 105)
    path = _write(tmp_path, "multi_weak.png", image)

    outcome = _run(tmp_path, path)

    assert outcome.result.status is Status.REVIEW
    assert outcome.result.reasons == (ReasonCode.MULTIPLE_WEAK_CANDIDATES,)
    assert len(outcome.result.components) >= 2


@pytest.mark.parametrize(
    ("error_code", "error_stage"),
    [
        (ErrorCode.E_INPUT_UNIFORM, Stage.VALIDATE_INPUT),
        (ErrorCode.E_INPUT_SIZE, Stage.VALIDATE_INPUT),
        (ErrorCode.E_INPUT_DECODE, Stage.VALIDATE_INPUT),
        (ErrorCode.E_MASK_SHAPE, Stage.VALIDATE_INPUT),
        (ErrorCode.E_MASK_MISMATCH, Stage.VALIDATE_INPUT),
        (ErrorCode.E_INPUT_MISSING, Stage.VALIDATE_INPUT),
    ],
)
def test_input_errors_are_error_status(
    tmp_path: Path,
    error_code: ErrorCode,
    error_stage: Stage,
) -> None:
    if error_code is ErrorCode.E_INPUT_UNIFORM:
        path = _write(
            tmp_path,
            "uniform.png",
            np.full((syn.HEIGHT, syn.WIDTH, 3), 128, dtype=np.uint8),
        )
        outcome = _run(tmp_path, path)
    elif error_code is ErrorCode.E_INPUT_SIZE:
        rng = np.random.default_rng(7)
        path = _write(
            tmp_path,
            "small.png",
            rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
        )
        outcome = _run(tmp_path, path)
    elif error_code is ErrorCode.E_INPUT_DECODE:
        path = tmp_path / "inputs" / "corrupt.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a png")
        outcome = _run(tmp_path, path)
    elif error_code is ErrorCode.E_MASK_SHAPE:
        image_path = _write(tmp_path, "mask_shape.png", syn.dark_defect_image()[0])
        mask_path = _write(
            tmp_path,
            "mask_shape_mask.png",
            np.full((30, 30), 255, dtype=np.uint8),
        )
        outcome = _run(
            tmp_path,
            image_path,
            mask_path=mask_path,
            label="defect",
        )
    elif error_code is ErrorCode.E_MASK_MISMATCH:
        image_path = _write(tmp_path, "mask_mismatch.png", syn.dark_defect_image()[0])
        mask_path = _write(
            tmp_path,
            "mask_mismatch_mask.png",
            np.zeros((syn.HEIGHT, syn.WIDTH), dtype=np.uint8),
        )
        outcome = _run(
            tmp_path,
            image_path,
            mask_path=mask_path,
            label="defect",
        )
    else:
        outcome = _run(tmp_path, tmp_path / "missing.png")

    result = outcome.result
    assert result.status is Status.ERROR
    assert result.error_code is error_code
    assert result.error_stage is error_stage
    assert result.reasons == ()
    assert result.score == 0.0
    assert outcome.debug is None


def test_good_label_with_nonempty_mask_is_mask_mismatch(tmp_path: Path) -> None:
    image_path = _write(tmp_path, "good_label.png", syn.dark_defect_image()[0])
    mask_path = _write(
        tmp_path,
        "good_label_mask.png",
        np.full((syn.HEIGHT, syn.WIDTH), 255, dtype=np.uint8),
    )

    outcome = _run(tmp_path, image_path, mask_path=mask_path, label="good")

    assert outcome.result.status is Status.ERROR
    assert outcome.result.error_code is ErrorCode.E_MASK_MISMATCH


@pytest.mark.parametrize("split", ["test", "official_test"])
def test_sealed_split_is_denied_before_file_access(
    tmp_path: Path,
    split: str,
) -> None:
    outcome = _run(tmp_path, tmp_path / "never-read.png", split=split)

    assert outcome.result.status is Status.ERROR
    assert outcome.result.error_code is ErrorCode.E_SEALED_DENIED
    assert outcome.result.error_stage is Stage.VALIDATE_INPUT
    assert outcome.result.input_sha256 == ""
    assert outcome.result.overlay_path == ""


def test_storage_failure_is_error_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_replace(source: object, destination: object) -> None:
        raise OSError("의도된 저장 실패")

    monkeypatch.setattr(
        "surface_sentinel.pipeline.evidence.os.replace",
        fail_replace,
    )
    image, mask = _dark_fixture()

    outcome = _run(tmp_path, image, mask_path=mask, label="defect")

    result = outcome.result
    assert result.status is Status.ERROR
    assert result.error_code is ErrorCode.E_STORAGE_WRITE
    assert result.error_stage is Stage.SAVE_EVIDENCE
    assert result.overlay_path == ""
    assert list((tmp_path / "out").glob("*_result.json")) == []


def test_inspect_is_deterministic(tmp_path: Path) -> None:
    image, mask = _dark_fixture()

    first = _run(tmp_path, image, mask_path=mask, label="defect", output_name="first")
    second = _run(tmp_path, image, mask_path=mask, label="defect", output_name="second")

    first_payload = first.result.to_dict()
    second_payload = second.result.to_dict()
    first_payload.pop("timing_ms")
    second_payload.pop("timing_ms")
    first_payload.pop("overlay_path")
    second_payload.pop("overlay_path")
    assert first_payload == second_payload

    assert first.debug is not None and second.debug is not None
    for field_name in (
        "gray",
        "roi_mask",
        "response",
        "z_map",
        "candidate_mask",
        "refined_mask",
    ):
        assert np.array_equal(
            getattr(first.debug, field_name),
            getattr(second.debug, field_name),
        )

    first_overlay = cv2.imread(first.result.overlay_path, cv2.IMREAD_UNCHANGED)
    second_overlay = cv2.imread(second.result.overlay_path, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(first_overlay, second_overlay)


def test_overlay_size_and_result_json(tmp_path: Path) -> None:
    image, mask = _dark_fixture()

    outcome = _run(tmp_path, image, mask_path=mask, label="defect")
    result = outcome.result

    overlay = cv2.imread(result.overlay_path, cv2.IMREAD_UNCHANGED)
    assert overlay.shape[:2] == (
        syn.HEIGHT * BASE_CONFIG.artifacts.overlay_scale,
        syn.WIDTH * BASE_CONFIG.artifacts.overlay_scale,
    )

    safe_id = sanitize_sample_id(result.sample_id)
    result_path = Path(result.overlay_path).parent / f"{safe_id}_result.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    expected_timing_keys = {stage.value for stage in STAGE_ORDER} | {"total"}
    assert set(payload["timing_ms"]) == expected_timing_keys
    assert payload["config_hash"] == BASE_CONFIG.config_hash
    assert payload["status"] == "NG"
    assert payload["schema_version"] == "result-v1"
    assert payload["overlay_path"] == result.overlay_path


def test_ng_overrides_quality_flag(tmp_path: Path) -> None:
    image, mask = _dark_fixture()
    config = replace(
        BASE_CONFIG,
        quality=replace(BASE_CONFIG.quality, dark_mean_min=200.0),
    )

    outcome = _run(tmp_path, image, config=config, mask_path=mask, label="defect")

    assert outcome.result.status is Status.NG
    assert outcome.result.reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)
    assert outcome.result.quality is not None
    assert outcome.result.quality.flags == (ReasonCode.INPUT_QUALITY_LOW,)


def test_ng_overrides_uncertain_roi(tmp_path: Path) -> None:
    image, mask = _dark_fixture()
    config = replace(
        BASE_CONFIG,
        roi=replace(BASE_CONFIG.roi, min_area_ratio=1.0),
    )

    outcome = _run(tmp_path, image, config=config, mask_path=mask, label="defect")

    assert outcome.result.status is Status.NG
    assert outcome.result.reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)
    assert outcome.result.roi is not None
    assert outcome.result.roi.uncertain is True


def test_save_stages_writes_response_and_candidate_images(tmp_path: Path) -> None:
    image, mask = _dark_fixture()

    outcome = _run(
        tmp_path,
        image,
        mask_path=mask,
        label="defect",
        save_stages=True,
    )

    safe_id = sanitize_sample_id(outcome.result.sample_id)
    directory = Path(outcome.result.overlay_path).parent
    assert (directory / f"{safe_id}_response.png").is_file()
    assert (directory / f"{safe_id}_candidates.png").is_file()
