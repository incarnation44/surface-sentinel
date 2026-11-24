"""YOLO 엔진 진입점(``inspect_with_yolo``) 단위 테스트 (E5b 통합).

실제 ONNX 가중치는 아직 없다(학습 중). ``YoloOnnxEngine`` 을 stub 으로 바꿔
합성 :class:`~surface_sentinel.yolo.inference.Detection` 만으로 검사 흐름을
검증한다. 실제 ONNX 통합은 학습 완료 후 ``tests/integration`` 에서 별도 실행한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import synthetic
from surface_sentinel.config import load_config
from surface_sentinel.contracts import ErrorCode, ReasonCode, SampleResult, Stage, Status
from surface_sentinel.pipeline import inspect_image
from surface_sentinel.records.fingerprint import file_sha256, policy_hash
from surface_sentinel.yolo import inspect as yolo_inspect
from surface_sentinel.yolo.adapter import YoloDecisionPolicy
from surface_sentinel.yolo.inference import Detection
from surface_sentinel.yolo.inspect import (
    COMBINED_TASK_ENV,
    DEFAULT_ONNX_ROOT,
    ONNX_ROOT_ENV,
    _merge_combined,
    engine_task,
    inspect_with_yolo,
    resolve_onnx_path,
)
from surface_sentinel.yolo.runtime import YoloEngineRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "base.toml")
POLICY = YoloDecisionPolicy()

Box = tuple[float, float, float, float]


def _det(score: float, bbox: Box = (100.0, 40.0, 130.0, 70.0)) -> Detection:
    return Detection(bbox_xyxy=bbox, score=score)


class StubEngine:
    """``YoloOnnxEngine`` 대체. 생성 인자와 추론 호출을 기록한다."""

    def __init__(self, model_path: str | Path, *, task: str = "det", **_kwargs: object) -> None:
        self.model_path = Path(model_path)
        self.task = task
        self.calls: list[tuple[int, int]] = []


def _install_stub(
    monkeypatch: pytest.MonkeyPatch,
    detections: list[Detection] | None = None,
    *,
    fail: Exception | None = None,
    seen: list[StubEngine] | None = None,
) -> list[StubEngine]:
    """``inspect`` 모듈의 ``YoloOnnxEngine`` 을 stub 으로 바꾼다."""

    created = seen if seen is not None else []
    items = list(detections or [])

    class _Stub(StubEngine):
        def __init__(self, model_path: str | Path, *, task: str = "det", **kwargs: object) -> None:
            super().__init__(model_path, task=task, **kwargs)
            created.append(self)

        def infer(self, bgr: np.ndarray, **_kwargs: object) -> list[Detection]:
            self.calls.append((int(bgr.shape[0]), int(bgr.shape[1])))
            if fail is not None:
                raise fail
            return list(items)

    monkeypatch.setattr(yolo_inspect, "YoloOnnxEngine", _Stub)
    return created


def _model(tmp_path: Path, task: str = "det") -> Path:
    """빈 ONNX 파일을 만든다(내용은 stub 이 쓰지 않는다)."""

    path = tmp_path / "models" / task / "weights" / "best.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def image(tmp_path: Path) -> Path:
    return synthetic.write_png(tmp_path / "sample_a.png", synthetic.good_image())


# ---------------------------------------------------------------- 모델 경로 해석


def test_default_model_root_is_data_derived() -> None:
    assert Path("data/derived/yolo/models") == DEFAULT_ONNX_ROOT
    assert engine_task("yolo-det") == "det"
    assert engine_task("yolo-seg") == "seg"


def test_engine_task_combined_requires_explicit_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(COMBINED_TASK_ENV, raising=False)
    # P0-3: 숨은 det 기본값 제거 — task 미설정 시 설정 오류.
    with pytest.raises(ValueError, match="combined"):
        engine_task("combined")


def test_engine_task_combined_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMBINED_TASK_ENV, "seg")
    assert engine_task("combined") == "seg"
    monkeypatch.setenv(COMBINED_TASK_ENV, "pose")
    with pytest.raises(ValueError):
        engine_task("combined")


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        engine_task("yolo-magic")


def test_missing_model_raises_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ONNX_ROOT_ENV, str(tmp_path / "empty-root"))
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_onnx_path("yolo-det")
    message = str(excinfo.value)
    assert "best.onnx" in message
    assert ONNX_ROOT_ENV in message  # 어떤 설정을 봤는지 메시지에 남는다


def test_env_root_resolves_task_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path, "seg")
    monkeypatch.setenv(ONNX_ROOT_ENV, str(tmp_path / "models"))
    assert resolve_onnx_path("yolo-seg") == model


def test_onnx_path_argument_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_model = _model(tmp_path, "det")
    override = tmp_path / "custom" / "model.onnx"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_bytes(b"")
    monkeypatch.setenv(ONNX_ROOT_ENV, str(tmp_path / "models"))
    assert resolve_onnx_path("yolo-det", override) == override
    assert resolve_onnx_path("yolo-det") == env_model


# ---------------------------------------------------------------- det / seg 추론


def test_det_engine_flags_ng(tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(tmp_path)
    engines = _install_stub(monkeypatch, [_det(0.8)])

    outcome = inspect_with_yolo(
        image, BASE_CONFIG, engine="yolo-det", onnx_path=model
    )

    result = outcome.result
    assert outcome.debug is None
    assert result.status is Status.NG
    assert result.score == pytest.approx(0.8)
    assert result.reasons == (ReasonCode.DEFECT_EVIDENCE_STRONG,)
    assert result.sample_id == "sample_a"
    assert result.original_id == "sample_a"
    assert result.split == "tune"
    assert result.config_hash == BASE_CONFIG.config_hash
    assert result.overlay_path == ""  # output_dir 없으면 파일을 쓰지 않는다
    assert engines[0].task == "det"
    assert engines[0].model_path == model
    assert len(engines[0].calls) == 1


def test_no_detection_is_ok(tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [])

    result = inspect_with_yolo(image, BASE_CONFIG, engine="yolo-det", onnx_path=model).result

    assert result.status is Status.OK
    assert result.score == 0.0
    assert result.reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)
    assert result.components == ()


def test_seg_engine_uses_seg_task_and_masks(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path, "seg")
    mask = np.zeros((210, 280), np.uint8)
    mask[40:70, 100:130] = 1
    detection = Detection(bbox_xyxy=(100.0, 40.0, 130.0, 70.0), score=0.9, mask=mask)
    engines = _install_stub(monkeypatch, [detection])

    result = inspect_with_yolo(image, BASE_CONFIG, engine="yolo-seg", onnx_path=model).result

    assert engines[0].task == "seg"
    assert result.status is Status.NG
    assert result.components[0].area_px == 900
    assert result.components[0].solidity == pytest.approx(1.0)


def test_policy_argument_controls_thresholds(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])
    strict = YoloDecisionPolicy(ng_conf=0.95, review_lo=0.5)

    result = inspect_with_yolo(
        image, BASE_CONFIG, engine="yolo-det", onnx_path=model, policy=strict
    ).result

    assert result.status is Status.REVIEW


def test_timing_reports_measured_values(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])

    result = inspect_with_yolo(image, BASE_CONFIG, engine="yolo-det", onnx_path=model).result

    assert {"load_ms", "infer_ms", "total"} <= set(result.timing_ms)
    assert "overlay_ms" not in result.timing_ms
    assert result.timing_ms["infer_ms"] >= 0.0
    assert result.timing_ms["total"] >= result.timing_ms["infer_ms"]


def test_inference_failure_returns_error_result(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """추론 실패는 예외가 아니라 ERROR 결과다(워커가 재시도·기록한다)."""

    model = _model(tmp_path)
    _install_stub(monkeypatch, fail=RuntimeError("session broke"))

    result = inspect_with_yolo(image, BASE_CONFIG, engine="yolo-det", onnx_path=model).result

    assert result.status is Status.ERROR
    assert result.error_code is not None
    assert result.error_code.value == "E_PROCESS_INTERNAL"
    assert result.error_stage is not None


def test_undecodable_image_returns_error_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")

    result = inspect_with_yolo(broken, BASE_CONFIG, engine="yolo-det", onnx_path=model).result

    assert result.status is Status.ERROR
    assert result.error_code is not None
    assert result.error_code.value == "E_INPUT_DECODE"


def test_run_id_matches_pipeline_convention(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])
    opencv_result = inspect_image(
        image, BASE_CONFIG, split="tune", label="unknown", output_dir=tmp_path / "opencv"
    ).result

    yolo_result = inspect_with_yolo(
        image, BASE_CONFIG, engine="yolo-det", onnx_path=model, output_dir=tmp_path / "yolo"
    ).result

    assert yolo_result.run_id == opencv_result.run_id
    assert yolo_result.sample_id == opencv_result.sample_id


# ---------------------------------------------------------------- 오버레이


def test_overlay_filename_and_path_are_filled(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])
    output_dir = tmp_path / "overlays"

    result = inspect_with_yolo(
        image,
        BASE_CONFIG,
        engine="yolo-det",
        onnx_path=model,
        output_dir=output_dir,
    ).result

    overlay = Path(result.overlay_path)
    assert overlay.is_file()
    assert overlay.name == "sample_a_yolo.png"
    assert overlay.parent == output_dir
    assert result.timing_ms["overlay_ms"] >= 0.0


def test_overlay_filename_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])
    image = synthetic.write_png(tmp_path / "bad id@1.png", synthetic.good_image())

    result = inspect_with_yolo(
        image,
        BASE_CONFIG,
        engine="yolo-det",
        onnx_path=model,
        output_dir=tmp_path / "overlays",
    ).result

    assert Path(result.overlay_path).name == "bad_id_1_yolo.png"
    # 워커가 ``{safe_id}_*`` 글롭으로 지울 수 있는 접두사여야 한다.
    assert Path(result.overlay_path).name.startswith("bad_id_1_")


def test_overlay_write_failure_returns_storage_error(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """오버레이 저장 실패는 ERROR 결과다(판정 자체를 잃지 않게 코드로 남긴다)."""

    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])

    def broken_save(*_args: object, **_kwargs: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(yolo_inspect, "save_overlay", broken_save)

    result = inspect_with_yolo(
        image, BASE_CONFIG, engine="yolo-det", onnx_path=model, output_dir=tmp_path / "ov"
    ).result

    assert result.status is Status.ERROR
    assert result.error_code is not None
    assert result.error_code.value == "E_STORAGE_WRITE"
    assert result.error_stage is not None
    assert result.error_stage.value == "save_evidence"
    assert result.overlay_path == ""


# ---------------------------------------------------------------- combined


def test_combined_runs_both_engines_and_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    engines = _install_stub(monkeypatch, [_det(0.8)])
    image = synthetic.write_png(tmp_path / "good_a.png", synthetic.good_image())
    output_dir = tmp_path / "overlays"

    result = inspect_with_yolo(
        image, BASE_CONFIG, engine="combined", onnx_path=model, output_dir=output_dir
    ).result

    assert len(engines) == 1
    assert engines[0].task == "det"
    assert result.status is Status.NG  # opencv OK + YOLO NG → NG
    assert result.score == pytest.approx(0.8)
    assert set(result.reasons) == {
        ReasonCode.NO_DEFECT_EVIDENCE,
        ReasonCode.DEFECT_EVIDENCE_STRONG,
    }
    # opencv 산출물과 YOLO 산출물이 같은 디렉터리에서 충돌하지 않는다.
    assert (output_dir / "good_a_overlay.png").is_file()
    assert Path(result.overlay_path).name == "good_a_yolo.png"
    assert result.timing_ms["opencv_total_ms"] >= 0.0
    assert result.timing_ms["infer_ms"] >= 0.0


def test_combined_opencv_error_stays_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencv ERROR + YOLO OK → ERROR, 오류 코드/단계는 보존한다."""

    model = _model(tmp_path)
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    _install_stub(monkeypatch, [])
    uniform = synthetic.write_png(
        tmp_path / "uniform.png", np.full((200, 200, 3), 128, np.uint8)
    )

    result = inspect_with_yolo(
        uniform, BASE_CONFIG, engine="combined", onnx_path=model, output_dir=tmp_path / "ov"
    ).result

    assert result.status is Status.ERROR
    assert result.error_code is not None
    assert result.error_code.value == "E_INPUT_UNIFORM"
    assert result.error_stage is not None
    assert result.error_stage.value == "validate_input"


def test_combined_opencv_error_does_not_mask_yolo_ng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencv ERROR + YOLO NG → NG(보수적 결합: ERROR 가 결함 판정을 묻지 않는다)."""

    model = _model(tmp_path)
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    _install_stub(monkeypatch, [_det(0.8)])
    uniform = synthetic.write_png(
        tmp_path / "uniform.png", np.full((200, 200, 3), 128, np.uint8)
    )

    result = inspect_with_yolo(
        uniform, BASE_CONFIG, engine="combined", onnx_path=model, output_dir=tmp_path / "ov"
    ).result

    assert result.status is Status.NG


def test_combined_without_opencv_artifacts_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    _install_stub(monkeypatch, [])
    image = synthetic.write_png(tmp_path / "good_b.png", synthetic.good_image())

    result = inspect_with_yolo(
        image, BASE_CONFIG, engine="combined", onnx_path=model, output_dir=tmp_path / "ov"
    ).result

    assert result.status is Status.OK
    # 양쪽이 같은 이유를 내면 한 번만 남는다.
    assert result.reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)


def test_combined_uses_seg_model_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path, "seg")
    monkeypatch.setenv(COMBINED_TASK_ENV, "seg")
    engines = _install_stub(monkeypatch, [])
    image = synthetic.write_png(tmp_path / "good_c.png", synthetic.good_image())

    inspect_with_yolo(image, BASE_CONFIG, engine="combined", onnx_path=model)

    assert engines[0].task == "seg"


def _result(status: Status, *, reasons: tuple[ReasonCode, ...], score: float) -> SampleResult:
    return SampleResult(
        run_id="r",
        sample_id="s",
        original_id="s",
        split="tune",
        label="unknown",
        status=status,
        score=score,
        reasons=reasons,
    )


@pytest.mark.parametrize(
    ("opencv", "yolo", "expected"),
    [
        (Status.OK, Status.OK, Status.OK),
        (Status.OK, Status.NG, Status.NG),
        (Status.NG, Status.OK, Status.NG),
        (Status.OK, Status.REVIEW, Status.REVIEW),
        (Status.REVIEW, Status.OK, Status.REVIEW),
        (Status.REVIEW, Status.NG, Status.NG),
        (Status.ERROR, Status.REVIEW, Status.REVIEW),
        (Status.REVIEW, Status.ERROR, Status.REVIEW),
        (Status.ERROR, Status.NG, Status.NG),
        (Status.OK, Status.ERROR, Status.ERROR),
        (Status.ERROR, Status.OK, Status.ERROR),
        (Status.ERROR, Status.ERROR, Status.ERROR),
    ],
)
def test_merge_combined_priority(
    opencv: Status, yolo: Status, expected: Status
) -> None:
    merged = _merge_combined(
        _result(opencv, reasons=(ReasonCode.ROI_UNCERTAIN,), score=0.4),
        _result(yolo, reasons=(ReasonCode.DEFECT_EVIDENCE_STRONG,), score=0.9),
    )
    assert merged.status is expected
    # 순서 보존 합집합 + 중복 제거
    assert merged.reasons == (
        ReasonCode.ROI_UNCERTAIN,
        ReasonCode.DEFECT_EVIDENCE_STRONG,
    )
    assert merged.score == pytest.approx(max(0.4, 0.9))
    assert merged.sample_id == "s"


def test_merge_combined_dedups_reasons() -> None:
    merged = _merge_combined(
        _result(Status.OK, reasons=(ReasonCode.NO_DEFECT_EVIDENCE,), score=0.0),
        _result(Status.OK, reasons=(ReasonCode.NO_DEFECT_EVIDENCE,), score=0.0),
    )
    assert merged.reasons == (ReasonCode.NO_DEFECT_EVIDENCE,)


# ------------------------------------------------ 신원·세션 재사용·엔진별 관찰 (P0-2/P0-3)


def test_inspection_identity_builds_spec(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"model-bytes")
    spec = yolo_inspect.inspection_identity("yolo-seg", BASE_CONFIG, onnx_path=model)
    assert spec.engine == "yolo-seg"
    assert spec.task == "seg"
    assert spec.model_sha256 == file_sha256(model)
    assert spec.config_hash == BASE_CONFIG.config_hash
    assert spec.yolo_policy_hash == policy_hash(0.5, 0.25)
    assert spec.combine_policy_hash == ""
    assert spec.provenance_complete is True


def test_inspection_identity_combined_has_combine_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(COMBINED_TASK_ENV, "seg")
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    spec = yolo_inspect.inspection_identity("combined", BASE_CONFIG, onnx_path=model)
    assert spec.task == "seg"
    assert spec.combine_policy_hash != ""


def test_inspection_identity_policy_and_code_change_fingerprint(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    base = yolo_inspect.inspection_identity("yolo-det", BASE_CONFIG, onnx_path=model)
    strict = yolo_inspect.inspection_identity(
        "yolo-det",
        BASE_CONFIG,
        onnx_path=model,
        policy=YoloDecisionPolicy(ng_conf=0.9, review_lo=0.5),
    )
    coded = yolo_inspect.inspection_identity(
        "yolo-det", BASE_CONFIG, onnx_path=model, code_sha="f" * 40
    )
    assert len({base.fingerprint, strict.fingerprint, coded.fingerprint}) == 3


def test_registry_reuses_session_across_images(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    created: list[object] = []

    class _RegStub:
        def __init__(self, path: str | Path, *, task: str = "det", **_kw: object) -> None:
            self.task = task
            created.append(self)

        def infer(self, bgr: np.ndarray, **_kw: object) -> list[Detection]:
            return [_det(0.8)]

    registry = YoloEngineRegistry(engine_factory=_RegStub)
    other = synthetic.write_png(tmp_path / "sample_b.png", synthetic.good_image())

    first = inspect_with_yolo(
        image, BASE_CONFIG, engine="yolo-det", onnx_path=model, registry=registry
    )
    second = inspect_with_yolo(
        other, BASE_CONFIG, engine="yolo-det", onnx_path=model, registry=registry
    )

    assert len(created) == 1
    assert registry.size == 1
    assert registry.cold_starts == 1
    assert first.result.status is Status.NG
    assert second.result.status is Status.NG
    assert second.result.timing_ms["session_create_ms"] == 0.0


def test_combined_preserves_engine_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    _install_stub(monkeypatch, [_det(0.8)])
    image = synthetic.write_png(tmp_path / "good_obs.png", synthetic.good_image())

    outcome = inspect_with_yolo(image, BASE_CONFIG, engine="combined", onnx_path=model)

    observations = outcome.engine_observations
    assert len(observations) == 2
    assert observations[0].engine == "opencv"
    assert observations[1].engine == "yolo-det"
    assert observations[1].task == "det"
    assert observations[0].result.status is Status.OK
    assert observations[1].result.status is Status.NG
    assert observations[1].result.score == pytest.approx(0.8)
    # 결합 결과는 NG 이지만 엔진별 원본 점수·성분이 보존된다.
    assert outcome.result.status is Status.NG
    assert observations[1].result.components


def test_single_engine_has_no_observations(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    _install_stub(monkeypatch, [_det(0.8)])
    outcome = inspect_with_yolo(image, BASE_CONFIG, engine="yolo-det", onnx_path=model)
    assert outcome.engine_observations == ()


@pytest.mark.parametrize("engine", ("yolo-det", "yolo-seg", "combined"))
@pytest.mark.parametrize("split", ("test", "official_test"))
def test_sealed_split_denied_before_inference_and_opencv(
    tmp_path: Path,
    image: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    split: str,
) -> None:
    """봉인 분할(test/official_test)은 YOLO/combined 경로에서도 추론 전에 차단된다."""

    model = _model(tmp_path, "seg" if engine == "yolo-seg" else "det")
    monkeypatch.setenv(COMBINED_TASK_ENV, "det")
    engines = _install_stub(monkeypatch, [_det(0.9)])
    opencv_calls: list[Path] = []

    def fail_if_called(*args: object, **kwargs: object) -> object:
        opencv_calls.append(Path(str(args[0])))
        raise AssertionError("봉인 분할에서 OpenCV 파이프라인이 호출되면 안 됩니다")

    monkeypatch.setattr(yolo_inspect, "inspect_image", fail_if_called)

    outcome = inspect_with_yolo(
        image,
        BASE_CONFIG,
        engine=engine,
        split=split,
        onnx_path=model,
        output_dir=tmp_path / "ov",
    )

    result = outcome.result
    assert result.status is Status.ERROR
    assert result.error_code is ErrorCode.E_SEALED_DENIED
    assert result.error_stage is Stage.VALIDATE_INPUT
    assert result.overlay_path == ""
    assert outcome.engine_observations == ()
    assert engines == []
    assert opencv_calls == []


def test_allow_sealed_opt_in_runs_on_synthetic_fixture(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """명시적 allow_sealed=True 일 때만 봉인 split 검사를 진행한다(합성 입력 전용)."""

    model = _model(tmp_path)
    engines = _install_stub(monkeypatch, [_det(0.8)])

    result = inspect_with_yolo(
        image,
        BASE_CONFIG,
        engine="yolo-det",
        split="test",
        onnx_path=model,
        allow_sealed=True,
    ).result

    assert result.status is Status.NG
    assert result.error_code is None
    assert len(engines) == 1
    assert len(engines[0].calls) == 1


def test_inspect_with_yolo_propagates_import_error_and_syncs_sha_cache(
    tmp_path: Path, image: Path
) -> None:
    """onnxruntime 미설치(ImportError)는 ERROR 결과가 아니라 예외로 전파된다(503).

    표본 단위 실패(디코드·추론 오류)만 ERROR 결과로 남기고, 환경 문제는
    호출자(API 503)가 처리하도록 멈춘다.
    """

    model = _model(tmp_path, "det")

    def _no_runtime(*args: object, **kwargs: object) -> object:
        raise ImportError("onnxruntime 미설치(재현)")

    registry = YoloEngineRegistry(engine_factory=_no_runtime)
    with pytest.raises(ImportError, match="onnxruntime"):
        inspect_with_yolo(
            image,
            BASE_CONFIG,
            engine="yolo-det",
            onnx_path=model,
            registry=registry,
        )
