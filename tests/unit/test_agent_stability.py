"""약후보 기하 패턴·섭동 민감도·패치 중재(`agent/stability.py`) 단위 테스트.

선형 스크래치/경계 잡음 분류, 결정론적 섭동 점수, 두 엔진 패치 중재 판정을 검증한다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from agent_fakes import make_component, make_observation, make_result
from surface_sentinel.agent.stability import (
    ArbitrationVerdict,
    StabilityLevel,
    StabilityPolicy,
    WeakPattern,
    analyze_weak_geometry,
    arbitrate_patches,
    assess_perturbation_stability,
)
from surface_sentinel.config import default_config
from surface_sentinel.contracts import RoiInfo, Status


def test_weak_geometry_none_and_single() -> None:
    empty = analyze_weak_geometry([])
    assert empty.pattern is WeakPattern.NONE
    assert empty.component_count == 0
    assert empty.collinearity_r2 == 0.0
    assert empty.cluster_compactness == 0.0
    assert empty.border_touch_ratio == 0.0

    single = analyze_weak_geometry([make_component(0, (40, 40, 20, 10), evidence=0.35)])
    assert single.pattern is WeakPattern.SINGLE_CANDIDATE
    assert single.component_count == 1
    assert single.max_evidence == 0.35


def test_weak_geometry_collinear_scratch_suspect() -> None:
    # 좁은 직선 회랑(수직 RMS <= 5px) 위에 놓인 4개의 컴포넌트(끊어진 스크래치 가설)
    comps = [
        make_component(0, (20, 40, 10, 4), evidence=0.32),
        make_component(1, (40, 42, 12, 4), evidence=0.31),
        make_component(2, (62, 44, 10, 4), evidence=0.30),
        make_component(3, (84, 46, 11, 4), evidence=0.29),
    ]
    roi = RoiInfo(mode="auto", bbox=(0, 0, 200, 400), area_px=80000, area_ratio=0.9)
    report = analyze_weak_geometry(comps, roi=roi)
    assert report.pattern is WeakPattern.COLLINEAR_SCRATCH_SUSPECT
    assert report.collinearity_r2 >= 0.85
    assert report.orthogonal_rms_px <= 5.0
    assert report.geometry_hash
    # 결정론 해시 재현성
    assert analyze_weak_geometry(comps, roi=roi).geometry_hash == report.geometry_hash


def test_weak_geometry_rejects_aspect_ratio_illusion_and_accepts_narrow_line() -> None:
    # ① KSDD2 1:2.7 세로형 ROI에서 가로 ±40px로 흩어진 점들은
    # 세로 분산 때문에 R^2 >= 0.85 가 되더라도 직선 스크래치로 오인하지 않는다.
    ksdd2_roi = RoiInfo(mode="auto", bbox=(0, 0, 230, 630), area_px=230 * 630, area_ratio=0.95)
    wide_scatter = [
        make_component(0, (70, 60, 10, 10), evidence=0.28),
        make_component(1, (150, 180, 10, 10), evidence=0.29),
        make_component(2, (72, 315, 10, 10), evidence=0.27),
        make_component(3, (152, 450, 10, 10), evidence=0.28),
        make_component(4, (74, 570, 10, 10), evidence=0.26),
    ]
    wide_report = analyze_weak_geometry(wide_scatter, roi=ksdd2_roi)
    assert wide_report.collinearity_r2 >= 0.85
    assert wide_report.orthogonal_rms_px > 14.0
    assert wide_report.pattern is not WeakPattern.COLLINEAR_SCRATCH_SUSPECT

    # ② 실제 좁은 직선(수직 RMS <= 5px) 위의 세로 스크래치 후보들은 COLLINEAR_SCRATCH_SUSPECT 유지
    narrow_line = [
        make_component(0, (110, 80, 6, 12), evidence=0.30),
        make_component(1, (112, 200, 6, 12), evidence=0.31),
        make_component(2, (111, 340, 6, 12), evidence=0.29),
        make_component(3, (113, 490, 6, 12), evidence=0.30),
    ]
    narrow_report = analyze_weak_geometry(narrow_line, roi=ksdd2_roi)
    assert narrow_report.orthogonal_rms_px <= 5.0
    assert narrow_report.pattern is WeakPattern.COLLINEAR_SCRATCH_SUSPECT

    # ③ 컴포넌트 2개 이하면 orthogonal_rms_px == 0.0
    two_report = analyze_weak_geometry(narrow_line[:2], roi=ksdd2_roi)
    assert two_report.orthogonal_rms_px == 0.0


def test_weak_geometry_border_artifact_suspect() -> None:
    comps = [
        make_component(0, (2, 100, 10, 10), evidence=0.33),
        make_component(1, (3, 150, 10, 10), evidence=0.31),
    ]
    # border_distance_px <= 8.0 이거나 touches_roi_border=True 인 비율 >= 0.6
    border_comps = [replace(c, border_distance_px=2.0, touches_roi_border=True) for c in comps]
    report = analyze_weak_geometry(border_comps)
    assert report.pattern is WeakPattern.BORDER_ARTIFACT_SUSPECT
    assert report.border_touch_ratio == 1.0


def test_weak_geometry_local_cluster_vs_diffuse() -> None:
    roi = RoiInfo(mode="auto", bbox=(0, 0, 200, 600), area_px=120000, area_ratio=0.9)
    tight = [
        make_component(0, (90, 280, 10, 10), evidence=0.30),
        make_component(1, (105, 290, 10, 10), evidence=0.31),
    ]
    tight_report = analyze_weak_geometry(tight, roi=roi)
    assert tight_report.pattern is WeakPattern.LOCAL_CLUSTER
    assert tight_report.cluster_compactness >= 0.72

    square_roi = RoiInfo(mode="auto", bbox=(0, 0, 400, 400), area_px=160000, area_ratio=0.9)
    diffuse = [
        make_component(0, (25, 25, 12, 12), evidence=0.28),
        make_component(1, (360, 25, 12, 12), evidence=0.29),
        make_component(2, (25, 360, 12, 12), evidence=0.27),
        make_component(3, (360, 360, 12, 12), evidence=0.26),
    ]
    diffuse_report = analyze_weak_geometry(diffuse, roi=square_roi)
    assert diffuse_report.pattern is WeakPattern.DIFFUSE_SURFACE_TEXTURE
    assert diffuse_report.cluster_compactness < 0.72


def test_perturbation_stability_deterministic_and_immutable(
    tmp_path: Path, monkeypatch
) -> None:
    # 디스크 쓰기 없이 메모리 실행 확인을 위해 작업 디렉터리를 tmp_path 로 이동
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(42)
    bgr = rng.integers(115, 125, size=(128, 128, 3), dtype=np.uint8)
    bgr[40:90, 62:66, :] = 35  # 뚜렷한 어두운 선형 결함
    original = bgr.copy()

    cfg = default_config()
    r1 = assess_perturbation_stability(bgr, cfg)
    r2 = assess_perturbation_stability(bgr, cfg)

    assert np.array_equal(bgr, original)  # 원본 배열 불변
    assert r1.stability_hash == r2.stability_hash
    assert len(r1.perturbations) == 2
    assert {p.name for p in r1.perturbations} == {"brightness_plus_5pct", "gaussian_blur_3x3"}
    assert r1.level in {StabilityLevel.STABLE, StabilityLevel.SENSITIVE, StabilityLevel.UNSTABLE}
    assert StabilityPolicy().policy_hash() == r1.policy_hash
    assert not (tmp_path / "runs").exists()


def test_perturbation_stability_zero_baseline_clean_image() -> None:
    bgr = np.full((128, 128, 3), 120, dtype=np.uint8)
    bgr[::2, ::2, :] = 122
    cfg = default_config()
    baseline = make_result(Status.OK, score=0.0)
    report = assess_perturbation_stability(bgr, cfg, baseline=baseline)
    assert report.level is StabilityLevel.STABLE
    assert report.mean_score_retention == 1.0
    assert report.max_abs_score_delta <= 0.05


def test_arbitrate_patches_engine_a_salient_and_border_noise() -> None:
    img = np.full((160, 160, 3), 120, dtype=np.uint8)
    img[60:80, 60:80, :] = 25
    obs_a = make_observation(
        "opencv",
        Status.REVIEW,
        components=[make_component(0, (58, 58, 24, 24), evidence=0.45)],
        score=0.45,
    )
    obs_b = make_observation("yolo-seg", Status.OK, task="seg")
    arb = arbitrate_patches(img, obs_a, obs_b)
    assert arb.verdict is ArbitrationVerdict.ENGINE_A_SALIENT
    assert arb.patch_a is not None
    assert arb.patch_b is None
    assert arb.patch_a.local_contrast > 30.0
    assert arb.patch_a.edge_sharpness > 20.0
    assert "opencv" in arb.summary
    assert arb.arbitration_hash == arbitrate_patches(img, obs_a, obs_b).arbitration_hash

    img_border = np.full((160, 160, 3), 120, dtype=np.uint8)
    img_border[0:10, 0:10, :] = 135
    border_comp = replace(make_component(0, (1, 1, 10, 10), evidence=0.32), touches_roi_border=True)
    obs_border = make_observation("opencv", Status.REVIEW, components=[border_comp], score=0.32)
    arb_border = arbitrate_patches(img_border, obs_border, obs_b)
    assert arb_border.verdict is ArbitrationVerdict.BORDER_NOISE_SUSPECT


def test_in_memory_pipeline_matches_inspect_image(tmp_path: Path) -> None:
    import cv2

    from surface_sentinel.agent.stability import _inspect_array_in_memory
    from surface_sentinel.pipeline import inspect_image

    repo_root = Path(__file__).resolve().parents[2]
    img_path = repo_root / "tests" / "fixtures" / "synthetic" / "good.png"
    cfg = default_config()
    disk_res = inspect_image(img_path, cfg, output_dir=tmp_path / "out").result
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert bgr is not None
    mem_res = _inspect_array_in_memory(bgr, cfg)
    assert mem_res.status is disk_res.status
    assert abs(mem_res.score - disk_res.score) < 1e-9
    assert mem_res.reasons == disk_res.reasons
    assert len(mem_res.components) == len(disk_res.components)


def test_arbitrate_patches_both_salient_and_aligned() -> None:
    img = np.full((160, 160, 3), 120, dtype=np.uint8)
    img[20:40, 20:40, :] = 20
    img[100:120, 100:120, :] = 20

    obs_a = make_observation(
        "opencv", Status.REVIEW, components=[make_component(0, (18, 18, 24, 24))], score=0.45
    )
    obs_b = make_observation(
        "yolo-seg",
        Status.REVIEW,
        components=[make_component(0, (98, 98, 24, 24))],
        score=0.45,
        task="seg",
    )
    both = arbitrate_patches(img, obs_a, obs_b)
    assert both.verdict is ArbitrationVerdict.BOTH_SALIENT

    obs_b_same = make_observation(
        "yolo-seg",
        Status.REVIEW,
        components=[make_component(0, (18, 18, 24, 24))],
        score=0.45,
        task="seg",
    )
    aligned = arbitrate_patches(img, obs_a, obs_b_same)
    assert aligned.verdict is ArbitrationVerdict.ALIGNED


def test_arbitrate_patches_single_zero_salience_patch_does_not_crash() -> None:
    """한쪽 패치가 None 이고 반대쪽이 경계 비접촉·salience=0 이어도 AssertionError 없이 중재한다."""

    uniform = np.full((160, 160, 3), 120, dtype=np.uint8)
    empty_a = make_observation("opencv", Status.OK, components=[], score=0.0)
    interior_b = make_observation(
        "yolo-seg",
        Status.REVIEW,
        components=[make_component(0, (60, 60, 20, 20))],
        score=0.30,
        task="seg",
    )

    res_b = arbitrate_patches(uniform, empty_a, interior_b)
    assert res_b.verdict is ArbitrationVerdict.ENGINE_B_SALIENT
    assert res_b.patch_a is None
    assert res_b.patch_b is not None
    assert res_b.patch_b.salience_index == 0.0
    assert res_b.patch_b.touches_border is False

    interior_a = make_observation(
        "opencv",
        Status.REVIEW,
        components=[make_component(0, (60, 60, 20, 20))],
        score=0.30,
    )
    empty_b = make_observation("yolo-seg", Status.OK, components=[], score=0.0, task="seg")
    res_a = arbitrate_patches(uniform, interior_a, empty_b)
    assert res_a.verdict is ArbitrationVerdict.ENGINE_A_SALIENT
    assert res_a.patch_a is not None
    assert res_a.patch_b is None
