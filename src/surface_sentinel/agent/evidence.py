"""Phase 1 교차 증거 — 엔진 간 비교 관찰 (에이전트 확장 제안 §2, observe-only).

결합(combined) 검사가 보존한 엔진별 관찰(``EngineObservation``)을 비교해
**별도 증거**(AgentAssessment)를 만든다. 기계 판정(``SampleResult.status``)은
만들거나 바꾸지 않는다 — 판정 소유권은 파이프라인에 있다(제안 §2 공통 계약).

계산은 전부 순수 함수다:

* localization: bbox IoU·containment·중심거리·면적비 (컴포넌트 bbox 는 x,y,w,h).
  관찰에는 마스크가 없으므로 bbox 기하를 쓴다. containment 는 실제 컴포넌트
  면적(``area_px``) 기준이라 bbox 만으로 과대평가되지 않는다.
* 매칭: 결정론 greedy — IoU 내림차순, 동률은 (index_a, index_b) 오름차순.
  입력 순서와 무관하게 같은 결과가 나온다(해시·replay 가능).
* 전용 플래그: 파이프라인 ``ReasonCode`` 와 별개 네임스페이스. 판정 이유가
  아니라 증거 관찰 결과이므로混ぜ지 않는다.
* dual overlay: 두 엔진 박스를 한 장에 다른 색으로 그린다(입력 불변).

임계값은 ``EvidencePolicy`` 동결 상수(버전 ``evidence-v1``)로 고정하고
정책 해시를 평가 해시에 묶는다. 설정 파일(agent.toml)은 변경하지 않는다 —
임계값 변경은 코드 개정 + 버전 bump 로만 한다(동결 계약 유지).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import ComponentFeature, EngineObservation, RoiInfo, Status
from ..records.fingerprint import canonical_json_sha256

#: 평가 산출물 스키마 버전. 필드가 바뀌면 bump.
EVIDENCE_SCHEMA_VERSION = "agent-assessment-v1"
#: 임계값 정책 버전. 임계값 기본값이 바뀌면 bump(해시로 추적된다).
EVIDENCE_POLICY_VERSION = "evidence-v1"

#: dual overlay 엔진 색(BGR). A=녹색, B=주황, 매칭 연결선=흰색.
DUAL_COLOR_A = (0, 200, 0)
DUAL_COLOR_B = (0, 165, 255)
DUAL_MATCH_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_TEXT_THICKNESS = 1


class EvidenceFlag(StrEnum):
    """교차 증거 전용 플래그. 선언 순서가 곧 출력 순서다(결정론)."""

    STATUS_MISMATCH = "STATUS_MISMATCH"
    NG_SINGLE_ENGINE = "NG_SINGLE_ENGINE"
    ERROR_ENGINE = "ERROR_ENGINE"
    SINGLE_ENGINE_EVIDENCE = "SINGLE_ENGINE_EVIDENCE"
    LOCALIZATION_MISMATCH = "LOCALIZATION_MISMATCH"
    UNMATCHED_COMPONENTS = "UNMATCHED_COMPONENTS"
    WEAK_MATCH = "WEAK_MATCH"
    CENTER_OFFSET_LARGE = "CENTER_OFFSET_LARGE"
    AREA_RATIO_LARGE = "AREA_RATIO_LARGE"
    SCORE_GAP_LARGE = "SCORE_GAP_LARGE"
    ROI_MISMATCH = "ROI_MISMATCH"
    ROI_UNCERTAIN_DIFF = "ROI_UNCERTAIN_DIFF"
    EVIDENCE_ALIGNED = "EVIDENCE_ALIGNED"


#: 사람 검토가 필요한 플래그(상태 불일치·단일 엔진 NG·엔진 오류).
_HUMAN_REVIEW_FLAGS = frozenset(
    {EvidenceFlag.STATUS_MISMATCH, EvidenceFlag.NG_SINGLE_ENGINE, EvidenceFlag.ERROR_ENGINE}
)


@dataclass(frozen=True)
class EvidencePolicy:
    """교차 증거 임계값(동결 기본값). 변경은 버전 bump 를 동반한다.

    * ``iou_match_min`` 0.10 — 매칭 성립 최소 IoU. 저장소 지역화 recall 기준
      (``IOU_MATCH_RECALL``)과 같은 값.
    * ``iou_strong_min`` 0.50 — 강한 위치 일치. ``IOU_MATCH_FULL`` 과 같은 값.
    * ``center_distance_norm_max`` 0.5 — 중심거리/큰 박스 대각선 상한.
    * ``area_ratio_max`` 4.0 — 매칭 쌍 면적비(큰/작은) 상한.
    * ``score_gap_max`` 0.25 — 엔진 점수 차이 허용치(보정 안 된 순위용 점수).
    * ``roi_iou_min`` 0.50 — ROI bbox 일치 하한.
    """

    iou_match_min: float = 0.10
    iou_strong_min: float = 0.50
    center_distance_norm_max: float = 0.5
    area_ratio_max: float = 4.0
    score_gap_max: float = 0.25
    roi_iou_min: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_match_min <= self.iou_strong_min <= 1.0:
            raise ValueError(
                "0 < iou_match_min <= iou_strong_min <= 1 이어야 합니다: "
                f"{self.iou_match_min}, {self.iou_strong_min}"
            )
        if self.center_distance_norm_max <= 0.0:
            raise ValueError("center_distance_norm_max 는 0보다 커야 합니다")
        if self.area_ratio_max < 1.0:
            raise ValueError("area_ratio_max 는 1 이상이어야 합니다")
        if self.score_gap_max < 0.0:
            raise ValueError("score_gap_max 는 0 이상이어야 합니다")
        if not 0.0 <= self.roi_iou_min <= 1.0:
            raise ValueError("roi_iou_min 는 0~1 이어야 합니다")

    @property
    def policy_hash(self) -> str:
        """버전 + 임계값 전체의 canonical 해시(평가 해시에 묶인다)."""

        return canonical_json_sha256({"version": EVIDENCE_POLICY_VERSION, **asdict(self)})


def default_evidence_policy() -> EvidencePolicy:
    return EvidencePolicy()


@dataclass(frozen=True)
class ComponentMatch:
    """매칭된 컴포넌트 쌍의 지역화 지표."""

    index_a: int
    index_b: int
    iou: float
    containment: float
    center_distance_px: float
    center_distance_norm: float
    area_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_a": self.index_a,
            "index_b": self.index_b,
            "iou": round(self.iou, 6),
            "containment": round(self.containment, 6),
            "center_distance_px": round(self.center_distance_px, 6),
            "center_distance_norm": round(self.center_distance_norm, 6),
            "area_ratio": round(self.area_ratio, 6),
        }


@dataclass(frozen=True)
class GreedyMatchResult:
    """결정론 greedy 매칭 결과(쌍 + 양쪽 미매칭 인덱스)."""

    matches: tuple[ComponentMatch, ...]
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]


@dataclass(frozen=True)
class RoiComparison:
    """이미지 단위 ROI 비교."""

    iou: float
    uncertain_a: bool
    uncertain_b: bool
    bbox_a: tuple[int, int, int, int]
    bbox_b: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou": round(self.iou, 6),
            "uncertain_a": self.uncertain_a,
            "uncertain_b": self.uncertain_b,
            "bbox_a": list(self.bbox_a),
            "bbox_b": list(self.bbox_b),
        }


@dataclass(frozen=True)
class CrossEvidenceAssessment:
    """교차 증거 평가 산출물. 기계 판정과 분리된 별도 기록이다(observe-only).

    ``assessment_hash`` 는 입력 해시·정책 해시·플래그·지표 전체의 canonical
    해시다. 같은 입력 + 같은 정책이면 항상 같은 해시 → 저장소 중복제거와
    replay 가 성립한다.
    """

    engine_a: str
    engine_b: str
    task_a: str
    task_b: str
    status_a: str
    status_b: str
    status_agree: bool
    score_a: float
    score_b: float
    score_gap: float
    flags: tuple[EvidenceFlag, ...]
    matches: tuple[ComponentMatch, ...]
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]
    roi: RoiComparison | None
    recommended_action: str
    policy_hash: str
    input_a_hash: str
    input_b_hash: str
    assessment_hash: str
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "task_a": self.task_a,
            "task_b": self.task_b,
            "status_a": self.status_a,
            "status_b": self.status_b,
            "status_agree": self.status_agree,
            "score_a": float(self.score_a),
            "score_b": float(self.score_b),
            "score_gap": round(float(self.score_gap), 6),
            "flags": [flag.value for flag in self.flags],
            "matches": [match.to_dict() for match in self.matches],
            "unmatched_a": list(self.unmatched_a),
            "unmatched_b": list(self.unmatched_b),
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "recommended_action": self.recommended_action,
            "policy_hash": self.policy_hash,
            "input_a_hash": self.input_a_hash,
            "input_b_hash": self.input_b_hash,
            "assessment_hash": self.assessment_hash,
        }


# ---------------------------------------------------------------------------
# bbox 기하 (ComponentFeature.bbox = x, y, w, h)
# ---------------------------------------------------------------------------


def bbox_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """두 bbox 의 교집합 면적(px). 겹치지 않으면 0."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """bbox IoU. 면적 0 박스는 0 을 돌려준다(0 으로 나누지 않는다)."""

    intersection = bbox_intersection_area(a, b)
    if intersection <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def pair_metrics(a: ComponentFeature, b: ComponentFeature) -> ComponentMatch:
    """컴포넌트 쌍의 지역화 지표. containment 는 실제 면적 기준이다."""

    intersection = bbox_intersection_area(a.bbox, b.bbox)
    min_area = min(a.area_px, b.area_px)
    containment = intersection / float(min_area) if min_area > 0 else 0.0
    center_a = bbox_center(a.bbox)
    center_b = bbox_center(b.bbox)
    distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    diagonal_a = math.hypot(a.bbox[2], a.bbox[3])
    diagonal_b = math.hypot(b.bbox[2], b.bbox[3])
    norm = distance / max(diagonal_a, diagonal_b, 1.0)
    area_ratio = max(a.area_px, b.area_px) / float(max(1, min_area))
    return ComponentMatch(
        index_a=a.index,
        index_b=b.index,
        iou=bbox_iou(a.bbox, b.bbox),
        containment=containment,
        center_distance_px=distance,
        center_distance_norm=norm,
        area_ratio=area_ratio,
    )


def greedy_match(
    components_a: Sequence[ComponentFeature],
    components_b: Sequence[ComponentFeature],
    *,
    iou_min: float,
) -> GreedyMatchResult:
    """결정론 greedy 매칭: IoU 내림차순, 동률은 (index_a, index_b) 오름차순.

    각 컴포넌트는 최대 한 번만 매칭된다. 입력 순서와 무관하게 같은 결과.
    """

    candidates: list[tuple[float, int, int, ComponentFeature, ComponentFeature]] = []
    for left in components_a:
        for right in components_b:
            iou = bbox_iou(left.bbox, right.bbox)
            if iou >= iou_min:
                candidates.append((-iou, left.index, right.index, left, right))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[ComponentMatch] = []
    for _neg_iou, index_a, index_b, left, right in candidates:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matches.append(pair_metrics(left, right))
    matches.sort(key=lambda match: (match.index_a, match.index_b))
    unmatched_a = tuple(sorted(c.index for c in components_a if c.index not in used_a))
    unmatched_b = tuple(sorted(c.index for c in components_b if c.index not in used_b))
    return GreedyMatchResult(
        matches=tuple(matches), unmatched_a=unmatched_a, unmatched_b=unmatched_b
    )


def roi_comparison(roi_a: RoiInfo | None, roi_b: RoiInfo | None) -> RoiComparison | None:
    """ROI bbox IoU + 불확실성 비교. 한쪽이라도 없으면 None."""

    if roi_a is None or roi_b is None:
        return None
    return RoiComparison(
        iou=bbox_iou(roi_a.bbox, roi_b.bbox),
        uncertain_a=bool(roi_a.uncertain),
        uncertain_b=bool(roi_b.uncertain),
        bbox_a=roi_a.bbox,
        bbox_b=roi_b.bbox,
    )


# ---------------------------------------------------------------------------
# 교차 평가
# ---------------------------------------------------------------------------


def _observation_input_hash(observation: EngineObservation) -> str:
    """증거 관련 입력만의 해시. 타이밍·run_id 같은 실행 값은 제외한다.

    같은 증거 내용이면 재실행해도 같은 해시 → 저장소 중복제거가 성립한다.
    """

    result = observation.result
    return canonical_json_sha256(
        {
            "engine": observation.engine,
            "task": observation.task,
            "status": result.status.value,
            "score": float(result.score),
            "reasons": [reason.value for reason in result.reasons],
            "components": [component.to_dict() for component in result.components],
            "roi": result.roi.to_dict() if result.roi is not None else None,
            "error_code": result.error_code.value if result.error_code is not None else None,
        }
    )


def assess_cross_evidence(
    observation_a: EngineObservation,
    observation_b: EngineObservation,
    *,
    policy: EvidencePolicy | None = None,
) -> CrossEvidenceAssessment:
    """엔진 관찰 둘을 비교해 별도 증거 평가를 만든다(순수 함수, 판정 불변)."""

    effective = policy if policy is not None else default_evidence_policy()
    result_a = observation_a.result
    result_b = observation_b.result
    components_a = result_a.components
    components_b = result_b.components

    matched = greedy_match(components_a, components_b, iou_min=effective.iou_match_min)
    roi = roi_comparison(result_a.roi, result_b.roi)
    score_gap = abs(float(result_a.score) - float(result_b.score))
    status_agree = result_a.status is result_b.status

    flags: set[EvidenceFlag] = set()
    if not status_agree:
        flags.add(EvidenceFlag.STATUS_MISMATCH)
    statuses = (result_a.status, result_b.status)
    if statuses.count(Status.NG) == 1:
        flags.add(EvidenceFlag.NG_SINGLE_ENGINE)
    if Status.ERROR in statuses:
        flags.add(EvidenceFlag.ERROR_ENGINE)

    has_a = bool(components_a)
    has_b = bool(components_b)
    if has_a != has_b:
        flags.add(EvidenceFlag.SINGLE_ENGINE_EVIDENCE)
    elif has_a and has_b and not matched.matches:
        flags.add(EvidenceFlag.LOCALIZATION_MISMATCH)
    if matched.matches and (matched.unmatched_a or matched.unmatched_b):
        flags.add(EvidenceFlag.UNMATCHED_COMPONENTS)
    if any(match.iou < effective.iou_strong_min for match in matched.matches):
        flags.add(EvidenceFlag.WEAK_MATCH)
    if any(
        match.center_distance_norm > effective.center_distance_norm_max
        for match in matched.matches
    ):
        flags.add(EvidenceFlag.CENTER_OFFSET_LARGE)
    if any(match.area_ratio > effective.area_ratio_max for match in matched.matches):
        flags.add(EvidenceFlag.AREA_RATIO_LARGE)
    if score_gap > effective.score_gap_max:
        flags.add(EvidenceFlag.SCORE_GAP_LARGE)
    if roi is not None:
        if roi.iou < effective.roi_iou_min:
            flags.add(EvidenceFlag.ROI_MISMATCH)
        if roi.uncertain_a != roi.uncertain_b:
            flags.add(EvidenceFlag.ROI_UNCERTAIN_DIFF)

    localization_aligned = (not has_a and not has_b) or (
        bool(matched.matches)
        and not matched.unmatched_a
        and not matched.unmatched_b
        and all(match.iou >= effective.iou_strong_min for match in matched.matches)
    )
    problem_flags = set(flags)
    if status_agree and not problem_flags and localization_aligned:
        flags.add(EvidenceFlag.EVIDENCE_ALIGNED)

    ordered = tuple(flag for flag in EvidenceFlag if flag in flags)
    if problem_flags & _HUMAN_REVIEW_FLAGS:
        recommended = "HUMAN_REVIEW"
    elif problem_flags:
        recommended = "INSPECT_EVIDENCE"
    else:
        recommended = "NONE"

    input_a_hash = _observation_input_hash(observation_a)
    input_b_hash = _observation_input_hash(observation_b)
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "engine_a": observation_a.engine,
        "engine_b": observation_b.engine,
        "task_a": observation_a.task,
        "task_b": observation_b.task,
        "status_a": result_a.status.value,
        "status_b": result_b.status.value,
        "status_agree": status_agree,
        "score_a": float(result_a.score),
        "score_b": float(result_b.score),
        "score_gap": round(score_gap, 6),
        "flags": [flag.value for flag in ordered],
        "matches": [match.to_dict() for match in matched.matches],
        "unmatched_a": list(matched.unmatched_a),
        "unmatched_b": list(matched.unmatched_b),
        "roi": roi.to_dict() if roi is not None else None,
        "recommended_action": recommended,
        "policy_hash": effective.policy_hash,
        "input_a_hash": input_a_hash,
        "input_b_hash": input_b_hash,
    }
    assessment_hash = canonical_json_sha256(payload)
    return CrossEvidenceAssessment(
        engine_a=observation_a.engine,
        engine_b=observation_b.engine,
        task_a=observation_a.task,
        task_b=observation_b.task,
        status_a=result_a.status.value,
        status_b=result_b.status.value,
        status_agree=status_agree,
        score_a=float(result_a.score),
        score_b=float(result_b.score),
        score_gap=score_gap,
        flags=ordered,
        matches=matched.matches,
        unmatched_a=matched.unmatched_a,
        unmatched_b=matched.unmatched_b,
        roi=roi,
        recommended_action=recommended,
        policy_hash=effective.policy_hash,
        input_a_hash=input_a_hash,
        input_b_hash=input_b_hash,
        assessment_hash=assessment_hash,
    )


# ---------------------------------------------------------------------------
# dual overlay (두 엔진 증거를 한 장에)
# ---------------------------------------------------------------------------


def draw_dual_overlay(
    bgr: np.ndarray,
    observation_a: EngineObservation,
    observation_b: EngineObservation,
    assessment: CrossEvidenceAssessment,
    *,
    color_a: tuple[int, int, int] = DUAL_COLOR_A,
    color_b: tuple[int, int, int] = DUAL_COLOR_B,
    thickness: int = 2,
) -> np.ndarray:
    """엔진 A/B 박스를 다른 색으로 그리고 매칭 쌍은 중심 연결선으로 잇는다.

    입력 배열은 변경하지 않는다. 매칭 선 위에 IoU 값을 적어 증거를 바로 읽을
    수 있게 한다.
    """

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"BGR 3채널 이미지가 필요합니다: shape={bgr.shape}")
    canvas = bgr.copy()
    components_a = {c.index: c for c in observation_a.result.components}
    components_b = {c.index: c for c in observation_b.result.components}

    for index, component in sorted(components_a.items()):
        x, y, w, h = component.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color_a, thickness)
        cv2.putText(
            canvas, f"A{index}", (x, max(8, y - 4)), _FONT, _FONT_SCALE, color_a,
            _TEXT_THICKNESS, cv2.LINE_8,
        )
    for index, component in sorted(components_b.items()):
        x, y, w, h = component.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color_b, thickness)
        cv2.putText(
            canvas, f"B{index}", (x, max(8, y + h + 12)), _FONT, _FONT_SCALE, color_b,
            _TEXT_THICKNESS, cv2.LINE_8,
        )
    for match in assessment.matches:
        left = components_a.get(match.index_a)
        right = components_b.get(match.index_b)
        if left is None or right is None:
            continue
        center_a = bbox_center(left.bbox)
        center_b = bbox_center(right.bbox)
        point_a = (int(round(center_a[0])), int(round(center_a[1])))
        point_b = (int(round(center_b[0])), int(round(center_b[1])))
        cv2.line(canvas, point_a, point_b, DUAL_MATCH_COLOR, 1, cv2.LINE_8)
        midpoint = (
            int(round((center_a[0] + center_b[0]) / 2.0)),
            int(round((center_a[1] + center_b[1]) / 2.0)),
        )
        cv2.putText(
            canvas, f"{match.iou:.2f}", midpoint, _FONT, _FONT_SCALE, DUAL_MATCH_COLOR,
            _TEXT_THICKNESS, cv2.LINE_8,
        )
    return canvas


def save_dual_overlay(
    bgr: np.ndarray,
    observation_a: EngineObservation,
    observation_b: EngineObservation,
    assessment: CrossEvidenceAssessment,
    path: str | Path,
    *,
    color_a: tuple[int, int, int] = DUAL_COLOR_A,
    color_b: tuple[int, int, int] = DUAL_COLOR_B,
    thickness: int = 2,
) -> Path:
    """dual overlay 를 PNG 로 저장하고 경로를 돌려준다(부모 디렉터리 자동 생성)."""

    target = Path(path)
    if target.suffix.lower() != ".png":
        target = target.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_dual_overlay(
        bgr, observation_a, observation_b, assessment,
        color_a=color_a, color_b=color_b, thickness=thickness,
    )
    if not cv2.imwrite(str(target), overlay):
        raise OSError(f"dual overlay 저장 실패: {target}")
    return target


__all__ = [
    "DUAL_COLOR_A",
    "DUAL_COLOR_B",
    "DUAL_MATCH_COLOR",
    "EVIDENCE_POLICY_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "ComponentMatch",
    "CrossEvidenceAssessment",
    "EvidenceFlag",
    "EvidencePolicy",
    "GreedyMatchResult",
    "RoiComparison",
    "assess_cross_evidence",
    "bbox_center",
    "bbox_intersection_area",
    "bbox_iou",
    "default_evidence_policy",
    "draw_dual_overlay",
    "greedy_match",
    "pair_metrics",
    "roi_comparison",
    "save_dual_overlay",
]
