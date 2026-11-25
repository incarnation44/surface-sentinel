"""ultralytics 없이 ONNX Runtime만으로 YOLO det/seg 전처리·추론·결정론적 NMS를 수행하는 엔진.

ultralytics 런타임 없이 ONNX 모델을 직접 실행한다. onnxruntime은
optional extra ``yolo`` 로 설치하며, 이 모듈 자체는 세션 생성 시점에만
lazy import 하므로 extra 미설치 환경에서도 import 가 깨지지 않는다.

출력 텐서 shape 는 하드코딩하지 않고 실제 shape 에서 유도한다.
(기대 shape: det output0 ``[1, 5, 3549]``, seg output0 ``[1, 37, 3549]`` +
output1 ``[1, 32, 104, 104]``, 416 입력 기준 — [NOT_MEASURED] export 후
통합 테스트에서 실측 확정)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: letterbox 패딩 값(ultralytics 관례와 동일).
LETTERBOX_VALUE = 114
#: 기본 네트워크 입력 한 변(정사각).
DEFAULT_INPUT_SIZE = 416
#: 기본 NMS IoU 임계값.
DEFAULT_IOU_THRESHOLD = 0.5
#: 기본 confidence 임계값. 정책 임계값(tune-val 조정)과는 별개다.
DEFAULT_CONF_THRESHOLD = 0.25


@dataclass(frozen=True)
class Detection:
    """탐지 하나. 좌표는 원본 이미지 픽셀 단위 ``xyxy``.

    ``mask`` 는 segmentation 전용이며 원본 크기 ``uint8`` {0, 1} 이다.
    numpy 배열은 동결 dataclass 의 비교 대상에서 제외한다(``compare=False``).
    """

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


def letterbox(
    image: np.ndarray,
    size: int = DEFAULT_INPUT_SIZE,
    *,
    value: int = LETTERBOX_VALUE,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """비율을 유지하며 ``size x size`` 로 리사이즈하고 114 로 패딩한다.

    반환값은 ``(패딩된 이미지, scale, (left, top))``. ``scale`` 은 원본 →
    letterbox 배율이며, 역변환은 :func:`undo_letterbox` 가 담당한다.
    """

    if image.ndim != 3:
        raise ValueError(f"이미지는 HWC 3채널이어야 합니다: shape={image.shape}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"이미지 크기가 0입니다: shape={image.shape}")

    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.full((size, size, image.shape[2]), value, dtype=image.dtype)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out, scale, (left, top)


def undo_letterbox(
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    scale: float,
    pad_x: float,
    pad_y: float,
    *,
    orig_w: int,
    orig_h: int,
) -> tuple[float, float, float, float]:
    """letterbox 좌표 ``xyxy`` 를 원본 좌표로 복원하고 이미지 안으로 클램프한다."""

    if scale <= 0:
        raise ValueError(f"scale 은 0보다 커야 합니다: scale={scale}")
    x1, y1, x2, y2 = (float(value) for value in np.asarray(box_xyxy, dtype=np.float64).reshape(4))
    left = (x1 - pad_x) / scale
    top = (y1 - pad_y) / scale
    right = (x2 - pad_x) / scale
    bottom = (y2 - pad_y) / scale
    return (
        float(min(max(left, 0.0), orig_w)),
        float(min(max(top, 0.0), orig_h)),
        float(min(max(right, 0.0), orig_w)),
        float(min(max(bottom, 0.0), orig_h)),
    )


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """``(N, 4)`` x ``(M, 4)`` xyxy 박스의 IoU 행렬 ``(N, M)``. 결정론적."""

    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[int]:
    """클래스 무관 NMS. 점수 내림차순 안정 정렬이라 결과가 결정론적이다(RNG 없음)."""

    box_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if box_array.shape[0] != score_array.shape[0]:
        raise ValueError(
            f"boxes와 scores 길이가 다릅니다: {box_array.shape[0]} != {score_array.shape[0]}"
        )
    if box_array.shape[0] == 0:
        return []

    order = np.argsort(-score_array, kind="stable")
    ious = _iou_matrix(box_array, box_array)
    suppressed = np.zeros(box_array.shape[0], dtype=bool)
    keep: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        keep.append(int(index))
        suppressed |= ious[index] > iou_threshold
        suppressed[index] = True
    return keep


#: 출력 텐서 shape 자동 추론 상한(전치 export ``[1, N, 4+nc]`` 오디코드 방어).
MAX_INFERRED_CLASSES = 1000
MAX_AUTO_TRANSPOSE_ROWS = 128
MAX_PROTO_CHANNELS = 256


def decode_det(
    raw: np.ndarray,
    *,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """detection 출력 ``[1, 4+nc, N]`` → ``(boxes_xyxy, scores)``.

    좌표는 네트워크 입력(letterbox) 좌표계다. 점수 내림차순 정렬이라
    결정론적이며, 점수가 같으면 원래 열 순서를 유지한다(안정 정렬).
    """

    pred = np.asarray(raw, dtype=np.float32)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"batch 크기 1만 지원합니다: shape={pred.shape}")
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"출력은 2차원이어야 합니다: shape={pred.shape}")

    rows, columns = pred.shape
    if rows > 32 and columns < 16 and columns < rows:
        raise ValueError(
            "소형 N 전치 export([1, N, 4+nc])는 거부합니다: "
            f"rows={rows} columns={columns}"
        )
    if num_classes is None:
        inferred = rows - 4
        if inferred > MAX_INFERRED_CLASSES or (
            rows > MAX_AUTO_TRANSPOSE_ROWS and columns < rows
        ):
            raise ValueError(
                "출력 텐서가 전치([1, N, 4+nc])됐거나 클래스 수가 비정상적으로 큽니다: "
                f"rows={rows} columns={columns}"
            )
        num_classes = inferred
    if num_classes < 1 or rows < 4 + num_classes:
        raise ValueError(f"출력 행이 부족합니다: rows={rows} num_classes={num_classes}")
    if rows > 4 + num_classes + MAX_PROTO_CHANNELS:
        raise ValueError(
            f"출력 행이 비정상적으로 많습니다: rows={rows} num_classes={num_classes}"
        )

    centers = pred[:4].T  # (N, 4) cx, cy, w, h
    class_scores = pred[4 : 4 + num_classes]  # (nc, N)
    conf = class_scores.max(axis=0).astype(np.float32)

    xyxy = np.empty((columns, 4), dtype=np.float32)
    x_center, y_center, width, height = (
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        centers[:, 3],
    )
    xyxy[:, 0] = x_center - width / 2.0
    xyxy[:, 1] = y_center - height / 2.0
    xyxy[:, 2] = x_center + width / 2.0
    xyxy[:, 3] = y_center + height / 2.0

    order = np.argsort(-conf, kind="stable")
    selected = order[conf[order] >= conf_threshold]
    return xyxy[selected], conf[selected]


def to_input_tensor(bgr: np.ndarray) -> np.ndarray:
    """BGR HWC → RGB CHW ``float32`` ``[0, 1]`` (batch 1)."""

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"BGR 3채널 이미지가 필요합니다: shape={bgr.shape}")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32) / 255.0, axis=0)


def decode_seg_masks(
    coeffs: np.ndarray,
    proto: np.ndarray,
    boxes_xyxy: np.ndarray,
    *,
    scale: float,
    pad_x: float,
    pad_y: float,
    orig_w: int,
    orig_h: int,
    input_size: int = DEFAULT_INPUT_SIZE,
    mask_threshold: float = 0.5,
) -> list[np.ndarray]:
    """mask 계수 × prototype → 원본 크기 이진 마스크 ``uint8`` {0, 1}.

    ``coeffs`` 는 ``(N, C)``, ``proto`` 는 ``(1, C, Hp, Wp)`` 또는 ``(C, Hp, Wp)``.
    마스크는 proto 해상도에서 박스 영역만 잘라 원본 박스 크기로 리사이즈한 뒤
    원본 좌표 캔버스에 붙인다(박스 밖은 0).
    """

    coeff_array = np.asarray(coeffs, dtype=np.float32)
    proto_array = np.asarray(proto, dtype=np.float32)
    box_array = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if proto_array.ndim == 4:
        if proto_array.shape[0] != 1:
            raise ValueError(f"proto batch 크기 1만 지원합니다: shape={proto_array.shape}")
        proto_array = proto_array[0]
    if proto_array.ndim != 3:
        raise ValueError(f"proto 는 3차원이어야 합니다: shape={proto_array.shape}")
    if coeff_array.ndim == 1:
        coeff_array = coeff_array.reshape(1, -1)
    if coeff_array.shape[0] != box_array.shape[0]:
        raise ValueError(
            f"계수 행과 박스 수가 다릅니다: {coeff_array.shape[0]} != {box_array.shape[0]}"
        )
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"원본 크기가 0입니다: orig_w={orig_w} orig_h={orig_h}")

    channels, proto_h, proto_w = proto_array.shape
    if coeff_array.shape[1] < channels:
        raise ValueError(
            f"마스크 계수 채널이 부족합니다: coeffs={coeff_array.shape[1]} < proto={channels}"
        )
    coeff_array = coeff_array[:, :channels]
    logits = coeff_array @ proto_array.reshape(channels, -1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    proto_scale_x = input_size / proto_w
    proto_scale_y = input_size / proto_h

    masks: list[np.ndarray] = []
    for prob_row, box in zip(probs, box_array, strict=True):
        canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
        x1, y1, x2, y2 = undo_letterbox(
            box, scale, pad_x, pad_y, orig_w=orig_w, orig_h=orig_h
        )
        if x2 <= x1 or y2 <= y1:
            masks.append(canvas)
            continue

        px1 = int(np.clip(np.floor(box[0] / proto_scale_x), 0, proto_w - 1))
        py1 = int(np.clip(np.floor(box[1] / proto_scale_y), 0, proto_h - 1))
        px2 = int(np.clip(np.ceil(box[2] / proto_scale_x), px1 + 1, proto_w))
        py2 = int(np.clip(np.ceil(box[3] / proto_scale_y), py1 + 1, proto_h))

        crop = prob_row.reshape(proto_h, proto_w)[py1:py2, px1:px2]
        target_w = max(1, int(round(x2 - x1)))
        target_h = max(1, int(round(y2 - y1)))
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        binary = (resized >= mask_threshold).astype(np.uint8)

        ox1 = int(round(x1))
        oy1 = int(round(y1))
        ox2 = min(orig_w, ox1 + target_w)
        oy2 = min(orig_h, oy1 + target_h)
        if ox2 > ox1 and oy2 > oy1:
            canvas[oy1:oy2, ox1:ox2] = binary[: oy2 - oy1, : ox2 - ox1]
        masks.append(canvas)
    return masks


def decode_seg(
    raw: np.ndarray,
    proto: np.ndarray,
    *,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    mask_threshold: float = 0.5,
    input_size: int = DEFAULT_INPUT_SIZE,
    scale: float = 1.0,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
    orig_w: int,
    orig_h: int,
    num_classes: int | None = None,
) -> list[Detection]:
    """segmentation 출력(output0 + proto) → 원본 좌표 :class:`Detection` 목록.

    클래스 수는 shape 에서 유도한다(``4 + nc + C = output0 행 수``,
    ``C = proto 채널``). 후처리 순서: conf 필터 → NMS → 마스크 생성.
    """

    pred = np.asarray(raw, dtype=np.float32)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"batch 크기 1만 지원합니다: shape={pred.shape}")
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"output0 은 2차원이어야 합니다: shape={pred.shape}")

    rows, columns = pred.shape
    if rows > 32 and columns < 16 and columns < rows:
        raise ValueError(
            "소형 N 전치 export([1, N, 4+nc+C])는 거부합니다: "
            f"rows={rows} columns={columns}"
        )

    proto_array = np.asarray(proto, dtype=np.float32)
    proto_channels = int(proto_array.shape[1] if proto_array.ndim == 4 else proto_array.shape[0])
    if proto_channels < 1 or proto_channels > MAX_PROTO_CHANNELS:
        raise ValueError(f"proto 채널 수가 비정상입니다: proto_channels={proto_channels}")
    if num_classes is None:
        inferred = pred.shape[0] - 4 - proto_channels
        if inferred > MAX_INFERRED_CLASSES or (
            pred.shape[0] > MAX_AUTO_TRANSPOSE_ROWS and pred.shape[1] < pred.shape[0]
        ):
            raise ValueError(
                "segmentation 출력 텐서가 전치([1, N, 4+nc+C])됐거나 채널 합이 비정상입니다: "
                f"rows={pred.shape[0]} columns={pred.shape[1]} proto_channels={proto_channels}"
            )
        num_classes = inferred
    if num_classes < 1 or pred.shape[0] != 4 + num_classes + proto_channels:
        raise ValueError(
            f"클래스 수/채널 합이 맞지 않습니다: rows={pred.shape[0]} "
            f"num_classes={num_classes} proto_channels={proto_channels}"
        )

    boxes, scores = decode_det(
        pred,
        conf_threshold=conf_threshold,
        num_classes=num_classes,
    )
    keep = nms(boxes, scores, iou_threshold=iou_threshold)
    if not keep:
        return []

    kept_boxes = boxes[keep]
    kept_scores = scores[keep]
    coeffs = pred[4 + num_classes :].T[keep]
    masks = decode_seg_masks(
        coeffs,
        proto_array,
        kept_boxes,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
        input_size=input_size,
        mask_threshold=mask_threshold,
    )

    detections: list[Detection] = []
    for box, score, mask in zip(kept_boxes, kept_scores, masks, strict=True):
        bbox = undo_letterbox(box, scale, pad_x, pad_y, orig_w=orig_w, orig_h=orig_h)
        detections.append(Detection(bbox_xyxy=bbox, score=float(score), mask=mask))
    return detections


def _create_session(model_path: Path, providers: list[str] | None) -> Any:
    """onnxruntime 세션을 만든다(optional extra: 이 함수 안에서만 import)."""

    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - extra 설치 환경에서는 실행되지 않음
        raise ImportError(
            "onnxruntime 이 없습니다. `uv sync --extra yolo` 로 설치하세요."
        ) from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers or ["CPUExecutionProvider"],
    )


class YoloOnnxEngine:
    """ONNX Runtime 세션 래퍼(det/seg 공용).

    ``onnxruntime`` 은 이 클래스 생성 시점에만 import 한다. extra ``yolo`` 가
    설치되지 않은 환경에서도 이 모듈의 import·단위 테스트는 동작한다.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        task: str = "det",
        input_size: int = DEFAULT_INPUT_SIZE,
        providers: list[str] | None = None,
    ) -> None:
        if task not in ("det", "seg"):
            raise ValueError(f"task 는 'det' 또는 'seg' 입니다: task={task!r}")
        self.model_path = Path(model_path)
        self.task = task
        self.input_size = input_size
        self._session = _create_session(self.model_path, providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

    def infer(
        self,
        bgr: np.ndarray,
        *,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        mask_threshold: float = 0.5,
    ) -> list[Detection]:
        """BGR 이미지 한 장 → 원본 좌표 :class:`Detection` 목록."""

        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"BGR 3채널 이미지가 필요합니다: shape={bgr.shape}")
        height, width = bgr.shape[:2]

        padded, scale, (pad_x, pad_y) = letterbox(bgr, self.input_size)
        tensor = to_input_tensor(padded)
        outputs = self._session.run(self._output_names, {self._input_name: tensor})

        if self.task == "det":
            boxes, scores = decode_det(
                outputs[0], conf_threshold=conf_threshold
            )
            keep = nms(boxes, scores, iou_threshold=iou_threshold)
            return [
                Detection(
                    bbox_xyxy=undo_letterbox(
                        boxes[index], scale, pad_x, pad_y, orig_w=width, orig_h=height
                    ),
                    score=float(scores[index]),
                )
                for index in keep
            ]

        return decode_seg(
            outputs[0],
            outputs[1],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            mask_threshold=mask_threshold,
            input_size=self.input_size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=width,
            orig_h=height,
        )


__all__ = [
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_IOU_THRESHOLD",
    "LETTERBOX_VALUE",
    "Detection",
    "YoloOnnxEngine",
    "decode_det",
    "decode_seg",
    "decode_seg_masks",
    "letterbox",
    "nms",
    "to_input_tensor",
    "undo_letterbox",
]
