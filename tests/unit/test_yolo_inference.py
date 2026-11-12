"""YOLO ONNX 추론 전처리·후처리 단위 테스트(합성 배열, ONNX 불필요).

모든 좌표는 픽셀 단위이며, letterbox된 416 좌표와 원본 좌표를 구분한다.
"""

from __future__ import annotations

import numpy as np

from surface_sentinel.yolo.inference import (
    Detection,
    decode_det,
    decode_seg_masks,
    letterbox,
    nms,
    to_input_tensor,
    undo_letterbox,
)


def test_letterbox_portrait_keeps_aspect() -> None:
    img = np.zeros((637, 229, 3), np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(img, 416)
    assert out.shape == (416, 416, 3)
    assert scale == 416 / 637
    assert pad_x > 0 and pad_y == 0


def test_letterbox_landscape_keeps_aspect() -> None:
    img = np.zeros((229, 637, 3), np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(img, 416)
    assert out.shape == (416, 416, 3)
    assert scale == 416 / 637
    assert pad_x == 0 and pad_y > 0


def test_letterbox_pads_with_114() -> None:
    img = np.full((100, 50, 3), 200, np.uint8)
    out, _, (pad_x, _pad_y) = letterbox(img, 416)
    assert out[0, 0, 0] == 114
    assert out[208, 208, 0] == 200
    assert pad_x > 0


def test_undo_letterbox_roundtrip() -> None:
    box = np.array([200.0, 100.0, 240.0, 160.0])  # xyxy, 416 letterbox 좌표
    scale, pad_x, pad_y = 416 / 637, 133.0, 0.0
    orig = undo_letterbox(box, scale, pad_x, pad_y, orig_w=229, orig_h=637)
    assert orig[2] > 0 and orig[3] > 0
    assert 0 <= orig[0] <= 229 and 0 <= orig[1] <= 637


def test_undo_letterbox_clamps_to_image() -> None:
    box = np.array([-40.0, -10.0, 600.0, 900.0])  # 경계 밖
    orig = undo_letterbox(box, 416 / 637, 133.0, 0.0, orig_w=229, orig_h=637)
    assert orig == (0.0, 0.0, 229.0, 637.0)


def test_nms_suppresses_overlap() -> None:
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 240, 240]], float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]


def test_nms_keeps_disjoint_and_is_deterministic() -> None:
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    scores = np.array([0.5, 0.5])
    assert nms(boxes, scores, iou_threshold=0.5) == [0, 1]
    assert nms(boxes, scores, iou_threshold=0.5) == nms(boxes, scores, iou_threshold=0.5)


def test_nms_empty_input() -> None:
    assert nms(np.zeros((0, 4)), np.zeros((0,)), iou_threshold=0.5) == []


def test_decode_det_filters_by_conf() -> None:
    raw = np.zeros((1, 5, 3), np.float32)
    raw[0, :4, 0] = [208, 208, 40, 40]
    raw[0, 4, 0] = 0.9
    raw[0, :4, 1] = [100, 100, 20, 20]
    raw[0, 4, 1] = 0.05
    raw[0, :4, 2] = [300, 300, 30, 30]
    raw[0, 4, 2] = 0.6
    boxes, scores = decode_det(raw, conf_threshold=0.25)
    assert len(boxes) == 2 and scores[0] >= scores[1]


def test_decode_det_boxes_are_xyxy() -> None:
    raw = np.zeros((1, 5, 1), np.float32)
    raw[0, :4, 0] = [100, 100, 40, 20]  # cx, cy, w, h
    raw[0, 4, 0] = 0.8
    boxes, _scores = decode_det(raw, conf_threshold=0.25)
    assert boxes[0].tolist() == [80.0, 90.0, 120.0, 110.0]


def test_decode_det_empty_when_all_below_threshold() -> None:
    raw = np.zeros((1, 5, 2), np.float32)
    raw[0, 4, :] = 0.1
    boxes, scores = decode_det(raw, conf_threshold=0.25)
    assert boxes.shape == (0, 4) and scores.shape == (0,)


def test_to_input_tensor_rgb_nchw_normalized() -> None:
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 2] = 30  # R
    tensor = to_input_tensor(bgr)
    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == np.float32(30 / 255)  # R 먼저
    assert tensor[0, 2, 0, 0] == np.float32(10 / 255)  # B 는 3번째 채널


def test_decode_seg_masks_binary_and_original_size() -> None:
    # proto 4x4, 1개 계수, 박스 하나(좌상단 8x8 픽셀)
    proto = np.full((1, 1, 4, 4), 4.0, np.float32)
    coeffs = np.ones((1, 1), np.float32)
    boxes = np.array([[0.0, 0.0, 8.0, 8.0]], np.float32)
    masks = decode_seg_masks(
        coeffs,
        proto,
        boxes,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        orig_w=16,
        orig_h=16,
        input_size=8,
    )
    assert len(masks) == 1
    mask = masks[0]
    assert mask.shape == (16, 16)
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask[0, 0] == 1
    assert mask[15, 15] == 0  # 박스 밖은 0


def test_detection_is_frozen_and_mask_not_compared() -> None:
    a = Detection(bbox_xyxy=(0.0, 0.0, 4.0, 4.0), score=0.5)
    b = Detection(bbox_xyxy=(0.0, 0.0, 4.0, 4.0), score=0.5)
    assert a == b
    mask_a = np.ones((4, 4), np.uint8)
    mask_b = np.zeros((4, 4), np.uint8)
    c = Detection(bbox_xyxy=(0.0, 0.0, 4.0, 4.0), score=0.5, mask=mask_a)
    d = Detection(bbox_xyxy=(0.0, 0.0, 4.0, 4.0), score=0.5, mask=mask_b)
    assert c == d  # numpy 필드는 eq=False → 객체 동일성 비교 아님
