"""OpenCV 기준선과 YOLO 판정의 결합 규칙.

어느 한쪽이 NG 면 NG, REVIEW 면 REVIEW, ERROR 면 ERROR, 둘 다 OK 일 때만 OK.
한쪽 판정이 다른 쪽을 덮어써서 OK 로 되돌리는 경로는 없다.
"""

from __future__ import annotations

from ..contracts import Status


def combine(opencv: Status, yolo: Status) -> Status:
    """두 판정을 하나로 결합한다(우선순위 NG > REVIEW > ERROR > OK)."""

    if opencv is Status.NG or yolo is Status.NG:
        return Status.NG
    if opencv is Status.REVIEW or yolo is Status.REVIEW:
        return Status.REVIEW
    if opencv is Status.ERROR or yolo is Status.ERROR:
        return Status.ERROR
    return Status.OK


__all__ = ["combine"]
