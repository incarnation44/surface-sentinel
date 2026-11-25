"""결정론적 10단계 OpenCV 표면 결함 검사 파이프라인 패키지.

외부 호출자는 공개 진입점 `inspect_image`를 통해 단일 이미지 검사를 실행한다.
"""

from .inspect import inspect_image
from .schema import SchemaError, validate_sample_result
from .types import InspectionOutcome

__all__ = [
    "InspectionOutcome",
    "SchemaError",
    "inspect_image",
    "validate_sample_result",
]
