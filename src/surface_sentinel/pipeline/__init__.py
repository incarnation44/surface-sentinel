"""결정론적 표면 검사 파이프라인."""

from .inspect import inspect_image
from .schema import SchemaError, validate_sample_result
from .types import InspectionOutcome

__all__ = [
    "InspectionOutcome",
    "SchemaError",
    "inspect_image",
    "validate_sample_result",
]
