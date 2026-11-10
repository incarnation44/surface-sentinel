"""파이프라인 단계에서 발생하는 계약 오류."""

from __future__ import annotations

from ..contracts import ErrorCode, Stage


class PipelineError(RuntimeError):
    """단계 실패를 오류 코드와 단계 이름으로 표현한다."""

    def __init__(self, error_code: ErrorCode, error_stage: Stage, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_stage = error_stage
        self.message = message

    def __str__(self) -> str:
        return f"[{self.error_code.value} @ {self.error_stage.value}] {self.message}"


__all__ = ["PipelineError"]
