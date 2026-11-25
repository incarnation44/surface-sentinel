"""검사 파이프라인 단계별 계약 오류(`PipelineError`) 정의.

오류 코드와 발생 단계를 구조화해 오케스트레이터가 일관된 ERROR 결과로 변환할 수 있게 한다.
"""

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
