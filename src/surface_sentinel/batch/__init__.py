"""데이터셋 단위 배치 검사 실행 패키지.

분할 CSV를 일괄 검사해 `runs/<run_id>/` 산출물과 요약을 원자적으로 생성한다.
"""

from .runner import (
    BATCH_SPLITS,
    MAX_RUN_ID_ATTEMPTS,
    RUN_ID_PATTERN,
    BatchError,
    BatchInputError,
    BatchOutcome,
    BatchStorageError,
    BatchUsageError,
    SealedSplitError,
    allocate_run_id,
    percentile,
    run_batch,
)

__all__ = [
    "BATCH_SPLITS",
    "MAX_RUN_ID_ATTEMPTS",
    "RUN_ID_PATTERN",
    "BatchError",
    "BatchInputError",
    "BatchOutcome",
    "BatchStorageError",
    "BatchUsageError",
    "SealedSplitError",
    "allocate_run_id",
    "percentile",
    "run_batch",
]
