"""FastAPI 검사 API(E4, ADR 0007).

``create_inspection_app``가 ``runs_dir`` 아래에 업로드·오버레이·DB를 만들고
``/api/inspect``·``/api/inspections``·``/api/stats``·``/healthz``를 제공한다.
127.0.0.1 전용, 외부 요청 없음. 기계 판정 OK/NG는 불변이고 사람 판정은
append-only로만 쌓인다.
"""

from __future__ import annotations

from .app import create_inspection_app, engine_status_counts
from .schemas import (
    EngineName,
    InspectionDetail,
    InspectionList,
    InspectResponse,
    ReviewDecision,
    ReviewRequest,
    StatsResponse,
)

__all__ = [
    "EngineName",
    "InspectResponse",
    "InspectionDetail",
    "InspectionList",
    "ReviewDecision",
    "ReviewRequest",
    "StatsResponse",
    "create_inspection_app",
    "engine_status_counts",
]
