"""단일 이미지 업로드 검사 및 기록 조회용 로컬 FastAPI 서버 패키지.

``create_inspection_app``가 ``runs_dir`` 아래에 업로드·오버레이·DB를 만들고
``/api/inspect``·``/api/inspections``·``/api/stats``·``/healthz``를 제공한다.
127.0.0.1 전용, 외부 요청 없음. 기계 판정 OK/NG는 불변이고 사람 판정은
append-only로만 쌓인다.
"""

from __future__ import annotations

from .app import create_inspection_app, engine_status_counts
from .schemas import (
    AgentAssessmentOut,
    EngineName,
    EngineObservationOut,
    InspectionDetail,
    InspectionList,
    InspectResponse,
    ReviewDecision,
    ReviewRequest,
    StatsResponse,
)

__all__ = [
    "AgentAssessmentOut",
    "EngineName",
    "EngineObservationOut",
    "InspectResponse",
    "InspectionDetail",
    "InspectionList",
    "ReviewDecision",
    "ReviewRequest",
    "StatsResponse",
    "create_inspection_app",
    "engine_status_counts",
]
