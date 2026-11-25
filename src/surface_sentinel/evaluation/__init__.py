"""정량 평가·기준선 비교·ablation·재보정·봉인 테스트 패키지.

공개 진입점은 ``run_evaluation`` 하나다. 세부 모듈은 다음과 같다.

============================  ============================================
``metrics``                   이미지/영역 지표, Wilson 95% 구간
``sampling``                  표본 설계(defect 전수 + good 층화)
``cache``                     샘플별 1회 파이프라인 캐시
``measure``                   후보 config 재채점·집계
``baselines``                 B0/B1/B2 독립 구현
``ablation``                  A1–A6 구조 ablation
``grid``                      튜닝 그리드(최대 24후보)와 선택 규칙
``failures``                  실패 색인과 §4.5 대표 사례 선택
``runner``                    전체 실행 진입점
============================  ============================================
"""

from .grid import MAX_CANDIDATES, TARGET_ESCAPE, TARGET_FALSE_NG, TARGET_REVIEW
from .measure import VariantReport, evaluate_cached
from .metrics import Ratio, wilson_interval
from .runner import EvalResult, run_evaluation
from .sampling import build_sample_plan

__all__ = [
    "MAX_CANDIDATES",
    "TARGET_ESCAPE",
    "TARGET_FALSE_NG",
    "TARGET_REVIEW",
    "EvalResult",
    "Ratio",
    "VariantReport",
    "build_sample_plan",
    "evaluate_cached",
    "run_evaluation",
    "wilson_interval",
]
