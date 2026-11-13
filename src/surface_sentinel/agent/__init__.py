"""Constrained tool-using agent.

T09 제한된 에이전트(설계 §6): 결정론적 정책 엔진(``engine``), 제한된 도구,
append-only 감사와 replay(``trace``), 실행 하네스(``run``), 승격 비교
(``comparison``), 설정 로더(``config``). E5 결정론 워커는 ``worker`` 에 있다.
판정은 항상 파이프라인이 소유하며 네트워크/셸/설정 변경 도구는 없다.
"""

