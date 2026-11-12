"""검사 기록 저장소(ADR 0007).

SQLite 우선, PostgreSQL 호환 스키마. run 산출물 ingest와
결정론적 워커(incoming 감시 → 검사 → 기록)를 담당한다.
워커는 판정을 바꾸지 않으며 판정 소유권은 파이프라인에 있다.
"""