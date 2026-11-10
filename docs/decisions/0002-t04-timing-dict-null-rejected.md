# ADR 0002 — `timing_ms`는 단계별 dict로 채우고 null은 스키마에서 거부한다

- 상태: 확정 · 2026-11-10 (T04에서 확정, 소급 기록)
- 맥락: T04 파이프라인은 10단계다. 소요 시간을 총합 하나로만 남기면 어느 단계가
  느린지 알 수 없고, T06의 p50/p95도 근거를 잃는다.
- 결정: `SampleResult.timing_ms`를 `{단계명: ms, "total": ms}` dict로 채우고,
  null 또는 빈 dict는 스키마 검증에서 거부한다.
- 대안: (a) 총합 단일 float — 병목 진단 불가로 폐기. (b) 선택 필드 — 빠뜨린 샘플이
  조용히 섞여 p50/p95가 오염되므로 폐기.
- 근거: `pipeline/schema.py`가 null을 거부하고, `inspect` 출력에 10단계 + total이
  실제로 찍힌다(`docs/evidence/T04_summary.txt`).
- 결과: T06 `run.json.timing`이 p50/p95를 계산할 수 있게 됐다. 단계 추가·개명은
  스키마 변경이므로 승인 대상이다.
