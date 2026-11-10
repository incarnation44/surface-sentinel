# ADR 0005 — run 저장은 임시 디렉터리 + fsync + os.replace, run.json은 마지막

- 상태: 확정 · 2026-11-10 (T06)
- 맥락: batch가 중간에 죽거나 디스크가 실패하면 반쯤 쓰인 run 디렉터리가 남는다.
  그 디렉터리를 완료된 run으로 오인하면 평가·리포트가 오염된다.
- 결정: `runs/.tmp-<run_id>-<pid>/`에 모든 산출물을 쓰고 fsync한 뒤
  `os.replace`로 최종 이름으로 교체한다. `run.json`은 **항상 마지막**에 쓴다.
  replace 실패 시 최종 디렉터리를 만들지 않고 임시 디렉터리를 지운 뒤 exit 5.
- 대안: (a) 최종 경로에 바로 쓰기 — 부분 산출물이 남아 폐기. (b) `.tmp` 후 rename
  만 하고 fsync 생략 — 크래시 시 0바이트 파일이 남아 폐기.
- 근거: `io/atomic.py`, `batch/runner.py`. `os.replace` monkeypatch 실패 주입
  테스트에서 최종 디렉터리 0개·exit 5·COMPLETE 없음이 확인된다.
- 세부 결정 2건(명세에 없던 부분):
  - `--raw-root` 추가: manifest·split의 경로가 원본 루트 상대경로라 필요하다.
  - `stages/` 생략: §5.2의 선택 항목이고 `inspect`가 쓰는 `_response.png` 이름과
    충돌해 run 계약이 흐려지므로 만들지 않는다. 필요해지면 별도 승인.
- 결과: 실패한 저장은 "없던 일"이 되고, 완료 판정은 `run.json` 존재로 단일화된다.
  Windows는 디렉터리 fsync를 지원하지 않아 파일 fsync만 수행한다.
