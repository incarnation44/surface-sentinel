# ADR 0003 — 원본 이미지와 파생 이미지는 커밋하지 않고 메타데이터만 커밋한다

- 상태: 확정 · 2026-11-10 (T03에서 확정, 소급 기록)
- 맥락: KSDD2는 CC BY-NC-SA 4.0이고, 원본 3,335장과 contact sheet·defect_map
  같은 파생 이미지는 저장소 크기를 빠르게 키운다.
- 결정: `data/manifests`, `data/splits`, `data/audit` 등 **텍스트 메타데이터만**
  Git에 넣는다. `data/raw/`와 `data/derived/`는 무시 규칙으로 제외한다.
  원본은 사용자가 직접 반입하고, 경로만 문서에 적는다.
- 대안: (a) 원본 커밋 — 라이선스 재배포 문제와 용량 초과로 폐기. (b) LFS —
  공용 PC·다중 기기 환경에서 인증·용량 관리 부담이 커 폐기. (c) 파생 이미지만
  커밋 — 95MB 한도와 재생성 가능성 때문에 폐기.
- 근거: `.gitignore`의 `data/raw/`, `data/derived/` 규칙. T03 산출물(manifest
  3,336행, split, audit JSON/MD)은 커밋되어 clone만으로 재현 가능하다.
- 결과: 다른 기기에서 `clone` 직후 데이터 검증은 `data` 마커로 선택 해제되고,
  원본을 반입한 뒤 `data build-manifest`부터 다시 돌린다.
