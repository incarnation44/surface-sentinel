# ADR 0007 — 엣지 확장 1~5단계: YOLO 비교·검사 API·기록 DB 도입, 6단계(RKNN) 보류

- 상태: 확정 · 2026-11-12 (확장 설계의 1~5단계 수행, 6단계는 보드 구매 보류로 제외)
- 맥락: 새 설계 문서가 6단계 로드맵을 제시했다. 1) OpenCV 규칙 기반 기준선 유지,
  2) YOLO Detection vs Segmentation 비교, 3) 결함 시각화·자동 판정, 4) FastAPI 검사
  시스템, 5) 검사 기록 DB, 6) RKNN/Orange Pi 엣지 배포. 6단계는 보드 구매 보류로
  수행하지 않는다. 기존 계약(OK/NG/REVIEW 3-class, 기계 OK/NG 불변, append-only
  검토 로그, 원자 run 저장, sealed test 동결 후 1회, T07 UI·테스트·증거)은 보존한다.
  개발 장비는 i5-1345U PC(CUDA 없음, Docker 없음)이다.
- 결정:
  1. 같은 저장소에 가산(additive) 모듈로 확장한다. 기존 `pipeline/`·`contracts.py`·
     `ui/`·테스트는 보존하고, 기존 T08–T11은 HOLD 유지, sealed test는 접근하지 않는다.
  2. YOLO 학습은 격리 환경 `.venv-train`(torch CPU + ultralytics)에서 수행한다.
     ultralytics의 `opencv-python` 의존이 메인 venv의 `opencv-python-headless`와
     cv2 파일을 충돌시키는 것을 회피한다. 학습 산출물은 ONNX로 export한다.
  3. 제품 추론은 ONNX Runtime(선택 extra `yolo`)과 자체 결정론적 후처리
     (letterbox·decode·NMS·마스크 디코드)로 수행한다. torch/ultralytics 런타임
     의존이 없어 CI 기본 `uv sync --frozen`이 불변이다. ONNX 산출물은 향후
     6단계 RKNN 변환 경로와 호환된다.
  4. 기록 DB는 SQLite로 시작한다(설계가 명시 허용). 스키마는 PostgreSQL 호환으로
     작성해 이전 경로를 남긴다. Docker는 요구하지 않는다.
  5. YOLO 학습/검증 분할은 tune 2,001장 내부에서 manifest `group_id` 기반
     group-aware 분할로 만든다(같은 부품 그룹의 누출 방지). 평가는 tune에서만
     수행하고 기존 `evaluation/metrics.py`의 동일 지표(분모·Wilson CI)를 쓴다.
  6. 워커는 결정론적이다: `incoming/` 감시 → 검사 → DB 기록, sha256 중복 제거,
     재시도, 감사 로그. 네트워크·셸 실행 없음. 제품 판정 소유권은 파이프라인이
     유지하며 워커는 판정을 바꾸지 않는다.
  7. 저장 정책: NG 원본은 전수 보관, OK 원본은 결정론적 해시 샘플링으로 보관한다.
- 대안:
  - 메인 venv에 ultralytics 직접 설치 — opencv-python/headless 충돌로 cv2 덮어쓰기
    위험이 있고 CI 기본 동기화가 깨진다.
  - torch 런타임 추론 — 의존성이 무겁고 CI에 torch 설치가 필요해 재현성이 낮아진다.
  - PostgreSQL/Docker 즉시 도입 — 개발 PC에 Docker가 없고 설계가 SQLite 시작을 허용한다.
  - 랜덤 train/val 분할 — 같은 그룹 이미지가 양쪽에 섞여 누출이 생긴다. group-aware 필수.
- 근거: 확장 설계 문서(1~5단계 수행·6단계 제외),
  E0 실측(전체 350 passed, ruff clean, config hash `a762ba639441…` 일치,
  배치 스모크 run `20261112T102419Z-a762ba63-tune` COMPLETE).
- 결과: E1–E5 트랙(plan.md §10)으로 수행한다. 성능 주장은 측정 후에만 기록한다.
  6단계는 보드 확보 후 별도 설계 개정으로 다룬다.