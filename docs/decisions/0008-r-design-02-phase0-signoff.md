# ADR 0008 — R-DESIGN-02 확정: Phase 0 무결성 기반(P0-2·P0-3) 구현 수용

- 상태: 확정 · 2026-11-13 (code_sha 주입 → 실 ONNX 짝 실측 → 재검증 → R-DESIGN-02 formal
  sign-off 순서로 수행 = Phase 0 개정안 수용)
- 맥락: `docs/design/agent-extension-proposal.md`(R-DESIGN-02 후보)는 판정 로드맵 2단계
  (P0-2 fingerprint 중복제거 · P0-3 평가/서빙 동일성)를 Phase 0으로 제안했다. §6 거버넌스는
  "착수 전 §1을 개정안으로 확정"을 요구했다. 이 ADR은
  구현·실측이 끝난 시점에 §7 미결 항목을 확정해 formal sign-off를 기록한다.
- 결정(확정 항목):
  1. **fingerprint 입력 집합** = engine · task · model_sha256 · config_hash · yolo_policy_hash ·
     combine_policy_hash · preprocessing_version · code_sha · code_dirty 의 canonical JSON SHA256
     (`records/fingerprint.py`). **image_sha256 제외**(별도 컬럼으로 UNIQUE 쌍을 이루므로 중복),
     **onnx_path 제외**(경로는 플랫폼 의존 — 모델 신원은 바이트 SHA). dedup 키는
     `UNIQUE(image_sha256, inspection_fingerprint)` 쌍.
  2. **code_sha 주입 위치** = 진입점 1회 계산: API 앱 생성(`create_inspection_app`) ·
     워커 시작(`run_worker`) · 평가 CLI(`run_yolo_evaluation`). 헬퍼는
     `evaluation/recalibration.py:git_code_identity()`(rev-parse + status --porcelain,
     실패 시 `("", False)`). 이미지별 핫패스에서는 subprocess를 호출하지 않는다.
     더러운 트리는 `code_dirty=True`로 fingerprint가 달라져 커밋 전 결과와 섞이지 않는다.
  3. **migration 롤백 정책** = 단일 트랜잭션(`records/db.py:migrate_to_fingerprint`).
     복사·백필·테이블 교체 후 `integrity_check`·`foreign_key_check` 검증, 실패 시
     `DatabaseError` → 트랜잭션 롤백으로 v1 원상 유지. 별도 백업 파일은 만들지 않는다
     (되돌림은 git·트랜잭션이 담당). 기존 id·`review_decisions`(사람 판정) 보존,
     과거 행은 `provenance_complete=0` 백필 — 모델 SHA를 추측해 채우지 않는다.
  4. **세션 재사용 동시성 모델** = 세션별 lock 직렬화 + 생성 전역 lock, 전역 싱글턴이 아니라
     API lifespan·워커 실행 범위·평가 실행 범위에 주입(`yolo/runtime.py:YoloEngineRegistry`).
     키 = (model_sha256, task, onnxruntime 버전, providers). lock vs 풀 중 lock 채택 —
     동시성 부하 실측 전 무제한 공유 금지 원칙.
  5. **버전 상수 확정** = `PREPROCESSING_VERSION="preprocess-v1"` ·
     `COMBINE_POLICY_VERSION="combine-v1"`. 전처리(letterbox·NMS·입력 변환) 또는 결합 규칙
     (NG > REVIEW > ERROR > OK) 변경 시 수동 bump — 같은 이미지도 다른 검사로 추적된다.
  6. **combined 명시 task** = `SURFACE_SENTINEL_YOLO_COMBINED_TASK` 필수(숨은 det 기본 제거).
     미설정 시 ValueError → API 400 · 워커 WorkerError. 평가 task = 서빙 task 보장.
- 대안(기각):
  - 이미지 SHA 단독 dedup 유지 — 엔진·모델·설정별 관찰이 덮이거나 잘못 재사용됨(F1~F3).
  - fingerprint에 onnx_path 포함 — 같은 모델이 다른 경로에 있으면 다른 검사로 오판.
  - code_sha를 이미지별 계산 — 워커 핫패스 subprocess 금지 규칙 위반.
  - migration 전 별도 백업 파일 생성 — 단일 트랜잭션 롤백으로 충분하고 사본 관리 부담만 늘어난다.
  - 전역 세션 싱글턴 — 실행 범위 밖 재사용·테스트 격리 불가. 주입이 검증 가능.
- 근거(실측):
  - 전체 테스트 **685 passed · 39.60s**(직전 639 → Phase 0 +39 → code_sha 주입 +7) · ruff clean.
  - **실 ONNX 평가·서빙 짝 실측**(val 400장 = defect 43 + good 357, onnxruntime 1.31.1):
    det·seg 모두 status 불일치 **0건** · score 최대차 **0.0** · fingerprint 일치(6개 신원 필드 전부).
    det 평가 24.49s/서빙 27.36s(평균 추론 64.6ms) · seg 28.63s/32.88s(78.51ms) ·
    세션 cold start 각 1회(재사용 확인). 증거 `docs/evidence/P0_parity_2026-11-13.json`,
    재현 `scripts/p0_parity_measure.py`(총 113.8s).
  - 동결 config hash `a762ba63…` 변경 0 · 봉인 test 미접촉(access_count=2 유지) ·
    `SampleResult.status` 판정 로직 변경 0(observe-only) · 기본 엔진 opencv 유지 ·
    combined 채택 보류 유지(E트랙 게이트 G1·G5 미통과).
- 결과: Phase 0(P0-2·P0-3)은 확정·구현·검증 완료. **§1.5 에이전트 전용 테이블 · §2 Phases 1~4 ·
  review-queue API/UI · dual overlay는 별도 개정 확정 전까지 구현 금지.**
  자동 채택·네트워크 없음 유지. 6단계(RKNN·Orange Pi) 보류 유지.
  이 ADR 이후 Phase 0 관련 문서 주장은 위 실측을 근거로 한다.
