# T11 릴리스 무결성 점검 보고서 (2026-11-13)

- 점검일: 2026-11-14 (작업 계약 기준 보고서 식별자 `2026-11-13`)
- 작업 디렉터리: `C:\surface-sentinel`
- 동결 기준선 설정: `configs/base.toml` (`r20`, `config_hash = a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098`, 파일 바이트 sha256 `352e7d38afaa282778339a9d2b20e1684438c50f7f228c8d4a4fba995a8b78a8`)
- 에이전트 설정: `configs/agent.toml` 무수정 (기본 설계 계약 일치)
- 봉인 분할 원장: `data/splits/test.lock.json` (`access_count = 2`, `post_seal_config_change = false` 불변)

## 1. 저장소 무결성 및 금지 항목 점검

| 점검 항목 | 기준 | 실측 결과 | 판정 |
|---|---|---|---|
| Git 작성자 설정 | `user.name` / `user.email` 실사용자 정보 | `incarnation44` / `chunildo44@gmail.com` | **PASS** |
| 금지 파일 혼입 점검 | `.pt`, `.onnx`, `data/raw/`, `data/derived/`, `runs/` 산출물, `.venv*`, `.env*` 미포함 | 추적/미추적(non-ignored) 파일 중 금지 항목 0건 | **PASS** |
| 단일 파일 크기 상한 | Git 포함 전 파일 `< 2,000,000` bytes | 전수 검사 2MB 초과 파일 0건 | **PASS** |
| 라이선스 · 고지 | `LICENSE`, `NOTICE` 파일 존재 | 양쪽 모두 존재 확인 | **PASS** |
| 동결 설정 · 봉인 잠금 | `base.toml` 해시 불변, `agent.toml` 무수정, `test.lock.json` `access_count == 2` | 기준선 해시 및 잠금 원장 완전 일치 | **PASS** |
| 일본어/중국어 문자 점검 | 활성 코드 · 문서 · HTML 내 문자 오타 0건 | 활성 코드/문서/HTML 0건 (보존된 T08 과거 실행 로그 3건의 깨짐 이력 제외) | **PASS** |

## 2. 테스트 · 정적 분석 · 하드 게이트 · 재현성 실측

| 검증 명령 | 결과 원문 요약 | 판정 |
|---|---|---|
| `uv lock --check` | `Resolved 36 packages` (잠금 파일 일치) | **PASS** |
| `uv run --no-sync ruff check src tests scripts` | `All checks passed!` (오류 0건) | **PASS** |
| `PYTHONUTF8=1 uv run --no-sync pytest -o addopts="" -q` | `840 passed` (데이터·브라우저 포함 전수 통과, `v1.0.1` `b45d322`) | **PASS** |
| `PYTHONUTF8=1 uv run --no-sync pytest -m "not data and not browser" -o addopts="" -q` | `826 passed, 14 deselected` (CI 동일 명령) | **PASS** |
| `PYTHONUTF8=1 uv run --no-sync pytest -m browser -o addopts="" -q` | `8 passed, 832 deselected` (시스템 Chrome 채널, 다크 테마 포함) | **PASS** |
| `PYTHONUTF8=1 uv run --no-sync surface-sentinel gate` | `PASS (통과 6/6, REJECT 0건, REVISE 0건)` (`G_CONFIG_FROZEN`, `G_AGENT_CONFIG`, `G_DATA_LEAKAGE`, `G_SEAL_LOCK`, `G_EVIDENCE_DENOMINATORS`, `G_GUIDE_H7`) | **PASS** |
| `PYTHONUTF8=1 uv run --no-sync surface-sentinel repro check --trace <agent_trace.jsonl>` | `트레이스 재현성 확인 완료 (이벤트 9건, 종료 DONE/NG)` (`exit 0`) | **PASS** |

## 3. Clean Clone · 원격 반영 정책

- 커밋 후 시스템 임시 디렉터리에 로컬 저장소 Clean Clone을 생성해 금지 파일 0건 · 2MB 초과 0건 · `surface-sentinel gate` 및 비데이터/비브라우저 테스트 정상 동작을 검증한 뒤 임시 폴더를 제거한다.
- 원격 푸시는 일반 fast-forward (`git push origin main`)만 수행하며, 강제 푸시(`--force`), 과거 커밋 이력 재작성, 무단 태그 생성은 금지한다. Clean Clone 검증 이후 사용자 명시적 승인에 따라 `v1.0.0`(`8815a10`) 및 `v1.0.1`(`b45d322`) 태그를 생성·푸시 완료했다.

## 4. 성능 한계 및 비주장 사항 (은폐 없음)

- **봉인 `test` (1,334장, T08):** `escape` 14.79%(21/142, Wilson 95% 상한 **21.55% > 20% FAIL**), `false-NG` 7.21%(86/1,192), `review` 34.71%(463/1,334), `recall@IoU0.10` 57.89%(88/152) — 목표 4/5 PASS · 1/5 FAIL 사실을 동결 상태로 그대로 유지한다.
- **미입증 범위:** 생산 유병률 보정 및 공개 데이터셋 밖 외부 공정 일반화는 미입증(`NOT_MEASURED`)이다.
