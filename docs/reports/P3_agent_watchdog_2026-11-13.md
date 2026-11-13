# Phase 3 무결성 감사 Watchdog · PSI 드리프트 · 재현성 실측 보고서

- 생성일: 2026-11-13
- 기준선 설정: `configs/base.toml` (`config_hash=a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098`)
- 봉인 분할(`test`·`official_test` CSV): **미접근** (`tune` 2,001장 캐시·잠금 JSON 메타데이터만 점검)
- 판정 소유권: Watchdog/드리프트 감시는 100% observe-only 이며 기계 판정·설정·모델을 수정하지 않음.

## 1. 저장소 하드 게이트 감사 (`audit_repository_gates` / `surface-sentinel gate`)

- 종합 판정: **PASS** (`C:\surface-sentinel: PASS (통과 5/5, REJECT 0건, REVISE 0건)`)
- 감사 해시: `ba0f0b26e3825aa4...`

| 게이트 | 결과 | 심각도 | 상세 |
|---|---|---|---|
| `G_CONFIG_FROZEN` | PASS | REJECT | configs/base.toml 해시 일치(a762ba639441) |
| `G_AGENT_CONFIG` | PASS | REJECT | configs/agent.toml 로드 및 기본 설계 계약 일치 |
| `G_DATA_LEAKAGE` | PASS | REJECT | T03_audit.json 분할 누출·무결성 위반 0건 확인 |
| `G_SEAL_LOCK` | PASS | REJECT | 봉인 잠금 불변 확인(test(access=2), official_test(access=1)) |
| `G_EVIDENCE_DENOMINATORS` | PASS | REJECT | 주요 증거 JSON 비율 객체 106건 분모 포함 확인 |

## 2. PSI 분포 드리프트 감시 (`tune` 2,001장 기준선, `t05_adopted_r20`)

- 기준선: `docs/evidence/t05_adopted.jsonl` + tune 캐시 2001장

| 시나리오 | 표본 수(기준/대상) | 종합 판정 | `score` PSI | `mean_gray` PSI | `blur_variance` PSI | 비고 |
|---|---:|---|---:|---:|---:|---|
| 정상 홀드아웃 (시드 20261113, 400장) | 1601 / 400 | **NORMAL** | 0.0238 | 0.0000 | 0.0122 | 모든 피처 PSI < 0.10 |
| 합성 드리프트 (점수 +0.35 / 밝기 +35) | 2001 / 400 | **DRIFT_ALERT** | 5.6026 | 14.4225 | 0.0060 | PSI >= 0.25 경보 감지 |
| 소표본 방어 (10장) | 2001 / 10 | **INSUFFICIENT_SAMPLE** | 9.5319 | - | - | 표본 20건 미만 억지 경보 차단 |

### 2.1 정상 홀드아웃 vs 합성 드리프트 상태 비율 (분모·Wilson 95% CI 포함)

| 시나리오 | OK 비율 | NG 비율 | REVIEW 비율 |
|---|---|---|---|
| 정상 홀드아웃 400장 | 48.00% (192/400) | 12.75% (51/400) | 39.25% (157/400) |
| 합성 드리프트 400장 | 0.00% (0/400) | 52.00% (208/400) | 48.00% (192/400) |

## 3. 재현성 검증 (`repro check`) 및 run/DB Watchdog 스모크

- 스모크 에이전트 트레이스: 이벤트 9건, 감사 오류 **0건**, 리플레이 결정론=True (`DONE/OK`)
- `surface-sentinel repro check --trace`: 종료 코드 **0**
- `surface-sentinel repro check --run-a ... --run-b ...`: 종료 코드 **0**
- 스모크 `audit_run_dir`: **PASS**
- 스모크 `audit_records_db` (`?mode=ro`): **PASS**
