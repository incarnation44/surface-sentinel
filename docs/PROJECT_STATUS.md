# 프로젝트 현황 — Surface Sentinel

## 현재 판정 요약

| 항목 | 상태 |
|---|---|
| **패키지 버전** | `1.0.1` (`v1.0.1` 기반) |
| **메인 검사 엔진** | OpenCV 10단계 결정론 파이프라인 **r20** (`configs/base.toml`, hash `a762ba639441…` 동결) |
| **비교 엔진 (YOLO)** | 격리 비교 트랙 유지 · **채택 보류** (독립 val 400장 짝이득 `+0.0000`) |
| **에이전트 (Phase 1–4)** | 고정 규칙 결정론 엔진 · **100% 관찰 전용 (observe-only)** · 기계 판정 불변 |
| **봉인 원장** | `test` `access_count=2` · `official_test` `access_count=1` · `post_seal_config_change=false` (3차 개봉 금지) |
| **생산 적용 판정** | **NO-GO 유지** (봉인 test escape Wilson 95% 상한 21.55% > 기준 20%) |

---

## 모듈 구현 및 검증 상태

| 영역 | 상태 | 핵심 근거 |
|---|---|---|
| 데이터 감사·분할 (T01·T03) | 완료 | 정본 3,335장 (결함 356 / 정상 2,979), `dHash <= 2` 군집 분할, L1–L8 누출 위반 0건 |
| 10단계 파이프라인·배치 (T04·T06) | 완료 | `inspect`·원자적 `batch` 구현, r20 동결, 검증 배치 후보 절단 0건 |
| 평가·봉인 검증 (T05·T08) | 완료 | `tune` 2,001장 및 봉인 `test` 1,334장 전수 측정 완료 (목표 4/5 PASS · G1 상한 FAIL 투명 기록) |
| 로컬 웹 콘솔·API (T07) | 완료 | `serve --with-api`, `/cinema`, `/triage`, `/watchdog`, `/inspections`, 외부 CDN 0건 |
| 관찰 전용 에이전트 (T09·Phase 0–4) | 완료 | 평가=서빙 패리티 0.0(val 400), 트리아지 Lift 4.09x, Watchdog 6게이트·PSI 드리프트, 반사실 면적 병목 95.12% 규명 |
| 통합 기술 가이드·릴리스 점검 (T10·T11) | 완료 | 30섹션 자립형 HTML 가이드(`G_GUIDE_H7` PASS), 금지 파일 0건, 단일 파일 `< 2MB` |
| 보안·안정화 하드닝 | 완료 | 이미지 헤더 fail-closed, `SSCACHE2`(JSON+zlib) 캐시, CSRF Origin 검사, API 동시성 락, 유니코드 경로 I/O, SQLite 트리거 8/8 |

---

## 자동 검증 기준선

* **테스트**: 전체 **1004건** (CI `not data and not browser` **989 passed** · nightly `not data` **997** · 로컬 `data` **7** · `browser` **8**)
* **정적 검사·잠금**: `ruff check .` 0오류 · `uv lock --check` 일치 (36 packages)
* **하드 게이트**: `uv run surface-sentinel gate` **6/6 PASS**
* **DB 무결성**: `uv run surface-sentinel records verify --db runs/records.db` 트리거 **8/8 PASS** (읽기 전용)
* **재현성**: `surface-sentinel repro check` exit 0 (동일 입력 점수차 0.0)

---

## 주요 실측 지표

| 지표 | tune (2,001장) | 봉인 test (1,334장) | 기준 / 비고 |
|---|---:|---:|---|
| Escape (미검출) | 11.68% (25/214) | 14.79% (21/142) | 목표 <= 15% (점추정 PASS) |
| Escape Wilson 95% 상한 | 16.68% | **21.55%** | 목표 <= 20% (**FAIL → NO-GO**) |
| False-NG (오검출) | 5.82% (104/1,787) | 7.21% (86/1,192) | 목표 <= 20% (PASS) |
| Review (사람 검토) | 38.93% (779/2,001) | 34.71% (463/1,334) | 목표 <= 40% (PASS) |
| Recall@IoU0.10 | 59.92% (145/242) | 57.89% (88/152) | 목표 >= 50% (PASS) |
| Recall@IoU0.50 | 20.66% (50/242) | 20.39% | 정밀 마스크·물리 mm 계측 불가 |

---

## 미해결 한계 (열린 과제)

1. 봉인 `test` escape Wilson 상한 21.55% (> 20%) 초과로 생산 투입 보류 (`NO-GO`).
2. `official_test`(1,004장)는 내부 `tune`(579)·`test`(425)와 중복돼 독립 검증 불가, 봉인 `test` 3차 개봉은 `G_SEAL_LOCK`으로 금지.
3. 실제 공장 유병률 보정, 다품종 일반화, 생산 PC 장시간 지연(`p95`/`p99`) 미측정(`NOT_MEASURED`).
4. 120건 피드백 캘리브레이션은 레이블 기반 결정론 시뮬레이션이며 실제 작업자 블라인드 검토는 별도 필요.
