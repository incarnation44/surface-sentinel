# 프로젝트 기획 및 설계 요약 — Surface Sentinel

> 설치·실행 방법과 프로젝트 구조는 [README.md](README.md), 상세 규격은 [docs/design/master-design.md](docs/design/master-design.md)를 참고한다.

## 1. 프로젝트 개요

### 프로젝트명
**Surface Sentinel — 결정론적 표면 결함 검사 및 사람 검토 보조 시스템**

### 목적과 문제 의식
제조 표면 검사에서 정상(`OK`)과 불량(`NG`) 둘로만 나누면 기준을 낮출 때 오검출이 늘고, 높일 때 실제 불량을 놓치게 된다. 딥러닝 블랙박스 모델은 오판 원인을 추적하기도 어렵다.

Surface Sentinel은 아래 세 가지 기준으로 설계했다.
1. 10단계 결정론 파이프라인: 같은 사진이면 항상 같은 점수(점수차 0.0)가 나오고, 단계별 중간 이미지를 확인할 수 있다.
2. 4상태 계약 (`OK` / `NG` / `REVIEW` / `ERROR`): 애매한 사진은 사람 검토(`REVIEW`)로 넘기고, 입력 오류는 `ERROR`로 격리한다.
3. 보수적 검증과 한계 공개: 봉인 테스트셋을 코드로 잠그고, 기준을 못 맞춘 지표(Wilson 95% 상한 21.55% > 20%)는 숨기지 않고 생산 `NO-GO`를 유지한다.

---

## 2. 데이터 구성 및 분할 정책

* **데이터셋**: KolektorSDD2 (CC BY-NC-SA 4.0) — 금속 표면 이미지, 폭 184–241px × 높이 597–665px(세로 약 1:2.7 비율).
* **정본 확정**: 원본 3,336행 중 완전 중복 `train/10301 (copy).png` 1쌍을 격리해 정본 **3,335장** (결함 356장 / 정상 2,979장) 확정.
* **누출 차단 분할**: 유사 이미지 군집(`dHash <= 2`) 단위 층화 분할. L1–L8 누출 검사 위반 0건.
  * `tune` (2,001장 = 결함 214 / 정상 1,787): 파라미터 탐색·재보정 및 내부 교차검증. YOLO 비교를 위해 내부를 `tune-train` 1,601장 + 독립 `val` 400장으로 추가 분할.
  * 봉인 `test` (1,334장 = 결함 142 / 정상 1,192): 최종 성능 측정 전용. 기본 실행 시 코드가 읽기 전 거부(`E_SEALED_DENIED`, exit 6)하며 `data/splits/test.lock.json`에 접근 횟수(`access_count=2`)와 사후 설정 변경 여부(`post_seal_config_change=false`)를 기록한다.
  * 보조 `official_test` (1,004장 = 결함 110 / 정상 894): 내부 `tune`(579장)·`test`(425장)와 중복되므로 참고 지표로만 보관한다.

---

## 3. 시스템 아키텍처 (4계층)

| 계층 | 담당 모듈 | 핵심 책임 |
|---|---|---|
| **L1. 데이터·계약 무결성** | `io/`, `config.py`, `contracts.py` | manifest·분할 검증, 이미지 헤더 fail-closed 가드, 동결 설정 `configs/base.toml` 해시(`a762ba63`) 검증 |
| **L2. 비전 코어 & 비교 트랙** | `pipeline/`, `yolo/`, `evaluation/` | 10단계 OpenCV r20 검사, 격리된 ONNX 비교 추론, 분모·신뢰구간 기반 평가 |
| **L3. 2계층 Observe-Only 에이전트** | `agent/`, `analysis/` | Tier 1(교차 증거·트리아지·반사실 병목 분석), Tier 2(Watchdog 6대 게이트·PSI 드리프트 감시) |
| **L4. 저장·서빙·로컬 콘솔** | `batch/`, `records/`, `review/`, `api/`, `ui/` | 원자적 배치 저장, SQLite append-only 원장(트리거 8/8), 오프라인 FastAPI 및 웹 콘솔(`/cinema`, `/triage`, `/watchdog`) |

---

## 4. 10단계 결정론 검사 파이프라인

1. `validate_input`: 경로·확장자·봉인 분할 차단(`E_SEALED_DENIED`)·이미지 헤더 픽셀 상한 검증
2. `load_image`: 유니코드 안전 로드, 채널·크기 확인, 입력 품질 지표(블러·밝기·포화) 측정
3. `prepare_roi`: 암부 배경(`dark_floor=8`) 제거 및 금속 표면 유효 영역(ROI) 마스크 확정
4. `preprocess`: 대형 커널 배경 추정으로 조명 편차 제거, 어두운 결함·밝은 긁힘 대비 응답 계산
5. `generate_candidates`: MAD 기반 강건 Z-Score(`z >= 7.6`) 및 최소 대비(`>= 16.0`)로 후보 추출
6. `refine_mask`: 모폴로지 연산(`open 3`·`close 5`·`merge 5`)으로 노이즈 제거 및 후보 병합(상한 20개)
7. `analyze_components`: 후보별 면적·대비·경계 접촉·형상 특징 계측
8. `calculate_score`: 최소 면적(`>= 40px`)·대비 통과 후보의 면적(`ng_area=800`)×대비(`ng_contrast=22`) 기하평균 최대값 산출
9. `decide_status`: 우선순위 진리표 적용 (`score >= 0.50` → `NG`, 후보 0 + 플래그 0 → `OK`, 그 외 → `REVIEW`)
10. `save_evidence`: 결과 JSON·오버레이 PNG·감사 메타데이터 원자적 저장

---

## 5. 엔진 비교 및 채택 기준

* **OpenCV r20 (채택·동결)**: 초기 기본 설정(`z=4.0`)의 정상 오검출 97.0% 문제를 후보 생성 축 재보정으로 해결해 `r20`(`z=7.6`, `contrast=16.0`, `min_area=40px`)을 동결했다(tune 오탐 5.82%, 후보 절단 0/2,001).
* **YOLO (det/seg) 결합 (채택 보류)**: `tune`에서는 학습 중복(1,601장) 때문에 미검출이 25→17로 줄어 보였으나, 독립 `val` 400장 짝비교에서 미검출 이득이 정확히 `+0.0000`(4/43 → 4/43)으로 확인되어 채택을 보류했다.
* **결정론 에이전트 (관찰 전용 유지)**: 안전 계약 테스트는 모두 통과했으나 `review` 비율 감소가 0.0pp로 승격 기준(>= 5.0pp)에 미달해 판정 비개입(observe-only) 보조 계층으로 고정했다.

---

## 6. 품질 게이트 및 거버넌스

저장소 무결성은 단일 명령 `uv run surface-sentinel gate`로 6개 하드 게이트를 자동 검증한다.

| 게이트 | 검증 내용 |
|---|---|
| `G_CONFIG_FROZEN` | `configs/base.toml` 정규화 해시(`a762ba639441`) 불변 확인 |
| `G_AGENT_CONFIG` | `configs/agent.toml` 로드 및 정책 계약 일치 확인 |
| `G_DATA_LEAKAGE` | `data/audit/T03_audit.json` 분할 누출(L1–L4, L8) 위반 0건 확인 |
| `G_SEAL_LOCK` | 봉인 잠금 원장 불변(`post_seal_config_change=false`, `access_count <= 2`) 확인 |
| `G_EVIDENCE_DENOMINATORS` | 핵심 증거 JSON의 모든 비율 객체에 분자/분모 포함 여부 확인 |
| `G_GUIDE_H7` | 통합 HTML 가이드 30개 필수 섹션, 고유 ID, 외부 CDN 0건, 증거 수치 일치 확인 |

---

## 7. 향후 개선 로드맵 (생산 GO 선행 조건)

현재 시스템은 `NO-GO` 상태이며, 향후 생산 적용 재검토를 위해 아래 3가지가 선행되어야 한다.
1. **신규 독립 데이터 확보**: 기존 봉인 `test`를 재개봉하지 않는 새 현장 데이터셋 확보 및 실제 양산 유병률 보정.
2. **면적 병목 및 위치 정밀도 개선**: `tune` REVIEW의 95.12%를 차지하는 미세 면적(`<= 133px`) 병목과 좌·우 가장자리 15% 집중 편향 해소, 전용 GPU 기반 후보 패치 CNN 분류기 등 대안 검토(`recall@IoU0.50 ≈ 20.4%` 보완).
3. **현장 장시간 안정성 실측**: 목표 생산 PC 환경에서의 p95/p99 처리 지연 및 24시간 연속 구동 안정성 검증.
