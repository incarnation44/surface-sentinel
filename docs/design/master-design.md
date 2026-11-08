# Surface Sentinel — 마스터 설계

- **설계 버전:** SS-DESIGN-1.1
- **패키지:** `surface_sentinel`
- **CLI:** `surface-sentinel`
- **현재 값 정본:** 채택 값은 **r20(`configs/base.toml`, config hash `a762ba63`)**이다. 본문의 초기 설계값(임계값·후보 설정·구조 서술)은 설계 기준선이며, 실제 동작 값은 `configs/`와 확정 보고서를 따른다.

> 이 문서는 구현 계약이다. 범위, 알고리즘, 데이터 분할, 평가 기준, UI 구조는 설계 개정(ADR)
> 없이 바꾸지 않는다.

## 1. 문제, 범위, 상태 계약

### 1.1 문제

KolektorSDD2에는 이미지가 3,335장 있다: 결함 356장(10.67%), 정상 2,979장.
이 프로젝트는 결정론적이고, 설명 가능하며, 감사 가능한 표면 검사 시스템을
만든다. 최신 정확도(state-of-the-art), 물리 측정, 현장 일반화, 운영 환경
준비 완료를 주장하지 않는다.

### 1.2 범위 안

- 이미지 단위 `OK / NG / REVIEW` 판정, 결함 오버레이, 사유 코드.
- 단일 이미지 및 배치 검사.
- 이미지 단위 및 영역 단위 평가, 명시적 분모 포함.
- 중복 그룹 분할과 봉인 test 접근을 포함한 tune/test 분리.
- 기준선(baseline)과 ablation.
- append-only 사람 결정을 포함한 로컬 검토 UI.
- CSV, JSONL, PNG, TOML, 감사 로그.
- 제한된 도구를 사용하는 결정론적 에이전트.

### 1.3 범위 밖

카메라 획득, 조명/광학 보정, 물리 mm 측정, 3D 검사, 고해상도 재검사,
딥러닝, 결함 유형 분류, MES/PLC 연동, 클라우드 배포, 다중 사용자 인증,
실시간 영상, 운영 환경 준비 완료 주장.

### 1.4 상태 계약

| 상태 | 의미 | 제품 판정 | 다음 행동 |
|---|---|---|---|
| `OK` | 증거가 OK 임계값 이하 | 정상 | 없음 |
| `NG` | 증거가 NG 임계값 이상이고 최소 타당성 통과 | 불량 | 없음 |
| `REVIEW` | 증거가 애매하거나, 입력 품질이 낮거나, ROI가 불확실하거나, 경계 후보이거나, 후보가 절단됨 | 보류 | 사람 검토 |
| `ERROR` | 입력, 처리, 저장 실패 | **판정 아님** | 수정 후 재실행 |

`ERROR`는 제품 상태와 별도로 기록한다. Run 상태는 `COMPLETE`,
`COMPLETE_WITH_ERRORS`, `FAILED`다.

## 2. 데이터 설계

### 2.1 검증된 사실 `[FACT]`

- 전체 이미지 3,335장: 눈에 보이는 결함 356장, 결함 없음 2,979장.
- 공식 train: 양성 246장 + 음성 2,085장.
- 공식 test: 양성 110장 + 음성 894장.
- 대략적인 이미지 크기: 230 × 630 픽셀.
- 라이선스: CC BY-NC-SA 4.0; 상업적 사용은 별도 계약이 필요하다.
- 출처: https://www.vicos.si/resources/kolektorsdd2/

T03에서 실제 아카이브를 확인할 때까지 아래 항목은 `[NOT_VERIFIED]`다:
마스크 파일 이름 규칙, 아카이브 하위 디렉터리 구조, 채널 수, 이미지가 물리적
부품 ID를 공유하는지 여부.

T03 확인 결과 `[FACT]` (`docs/research/T03_data_audit.md`): 실제
아카이브는 `train/`, `test/` 아래에 이미지와 `<id>_GT.png` 마스크가 있고, 복사본은
`train/10301 (copy).png`와 `train/10301_GT (copy).png`다. 이미지는 3채널 uint8
PNG, 마스크는 1채널이며, 폭 184–241, 높이 597–665다. 이미지·마스크 sha256이 모두
같은 중복 1쌍(`train/10301.png` ↔ `train/10301 (copy).png`)이 있어 정본
3,335장, 전체 3,336행이다. 정상 이미지의 마스크 2,979개는 픽셀 기준 빈
마스크다. 위 `[NOT_VERIFIED]` 항목은 T03에서 확인되었다.

### 2.2 Manifest

`data/manifests/dataset_manifest.csv` 컬럼:

`sample_id,original_id,group_id,image_path,mask_path,label,source_split,width,height,channels,format,image_sha256,mask_sha256,defect_pixels`

- `sample_id`: 이미지 파일 stem. 예: `10023`.
- `original_id`: 원본 촬영 ID. 기본값은 `sample_id`.
- `group_id`: 중복 그룹 ID. `g` + 10자리 hex.
- `label`: `defect` 또는 `good`.
- `source_split`: 공식 `train` 또는 `test`. 참고용.
- `mask_path`: 정상 이미지에서만 비어 있다. 결함 이미지에 마스크가 없으면
  T03을 중단한다.

### 2.3 분할

1. 정확히 같은 `image_sha256` 중복을 합친다.
2. 9×8 dHash Hamming 거리 ≤ 2인 근접 중복을 합친다.
3. 그룹이 데이터셋의 2%를 넘으면 dHash 임계값을 0으로 낮추고 그룹을 다시
   만든다.
4. 시드 42로 그룹을 섞는다.
5. `label`로 층화한 뒤 tune/test 목표 60% / 40%에 그룹을 탐욕적으로 배정한다.
6. `P_test >= 100`을 보장한다. 필요하면 tune에서 그룹을 옮기고 그 사실을
   기록한다.
7. `test.csv`를 동결하고 `test.sha256`을 쓰며 `data/splits/split_report.json`을
   만든다.
8. `source_split=test`로 `official_test.csv`를 만들고, 내부 test와의 중복을
   보고한다.

금지: 이미지 단위 무작위 분할, 파생 이미지를 따로 분할, 평가 후 test 그룹을
tune으로 되돌리기.

### 2.4 누출 검사

| ID | 검사 | 통과 조건 |
|---|---|---|
| L1 | tune과 test에 같은 `original_id` | 0 |
| L2 | 양쪽에 같은 `image_sha256` | 0 |
| L3 | 양쪽에 같은 `group_id` | 0 |
| L4 | 파생 이미지가 `group_id`를 상속 | 100% |
| L5 | `test.csv` 해시가 `test.sha256`과 일치 | 일치 |
| L6 | tune run 로그에 `split=test` 포함 | 0 |
| L7 | 공식/내부 test 중복 | 보고만, 절대 튜닝하지 않음 |
| L8 | 라벨/마스크 불일치 | 0 |

### 2.5 봉인 test

- `batch --split test`와 `eval --split test`는 `--allow-sealed`가 필요하다.
- 첫 접근 시 `data/splits/test.lock.json`에 split 해시, config 해시, 코드 SHA,
  타임스탬프, 목적을 기록한다.
- 이후 접근은 `access_count`를 증가시키고 `reuse=true`로 설정한다.
- `official_test`는 별도 잠금을 사용한다.
- 첫 봉인 실행 이후의 설정 변경은 봉인 후 새 리비전으로 보고한다.

### 2.6 데이터 QA와 중단 조건

T03은 3,335장 전체와 공식 분할 수량이 일치하고, 모든 이미지가 디코드되며,
모든 결함 마스크가 이미지 크기와 맞고 결함 픽셀을 포함하며, 모든 정상
마스크가 없거나 비어 있고, L1–L4/L8을 통과할 때만 PASS한다.

다운로드에 대화형 양식이 필요하거나, 수량이 1% 넘게 다르거나, 마스크 규칙에
설계 변경이 필요하거나, `P_test >= 100`을 만족하는 분할이 없으면 중단한다.

### 2.7 한계

단일 통제 데이터셋이며 현장 일반화는 없다. 물리 스케일도 없다. 무작위 그룹
분할은 시간적 또는 생산 배치 이동을 모델링하지 않는다. 공식 test는 보조
감사이며 독립적인 현장 시험이 아니다.

## 3. 검사 파이프라인

고정 순서:

`validate_input → load_image → prepare_roi → preprocess → generate_candidates → refine_mask → analyze_components → calculate_score → decide_status → save_evidence`

| # | 단계 | 목적 | 입력 → 출력 | 알고리즘 / 설정 | 실패 또는 플래그 |
|---|---|---|---|---|---|
| 1 | `validate_input` | 계약 검사 | path/bytes → metadata | 존재, 디코드, 최소 64×64, std ≥ 1.0, SHA256, 마스크 크기 | `E_INPUT_*`, `E_MASK_SHAPE` |
| 2 | `load_image` | 원본 보존 + 회색조 | file → BGR, gray | `IMREAD_COLOR`, `cvtColor` | `E_INPUT_DECODE` |
| 3 | `prepare_roi` | 표면으로 제한 | gray → ROI mask/bbox | `dark_floor=8`, close 7, 최대 연결 요소, 채우기, erode 2 | `ROI_UNCERTAIN` |
| 4 | `preprocess` | 배경 보정과 강건 정규화 | gray, ROI → response, z-map | median 3, black-hat/top-hat, kernel 0.10·min(H,W)를 [21,61]로 제한, max, MAD | `INPUT_QUALITY_LOW` |
| 5 | `generate_candidates` | 후보 픽셀 | z-map → binary | z ≥ 4.0, contrast ≥ 12, open 3, close 5, merge 7 | 비어 있어도 허용 |
| 6 | `refine_mask` | 컴포넌트 정제 | binary → components | 재임계, open 3, 최소 면적 12, 경계 플래그, 최대 20 | `CANDIDATE_TRUNCATED` |
| 7 | `analyze_components` | 특징 | components → feature table | 면적, bbox, 종횡비, elongation, solidity, 대비, 경계 거리 | 없음 |
| 8 | `calculate_score` | 증거 점수 | features → score | 면적/대비 정규화, 기하평균, 최대값 | `E_PROCESS_INTERNAL` |
| 9 | `decide_status` | 상태 계약 | score/flags → status/reasons | 아래 진리표 | 없음 |
| 10 | `save_evidence` | 저장 | result → run artifacts | PNG 오버레이, JSONL, 원자 rename | `E_STORAGE_WRITE` |

### 3.1 점수

```
area_norm(c)     = min(area_px(c) / 48.0, 1.5)
contrast_norm(c) = min(mean_contrast_gray(c) / 24.0, 1.5)
evidence(c)      = min(sqrt(area_norm(c) * contrast_norm(c)), 1.5)
image_score      = max(evidence(c))
```

점수는 확률이 아니다. 최소 타당성: `area_px >= 12`이고
`mean_contrast_gray >= 12`. 모든 초기값은 `[DESIGN]`이며 tune에서만 조정할 수
있다.

### 3.2 판정 진리표

| 우선순위 | 조건 | 상태 | 사유 |
|---:|---|---|---|
| 1 | 입력/처리 오류 | `ERROR` | `error_code` |
| 2 | 타당한 컴포넌트 있고 점수 ≥ `ng_min` | `NG` | `DEFECT_EVIDENCE_STRONG` |
| 3 | 타당한 컴포넌트 없고 플래그 없음 | `OK` | `NO_DEFECT_EVIDENCE` |
| 4 | 입력 품질 낮음 | `REVIEW` | `INPUT_QUALITY_LOW` |
| 5 | ROI 불확실 | `REVIEW` | `ROI_UNCERTAIN` |
| 6 | 경계에 닿은 후보이고 점수 ≥ `REVIEW_BAND_MAX`(0.40 동결) | `REVIEW` | `BORDER_TOUCHING_CANDIDATE` |
| 7 | 후보 절단 | `REVIEW` | `CANDIDATE_TRUNCATED` |
| 8 | 약한 후보가 여러 개 | `REVIEW` | `MULTIPLE_WEAK_CANDIDATES` |
| 9 | 그 외 | `REVIEW` | `EVIDENCE_IN_REVIEW_BAND` |

품질 및 ROI 플래그는 `NG`를 절대 덮어쓰지 않는다.

> F4 채택: `scoring.ok_max`를 제거하고 `REVIEW_BAND_MAX = 0.40`으로
> 동결했다. OK는 타당한 컴포넌트가 0개이고 플래그가 없을 때만 열리며, 경계
> 에스컬레이션과 다중 약후보 하한은 동결 상수를 쓴다. 채택 이전 TOML의
> `ok_max`는 로드 시 무시되고 config hash에도 들어가지 않는다. 이 변경으로
> config hash가 1회 바뀌었다(제거 이전 base `22ff6bcb…`, r20 후보 `de1da847…`).

### 3.3 오류 코드

`E_INPUT_MISSING`, `E_INPUT_DECODE`, `E_INPUT_SIZE`, `E_INPUT_UNIFORM`,
`E_MASK_SHAPE`, `E_MASK_MISMATCH`, `E_CONFIG_INVALID`, `E_PROCESS_INTERNAL`,
`E_STORAGE_WRITE`, `E_SEALED_DENIED`, `E_PROFILE_UNKNOWN`,
`E_BUDGET_EXCEEDED`, `E_TOOL_FAILED`.

## 4. 평가

### 4.1 비교 집합

| ID | 구성 | 역할 |
|---|---|---|
| B0 | 항상 `OK` | 유병률/하한 |
| B1 | ROI + median + Otsu + 최소 면적 | 단순 임계 기준선 |
| B2 | 배경 보정 + 고정 임계 | 정규화 ablation |
| P | 제안 파이프라인 | 최종 |
| A1 | 배경 보정 제거 | morphology 기여 |
| A2 | ROI 제거 | ROI 기여 |
| A3 | 컴포넌트 정제 제거 | 정제 기여 |
| A4 | 대비 하한 제거 | 대비 하한 기여 |
| A5 | median blur 제거 | 잡음 민감도 |
| A6 | 어두운 채널만 사용 | 밝은 채널 기여 |

B0/B1/P는 test에서 한 번 실행한다. Ablation은 tune에서만 실행한다. Ablation은
기본 판정 임계값을 유지한다. A1은 점수 스케일이 바뀌므로 서술적으로만 본다.

### 4.2 이미지 지표와 분모

`P` = test 결함 이미지 수, `N` = test 정상 이미지 수, `T = P + N`.

| 지표 | 공식 | 의미 |
|---|---|---|
| `defect_escape_rate` | `OK_on_defect / P` | 가장 중요한 실패 |
| `defect_ng_rate` | `NG_on_defect / P` | 검출된 결함 |
| `defect_review_rate` | `REVIEW_on_defect / P` | 보류된 결함 |
| `good_false_ng_rate` | `NG_on_good / N` | 오탐 |
| `good_review_rate` | `REVIEW_on_good / N` | 보류된 정상 |
| `review_rate` | `(REVIEW_defect+REVIEW_good)/T` | 전체 보류율 |
| `automation_rate` | `(OK+NG)/T` | 자동 결정 |
| `decided_accuracy` | `(NG_defect+OK_good)/(OK+NG)` | 결정된 것 중 정확도 |
| `balanced_accuracy` | `(NG_defect/P + OK_good/N)/2` | 불균형 고려 |

REVIEW는 맞음도 틀림도 아니다. 모든 표는 분모를 명시한다.

### 4.3 영역 지표

마스크가 있는 결함 이미지만 대상. GT 컴포넌트는 8-연결 마스크 컴포넌트다.

- `localization_recall@IoU0.10`: 매칭된 GT / 전체 GT.
- `localization_recall@IoU0.30`: 엄격 매칭.
- `pixel_precision = TP_pixels / predicted_pixels`.
- `pixel_recall = TP_pixels / GT_pixels`.
- `mean_iou_matched`: 매칭된 쌍의 평균 IoU.

분모 두 개를 보고한다: 전체 GT, 그리고 NG/REVIEW로 판정된 이미지 안의 GT.

### 4.4 튜닝 및 test 프로토콜

- 비율에는 Wilson 95% CI를 붙인다.
- 시간 경계: `load_image` 시작부터 `save_evidence` 끝까지. manifest, config
  로드, 프로세스 시작, 디스크 flush는 제외. 배치 wall time은 별도.
- Tune: 모든 임계값, ablation, 실패 분석.
- Test: 동결 후 봉인 실행 1회, B0/B1/P.
- Official test: 동결 config로 봉인 보조 실행 1회.
- 튜닝 그리드: 최대 24회 실행, `docs/evidence/tune_grid.jsonl`에 기록.
- 선택: false-NG ≤ 0.20, review rate ≤ 0.40을 만족하면서 escape를 최소화.
  없으면 Pareto 점을 골라 보고한다. 후보 절단(`CANDIDATE_TRUNCATED`) 수는
  하드 조건이 아닌 비교 축으로 함께 보고한다. 절단 0을 필수 조건으로
  확정하려면 사용자 정책 결정과 설계 개정이 필요하다.
- test를 본 뒤 임계값을 절대 바꾸지 않는다.

### 4.5 실패 사례 선택

규칙으로 선택하고 임의로 고르지 않는다: 고득점 escape 상위 2개, 고득점 오탐
상위 2개, 가장 큰 REVIEW 클러스터, 경계 사례, 품질 플래그 사례, 부분 위치
사례. 전체 목록은 `docs/evidence/failure_cases/index.json`에 둔다.

## 5. 인터페이스와 산출물

### 5.1 CLI

`doctor`, `data build-manifest`, `data make-splits`, `data verify`, `inspect`,
`batch`, `eval`, `gate`, `serve`, `agent run`, `agent replay`, `repro check`.

종료 코드: 0 성공, 2 사용법/설정, 3 입력 검증, 4 샘플 오류와 함께 완료,
5 저장/미완료, 6 봉인 접근 거부, 7 릴리스 게이트 실패.

### 5.2 Run 디렉터리

```
runs/<run_id>/
  run.json
  results.jsonl
  summary.csv
  summary.json
  config_snapshot.toml
  manifest_snapshot.csv
  logs/run.log.jsonl
  overlays/<sample_id>_overlay.png
  stages/<sample_id>_response.png      # 선택
  review_decisions.jsonl               # 검토한 경우
  agent_trace.jsonl                    # 에이전트 모드인 경우
```

`run_id`: `YYYYMMDDTHHMMSSZ-<config_hash8>-<split>[-NN]`.

### 5.3 스키마

`results.jsonl` (스키마 `result-v1`):
`run_id,sample_id,original_id,split,label,status,score,reasons,components,quality,roi,timing_ms,error_code,error_stage,input_sha256,mask_sha256,overlay_path,config_hash`.

`summary.csv`:
`run_id,sample_id,original_id,split,label,status,score,top_area_px,top_contrast_gray,top_evidence,reason_codes,component_count,duration_ms,error_code,overlay_path`.

`run.json` (스키마 `run-v1`):
`run_id,mode,split,config_path,config_hash,code,env,manifest_sha256,split_sha256,started_at,finished_at,status,counts,timing,artifacts`.

`review_decisions.jsonl`:
`decision_id,run_id,sample_id,machine_status,final_status,reason_code,note,reviewer,decided_at,config_hash,overlay_sha256`.

`REVIEW`만 덮어쓸 수 있다. `OK`/`NG`는 불변이다. 결정은 append-only이며 마지막
결정이 최종 상태다.

### 5.4 원자 저장

`runs/.tmp-<run_id>-<pid>/`에 쓰고, 파일을 fsync한 뒤 `os.replace`한다.
`run.json`은 마지막에 쓴다. replace 실패 시 최종 디렉터리를 남기지 않고
종료 코드 5로 끝낸다. `os.replace`를 monkeypatch해 시험한다.

### 5.5 UI

로컬 `127.0.0.1` 전용. 설정 편집과 삭제는 없다. 이미지 업로드 검사는
`serve --with-api`로 실행했을 때만 열린다. 사용법은 [WEB_GUI_GUIDE.md](../WEB_GUI_GUIDE.md)에 있다.

| 경로 | 화면 |
|---|---|
| `/` | 검사 허브: run 목록 |
| `/runs/{run_id}` | 요약과 샘플 그리드 |
| `/runs/{run_id}/samples/{sample_id}` | 원본, ROI, response, 오버레이, 점수, 사유, 컴포넌트, 해시 |
| `/runs/{run_id}/review` | REVIEW 큐, 키보드 J/K, OK/NG/보류, 사유 필수 |
| `/runs/{run_id}/metrics` | 혼동행렬, 지표, 실패 링크 |
| `/runs/{run_id}/triage` | REVIEW 우선순위 트리아지, Recall@Top-K |
| `/cinema` | 10단계 검사 과정 시각화 |
| `/inspections` | 사진 검사 워크벤치와 검사 기록 DB, CSV 내보내기 |
| `/stats` | 검사 기록 통계 |
| `/watchdog` | 저장소 게이트·드리프트 감사 |
| `/compare` | 엔진 비교 |
| `/healthz` | 상태 확인 |

한국어 UI 문구; 영어 식별자와 상태 배지. 색상만으로 의미를 전달하지 않는다.

## 6. 제한된 에이전트

에이전트는 고정 규칙으로 동작하는 결정론적 정책 엔진이다. 제품 판정은 항상
`decide_status`에서 나온다.

상태: `OBSERVE → PLAN → ACT → VERIFY → PLAN | DONE | REVIEW | ERROR`.

| 항목 | 계약 |
|---|---|
| 관찰 | 품질 플래그, ROI 상태, 최상위 증거, 프로파일 ID, 남은 예산 |
| 행동 | `run_profile`, `inspect_artifact`, `compare`, `escalate_review`, `stop` |
| 금지 | 파일 쓰기, 셸, 네트워크, 설정 변경, 직접 판정 |
| 예산 | 행동 3회, 프로파일 2개, 추가 지연 5,000 ms |
| 가드 | `action_hash` 중복 차단; NG 이후 재시도 없음 |
| 소진 | `REVIEW`, 낙관적 `OK` 아님 |
| 감사 | 상태, 도구, 인자 해시, 결과 해시, 예산을 append-only `agent_trace.jsonl`에 기록 |

정책:

| 관찰 | 행동 |
|---|---|
| 기본 프로파일에서 `INPUT_QUALITY_LOW` | `denoise_strong` 1회 |
| 기본 프로파일에서 `ROI_UNCERTAIN` | `roi_fixed` 1회 |
| 두 프로파일 모두 OK | DONE OK |
| 어느 하나라도 NG | DONE NG |
| 프로파일 불일치 | REVIEW |
| 경계 + REVIEW 구간 | REVIEW |
| 알 수 없는 프로파일 | ERROR |

비교: 120장 층화 tune 부분집합에서 고정 `base`와 정책을 비교한다. review rate가
5퍼센트포인트 이상 감소하고, escape가 1포인트 넘게 증가하지 않고, 추가 p95
지연이 1.5초 이하이고, 잘못된 행동이 0이고, 감사가 완전할 때만 승격한다.
그렇지 않으면 고정 모드를 유지하고 보고한다.

## 7. 테스트와 실패 처리

| 계층 | 필수 사례 |
|---|---|
| 단위 config | 알 수 없는 키, 범위 위반, `ng_min <= REVIEW_BAND_MAX`, 레거시 `ok_max` 무시 |
| 단위 features | 합성 스크래치/점/빈 표면/경계/대비 |
| 단위 score | 단조성, 0으로 나누기, 비유한 값 |
| 단위 decision | 9행 진리표, NG 우선, 플래그가 NG를 못 덮음 |
| 단위 metrics | 손계산 혼동행렬, 0 분모, Wilson CI |
| 통합 pipeline | 합성 이미지 10장, 기대 상태, 오버레이 크기 |
| 통합 batch | 이미지 50장, summary 수 = results 수 |
| 통합 persistence | `os.replace` 실패 → COMPLETE 없음, 종료 코드 5 |
| 통합 input | 손상 PNG, 0바이트, 마스크 크기 불일치 |
| 회귀 | golden fixture 결과 해시 |
| 재현성 | 두 실행, timing/path 제외 결과 동일 |
| UI | 경로, 잘못된 전이, HTML escape |
| 브라우저 | 데스크톱/모바일, 콘솔 오류 0, 키보드 흐름 |
| 데이터 | 로컬 마커 전용 |

하드 게이트:

- **H1** 데이터 무결성과 분할 누출 0.
- **H2** 스키마와 상태 계약.
- **H3** 결정론.
- **H4** 저장 실패 시 COMPLETE 마커 없음.
- **H5** 봉인/tune 분리.
- **H6** clean clone 설치와 테스트.
- **H7** 문서 수치가 증거와 일치.

성능 목표 `[DESIGN]`: escape ≤ 0.15, false-NG ≤ 0.20, review ≤ 0.40,
localization recall@IoU0.10 ≥ 0.50. 목표 미달은 보고한다. 목표를 맞추기 위해
test를 재사용하는 것은 금지한다.

명시: 위 escape ≤ 0.15는 **점추정** 기준이고, T04 재보정 통과
기준은 Wilson 95% **CI 상한 ≤ 0.20**을 별도 조건으로 함께 본다(점추정 ≤ 0.15
+ CI 상한 ≤ 0.20 동시 충족). 두 기준을 혼동하지 않는다.

F6: 채택 검증에는 이미지 단위 **그룹 부트스트랩**(b=2000,
seed=20261111)과 동일 이미지 인덱스를 쓰는 **짝 부트스트랩 delta**를 함께
기록한다. 비율 분모가 이미지마다 다른 recall은 GT 수를 분모로 둔다. Wilson
구간은 기존 보고서 호환을 위해 유지한다.

## 8. 저장소 구조

```
surface-sentinel/
├─ .github/workflows/{ci,nightly}.yml
├─ .gitignore / .gitattributes
├─ README.md / LICENSE / NOTICE / plan.md
├─ pyproject.toml / uv.lock
├─ configs/
│  ├─ base.toml                 # r20 동결
│  ├─ agent.toml
│  ├─ candidates/               # 재보정·전수 검증 후보 (recalib, fullverify, adopted)
│  └─ profiles/*.toml
├─ data/ (원본과 파생 데이터는 무시)
│  ├─ audit/T03_audit.json
│  ├─ manifests/dataset_manifest.csv
│  └─ splits/{tune,test,official_test}.csv, split_report.json, test.sha256, *.lock.json
├─ docs/
│  ├─ design/                   # 설계 문서
│  ├─ decisions/                # ADR 0001–0008
│  ├─ evidence/                 # 평가·감사 원시 산출물
│  ├─ reports/                  # 실험·봉인·에이전트 보고서
│  ├─ research/                 # 데이터셋 사실·감사
│  └─ surface-sentinel-integrated-guide.html
├─ runs/ (.gitkeep 외 무시)
├─ scripts/
├─ src/surface_sentinel/
│  ├─ cli.py, config.py, contracts.py, version.py
│  ├─ io/, pipeline/, evaluation/, batch/
│  ├─ records/, review/, api/, ui/
│  └─ agent/, analysis/, yolo/
└─ tests/{unit,integration,ui,data,regression,fixtures,golden}
```

## 9. 구현 단위와 검증 결과

| 단위 | 내용 | 검증 결과 |
|---|---|---|
| T01 데이터 사실 | 공식 KSDD2 수량·라이선스·인용 검증 (`docs/research/T01_dataset_facts.md`) | PASS — 공식 수량 3,335/2,331/1,004, 라이선스 확인 |
| T02 저장소 골격 | uv 환경, 패키지 골격, 계약, config 로더, CLI, 테스트, CI | PASS — `uv sync --frozen`, ruff clean |
| T03 데이터 기반 | manifest, tune/test/official_test 분할, 누출 감사, contact sheet | PASS — 전체 3,336행/정본 3,335, tune 2,001 / test 1,334 / official_test 1,004, L1–L4/L8 위반 0, `P_test` 142 |
| T04 파이프라인 | 10단계 파이프라인, config 스키마, fixtures, `inspect` | PASS — 결정론 fixtures, 오버레이, ERROR 분리 |
| T05 평가 | metrics, B0/B1/B2/P, A1–A6, tune grid, failure index, r20 채택 | PASS — 모든 비율에 분모, test 흔적 없음 |
| T06 배치·저장 | batch runner, run 디렉터리, 원자 저장, 종료 코드 | PASS — 실패 주입 시 COMPLETE 없음, exit 5 |
| T07 로컬 UI·검토 | 로컬 웹 콘솔, append-only 검토 결정, `serve` CLI | PASS — 잘못된 전이 거부, 외부 요청 없음 |
| T08 봉인 test | 봉인 test·official_test 1회 측정, 게이트 보고서 | 목표 4/5 PASS · escape Wilson 상한 FAIL 그대로 기록 (`docs/reports/T08_sealed_2026-11-13.md`) |
| T09 제한된 에이전트 | 상태 머신, 도구, 정책, 감사, 비교 | 계약 PASS · 승격 조건 미충족(review 감소 0pp < 5pp) → 고정 모드 유지 (`docs/reports/T09_agent_2026-11-13.md`) |
| T10 통합 HTML 가이드 | 30섹션 자립형 HTML (`docs/surface-sentinel-integrated-guide.html`) | PASS — `G_GUIDE_H7` |
| T11 릴리스 점검 | clean clone, 금지 파일, 파일 크기, 라이선스 고지 | PASS (`docs/reports/T11_release_check_2026-11-13.md`) |

각 단위의 허용 범위는 다음 원칙을 따른다.

- 데이터 단위(T01·T03)는 원본 이미지를 커밋하지 않고, test로 튜닝하지 않는다.
- 파이프라인·평가 단위(T04·T05)는 봉인 test에 접근하지 않고, 분모 없는 비율을 보고하지 않는다.
- 저장 단위(T06)는 실패한 저장을 완료로 표시하지 않고, run ID를 덮어쓰지 않는다.
- UI 단위(T07)는 설정 편집·삭제·OK/NG 덮어쓰기·외부 CDN을 허용하지 않는다.
- 봉인 단위(T08)는 test 이후 config를 바꾸지 않고, 실패를 숨기지 않는다.
- 에이전트 단위(T09)는 네트워크·API 키·파일/셸/config 도구·직접 판정을 갖지 않는다.

구현 의존 순서는 `T02 → T01 → T03 → T04 → T06 → T05 → T07 → T08 → T09 → T10 → T11`이다.

## 10. 확장 트랙

- **E 트랙 (ADR 0007):** YOLO ONNX 비교 추론, SQLite 기록 DB, 로컬 검사 API, 결정론 워커. 독립 val 400장 짝평가에서 이득 `+0.0000`으로 기본 채택은 보류하고 r20을 유지한다 (`docs/reports/E_val_gate_2026-11-13.md`).
- **Phase 0 무결성 기반 (ADR 0008):** 검사 fingerprint 중복 제거, 평가·서빙 신원 일치, `code_sha` 진입점 1회 주입.
- **Phase 1–4 관찰 전용 에이전트:** 교차 증거, 트리아지, Watchdog·드리프트 감시, 반사실·캘리브레이션. 기계 판정은 바꾸지 않는다 (`docs/design/agent-extension-proposal.md`).

## 11. 통합 HTML 가이드 개요

필수 섹션: 표지; 한 페이지 요약; 문제/차별점; 범위; 사용자 흐름; 화면; 데이터
출처/라이선스; 데이터 구조; 분할/누출; 아키텍처; 파이프라인; 알고리즘;
특징/점수; 상태 계약; 기준선/ablation; 지표/분모; 결과; 실패; 테스트/재현성;
산출물; CLI/폴더; 설치; 에이전트 설계; 에이전트 결과; 게이트; 한계;
설계 결정 이력; 용어; 출처; 버전 기록.

규칙: 한국어 본문, 영어 식별자, 내부 CSS/JS, 외부 CDN 없음, 반응형 및 인쇄
스타일, 모든 수치에 `[FACT]/[DESIGN]/[MEASURED]/[NOT_MEASURED]` 태그, 가짜
결과 없음.

## 12. 확인된 항목

| 항목 | 상태 | 결과 |
|---|---|---|
| 마스크 이름/아카이브 구조 | 확인 완료 | `train/`, `test/`; `<id>_GT.png`; 복사본 `10301_GT (copy).png`; 설계 변경 없음 |
| 채널 수 | 확인 완료 | 이미지 3채널 uint8, 마스크 1채널 |
| 중복/부품 식별 | 확인 완료 | 이미지·마스크 sha256 동일 중복 1쌍; dHash 2, 최대 2행, 2% 초과 없음 |
| 공식 수량 | 확인 완료 | 전체 3,336행/정본 3,335, 공식 수량 일치 |
| 브라우저 테스트 | 확인 완료 | 시스템 Chrome 우선, 없으면 Playwright Chromium |
| 성능 목표 | DESIGN | 실제 값 보고; test 재사용 금지 |
| 에이전트 가치 | DESIGN | 게이트 통과 시에만 승격 |
