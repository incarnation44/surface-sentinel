# Phase 2 에이전트 트리아지·민감도·능동 패치 중재 실측 보고서

- 생성일: 2026-11-13
- 기준선 설정: `configs/base.toml` (`config_hash=a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098`)
- 봉인 분할(`test.csv`, `official_test.csv`): **미접근** (`tune` 및 내부 `val`만 사용)
- 판정 소유권: 기계 판정(`SampleResult.status`)은 파이프라인 소유이며, 에이전트 위반 **0건**.

## 1. tune REVIEW 779장 정량 검증 (OpenCV r20 기준선)

- 대상: `tune` 2,001장 중 REVIEW **779장** (실제 결함 **44건** / 정상 735건)
- 실행 시간: 64.017초 (이미지당 p50=66.2178 ms, p95=95.2887 ms)
- 판정 불변 위반: **0건**

### 1.1 Recall@Top-K% 비교 (기본 ID순 vs 에이전트 트리아지)

| 구간 | K (장) | ID순 회수율 (분모 44) | 에이전트 Recall@Top-K (분모 44, Wilson 95% CI) | PPV (정밀도) | Lift (무작위 대비) |
|---|---:|---|---|---|---:|
| Top-10% | 78 | 4.55% (2/44, CI95 [0.013, 0.151]) | 40.91% (18/44, CI95 [0.277, 0.556]) | 23.08% (18/78, CI95 [0.151, 0.336]) | 4.09x |
| Top-20% | 156 | 13.64% (6/44, CI95 [0.064, 0.267]) | 54.55% (24/44, CI95 [0.401, 0.683]) | 15.38% (24/156, CI95 [0.106, 0.219]) | 2.72x |
| Top-30% | 234 | 22.73% (10/44, CI95 [0.128, 0.370]) | 61.36% (27/44, CI95 [0.466, 0.743]) | 11.54% (27/234, CI95 [0.081, 0.163]) | 2.04x |

### 1.2 우선순위 등급(P0~P3)별 결함 밀집도(PPV)

| 등급 | 표본 수 | 실제 결함 수 | 정상 수 | PPV (결함 비율, Wilson 95% CI) |
|---|---:|---:|---:|---|
| P0 | 66 | 17 | 49 | 25.76% (17/66, CI95 [0.167, 0.374]) |
| P1 | 190 | 13 | 177 | 6.84% (13/190, CI95 [0.040, 0.114]) |
| P2 | 268 | 8 | 260 | 2.99% (8/268, CI95 [0.015, 0.058]) |
| P3 | 255 | 6 | 249 | 2.35% (6/255, CI95 [0.011, 0.050]) |

### 1.3 섭동 안정성 · 약후보 기하 분포

- `StabilityLevel` 분포: {'SENSITIVE': 120, 'STABLE': 183, 'UNSTABLE': 476}
- `WeakPattern` 분포: {'BORDER_ARTIFACT_SUSPECT': 1, 'COLLINEAR_SCRATCH_SUSPECT': 57, 'DIFFUSE_SURFACE_TEXTURE': 378, 'LOCAL_CLUSTER': 125, 'SINGLE_CANDIDATE': 218}

## 2. val 400장 결합(yolo-seg + opencv) 교차 증거 · 능동 패치 중재 실측

- 대상: `val` 400장 (task=`seg`, 총 소요 90.76초)
- 결합 상태 분포: {'NG': 55, 'OK': 210, 'REVIEW': 135}
- 교차 증거 플래그 분포: {'AREA_RATIO_LARGE': 2, 'EVIDENCE_ALIGNED': 210, 'LOCALIZATION_MISMATCH': 22, 'NG_SINGLE_ENGINE': 40, 'SCORE_GAP_LARGE': 183, 'SINGLE_ENGINE_EVIDENCE': 163, 'STATUS_MISMATCH': 172, 'UNMATCHED_COMPONENTS': 5, 'WEAK_MATCH': 5}
- 능동 에이전트 에피소드: 188건 실행, 패치 중재(`arbitrate_patches`) **56건** 발동 (p50=1.0462 ms, p95=4.2776 ms)
- 패치 중재 결론 분포: {'BORDER_NOISE_SUSPECT': 1, 'BOTH_SALIENT': 2, 'ENGINE_A_SALIENT': 52, 'ENGINE_B_SALIENT': 1}
- 계약 검증: 판정 불변 위반 **0건**, 감사 오류 **0건**, replay 불일치 **0건**

### 2.1 val 결합 REVIEW 135장 (결함 7건) 트리아지

| 구간 | K (장) | ID순 회수율 | 에이전트 Recall@Top-K (Wilson 95% CI) | PPV | Lift |
|---|---:|---|---|---|---:|
| Top-10% | 14 | 14.29% (1/7, CI95 [0.026, 0.513]) | 14.29% (1/7, CI95 [0.026, 0.513]) | 7.14% (1/14, CI95 [0.013, 0.315]) | 1.38x |
| Top-20% | 27 | 28.57% (2/7, CI95 [0.082, 0.641]) | 28.57% (2/7, CI95 [0.082, 0.641]) | 7.41% (2/27, CI95 [0.021, 0.234]) | 1.43x |
| Top-30% | 41 | 28.57% (2/7, CI95 [0.082, 0.641]) | 71.43% (5/7, CI95 [0.359, 0.918]) | 12.20% (5/41, CI95 [0.053, 0.255]) | 2.35x |

## [개선 이력] 기하 공선성 종횡비 착시 보정 전 vs 보정 후 비교

KSDD2 이미지는 폭~230px·높이~630px(약 1:2.7 세로형)이므로 공분산 고유값 비율 `lambda_max / (lambda_max + lambda_min)` 만 사용하면 가로로 넓게 퍼진 정상 표면 노이즈도 세로 분산 때문에 `COLLINEAR_SCRATCH_SUSPECT` 로 과잉 분류되었다. Wave 2에서 주축 직선 수직 RMS 폭(`orthogonal_rms_px <= 14.0px`, 또는 `max_elongation >= 4.0`·`R² >= 0.75`·`orthogonal_rms_px <= 18.0px`) 제약을 추가한 전/후 실측 비교는 다음과 같다.

| 지표 (tune REVIEW 779장 기준) | 보정 전 (Wave 1) | 보정 후 (Wave 2, `orthogonal_rms_px <= 14px` 적용) | 변화 |
|---|---:|---:|---:|
| `COLLINEAR_SCRATCH_SUSPECT` 건수 (비율) | 391/779 (50.2%) | 57/779 (7.3%) | -334건 (-42.9%p) |
| `DIFFUSE_SURFACE_TEXTURE` 건수 | 88 | 378 | +290건 |
| `LOCAL_CLUSTER` 건수 | 81 | 125 | +44건 |
| Recall@Top-10% (K=78, 분모 44) | 38.64% (17/44) | 40.91% (18/44) | +2.27%p |
| Recall@Top-20% (K=156, 분모 44) | 52.27% (23/44) | 54.55% (24/44) | +2.28%p |
| Recall@Top-30% (K=234, 분모 44) | 59.09% (26/44) | 61.36% (27/44) | +2.27%p |
| P0 등급 PPV (실제 결함/표본) | 18.75% (18/96) | 25.76% (17/66) | +7.01%p |

## 3. 한계 및 실패 사례 분석

1. **종횡비 편향 보정 완료 (`orthogonal_rms_px <= 14.0px` 회랑 폭 제한)**: KSDD2 1:2.7 세로형 이미지에서 비등방 분산 비율만으로 과잉 분류되던 `COLLINEAR_SCRATCH_SUSPECT` 를 Wave 1 391/779건(50.2%)에서 Wave 2 57/779건(7.3%)으로 보정하였다.
2. **약한/저대비 결함의 후순위화 (`P3` 누락)**: 블러 섭동에서 급감하거나 국소 대비가 낮은 미세 결함은 P3 로 내려갈 수 있다(작업자 전수 검토 보류 시 조기 발견 지연 가능).
   - P3 누락 결함 예: `10045` (score=0.285, UNSTABLE, DIFFUSE_SURFACE_TEXTURE)
   - P3 누락 결함 예: `11169` (score=0.259, UNSTABLE, DIFFUSE_SURFACE_TEXTURE)
   - P3 누락 결함 예: `20564` (score=0.233, UNSTABLE, DIFFUSE_SURFACE_TEXTURE)
3. **고대비 정상 반사광의 `P0` 상승**: 국소 대비와 안정성이 높은 표면 기포·조명 경계는 결함 신호와 유사해 `P0` 로 올라올 수 있다.
   - P0 정상 오경보 예: `20472` (score=0.919, STABLE, SINGLE_CANDIDATE)
   - P0 정상 오경보 예: `11136` (score=0.918, STABLE, SINGLE_CANDIDATE)
   - P0 정상 오경보 예: `10058` (score=0.859, STABLE, SINGLE_CANDIDATE)
4. **결합(`yolo-seg+opencv`) 후 잔여 `val` REVIEW 의 저신호 집중**: 고대비 결함은 `yolo-seg` 가 이미 `NG` 로 확정하므로, 잔여 REVIEW 135장 중 실제 결함 7건은 저대비·미세 결함이 많으나, 종횡비 착시 보정 후 Top-10%(1/7, 1.38x Lift), Top-30%(5/7, 2.35x Lift)로 개선되었다 (단, 소표본으로 Wilson 구간 폭은 여전히 넓다).
