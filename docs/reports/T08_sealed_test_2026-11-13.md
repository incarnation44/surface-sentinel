# T08 봉인 측정 보고서 (test, 2026-11-13)

- 분할: `test` (봉인, `--allow-sealed` 일회성 측정)
- 표본: defect 142 (전수) + good 1192 (전수) = 1334
- 동결 config: `configs/base.toml` (hash `a762ba639441`)
- 코드 SHA: `eef2a0767b57a1adab2f5a93c909842aceb17f02`
- 봉인 접근: access_count=1, reuse=false, post_seal_config_change=false
- 목적: T08 봉인 test 일회성 측정

모든 비율은 `값 [Wilson 95% CI] (분모 n)`으로 적는다. 분모가 0이면 값을 쓰지 않는다.
**test를 본 뒤 임계값·config는 바꾸지 않는다**(설계 §4.4). 목표 미달도 숨기지 않는다.

## 요약 판정 (목표 대비 실제 값)

| 목표 | 기준 | 실제 값 | 판정 |
|---|---|---|---|
| escape 점추정 | <= 0.15 | - | 판정 불가(분모 0) |
| escape Wilson 상한 | <= 0.20 | - | 판정 불가(분모 0) |
| false-NG | <= 0.20 | - | 판정 불가(분모 0) |
| review | <= 0.40 | - | 판정 불가(분모 0) |
| recall@IoU0.10 | >= 0.50 | - | 판정 불가(분모 0) |

## 1. 기준선·제안 파이프라인 (설계 §4.1·§4.2)

| 지표 | B0 | B1 | P |
|---|---|---|---|
| defect_escape_rate | 1.0000 [0.974, 1.000] (분모 142) | 0.0000 [0.000, 0.026] (분모 142) | - (분모 0) |
| defect_ng_rate | 0.0000 [0.000, 0.026] (분모 142) | 1.0000 [0.974, 1.000] (분모 142) | - (분모 0) |
| defect_review_rate | 0.0000 [0.000, 0.026] (분모 142) | 0.0000 [0.000, 0.026] (분모 142) | - (분모 0) |
| good_false_ng_rate | 0.0000 [0.000, 0.003] (분모 1192) | 1.0000 [0.997, 1.000] (분모 1192) | - (분모 0) |
| good_review_rate | 0.0000 [0.000, 0.003] (분모 1192) | 0.0000 [0.000, 0.003] (분모 1192) | - (분모 0) |
| review_rate | 0.0000 [0.000, 0.003] (분모 1334) | 0.0000 [0.000, 0.003] (분모 1334) | - (분모 0) |
| localization_recall_iou0.10 | - (분모 0) | - (분모 0) | - (분모 0) |
| localization_recall_iou0.30 | - (분모 0) | - (분모 0) | - (분모 0) |

B0/B1은 REVIEW를 내지 않는 이진 기준선이다(review 지표 0이 정상).

## 2. 그룹 부트스트랩 CI (F6)

b=2000, seed=20261111, alpha=0.05. 이미지 단위 복원추출, 분모 0 이미지는 제외.

| 지표(P) | 값 | CI low | CI high | n |
|---|---|---|---|---|
| escape | - | - | - | 0 |
| false_ng | - | - | - | 0 |
| review | - | - | - | 0 |
| recall_iou0.10 | - | - | - | 0 |
| recall_iou0.50 | - | - | - | 0 |

## 3. 실패 사례 (설계 §4.5 규칙 선택)

카테고리별 실패 수: error 1334

### high_score_escape_top2

해당 사례 없음.

### high_score_false_ng_top2

해당 사례 없음.

전체 실패 색인: `docs/evidence/failure_cases/T08_test_failure_index.csv` / 대표 사례: `docs/evidence/failure_cases/T08_test_failure_selection.json`

## 4. 재현

```powershell
uv run surface-sentinel eval --split test --allow-sealed
```

산출물:

- evidence: `docs/evidence/T08_sealed_test.json`
- report: `docs/reports/T08_sealed_test_2026-11-13.md`
- failure_index: `docs/evidence/failure_cases/T08_test_failure_index.csv`
- failure_selection: `docs/evidence/failure_cases/T08_test_failure_selection.json`
- lock: `data/splits/test.lock.json`
