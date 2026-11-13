# T08 봉인 측정 보고서 (official_test, 2026-11-13)

- 분할: `official_test` (봉인, `--allow-sealed` 일회성 측정)
- 표본: defect 110 (전수) + good 894 (전수) = 1004
- 동결 config: `configs/base.toml` (hash `a762ba639441`)
- 코드 SHA: `a6671e42a855dc889d7cefe19aa2e674b21329df`
- 봉인 접근: access_count=1, reuse=false, post_seal_config_change=false
- 목적: T08 봉인 test 일회성 측정

모든 비율은 `값 [Wilson 95% CI] (분모 n)`으로 적는다. 분모가 0이면 값을 쓰지 않는다.
**test를 본 뒤 임계값·config는 바꾸지 않는다**(설계 §4.4). 목표 미달도 숨기지 않는다.

> official_test는 내부 tune/test와 중복이 있어 **독립 감사가 아니다**(보조 측정).
> 중복 수량은 `docs/evidence/T08_data_verify.md` 감사 증거를 참조한다.

## 요약 판정 (목표 대비 실제 값)

| 목표 | 기준 | 실제 값 | 판정 |
|---|---|---|---|
| escape 점추정 | <= 0.15 | 0.1091 | PASS |
| escape Wilson 상한 | <= 0.20 | 0.1810 | PASS |
| false-NG | <= 0.20 | 0.0805 | PASS |
| review | <= 0.40 | 0.3775 | PASS |
| recall@IoU0.10 | >= 0.50 | 0.5462 | PASS |

## 1. 기준선·제안 파이프라인 (설계 §4.1·§4.2)

| 지표 | P |
|---|---|
| defect_escape_rate | 0.1091 [0.064, 0.181] (분모 110) |
| defect_ng_rate | 0.6273 [0.534, 0.712] (분모 110) |
| defect_review_rate | 0.2636 [0.190, 0.353] (분모 110) |
| good_false_ng_rate | 0.0805 [0.064, 0.100] (분모 894) |
| good_review_rate | 0.3915 [0.360, 0.424] (분모 894) |
| review_rate | 0.3775 [0.348, 0.408] (분모 1004) |
| localization_recall_iou0.10 | 0.5462 [0.457, 0.633] (분모 119) |
| localization_recall_iou0.30 | 0.3445 [0.265, 0.434] (분모 119) |

B0/B1은 REVIEW를 내지 않는 이진 기준선이다(review 지표 0이 정상).

## 2. 그룹 부트스트랩 CI (F6)

b=2000, seed=20261111, alpha=0.05. 이미지 단위 복원추출, 분모 0 이미지는 제외.

| 지표(P) | 값 | CI low | CI high | n |
|---|---|---|---|---|
| escape | 0.1091 | 0.0545 | 0.1727 | 110 |
| false_ng | 0.0805 | 0.0649 | 0.0996 | 894 |
| review | 0.3775 | 0.3476 | 0.4084 | 1004 |
| recall_iou0.10 | 0.5462 | 0.4530 | 0.6293 | 110 |
| recall_iou0.50 | 0.2101 | 0.1391 | 0.2893 | 110 |

## 3. 실패 사례 (설계 §4.5 규칙 선택)

카테고리별 실패 수: defect_review 29, escape 12, false_ng 72, good_review 350

### high_score_escape_top2

| sample_id | label | status | score | best_iou |
|---|---|---|---|---|
| 20139 | defect | OK | 0.0000 | 0.0000 |
| 20159 | defect | OK | 0.0000 | 0.0000 |

### high_score_false_ng_top2

| sample_id | label | status | score | best_iou |
|---|---|---|---|---|
| 20462 | good | NG | 1.2070 | 0.0000 |
| 20756 | good | NG | 1.1399 | 0.0000 |

전체 실패 색인: `docs/evidence/failure_cases/T08_official_test_failure_index.csv` / 대표 사례: `docs/evidence/failure_cases/T08_official_test_failure_selection.json`

## 4. 재현

```powershell
uv run surface-sentinel eval --split official_test --allow-sealed
```

산출물:

- evidence: `docs/evidence/T08_sealed_official_test.json`
- report: `docs/reports/T08_sealed_official_test_2026-11-13.md`
- failure_index: `docs/evidence/failure_cases/T08_official_test_failure_index.csv`
- failure_selection: `docs/evidence/failure_cases/T08_official_test_failure_selection.json`
- lock: `data/splits/official_test.lock.json`
