# 신뢰구간 통계 해석 개선 (감사 5.10)

- 날짜: 2026-11-17
- 범위: `src/surface_sentinel/evaluation/metrics.py`, `src/surface_sentinel/evaluation/bootstrap.py`
- 새 테스트: `tests/unit/test_metrics_bootstrap_ci.py`
- 데이터: tune 분할 캐시(`data/derived/t05_cache`, split=tune)만 사용. **봉인 split(`test`, `official_test`)은 읽지 않았다.**

## 1. 문제

1. `image_metrics`의 `balanced_accuracy` 구간은 두 성분(`NG_defect/P`, `OK_good/N`)
   Wilson 구간의 **평균**이다(`_average_interval`). 두 성분의 상관과 서로 다른
   표본 수(P≠N)를 반영하지 못하므로 평균 지표의 엄밀한 95% CI가 아니다.
2. `localization_metrics`의 `pixel_precision`/`pixel_recall` 구간은 픽셀을
   **독립 시행**으로 세는 Wilson 구간이다. 같은 이미지 안 픽셀은 공간 상관이
   있어 실제 불확실성을 과소평가한다(아래 실측에서 폭이 약 30~50배 좁다).
3. `escape` 등 이미지 단위 비율의 Wilson 구간은 이미지 하나가 시행 하나이므로
   정당하다. 이 값들은 **그대로 둔다.**

## 2. 새 함수 (하위 호환 우선)

기존 함수의 반환값·정의 문자열·`Ratio` 스키마는 **하나도 바꾸지 않았다.**
새 함수만 `evaluation/bootstrap.py`에 추가했다(기존 부트스트랩 모듈의 자리).

```python
from surface_sentinel.evaluation.bootstrap import (
    balanced_accuracy_bootstrap_ci,
    pixel_ratio_bootstrap_ci,
)
from surface_sentinel.evaluation.metrics import Confusion, PixelCounts

pairs = [(outcome.label, outcome.status.value) for outcome in report.outcomes]
balanced = balanced_accuracy_bootstrap_ci(pairs, seed=20261117, replicates=2000)
# {'value': 0.6040, 'ci_low': 0.5708, 'ci_high': 0.6367, 'n': 2001, ...}

counts = [...]  # 이미지별 PixelCounts
pixels = pixel_ratio_bootstrap_ci(counts, seed=20261117, replicates=2000)
# pixels['precision'] / pixels['recall'] 각각 value·ci_low·ci_high·분자·분모

# 그룹 단위(=같은 원본 이미지에서 나온 증강/크롭) 리샘플
grouped = balanced_accuracy_bootstrap_ci(pairs, seed=20261117, group_ids=groups)
```

| 항목 | `balanced_accuracy_bootstrap_ci` | `pixel_ratio_bootstrap_ci` |
|---|---|---|
| 리샘플 단위 | `(label, status)` 쌍 = 이미지 1장, `group_ids`가 있으면 그룹 | 이미지별 `PixelCounts`, `group_ids`가 있으면 그룹 |
| 점추정 | `(NG_defect/P + OK_good/N)/2` | `Σtp/Σpred`, `Σtp/Σgt` |
| 리샘플 수 | `replicates`(기본 2000) | `replicates`(기본 2000) |
| 구간 | percentile 95% (`alpha=0.05`) | percentile 95% |
| 결정론 | `random.Random(seed)` | `random.Random(seed)` |
| `ERROR` 처리 | 분모에서 제외(`Confusion.evaluated`와 동일) | 입력에 넣지 않는다(이미지별 카운트를 호출자가 만든다) |
| 반환 | `value`, `ci_low`, `ci_high`, `n`, `images`, `unit`, `defect_evaluated`, `good_evaluated`, `b`, `valid_b`, `skipped_b`, `seed`, `alpha` | `precision`, `recall`(각 `value`/`ci_low`/`ci_high`/`numerator`/`denominator`), `n`, `images`, `unit`, `b`, `valid_b`, `seed`, `alpha` |

**퇴화 규칙(예외 없이 `None`).**

| 상황 | 결과 |
|---|---|
| 빈 입력 | `value`/`ci_low`/`ci_high` 모두 `None`, `n=0` |
| 한쪽 클래스 분모 0(단일 클래스) | `value=None`, `ci=None`, `skipped_b=replicates` |
| 리샘플 중 한쪽 클래스 분모 0 | 그 리샘플만 건너뜀(`valid_b`/`skipped_b`에 기록) |
| 유효 리샘플 0개 | 구간만 `None`(점추정은 유지) |
| pixel 전체 분모 0 | 해당 지표만 `None` |
| `group_ids` 길이 불일치 | `ValueError`(조용한 오작동 방지) |

## 3. 기존 Wilson 근사 vs 새 bootstrap — 합성 예시

`seed=20261117`, `replicates=2000`. 합성 데이터만 쓴다.

| balanced accuracy 사례 | 점추정 | 기존 Wilson 평균 | 폭 | 새 bootstrap(이미지) | 폭 |
|---|---|---|---|---|---|
| P=5(4 NG), N=200(190 OK) | 0.8750 | [0.6430, 0.9682] | 0.3252 | [0.6495, 0.9852] | 0.3358 |
| P=3(2 NG), N=300(285 OK) | 0.8083 | [0.5634, 0.9540] | 0.3906 | [0.4735, 0.9834] | 0.5099 |
| P=10(8 NG), N=10(8 OK) | 0.8000 | [0.4902, 0.9433] | 0.4532 | [0.5989, 0.9583] | 0.3594 |
| defect 20 NG+40 ERROR, good 150 OK+10 NG | 0.9688 | [0.8638, 0.9829] | 0.1190 | [0.9490, 0.9851] | 0.0361 |

- 결함 표본이 3장뿐이면 기존 구간은 **하한을 너무 높게**(0.5634 vs 0.4735) 잡는다.
- 두 클래스 표본이 비슷하면(P=N=10) 기존 평균 구간이 오히려 더 넓다.
- ERROR가 많은 경우 기존 구간은 ERROR를 분모에서 빼고도 "평균한 Wilson"이라
  훨씬 넓게 남는다(폭 0.1190 → 0.0361).

픽셀(10장, 각 gt=pred=1000, tp 600~960):

| 산정 방식 | 점추정 | 구간 | 폭 |
|---|---|---|---|
| 픽셀 Wilson(10,000 시행) | 0.7800 | [0.7718, 0.7880] | 0.0162 |
| 이미지 bootstrap(10장) | 0.7800 | [0.7080, 0.8520] | 0.1440 |
| 그룹 bootstrap(5그룹) | 0.7800 | [0.6840, 0.8760] | 0.1920 |

같은 데이터에서 Wilson 폭이 bootstrap 폭의 **1/9** 수준이다. 사례를 하나로 묶으면
(그룹) 폭이 더 넓어진다 — 리샘플 단위가 실제로 바뀐다는 증거다.

## 4. 기존 Wilson 근사 vs 새 bootstrap — tune 캐시 실측

- 캐시: `data/derived/t05_cache/t05-cache-v1-tune-a762ba639441-46c06acb3b53.pkl` (`meta.split=tune`)
- 설정: `default_config()`(config_hash `a762ba639441…`), 표본 2001장 = defect 214 / good 1787
- 판정: defect OK 25 · NG 145 · REVIEW 44 · ERROR 0 / good OK 948 · NG 104 · REVIEW 735 · ERROR 0
- `seed=20261117`, `replicates=2000`

| 지표 | 점추정 | 기존 Wilson 근사 [lo, hi] | 폭 | 새 bootstrap [lo, hi] | 폭 |
|---|---|---|---|---|---|
| balanced accuracy | 0.6040 | [0.5598, 0.6451] | 0.0853 | [0.5708, 0.6367] | 0.0658 |
| pixel precision | 0.8656 | [0.8641, 0.8670] | 0.0028 | [0.8313, 0.8951] | 0.0638 |
| pixel recall | 0.2149 | [0.2140, 0.2157] | 0.0017 | [0.1710, 0.2670] | 0.0959 |

- 점추정값은 세 지표 모두 기존 값과 **정확히 같다**(정의 불변).
- pixel 구간 폭이 각각 **약 23배, 56배**로 벌어진다. 실제 불확실성은 Wilson 값이
  아니라 bootstrap 쪽이다(픽셀 890,783개는 독립 시행이 아니다).
- balanced accuracy는 Wilson 평균이 더 넓다. 기존 보고서의 `[0.5598, 0.6451]`은
  "보수적 근사"였고, 새 구간 `[0.5708, 0.6367]`이 평균 지표의 실제 표본 변동에
  가깝다.
- tune 캐시의 defect 214장은 `original_id`가 모두 달라 그룹 단위 리샘플 결과가
  이미지 단위와 같다(`n=214`). 그룹 리샘플 자체는 합성 5그룹 예시로 검증했다.

## 5. 채택 결론에는 영향 없음

현행 채택 게이트(`recalibration.py`, `sealed.py`)가 쓰는 값은
**escape 점추정 · escape Wilson 상한(≤0.20) · false-NG · review · recall@IoU0.10**
뿐이고 **balanced accuracy CI는 게이트에 없다.** 따라서 r20 채택 결론은 이번
추가로 뒤집히지 않는다. escape Wilson 구간은 이미지 단위 시행이라 그대로 유효하다.

## 6. 재현

```powershell
uv run --no-sync python -c "from surface_sentinel.evaluation.bootstrap import balanced_accuracy_bootstrap_ci as f; print(f([('defect','NG')]*4+[('defect','OK')]+[('good','OK')]*190+[('good','NG')]*10, seed=20261117))"
uv run --no-sync pytest tests/unit/test_evaluation_core.py tests/unit/test_metrics_localization.py tests/unit/test_evaluation_bootstrap.py tests/unit/test_metrics_bootstrap_ci.py
```

## 7. 한계

- percentile 구간이라 표본이 아주 작고 비율이 0/1에 붙으면 구간이 점에 붙는다
  (BCa 보정은 넣지 않았다 — 재현성·단순성 우선).
- 그룹 정의는 호출자 책임이다. `original_id`를 그룹 키로 쓰면 같은 원본에서
  나온 증강이 함께 뽑힌다.
- `balanced accuracy`의 Wilson 근사값은 **동결값**이라 코드에서 계속 나온다.
  새 값을 쓰려면 위 함수를 별도로 호출해 보고서에 병기한다.