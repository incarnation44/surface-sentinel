# PROJECT 02 과제 요건 검증 보고서 (2026-11-25)

> **범위**: KolektorSDD2 표면 결함 데이터셋에서 픽셀 기반 결함 후보 검사 및 사람 검토(`REVIEW`) 보조 파이프라인의 과제 요건(1~6번) 충족 여부를 실측 검증한다. 물리적 mm 정밀 광학 계측기가 아니며, 봉인 분할(`test` 1,334장, `official_test` 1,004장) 이미지는 일체 열지 않았다(`--allow-sealed` 미사용, 기존 JSON 증거만 대조).

## 1. 핵심 요약 (과제 1~6번 1:1 매핑)

| 과제 요건 | 프로젝트 구현 근거 | 실측·증거 산출물 | 판정/핵심 수치 |
|---|---|---|---|
| **1. 오검·미검 동시 관리 & 현장형 `REVIEW`** | 4상태(`OK`/`NG`/`REVIEW`/`ERROR`) 진리표 (`stages.py::decide_status`) · `r20` 동결(`base.toml`, `a762ba63`) | `docs/reports/T05_adopted_2026-11-11.md`, `docs/evidence/t05_adopted_verify.json` | `tune` 2,001장: `escape` 11.68%, `false-NG` 5.82%, `review` 38.93%, 절단 0건 |
| **2. 조명·배경 불균일 대응 전처리 3종 비교** | `CLAHE` vs `Sobel` vs 정본 `Dual Morphology (Black-hat + Top-hat + Median Blur)` 동일 후단 비교 | `scripts/verify_project02_requirements.py`, `docs/evidence/project02_preprocess_comparison.json` | `tune∩train` 1,422장 기준 정본만 목표 충족(`escape` **11.64%**, `false-NG` **4.94%**) |
| **3. 결함 면적·길이·방향 기하 계측** | `cv2.minAreaRect` 기반 장축(`length_px`)·단축(`width_px`)·방향(`angle_deg`)·연신율(`elongation`) | `src/surface_sentinel/pipeline/stages.py#L677`, `docs/evidence/failure_cases/project02_ng_and_false_ng_cases.csv` | 검출 `NG` 167건(`true_ng` 104 + `false_ng` 63) 기하 계측 완료 |
| **4. 정상 오검출(false-NG) 원인·실패사례 분석** | 경계 인접·다중 후보·조명 반사·음영 자동 분류 + T08 전수 실패 색인 | `docs/evidence/failure_cases/project02_ng_and_false_ng_cases.csv`, `T08_test_failure_index.csv` | `false-NG` 63건 중 경계 인접 54건(85.7%) · 미세 다발 7건 · 반사 1건 · 음영 1건 |
| **5. KPI 달성 여부 및 CPU 실행 속도** | `official_test` 1,004장(보조) vs 독립 봉인 `test` 1,334장 + 배치 CPU 지연 | `docs/evidence/T08_sealed_official_test.json`, `T08_sealed_test.json`, `runs/.../run.json` | `official_test` 5/5 PASS · 독립 `test` 4/5 PASS(Wilson 상한 21.55% FAIL) · 배치 P95 **158.00 ms** |
| **6. 심플 오프라인 GUI · CSV 저장 · 감사 무결성** | Zero-CDN 4메뉴 콘솔, `/inspections` 20장 자동검사 + CSV 저장, SQLite `8/8 Append-Only`, `gate` 6/6 | `src/surface_sentinel/ui/app.py`, `templates/inspections.html`, `records/schema.sql` | `GET /inspections/export.csv` (UTF-8 BOM) · `gate` 6/6 PASS · `repro check` exit 0 |

---

## 2. 전처리 3종 실측 비교 (`tune ∩ train` 1,422장)

- **표본 범위**: `data/splits/tune.csv` 중 `source_split == "train"` 전수 **1,422장** (`defect` 146장 / GT 컴포넌트 168개, `good` 1,276장).
- **봉인 격리**: `data/splits/test.csv`(1,334장) 및 `data/splits/official_test.csv`(1,004장)와 교집합 **0장** (`allow_sealed=False`).
- **공통 후단 조건**: ROI 추출, 강건 MAD Z-정규화(`noise_floor_gray=2.0`), 후보 생성·정제·점수·판정은 `configs/base.toml`(`r20`, `a762ba63`)과 동일.

| 전처리 방식 | Escape (미검) | False-NG (과검) | Review (보류) | Recall@IoU0.10 | 전처리 P50 / P95 | 파이프라인 P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| ① **CLAHE 대비 강조** (`cv2.createCLAHE`) | 52.74% (77/146) | 0.00% (0/1,276) | 1.83% (26/1,422) | 15.48% (26/168) | 3.72 ms / 5.39 ms | 5.62 ms / 8.40 ms |
| ② **Sobel 에지 그라디언트** (`cv2.Sobel`) | 44.52% (65/146) | 0.00% (0/1,276) | 5.70% (81/1,422) | 25.00% (42/168) | 2.82 ms / 3.92 ms | 4.72 ms / 6.83 ms |
| ③ **정본 Dual Morphology + Median Blur** | **11.64%** (17/146) | **4.94%** (63/1,276) | **39.31%** (559/1,422) | **63.69%** (107/168) | 8.23 ms / 12.53 ms | 10.97 ms / 16.10 ms |

### 방식별 거동 해석
1. **CLAHE (`escape` 52.74%)**: 전역·국소 히스토그램 평활화가 배경 조명 기울기와 미세 가공 무늬까지 동시에 증폭시켜 배경 MAD(`sigma`)가 커지고, 결과적으로 결함의 강건 Z-score가 `z_threshold=7.6` 아래로 희석되어 절반 이상을 놓친다.
2. **Sobel (`escape` 44.52%)**: 1차 미분 경계 검출은 결함 외곽선에만 얇게 반응하고 결함 내부 면적을 비워버린다. 이로 인해 `min_area_px=40` 및 `ng_area_px=800` 기준에서 결함이 파편화되어 미검이 급증한다.
3. **정본 Dual Morphology (`escape` 11.64%)**: `Median Blur(3)`로 점 잡음을 억제한 뒤 타원 커널(`~23px`) `Black-hat`(어두운 긁힘)과 `Top-hat`(밝은 반사)을 결합해 배경 조명을 제거하고 결함 면적을 온전히 보존한다. 인메모리 전체 파이프라인 P95는 **16.10 ms**다.

---

## 3. 길이·면적·방향(`cv2.minAreaRect`) 및 오검출 실패사례 분석

`src/surface_sentinel/pipeline/stages.py`의 `analyze_components`는 정제된 각 연결 요소 외곽선에 대해 `cv2.minAreaRect(contour)`를 호출해 최소 외접 회전 사각형의 장축·단축·연신율(`elongation`)을 산출한다. 검증 스크립트는 1,422장 중 정본 파이프라인이 `NG`로 판정한 **167건**(`true_ng` 104건 + `false_ng` 63건)의 기하·대비·원인 지표를 `docs/evidence/failure_cases/project02_ng_and_false_ng_cases.csv`에 기록했다.

### 정상 오검출(`false-NG` 63건) 원인 분포

| 오검출 원인 분류 | 건수 (비중) | 대표 사례 (`sample_id`) | 기하·대비 실측 특징 |
|---|---:|---|---|
| **경계 인접 (치구·가장자리 경계선)** | 54건 (85.7%) | `10116`, `11014`, `10814` | `edge_margin_px=0`, 장축 `124~131 px`·단축 `3~6 px`의 세장형 암부 경계(`elongation 20.7~42.3`, `score 0.78~0.86`) |
| **미세 표면 결 다발 (반복 가공 무늬)** | 7건 (11.1%) | `10190`, `10122`, `10091` | 부품 내부 다수 후보(`component_count 9~15개`), `mean_contrast_gray 28.0~40.8`, `score 0.53~0.56` |
| **조명 반사 (고휘도 포화 영역)** | 1건 (1.6%) | `10820` | `area_px=460`, `length=43.05 px`, `width=24.88 px`, `angle=39.47°`, `gray_max=255.0`, `score=0.9287` |
| **국소 표면 음영/오염** | 1건 (1.6%) | `11790` | `area_px=153`, `length=27.86 px`, `width=10.44 px`, `angle=11.77°`, `score=0.5108` |

> 참고: `true_ng` 104건은 선형 스크래치(`elongation >= 3.0`) 16건과 국소 표면 결함 88건으로 계측되었으며, 전체 계측이 픽셀 단위이므로 물리적 mm 치수 변환은 제공하지 않는다(`recall@IoU0.50`은 `tune∩train` 20.83%, 봉인 `test` 20.39%).

---

## 4. KPI · 봉인 평가 · 확장 비교 대조

| 평가 분할 / 트랙 | 표본 (결함/정상) | Escape (미검) | False-NG (과검) | Review (보류) | Recall@IoU0.10 | 비고 |
|---|---:|---:|---:|---:|---:|---|
| **`tune` (개발 동결, `r20`)** | 2,001 (214 / 1,787) | 11.68% (25/214) | 5.82% (104/1,787) | 38.93% (779/2,001) | 59.92% (145/242) | 5/5 PASS · 절단 0/2,001 |
| **`official_test` (보조 측정)** | 1,004 (110 / 894) | 10.91% [CI 상한 18.10%] (12/110) | 8.05% (72/894) | 37.75% (379/1,004) | 54.62% (65/119) | 5/5 PASS (단 `tune` 579·`test` 425 중복으로 독립 감사 아님) |
| **독립 봉인 `test` (주 측정)** | 1,334 (142 / 1,192) | 14.79% [**CI 상한 21.55% FAIL**] (21/142) | 7.21% (86/1,192) | 34.71% (463/1,334) | 57.89% (88/152) | **4/5 PASS · 1/5 FAIL → 생산 NO-GO** |

* **실행 속도 (배치 vs 인메모리)**: 디스크 I/O·오버레이 PNG 저장을 포함한 배치 20장 CPU 실측(`runs/20261112T102419Z-a762ba63-tune/run.json`)은 P50 **104.90 ms**, P95 **158.00 ms**로 200 ms 기준 이내다. 순수 인메모리 파이프라인(1,422장)은 P50 **10.97 ms**, P95 **16.10 ms**다.
* **확장 비교(YOLO / GBDT / 에이전트) 결론**: `YOLO seg` 결합은 독립 `val` 400장에서 짝차이 이득 +0.0000 (`escape` 9.30% 동일), 에이전트는 `observe-only`로 판정 변경 0.0pp를 유지하되 `REVIEW` 상위 10% 우선 검토 시 결함 40.91%(4.09x Lift)를 회수한다.

---

## 5. 웹 콘솔 CSV 저장 · 무결성 및 한계

1. **웹 콘솔 CSV 저장 (`GET /inspections/export.csv`)**: `/inspections` 필터 툴바에서 전체 이력(`CSV 저장`) 및 불량 이력(`NG 이력 CSV`, `?status=NG`)을 `_read_only_connection`으로 조회해 UTF-8 BOM(`utf-8-sig`) CSV로 다운로드한다.
2. **감사·무결성**: `configs/base.toml` 해시(`a762ba63...`) 불변, `surface-sentinel gate` 6/6 PASS, SQLite `8/8 Append-Only` 트리거 및 `repro check` exit 0을 유지한다.
3. **4대 명시적 한계**: (1) 물리적 mm 계측 아님(`recall@IoU0.50` 20.39%), (2) 생산 적용 **NO-GO**(봉인 `test` escape Wilson 상한 21.55% > 20%), (3) 사람 검토 필수(`REVIEW` 34.71%), (4) 단일 데이터셋(`KolektorSDD2`) 기반 현장 유병률 보정 미입증.

## 6. 재현 명령

```powershell
uv run python scripts/verify_project02_requirements.py
uv run pytest tests/ui/ tests/unit/test_records_append_only.py tests/unit/test_cli_gate_repro.py -q
uv run surface-sentinel gate
```
