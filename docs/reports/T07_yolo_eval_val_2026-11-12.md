# T07_yolo_eval — YOLO vs OpenCV 비교 (split=val)

- 생성일: 2026-11-12
- 기준선 config: configs/candidates/adopted/t05_adopted_r20.toml (hash a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098)
- 기준선 측정 분모: tune 2001 (defect 214 + good 1787)
- 표의 괄호는 그 행의 분자/분모다. 기준선 열 이름의 분모(/214·/1787·/2001·/242)는 기준선 분모이고, val 측정 행은 자기 분모를 칸에 적는다.
- test/official_test: SEALED — 이 보고서는 열지 않는다.

## 1. 요약

| 엔진 | escape(/214) | false-NG(/1787) | review(/2001) | recall@0.10(/242) | recall@0.50 |
|---|---|---|---|---|---|
| OpenCV r20(기준선) | 11.68% (25/214) | 5.82% (104/1787) | 38.93% (779/2001) | 59.92% (145/242) | 20.66% (50/242) |
| YOLO det (yolo-det) [MEASURED, val] | 55.81% (24/43) | 0.00% (0/357) | 3.50% (14/400) | 46.00% (23/50) | 38.00% (19/50) |
| YOLO seg (yolo-seg) [MEASURED, val] | 39.53% (17/43) | 0.00% (0/357) | 3.00% (12/400) | 62.00% (31/50) | 56.00% (28/50) |

## 2. 지표 상세 (분모 · Wilson 95%)

### OpenCV r20(기준선) — [MEASURED, tune 2001]
- defect_escape_rate: 0.1168 [0.080, 0.167] (분모 214)
- good_false_ng_rate: 0.0582 [0.048, 0.070] (분모 1787)
- review_rate: 0.3893 [0.368, 0.411] (분모 2001)
- recall_iou0.10: 0.5992 [0.536, 0.659] (분모 242)
- recall_iou0.50: 0.2066 [0.160, 0.262] (분모 242)

### YOLO det (yolo-det) — [MEASURED, val]
- automation_rate: 0.9650 [0.942, 0.979] (분모 400)
- balanced_accuracy: 0.5581 [0.520, 0.622] (분모 -)
- decided_accuracy: 0.9378 [0.909, 0.958] (분모 386)
- defect_escape_rate: 0.5581 [0.411, 0.696] (분모 43)
- defect_ng_rate: 0.1163 [0.051, 0.245] (분모 43)
- defect_review_rate: 0.3256 [0.205, 0.475] (분모 43)
- good_false_ng_rate: 0.0000 [0.000, 0.011] (분모 357)
- good_review_rate: 0.0000 [0.000, 0.011] (분모 357)
- review_rate: 0.0350 [0.021, 0.058] (분모 400)
- localization_recall_iou0.10: 0.4600 [0.330, 0.596] (분모 50)
- localization_recall_iou0.10_decided: 0.9583 [0.798, 0.993] (분모 24)
- localization_recall_iou0.30: 0.4600 [0.330, 0.596] (분모 50)
- localization_recall_iou0.30_decided: 0.9583 [0.798, 0.993] (분모 24)
- mean_iou_matched: 0.6038 (분모 23)
- pixel_precision: 0.5515 [0.549, 0.554] (분모 112535)
- pixel_recall: 0.5166 [0.514, 0.519] (분모 120139)
- recall_iou0.10: 0.4600 [0.330, 0.596] (분모 50)
- recall_iou0.50: 0.3800 [0.259, 0.518] (분모 50)

### YOLO seg (yolo-seg) — [MEASURED, val]
- defect_escape_rate: 0.3953 [0.264, 0.544] (분모 43)
- defect_ng_rate: 0.3721 [0.244, 0.521] (분모 43)
- defect_review_rate: 0.2326 [0.132, 0.377] (분모 43)
- good_false_ng_rate: 0.0000 [0.000, 0.011] (분모 357)
- good_review_rate: 0.0056 [0.002, 0.020] (분모 357)
- review_rate: 0.0300 [0.017, 0.052] (분모 400)
- automation_rate: 0.9700 [0.948, 0.983] (분모 400)
- decided_accuracy: 0.9562 [0.931, 0.972] (분모 388)
- balanced_accuracy: 0.6832 [0.612, 0.760] (분모 -)
- localization_recall_iou0.10: 0.6200 [0.482, 0.741] (분모 50)
- localization_recall_iou0.10_decided: 0.9688 [0.843, 0.994] (분모 32)
- localization_recall_iou0.30: 0.6200 [0.482, 0.741] (분모 50)
- localization_recall_iou0.30_decided: 0.9688 [0.843, 0.994] (분모 32)
- pixel_precision: 0.7987 [0.796, 0.801] (분모 77650)
- pixel_recall: 0.5162 [0.513, 0.519] (분모 120139)
- mean_iou_matched: 0.6611 (분모 31)
- recall_iou0.10: 0.6200 [0.482, 0.741] (분모 50)
- recall_iou0.50: 0.5600 [0.423, 0.688] (분모 50)

## 3. 측정 조건

- 기준선: OpenCV r20(기준선) / configs/candidates/adopted/t05_adopted_r20.toml
- 이 보고서는 자동 생성물이며 수치를 손으로 고치지 않는다.
- yolo-det: task=det / split=val / onnx=data\derived\yolo\models\det\weights\best.onnx / config_hash=a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098 / combine=아니오 / 정책 ng_conf=0.50, review_lo=0.25 / 이미지 400장 (이전 실행 JSON)
- yolo-seg: task=seg / split=val / onnx=data\derived\yolo\models\seg\weights\best.onnx / config_hash=a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098 / combine=아니오 / 정책 ng_conf=0.50, review_lo=0.25 / 이미지 400장
- 임계값 조정: 조정 없음 — 기본 정책(ng_conf=0.50, review_lo=0.25). tune 결합 평가 전 검증용.
- test/official_test: SEALED — 미사용

## 4. 결론

- 결론은 위 표의 수치로만 판단한다. 측정하지 않은 칸은 추정으로 채우지 않는다. test/official_test는 SEALED이므로 이 비교에 쓰지 않는다.
