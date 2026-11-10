# T03 데이터 감사 — KolektorSDD2 반입본, manifest, 분할, 누출 검사

- 감사 범위: T03. 원본 재디코드, manifest/분할 수량, L1–L8, contact sheet.
- 코드 리비전과 실행 시각: 지어내지 않는다. 증거 파일을 참조한다
  (`docs/evidence/T03_commands.log`, `docs/evidence/T03_summary.txt`).
- sealed test 접근: 없음. `batch`/`eval`을 실행하지 않았고 `--allow-sealed`를 쓰지 않았다.

## 1. 요약

- 데이터 감사 결과: **PASS**
- 데이터 위반: 0건, 경고: 1건
- 분할 검증 결과: **PASS**, 위반 0건
- manifest 전체 행: 3336
- 정본 행: 3335
- P_test: 142 (하한 100)
- test 비율: 0.4000 (허용 0.38–0.42)

## 2. 공식 수량 vs 로컬 측정

공식 수량은 T01 사실표의 공식 페이지 진술과 합산 도출값이다. 로컬 값은
이번 감사에서 manifest/분할 CSV를 다시 읽어 센 값이다.

| 항목 | 공식 수량 | 로컬 측정 |
|---|---:|---:|
| 전체 정본 이미지 | 3335 | 3335 |
| 결함 이미지 | 356 | 356 |
| 정상 이미지 | 2979 | 2979 |
| 공식 train | 2331 | 2331 |
| 공식 train 결함 | 246 | 246 |
| 공식 train 정상 | 2085 | 2085 |
| 공식 test | 1004 | 1004 |
| 공식 test 결함 | 110 | 110 |
| 공식 test 정상 | 894 | 894 |

## 3. 3,336장과 3,335장

- manifest 전체 행은 3336장이다. 복사본 1장이 포함된 값이다.
- 정본은 3335장이다. 정본 결함 356장 + 정상 2979장.
- 복사본 행: 10301 (copy)
- 복사본은 이미지와 마스크의 sha256이 정본과 같다. 원본 이미지/마스크를 삭제하거나 이동하지 않았다.
- 복사본 마스크 이름은 `_GT`가 ` (copy)` 앞에 들어간다: `<id>_GT (copy).png`. stem + `_GT.png`로 찾으면 이 마스크를 놓친다.

## 4. 이미지와 마스크

- 디코드한 이미지: 3336장
- 이미지 폭: 184–241
- 이미지 높이: 597–665
- 채널 분포: {'3': 3336}
- dtype 분포: {'uint8': 3336}
- format 분포: {'PNG': 3336}
- 디코드한 마스크: 3336개
- 정상 마스크: 비어 있음 2979개, 비어 있지 않음 0개
- 결함 마스크: 비어 있지 않음 357개, 비어 있음 0개
- 마스크 sha256 중복 그룹: 42개 (최대 2행)
- 경고: 마스크 sha256은 빈 마스크 등에서 중복되므로 이미지 그룹 키로 쓰지 않았다.

## 5. 중복 그룹핑

- dHash threshold: 2
- 그룹 수: 3335
- 최대 그룹 크기: 2행
- 2% 상한: 66.72행 (비율 0.02)
- 2% 초과 그룹: 0개
- manifest group_id 수: 3335
- 정확히 같은 image_sha256을 먼저 합치고, dHash Hamming 거리 ≤ 2를 추가로 합쳤다.
- dHash는 grayscale 9×8 INTER_AREA 축소 후 가로 인접 비교 64bit다.

## 6. 분할

| 분할 | 행 | 결함 | 정상 | sha256 |
|---|---:|---:|---:|---|
| tune | 2001 | 214 | 1787 | `399e50eb61b32f5a60f2c069a38fdafd383260df4c8d2e94089e7bea4512e463` |
| test | 1334 | 142 | 1192 | `c467b8a71d616fc0bd9c093dad7eed20f28088e53480c1c1de81aa16d4614962` |
| official_test | 1004 | 110 | 894 | `060bd376189cefb296e07e4b2874d5719ba13fe27ac9fc80f47a1db5e1fbc7a8` |

- test 비율: 0.4000
- P_test: 142 (하한 100)
- seed: 42
- split_hash: `2490f3cacd80e7a8cd3b9756bad67d21a8e841b4f3bff06bae3c324bf0ffb314`
- split_hash recipe: `sha256(canonical_json({seed,test_ratio_target,p_test_min,dhash_threshold,tune_sha256,test_sha256,official_test_sha256}))`
- 공식 test와 내부 test 중복: sample 425개, group 425개
- 공식 test와 내부 tune 중복: sample 579개, group 579개
- 공식 test 중복은 보고만 하며 임계값 튜닝에 사용하지 않았다.

## 7. 누출 검사 L1–L8

| 검사 | 결과 | 값 | 설명 |
|---|---|---:|---|
| L1 | PASS | 0 | tune/test original_id 교집합 |
| L2 | PASS | 0 | tune/test image_sha256 교집합 |
| L3 | PASS | 0 | tune/test group_id 교집합 |
| L4 | PASS | 0 | 그룹 상속/일관성 위반 |
| L5 | PASS | 1 | test.csv 해시와 test.sha256 일치 |
| L6 | NOT_APPLICABLE | 0 | T03 범위에서는 run 로그를 만들지 않습니다. |
| L7 | REPORTED | 1004 | 공식 test와 내부 test/tune 중복(보고만, 튜닝 없음) |
| L8 | PASS | 0 | 라벨/마스크 일치 |
| official_test_identity | PASS | 0 | official_test.csv 행 집합/라벨/defect_pixels/group_id/original_id 일치 |
| split_report | PASS | 0 | split_report.json 분할 sha256과 split_hash 재계산 검증 |

## 8. Contact sheet

- 경로: `data\derived\contact_sheets\tune_contact_sheet.png`
- 선정 규칙: tune 결함 전체(최대 400장) + tune 정상에서 정렬 후 균등 간격 표본 120장
- 선정 수: 결함 214장, 정상 120장, 전체 334장
- 배치: 8열, 썸네일 높이 150px
- contact sheet에는 tune 행만 넣는다. 내부 test sample_id가 들어가면 예외로 중단한다.

## 9. Sealed test

- sealed test 접근: 없음.
- `sealed_test_accessed`: false
- 이 감사는 test/official_test 파일을 읽어 수량과 누출만 확인하며, 성능 측정·임계값 튜닝·시각 검토에 사용하지 않았다.

## 10. 한계와 미해결

- 단일 통제 데이터셋이며 현장 일반화를 주장하지 않는다.
- 공식 train/test 분할은 보조 감사용이며 독립적인 현장 시험이 아니다.
- 공식 test 1,004장 중 internal tune 579장, internal test 425장이 포함된다. 설계는 internal test 중복만 보고하도록 명시했으나, tune에 포함된 579장은 official_test의 독립 감사를 제한한다. T05/T08에서 official_test를 보조 감사로 쓸 때 이 한계를 명시해야 한다.
- dHash 근접 중복은 휴리스틱이다. 물리적 부품 ID는 확인되지 않았다.
- 마스크 sha256은 빈 마스크 때문에 중복되므로 그룹 키로 쓰지 않았다.
- `split_weakly_*.pyb` 5개는 공식 약지도 split 정의이며 현재 설계에서 사용하지 않는다. manifest/분할에서 제외하고 존재만 기록했다.
- 코드 리비전과 실행 시각은 이 보고서에 직접 적지 않는다. `docs/evidence/T03_commands.log`와 `docs/evidence/T03_summary.txt`를 참조한다.

## 11. 재현 명령

```
uv run surface-sentinel data build-manifest --raw data/raw/kolektorsdd2 --manifest data/manifests/dataset_manifest.csv --errors data/manifests/build_errors.json
uv run surface-sentinel data make-splits --manifest data/manifests/dataset_manifest.csv --out-dir data/splits --seed 42
uv run surface-sentinel data verify --raw data/raw/kolektorsdd2 --manifest data/manifests/dataset_manifest.csv --splits-dir data/splits --audit-json data/audit/T03_audit.json --audit-md docs/research/T03_data_audit.md --contact-sheet data/derived/contact_sheets/tune_contact_sheet.png
```

## 13. 경고 목록

- 마스크 sha256은 빈 마스크 등에서 중복되므로 중복 그룹 키로 사용하지 않았습니다. 중복 마스크 해시 그룹: 42개
