# 기기 간 재현성 전체 검증 보고서

- 기기: 검증 PC B · Windows 11 Pro 10.0.26300 x64 · i5-12400 · RAM 16 GB
- 기준 기기: 개발 PC A (동일 onnxruntime 버전)
- 환경: Python 3.11.16 · OpenCV 4.14.0 · numpy 2.4.6 · onnxruntime 1.31.1
- 봉인 접근: 없음. `test`/`official_test`는 **해시 대조만** 했고 열지 않았다. `--allow-sealed` 미사용.
- 코드·설정 변경: 0건. 이 검증은 읽기·재실행만으로 수행했다.

## 1. 결론

**전 항목 PASS.** 개발 PC A(최종 검증 546 passed)와 검증 PC B 수치가 일치하고,
YOLO val 재측정 보고서가 커밋본과 **바이트 단위 동일**하게 재생성됐다(결정론 재현 입증).
모델 4종·manifest·split·config 해시가 전부 기준선과 일치한다.

## 2. 검증 항목과 결과

| # | 항목 | 명령/방법 | 결과 |
|---|---|---|---|
| 1 | git 동기화 | `git fetch` + `rev-list --left-right --count main...origin/main` | `0 0` |
| 2 | 작업 트리 | `git status --porcelain` | clean |
| 3 | 의존성 동기화 | `uv sync --frozen --extra yolo` | 36 packages, lock 일치 |
| 4 | lock 무결성 | `uv lock --check` | exit 0 (Resolved 36 packages) |
| 5 | lint | `uv run ruff check .` | All checks passed! |
| 6 | 전체 테스트 | `uv run pytest` | **546 passed** (27.86s), 0 skipped |
| 7 | 비데이터·비브라우저 | `uv run pytest -m "not data and not browser"` | **537 passed / 9 deselected** (17.89s) |
| 8 | 환경 점검 | `uv run surface-sentinel doctor` | Python 3.11.16 · OpenCV 4.14.0 · numpy 2.4.6 |
| 9 | 모델 해시 4종 | `Get-FileHash` SHA256 | 기준 해시 4/4 일치 (§3) |
| 11 | manifest·split 해시 | `Get-FileHash` SHA256 | manifest `c255d66c…` · tune `399e50eb…` · test `c467b8a7…` · official_test `060bd376…` 전부 일치 (개봉 없음) |
| 12 | config 동치 | `Get-FileHash` SHA256 | `configs/base.toml` = `configs/candidates/adopted/t05_adopted_r20.toml` (파일 바이트 `352e7d38…`, config hash `a762ba639441…`) |
| 13 | YOLO val 재측정 | `yolo evaluate --task det/seg --split val` | 개발 PC A 수치와 완전 일치 (§4) |
| 14 | 보고서 결정론 | 재측정 후 `git status --porcelain` | `T07_yolo_eval_2026-11-13.md` 바이트 동일 재생성 — 차이 0 |

## 3. 모델 해시 대조 (4/4 일치)

| 파일 | sha256 | 크기 |
|---|---|---|
| models/det/weights/best.pt | `adc9a8109fcb666da6182ed7199956b4a4fa76b098be9e71b19ca74d3c64f072` | 6,457,834 B |
| models/det/weights/best.onnx | `293f9945075b693b2a0981b4b9cd7363da3f47f2fb2a3deb9aceb7fbf72bb4e6` | 12,649,223 B |
| models/seg/weights/best.pt | `f1d01dbd5492bcd6f46bdc4d3d7610bef062a23bc6a649e3673b49bb6d6d9bdc` | 6,989,300 B |
| models/seg/weights/best.onnx | `992c0e46cf0e4f351bffa7e4f7d3cbc13948a10d77e35f01440407dbeb1de6a9` | 13,667,278 B |


## 4. YOLO val 재측정 (400장: defect 43 · good 357 · GT 컴포넌트 50)

임계값 조정 없음(기본 ng_conf=0.50, review_lo=0.25). 기준선 config hash `a762ba639441…`.

| 지표 | YOLO det | YOLO seg | 개발 PC A 기록 |
|---|---|---|---|
| escape(/43) | 55.81% (24/43) | 39.53% (17/43) | 동일 |
| false-NG(/357) | 0.00% (0/357) | 0.00% (0/357) | 동일 |
| review(/400) | 3.50% (14/400) | 3.00% (12/400) | 동일 |
| recall@0.10(/50) | 46.00% (23/50) | 62.00% (31/50) | 동일 |
| recall@0.50(/50) | 38.00% (19/50) | 56.00% (28/50) | 동일 |

- Wilson 95% CI·픽셀 지표 등 상세: `docs/reports/T07_yolo_eval_2026-11-13.md` (재생성본 = 커밋본, 바이트 동일).
- 감사 JSONL: `data/derived/yolo/eval/yolo-{det,seg}-val.jsonl` (gitignore).
- tune 2001 결합 수치는 이번 재측정 범위가 아니다 — 개발 PC A 실측(`T07_yolo_eval_2026-11-12.md`)을 그대로 인용한다: seg+OpenCV escape 17/214 · recall@0.10 168/242 · recall@0.50 153/242.

## 5. 소요 시간 (실측)

| 작업 | 시간 |
|---|---|
| 전체 pytest 546 | 27.86s |
| 비데이터·비브라우저 537 | 17.89s |
| YOLO det val 400장 | 약 55.5s (100장당 20.9 / 10.7 / 11.5 / 12.3s) |
| YOLO seg val 400장 | 약 55.5s (동일 구간) |
| 해시 대조·lock·doctor | 수 초 |

## 6. 특이사항

1. **콘솔 한글 깨짐**: 셸 출력에서 한글 파일명이 깨져 보였다(코드 페이지 문제). 파일 작업·
   해시에는 영향 없음 — 해시 일치로 확인했다. 경로 조작 없이 그대로 진행.
2. **보고서 바이트 동일 재생성**: `yolo evaluate` 재실행이 `T07_yolo_eval_2026-11-13.md`를
   커밋본과 바이트 단위로 동일하게 덮어썼다(git diff 0). 같은 이미지·같은 정책·같은
   onnxruntime 1.31.1이면 기기 간 결정론이 성립한다는 직접 증거다.
3. **테스트 수 일치**: 개발 PC A 546 passed · 0 skipped = 검증 PC B 546 passed · 0 skipped.
   ONNX 통합 4건은 가중치 배치 후 양쪽 모두 실행 통과(skip 0).

## 7. 재현 명령 (PowerShell, 저장소 루트)

```powershell
uv sync --frozen --extra yolo
uv lock --check
uv run ruff check .
uv run pytest
uv run pytest -m "not data and not browser"
uv run surface-sentinel doctor
uv run surface-sentinel yolo evaluate --task det --onnx data/derived/yolo/models/det/weights/best.onnx --split val
uv run surface-sentinel yolo evaluate --task seg --onnx data/derived/yolo/models/seg/weights/best.onnx --split val
git status --porcelain   # 보고서 바이트 동일이면 비어 있음
```

## 8. 산출물

- 이 보고서: `docs/reports/fullverify_2026-11-13.md`
- 재생성 보고서(커밋본과 동일): `docs/reports/T07_yolo_eval_2026-11-13.md`
- 평가 JSON·JSONL: `data/derived/yolo/eval/` (gitignore)
- 상태 문서 갱신: `docs/PROJECT_STATUS.md` (전체 검증 항목 추가)
