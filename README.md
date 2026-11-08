# Surface Sentinel

KolektorSDD2 금속 표면 사진에서 결함 후보를 찾고 사람 검토를 돕는 로컬 검사 프로그램이다. 10단계 OpenCV 규칙 파이프라인으로 이미지를 처리해 정상(OK), 불량(NG), 재확인(REVIEW), 오류(ERROR) 4가지 상태로 나누고, 오버레이 이미지와 실행 기록을 남긴다. 같은 사진이면 항상 같은 점수가 나오며(점수차 0.0), 외부 네트워크 없이 오프라인에서 돌아간다.

> **범위**: 실제 밀리미터(mm)를 재는 정밀 계측기가 아니라 픽셀 기반 결함 후보 검사·검토 보조 도구다. 봉인 테스트에서 불량 놓침률(escape)의 Wilson 95% 상한이 기준(20%)을 넘어(21.55%) 생산 투입은 보류(NO-GO) 상태다.

| 구성 | 방식 | 역할 | 현재 상태 |
|---|---|---|---|
| OpenCV r20 | 10단계 규칙 영상처리 | 배경 보정, 강건 Z-Score, 모폴로지, 면적·대비 점수 계산 | 메인 엔진 (configs/base.toml 동결 `a762ba63`) |
| YOLO (det/seg) | ONNX 딥러닝 | 격리 비교 실험 (독립 검증용) | 채택 보류 (독립 val 400장 이득 +0.0000) |
| 결정론 에이전트 | 고정 규칙 정책 엔진 | 검토 우선순위 정렬, 드리프트 감시, 원인 분석 | 관찰 전용 (기계 판정 변경 불가) |

* 판정 기준: 유효 결함 후보가 있고 `score >= 0.50`이면 `NG`, 유효 후보와 품질·ROI·경계·절단·다중후보 플래그가 모두 없으면 `OK`, 그 외(유효 후보 `score < 0.50` 또는 플래그 발생)는 모두 `REVIEW`로 넘긴다. 손상 파일이나 봉인 위반은 `ERROR`로 격리해 비율 계산 분모에서 뺀다.
* 판정 권한: 기계 판정(`OK`/`NG`/`REVIEW`)은 10단계 파이프라인만 내린다. 에이전트와 비교 엔진은 판정을 바꾸지 못하며, 사람 검토도 `REVIEW` 건에만 확정·기각 기록을 덧붙인다.
* 데이터: KolektorSDD2 정본 3,335장(결함 356 / 정상 2,979, 중복 `10301 (copy)` 격리). 유사 이미지 군집(`dHash <= 2`) 단위로 나눠 누출 0건(L1–L8 PASS)을 확인했다.
  * `tune`: 2,001장(결함 214 / 정상 1,787). YOLO 비교를 위해 내부에 `tune-train` 1,601장 + 독립 `val` 400장을 따로 나눴다.
  * 봉인 `test`: 1,334장(결함 142 / 정상 1,192). 기본 실행 시 코드가 읽기 전 거부(`E_SEALED_DENIED`, exit 6)하며, 잠금 원장(`test.lock.json`)에 접근 횟수(`access_count=2`)와 사후 설정 무변경(`post_seal_config_change=false`)이 남아 있다.
  * `official_test`: 1,004장(결함 110 / 정상 894). 내부 `tune`(579)과 `test`(425)가 절반 이상 겹쳐 독립 검증으로 쓰지 않는다.

설계 요약은 [plan.md](plan.md), 상세 규격은 [docs/design/master-design.md](docs/design/master-design.md), 현황은 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md), 실행 환경은 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), 웹 콘솔 사용법은 [docs/WEB_GUI_GUIDE.md](docs/WEB_GUI_GUIDE.md), 30섹션 통합 기술 문서는 [docs/surface-sentinel-integrated-guide.html](docs/surface-sentinel-integrated-guide.html)을 참고한다.

## 구현 항목

- [x] 10단계 OpenCV 검사 파이프라인 (`inspect`)
- [x] 원자적 배치 실행 (`batch`: 임시 폴더 기록 후 `os.replace`, `run.json` 마지막 기록, 실패 시 exit 5, 봉인 거부 exit 6)
- [x] 평가·기준선·봉인 검증 (`eval`: 분모, Wilson 95% 구간, 부트스트랩 구간, B0/B1 기준선, 실패 색인)
- [x] 로컬 웹 콘솔 및 API (`serve --with-api`: 4메뉴 `10단계 검사 과정` `/cinema`, `사진 검사` `/inspections` 20장 자동검사·CSV 저장 `/inspections/export.csv`, `검사 허브` `/`, `검사 통계` `/stats` · 보조 `/watchdog`, `/compare` · SQLite append-only 트리거 8/8, 외부 CDN 0건)
- [x] 관찰 전용 에이전트 (평가·서빙 복합 신원, 교차 증거, 세로 종횡비 착시 보정 트리아지, 6대 게이트·PSI 드리프트 감시, 반사실 병목 분석 및 6×14 스트립 핫스팟)

## 성능

### 파이프라인과 봉인 테스트 (`configs/base.toml`, r20 `a762ba63`)

| 분할 / 비교 | 표본 수 | Escape (놓침) | False-NG (오검출) | Review (검토) | Recall@IoU0.10 | 결과 |
|---|---:|---:|---:|---:|---:|---|
| tune (개발) | 2,001 | 11.68% (25/214) | 5.82% (104/1,787) | 38.93% (779/2,001) | 59.92% (145/242) | 기준선 동결 (절단 0) |
| 봉인 test (독립) | 1,334 | 14.79% (21/142), Wilson 상한 21.55% | 7.21% (86/1,192) | 34.71% (463/1,334) | 57.89% (88/152) | 목표 4/5 PASS · NO-GO |
| 독립 val 400 (r20 vs YOLO 결합) | 400 | 9.30% (4/43) 동일 (짝차이 +0.0000) | 6.44% (23/357) 동일 | 33.75% (135/400) 동일 | - | YOLO 보류 · r20 유지 |

* 봉인 `test`에서 놓침 점추정(14.79%)은 목표(15% 이하) 안이지만, 결함 표본이 142장이라 Wilson 95% 상한이 21.55%(> 20%)를 넘겼다. 시험 뒤 임계값은 건드리지 않았다.
* `recall@IoU0.50`은 tune 20.66%(50/242), 봉인 test 20.39%다. 정밀한 위치 정렬이나 로봇 제어 용도로는 쓰기 어렵다.

### 관찰 전용 에이전트 결과

* 트리아지: 사진의 1:2.7 세로 비율 때문에 정상 무늬가 긁힘으로 오해되던 문제를 수직 RMS 폭 제한(`<= 14px`)으로 391건에서 57건으로 줄였다. `tune` REVIEW 779장에서 상위 10%만 먼저 봐도 결함의 40.91%(18/44, Lift 4.09x)를 찾고, P0 정밀도는 25.76%(17/66)다. 독립 `val` 상위 30%(K=41)에서는 ID순 28.57%(2/7)에서 71.43%(5/7, Lift 2.35x)로 올랐다.
* Watchdog: 저장소 6개 게이트(`surface-sentinel gate`) 통과, 재현성 검증(`repro check`) exit 0. PSI 분포 감시는 정상 홀드아웃 0.0238(NORMAL)과 합성 드리프트 5.6026(DRIFT_ALERT)을 구분하고, 표본 20건 미만에서는 경보를 막는다.
* 반사실 원인 분석: `tune` REVIEW 779장 중 95.12%(741/779)가 면적 `<= 133px` 때문에 대비가 포화돼도 NG 점수(0.50)에 못 미치는 구조였다. 6×14 스트립 좌·우 가장자리 15% 영역에 NG 43.67%, REVIEW 45.86%가 몰려 있다. 120건 레이블 기반 결정론 시뮬레이션(seed 20261113, 실제 작업자 리뷰 아님)에서는 P0 50.0%(2/4), P2 1.7%(1/60)로 나왔다.

## 한계

* 생산 NO-GO: 봉인 `test` escape Wilson 95% 상한 21.55% > 20%.
* 사람 검토량: 봉인 `test` REVIEW 34.71%(463/1,334장, 약 3장 중 1장). P3 자동 생략은 금지한다.
* 계측 범위: `recall@IoU0.50 ≈ 20.4%`로 물리적 mm 계측 불가.
* 일반화 미입증: KolektorSDD2 한 데이터셋 기준이며, 실제 공장 불량률(유병률) 보정과 다른 품종 검증은 하지 못했다.
* 속도 조건: 배치 20장 CPU 실측(`runs/20261112T102419Z-a762ba63-tune/run.json`, Windows, OpenCV r20) 기준 p50 104.90 ms / p95 158.00 ms다. 실패한 `22ff6bcb` 및 중간 실험 실행 기록은 `runs/_experiments/`로 분리 보관하며, 현장 PC 장시간 구동 지연은 미측정(`NOT_MEASURED`)이다.

## 실행

Python 3.11, [uv](https://docs.astral.sh/uv/) 0.12 이상, Git이 필요하다.

```bash
uv sync --frozen                 # ONNX 비교 트랙 포함 시 --extra yolo 추가
uv run surface-sentinel --version
uv run surface-sentinel doctor
```

웹 콘솔과 API를 같이 켜려면:

```bash
uv run surface-sentinel serve --with-api
```

`http://127.0.0.1:8000`에서 열리고, 터미널에서 Ctrl+C로 종료한다. 상단 4메뉴는 **10단계 검사 과정** (`/cinema`: 10단계 분해), **사진 검사** (`/inspections`: 20장 자동 연속 검사·요약·CSV 저장 `/inspections/export.csv`), **검사 허브** (`/`, `/runs/<run_id>/review?sort=priority`: 위험순 REVIEW 검토), **검사 통계** (`/stats`)이며, 하단 보조 메뉴로 `/watchdog`(내부 감사)과 `/compare`(run 비교)가 있다.

메뉴별 사용법과 문제 해결은 **[웹 콘솔 사용 가이드 (docs/WEB_GUI_GUIDE.md)](docs/WEB_GUI_GUIDE.md)**에 정리했다.

단일 검사(합성 테스트 이미지)와 배치 실행:

```bash
uv run surface-sentinel inspect --image tests/fixtures/synthetic/dark_defect.png --mask tests/fixtures/synthetic/dark_defect_mask.png --label defect --split tune --config configs/base.toml --output runs/inspect
uv run surface-sentinel batch --split tune --limit 20
```

## 데이터 준비

원본 KolektorSDD2 이미지와 학습 산출물은 용량과 라이선스(CC BY-NC-SA 4.0) 때문에 Git에 올리지 않는다. 직접 받은 원본을 `data/raw/kolektorsdd2/`에 둔 뒤 아래 순서로 manifest와 분할을 만든다.

```bash
uv run surface-sentinel data build-manifest --raw data/raw/kolektorsdd2 --manifest data/manifests/dataset_manifest.csv --errors data/manifests/build_errors.json
uv run surface-sentinel data make-splits --manifest data/manifests/dataset_manifest.csv --out-dir data/splits --seed 42
uv run surface-sentinel data verify --raw data/raw/kolektorsdd2 --manifest data/manifests/dataset_manifest.csv --splits-dir data/splits --audit-json data/audit/T03_audit.json --audit-md docs/research/T03_data_audit.md --contact-sheet data/derived/contact_sheets/tune_contact_sheet.png
```

출처: [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/), 논문 [prints.vicos.si/publications/385](https://prints.vicos.si/publications/385) · [arXiv:2104.06064](https://arxiv.org/abs/2104.06064) (DOI `10.1016/j.compind.2021.103459`). 사용 조건은 `NOTICE`를 따른다.

## 테스트 및 점검

```bash
uv run pytest                                  # 전체 1004건 (CI 989 + data 7 + browser 8)
uv run pytest -m "not data and not browser"    # CI 기준 989 passed
uv run ruff check .                            # 린트 0오류
uv run surface-sentinel gate                   # 저장소 게이트 6/6 PASS
uv run surface-sentinel records verify --db runs/records.db  # SQLite 트리거 8/8 PASS
uv lock --check                                # 36 packages 일치
```

## 프로젝트 구조

```text
surface-sentinel/
├── src/surface_sentinel/
│   ├── pipeline/        # 10단계 OpenCV 검사 및 증거 저장
│   ├── evaluation/      # 지표, Wilson/부트스트랩 CI, 기준선, 봉인 평가
│   ├── batch/           # 원자적 배치 실행
│   ├── io/              # manifest, 분할, 누출 검사, 헤더 가드, 유니코드 I/O
│   ├── agent/           # 정책, 교차 증거, 트리아지, Watchdog, 반사실, 워커
│   ├── records/         # SQLite append-only 저장소 및 복합 신원 키
│   ├── review/          # 사람 검토(REVIEW 전용) 저장소
│   ├── api/             # 로컬 FastAPI 서버
│   ├── ui/              # 오프라인 웹 콘솔 (Runs, Cinema, Triage, Watchdog)
│   ├── yolo/            # 격리된 ONNX 비교 트랙
│   ├── analysis/        # 6×14 스트립 공간 핫스팟 분석
│   └── cli.py           # CLI 진입점
├── configs/             # 동결 설정 base.toml(r20), agent.toml, 후보 프로필
├── data/                # manifest, 분할, 감사 JSON (raw/derived는 Git 제외)
├── docs/
│   ├── design/          # 설계 문서 및 권한 계약
│   ├── decisions/       # 아키텍처 결정 기록 (ADR 0001~0008)
│   ├── evidence/        # 평가·감사 원시 JSON
│   ├── reports/         # 실험·봉인·에이전트 보고서
│   ├── research/        # 데이터셋 사실 및 감사 문서
│   ├── PROJECT_STATUS.md
│   ├── ENVIRONMENT.md
│   ├── WEB_GUI_GUIDE.md                       # 웹 콘솔 실행·사용 가이드
│   └── surface-sentinel-integrated-guide.html  # 30섹션 자립형 기술 가이드
├── scripts/             # 짝평가, 패리티 측정, 스모크, 요구사항 검증 스크립트
├── tests/               # 단위, 통합, 데이터, 브라우저 테스트 (1000건)
├── plan.md              # 기획 및 설계 요약
├── pyproject.toml       # 패키지 설정 (v1.0.1)
└── uv.lock              # 의존성 잠금 파일
```
