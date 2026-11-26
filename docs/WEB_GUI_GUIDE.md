# 로컬 웹 콘솔 사용 가이드 — Surface Sentinel

Surface Sentinel의 웹 콘솔은 내 PC(`127.0.0.1`)에서만 열리는 오프라인 화면이다. 외부 CDN이나 네트워크 요청 없이 검사 과정 확인, 사진 검사, 검토 큐 처리, 통계 확인을 할 수 있다.

## 1. 사전 준비

| 항목 | 기준 |
|---|---|
| Python | 3.11 (`uv`가 가상환경 `.venv`를 만든다) |
| uv | 0.12 이상 |
| 브라우저 | Chrome·Edge 등 최신 브라우저 |
| 원본 데이터 (선택) | `data/raw/kolektorsdd2/` (`train/`, `test/`) — 없어도 콘솔은 열린다 |

저장소 루트에서 의존성을 맞춘다.

```bash
uv sync --frozen                 # 기본 의존성
uv sync --frozen --extra yolo    # YOLO(ONNX) 비교 트랙까지 쓸 때
uv run surface-sentinel doctor   # 환경 점검
```

`runs/` 폴더가 없으면 서버가 시작되지 않는다. 처음이라면 배치를 한 번 돌려 run을 만든다(원본 데이터 필요).

```bash
uv run surface-sentinel batch --split tune --limit 20
```

## 2. 실행·접속·종료

```bash
uv run surface-sentinel serve --with-api
```

- 접속 주소: **http://127.0.0.1:8000**
- `--with-api`: 사진 검사 워크벤치(`/inspections`)와 로컬 검사 API(`/api/*`)를 함께 켠다. 빼면 읽기 전용 콘솔만 열린다.
- 종료: 서버를 띄운 터미널에서 **Ctrl+C**.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--port` | `8000` | 포트. 호스트는 `127.0.0.1`로 고정 |
| `--runs-dir` | `runs` | run 디렉터리와 `records.db` 위치 |
| `--raw-root` | `data/raw/kolektorsdd2` | 원본 이미지 루트. 없으면 원본 없이 표시 |
| `--with-api` | 꺼짐 | 사진 검사 워크벤치와 `/api/*` 활성화 |

## 3. 메뉴별 사용법

상단 메뉴 4개와 하단 보조 링크 2개로 구성된다.

### 3.1 10단계 검사 과정 — `/cinema`

한 장의 사진이 10단계 OpenCV 파이프라인(입력 검증 → ROI → 전처리 → 후보 생성 → 정제 → 특징 → 점수 → 판정 → 저장)을 거치는 과정을 단계별로 보여 준다.

1. 상단 프리셋 줄에서 사진을 고른다. 첫 화면 기본값은 대표 불량 샘플 `#20117`이다. 원본 데이터가 없으면 합성 샘플(`tests/fixtures/synthetic/`)이 나온다.
2. **▶ 재생**(속도 0.5×~2.0×)이나 타임라인 슬라이더, 키보드 **←/→**로 단계를 넘기며 중간 이미지와 수치(점수, 후보 수, 판정 사유)를 확인한다. **↺**는 처음부터 다시 재생, **F**는 전체화면이다.
3. 보기 방식은 `3D 분해 스택`, `2D 시네마 와이프`, `결함 줌인 콜아웃` 중에서 고른다.
4. 내 이미지 업로드도 가능하다(PNG/JPG 등, 10MB 이하). 업로드 결과는 DB와 `runs/`에 저장하지 않는 임시 실행이다.

### 3.2 사진 검사 — `/inspections`

`--with-api`로 켰을 때 실시간 검사 워크벤치가 보인다.

- **▶ 무작위 샘플 20장 검사**: tune 분할의 공식 train 사진에서 불량 약 25%를 섞어 20장을 뽑아 연속 검사한다. 서버가 켜져 있는 동안 같은 사진을 다시 뽑지 않는다. 봉인 test는 읽지 않는다.
- **내 사진 검사**: 파일을 하나 이상 골라 검사한다.
- 결과 카드의 **상세 보기 →**에서 판정 근거(오버레이, 점수, 사유, 해시)를 볼 수 있다.
- 아래 **검사 기록** 표에서 판정(OK/NG/REVIEW/ERROR)으로 거른다.

**CSV 내보내기**

| 버튼 | 주소 | 내용 |
|---|---|---|
| CSV 저장 | `/inspections/export.csv` | 검사 기록 전체 |
| NG 이력 CSV | `/inspections/export.csv?status=NG` | NG 판정만 |

CSV는 UTF-8 BOM으로 저장되어 Excel에서 한글이 깨지지 않는다. `status`, `engine` 쿼리로 거를 수 있다.

### 3.3 검사 허브 — `/`

`runs/`에 있는 배치 run 목록이다.

- run을 누르면 `/runs/<run_id>`에서 요약과 샘플 그리드를 본다.
- 샘플 상세(`/runs/<run_id>/samples/<sample_id>`)에서 원본·ROI·response·오버레이와 컴포넌트 표를 확인한다.
- **검토 큐** `/runs/<run_id>/review?sort=priority`: REVIEW 건을 위험순으로 정렬한다. **J/K** 키로 다음·이전 샘플로 이동하고, OK/NG/보류 중 하나와 사유를 입력해 저장한다. 결정은 `review_decisions.jsonl`에 덧붙여 기록(append-only)되며, 기계 판정 OK/NG는 바뀌지 않는다.
- **지표** `/runs/<run_id>/metrics`: 혼동행렬과 분모가 붙은 비율 지표.
- **트리아지** `/runs/<run_id>/triage`: REVIEW 우선순위(P0~P3)와 Recall@Top-K 회수 곡선.

### 3.4 검사 통계 — `/stats`

검사 기록 DB(`runs/records.db`)를 읽어 상태 분포, 엔진별 비교, 분모가 붙은 비율 지표, 사람 검토 이력, run별 요약을 보여 준다. 읽기 전용이다.

### 3.5 내부 감사 — `/watchdog` (하단 링크)

- 저장소 하드 게이트(`surface-sentinel gate`와 같은 6개 검사) 결과
- run 디렉터리와 DB 무결성 감사
- PSI 분포 드리프트 감시(기준 run vs 대상 run 선택)
- 사람-기계 피드백 캘리브레이션과 하드 샘플 큐

모두 관찰 전용이며 판정이나 설정을 바꾸지 않는다.

### 3.6 run 비교 — `/compare` (하단 링크)

기준(base)과 대상(target) run을 골라 지표를 나란히 비교한다. Δ(= target − base)는 좋아진 방향이면 초록, 나빠진 방향이면 빨강으로 표시된다.

## 4. 자주 막히는 상황

| 증상 | 원인 | 해결 |
|---|---|---|
| `runs 디렉터리가 없습니다` 오류 후 종료 | `runs/`가 없음 | `uv run surface-sentinel batch --split tune --limit 20`으로 run을 만들거나 `--runs-dir`로 경로 지정 |
| `/inspections`에 검사 버튼이 없음 | `--with-api` 없이 실행 | `serve --with-api`로 다시 실행 |
| "샘플 폴더를 찾을 수 없습니다" | 원본 데이터 없음 | `data/raw/kolektorsdd2/train/`을 배치하거나 `--raw-root` 지정. **내 사진 검사**는 원본 없이도 동작 |
| "샘플 N장을 모두 검사했습니다" | 무작위 대기열 소진 | 서버를 재시작하면 대기열이 다시 채워진다(검사 기록은 유지) |
| 포트 사용 중(`address already in use`) | 다른 프로그램이 8000번 사용 | `--port 8010` 등으로 변경 |
| 저장·업로드 시 403 | 다른 주소에서 요청(Origin 검사) | 주소창에 서버가 출력한 `http://127.0.0.1:<포트>`를 그대로 사용 |
| `/stats`·검사 기록이 비어 있음 | `records.db`에 기록 없음 | `/inspections`에서 검사를 실행하거나 워커로 기록 생성 |
| 원본 이미지 칸이 비어 있음 | `--raw-root` 경로에 원본 없음 | 원본 배치 후 새로고침 |
| 터미널 한글 깨짐(Windows) | 콘솔 코드페이지 | PowerShell에서 `$env:PYTHONUTF8 = "1"` 설정 후 실행 |
| `ModuleNotFoundError: onnxruntime` | YOLO extra 미설치 | `uv sync --frozen --extra yolo` |

## 5. 보안·데이터 원칙

- 서버는 `127.0.0.1`에만 바인딩되며 외부에서 접속할 수 없다.
- 설정 편집, 데이터 삭제 기능은 없다. 쓰기는 검사 기록 추가와 REVIEW 검토 결정 추가뿐이다.
- 봉인 `test` 분할은 콘솔에서 읽지 않는다.
