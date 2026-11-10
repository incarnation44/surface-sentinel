# T01 데이터셋 공식 사실 확인 — KolektorSDD2

## 작성 목적

Surface Sentinel 프로젝트의 T01 범위에서, KolektorSDD2 데이터셋에 대해 **공식 페이지만으로 확인 가능한 사실**을 원문 인용과 함께 고정한다. 이후 T03 데이터 검증 및 로컬 반입본 점검에서 "공식 진술"과 "로컬 실제 확인 결과"를 섞지 않기 위한 기준 문서다.

- 확인일: 2026-11-10
- 확인 방법: 아래 공식 URL 3개만 WebFetch로 직접 읽었다. 데이터 파일 다운로드 없음, 미러 사이트 사용 없음, 로컬 반입본(`data/raw/kolektorsdd2`) 열람 없음.
- 상태 태그: `FACT`(공식 페이지 원문에서 직접 확인), `FACT (합산 도출)`(공식 원문 수치의 합산 도출), `NOT_VERIFIED`(이 문서에서 확인하지 않음)

## 사실표

| 항목 | 확인된 값 | 상태 | 공식 URL | 원문 인용 | 한국어 설명 |
|---|---|---|---|---|---|
| 결함이 보이는 이미지 수 | 356 | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "356 images with visible defects" | 공식 데이터셋 페이지가 직접 밝힌 수치다. |
| 결함이 없는 이미지 수 | 2979 | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "2979 images without any defect" | 페이지는 천 단위 구분자 없이 `2979`로 표기한다. |
| 전체 이미지 수 | 3,335 | FACT (합산 도출) | https://www.vicos.si/resources/kolektorsdd2/ | "356 images with visible defects" + "2979 images without any defect" | 356 + 2,979로 도출한 값이다. 페이지에 총계를 직접 밝히는 문장은 없다(아래 "총계·합계 수치의 도출 표시" 참조). |
| 학습 세트 양성/음성 이미지 수 | 246 / 2085 | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "train set with 246 positive and 2085 negative images" | 학습 세트 구성에 대한 공식 진술이다. |
| 학습 세트 합계 | 2,331 | FACT (합산 도출) | https://www.vicos.si/resources/kolektorsdd2/ | "train set with 246 positive and 2085 negative images" | 246 + 2,085로 도출한 값이며, 페이지가 직접 인쇄한 수치는 아니다. |
| 테스트 세트 양성/음성 이미지 수 | 110 / 894 | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "test set with 110 positive and 894 negative images" | 테스트 세트 구성에 대한 공식 진술이다. |
| 테스트 세트 합계 | 1,004 | FACT (합산 도출) | https://www.vicos.si/resources/kolektorsdd2/ | "test set with 110 positive and 894 negative images" | 110 + 894로 도출한 값이며, 페이지가 직접 인쇄한 수치는 아니다. |
| 이미지 크기 | approximately 230 x 630 pixels | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "image sizes of approximately 230 x 630 pixels" | 공식 진술은 근사치다. 로컬 실제 이미지 크기 범위는 이 문서에서 확인하지 않았다. |
| 결함 유형 | scratches, minor spots, surface imperfections, etc. | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "several different types of defects (scratches, minor spots, surface imperfections, etc.)" | 결함 유형을 예시로 든 공식 진술이다. |
| 촬영 환경 | controlled industrial environment | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "The images were captured in a controlled industrial environment." | 통제된 산업 환경에서 촬영되었다는 공식 진술이다. |
| 데이터 출처·주석 | Kolektor Group d.o.o. 제공·주석 | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "The dataset is constructed from images of defected production items that were provided and annotated by Kolektor Group d.o.o." | 결함 생산품 이미지를 Kolektor Group d.o.o.가 제공하고 주석했다는 공식 진술이다. |
| 라이선스 | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "The dataset is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License." | 데이터셋 라이선스에 대한 공식 진술이다. |
| 상업적 이용 문의 | Danijel Skočaj | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "For comerical use please contact Danijel Skočaj." | 원문에 `comerical` 철자가 그대로 쓰여 있다. 인용 시 원형을 유지한다. |
| 인용 논문 (데이터셋 페이지 요청) | Computers in Industry journal 게재 논문 | FACT | https://www.vicos.si/resources/kolektorsdd2/ | "Please cite our paper published in the Computers in Industry journal when using this dataset:" | 데이터셋 사용 시 Computers in Industry 논문을 인용하라는 공식 요청이다. |
| 논문 DOI | 10.1016/j.compind.2021.103459 | FACT | https://prints.vicos.si/publications/385 | "10.1016/j.compind.2021.103459" | ViCoS 논문 페이지가 표시하는 DOI다. |
| 논문 저자 | Jakob Božič, Domen Tabernik, Danijel Skočaj | FACT | https://prints.vicos.si/publications/385 | "Jakob Božič, Domen Tabernik, and Danijel Skočaj" | 저자 이름은 원문 표기를 유지한다(성명 사이 `and` 포함). |
| 논문 제목 | Mixed supervision for surface-defect detection: from weakly to fully supervised learning | FACT | https://prints.vicos.si/publications/385 | "Mixed supervision for surface-defect detection: from weakly to fully supervised learning" | 논문 제목 원문이다. |
| 게재지·출판사·연도 | Computers in Industry, Elsevier, 2021 | FACT | https://prints.vicos.si/publications/385 | "Computers in Industry, Elsevier, 2021" | 학술지, 출판사, 연도에 대한 공식 표기다. |
| arXiv 초록의 데이터셋 규모 표현 | over 3000 images | FACT | https://arxiv.org/abs/2104.06064 | "We also present a new dataset termed KolektorSDD2 with over 3000 images containing several types of defects …" | `over 3000`은 하한 표현이다. 3,335와 모순되지 않지만 총계 진술이 아니므로 총계 근거로 대체할 수 없다. |

### 총계·합계 수치의 도출 표시

- 356과 2,979는 공식 문장에 직접 적힌 값이다(`FACT`).
- 3,335는 공식 페이지에 직접 적힌 총계가 **아니다**. 356 + 2,979의 합산 도출값(`FACT (합산 도출)`)이다. 공식 데이터셋 페이지에는 전체 이미지 수를 직접 밝히는 문장이 없다.
- 2,331은 246 + 2,085의 합산 도출값(`FACT (합산 도출)`)이다.
- 1,004는 110 + 894의 합산 도출값(`FACT (합산 도출)`)이다.
- 위 합계 값들은 공식 페이지가 인쇄한 수치가 아니므로, 인용할 때는 반드시 "공식 문장의 합산 도출"이라고 표시해야 한다.

## 설계 문서 확인 문장과의 대조

설계 문서에 인용된 공식 페이지 문장 11개, ViCoS 논문 페이지 항목 4개, arXiv 초록 문장 1개를 이번에 직접 읽은 원문과 대조했다. **실질적 불일치는 없었다.** 모든 문장이 원문과 일치했다. 표기상 차이만 다음과 같이 기록한다.

- arXiv 인용은 설계 문서에서 소문자로 시작하는 문장 중간 조각(`"also present a new dataset termed KolektorSDD2 with over 3000 images containing several types of defects"`)으로 제시했고, 원문에서는 `"We also present a new dataset termed KolektorSDD2 with over 3000 images containing several types of defects …"` 문장의 일부다. 내용은 동일하다.
- ViCoS 논문 페이지의 저자 표기에는 마지막 저자 앞에 `and`가 있다. 이름 자체는 일치한다.
- 데이터셋 페이지의 `comerical`은 원문 철자 그대로이며, 오타를 임의로 고치지 않았다.

## NOT_VERIFIED — 로컬 반입본 관련 (이 문서에서 확인하지 않음)

아래 6개 항목은 공식 페이지 진술만으로는 확인할 수 없고, 이 문서 작성 과정에서 로컬 반입본(`data/raw/kolektorsdd2`)을 열어보지도 않았다. 따라서 모두 `NOT_VERIFIED`이며, T03 데이터 검증 또는 별도의 로컬 점검에서만 다룰 수 있다.

| 항목 | 상태 | 설명 |
|---|---|---|
| 실제 반입본의 마스크 파일명 | NOT_VERIFIED | 공식 페이지에 마스크 파일명에 대한 진술이 없다. 로컬 파일 목록도 확인하지 않았다. |
| 압축 해제 후 디렉터리 구조 | NOT_VERIFIED | 공식 페이지에 디렉터리 구조 진술이 없다. 로컬 압축 해제 결과도 확인하지 않았다. |
| 채널 수와 dtype | NOT_VERIFIED | 공식 페이지에 채널 수나 dtype 진술이 없다. 로컬 이미지를 열어 확인하지 않았다. |
| 실제 중복 파일 | NOT_VERIFIED | 공식 페이지에 중복 파일 여부 진술이 없다. 로컬 파일 해시 등도 계산하지 않았다. |
| 실제 부품·원본 ID 식별 방법 | NOT_VERIFIED | 공식 페이지에 부품 ID나 원본 ID 체계에 대한 진술이 없다. 로컬 파일명 규칙도 확인하지 않았다. |
| 로컬 이미지 크기 범위 | NOT_VERIFIED | 공식 진술은 "approximately 230 x 630 pixels"일 뿐이며, 로컬 실제 이미지 크기 범위는 확인하지 않았다. |

## 검산

| 검산식 | 결과 |
|---|---|
| 3,335 = 356 + 2,979 | 참 (3,335) |
| 2,331 = 246 + 2,085 | 참 (2,331) |
| 1,004 = 110 + 894 | 참 (1,004) |
| 3,335 = 2,331 + 1,004 | 참 (3,335) |
| 356 = 246 + 110 | 참 (356) |
| 2,979 = 2,085 + 894 | 참 (2,979) |

## 출처 목록과 접근 결과

| URL | 접근 결과 | 확인한 내용 |
|---|---|---|
| https://www.vicos.si/resources/kolektorsdd2/ | 접근 성공, 페이지 내용 읽음 | 위 11개 문장 원문 일치 확인. 전체 이미지 수를 직접 밝히는 총계 문장이 없음도 확인. |
| https://prints.vicos.si/publications/385 | 접근 성공, 페이지 내용 읽음 | DOI, 저자, 제목, 게재지·출판사·연도 확인. |
| https://arxiv.org/abs/2104.06064 | 접근 성공, 초록 페이지 읽음 | KolektorSDD2와 over 3000 images를 언급하는 초록 문장 확인. |

- 사용한 접근: 위 3개 공식 URL에 대한 WebFetch 직접 읽기.
- 사용하지 않은 것: 데이터 다운로드, 미러 사이트, 캐시/아카이브, 로컬 반입본 열람.

## 경계 문장

이 문서는 **2026-11-10 기준 공식 페이지에 적힌 진술만** 기록한다. 이 문서는 T03 데이터 검증이나 로컬 반입본(`data/raw/kolektorsdd2`) 확인을 대체하지 않는다. 이 문서의 수치와 인용을 로컬 파일의 실제 속성(파일명, 구조, 채널, dtype, 중복, ID, 크기 범위)에 대한 증거로 사용해서는 안 된다.
