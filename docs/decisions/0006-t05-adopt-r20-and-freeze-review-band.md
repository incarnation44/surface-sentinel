# ADR 0006 — T05는 r20을 채택하고 OK/REVIEW 밴드 상한을 0.40으로 동결한다

- 상태: 확정 · 2026-11-11 (T05 채택, 채택 커밋 승인 대기)
- 맥락: 전수 tune 2,001장(결함 214 + 정상 1,787)에서 통과 7건 중 5건이
  5게이트를 만족했다. r20은 escape 11.68%(25/214, CI 상한 16.68%),
  false-NG 5.82%(104/1,787), review 38.93%(779/2,001),
  recall@IoU0.10 59.92%(145/242), 절단 0/2,001로 균형이 가장 좋았다.
  r17은 escape 9.81%로 더 낮지만 false-NG 13.26%·절단 163/2,001이고,
  r23은 false-NG 3.75%지만 review 41.63%로 게이트를 넘는다.
- 결정:
  1. `configs/base.toml`의 `[candidates]`를 r20 값(z 7.6, contrast 16,
     open 3, close 5, merge 5, min_area 40, max_candidates 20)으로,
     `[scoring]`을 S4 값(ng_area 800, ng_contrast 22, ng_min 0.50,
     multi_candidate_margin 0.15)으로 바꾼다.
  2. `ScoringConfig.ok_max`를 제거하고 `REVIEW_BAND_MAX = 0.40`을 모듈
     상수로 동결한다. OK 분기는 "유효 컴포넌트 없음 + 플래그 없음"으로
     단순화한다(점수 비교는 0=score라 동치). 경계 에스컬레이션과 다중
     약후보 하한은 동결 상수를 쓴다.
  3. 채택 이전 TOML의 `ok_max`는 로드 시 무시하고 config hash에 넣지 않는다.
     과거 후보·증거 스냅샷을 수정하지 않기 위한 호환 규칙이다.
  4. config hash 변경은 이번 채택에서 1회로 묶는다. 제거 이전 base
     `22ff6bcb…`, r20 후보 `de1da847…`은 "제거 이전 값"으로만 기록한다.
  5. F6로 이미지 단위 그룹 부트스트랩(b=2000, seed=20261111)과 r18 대비
     짝 부트스트랩 delta를 `docs/evidence/t05_adopted_verify.json`에 남긴다.
- 대안:
  - r17 채택 — escape는 낮지만 false-NG·절단이 커져 균형 목적과 다르다.
  - r23 채택 — false-NG는 최저지만 review 41.63%로 현행 게이트를 넘는다.
    게이트 완화는 별도 정책 결정이므로 채택안에서 제외했다.
  - `ok_max` 유지 — 유효 컴포넌트 0개일 때 score=0이라 조건이 항상 참인
    죽은 손잡이이고, 해시·문서 혼선만 남긴다.
- 근거: `docs/reports/T05_adopted_2026-11-11.md`,
  `docs/evidence/t05_adopted.jsonl`,
  `docs/evidence/t05_adopted_verify.json`. 채택 캐시와 r20 이전 캐시는
  샘플 2,001장에서 status·reason·컴포넌트가 0건 차이였다.
- 결과: 새 base config hash는 `a762ba63944133ddfbc2f962342001e09d0dffd0f0ea14fd79095274865d8098`.
  tune 성능은 측정됐지만 생산 일반화·sealed test는 아직 미입증이다.
  recall@IoU0.50 20.66%, escape 병목은 면적/모폴로지로 남아 있다.
  다음 단계는 T07 로컬 UI다.
