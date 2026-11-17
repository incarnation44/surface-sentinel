# append-only 스키마 강제와 무결성 검사 (감사 5.13)

대상 코드: `records/schema.sql`, `records/db.py`, `cli.py`(`records verify`)

## 1. 규칙

기록 DB(`data/derived/records.db`)의 4개 테이블은 **쓰기 방향이 한쪽뿐**이다.

| 테이블 | 의미 | 허용 | 금지 |
|---|---|---|---|
| `inspections` | 기계 판정 대표 행 | INSERT | UPDATE, DELETE |
| `review_decisions` | 사람 판정(append-only 로그) | INSERT | UPDATE, DELETE |
| `engine_observations` | 결합 검사의 엔진별 원본 관찰 | INSERT | UPDATE, DELETE |
| `agent_assessments` | 에이전트 교차 증거 평가 | INSERT | UPDATE, DELETE |

정정이 필요하면 **행을 고치지 않고 새 행을 쌓는다.** 사람 판정 정정도
`review_decisions` 에 새 행을 추가하는 방식이다(마지막 행이 유효 판정).

감사 5.13 의 지적은 "이 규칙이 애플리케이션 코드 관례(`store.py` 가 INSERT 만
쓴다)로만 지켜지고, DB 를 직접 여는 도구·스크립트·사고에는 아무 방어가 없다"였다.
그래서 규칙을 **스키마 수준 트리거**로 내렸다.

## 2. 트리거 목록 (8개)

`schema.sql` 끝의 `-- APPEND-ONLY TRIGGERS (SQLite 전용) BEGIN` ~
`-- END APPEND-ONLY TRIGGERS` 구간이 정본이다. 전부 `CREATE TRIGGER IF NOT EXISTS`
라서 재실행이 안전하다. 위반 시 `sqlite3.IntegrityError`(`RAISE(ABORT, ...)`)가
발생하고 한국어 메시지가 그대로 올라온다.

| 트리거 | 대상 | 시점 |
|---|---|---|
| `trg_inspections_no_update` | `inspections` | BEFORE UPDATE |
| `trg_inspections_no_delete` | `inspections` | BEFORE DELETE |
| `trg_review_decisions_no_update` | `review_decisions` | BEFORE UPDATE |
| `trg_review_decisions_no_delete` | `review_decisions` | BEFORE DELETE |
| `trg_engine_observations_no_update` | `engine_observations` | BEFORE UPDATE |
| `trg_engine_observations_no_delete` | `engine_observations` | BEFORE DELETE |
| `trg_agent_assessments_no_update` | `agent_assessments` | BEFORE UPDATE |
| `trg_agent_assessments_no_delete` | `agent_assessments` | BEFORE DELETE |

트리거 이름은 `records/db.py` 의 `APPEND_ONLY_TRIGGER_NAMES` 와 1:1 이다.
이름을 바꾸면 `records verify` 가 누락으로 잡는다(의도된 안전장치).

### 트리거 DDL 의 정본은 schema.sql 하나

`db.py` 는 트리거 DDL 을 파이썬 문자열로 복제하지 않는다. `append_only_trigger_sql()`
이 마커 구간만 잘라 오고, `ensure_append_only_triggers(connection)` 가 그것을
`executescript` 로 실행한다. 테이블 DDL 을 `_schema_table_block()` 로 뽑아 쓰는
기존 패턴과 같다.

## 3. 마이그레이션과의 공존 (중요)

`migrate_to_fingerprint()` 는 v1 `inspections` 를 다음 순서로 갈아끼운다.

1. `inspections_v2` 생성 → 데이터 복사 → fingerprint UPDATE
2. `DROP TABLE inspections` ← **이 테이블의 트리거도 함께 삭제된다**
3. `ALTER TABLE inspections_v2 RENAME TO inspections`

따라서 마이그레이션만 돌리면 `inspections` 트리거 2개가 사라진다. 막는 방법:

* `init_db()` 는 `schema.sql` 실행 → 마이그레이션 → **`ensure_append_only_triggers()`**
  순서로 돈다. 마이그레이션 뒤에 항상 다시 건다(멱등).
* 순서도 안전하다. `schema.sql` 이 먼저 돌면서 v1 `inspections` 에 트리거를 걸지만,
  마이그레이션의 UPDATE 는 `inspections_v2` 를 건드리므로 트리거에 걸리지 않는다.
  `inspections_v2` 에는 트리거가 없다(트리거는 테이블을 만들 때 붙지 않는다).
* `ensure_append_only_triggers()` 는 `executescript` 를 쓰므로 **열린 트랜잭션을
  먼저 커밋**한다. 마이그레이션 트랜잭션 안에서 호출하지 말고, 끝난 뒤 호출한다.

## 4. 마이그레이션 전용 예외 절차

스키마 자체를 바꾸는 마이그레이션(예: 컬럼 추가, 테이블 재작성)은 트리거를 잠시
내려야 한다. 예외는 **마이그레이션 트랜잭션 안에서만** 허용하고, 끝나면 즉시
`init_db()` 로 재보장한다.

```python
# 마이그레이션 스크립트 예 (파이썬, 트랜잭션 1개)
from surface_sentinel.records.db import (
    APPEND_ONLY_TRIGGER_NAMES,
    connect,
    ensure_append_only_triggers,
)

connection = connect("data/derived/records.db")
try:
    with connection:  # BEGIN ... COMMIT
        for name in APPEND_ONLY_TRIGGER_NAMES:
            connection.execute(f"DROP TRIGGER IF EXISTS {name}")
        connection.execute("ALTER TABLE inspections ADD COLUMN new_col TEXT NOT NULL DEFAULT ''")
    # 트랜잭션 밖에서 재보장(executescript 가 커밋을 하므로 안/밖을 섞지 않는다)
    ensure_append_only_triggers(connection)
finally:
    connection.close()
```

수동 SQL 로 할 때(예: `sqlite3` CLI)는 같은 순서를 지킨다.

```sql
BEGIN;
DROP TRIGGER IF EXISTS trg_inspections_no_update;
DROP TRIGGER IF EXISTS trg_inspections_no_delete;
DROP TRIGGER IF EXISTS trg_review_decisions_no_update;
DROP TRIGGER IF EXISTS trg_review_decisions_no_delete;
DROP TRIGGER IF EXISTS trg_engine_observations_no_update;
DROP TRIGGER IF EXISTS trg_engine_observations_no_delete;
DROP TRIGGER IF EXISTS trg_agent_assessments_no_update;
DROP TRIGGER IF EXISTS trg_agent_assessments_no_delete;
-- ... 스키마 변경 ...
COMMIT;
-- 그리고 반드시: python -m surface_sentinel.cli records init --db <경로>
--            또는: python -m surface_sentinel.cli records verify --db <경로>  (PASS 확인)
```

주의: `PRAGMA foreign_keys` 처럼 트리거도 **연결 단위가 아니라 DB 파일에 저장**된다.
DROP 한 채로 두면 그 파일은 영구히 append-only 보호를 잃는다. 그래서 `records verify`
가 트리거 개수를 매번 확인한다.

## 5. `records verify` (무결성 검사)

`verify_records_integrity(connection)`(`records/db.py`)가 세 가지를 본다.

| 검사 | 방법 | 정상 조건 |
|---|---|---|
| 파일/페이지 무결성 | `PRAGMA integrity_check` | 결과가 `ok` |
| 외래키 | `PRAGMA foreign_key_check` | 위반 0건 |
| append-only 트리거 | `sqlite_master` 의 `type='trigger'` | 필수 8개 모두 존재 |

CLI 는 `records verify` 다. 읽기 전용이며 아무것도 쓰지 않는다.

```
python -m surface_sentinel.cli records verify --db data/derived/records.db
대상: data/derived/records.db
integrity_check: ok
foreign_key_check: 위반 0건
append-only 트리거: 8/8 존재
PASS: 기록 DB 무결성·append-only 트리거 정상
```

* 정상: exit `0`(`EXIT_OK`), 표준출력 `PASS`
* 위반: exit `5`(`EXIT_STORAGE` — 기존 DB 오류 관례), 표준에러 `FAIL` + 누락 트리거 이름
* DB 파일 없음: exit `3`(`EXIT_INPUT`), `기록 DB가 없습니다: ... (먼저 records init)`

트리거가 빠졌으면 `records init` 한 번으로 복구된다(멱등).

## 6. error_code / error_stage 조회 계약 (스키마 열 추가 대신 join)

감사 5.13 은 "`inspections` 에 `error_code` 열이 없어 오류 원인을 대표 행만으로
알 수 없다"도 지적했다. **스키마 열 추가는 채택하지 않았다.** 근거:

* `error_code` 는 원래 **엔진별 관찰**의 성질이다. 결합 검사에서 성분 엔진마다
  코드가 다를 수 있어 대표 행 한 칸에 담으면 정보가 뭉개진다.
* 이미 `engine_observations.error_code` 에 값이 있다(`store.insert_engine_observation`).
* 열을 추가하면 기존 행 backfill·`schema.sql` 정본 변경·마이그레이션 예외 절차가
  따라붙는데, 얻는 것이 조회 편의뿐이다.

그래서 **조회 계약(join)** 으로 해결한다. 대표 행 + 관찰을 함께 읽는다.

```sql
SELECT i.id, i.sample_id, i.status, i.score,
       o.engine, o.status AS engine_status, o.error_code
FROM inspections AS i
LEFT JOIN engine_observations AS o ON o.inspection_id = i.id
WHERE i.id = ?;
```

계약 요점:

* `inspections` 는 대표 판정만 갖는다. `error_code`/`error_stage` 를 여기서 찾지 않는다.
* 오류 원인은 `engine_observations.error_code`(빈 문자열 = 오류 없음)에서 읽는다.
  단일 엔진 검사는 관찰 행을 남기지 않으므로(UNIQUE 계약) 결과가 없을 수 있다 —
  이때는 빈 결과가 정상이고, NULL 을 "오류 없음"으로 단정하지 않는다.
* `error_stage` 는 현재 DB 에 영속하지 않는다(`SampleResult` 에만 있다). 필요해지면
  같은 "관찰 행 소유" 원칙으로 `engine_observations` 에 추가하는 것이 순서다.

## 7. 테스트 픽스처 주의 (알려진 충돌)

append-only 는 이제 스키마에 있으므로, **일부러 값을 변조해 하류 동작을 확인하는
테스트**는 변조 전에 트리거를 내려야 한다.

```python
connection.execute("DROP TRIGGER IF EXISTS trg_inspections_no_update")
```

현재 이 방식으로 깨지는 기존 테스트(이번 레인 스코프 밖, 미수정):

* `tests/unit/test_agent_audit.py` — `inspections.provenance_complete` UPDATE
* `tests/unit/test_api_boundary.py` — `inspections` dedup 응답 확인용 UPDATE
* `tests/integration/test_api_evidence.py` — `engine_observations` UPDATE

해결은 각 테스트에서 변조 직전 `DROP TRIGGER`(+ 필요 시 이후 `init_db` 로 복구)다.
프로덕션 코드는 영향받지 않는다(`store.py` 는 INSERT 만 쓴다).

## 8. PostgreSQL 이식 노트

SQLite 트리거는 이식되지 않는다. PG 로 옮길 때 같은 8개 규칙을 다시 세운다.

```sql
CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% 는 append-only 입니다', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_inspections_no_update BEFORE UPDATE OR DELETE ON inspections
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
-- review_decisions / engine_observations / agent_assessments 동일
```

권한으로 한 겹 더 막는 것을 권장한다(SQLite 에는 없는 층).

```sql
REVOKE UPDATE, DELETE ON inspections, review_decisions,
    engine_observations, agent_assessments FROM app_role;
```

PG 에서는 `ALTER TABLE ... ADD COLUMN` 이 트리거를 지우지 않으므로 3절의
DROP/재보장 절차가 필요 없다. 대신 `TRUNCATE` 는 별도 트리거
(`BEFORE TRUNCATE ... FOR EACH STATEMENT`)로 막아야 한다 — SQLite 에는 TRUNCATE 가 없다.