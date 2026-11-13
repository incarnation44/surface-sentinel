"""Phase 3 무결성 감사 Watchdog 에이전트 (observe-only).

개별 이미지 에이전트(Tier 1) 상위에서 배치 실행 산출물(``runs/<run_id>/``),
검사 기록 DB(``records.db``), 저장소 설계 하드 게이트(H1~H7 핵심 불변 조건)를
대조·검증한다.

계약:
* 판정(``SampleResult.status`` / ``inspections.status``)이나 설정을 절대 수정하지 않는다.
* DB 는 항상 읽기 전용 URI(``?mode=ro``)로만 열며 미존재 시 새 파일을 만들지 않는다.
* 봉인 분할 CSV 파일은 절대 열거나 읽지 않는다(잠금 JSON 메타데이터만 점검).
* 외부 네트워크·셸·subprocess 를 사용하지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..config import ConfigError, default_config, load_config
from ..contracts import Status
from ..io.artifacts import CONFIG_SNAPSHOT, RESULTS_JSONL, RUN_JSON
from ..records.fingerprint import canonical_json_sha256
from ..records.store import RECOMMENDED_ACTIONS, REVIEW_PRIORITIES, STABILITY_LEVELS
from ..review.store import REVIEW_DECISIONS_JSONL, ReviewDecision, ReviewError
from .config import default_agent_config, load_agent_config
from .trace import TraceError, audit_trace, load_trace

AUDIT_SCHEMA_VERSION = "agent-audit-v1"
AGENT_TRACE_JSONL = "agent_trace.jsonl"
_SEALED_SPLIT_NAMES: tuple[str, ...] = ("test", "official_test")


class AuditVerdict(StrEnum):
    """Watchdog 감사 최종 판정 등급."""

    PASS = "PASS"
    REVISE = "REVISE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class AuditCheck:
    """단일 감사 항목 점검 결과."""

    name: str
    passed: bool
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AuditReport:
    """Watchdog 무결성 감사 종합 보고서."""

    target: str
    verdict: AuditVerdict
    checks: tuple[AuditCheck, ...]
    summary: str
    audit_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "target": self.target,
            "verdict": self.verdict.value,
            "checks": [check.to_dict() for check in self.checks],
            "summary": self.summary,
            "audit_hash": self.audit_hash,
        }


def _compute_verdict(checks: Sequence[AuditCheck]) -> AuditVerdict:
    has_reject = any((not c.passed) and c.severity == "REJECT" for c in checks)
    if has_reject:
        return AuditVerdict.REJECT
    has_revise = any((not c.passed) and c.severity == "REVISE" for c in checks)
    if has_revise:
        return AuditVerdict.REVISE
    return AuditVerdict.PASS


def _build_report(
    target: str,
    checks: Sequence[AuditCheck],
    summary: str | None = None,
) -> AuditReport:
    frozen_checks = tuple(checks)
    verdict = _compute_verdict(frozen_checks)
    passed_count = sum(1 for c in frozen_checks if c.passed)
    reject_fails = sum(1 for c in frozen_checks if (not c.passed) and c.severity == "REJECT")
    revise_fails = sum(1 for c in frozen_checks if (not c.passed) and c.severity == "REVISE")
    resolved_summary = summary or (
        f"{target}: {verdict.value} "
        f"(통과 {passed_count}/{len(frozen_checks)}, "
        f"REJECT {reject_fails}건, REVISE {revise_fails}건)"
    )
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "target": target,
        "verdict": verdict.value,
        "checks": [c.to_dict() for c in frozen_checks],
        "summary": resolved_summary,
    }
    return AuditReport(
        target=target,
        verdict=verdict,
        checks=frozen_checks,
        summary=resolved_summary,
        audit_hash=canonical_json_sha256(payload),
    )


def _read_json_dict(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"파일이 없습니다: {path.name}"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"JSON 파싱 실패({path.name}): {exc}"
    if not isinstance(loaded, dict):
        return None, f"JSON 최상위가 객체가 아닙니다: {path.name}"
    return loaded, None


def _read_results_jsonl(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.is_file():
        return None, f"파일이 없습니다: {path.name}"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return None, f"파일을 읽지 못했습니다({path.name}): {exc}"
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"{path.name} {line_no}번째 줄 JSON 오류: {exc}"
        if not isinstance(obj, dict):
            return None, f"{path.name} {line_no}번째 줄이 JSON 객체가 아닙니다"
        records.append(obj)
    return records, None


def audit_run_dir(run_dir: str | Path) -> AuditReport:
    """배치 실행 산출물 디렉터리(``runs/<run_id>/``)의 무결성을 감사한다."""

    root = Path(run_dir)
    checks: list[AuditCheck] = []

    run_json, run_err = _read_json_dict(root / RUN_JSON)
    if run_err is not None or run_json is None:
        checks.append(
            AuditCheck(
                name="run_json_complete",
                passed=False,
                severity="REJECT",
                detail=run_err or "run.json 을 읽을 수 없습니다",
            )
        )
    else:
        status_val = str(run_json.get("status", ""))
        passed = status_val == "COMPLETE"
        checks.append(
            AuditCheck(
                name="run_json_complete",
                passed=passed,
                severity="REJECT",
                detail=(
                    "run.json status=COMPLETE 확인"
                    if passed
                    else f"run.json status가 COMPLETE가 아닙니다: {status_val!r}"
                ),
            )
        )

    records, res_err = _read_results_jsonl(root / RESULTS_JSONL)
    if res_err is not None or records is None:
        checks.append(
            AuditCheck(
                name="results_counts_match",
                passed=False,
                severity="REJECT",
                detail=res_err or "results.jsonl 을 읽을 수 없습니다",
            )
        )
    elif run_json is None or not isinstance(run_json.get("counts"), Mapping):
        checks.append(
            AuditCheck(
                name="results_counts_match",
                passed=False,
                severity="REJECT",
                detail="run.json counts 필드가 없어 대조할 수 없습니다",
            )
        )
    else:
        counts = run_json["counts"]
        status_counter = Counter(str(rec.get("status", "")) for rec in records)
        expected_samples = counts.get("samples")
        expected_map = {
            "ok": counts.get("ok"),
            "ng": counts.get("ng"),
            "review": counts.get("review"),
            "error": counts.get("error"),
        }
        actual_map = {
            "ok": status_counter.get(Status.OK.value, 0),
            "ng": status_counter.get(Status.NG.value, 0),
            "review": status_counter.get(Status.REVIEW.value, 0),
            "error": status_counter.get(Status.ERROR.value, 0),
        }
        mismatches: list[str] = []
        if expected_samples != len(records):
            mismatches.append(f"samples(run.json={expected_samples}, results={len(records)})")
        for key in ("ok", "ng", "review", "error"):
            if expected_map[key] != actual_map[key]:
                mismatches.append(
                    f"{key}(run.json={expected_map[key]}, results={actual_map[key]})"
                )
        checks.append(
            AuditCheck(
                name="results_counts_match",
                passed=not mismatches,
                severity="REJECT",
                detail=(
                    f"results.jsonl {len(records)}건 및 상태별 건수 정확 일치"
                    if not mismatches
                    else "건수 불일치: " + ", ".join(mismatches)
                ),
            )
        )

    config_snapshot = root / CONFIG_SNAPSHOT
    if not config_snapshot.is_file():
        checks.append(
            AuditCheck(
                name="config_snapshot_hash",
                passed=True,
                severity="REJECT",
                detail="config_snapshot.toml 없음(검사 생략)",
            )
        )
    elif run_json is None or not run_json.get("config_hash"):
        checks.append(
            AuditCheck(
                name="config_snapshot_hash",
                passed=False,
                severity="REJECT",
                detail="run.json config_hash 가 없어 설정 스냅샷을 대조할 수 없습니다",
            )
        )
    else:
        try:
            snap_cfg = load_config(config_snapshot)
            expected_hash = str(run_json["config_hash"])
            matched = snap_cfg.config_hash == expected_hash
            checks.append(
                AuditCheck(
                    name="config_snapshot_hash",
                    passed=matched,
                    severity="REJECT",
                    detail=(
                        f"config_snapshot.toml 해시 일치({expected_hash[:12]})"
                        if matched
                        else (
                            "config_snapshot.toml 해시 불일치: "
                            f"snapshot={snap_cfg.config_hash[:12]} != run.json={expected_hash[:12]}"
                        )
                    ),
                )
            )
        except ConfigError as exc:
            checks.append(
                AuditCheck(
                    name="config_snapshot_hash",
                    passed=False,
                    severity="REJECT",
                    detail=f"config_snapshot.toml 로드 실패: {exc}",
                )
            )

    review_path = root / REVIEW_DECISIONS_JSONL
    if not review_path.is_file():
        checks.append(
            AuditCheck(
                name="review_decisions_immutable",
                passed=True,
                severity="REJECT",
                detail="review_decisions.jsonl 없음(검토 기록 없음)",
            )
        )
    elif records is None:
        checks.append(
            AuditCheck(
                name="review_decisions_immutable",
                passed=False,
                severity="REJECT",
                detail="results.jsonl 이 없어 검토 결정을 대조할 수 없습니다",
            )
        )
    else:
        by_sample = {str(rec.get("sample_id", "")): rec for rec in records}
        violations: list[str] = []
        decision_count = 0
        try:
            lines = review_path.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                decision_count += 1
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    violations.append(f"{line_no}행: JSON 객체 아님")
                    continue
                machine_status = obj.get("machine_status")
                sample_id = str(obj.get("sample_id", ""))
                if machine_status != Status.REVIEW.value:
                    violations.append(
                        f"{sample_id or line_no}행: machine_status={machine_status!r} (!= REVIEW)"
                    )
                    continue
                try:
                    ReviewDecision.from_dict(obj)
                except ReviewError as exc:
                    violations.append(f"{sample_id or line_no}행 스키마 위반: {exc}")
                    continue
                orig = by_sample.get(sample_id)
                if orig is None:
                    violations.append(f"{sample_id}: results.jsonl에 존재하지 않는 샘플")
                elif str(orig.get("status", "")) != Status.REVIEW.value:
                    violations.append(
                        f"{sample_id}: 원본 results.jsonl 상태가 {orig.get('status')!r}(!= REVIEW)"
                    )
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(f"review_decisions.jsonl 읽기/파싱 오류: {exc}")

        checks.append(
            AuditCheck(
                name="review_decisions_immutable",
                passed=not violations,
                severity="REJECT",
                detail=(
                    f"검토 결정 {decision_count}건 모두 원본 REVIEW 불변 규칙 준수"
                    if not violations
                    else "판정 불변 위반: " + "; ".join(violations[:5])
                ),
            )
        )

    trace_path = root / AGENT_TRACE_JSONL
    if not trace_path.is_file():
        checks.append(
            AuditCheck(
                name="agent_trace_integrity",
                passed=True,
                severity="REJECT",
                detail="agent_trace.jsonl 없음(에이전트 로그 없음)",
            )
        )
    else:
        try:
            events = load_trace(trace_path)
            trace_errors = audit_trace(events)
            checks.append(
                AuditCheck(
                    name="agent_trace_integrity",
                    passed=not trace_errors,
                    severity="REJECT",
                    detail=(
                        f"agent_trace.jsonl {len(events)}개 이벤트 무결성 통과"
                        if not trace_errors
                        else "트레이스 오류: " + "; ".join(trace_errors[:5])
                    ),
                )
            )
        except TraceError as exc:
            checks.append(
                AuditCheck(
                    name="agent_trace_integrity",
                    passed=False,
                    severity="REJECT",
                    detail=f"agent_trace.jsonl 로드 실패: {exc}",
                )
            )

    if records is None:
        checks.append(
            AuditCheck(
                name="overlay_files_exist",
                passed=False,
                severity="REVISE",
                detail="results.jsonl 을 읽지 못해 오버레이 파일 존재를 확인하지 못했습니다",
            )
        )
    else:
        resolved_root = root.resolve()
        missing_overlays: list[str] = []
        checked_overlays = 0
        for rec in records:
            rel = str(rec.get("overlay_path") or "").strip()
            if not rel:
                continue
            checked_overlays += 1
            rel_path = Path(rel)
            if rel_path.is_absolute():
                missing_overlays.append(f"{rec.get('sample_id', '?')}(절대경로)")
                continue
            candidate = (root / rel_path).resolve()
            if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
                missing_overlays.append(str(rec.get("sample_id", rel)))
        checks.append(
            AuditCheck(
                name="overlay_files_exist",
                passed=not missing_overlays,
                severity="REVISE",
                detail=(
                    f"기록된 오버레이 파일 {checked_overlays}건 실존 확인"
                    if not missing_overlays
                    else f"누락된 오버레이 파일 {len(missing_overlays)}건: "
                    + ", ".join(missing_overlays[:5])
                ),
            )
        )

    return _build_report(str(root), checks)


def audit_records_db(db_path: str | Path) -> AuditReport:
    """검사 기록 DB(SQLite)를 읽기 전용(``?mode=ro``)으로 열어 무결성을 감사한다."""

    path = Path(db_path)
    if not path.is_file():
        return _build_report(
            str(path),
            (
                AuditCheck(
                    name="db_exists",
                    passed=False,
                    severity="REVISE",
                    detail="검사 기록 DB 파일이 없습니다",
                ),
            ),
            summary="검사 기록 DB 파일이 없습니다",
        )

    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    checks: list[AuditCheck] = []
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "inspections" not in tables or "review_decisions" not in tables:
            checks.append(
                AuditCheck(
                    name="schema_tables_present",
                    passed=False,
                    severity="REJECT",
                    detail="필수 테이블(inspections, review_decisions)이 없습니다",
                )
            )
            return _build_report(str(path), checks)

        illegal_row = connection.execute(
            "SELECT COUNT(*) AS n FROM review_decisions r "
            "LEFT JOIN inspections i ON i.id = r.inspection_id "
            "WHERE i.id IS NULL OR i.status <> 'REVIEW'"
        ).fetchone()
        illegal_count = int(illegal_row["n"]) if illegal_row is not None else 0
        checks.append(
            AuditCheck(
                name="illegal_review_decisions",
                passed=illegal_count == 0,
                severity="REJECT",
                detail=(
                    "OK/NG/ERROR 행에 달린 불법 review_decisions 0건 확인"
                    if illegal_count == 0
                    else (
                        f"REVIEW가 아닌 행(또는 고아 행)에 연결된 "
                        f"review_decisions {illegal_count}건 발견"
                    )
                ),
            )
        )

        insp_cols = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(inspections)").fetchall()
        }
        if "inspection_fingerprint" in insp_cols:
            dup_row = connection.execute(
                "SELECT COUNT(*) AS n FROM ("
                "  SELECT image_sha256, inspection_fingerprint, COUNT(*) AS c "
                "  FROM inspections GROUP BY image_sha256, inspection_fingerprint HAVING c > 1"
                ")"
            ).fetchone()
            dup_count = int(dup_row["n"]) if dup_row is not None else 0
            checks.append(
                AuditCheck(
                    name="unique_image_fingerprint",
                    passed=dup_count == 0,
                    severity="REJECT",
                    detail=(
                        "(image_sha256, inspection_fingerprint) 중복 위반 0건 확인"
                        if dup_count == 0
                        else f"(image_sha256, inspection_fingerprint) 중복 그룹 {dup_count}건 발견"
                    ),
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name="unique_image_fingerprint",
                    passed=False,
                    severity="REJECT",
                    detail="inspections 테이블에 inspection_fingerprint 컬럼이 없습니다(v1 미이관)",
                )
            )

        if "agent_assessments" in tables:
            bad_vocab = connection.execute(
                "SELECT COUNT(*) AS n FROM agent_assessments "
                "WHERE recommended_action NOT IN (?, ?, ?) "
                "   OR (review_priority IS NOT NULL AND review_priority NOT IN (?, ?, ?, ?)) "
                "   OR (stability IS NOT NULL AND stability NOT IN (?, ?, ?))",
                (*RECOMMENDED_ACTIONS, *REVIEW_PRIORITIES, *STABILITY_LEVELS),
            ).fetchone()
            bad_count = int(bad_vocab["n"]) if bad_vocab is not None else 0
            checks.append(
                AuditCheck(
                    name="agent_assessments_vocabulary",
                    passed=bad_count == 0,
                    severity="REJECT",
                    detail=(
                        "agent_assessments 허용 어휘 준수 확인"
                        if bad_count == 0
                        else f"agent_assessments 허용 어휘 위반 {bad_count}건 발견"
                    ),
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name="agent_assessments_vocabulary",
                    passed=True,
                    severity="REJECT",
                    detail="agent_assessments 테이블 없음(생략)",
                )
            )

        if "provenance_complete" in insp_cols:
            legacy_row = connection.execute(
                "SELECT COUNT(*) AS n FROM inspections WHERE provenance_complete = 0"
            ).fetchone()
            legacy_count = int(legacy_row["n"]) if legacy_row is not None else 0
            checks.append(
                AuditCheck(
                    name="provenance_completeness",
                    passed=legacy_count == 0,
                    severity="REVISE",
                    detail=(
                        "모든 검사 행 provenance_complete=1 확인"
                        if legacy_count == 0
                        else f"provenance_complete=0 레거시/불완전 행 {legacy_count}건 존재"
                    ),
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name="provenance_completeness",
                    passed=False,
                    severity="REVISE",
                    detail="provenance_complete 컬럼이 없는 구형 스키마입니다",
                )
            )
    except sqlite3.Error as exc:
        checks.append(
            AuditCheck(
                name="db_readable",
                passed=False,
                severity="REJECT",
                detail=f"검사 기록 DB를 읽기 전용으로 열지 못했습니다: {exc}",
            )
        )
    finally:
        if connection is not None:
            connection.close()

    return _build_report(str(path), checks)


def _scan_ratio_denominators(
    node: Any,
    *,
    prefix: str,
    issues: list[str],
) -> int:
    """증거 JSON 내부의 Ratio 객체 분모 누락 여부를 재귀 점검한다."""

    checked = 0
    if isinstance(node, Mapping):
        is_ratio_obj = "numerator" in node or (
            "definition" in node
            and "value" in node
            and ("ci95_low" in node or "denominator" in node)
        )
        if is_ratio_obj:
            checked += 1
            has_denom_key = "denominator" in node
            denom_val = node.get("denominator")
            num_val = node.get("numerator")
            if not has_denom_key:
                issues.append(f"{prefix}: denominator 키 누락")
            elif num_val is not None and (
                denom_val is None
                or isinstance(denom_val, bool)
                or not isinstance(denom_val, (int, float))
            ):
                issues.append(
                    f"{prefix}: numerator가 존재하나 denominator가 비어 있음({denom_val!r})"
                )
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            checked += _scan_ratio_denominators(value, prefix=child_prefix, issues=issues)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            checked += _scan_ratio_denominators(item, prefix=f"{prefix}[{idx}]", issues=issues)
    return checked




class _GuideAuditParser(HTMLParser):
    """T10 통합 HTML 가이드의 ID·내부 링크·본문 텍스트를 수집한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.duplicate_ids: list[str] = []
        self.seen_ids: set[str] = set()
        self.internal_hrefs: list[str] = []
        self._text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1
        for k, v in attrs:
            if k == "id" and v:
                self.ids.append(v)
                if v in self.seen_ids:
                    self.duplicate_ids.append(v)
                self.seen_ids.add(v)
            elif k == "href" and v and v.startswith("#"):
                self.internal_hrefs.append(v)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._text_parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._text_parts)


def _check_t10_guide_h7(root: Path) -> AuditCheck:
    """T10 통합 HTML 가이드 존재 시 H7 기본 무결성(중복 ID 0·수치 일치)을 점검한다."""

    guide_path = root / "docs" / "surface-sentinel-integrated-guide.html"
    if not guide_path.is_file():
        return AuditCheck(
            name="G_GUIDE_H7",
            passed=True,
            severity="REJECT",
            detail="T10 통합 가이드 미존재(검사 생략)",
        )
    try:
        raw_bytes = guide_path.read_bytes()
        if len(raw_bytes) >= 2_000_000:
            return AuditCheck(
                name="G_GUIDE_H7",
                passed=False,
                severity="REJECT",
                detail=f"T10 통합 가이드 크기 초과: {len(raw_bytes)} bytes (>= 2,000,000)",
            )
        html_text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return AuditCheck(
            name="G_GUIDE_H7",
            passed=False,
            severity="REJECT",
            detail=f"T10 통합 가이드 읽기 실패: {exc}",
        )

    parser = _GuideAuditParser()
    parser.feed(html_text)
    parser.close()
    body_text = parser.text

    issues: list[str] = []
    if parser.duplicate_ids:
        issues.append(f"중복 id {len(parser.duplicate_ids)}건({parser.duplicate_ids[0]})")

    missing_sections = [
        f"sec-{idx:02d}"
        for idx in range(1, 31)
        if not any(
            ident == f"sec-{idx:02d}" or ident.startswith(f"sec-{idx:02d}-")
            for ident in parser.seen_ids
        )
    ]
    if missing_sections:
        issues.append(f"누락 섹션 {len(missing_sections)}건({missing_sections[0]})")

    broken_anchors = sorted(
        {
            href[1:]
            for href in parser.internal_hrefs
            if len(href) > 1 and href[1:] not in parser.seen_ids
        }
    )
    if broken_anchors:
        issues.append(f"깨진 내부 링크 {len(broken_anchors)}건(#{broken_anchors[0]})")

    required_tokens: list[str] = ["3,335", "2,001", "1,334", "11.68%", "a762ba63"]
    ev_dir = root / "docs" / "evidence"
    t08_path = ev_dir / "T08_sealed_test.json"
    if t08_path.is_file():
        t08, err = _read_json_dict(t08_path)
        if err is None and t08 is not None:
            goals = t08.get("goals") or {}
            esc = (goals.get("escape 점추정") or {}).get("value")
            esc_hi = (goals.get("escape Wilson 상한") or {}).get("value")
            if isinstance(esc, (int, float)):
                required_tokens.append(f"{esc * 100:.2f}%")
            if isinstance(esc_hi, (int, float)):
                required_tokens.append(f"{esc_hi * 100:.2f}%")

    p2_path = ev_dir / "P2_triage_eval_2026-11-13.json"
    if p2_path.is_file():
        p2, err = _read_json_dict(p2_path)
        if err is None and p2 is not None:
            top_k = ((p2.get("tune_review") or {}).get("triage_lift") or {}).get("top_k") or {}
            r10 = ((top_k.get("0.10") or {}).get("recall_at_k") or {}).get("value")
            r20 = ((top_k.get("0.20") or {}).get("recall_at_k") or {}).get("value")
            if isinstance(r10, (int, float)):
                required_tokens.append(f"{r10 * 100:.2f}%")
            if isinstance(r20, (int, float)):
                required_tokens.append(f"{r20 * 100:.2f}%")

    p4_path = ev_dir / "P4_counterfactual_eval_2026-11-13.json"
    if p4_path.is_file():
        p4, err = _read_json_dict(p4_path)
        if err is None and p4 is not None:
            bdown = p4.get("review_bottleneck_breakdown") or {}
            ab = (bdown.get("area_bottleneck_ratio_all_review") or {}).get("value")
            if isinstance(ab, (int, float)):
                required_tokens.append(f"{ab * 100:.2f}%")

    missing_tokens = [tok for tok in dict.fromkeys(required_tokens) if tok not in body_text]
    if missing_tokens:
        issues.append("증거 수치 누락: " + ", ".join(missing_tokens[:5]))

    if issues:
        return AuditCheck(
            name="G_GUIDE_H7",
            passed=False,
            severity="REJECT",
            detail="; ".join(issues),
        )
    return AuditCheck(
        name="G_GUIDE_H7",
        passed=True,
        severity="REJECT",
        detail=(
            f"T10 통합 가이드 H7 무결성 확인(섹션 30/30, "
            f"고유 ID {len(parser.seen_ids)}개, 중복 0, 수치 일치)"
        ),
    )


def audit_repository_gates(repo_root: str | Path = ".") -> AuditReport:
    """저장소 설계 계약 하드 게이트(설정 동결·누출·봉인 잠금·증거 분모)를 점검한다."""

    root = Path(repo_root)
    checks: list[AuditCheck] = []

    base_toml = root / "configs" / "base.toml"
    expected_base_hash = default_config().config_hash
    if not base_toml.is_file():
        checks.append(
            AuditCheck(
                name="G_CONFIG_FROZEN",
                passed=False,
                severity="REJECT",
                detail=f"동결 설정 파일이 없습니다: {base_toml}",
            )
        )
    else:
        try:
            loaded_base = load_config(base_toml)
            matched = loaded_base.config_hash == expected_base_hash
            checks.append(
                AuditCheck(
                    name="G_CONFIG_FROZEN",
                    passed=matched,
                    severity="REJECT",
                    detail=(
                        f"configs/base.toml 해시 일치({expected_base_hash[:12]})"
                        if matched
                        else (
                            f"configs/base.toml 해시 변조: "
                            f"실제={loaded_base.config_hash[:12]} != 기대={expected_base_hash[:12]}"
                        )
                    ),
                )
            )
        except ConfigError as exc:
            checks.append(
                AuditCheck(
                    name="G_CONFIG_FROZEN",
                    passed=False,
                    severity="REJECT",
                    detail=f"configs/base.toml 로드 실패: {exc}",
                )
            )

    agent_toml = root / "configs" / "agent.toml"
    if not agent_toml.is_file():
        checks.append(
            AuditCheck(
                name="G_AGENT_CONFIG",
                passed=False,
                severity="REJECT",
                detail=f"에이전트 설정 파일이 없습니다: {agent_toml}",
            )
        )
    else:
        try:
            loaded_agent = load_agent_config(agent_toml)
            expected_agent = default_agent_config()
            matched_agent = loaded_agent == expected_agent
            checks.append(
                AuditCheck(
                    name="G_AGENT_CONFIG",
                    passed=matched_agent,
                    severity="REJECT",
                    detail=(
                        "configs/agent.toml 로드 및 기본 설계 계약 일치"
                        if matched_agent
                        else "configs/agent.toml 값이 기본 설계 계약과 다릅니다"
                    ),
                )
            )
        except ConfigError as exc:
            checks.append(
                AuditCheck(
                    name="G_AGENT_CONFIG",
                    passed=False,
                    severity="REJECT",
                    detail=f"configs/agent.toml 로드 실패: {exc}",
                )
            )

    audit_json_path = root / "data" / "audit" / "T03_audit.json"
    if not audit_json_path.is_file():
        checks.append(
            AuditCheck(
                name="G_DATA_LEAKAGE",
                passed=False,
                severity="REVISE",
                detail="data/audit/T03_audit.json 파일이 없습니다",
            )
        )
    else:
        audit_payload, audit_err = _read_json_dict(audit_json_path)
        if audit_err is not None or audit_payload is None:
            checks.append(
                AuditCheck(
                    name="G_DATA_LEAKAGE",
                    passed=False,
                    severity="REJECT",
                    detail=audit_err or "T03_audit.json 파싱 실패",
                )
            )
        else:
            top_ok = audit_payload.get("ok") is True
            top_viol = audit_payload.get("violations") or []
            split_block = audit_payload.get("split") or {}
            split_ok = split_block.get("ok") is True if isinstance(split_block, Mapping) else False
            split_viol = (
                split_block.get("violations") or [] if isinstance(split_block, Mapping) else []
            )
            l_checks = audit_payload.get("l_checks") or {}
            leak_bad: list[str] = []
            for key in ("L1", "L2", "L3", "L4", "L8"):
                entry = l_checks.get(key) if isinstance(l_checks, Mapping) else None
                if isinstance(entry, Mapping) and (
                    entry.get("status") != "PASS" or int(entry.get("value", 0)) != 0
                ):
                    leak_bad.append(f"{key}={entry.get('status')}({entry.get('value')})")
            passed_leak = top_ok and not top_viol and split_ok and not split_viol and not leak_bad
            checks.append(
                AuditCheck(
                    name="G_DATA_LEAKAGE",
                    passed=passed_leak,
                    severity="REJECT",
                    detail=(
                        "T03_audit.json 분할 누출·무결성 위반 0건 확인"
                        if passed_leak
                        else (
                            f"누출/무결성 위반 감지: violations={len(top_viol) + len(split_viol)}, "
                            f"l_checks={leak_bad}"
                        )
                    ),
                )
            )

    splits_dir = root / "data" / "splits"
    lock_errors: list[str] = []
    lock_checked: list[str] = []
    for split_name in _SEALED_SPLIT_NAMES:
        lock_file = splits_dir / f"{split_name}.lock.json"
        if not lock_file.is_file():
            continue
        lock_payload, lock_err = _read_json_dict(lock_file)
        if lock_err is not None or lock_payload is None:
            lock_errors.append(f"{lock_file.name}: {lock_err}")
            continue
        post_change = lock_payload.get("post_seal_config_change")
        access_count = lock_payload.get("access_count")
        lock_checked.append(f"{split_name}(access={access_count})")
        if post_change is not False:
            lock_errors.append(f"{lock_file.name}: post_seal_config_change={post_change!r}")
        if isinstance(access_count, bool) or not isinstance(access_count, int) or access_count > 2:
            lock_errors.append(f"{lock_file.name}: access_count={access_count!r} (> 2)")

    checks.append(
        AuditCheck(
            name="G_SEAL_LOCK",
            passed=not lock_errors,
            severity="REJECT",
            detail=(
                (
                    f"봉인 잠금 불변 확인({', '.join(lock_checked)})"
                    if lock_checked
                    else "봉인 잠금 파일 미생성(봉인 미개봉)"
                )
                if not lock_errors
                else "봉인 잠금 위반: " + "; ".join(lock_errors)
            ),
        )
    )

    evidence_targets = (
        root / "docs" / "evidence" / "T08_sealed_test.json",
        root / "docs" / "evidence" / "P2_triage_eval_2026-11-13.json",
    )
    missing_evidence: list[str] = []
    denom_issues: list[str] = []
    checked_ratios = 0
    for ev_path in evidence_targets:
        if not ev_path.is_file():
            missing_evidence.append(ev_path.name)
            continue
        ev_payload, ev_err = _read_json_dict(ev_path)
        if ev_err is not None or ev_payload is None:
            denom_issues.append(f"{ev_path.name}: {ev_err}")
            continue
        checked_ratios += _scan_ratio_denominators(
            ev_payload, prefix=ev_path.name, issues=denom_issues
        )

    if denom_issues:
        checks.append(
            AuditCheck(
                name="G_EVIDENCE_DENOMINATORS",
                passed=False,
                severity="REJECT",
                detail="증거 JSON 비율 분모 누락: " + "; ".join(denom_issues[:5]),
            )
        )
    elif missing_evidence:
        checks.append(
            AuditCheck(
                name="G_EVIDENCE_DENOMINATORS",
                passed=False,
                severity="REVISE",
                detail="증거 JSON 파일 없음: " + ", ".join(missing_evidence),
            )
        )
    else:
        checks.append(
            AuditCheck(
                name="G_EVIDENCE_DENOMINATORS",
                passed=True,
                severity="REJECT",
                detail=f"주요 증거 JSON 비율 객체 {checked_ratios}건 분모 포함 확인",
            )
        )

    checks.append(_check_t10_guide_h7(root))

    return _build_report(str(root), checks)


__all__ = [
    "AGENT_TRACE_JSONL",
    "AUDIT_SCHEMA_VERSION",
    "AuditCheck",
    "AuditReport",
    "AuditVerdict",
    "audit_records_db",
    "audit_repository_gates",
    "audit_run_dir",
]
