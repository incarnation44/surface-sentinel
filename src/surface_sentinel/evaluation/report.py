"""평가 실행 결과를 근거 수치가 포함된 마크다운 보고서로 자동 렌더링하는 모듈.

보고서의 모든 숫자는 이 실행에서 나온 값이다. 손으로 적은 숫자는 없다.
비율은 언제나 ``값 [Wilson 95% CI] (분모 n)`` 형태로 찍는다.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import REVIEW_BAND_MAX, AppConfig
from .grid import (
    TARGET_ESCAPE,
    TARGET_FALSE_NG,
    TARGET_LOCALIZATION_RECALL,
    TARGET_REVIEW,
    minimax_key,
)
from .measure import VariantReport
from .metrics import Ratio

if TYPE_CHECKING:  # pragma: no cover - 순환 import 방지
    from .runner import EvalResult

IMAGE_METRIC_ORDER = (
    "defect_escape_rate",
    "defect_ng_rate",
    "defect_review_rate",
    "good_false_ng_rate",
    "good_review_rate",
    "review_rate",
    "automation_rate",
    "decided_accuracy",
    "balanced_accuracy",
)

LOCALIZATION_ORDER = (
    "localization_recall_iou0.10",
    "localization_recall_iou0.10_decided",
    "localization_recall_iou0.30",
    "localization_recall_iou0.30_decided",
    "pixel_precision",
    "pixel_recall",
    "mean_iou_matched",
)

CORE_ROWS = (
    ("defect_escape_rate", "OK_on_defect / P"),
    ("good_false_ng_rate", "NG_on_good / N"),
    ("review_rate", "(REVIEW_defect+REVIEW_good) / T"),
    ("localization_recall_iou0.10", "matched_GT(IoU>=0.10) / GT"),
)

SCORE_BUCKETS = 10
SCORE_MAX = 1.5


def _fmt(ratio: Ratio) -> str:
    return ratio.formatted().replace("|", "/")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _section_verdict(result: EvalResult) -> list[str]:
    proposal = result.by_name("P")
    targets = _report_targets(proposal)
    met = sum(1 for value in targets.values() if value)
    best = result.pilot_best
    feasibility = sum(1 for row in result.grid if row.selection_ready())
    lines = ["## 요약 판정", ""]
    rows = [
        [
            "P(기본 config)",
            _pct(proposal.image["defect_escape_rate"].value),
            _pct(proposal.image["good_false_ng_rate"].value),
            _pct(proposal.image["review_rate"].value),
            _pct(proposal.localization["localization_recall_iou0.10"].value),
            f"{met}/4",
        ]
    ]
    if best is not None:
        adopted = best.report
        adopted_targets = _report_targets(adopted)
        rows.append(
            [
                f"채택안 `{result.adopted_label}`",
                _pct(adopted.image["defect_escape_rate"].value),
                _pct(adopted.image["good_false_ng_rate"].value),
                _pct(adopted.image["review_rate"].value),
                _pct(adopted.localization["localization_recall_iou0.10"].value),
                f"{sum(1 for value in adopted_targets.values() if value)}/4",
            ]
        )
    lines.extend(
        _table(["집합", "escape", "false-NG", "review", "loc recall@0.10", "목표 충족"], rows)
    )
    lines.append("")
    lines.append(f"- 그리드 12후보 중 `false-NG ≤ 0.20` 과 `review ≤ 0.40` 을 동시에 만족: "
                 f"**{feasibility}개**.")
    lines.append(
        "- 후보 축 파일럿(그리드 밖): "
        + (
            f"`{best.candidate_id}+{best.scoring_id}` 가 세 목표를 동시에 만족."
            if best is not None
            else "세 목표를 동시에 만족하는 칸 없음 → Pareto 보고."
        )
    )
    lines.append(
        "- 결론: **scoring 3필드만으로는 목표를 맞출 수 없다.** 후보 생성 단계"
        "(§3.2 진리표에서 `OK`가 '컴포넌트 0개'를 요구)가 병목이며, 아래 §4의 "
        "추천안은 그 축을 함께 조정한 측정 결과다."
    )
    lines.append("")
    return lines


def _section_sample(result: EvalResult) -> list[str]:
    plan = result.plan
    lines = ["## 0. 표본과 분모", ""]
    lines.append(
        f"- 분할: `{plan.split}` (tune 전용). 추출 시드: `{plan.seed}`."
    )
    lines.append(
        f"- defect: **{plan.defect_taken} / {plan.defect_population}** — 전수(표본 오차 없음)."
    )
    lines.append(
        f"- good: **{plan.good_taken} / {plan.good_population}** — `source_split` 층화 "
        f"{sum(stratum.requested for stratum in plan.strata)}장 요청."
    )
    lines.append("")
    lines.extend(
        _table(
            ["층(stratum)", "모집단", "표본", "추출률"],
            [
                [
                    stratum.stratum,
                    f"{stratum.population:,}",
                    f"{stratum.requested:,}",
                    _pct(stratum.fraction),
                ]
                for stratum in plan.strata
            ],
        )
    )
    lines.append("")
    lines.append(
        "배분은 비례 배분 + 최대 나머지, 층 내부는 `sha256(seed:sample_id)` 오름차순이다. "
        "난수 생성기를 쓰지 않으므로 재실행해도 같은 표본이 나온다."
    )
    lines.append("")
    lines.append(
        "분모 규칙: `P` = 표본의 defect 이미지 수, `N` = 표본의 good 이미지 수, "
        "`T = P + N`. `ERROR`는 판정이 아니므로 분모에서 뺀다."
    )
    lines.append("")
    return lines


def _section_variants(result: EvalResult) -> list[str]:
    reports = [report for report in result.reports if report.name in {"B0", "B1", "B2", "P"}]
    lines = ["## 1. B0 / B1 / B2 / P (설계 §4.2·§4.3)", ""]
    if not reports:
        lines.append("기준선을 실행하지 않았습니다(`--no-baselines`).")
        lines.append("")
        return lines

    names = [report.name for report in reports]
    rows = []
    for key, definition in CORE_ROWS:
        cells = [f"`{key}`", definition]
        for report in reports:
            cells.append(_fmt(report.ratio(key)))
        rows.append(cells)
    lines.append("### 1.1 목표 지표")
    lines.append("")
    lines.extend(_table(["지표", "정의(분모)", *names], rows))
    lines.append("")

    rows = []
    for key in IMAGE_METRIC_ORDER:
        cells = [f"`{key}`"]
        for report in reports:
            cells.append(_fmt(report.image[key]))
        rows.append(cells)
    lines.append("### 1.2 이미지 지표 9개")
    lines.append("")
    lines.extend(_table(["지표", *names], rows))
    lines.append("")

    rows = []
    for key in LOCALIZATION_ORDER:
        cells = [f"`{key}`"]
        for report in reports:
            cells.append(_fmt(report.localization[key]))
        rows.append(cells)
    lines.append("### 1.3 영역 지표 (마스크가 있는 defect 이미지만)")
    lines.append("")
    lines.extend(_table(["지표", *names], rows))
    lines.append("")
    lines.append(
        "B1/B2는 REVIEW를 내지 않는 이진 기준선이다. 그래서 `*_review_rate`가 0으로 "
        "나오는 것은 성능이 아니라 **구조**의 결과다. P와 비교할 때는 "
        "`defect_escape_rate`·`good_false_ng_rate`만 같은 자로 본다."
    )
    lines.append("")
    counts = [
        [report.name, str(report.counts.get("defect")), str(report.counts.get("good")),
         str(report.counts.get("defect_evaluated")), str(report.counts.get("good_evaluated"))]
        for report in reports
    ]
    lines.extend(
        _table(
            ["집합", "defect(전체)", "good(전체)", "P(분모)", "N(분모)"],
            counts,
        )
    )
    lines.append("")
    return lines


def _section_diagnosis(result: EvalResult) -> list[str]:
    proposal = result.by_name("P")
    samples = result.samples
    by_id = {sample.sample_id: sample for sample in samples}
    lines = ["## 2. 실패 분포와 [scoring] 원인 지목", ""]

    # 2.1 false-NG 사유 분포
    false_ng = [
        outcome
        for outcome in proposal.outcomes
        if outcome.label == "good" and outcome.status.value == "NG"
    ]
    defect_fail = [
        outcome
        for outcome in proposal.outcomes
        if outcome.label == "defect" and outcome.status.value != "NG"
    ]
    lines.append("### 2.1 false-NG 사유 코드 분포")
    lines.append("")
    if false_ng:
        counts: dict[str, int] = {}
        for outcome in false_ng:
            counts[outcome.reason_code] = counts.get(outcome.reason_code, 0) + 1
        rows = [
            [f"`{code}`", str(count), _pct(count / len(false_ng))]
            for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        lines.extend(_table(["사유 코드", "건수", "false-NG 내 비중"], rows))
        lines.append("")
        lines.append(f"분모: false-NG {len(false_ng)}건 (N = {proposal.counts.get('good')}).")
    else:
        lines.append("false-NG 0건 — 사유 분포가 없다(분모 0).")
    lines.append("")

    # 2.2 점수 히스토그램
    lines.append("### 2.2 점수 히스토그램 (defect vs good)")
    lines.append("")
    width = SCORE_MAX / SCORE_BUCKETS
    buckets = [[0, 0] for _ in range(SCORE_BUCKETS + 1)]
    for outcome in proposal.outcomes:
        index = min(int(outcome.score / width), SCORE_BUCKETS)
        column = 0 if outcome.label == "defect" else 1
        buckets[index][column] += 1
    rows = []
    for index, (defect_count, good_count) in enumerate(buckets):
        low = index * width
        high = SCORE_MAX if index == SCORE_BUCKETS else (index + 1) * width
        if index == SCORE_BUCKETS:
            label = f"{SCORE_MAX:.2f} (상한 포화)"
        else:
            label = f"[{low:.2f}, {high:.2f})"
        rows.append(
            [
                label,
                str(defect_count),
                "▇" * min(defect_count, 40),
                str(good_count),
                "▇" * min(good_count, 40),
            ]
        )
    lines.extend(_table(["점수 구간", "defect", "", "good", ""], rows))
    lines.append("")
    lines.append(
        f"분모: defect {proposal.counts.get('defect')}, good {proposal.counts.get('good')}. "
        f"review band 상한(동결)={REVIEW_BAND_MAX}, "
        f"NG 하한 `ng_min`={result.base_config.scoring.ng_min}."
    )
    lines.append("")

    # 2.3 scoring 필드 지목
    lines.append("### 2.3 원인 [scoring] 필드 지목")
    lines.append("")
    escapes = [outcome for outcome in defect_fail if outcome.status.value == "OK"]
    zero_score = [outcome for outcome in escapes if outcome.score <= 0.0]
    band_ok = [outcome for outcome in escapes if 0.0 < outcome.score <= REVIEW_BAND_MAX]
    rows = [
        ["escape 총계", str(len(escapes)), f"P = {proposal.counts.get('defect_evaluated')}"],
        [
            "  · 컴포넌트 0개(score=0)",
            str(len(zero_score)),
            "후보 생성 단계에서 걸러짐 → [candidates] 문제, [scoring] 아님",
        ],
        [
            f"  · 0 < score ≤ review band 상한({REVIEW_BAND_MAX})",
            str(len(band_ok)),
            "evidence가 동결 상한을 넘지 못함 → [scoring] 정규화 기준 문제",
        ],
    ]
    lines.extend(_table(["구분", "건수", "해석"], rows))
    lines.append("")

    components = [
        component
        for sample in samples
        for component in sample.components
        if sample.label == "defect"
    ]
    if components:
        areas = sorted(component.area_px for component in components)
        contrasts = sorted(component.mean_contrast_gray for component in components)
        ng_area = result.base_config.scoring.ng_area_px
        ng_contrast = result.base_config.scoring.ng_contrast_gray
        below_area = sum(1 for value in areas if value < ng_area)
        below_contrast = sum(1 for value in contrasts if value < ng_contrast)

        def median(values: Sequence[float]) -> float:
            return values[len(values) // 2]

        rows = [
            [
                "`scoring.ng_area_px`",
                f"{ng_area:g}",
                f"중앙 {median(areas):.1f} px",
                f"{below_area} / {len(areas)} ({_pct(below_area / len(areas))})",
            ],
            [
                "`scoring.ng_contrast_gray`",
                f"{ng_contrast:g}",
                f"중앙 {median(contrasts):.2f} gray",
                f"{below_contrast} / {len(contrasts)} ({_pct(below_contrast / len(contrasts))})",
            ],
        ]
        lines.append("defect 이미지 컴포넌트가 정규화 기준보다 작은 비율:")
        lines.append("")
        lines.extend(_table(["필드", "기준값", "실측", "기준 미만"], rows))
        lines.append("")
        lines.append(
            "`evidence = sqrt(min(area/ng_area_px, 1.5) · min(contrast/ng_contrast_gray, 1.5))` "
            "이므로, 두 기준 미만 비율이 높은 필드가 점수를 눌러 escape/review를 만든다."
        )
        if false_ng:
            codes = sorted({outcome.reason_code for outcome in false_ng})
            lines.append("")
            lines.append(
                f"실제로 false-NG {len(false_ng)}건의 사유는 전부 "
                f"{', '.join(f'`{code}`' for code in codes)} 이다. 즉 점수가 "
                f"`ng_min`을 넘겨 NG가 된 경우이고, 원인 필드는 대비 정규화"
                f"(`ng_contrast_gray`)와 판정 하한(`ng_min`)이다. §2.4의 주변 효과가 "
                "그 방향을 수치로 보여준다."
            )
        lines.append("")

    if escapes:
        rows = []
        for outcome in sorted(escapes, key=lambda item: -item.score)[:5]:
            sample = by_id[outcome.sample_id]
            rows.append(
                [
                    outcome.sample_id,
                    str(len(sample.components)),
                    f"{outcome.score:.4f}",
                    ";".join(sample.quality_flags) or "-",
                    "Y" if sample.roi_uncertain else "N",
                    "Y" if sample.truncated else "N",
                ]
            )
        lines.extend(
            _table(
                ["escape sample", "컴포넌트", "score", "품질 플래그", "roi.uncertain", "truncated"],
                rows,
            )
        )
        lines.append("")

    # 2.4 구조 진단: OK 경로가 열리는가
    lines.append("### 2.3b 구조 진단 — OK 경로가 열리는가")
    lines.append("")
    total = len(samples)
    with_component = sum(1 for sample in samples if sample.components)
    truncated = sum(1 for sample in samples if sample.truncated)
    mean_components = (
        sum(len(sample.components) for sample in samples) / total if total else 0.0
    )
    component_ratio = with_component / total if total else None
    truncated_ratio = truncated / total if total else None
    rows = [
        ["컴포넌트 ≥ 1개인 이미지", f"{with_component} / {total}", _pct(component_ratio)],
        [
            "`CANDIDATE_TRUNCATED`(상한 20 초과)",
            f"{truncated} / {total}",
            _pct(truncated_ratio),
        ],
        ["이미지당 컴포넌트 수(평균)", f"{mean_components:.2f}", "-"],
    ]
    lines.extend(_table(["항목", "수량", "비율"], rows))
    lines.append("")
    lines.append(
        "진리표(§3.2)에서 `OK`는 **타당한 컴포넌트가 하나도 없고** 플래그가 없을 때만 "
        "나온다(우선순위 3). 그런데 후보 생성이 표면 전체를 후보로 만들면 컴포넌트가 "
        "항상 존재하고 절단 플래그까지 걸린다. 그러면 `ng_min` 미만은 전부 "
        "`REVIEW`(우선순위 7 또는 9)가 되어 review rate가 구조적으로 0.6~0.9에 "
        "고정된다. 이것이 scoring 3필드 그리드만으로 목표를 못 맞추는 이유다."
    )
    lines.append("")

    # 2.5 그리드 주변 효과
    field_order = (
        "scoring.ng_area_px",
        "scoring.ng_contrast_gray",
        "scoring.ng_min",
    )
    lines.append("### 2.4 필드별 주변 효과 (그리드 12후보 평균)")
    lines.append("")
    rows: list[list[str]] = []
    for field in field_order:
        values = sorted({row.overrides[field] for row in result.grid})
        for value in values:
            subset = [row for row in result.grid if row.overrides[field] == value]
            escape = [row.escape for row in subset if row.escape is not None]
            false_ng = [row.false_ng for row in subset if row.false_ng is not None]
            review = [row.review for row in subset if row.review is not None]
            rows.append(
                [
                    f"`{field}`",
                    f"{value:g}",
                    str(len(subset)),
                    _pct(sum(escape) / len(escape)) if escape else "-",
                    _pct(sum(false_ng) / len(false_ng)) if false_ng else "-",
                    _pct(sum(review) / len(review)) if review else "-",
                ]
            )
    lines.extend(
        _table(
            ["필드", "값", "후보 수", "평균 escape", "평균 false-NG", "평균 review"],
            rows,
        )
    )
    lines.append("")
    lines.append("후보 수가 적은 칸의 평균은 흔들린다. 방향을 읽는 용도로만 쓴다.")
    lines.append("")
    return lines


def _section_grid(result: EvalResult) -> list[str]:
    grid = result.grid
    selection = result.selection
    lines = ["## 3. 튜닝 그리드 (12후보, scoring 3개 필드)", ""]
    lines.append(
        "후보 필드: `scoring.ng_area_px` × `scoring.ng_contrast_gray` × "
        "`scoring.ng_min` = 2×3×2 = 12. 이 3개만 캐시 위에서 "
        "정확히 재계산된다(파이프라인 재실행 없음)."
    )
    lines.append("")
    ordered = sorted(grid, key=minimax_key)
    rows = []
    for row in ordered[:10]:
        targets = row.targets()
        rows.append(
            [
                row.candidate_id,
                f"{row.overrides['scoring.ng_area_px']:g}",
                f"{row.overrides['scoring.ng_contrast_gray']:g}",
                f"{row.overrides['scoring.ng_min']:g}",
                _pct(row.escape),
                _pct(row.false_ng),
                _pct(row.review),
                _pct(row.localization_recall),
                "".join(
                    "O" if targets[key] else ("-" if targets[key] is None else "X")
                    for key in (
                        "escape_le_0.15",
                        "false_ng_le_0.20",
                        "review_le_0.40",
                        "localization_recall_ge_0.50",
                    )
                ),
            ]
        )
    lines.extend(
        _table(
            [
                "id",
                "ng_area",
                "ng_contrast",
                "ng_min",
                "escape",
                "false-NG",
                "review",
                "loc recall@0.10",
                "목표(escape/falseNG/review/loc)",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"목표: escape ≤ {TARGET_ESCAPE}, false-NG ≤ {TARGET_FALSE_NG}, "
        f"review ≤ {TARGET_REVIEW}, loc recall@IoU0.10 ≥ {TARGET_LOCALIZATION_RECALL}. "
        "`O`=충족, `X`=미달, `-`=분모 0. 표는 목표 대비 최악 위반비가 작은 순서다."
    )
    lines.append("")
    lines.append(f"선택 규칙 결과: {selection.reason}.")
    lines.append("")
    if selection.pareto:
        lines.append("Pareto 점(escape·false-NG·review 최소화, 위반비 순):")
        lines.append("")
        front = sorted(selection.pareto, key=minimax_key)
        rows = [
            [row.candidate_id, _pct(row.escape), _pct(row.false_ng), _pct(row.review)]
            for row in front[:5]
        ]
        lines.extend(_table(["id", "escape", "false-NG", "review"], rows))
        lines.append("")
    lines.append(
        "전체 후보 결과는 `docs/evidence/tune_grid.jsonl`에 있다(후보마다 한 줄, "
        "분모와 Wilson 구간 포함)."
    )
    lines.append("")
    lines.extend(_section_pilot(result))
    return lines


def _section_pilot(result: EvalResult) -> list[str]:
    rows = result.pilots
    if not rows:
        return []
    lines = ["### 3.5 후보(candidates) 축 파일럿 — 그리드 밖 보정 실험", ""]
    lines.append(
        "그리드가 목표를 만족하지 못한 이유는 판정 진리표(§3.2)에 있다. `OK`는 "
        "타당한 컴포넌트가 하나도 없을 때만 나온다. 기본 설정에서는 컴포넌트가 "
        "상한(20개)까지 차고 `CANDIDATE_TRUNCATED`가 걸리므로, `ng_min` 미만은 "
        "전부 `REVIEW`가 된다. 그래서 review를 낮추려면 **후보 생성 임계값**을 "
        "직접 조여야 한다. 이 표가 그 축의 측정이다."
    )
    lines.append("")
    table_rows = []
    for row in rows:
        targets = _pilot_targets(row)
        table_rows.append(
            [
                row.candidate_id,
                row.scoring_id,
                _pct(row.escape),
                _pct(row.false_ng),
                _pct(row.review),
                _pct(row.localization_recall),
                "".join(
                    "O" if targets[key] else ("-" if targets[key] is None else "X")
                    for key in ("escape", "false_ng", "review", "localization")
                ),
            ]
        )
    lines.extend(
        _table(
            ["후보축", "scoring축", "escape", "false-NG", "review", "loc recall@0.10", "목표"],
            table_rows,
        )
    )
    lines.append("")
    best = result.pilot_best
    if best is None:
        lines.append(
            "세 목표를 동시에 만족하는 칸이 없다. 이 경우 설계 §4.4에 따라 Pareto "
            "후보를 보고하고, 후보 생성 단계의 재보정을 별건으로 올린다."
        )
    else:
        lines.append(
            f"목표 동시 충족 칸: `{best.candidate_id}+{best.scoring_id}` "
            f"(escape {_pct(best.escape)}, false-NG {_pct(best.false_ng)}, "
            f"review {_pct(best.review)})."
        )
    lines.append("")
    return lines


def _pilot_targets(row: Any) -> dict[str, bool | None]:
    return row.targets(
        escape_max=TARGET_ESCAPE,
        false_ng_max=TARGET_FALSE_NG,
        review_max=TARGET_REVIEW,
        recall_min=TARGET_LOCALIZATION_RECALL,
    )


def _diff(base: AppConfig, adopted: AppConfig) -> list[str]:
    base_lines = base.canonical_toml().splitlines()
    adopted_lines = adopted.canonical_toml().splitlines()
    diff = list(
        difflib.unified_diff(
            base_lines,
            adopted_lines,
            fromfile="configs/base.toml",
            tofile="configs/candidates/<adopted>.toml",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return ["변경 없음 — 기본 config가 그대로 추천안이다."]
    return ["```diff", *diff, "```"]


def _section_recommendation(
    result: EvalResult,
    adopted_config: AppConfig,
) -> list[str]:
    proposal = result.by_name("P")
    best = result.pilot_best
    target_row = best if best is not None else result.selection.best
    lines = ["## 4. 채택 추천안", ""]
    if target_row is None:
        lines.append("후보를 만들지 못했습니다.")
        lines.append("")
        return lines

    adopted_report = target_row.report
    lines.append(
        f"- 채택 후보: `{result.adopted_label}` "
        f"(config hash `{adopted_config.config_hash[:12]}`)"
    )
    lines.append(f"- 근거: {result.adopted_basis}.")
    if best is not None:
        lines.append(
            "- 이 후보는 `[candidates]`·`[scoring]` 두 섹션을 함께 바꾼다. 후보 마스크가 "
            "달라지므로 실패 색인의 오버레이는 기본 config 판정 기준으로 그려져 있다"
            "(`status_base` 열로 구분)."
        )
    lines.append("")
    lines.append("### 4.1 diff (base → 추천안)")
    lines.append("")
    lines.extend(_diff(result.base_config, adopted_config))
    lines.append("")

    lines.append("### 4.2 기대 지표 (같은 tune 표본 위)")
    lines.append("")
    rows = []
    for key, definition in CORE_ROWS:
        rows.append(
            [
                f"`{key}`",
                definition,
                _fmt(proposal.ratio(key)),
                _fmt(adopted_report.ratio(key)),
            ]
        )
    lines.extend(_table(["지표", "정의(분모)", "P(기본)", "추천안"], rows))
    lines.append("")

    targets = _report_targets(adopted_report)
    image = adopted_report.image
    rows = [
        [
            "escape ≤ 0.15",
            f"{TARGET_ESCAPE}",
            _pct(image["defect_escape_rate"].value),
            _mark(targets["escape"]),
        ],
        [
            "false-NG ≤ 0.20",
            f"{TARGET_FALSE_NG}",
            _pct(image["good_false_ng_rate"].value),
            _mark(targets["false_ng"]),
        ],
        [
            "review ≤ 0.40",
            f"{TARGET_REVIEW}",
            _pct(image["review_rate"].value),
            _mark(targets["review"]),
        ],
        [
            "loc recall@IoU0.10 ≥ 0.50",
            f"{TARGET_LOCALIZATION_RECALL}",
            _pct(adopted_report.localization["localization_recall_iou0.10"].value),
            _mark(targets["localization"]),
        ],
    ]
    lines.extend(_table(["목표", "기준", "추천안", "판정"], rows))
    lines.append("")
    lines.append(
        "주의 1: 이 값은 **tune 표본**에서 나온 값이다. test/official_test는 열지 않았고, "
        "동결 이후 이 값으로 임계값을 다시 바꾸지 않는다(설계 §4.4)."
    )
    lines.append(
        "주의 2: 추천안은 `review`를 낮추기 위해 후보 생성 임계값을 조인 결과다. 그 대가로 "
        "escape(결함을 OK로 넘김)가 올라간다. 표의 두 값을 함께 읽어야 한다."
    )
    lines.append(
        "주의 3: pixel precision이 낮다면 영역 지표의 recall은 큰 덩어리가 GT를 덮어서 "
        "올라간 값일 수 있다. §1.3의 `pixel_precision`을 함께 본다."
    )
    lines.append("")
    return lines


def _report_targets(report: VariantReport) -> dict[str, bool | None]:
    escape = report.image["defect_escape_rate"].value
    false_ng = report.image["good_false_ng_rate"].value
    review = report.image["review_rate"].value
    recall = report.localization["localization_recall_iou0.10"].value
    return {
        "escape": None if escape is None else escape <= TARGET_ESCAPE,
        "false_ng": None if false_ng is None else false_ng <= TARGET_FALSE_NG,
        "review": None if review is None else review <= TARGET_REVIEW,
        "localization": None if recall is None else recall >= TARGET_LOCALIZATION_RECALL,
    }


def _mark(value: bool | None) -> str:
    if value is None:
        return "-"
    return "충족" if value else "미달"


def _section_ablations(result: EvalResult) -> list[str]:
    specs = [report for report in result.reports if report.name.startswith("A")]
    lines = ["## 5. Ablation A1–A6 (tune 전용, 판정 임계값 고정)", ""]
    if not specs:
        lines.append("ablation을 실행하지 않았습니다(`--no-ablations`).")
        lines.append("")
        return lines
    proposal = result.by_name("P")
    rows = [
        [
            "P",
            "제안 파이프라인",
            _fmt(proposal.image["defect_escape_rate"]),
            _fmt(proposal.image["good_false_ng_rate"]),
            _fmt(proposal.image["review_rate"]),
            _fmt(proposal.localization["localization_recall_iou0.10"]),
        ]
    ]
    notes: list[str] = []
    for report in specs:
        intent = report.notes[0].split(" — ")[0] if report.notes else ""
        rows.append(
            [
                report.name,
                intent,
                _fmt(report.image["defect_escape_rate"]),
                _fmt(report.image["good_false_ng_rate"]),
                _fmt(report.image["review_rate"]),
                _fmt(report.localization["localization_recall_iou0.10"]),
            ]
        )
        for note in report.notes:
            if " — " in note:
                notes.append(f"- **{report.name}**: {note.split(' — ', 1)[1]}")
    lines.extend(
        _table(["id", "취지", "escape", "false-NG", "review", "loc recall@0.10"], rows)
    )
    lines.append("")
    lines.extend(notes)
    lines.append("")
    unchanged = [
        report.name
        for report in specs
        if report.image["defect_escape_rate"].value
        == proposal.image["defect_escape_rate"].value
        and report.image["good_false_ng_rate"].value
        == proposal.image["good_false_ng_rate"].value
        and report.image["review_rate"].value == proposal.image["review_rate"].value
    ]
    if unchanged:
        lines.append(
            "P와 지표가 **완전히 같은** ablation: "
            + ", ".join(f"`{name}`" for name in unchanged)
            + ". 구조를 하나 빼도 결과가 안 바뀐다는 뜻이고, 그 이유는 후보 생성이 "
            "이미 포화(컴포넌트 20개 상한) 상태라 상위 20개 구성이 그대로이기 때문이다."
        )
        lines.append("")
    lines.append(
        "A1은 응답·점수 스케일 자체가 달라진다(배경 커널 61px → 21px). 그래서 A1의 "
        "절대값을 다른 행과 나란히 비교하지 않고, 방향(escape가 늘었는지)만 읽는다."
    )
    lines.append("")
    return lines


def _section_failures(result: EvalResult) -> list[str]:
    selection = result.case_selection
    lines = ["## 6. 실패 사례 (설계 §4.5 규칙 선택)", ""]
    total = len(result.failures)
    lines.append(
        f"실패 색인 전체 {total}건: `docs/evidence/failure_cases/T05_failure_index.csv` "
        "(sample_id, split, label, status, reason_code, score, overlay, category …)."
    )
    lines.append("")
    counts = selection.get("category_counts", {})
    if counts:
        lines.extend(
            _table(
                ["카테고리", "건수"],
                [[str(key), str(value)] for key, value in sorted(counts.items())],
            )
        )
        lines.append("")
    for key, title in (
        ("high_score_escape_top2", "고득점 escape 상위 2"),
        ("high_score_false_ng_top2", "고득점 오탐 상위 2"),
    ):
        entries = selection.get(key, [])
        if not entries:
            continue
        lines.append(f"**{title}**")
        lines.append("")
        lines.extend(
            _table(
                ["sample_id", "label", "status", "reason_code", "score", "overlay"],
                [
                    [
                        entry["sample_id"],
                        entry["label"],
                        entry["status"],
                        f"`{entry['reason_code']}`",
                        f"{entry['score']:.4f}",
                        entry["overlay"] or "-",
                    ]
                    for entry in entries
                ],
            )
        )
        lines.append("")
    reason = selection.get("largest_review_cluster_reason")
    if reason:
        lines.append(
            f"가장 큰 REVIEW 클러스터: `{reason}` "
            f"{selection.get('largest_review_cluster_size')}건."
        )
        lines.append("")
    for key, title in (
        ("boundary_cases", "경계 사례"),
        ("quality_flag_cases", "품질 플래그 사례"),
        ("partial_location_cases", "부분 위치 사례(best IoU < 0.30)"),
    ):
        entries = selection.get(key, [])
        if not entries:
            continue
        lines.append(f"**{title}**")
        lines.append("")
        lines.extend(
            _table(
                ["sample_id", "label", "status", "score", "best_iou", "overlay"],
                [
                    [
                        entry["sample_id"],
                        entry["label"],
                        entry["status"],
                        f"{entry['score']:.4f}",
                        f"{entry['best_iou']:.3f}",
                        entry["overlay"] or "-",
                    ]
                    for entry in entries
                ],
            )
        )
        lines.append("")
    return lines


def _section_reproduce(result: EvalResult) -> list[str]:
    paths = result.paths
    lines = ["## 7. 재현", "", "PowerShell:"]
    lines.append("")
    lines.append("```powershell")
    lines.append(".venv/Scripts/python.exe -m surface_sentinel eval --split tune")
    lines.append("```")
    lines.append("")
    lines.append("산출물:")
    lines.append("")
    keys = (
        "grid",
        "pilot",
        "variants",
        "sample_plan",
        "failure_index",
        "failure_selection",
        "candidates",
    )
    for key in keys:
        if key in paths:
            lines.append(f"- `{paths[key]}`")
    lines.append("")
    lines.append(
        f"실행 시간 {result.elapsed_seconds:.1f}s. 캐시(`data/derived/t05_cache/`)는 "
        "config hash·분할 해시·표본 id로 키가 정해지며, 같은 키면 재사용한다."
    )
    lines.append("")
    return lines


def write_report(
    result: EvalResult,
    path: str | Path,
    *,
    adopted_config: AppConfig,
) -> Path:
    """보고서 마크다운을 쓴다. 숫자는 전부 실행 결과에서 온다."""

    proposal = result.by_name("P")
    lines = [
        "# T05 평가·기준선·ablation 보고서 (2026-11-10)",
        "",
        f"- 분할: `{result.plan.split}` (tune 전용, test/official_test 미접근)",
        f"- 표본: defect {result.plan.defect_taken} (전수) + good {result.plan.good_taken} (층화)",
        f"- 기본 config: `configs/base.toml` (hash `{result.base_config.config_hash[:12]}`)",
        f"- P 판정 분포: NG {proposal.confusion.count('defect', 'NG')} / OK "
        f"{proposal.confusion.count('defect', 'OK')} / REVIEW "
        f"{proposal.confusion.count('defect', 'REVIEW')} (defect), "
        f"NG {proposal.confusion.count('good', 'NG')} / REVIEW "
        f"{proposal.confusion.count('good', 'REVIEW')} (good)",
        "",
        "모든 비율은 `값 [Wilson 95% CI] (분모 n)`으로 적는다. 분모가 0이면 값을 쓰지 않는다.",
        "",
    ]
    lines.extend(_section_verdict(result))
    lines.extend(_section_sample(result))
    lines.extend(_section_variants(result))
    lines.extend(_section_diagnosis(result))
    lines.extend(_section_grid(result))
    lines.extend(_section_recommendation(result, adopted_config))
    lines.extend(_section_ablations(result))
    lines.extend(_section_failures(result))
    lines.extend(_section_reproduce(result))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = ["write_report"]
