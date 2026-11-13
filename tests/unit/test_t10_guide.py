"""T10 공식 30섹션 통합 HTML 가이드 및 H7 증거-문서 수치 일치 검증."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = REPO_ROOT / "docs" / "surface-sentinel-integrated-guide.html"
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence"


class _GuideParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.duplicate_ids: list[str] = []
        self.seen_ids: set[str] = set()
        self.internal_hrefs: list[str] = []
        self.external_resource_urls: list[str] = []
        self.style_blocks: list[str] = []
        self._text_parts: list[str] = []
        self._in_style = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in {"script", "style"}:
            self._skip_depth += 1
        if lower == "style":
            self._in_style = True

        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if "id" in attr_map and attr_map["id"]:
            ident = attr_map["id"]
            self.ids.append(ident)
            if ident in self.seen_ids:
                self.duplicate_ids.append(ident)
            self.seen_ids.add(ident)

        if lower in {"script", "img"}:
            src = attr_map.get("src", "")
            if re.match(r"^https?://", src, re.IGNORECASE):
                self.external_resource_urls.append(f"{lower}[src]={src}")
        elif lower == "link":
            href = attr_map.get("href", "")
            if re.match(r"^https?://", href, re.IGNORECASE):
                self.external_resource_urls.append(f"link[href]={href}")
        elif lower == "a":
            href = attr_map.get("href", "")
            if href.startswith("#"):
                self.internal_hrefs.append(href)

        style_attr = attr_map.get("style", "")
        if style_attr:
            self.style_blocks.append(style_attr)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if lower == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.style_blocks.append(data)
        if self._skip_depth == 0 and data:
            self._text_parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._text_parts)


def test_t10_guide_structure_size_and_links() -> None:
    assert GUIDE_PATH.is_file(), f"T10 가이드 파일이 없습니다: {GUIDE_PATH}"
    raw = GUIDE_PATH.read_bytes()
    assert len(raw) < 2_000_000, f"단일 파일 2MB 한도 초과: {len(raw)} bytes"
    assert len(raw) <= 300 * 1024, f"권장 300KB 상한 초과: {len(raw)} bytes"

    html_text = raw.decode("utf-8")
    parser = _GuideParser()
    parser.feed(html_text)
    parser.close()

    # 30개 필수 섹션(sec-01 ~ sec-30) 존재
    missing_sections = [
        f"sec-{idx:02d}"
        for idx in range(1, 31)
        if not any(
            ident == f"sec-{idx:02d}" or ident.startswith(f"sec-{idx:02d}-")
            for ident in parser.seen_ids
        )
    ]
    assert missing_sections == [], f"누락된 섹션 ID: {missing_sections}"

    # 중복 id 0건
    assert parser.duplicate_ids == [], f"중복 id 발견: {parser.duplicate_ids}"

    # 깨진 내부 앵커 링크 0건
    broken_links = sorted(
        {
            href
            for href in parser.internal_hrefs
            if len(href) > 1 and href[1:] not in parser.seen_ids
        }
    )
    assert broken_links == [], f"깨진 내부 앵커 링크: {broken_links}"

    # 외부 URL 리소스 0건 (<script src>, <link href>, <img src>, @import, url())
    assert parser.external_resource_urls == []
    css_blob = "\n".join(parser.style_blocks)
    assert not re.search(r"@import\s+(?:url\()?['\"]?https?://", css_blob, re.IGNORECASE)
    assert not re.search(r"url\(\s*['\"]?https?://", css_blob, re.IGNORECASE)

    # 4종 검증 태그 존재
    body_text = parser.text
    for tag in ("[FACT]", "[DESIGN]", "[MEASURED]", "[NOT_MEASURED]"):
        assert tag in body_text, f"필수 태그 누락: {tag}"


def test_t10_guide_h7_numbers_match_evidence() -> None:
    html_text = GUIDE_PATH.read_text(encoding="utf-8")
    parser = _GuideParser()
    parser.feed(html_text)
    parser.close()
    body_text = parser.text

    t08 = json.loads((EVIDENCE_DIR / "T08_sealed_test.json").read_text(encoding="utf-8"))
    p2 = json.loads((EVIDENCE_DIR / "P2_triage_eval_2026-11-13.json").read_text(encoding="utf-8"))
    p3 = json.loads((EVIDENCE_DIR / "P3_watchdog_eval_2026-11-13.json").read_text(encoding="utf-8"))
    p4 = json.loads(
        (EVIDENCE_DIR / "P4_counterfactual_eval_2026-11-13.json").read_text(encoding="utf-8")
    )

    p_report = t08["variants"]["P"]["report"]
    p_img = p_report["image_metrics"]
    p_loc = p_report["localization_metrics"]
    p2_tune = p2["tune_review"]
    p2_top = p2_tune["triage_lift"]["top_k"]
    p3_norm = p3["drift_evaluation"]["normal_holdout_400"]["features"]
    p3_synth = p3["drift_evaluation"]["synthetic_drift_score_and_brightness"]["features"]
    p4_bdown = p4["review_bottleneck_breakdown"]
    p4_spot = p4["spatial_hotspot_tune"]

    expected_tokens = [
        "3,335",
        "2,001",
        "1,334",
        "11.68%",
        t08["config_hash"][:8],
        f"{t08['goals']['escape 점추정']['value'] * 100:.2f}%",
        f"{t08['goals']['escape Wilson 상한']['value'] * 100:.2f}%",
        f"{p_img['good_false_ng_rate']['value'] * 100:.2f}%",
        f"{p_img['review_rate']['value'] * 100:.2f}%",
        f"{p_loc['localization_recall_iou0.10']['value'] * 100:.2f}%",
        f"{p2_top['0.10']['recall_at_k']['value'] * 100:.2f}%",
        f"{p2_top['0.20']['recall_at_k']['value'] * 100:.2f}%",
        f"{p2_tune['priority_breakdown']['P0']['ppv']['value'] * 100:.2f}%",
        f"{p3_norm[0]['psi']:.4f}",
        f"{p3_synth[0]['psi']:.4f}",
        f"{p4_bdown['area_bottleneck_ratio_all_review']['value'] * 100:.2f}%",
        f"{p4_spot['ng_edge_15pct_ratio']['value'] * 100:.2f}%",
        f"{p4_spot['review_edge_15pct_ratio']['value'] * 100:.2f}%",
    ]

    missing = [tok for tok in expected_tokens if tok not in body_text]
    assert missing == [], f"H7 증거-문서 수치 불일치/누락: {missing}"
