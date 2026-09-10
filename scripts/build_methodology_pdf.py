"""
Render methodology.md to a two-page PDF.

The organiser caps the methodology note at two pages, which is only meaningful
against a paginated format. This produces that PDF and fails loudly if the
result runs over, so the limit is checked rather than assumed.

    python scripts/build_methodology_pdf.py

Needs reportlab, which is a build-time dependency only — it is deliberately not
in requirements.txt, since producing the deliverables does not require it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "methodology.md"
OUT = ROOT / "methodology.pdf"
PAGE_LIMIT = 2

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=8.4, leading=10.8,
                      alignment=TA_JUSTIFY, spaceAfter=3.6)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=16,
                    spaceAfter=2)
SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=8, leading=10,
                     textColor="#555555", spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.6, leading=12,
                    spaceBefore=6, spaceAfter=2.5)
BULLET = ParagraphStyle("bullet", parent=BODY, spaceAfter=2.2)


def inline(text: str) -> str:
    """Markdown inline markup -> reportlab markup."""
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


def build() -> int:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    story, pending = [], []

    def flush() -> None:
        """Emit any accumulated list items as one block."""
        if pending:
            numbered = pending[0][1] == "num"
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(t), BULLET), leftIndent=10)
                 for t, _ in pending],
                bulletType="1" if numbered else "bullet",
                start=1 if numbered else None,
                bulletFontSize=7, leftIndent=11, spaceAfter=3,
            ))
            pending.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line or line.strip() == "---":
            flush()
            continue
        if line.startswith("# "):
            flush(); story.append(Paragraph(inline(line[2:]), H1))
        elif line.startswith("## "):
            flush(); story.append(Paragraph(inline(line[3:]), H2))
        elif line.startswith("- "):
            pending.append((line[2:], "bullet"))
        elif re.match(r"^\d+\.\s", line):
            pending.append((re.sub(r"^\d+\.\s", "", line), "num"))
        elif line.startswith("**Team:**") or line.startswith("*Benchmarks:"):
            # Subtitle / footer line. Pass through inline() untouched — stripping
            # asterisks here would break the ** ** pairs it needs to see.
            flush(); story.append(Paragraph(inline(line), SUB))
        else:
            flush(); story.append(Paragraph(inline(line), BODY))
    flush()

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=13 * mm, bottomMargin=13 * mm,
        title="π-thon — Methodology Note", author="π-thon",
    )
    doc.build(story)

    pages = doc.page
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.name}: {pages} page(s), {size_kb:.0f} KB")
    if pages > PAGE_LIMIT:
        print(f"FAIL: methodology note is {pages} pages, limit is {PAGE_LIMIT}")
        return 1
    print(f"PASS: within the {PAGE_LIMIT}-page limit")
    return 0


if __name__ == "__main__":
    sys.exit(build())
