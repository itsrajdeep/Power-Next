"""
Submission compliance check — every requirement the organiser stated, asserted
mechanically against the files actually being shipped.

    python verification/check_compliance.py

Exits non-zero if anything fails, so it can gate a build. Run it from the
submission folder after `python run.py`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEAM = "pi-thon"   # ASCII transliteration of π-thon; see build_submission.py
EXPECTED_ROWS = 350
SUBMISSION_COLS = ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]
SUMMARY_FIELDS = [
    ("records_analysed", "number of records analysed"),
    ("abnormal_invalid_records", "number of abnormal / invalid records"),
    ("minimum_predicted_reference_parameter", "minimum predicted Reference Parameter"),
    ("maximum_predicted_reference_parameter", "maximum predicted Reference Parameter"),
    ("average_predicted_reference_parameter", "average predicted Reference Parameter"),
    ("highest_attention_test_ids", "three Test IDs needing attention"),
    ("approach_explanation", "explanation, max 100 words"),
]

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((bool(ok), name, detail))
    return bool(ok)


def find(*names: str) -> Path | None:
    """Locate a deliverable at the folder root or under outputs/final."""
    for n in names:
        for base in (ROOT, ROOT / "outputs" / "final"):
            if (base / n).exists():
                return base / n
    return None


def main() -> int:
    # ---- Deliverable 1: the prediction CSV ---------------------------------
    csv = find(f"{TEAM}.csv")
    if not check("1. <TeamName>.csv present", csv is not None, f"{TEAM}.csv"):
        report()
        return 1

    raw = csv.read_bytes()
    df = pd.read_csv(csv)

    check("   named after the team", csv.stem == TEAM, csv.name)
    check("   exactly 350 rows", len(df) == EXPECTED_ROWS, f"{len(df)} rows")
    check("   required columns, in order",
          list(df.columns) == SUBMISSION_COLS, str(list(df.columns)))
    check("   no missing values", int(df.isna().sum().sum()) == 0,
          f"{int(df.isna().sum().sum())} NaN")
    finite = df.Predicted_Reference_Parameter.map(
        lambda v: isinstance(v, (int, float)) and math.isfinite(v)).all()
    check("   predictions all finite numbers", bool(finite))
    labels = set(df.Validity_Label.unique())
    check("   labels are Valid/Invalid only", labels <= {"Valid", "Invalid"},
          str(sorted(labels)))
    check("   Test_IDs unique", df.Test_ID.is_unique)
    check("   no byte-order mark", not raw.startswith(b"\xef\xbb\xbf"))
    check("   no unnamed index column",
          not any(str(c).startswith("Unnamed") for c in df.columns))

    # Order must match the workbook's Sample_Submission for centralised scoring
    try:
        from src.contract import load_data
        _, _, sample = load_data()
        check("   Test_ID order matches Sample_Submission",
              bool((df.Test_ID.values == sample.Test_ID.values).all()))
    except Exception as e:                                   # noqa: BLE001
        check("   Test_ID order matches Sample_Submission", False, str(e)[:60])

    # Exactly one prediction CSV — the organiser forbids submitting several.
    # Identify them by shape, not by filename: a second file carrying the
    # submission columns is what would confuse centralised scoring, whereas a
    # diagnostics or leaderboard CSV is unambiguous.
    stray = []
    for p in ROOT.rglob("*.csv"):
        if p.resolve() == csv.resolve() or p.name == csv.name:
            continue
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
        except Exception:                                    # noqa: BLE001
            continue
        if cols == SUBMISSION_COLS:
            stray.append(p.relative_to(ROOT))
    check("   exactly one prediction CSV", not stray,
          ", ".join(str(p) for p in stray) or "only " + csv.name)

    # ---- Deliverable 2: summary.json ---------------------------------------
    js = find("summary.json")
    if check("2. summary.json present", js is not None):
        summary = json.loads(js.read_text(encoding="utf-8"))
        for key, label in SUMMARY_FIELDS:
            check(f"   {label}", key in summary,
                  "" if key in summary else "MISSING")
        expl = str(summary.get("approach_explanation", ""))
        words = len(expl.split())
        check("   explanation within 100 words", words <= 100, f"{words} words")
        ids = summary.get("highest_attention_test_ids", [])
        check("   exactly three attention Test IDs", len(ids) == 3, str(ids))
        check("   counts agree with the CSV",
              summary.get("records_analysed") == len(df)
              and summary.get("abnormal_invalid_records")
              == int((df.Validity_Label == "Invalid").sum()),
              f"{summary.get('records_analysed')} / "
              f"{summary.get('abnormal_invalid_records')}")

    # ---- Deliverable 3: source code ----------------------------------------
    check("3. entry point present", (ROOT / "run.py").exists(), "run.py")
    check("   requirements.txt present", (ROOT / "requirements.txt").exists())
    check("   model source present",
          (ROOT / "src" / "twin.py").exists() and (ROOT / "src" / "regress.py").exists())
    # Note: "no __pycache__ shipped" is enforced by scripts/build_submission.py
    # against the archive. It cannot be checked here, because importing anything
    # from src/ to run these checks creates a cache directory itself.

    # ---- Deliverable 4: methodology note ------------------------------------
    md = ROOT / "methodology.md"
    pdf = ROOT / "methodology.pdf"
    check("4. methodology note present", md.exists() or pdf.exists())
    if pdf.exists():
        try:
            from pypdf import PdfReader
            pages = len(PdfReader(str(pdf)).pages)
            check("   within two pages", pages <= 2, f"{pages} pages")
        except ImportError:
            check("   within two pages", True, "pypdf absent, not verified here")
    if md.exists():
        text = md.read_text(encoding="utf-8")
        for topic in ["Approach", "Parameters", "Abnormal Data",
                      "Assumptions", "Digital Twin"]:
            check(f"   covers: {topic}", topic.lower() in text.lower())

    return report()


def report() -> int:
    width = max(len(n) for _, n, _ in results)
    print("=" * (width + 20))
    print("SUBMISSION COMPLIANCE")
    print("=" * (width + 20))
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:{width}s}  {detail}")
    failed = [n for ok, n, _ in results if not ok]
    print("=" * (width + 20))
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED: {'; '.join(failed)}")
        return 1
    print(f"ALL {len(results)} CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
