"""
Build the submission archive, end to end and reproducibly.

    python scripts/build_submission.py

Regenerates the deliverables, renders the methodology PDF, assembles the
`pi-thon/` folder, zips it, and then re-verifies the archive from the outside:
structure, the four deliverables, and no compiled caches. Fails loudly rather
than shipping something unchecked.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
# Filenames and paths use the ASCII transliteration of the team name
# π-thon. The π is kept in the documents themselves, where it cannot
# break a scoring script that assumes ASCII paths.
TEAM = "pi-thon"
STAGE = ROOT / TEAM
ARCHIVE = ROOT / f"{TEAM}-submission.zip"

# Copied into the archive root alongside the generated deliverables.
ROOT_FILES = ["methodology.md", "methodology.pdf", "run.py",
              "requirements.txt", "folds.csv"]
# The package README is written for the judges and is a different document from
# the repository README, which is written for the team.
PACKAGE_README = ROOT / "docs" / "submission_README.md"
TREES = ["src", "tests", "verification", "scripts"]
DELIVERABLES = [f"{TEAM}.csv", "summary.json", "methodology.pdf", "run.py"]


def run(*cmd: str) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")


def strip_caches(base: Path) -> None:
    for d in list(base.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    for f in base.rglob("*.pyc"):
        f.unlink(missing_ok=True)


def main() -> int:
    print("[1/5] Regenerating deliverables")
    # Drop prediction files left over from an earlier team name. The organiser
    # allows exactly one prediction CSV, and a stale one sitting in outputs/
    # would be swept into the archive alongside the current one.
    import csv as _csv
    final_dir = ROOT / "outputs" / "final"
    for old in final_dir.glob("*.csv"):
        if old.name == f"{TEAM}.csv":
            continue
        with old.open(newline="", encoding="utf-8") as fh:
            header = next(_csv.reader(fh), [])
        if header == ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]:
            print(f"  removing stale prediction file: {old.name}")
            old.unlink()
    run("run.py")

    print("[2/5] Rendering the methodology PDF")
    run("scripts/build_methodology_pdf.py")

    print("[3/5] Staging the submission folder")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "dataset").mkdir(parents=True)
    (STAGE / "outputs").mkdir()

    final = ROOT / "outputs" / "final"
    for name in [f"{TEAM}.csv", "summary.json"]:
        shutil.copy2(final / name, STAGE / name)
    for name in ROOT_FILES:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, STAGE / name)
    shutil.copy2(PACKAGE_README, STAGE / "README.md")
    for tree in TREES:
        if (ROOT / tree).exists():
            shutil.copytree(ROOT / tree, STAGE / tree,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for xlsx in (ROOT / "dataset").glob("*.xlsx"):
        shutil.copy2(xlsx, STAGE / "dataset" / xlsx.name)
    for sub in ["final", "charts", "benchmark"]:
        if (ROOT / "outputs" / sub).exists():
            shutil.copytree(ROOT / "outputs" / sub, STAGE / "outputs" / sub)
    mc = ROOT / "outputs" / "model_comparison.csv"
    if mc.exists():
        shutil.copy2(mc, STAGE / "outputs" / mc.name)
    strip_caches(STAGE)

    print("[4/5] Writing the archive")
    ARCHIVE.unlink(missing_ok=True)
    files = sorted(p for p in STAGE.rglob("*") if p.is_file())
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())

    print("[5/5] Verifying the archive from the outside")
    z = zipfile.ZipFile(ARCHIVE)
    names = z.namelist()
    problems = []

    if z.testzip() is not None:
        problems.append("archive fails its own CRC check")
    roots = {n.split("/")[0] for n in names}
    if roots != {TEAM}:
        problems.append(f"expected one top-level folder {TEAM!r}, found {sorted(roots)}")
    for d in DELIVERABLES:
        if f"{TEAM}/{d}" not in names:
            problems.append(f"deliverable missing from archive root: {d}")
    caches = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    if caches:
        problems.append(f"{len(caches)} compiled cache entries present")
    predictions = [n for n in names
                   if n.endswith(".csv") and Path(n).name == f"{TEAM}.csv"]
    if not predictions:
        problems.append("no prediction CSV in the archive")
    # Any path a scoring script might read should be plain ASCII, so that it
    # cannot be mangled by an extractor or a codepage. If a non-ASCII name is
    # ever reintroduced, the archive must at least set the UTF-8 name flag.
    non_ascii = [n for n in names if not n.isascii()]
    if non_ascii:
        unflagged = [n for n in non_ascii if not z.getinfo(n).flag_bits & 0x800]
        if unflagged:
            problems.append(f"{len(unflagged)} non-ASCII name(s) without the UTF-8 flag")
        else:
            print(f"  note    : {len(non_ascii)} non-ASCII path(s), UTF-8 flag set")

    size_kb = ARCHIVE.stat().st_size / 1024
    print()
    print(f"  archive : {ARCHIVE.name}")
    print(f"  size    : {size_kb:.0f} KB")
    print(f"  entries : {len(names)}")
    print(f"  root    : {sorted(roots)[0]}/")
    print()
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    print("  [PASS] structure, deliverables, encoding and cache hygiene all verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
