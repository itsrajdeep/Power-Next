# Team Briefs — How to use these

Each person owns one file in this folder. Open your agentic coding tool in the
repository root and give it this instruction:

```
Read briefs/P1_sensor_twin.md and carry it out completely.
Follow it exactly. Do not ask me questions - the brief tells you what to do
when something is ambiguous. Report back only when every acceptance test passes.
```

Substitute your own brief filename.

## Who has which brief

| Person | Brief | Owns | Weight of marks |
|---|---|---|---|
| 1 | `P1_sensor_twin.md` | Invalid detection | 25% |
| 2 | `P2_regression.md` | Reference Parameter | 35% |
| 3 | `P3_benchmark.md` | Leaderboard + methodology note | 10% |
| 4 | `P4_pipeline.md` | Pipeline + deliverables | 10% |

The remaining 20% (second hidden dataset) is protected by all four working correctly.

## Before anyone starts

One person runs this once and commits the result:

```bash
pip install -r requirements.txt
python src/contract.py
git add src/contract.py folds.csv && git commit -m "chore: shared contract and folds" && git push
```

It is deterministic, so if two people run it they get an identical `folds.csv`.
If yours differs from what is committed, delete yours and pull.

## Branching

```bash
git checkout -b p1-twin        # p2-regression / p3-benchmark / p4-pipeline
```

Never commit to `main` directly. Open a pull request, one reviewer.

## File ownership — do not edit outside your lane

| Path | Owner |
|---|---|
| `src/contract.py`, `folds.csv` | **Nobody.** Shared. Changes need team agreement. |
| `src/twin.py` | Person 1 |
| `src/regress.py` | Person 2 |
| `src/benchmark.py`, `methodology.md`, `outputs/charts/` | Person 3 |
| `run.py`, `src/finalize.py`, `tests/`, `outputs/final/` | Person 4 |
| `analysis/`, `verification/`, `PROJECT_GUIDE.md` | Read-only reference |

If your work seems to need a change in someone else's file, do not make it.
Write the request into your final report instead.

## Nobody is blocked

Person 4 starts with stub versions of Person 1 and 2's functions and builds the
whole pipeline against them. Person 3 runs every off-the-shelf model immediately
and adds two rows at the end. All four lanes begin at the same moment.

## Context you need

Read `PROJECT_GUIDE.md` for the full picture. Every brief also contains the
verified findings relevant to it, so you can start without reading anything else.

All findings are reproducible with `python verification/verify_findings.py`.
