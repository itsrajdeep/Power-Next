# π-thon — CPRI Hackathon Screening Round Submission

## Required deliverables

**A note on the file names.** Our team name is **π-thon**. Filenames and folder
names use the ASCII transliteration `pi-thon`, because a non-ASCII character in
a path can break automated scoring on some systems. The π is preserved
everywhere it is safe — in this document, the methodology note and the team
identity itself.


| File | Deliverable |
|---|---|
| `pi-thon.csv` | 1. Prediction file — 350 rows: Test_ID, Predicted_Reference_Parameter, Validity_Label |
| `summary.json` | 2. Automated summary — produced by the program, not written by hand |
| `run.py`, `src/` | 3. Source code — runs beginning to end in one command |
| `methodology.pdf` | 4. Methodology note — two pages (`methodology.md` is the same text in source form) |

## Running it

```
pip install -r requirements.txt
python run.py
```

No arguments, no manual steps, no per-record editing. Deterministic: seed 42,
validation folds fixed in `folds.csv`. Writes `outputs/final/pi-thon.csv`,
`outputs/final/summary.json` and `outputs/final/test_diagnostics.csv`. The
copies at the root of this folder are the output of exactly that command.

Validated end to end on Python 3.13 in a clean virtual environment installing
only from `requirements.txt`.

## Checking it

```
python verification/check_compliance.py   # every stated requirement, asserted
python verification/verify_findings.py    # each data finding, from the raw workbook
python verification/p1_report.py          # invalid detection: OOF precision/recall/F1
python verification/p2_report.py          # regression: OOF error, regimes, extrapolation
python src/benchmark.py                   # 51-row model leaderboard and ablations
python -m pytest tests/ -q                # 90 tests
```

## Method in brief

**Invalid detection.** A physics twin predicts each sensor from the four
operating conditions. A record is Invalid if a sensor is missing, its
measurement vector is duplicated, or a residual exceeds a self-calibrating
threshold. Out-of-fold precision, recall and F1 are all 1.000 across the 1,000
training rows. On the 350 test rows it flags 46 (13.1%, against a 13.4%
training rate), and the residual separation gap seen in training holds.

**Reference Parameter.** A degree-4 polynomial Ridge blended with
gradient-boosted trees, weight 0.65 selected out-of-fold, trained on the 866
Valid rows only. No sensor is used as an input, so a faulty sensor cannot
corrupt the temperature estimate. Out-of-fold MAE 0.4578, R² 0.99625, worst
fold 0.5016. Predictions are clipped to the training range widened by 20%.

**Both entries lead the benchmark** in `outputs/model_comparison.csv`: the twin
beats every off-the-shelf classifier (best learned F1 0.71) and the blend beats
every off-the-shelf regressor (next best MAE 0.4736).
