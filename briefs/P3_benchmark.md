# BRIEF — Person 3: Benchmark Leaderboard & Methodology Note

You are an ML engineer and technical writer working on the CPRI PowerNext-AI
hackathon. Carry out this brief end to end. **Do not ask the user questions** —
every ambiguity is resolved below. Report back only when all acceptance tests pass.

Work on branch `p3-benchmark`. You own `src/benchmark.py`, `methodology.md`,
`outputs/model_comparison.csv`, `outputs/charts/`.

---

## THE WHOLE SYSTEM — how your work gets combined

Read this before starting. It tells you what everyone else is building, what the
finished machine looks like, and exactly where your piece plugs in.

### The problem, in one paragraph

A lab ran 1,350 tests on electrical equipment. Each test applies a voltage and a
current at some ambient temperature for some duration, and four sensors record how
hot the equipment gets. For 1,000 tests we know two extra things: the true hot-spot
temperature (`Reference_Parameter`, measured with a calibrated instrument) and an
engineer's verdict on whether the test was trustworthy (`Validity_Label`). For the
other 350, both are missing. We must fill them in.

### The four lanes

```
                       src/contract.py  +  folds.csv
                    (shared: data, duplicate groups, 5 folds)
                                     |
       +--------------+--------------+--------------+--------------+
       |              |              |              |              |
   PERSON 1       PERSON 2       PERSON 3       PERSON 4
   src/twin.py    src/regress.py src/benchmark. run.py
                                 py             src/finalize.py
       |              |              |              |
  Valid/Invalid   predicted      leaderboard    assembles
  verdict         temperature    + methodology  everything
       |              |              |              |
       +------------->+<-------------+              |
                      |                             |
                      +---------------------------->+
                                                    |
                                                    v
                                      outputs/final/<TEAM>.csv
                                      outputs/final/summary.json
                                      methodology.md
                                      -> zipped submission
```

### Who produces what

| Person | File | Produces | Feeds into |
|---|---|---|---|
| 1 | `src/twin.py` | `Valid` / `Invalid` per record | P4's CSV column 3, P3's leaderboard |
| 2 | `src/regress.py` | Predicted temperature per record | P4's CSV column 2, P3's leaderboard |
| 3 | `src/benchmark.py`, `methodology.md` | Proof our methods won, plus the write-up | Judges directly |
| 4 | `run.py`, `src/finalize.py` | The submission files | Judges directly |

### The interface contract — this is what makes it fit together

Person 4's `run.py` calls exactly these. Signatures are fixed; do not change them.

```python
from src.twin import fit_twin, derive_threshold, predict_validity
from src.regress import fit_reference_model, predict_reference

twin      = fit_twin(train_valid)                       # P1
threshold = derive_threshold(train_valid, twin)         # P1
model     = fit_reference_model(train_valid)            # P2

labels = predict_validity(test, twin, threshold)        # P1 -> "Valid"/"Invalid"
preds  = predict_reference(test, model)                 # P2 -> float

submission = pd.DataFrame({
    "Test_ID": test["Test_ID"],
    "Predicted_Reference_Parameter": preds,
    "Validity_Label": labels,
})
```

Both prediction functions take a dataframe and return a Series of the same length,
in the same row order. That is the entire coupling between lanes.

### Nobody is blocked

- **P1 and P2 do not depend on each other at all.** P2 trains on rows already
  labelled Valid in the training data, and predicts from control settings only, so
  it never needs P1's verdict.
- **P4 starts immediately with stubs** — `predict_validity` returning `"Valid"`,
  `predict_reference` returning `25.0` — and builds the whole pipeline against
  them. When the real modules land, two stub files get deleted. Nothing else
  changes.
- **P3 runs every off-the-shelf model immediately** and adds two rows for P1 and
  P2's methods at the end.

Everyone starts at the same moment.

### The shared contract — never modify

`src/contract.py` and `folds.csv` are shared. Every lane imports from the contract
so all four are guaranteed to use identical data loading, identical duplicate
grouping, and **identical validation folds**. If two people validate on different
folds, their numbers are not comparable and the decision gate below collapses.

```python
from src.contract import OPS, S3, SENSORS, MEASUREMENTS, SEED
from src.contract import load_data, valid_rows, is_invalid, is_duplicate, fold_indices
```

Run everything from the repository root so imports resolve.

### File ownership — do not edit outside your lane

| Path | Owner |
|---|---|
| `src/contract.py`, `folds.csv` | Nobody. Shared. Changes need team agreement. |
| `src/twin.py` | Person 1 |
| `src/regress.py` | Person 2 |
| `src/benchmark.py`, `methodology.md`, `outputs/charts/` | Person 3 |
| `run.py`, `src/finalize.py`, `tests/`, `outputs/final/`, `README.md`, `requirements.txt` | Person 4 |
| `analysis/`, `verification/`, `PROJECT_GUIDE.md`, `briefs/` | Read-only reference |

If your work seems to require a change in someone else's file, **do not make it**.
Write the request into your final report and a human will arbitrate.

### The timeline

| Phase | What happens |
|---|---|
| 0 | One person commits `src/contract.py` and `folds.csv`. Done already if they are in your pull. |
| 1 | All four lanes build in parallel. **You are here.** |
| 2 | P4 swaps stubs for the real modules, first full end-to-end run |
| 3 | **Decision gate** — P3 presents the leaderboard, a human decides what ships |
| 4 | P4 runs robustness attacks, P1 and P2 patch what breaks |
| 5 | Freeze, verify, zip, submit |

### The decision gate

When P3's leaderboard lands, a human compares every method on identical folds and
decides what goes into the final submission. Agents do not make this call. Your job
is to make your lane as good as it can honestly be and report real numbers — not to
argue for your own method.

### What the team is scored on

| Criterion | Weight | Owner |
|---|---|---|
| Reference Parameter accuracy | 35% | Person 2 |
| Invalid record detection | 25% | Person 1 |
| Second unseen dataset | 20% | All — via honest validation and robustness |
| Reproducibility and code quality | 10% | Person 4 |
| Engineering reasoning | 10% | Person 3 |

### Two facts that shape every lane

1. **The sensors are predictable from the control settings.** A degree-4 polynomial
   from voltage, current, ambient temperature and duration predicts each of
   Sensor_S1/S2/S3 at R² ≈ 0.997. This is why detection works: compare a reading to
   what it should have been, and a spike becomes obvious while a genuinely hot test
   does not.

2. **Invalid records have corrupted `Reference_Parameter` values.** Train regression
   on Valid rows only. Twelve duplicate groups have identical inputs but different
   target values, which proves the target is unreliable on those rows.

Everything above is reproducible with `python verification/verify_findings.py`.

---

## Goal

Two things, both about **evidence**:

1. Run every credible model as a fair contest on the shared folds, so we can prove
   our chosen method won rather than asserting it.
2. Write the 2-page methodology note. Worth **10%** of the score directly, and it
   is the document a judge actually reads.

**You are not blocked by anyone.** Run every off-the-shelf model immediately.
Person 1 and Person 2's entries are two extra rows you add at the end.

## Verified context

Confirmed by `verification/verify_findings.py`. Use these as the target rows your
leaderboard must reproduce.

- Classification: a physics-based rule scores out-of-fold **F1 = 1.000**, while a
  tuned boosted-tree classifier on engineered features reaches only about 0.73.
- Regression on Valid rows, control settings only:

  | Model | CV MAE | CV R² |
  |---|---|---|
  | Boosted trees | 0.6600 | 0.98949 |
  | Degree-4 polynomial | 0.5134 | 0.99515 |
  | Blend 0.65 / 0.35 | 0.4606 | 0.99607 |

- `Sensor_S4` is irrelevant (max absolute correlation 0.071).
- No train/test drift on any column.

---

## Part A — The benchmark

Create `src/benchmark.py`. It must use `fold_indices` from `src/contract.py` so
every model is scored on identical folds. Anything scored on different folds is
not comparable and must not appear in the same table.

### Classification lane

Target: `Validity_Label`. Run at minimum:

```
DummyClassifier (most_frequent)      LogisticRegression (class_weight=balanced)
RandomForestClassifier               ExtraTreesClassifier
HistGradientBoostingClassifier       XGBClassifier
LGBMClassifier                       CatBoostClassifier
```

Feature sets to run each against:

- `raw_operating` — the four control settings
- `all_raw` — control settings plus all four sensors
- `sensor_consistency` — pairwise sensor differences, spread, deviation from
  median, missing-count flags, duplicate flag

Metrics, all computed out-of-fold: precision / recall / F1 for the **Invalid**
class, macro F1, balanced accuracy, ROC-AUC, PR-AUC, MCC.

**Do not rank on plain accuracy.** With 13.4% Invalid, predicting everything Valid
scores 86.6% accuracy and is useless. The Dummy row exists to make that visible.

### Regression lane

Target: `Reference_Parameter`, trained on Valid rows only. Run at minimum:

```
DummyRegressor (median)      Ridge              ElasticNet
HuberRegressor               RandomForest       ExtraTrees
HistGradientBoosting         XGBRegressor       LGBMRegressor
CatBoostRegressor            SVR
```

Feature sets:

- `raw_operating` — four control settings
- `all_raw` — control settings plus sensors
- `poly4_operating` — degree-4 expansion of the control settings

Metrics out-of-fold: MAE, RMSE, R², median absolute error, max error, per-fold
standard deviation, worst-fold MAE.

### Ablations to include

Run these explicitly, because each one is a claim the methodology note makes:

| Ablation | What it demonstrates |
|---|---|
| Regression trained on all rows vs Valid rows only | Invalid targets are corrupted |
| With `Sensor_S4` vs without | S4 is irrelevant |
| With `Test_Duration_min` vs without | Duration matters despite zero correlation |
| Polynomial degree 2 vs 3 vs 4 vs 5 | Stopping at degree 2 loses accuracy |

### Adding the team's entries

At the end, import `predict_validity` from `src/twin.py` and `predict_reference`
from `src/regress.py` and score them on the same folds. If those files do not
exist yet, write the leaderboard without them and leave a clearly marked
placeholder row; rerun once they land.

Output `outputs/model_comparison.csv` with one row per (task, model, feature set),
sorted best first within each task, including a `rank` column.

---

## Part B — Charts

Save to `outputs/charts/`, readable at a glance, axis labels and titles on all:

1. `gap_separation.png` — histogram of max absolute sensor residual, Valid vs
   Invalid overlaid, log x-axis, with the threshold marked. This is the single
   most persuasive image in the submission.
2. `fault_taxonomy.png` — bar chart of 15 / 24 / 95 summing to 134.
3. `leaderboard_classification.png` — Invalid-class F1 by model.
4. `leaderboard_regression.png` — MAE by model.
5. `pred_vs_actual.png` — predicted vs actual Reference Parameter with the
   diagonal, out-of-fold.
6. `residual_by_regime.png` — regression residuals across current terciles.

## Part C — The methodology note

Write `methodology.md`. **Maximum two pages when rendered.** Cover, in this order:

1. **Approach** — we modelled how the equipment behaves rather than fitting a
   black box to the labels.
2. **Parameters considered important** — the four control settings; S1/S2/S3 as
   monitored outputs rather than inputs; S4 excluded with the measured correlation
   as justification.
3. **Method for detecting abnormal data** — the twin, the residual, the
   self-calibrating threshold, the three fault buckets, and explicitly how this
   separates a genuine high-load regime from a sensor fault.
4. **Assumptions** — Valid-row targets are trustworthy and Invalid ones are not;
   sensor faults affect one sensor at a time; the second dataset comes from the
   same equipment.
5. **Limitations** — polynomial extrapolation beyond the training envelope; the
   threshold assumes a comparable noise level; we deliberately declined the
   missing-S4 shortcut and why.
6. **Digital twin deployment steps** — this is the differentiator. We already have
   a working twin, so describe the real system: ingest live readings, evaluate the
   twin, compute residuals, compare to a rolling threshold, classify the fault
   bucket, raise an alarm with the named suspect sensor, log for periodic
   recalibration against fresh calibrated reference tests. Include the measured
   false-alarm rate.

Cite real numbers throughout. Generic prose scores generically.

---

## Tasks

1. Read `src/contract.py`.
2. Build `src/benchmark.py` with both lanes and all ablations.
3. Run it, produce `outputs/model_comparison.csv`.
4. Produce the six charts.
5. Write `methodology.md`.
6. Save console output to `verification/p3_report.txt`.

---

## Acceptance tests — all must pass

- [ ] `python src/benchmark.py` runs start to finish without error
- [ ] `outputs/model_comparison.csv` exists with at least **25** rows
- [ ] Every row was scored using `fold_indices` from the contract
- [ ] Both lanes include a Dummy baseline
- [ ] All four ablations appear in the table
- [ ] Classification ranking is by Invalid-class F1, not accuracy
- [ ] All six charts exist in `outputs/charts/` and are non-empty
- [ ] `methodology.md` exists, is under two rendered pages, and contains all six sections
- [ ] The methodology note cites at least eight specific measured numbers
- [ ] The digital twin section describes the built system, not a hypothetical one

If an acceptance test fails, fix it. Do not weaken the test.

---

## Do not

- Do not edit `src/twin.py`, `src/regress.py`, `run.py`, `src/contract.py`, `folds.csv`
- Do not score any model on folds other than the contract folds
- Do not rank classifiers by accuracy
- Do not tune hyperparameters extensively — this is a fair contest at sensible
  defaults, not a tuning competition. Keep total runtime under 30 minutes.
- Do not use the test set to select or score anything
- Do not overstate results. If our method loses a row, report it honestly — an
  honest table we win is worth far more than a flattering one.

## If you get stuck

| Problem | Fallback |
|---|---|
| CatBoost / XGBoost / LightGBM will not install | Skip it, add a row marked `not_available`, note it |
| A model is too slow | Reduce estimators, document the setting used |
| `src/twin.py` or `src/regress.py` missing | Build the leaderboard without them, mark placeholder rows, rerun later |
| A chart is unreadable | Increase figure size and font, do not delete the chart |

---

## Final report

1. Top five rows of each leaderboard
2. Where the team's rule and blend ranked
3. All four ablation results with the numbers
4. Confirmation the six charts rendered
5. The methodology note's section list and rendered length
6. Any model you had to skip and why
7. Anything you would need changed in another person's file (describe, do not edit)
