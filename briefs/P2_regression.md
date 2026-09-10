# BRIEF — Person 2: Reference Parameter Model

You are an ML engineer working on the CPRI PowerNext-AI hackathon. Carry out this
brief end to end. **Do not ask the user questions** — every ambiguity is resolved
below. Report back only when all acceptance tests pass.

Work on branch `p2-regression`. You own **`src/regress.py` only**.

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

Predict `Reference_Parameter` — the verified hot-spot temperature rise, in °C
above ambient — for every test record. Worth **35%** of the competition score,
the heaviest single criterion.

## Verified context — do not re-derive this

Confirmed against the workbook by `verification/verify_findings.py`.

1. **Train on Valid rows only.** Invalid rows carry corrupted targets. Twelve
   duplicate groups have identical inputs but different `Reference_Parameter`
   values, which proves the target is unreliable on those rows.

   ```
   trained on all 1000 rows : MAE 0.8963   R² 0.95895
   trained on 866 Valid rows: MAE 0.6600   R² 0.98949    <- 27% better
   ```

2. **Use the four control settings only.** `Applied_Voltage_kV`,
   `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`. Adding the raw
   sensors buys roughly 5% and makes the prediction vulnerable to sensor faults.
   Staying sensor-free means you can predict the temperature even for records
   where a sensor is broken, and it keeps this lane fully independent of Person 1.

3. **A polynomial beats the boosted trees, and a blend beats both.**

   | Model | CV MAE | CV R² |
   |---|---|---|
   | HistGradientBoostingRegressor | 0.6600 | 0.98949 |
   | Degree-4 polynomial + Ridge | 0.5134 | 0.99515 |
   | Blend 0.65 poly + 0.35 trees | **0.4606** | **0.99607** |

4. **`Test_Duration_min` looks useless but is not.** Its raw correlation with the
   target is −0.01, yet dropping it moves MAE from 0.665 to 0.733. It is a
   suppressor variable. Keep it.

5. **`Sensor_S4` is irrelevant** — max absolute correlation 0.071. Excluded anyway
   by the sensor-free design.

6. **No train/test drift.** KS tests pass on all eight columns, so
   cross-validated scores are a trustworthy guide to test performance.

---

## What to build

Create `src/regress.py` exposing exactly these three functions:

```python
def fit_reference_model(train_valid: pd.DataFrame) -> dict:
    """Fit the polynomial, the tree model, and store the blend weight.
    Returns a dict holding both fitted models and the weight."""

def predict_reference(df: pd.DataFrame, model: dict) -> pd.Series:
    """Predicted Reference_Parameter for every row. Never NaN, never infinite."""

def evaluate(train_valid: pd.DataFrame) -> pd.DataFrame:
    """Cross-validated comparison table across the candidates you tried."""
```

Import `OPS`, `load_data`, `valid_rows`, `fold_indices`, `SEED` from
`src/contract.py`. Do not redefine them.

### Choosing the blend weight

The weight must be chosen from **out-of-fold predictions**, never from a fit on
the full training set, and never from the test data. Procedure:

1. Generate out-of-fold predictions for the polynomial and for the tree model
   using `fold_indices` from the contract.
2. Sweep the weight from 0 to 1 in steps of 0.05.
3. Pick the weight minimising out-of-fold MAE.
4. Store it in the returned model dict.

### Guarding against runaway extrapolation

A degree-4 polynomial can behave badly outside the range it was trained on. Add a
safety net inside `predict_reference`:

- Record the min and max of `Reference_Parameter` seen in training.
- Clip predictions to that range widened by 20% on each side.
- Count how many predictions needed clipping and expose it for the report.

This costs nothing on in-range data and prevents an absurd value on the second
hidden dataset.

---

## Tasks

1. Read `src/contract.py` so you use the shared helpers correctly.
2. Implement `src/regress.py` per the signatures above.
3. Sweep polynomial degree over `{3, 4, 5}` and Ridge alpha over
   `{1e-4, 1e-3, 1e-2, 1e-1, 1.0}`, all evaluated on the contract folds. Keep the
   best by out-of-fold MAE.
4. Fit `HistGradientBoostingRegressor`, lightly tuned. Do not run a long search.
5. Choose the blend weight as described above.
6. Write `verification/p2_report.py` that prints:
   - The full candidate comparison table with mean and standard deviation per fold
   - The chosen degree, alpha and blend weight
   - Final out-of-fold MAE, RMSE, R², median absolute error, max error
   - Error broken down by operating regime: current terciles, voltage terciles,
     and ambient temperature above vs below 40 °C
   - Worst-fold MAE, so we can see stability
   - Predicted min / max / mean across the 350 test rows
   - Clipping count on the test set
7. Run an extrapolation stress test in the same script: feed inputs at 1.2× and
   1.5× the maximum training current and voltage, and report whether predictions
   stay physically sensible (monotonically increasing with current, no negatives,
   no absurd magnitudes).
8. Save the printed output to `verification/p2_report.txt`.

---

## Acceptance tests — all must pass

- [ ] `python -c "from src.regress import fit_reference_model, predict_reference, evaluate"` succeeds
- [ ] Out-of-fold MAE ≤ **0.55** on the contract folds
- [ ] Out-of-fold R² ≥ **0.994**
- [ ] Worst individual fold MAE ≤ 0.70
- [ ] Blend weight was selected from out-of-fold predictions, provably not from a full-data fit
- [ ] `predict_reference` returns exactly one finite value per input row, no NaN, no infinity
- [ ] Predictions on the 350 test rows fall roughly within 10–70 °C
- [ ] Model trains on Valid rows only — assert this in code
- [ ] No sensor column appears among the model inputs — assert this in code
- [ ] Extrapolation test produces no negative and no absurd values

If an acceptance test fails, fix your implementation. Do not weaken the test.

---

## Do not

- Do not edit any file except `src/regress.py`, `verification/p2_report.py`, `verification/p2_report.txt`
- Do not modify `src/contract.py` or `folds.csv`
- Do not train on Invalid rows
- Do not use any sensor column as an input
- Do not use `Test_ID` or any part of it
- Do not use `Validity_Label` as an input — it does not exist in the test data
- Do not tune anything against the test set
- Do not build a gated model that depends on Person 1's validity output. It would
  couple the two lanes for roughly 5% gain. If you believe it is worth it, write
  the argument in your report instead of implementing it.
- Do not run an unbounded hyperparameter search. Keep total runtime under 10 minutes.

## If you get stuck

Do not stop and ask. Use the fallback, note it, continue:

| Problem | Fallback |
|---|---|
| Degree-5 polynomial is unstable or slow | Cap at degree 4, document it |
| Ridge is ill-conditioned | Standardise before the polynomial expansion, raise alpha |
| Blend weight lands at 0 or 1 | Accept it, report the single model won, show the sweep |
| Tree model is slow | Reduce `max_iter` to 300, document it |
| A library is missing | `pip install` it and add it to `requirements.txt` |

---

## Final report

When every acceptance test passes, reply with:

1. The candidate comparison table
2. Chosen degree, alpha, blend weight, and the out-of-fold sweep that picked it
3. Final MAE / RMSE / R² / median AE / max error, plus worst-fold MAE
4. Error by operating regime
5. Test-set prediction min / max / mean and clipping count
6. Extrapolation stress-test results
7. Any fallback you used
8. Anything you would need changed in another person's file (describe, do not edit)
