# BRIEF — Person 1: Sensor Twin & Invalid Detection

You are an ML engineer working on the CPRI PowerNext-AI hackathon. Carry out this
brief end to end. **Do not ask the user questions** — every ambiguity is resolved
below. Report back only when all acceptance tests pass.

Work on branch `p1-twin`. You own **`src/twin.py` only**.

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

Decide, for any test record, whether it is `Valid` or `Invalid`. Worth **25%** of
the competition score.

## Verified context — do not re-derive this

These facts are already confirmed against the workbook by
`verification/verify_findings.py`. Trust them; your job is to implement and
validate, not rediscover.

1. **The three real sensors are deterministic functions of the four control
   settings.** Fitting a degree-4 polynomial from `[Applied_Voltage_kV,
   Load_Current_A, Ambient_Temperature_C, Test_Duration_min]` to each sensor,
   using Valid rows only, gives:

   | Sensor | CV R² | residual σ |
   |---|---|---|
   | Sensor_S1 | 0.99748 | 0.19 |
   | Sensor_S2 | 0.99727 | 0.18 |
   | Sensor_S3 | 0.99595 | 0.31 |

   Call these models the **twin**. `residual = actual − twin prediction`.

2. **Every Invalid record falls into exactly one of three buckets**, and they sum
   exactly to the 134 Invalid rows:

   | Cause | Count | Detection |
   |---|---|---|
   | A sensor reading is missing | 15 | any of S1/S2/S3 is NaN |
   | The row duplicates another row | 24 | measurement vector appears twice |
   | One sensor is spiked | 95 | `max abs residual` is large |

3. **The spike separation is a canyon, not a boundary:**

   ```
   largest residual on a Valid row   : 1.0551
   smallest residual on a spiked row : 4.1774
   ```

   Any threshold in `[1.1, 4.0]` scores a perfect F1 on training data.

4. **Sensor_S4 is irrelevant** — maximum absolute correlation with anything in the
   dataset is 0.071. Do not use it as a model input.

5. **FORBIDDEN SHORTCUT.** In training, all 29 rows with a missing `Sensor_S4`
   happen to be Valid. Do **not** use this. There is no physical reason for it and
   it will not survive the second hidden dataset. Missing S4 must have no effect
   on your verdict.

---

## What to build

Create `src/twin.py` exposing exactly these four functions:

```python
def fit_twin(train_valid: pd.DataFrame) -> dict:
    """Fit one sensor model per sensor in S3. Returns {sensor_name: model}."""

def sensor_residuals(df: pd.DataFrame, twin: dict) -> pd.DataFrame:
    """Columns S1/S2/S3, values = actual - expected. NaN where sensor is missing."""

def derive_threshold(train_valid: pd.DataFrame, twin: dict) -> float:
    """Compute the spike threshold FROM THE DATA. See rules below."""

def predict_validity(df: pd.DataFrame, twin: dict, threshold: float) -> pd.Series:
    """Returns a Series of the strings "Valid" / "Invalid", indexed like df."""
```

Import all shared pieces from `src/contract.py` — `OPS`, `S3`, `MEASUREMENTS`,
`load_data`, `valid_rows`, `is_invalid`, `is_duplicate`, `fold_indices`, `SEED`.
Do not redefine them.

### The threshold must be self-calibrating

Do **not** hardcode a number. The second hidden dataset may have different noise
levels, and a fixed threshold is our single biggest risk. Derive it at runtime
from the Valid-row residual distribution, for example:

```
threshold = max(k * robust_sigma_of_valid_residuals, small_floor)
```

Pick `k` so that the resulting threshold lands inside `[1.1, 4.0]` on this
dataset, and document your choice in the docstring. A value around 6–8 robust
sigmas works because the Valid residual σ is roughly 0.15. Log the derived
threshold whenever the function runs.

### Verdict logic

```
Invalid  if  any of S1/S2/S3 is missing
         or  the measurement vector is duplicated within this dataframe
         or  max(|residual|) over the available sensors > threshold
Valid    otherwise
```

Rows where a sensor is missing are already Invalid by rule one, so a NaN residual
must never silently become Valid. Handle that explicitly.

Compute duplicates **within a single dataframe only**. Never concatenate train and
test before checking.

---

## Tasks

1. Read `src/contract.py` so you use the shared helpers correctly.
2. Implement `src/twin.py` per the signatures above.
3. Write `verification/p1_report.py` that prints:
   - Per-sensor twin CV R², MAE, residual σ (fit on Valid rows only)
   - The derived threshold
   - The fault taxonomy counts and whether they sum to 134
   - The separation gap: max Valid residual vs min spiked residual
   - Out-of-fold precision / recall / F1 using `fold_indices` from the contract,
     **with the twin refit inside every fold** on that fold's Valid training rows
   - Counts when applied to the 350 test rows, plus the flagged rate
4. Run a robustness check in the same script: add Gaussian noise of σ = 0.1, 0.25
   and 0.5 to the sensors of Valid rows and report whether the gap still separates.
5. Save the printed output to `verification/p1_report.txt`.

---

## Acceptance tests — all must pass

Run these yourself and confirm each one:

- [ ] `python -c "from src.twin import fit_twin, sensor_residuals, derive_threshold, predict_validity"` succeeds
- [ ] Each sensor twin achieves CV R² ≥ 0.995
- [ ] Fault taxonomy sums to exactly **134**
- [ ] Derived threshold falls inside `[1.1, 4.0]` and is not a literal in the code
- [ ] Out-of-fold F1 ≥ 0.99 with the twin refit inside each fold
- [ ] Applied to the 350 test rows, the flagged rate lands between 10% and 17%
- [ ] `predict_validity` returns only the strings `"Valid"` and `"Invalid"`, never NaN
- [ ] Output length always equals input length
- [ ] Removing `Sensor_S4` from the test dataframe entirely does not change any verdict
- [ ] Gap still separates at added noise σ = 0.25

If an acceptance test fails, fix your implementation. Do not weaken the test.

---

## Do not

- Do not edit any file except `src/twin.py`, `verification/p1_report.py`, `verification/p1_report.txt`
- Do not modify `src/contract.py` or `folds.csv`
- Do not use `Sensor_S4` values, or its missingness, as a signal
- Do not use `Test_ID` or any part of it as an input
- Do not use `Reference_Parameter` as an input — it does not exist in the test data
- Do not fit the twin on Invalid rows
- Do not hardcode the threshold
- Do not train a supervised classifier here — that is Person 3's benchmark lane

## If you get stuck

Do not stop and ask. Use the documented fallback, note it in your report, continue:

| Problem | Fallback |
|---|---|
| Degree-4 polynomial is numerically unstable | Standardise inputs before the polynomial expansion, raise Ridge alpha to 1e-2 |
| A fold has too few rows for a twin | Fall back to the twin fitted on all Valid training rows, log it |
| Threshold lands outside `[1.1, 4.0]` | Adjust `k`, document the final value and why |
| A library is missing | `pip install` it and add it to `requirements.txt` |

---

## Final report

When every acceptance test passes, reply with:

1. The per-sensor twin table (R², MAE, σ)
2. The derived threshold and the rule that produced it
3. The taxonomy counts and the sum
4. The separation gap
5. Out-of-fold precision / recall / F1
6. Test-set flag counts and rate
7. Noise-robustness results
8. Any fallback you used
9. Anything you would need changed in another person's file (describe, do not edit)
