# BRIEF — Person 4: Pipeline, Deliverables & Integration QA

You are a software engineer working on the CPRI PowerNext-AI hackathon. Carry out
this brief end to end. **Do not ask the user questions** — every ambiguity is
resolved below. Report back only when all acceptance tests pass.

Work on branch `p4-pipeline`. You own `run.py`, `src/finalize.py`, `tests/`,
`outputs/final/`, `requirements.txt`, `README.md`.

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

Own everything the judges actually receive. Worth **10%** directly for
reproducibility and code quality, and you are the last line of defence for the
**20%** second-hidden-dataset criterion.

## You are not blocked — start with stubs

Person 1 and Person 2 are building `src/twin.py` and `src/regress.py` in parallel.
Do not wait. Create these stubs immediately and build the entire pipeline against
them:

```python
# src/twin_stub.py
def predict_validity(df, twin=None, threshold=None):
    return pd.Series(["Valid"] * len(df), index=df.index)

# src/regress_stub.py
def predict_reference(df, model=None):
    return pd.Series([25.0] * len(df), index=df.index)
```

In `run.py`, import the real module if it exists and fall back to the stub:

```python
try:
    from src.twin import fit_twin, derive_threshold, predict_validity
    from src.regress import fit_reference_model, predict_reference
    USING_STUBS = False
except ImportError:
    from src.twin_stub import predict_validity
    from src.regress_stub import predict_reference
    USING_STUBS = True
```

Print a loud warning whenever stubs are active. When the real modules land,
nothing else in your code changes. Delete the stubs before final submission.

---

## Required deliverables

### 1. `outputs/final/<TEAM_NAME>.csv`

Exactly these columns, in this order:

```
Test_ID,Predicted_Reference_Parameter,Validity_Label
```

- Exactly **350** rows
- `Test_ID` values and order must match the `Sample_Submission` sheet exactly
- `Predicted_Reference_Parameter` finite, no NaN, no infinity
- `Validity_Label` contains only the strings `Valid` and `Invalid`
- No pandas index column

Team name comes from `--team-name` on the command line, defaulting to a constant
at the top of `run.py`.

### 2. `outputs/final/summary.json`

```json
{
  "records_analysed": 350,
  "abnormal_invalid_records": 0,
  "minimum_predicted_reference_parameter": 0.0,
  "maximum_predicted_reference_parameter": 0.0,
  "average_predicted_reference_parameter": 0.0,
  "highest_attention_test_ids": ["ID1", "ID2", "ID3"],
  "approach_explanation": "Maximum 100 words."
}
```

Every value computed from the final predictions — no hardcoding. Enforce the
100-word limit programmatically and fail the build if it is exceeded.

### 3. The attention score

Pick the three Test IDs most deserving human review. Combine, each normalised to
0–1 and equally weighted unless you document otherwise:

- Whether the record was flagged Invalid
- Severity of the sensor residual, relative to the threshold
- Missing-measurement count
- Percentile rank of the predicted Reference Parameter (hottest tests matter most)

Document the formula in `README.md`. Write per-record components to
`outputs/final/test_diagnostics.csv` so the choice is auditable.

### 4. `run.py`

One command, start to finish, zero manual steps:

```bash
python run.py --team-name YourTeamName
```

Should also support `--output-dir` and `--skip-charts`. Print a clear progress log
and a final summary of what was written.

### 5. `tests/`

Pytest suite covering:

- Submission has 350 rows and the exact required columns
- Test IDs match `Sample_Submission` in the same order
- No NaN or infinite predictions
- Labels are only `Valid` / `Invalid`
- `summary.json` values agree with the CSV — invalid count, min, max, mean
- The explanation is 100 words or fewer
- Identical measurement vectors in the test set receive identical outputs
- No duplicate or missing Test IDs

### 6. `README.md` and `requirements.txt`

README covers: what this is, how to install, how to run, what each output is, the
attention-score formula, and the repository layout. Pin versions in
`requirements.txt` so the judges get what we got.

---

## Integration QA — your second job

This protects the 20% hidden-dataset criterion. Write
`tests/test_robustness.py` that runs the full pipeline against deliberately
damaged copies of the test data and asserts it never crashes and always emits 350
valid rows:

| Attack | Expectation |
|---|---|
| Randomly NaN 10% of sensor readings | Runs, produces 350 rows |
| Drop the `Sensor_S4` column entirely | Runs, verdicts unchanged |
| Duplicate 20 random rows | Runs, duplicates flagged Invalid |
| Multiply all currents by 1.3 (beyond training range) | Runs, predictions stay finite and sensible |
| Insert a row of all-zero sensors | Runs, flagged Invalid |
| Shuffle the column order | Runs, identical output |

Report anything that breaks to Person 1 or Person 2 in your final report.
**Do not fix their files yourself.**

---

## Tasks

1. Read `src/contract.py` and `briefs/00_START_HERE.md`.
2. Create the stubs and get an end-to-end run working today.
3. Build `src/finalize.py` — submission writer, summary builder, attention score,
   validators.
4. Build `run.py` as the single entry point.
5. Write the pytest suite, including robustness.
6. Write `README.md` and pin `requirements.txt`.
7. Verify on a clean clone in a fresh virtual environment.
8. Save console output to `verification/p4_report.txt`.

---

## Acceptance tests — all must pass

- [ ] `python run.py --team-name TestTeam` completes with exit code 0
- [ ] `outputs/final/TestTeam.csv` has exactly 350 rows and the exact three columns
- [ ] Test IDs match `Sample_Submission` exactly, in order
- [ ] No NaN, no infinity anywhere in the CSV
- [ ] Labels are only `Valid` / `Invalid`
- [ ] `summary.json` has all seven keys with correct types
- [ ] Summary statistics match the CSV when recomputed independently
- [ ] Explanation is 100 words or fewer, checked in code
- [ ] Exactly three attention IDs, all present in the test data
- [ ] `pytest tests/ -v` passes with zero failures
- [ ] All six robustness attacks pass
- [ ] Fresh clone into a clean virtualenv: `pip install -r requirements.txt && python run.py` works
- [ ] No file outside your ownership list was modified

If an acceptance test fails, fix it. Do not weaken the test.

---

## Do not

- Do not edit `src/twin.py`, `src/regress.py`, `src/benchmark.py`, `src/contract.py`, `folds.csv`, `methodology.md`
- Do not fix bugs in someone else's module — report them instead
- Do not hardcode any value in `summary.json`
- Do not write the pandas index into the CSV
- Do not change Test ID order
- Do not let the pipeline silently swallow an exception — fail loudly with a clear message
- Do not commit `outputs/final/` artifacts as if they were final while stubs are active

## If you get stuck

| Problem | Fallback |
|---|---|
| `src/twin.py` or `src/regress.py` not ready | Keep using stubs, mark output clearly as provisional |
| Real module has a different signature than the brief specified | Adapt with a thin wrapper, report the mismatch |
| A robustness attack crashes inside someone else's module | Record it, mark the test `xfail` with a comment naming the owner, continue |
| Fresh-clone test fails on a missing dependency | Add it to `requirements.txt`, rerun |

---

## Final report

1. Confirmation `run.py` completes end to end and whether stubs were active
2. Submission file validation results, line by line
3. `summary.json` contents
4. The three attention Test IDs and why they were chosen
5. Pytest output summary
6. Robustness results for all six attacks
7. Fresh-clone verification result
8. Any bug found in another person's module — describe precisely, do not fix
