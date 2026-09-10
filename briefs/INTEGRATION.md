# How the four lanes fit together

Standalone copy of the integration section that is already embedded in every
P1-P4 brief. Useful for humans; agents do not need it separately.
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
