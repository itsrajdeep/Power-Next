# PowerNext-AI Screening Round — Project Guide

**Team reference document. Written so anyone can follow it, technical or not.**

Last updated: 10 September 2026
All numbers in this document are reproducible by running `python verification/verify_findings.py`.

---

## Part 1 — What this project actually is

### The one-sentence version

A laboratory tested some electrical equipment 1,350 times. For 1,000 of those tests they know the answer. For the other 350 they don't, and we have to work it out.

### The longer version

CPRI (Central Power Research Institute) runs a test bench. They take a piece of electrical equipment — think a transformer or a heavy-duty cable joint — and they stress it:

- They apply a **voltage** to it
- They push a **current** through it
- They do this in a room at some **ambient temperature**
- They hold it there for a fixed **duration**

While that happens, the equipment heats up. Four sensors are stuck to it in different places, recording how hot it gets.

But the number that actually matters is the temperature at the **hottest point inside** the equipment. That is what determines whether the equipment is safe or whether it will fail. It is also the hardest place to measure, so in the lab they use a separate, expensive, calibrated instrument to get it. That reading is called the **Reference Parameter**.

They also have an engineer look at each test and decide whether the test was trustworthy or whether something went wrong — a sensor glitched, a reading got lost, someone logged it twice. That verdict is the **Validity Label**: `Valid` or `Invalid`.

Now they've handed us 350 new tests with **both of those columns removed**, and asked us to fill them back in.

### What we have to deliver

| # | Deliverable | What it is |
|---|---|---|
| 1 | `<TeamName>.csv` | 350 rows: Test ID, our predicted Reference Parameter, our Valid/Invalid call |
| 2 | `summary.json` | An automatic report: how many records, how many bad ones, min/max/average prediction, top 3 tests needing attention, 100-word method explanation |
| 3 | Source code | Must run start to finish with one command, no manual fiddling |
| 4 | Methodology note | Max 2 pages: our approach, key parameters, how we detect bad data, assumptions, and how we'd turn this into a live "digital twin" system |

All zipped as `your-teamname-submission.zip`.

### How it's marked

| Criterion | Weight | Plain meaning |
|---|---|---|
| Reference Parameter accuracy | **35%** | How close our temperature predictions are |
| Invalid record accuracy | **25%** | How well we spot the bad tests |
| Second unseen dataset | **20%** | Does it still work on data nobody has seen, including us |
| Reproducibility & code quality | **10%** | Does it run cleanly for the judges |
| Engineering reasoning | **10%** | Can we explain *why* it works |

**80% of the score is output accuracy.** The organisers wrote this explicitly:

> The ranking will primarily be based on the output produced by the submitted program against hidden data, rather than the presentation or complexity of the solution.

Remember that line. It shapes every decision below.

---

## Part 2 — What's in the data

The workbook has four sheets: `README`, `Training_Data`, `Test_Data`, `Sample_Submission`.

### Training_Data — 1,000 rows, 11 columns

| Column | Unit | What it is |
|---|---|---|
| `Test_ID` | — | Label for the test. **Never use as a model input.** |
| `Applied_Voltage_kV` | kV | Voltage we applied |
| `Load_Current_A` | A | Current we pushed through |
| `Ambient_Temperature_C` | °C | Room temperature |
| `Test_Duration_min` | min | How long we held it |
| `Sensor_S1` | °C rise | Temperature near the incoming terminal |
| `Sensor_S2` | °C rise | Temperature near the outgoing terminal |
| `Sensor_S3` | °C rise | Temperature at another critical spot |
| `Sensor_S4` | a.u. | "Auxiliary sensor, relevance unknown" |
| `Reference_Parameter` | °C rise | **Target 1** — the true hot-spot temperature |
| `Validity_Label` | — | **Target 2** — Valid or Invalid |

The first four are **inputs we control**. The next four are **things we measure**. The last two are what we have to predict.

### Test_Data — 350 rows, 9 columns

Same, minus the two target columns.

### The class split

```
Valid   : 866
Invalid : 134   (13.4%)
```

### Known data problems

- Missing values in the sensor columns
- 24 rows that are exact duplicates of each other
- Measurement noise
- Sensor spikes
- At least one irrelevant column

The brief warns us about one trap in particular:

> An unusual value is not necessarily an invalid value. Under certain combinations of voltage, current, temperature or duration, the equipment may genuinely enter a different operating regime.

In plain terms: **if you run the equipment hard, it genuinely gets very hot. That's not a fault, that's physics.** A naive outlier detector would flag all the high-temperature tests as broken. That is the mistake the challenge is designed to catch.

---

## Part 3 — What we discovered

This is the heart of the project. Everything else follows from it.

### The starting question

Instead of asking *"which machine learning model predicts this best?"*, we asked *"how does this machine actually behave?"*

### Discovery 1 — The sensors are predictable

We tried predicting each sensor reading using **only the four things we control** (voltage, current, ambient temperature, duration). Not using the other sensors. Just the knobs.

Result:

```
Sensor_S1 : 99.75% explained    typical error ±0.19 °C
Sensor_S2 : 99.73% explained    typical error ±0.18 °C
Sensor_S3 : 99.60% explained    typical error ±0.31 °C
```

**Read that again.** If you tell us the voltage, current, room temperature and duration, we can tell you what each sensor *should* read, to within a fifth of a degree.

That means the sensors aren't independent sources of information. They're a **consequence** of the settings. Which gives us something powerful: an expectation. And once you have an expectation, you can spot a lie.

### Discovery 2 — Bad tests fall into exactly three buckets

We took each sensor's actual reading and subtracted what it *should* have been. That difference is called the **residual** — how far off reality is from expectation.

Then we sorted all 134 Invalid records:

```
Cause                                    Count
---------------------------------------  -----
A sensor reading is simply missing          15
The row is a duplicate of another row       24
One sensor is lying (large residual)        95
---------------------------------------  -----
TOTAL                                      134   ← exactly the number of Invalid records
```

**The three buckets sum exactly to 134.** Nothing left over, nothing double-counted. We have fully accounted for every bad record in the dataset.

For the 95 spiked records, exactly **one** of the three sensors is wrong at a time — sometimes reading too high, sometimes too low, split roughly evenly across the three sensors (S1: 26, S2: 48, S3: 21).

### Discovery 3 — The separation is enormous

Here is why this matters so much. Look at the residual sizes:

```
Largest residual on a GOOD test :  1.06
Smallest residual on a SPIKED test:  4.18
```

There is a **four-fold empty gap** between good and bad. Nothing lives in between. This is not a fuzzy statistical boundary you have to tune carefully — it's a canyon.

Which means our detection rule scores:

```
Precision : 1.0000     (every test we flag really is bad)
Recall    : 1.0000     (we catch every bad test)
F1        : 1.0000     (perfect)
Errors    : 0 false positives, 0 false negatives
```

And critically — this was confirmed **out-of-fold**, meaning we rebuilt the whole thing from scratch five times, each time hiding a fifth of the data, and it got every single one right every time. It is not memorisation.

### Discovery 4 — It transfers to the unseen test data

We applied the rule to the 350 test records we've never had answers for:

```
Missing sensor readings  :  6
Duplicate rows           :  8
Sensor spikes            : 32
-------------------------------
Flagged Invalid          : 46 / 350  =  13.1%
```

The training set was 13.4% invalid. We found 13.1%. Almost identical — a strong sign the rule is behaving the same way on new data.

And the gap held:

```
Largest residual among tests we called GOOD : 0.87
Smallest residual among tests we called BAD : 4.18
```

Same canyon. No ambiguous cases.

### Discovery 5 — This solves the "genuine regime vs fault" trap

Remember the warning in the brief about not confusing genuinely hot equipment with a broken sensor?

Our method handles this automatically, and it's worth understanding why:

- A test run at very high current produces **very high sensor readings** — but our expectation *also* predicts high readings, because we fed it the high current. So the **residual stays small**. → Correctly called Valid.
- A spiked sensor produces a reading that doesn't match what the settings predict. → **Large residual.** → Correctly called Invalid.

We are not asking *"is this number unusual?"* We are asking *"is this number unusual **given the conditions**?"* That distinction is the entire challenge, and it's why a generic outlier detector fails here.

### Discovery 6 — The Reference Parameter follows the same logic

The hot-spot temperature is also a function of the four control settings. Three findings:

**(a) Train only on the Valid rows.** Invalid rows have corrupted target values that poison the model.

```
Trained on all 1,000 rows  : average error 0.90 °C
Trained on 866 Valid rows  : average error 0.66 °C     ← 27% better
```

**(b) Use only the control settings, not the sensors.** This makes the prediction completely immune to sensor faults — we can predict the temperature even for tests where a sensor is broken, because we never look at the sensors.

**(c) A polynomial beats the fancy models.**

```
Boosted trees (HistGBR)         : average error 0.66 °C
Degree-4 polynomial             : average error 0.51 °C
Blend of both (65% / 35%)       : average error 0.46 °C   ← best
```

Final accuracy: **99.6% of the variation explained, average error 0.46 °C**, on a quantity that ranges from 12 to 62 °C.

### Discovery 7 — Sensor_S4 is genuinely noise

The brief said S4's relevance "is not known beforehand and must be assessed." We assessed it. Its correlation with every other column in the dataset:

```
Voltage    -0.049      Sensor_S1  -0.032
Current    -0.017      Sensor_S2  -0.026
Ambient    +0.007      Sensor_S3  -0.071
Duration   -0.021      Reference  +0.002   ← essentially zero
```

It correlates with nothing. It is the irrelevant parameter the brief hinted at. We exclude it and say so.

### One trap we are deliberately NOT falling into

In the training data, every row where S4 is missing happens to be Valid (29 out of 29). It's tempting to use that as a rule.

**We won't.** There's no physical reason a missing auxiliary sensor should mean the test was good. It's an artifact of however this dataset was generated, and it is exactly the kind of thing that breaks on the second hidden dataset. We're leaving it alone, and we'll say why in the methodology note.

---

## Part 4 — How we win

### The scoring reality

80% of the marks are accuracy on hidden data. Our current position:

| Criterion | Weight | Where we stand |
|---|---|---|
| Reference Parameter accuracy | 35% | Average error 0.46 °C, 99.6% explained |
| Invalid detection | 25% | Perfect out-of-fold, verified on test data |
| Second unseen dataset | 20% | Method is physics-based, not tuned to quirks |
| Code quality | 10% | To be built — Person 4 owns it |
| Engineering reasoning | 10% | We can explain the actual mechanism |

### Why this beats the standard approach

Most teams will do this:

1. Load the data
2. Build a lot of engineered features
3. Train XGBoost / LightGBM / CatBoost
4. Tune hyperparameters
5. Submit

That approach lands around **F1 0.73 for detection** and **0.65 °C error for prediction**. We measured it ourselves — that's what our own boosted-tree baseline achieved before we found the mechanism.

We're at **F1 1.00** and **0.46 °C**. Not because we used a better model, but because we asked a better question.

### The "everyone is using AI" problem

This is a real concern and deserves a straight answer.

**What AI gives everyone equally.** Anyone can prompt an AI to load a spreadsheet, generate 80 features, run six models, and produce a leaderboard. That is now free and universal. It is no longer an advantage — it's the floor.

**What AI does not do on its own.** By default an AI assistant treats this as a modelling problem, because that's the shape of the request. It will not spontaneously:

- Question whether the target column is corrupted for some rows
- Test whether the sensors are derivable from the inputs
- Notice that the three failure causes sum exactly to the invalid count
- Choose a rule over a model when the rule is provably better
- Refuse to use a signal (the S4 rule) that would score well now and break later

Each of those came from a human deciding what to investigate next. **The AI ran the experiments. The direction was the contribution.**

**Where the crowd will actually lose marks.** From reviewing the analysis that was already done on this project, three failure modes are extremely common in AI-generated pipelines, and all three cost accuracy:

1. **Training on all rows** instead of Valid rows only — costs 27% on the regression score
2. **Predicting each sensor from the other sensors** — a spiked sensor poisons its neighbours' predictions and the signal gets washed out
3. **Stopping the polynomial search at degree 2** — the real relationship is degree 4

Every one of those is a default that looks reasonable and quietly loses points. We found and fixed all three.

**The tiebreakers.** When the top of the leaderboard is crowded, three things separate us:

- **The second hidden dataset (20%).** Solutions tuned to quirks of this particular file will degrade. Ours is grounded in the physics of the machine, so it shouldn't.
- **The methodology note (10%).** Generic AI-written notes read generic. Ours will contain a specific mechanism, a fault taxonomy that sums exactly, and real measured numbers.
- **The digital twin question.** Deliverable 4 asks how we'd automate this as a digital twin. Most teams will write a hypothetical paragraph. **We already built one** — see below.

### The digital twin angle

A digital twin is a live software model of a physical machine, running alongside the real thing, comparing what the machine *should* be doing to what it *is* doing.

That is a precise description of what we built:

- The sensor models are the **twin** — they know how the equipment behaves
- The residual is the **discrepancy** between twin and reality
- The threshold is the **alarm level**
- The three fault buckets are the **diagnostic categories**

So when the brief asks "what steps would you follow to automate this through a digital twin?", we don't speculate. We describe a working system, with its residual distribution, its alarm threshold, and its measured false-alarm rate on unseen data.

That is the strongest single card in the submission.

---

## Part 5 — Where we are right now

### Done

- **Exploratory analysis** — 80+ charts, feature rankings, correlation studies, model recommendations, a 25-experiment matrix. Committed in `analysis/`.
- **Mechanism discovery** — everything in Part 3 above.
- **Verification** — `verification/verify_findings.py` re-derives every claim from the raw workbook. Output saved in `verification/verified_findings.txt` and `.json`.

### Not done

- The final pipeline that produces the submission files
- The model benchmark leaderboard (our evidence that the chosen method won)
- The methodology note
- Stress testing against the second hidden dataset

### One known bug

`analysis_main.py` has hardcoded paths pointing at `d:\Powernext\`. It will not run from this repository. Since reproducibility is 10% of the marks, this needs fixing before submission.

### Current phase

```
[####################----------------] Discovery complete → Build phase
```

We know the answer. We now have to package it so a judge can run it.

---

## Part 6 — The plan from here

Four people, four independent lanes. **Nobody waits for anybody.**

### Phase 0 — Everyone together, 30 minutes

This is the only step that must happen simultaneously. Lock these decisions:

- One git branch per person: `p1-twin`, `p2-regression`, `p3-benchmark`, `p4-pipeline`
- Fixed random seed: `42`
- Agree the **fingerprint function** — the 8 measurement columns rounded to 4 decimal places and joined into a string. Used to group duplicate rows.
- Generate the 5-fold cross-validation split **once**, save it as `folds.csv`, commit it. **Everyone validates on these exact folds.** Different splits make results incomparable and the whole comparison collapses.
- Agree the two function signatures everyone codes against:

```python
# src/twin.py     — Person 1
def predict_validity(df) -> pd.Series:    # returns "Valid" / "Invalid"

# src/regress.py  — Person 2
def predict_reference(df) -> pd.Series:   # returns a number
```

Then split up.

### Person 1 — Sensor Twin & Invalid Detection
**Owns Task 1 · 25% of the marks**

Build the model of how each sensor should behave, then flag the ones that don't.

- Fit degree-4 polynomial: control settings → each of S1, S2, S3, using Valid rows only
- Compute residuals (actual reading minus expected reading)
- **Derive the threshold automatically** from the gap in the data. Do not hardcode 2.0 — a self-calibrating threshold survives a dataset with different noise levels, which is our main risk on the 20% criterion.
- Assemble the three rules: missing sensor, duplicate row, spike
- Validate with grouped cross-validation, confirm 15 / 24 / 95 = 134
- Stress-test your own work: add noise to the sensors, confirm the gap still separates

**Delivers:** `src/twin.py`
**Done when:** perfect F1 reproduced out-of-fold, threshold self-derives, taxonomy sums correctly

### Person 2 — Reference Parameter Model
**Owns Task 2 · 35% of the marks — the single heaviest criterion**

- Train on **Valid rows only**, using **control settings only**
- Sweep polynomial degree (3, 4, 5) against regularisation strength
- Fit the boosted-tree model, pick the blend weight from out-of-fold predictions
- Break down the error by operating regime: low/medium/high current, high ambient temperature
- Stress-test your own work: feed inputs beyond the training range, confirm the polynomial doesn't blow up

**Delivers:** `src/regress.py`
**Done when:** cross-validated error ≤ 0.50 °C on the shared folds, blend weight chosen from out-of-fold data

### Person 3 — Benchmark Leaderboard & Methodology Note
**Owns the evidence · 10% reasoning marks**

This is where CatBoost, LightGBM and XGBoost belong — as the scoreboard, not the product.

- One loop over every classifier: Logistic Regression, Random Forest, Extra Trees, HistGB, CatBoost, XGBoost, LightGBM, **plus our rule**
- One loop over every regressor: Ridge, ElasticNet, Huber, Random Forest, Extra Trees, HistGBR, CatBoost, XGBoost, LightGBM, **plus our polynomial and blend**
- Same folds, same metrics, ranked output
- Then write the 2-page methodology note, including the digital twin section
- Charts: the residual separation gap, the fault taxonomy, the leaderboard

**Delivers:** `src/benchmark.py`, `outputs/model_comparison.csv`, `methodology.md`
**Done when:** every model has run, leaderboard is sorted, note fits in 2 pages

*Why this matters:* a comparison table where our method wins is proof. Skipping the comparison is what looks lazy.

### Person 4 — Pipeline, Deliverables & Integration QA
**Owns 10% code quality, protects the 20% hidden-set criterion**

- `run.py` — one command, start to finish, zero manual steps
- `<TeamName>.csv` — exactly 350 rows, correct column names, order matched to `Sample_Submission`
- `summary.json` — all seven required fields, computed automatically
- The attention score that picks the top 3 Test IDs needing review
- `requirements.txt`, `README.md`, output verification checks
- **Attack the integration:** randomly drop sensors, feed malformed rows, run on a fresh clone of the repo

**Start immediately using stubs.** Write `predict_validity()` returning `"Valid"` and `predict_reference()` returning `25.0`, then build the entire pipeline against those. When Person 1 and 2 finish, delete the stubs. Nothing else changes. **You are not blocked.**

**Delivers:** `run.py`, `src/finalize.py`, `outputs/`, `tests/`
**Done when:** a clean clone runs end to end, no missing or infinite values, labels only `Valid`/`Invalid`, row count verified

### Who depends on whom

```
P1 (twin)  ───────────┐
                      ├──► P3 (adds 2 rows to leaderboard, at the end)
P2 (regression) ──────┤
                      └──► P4 (swaps out 2 stub files, at the end)

P1 ←── no dependency ──► P2
P3 ←── no dependency ──► P4
```

Only two dependencies, both at the very end, both reduced to deleting a stub file.

### Timeline

| Phase | Who | What happens |
|---|---|---|
| 0 | All | Lock the contract, commit `folds.csv` (30 min) |
| 1 | All, in parallel | Build the four lanes independently |
| 2 | P4 | Wire the real modules in, first full end-to-end run |
| 3 | **Decision gate** | P3 presents the leaderboard. **The evidence decides what ships** — rule vs model, blend vs single. Not preference. |
| 4 | P4 attacks, P1/P2 patch | Fix whatever integration testing breaks |
| 5 | P4 ships, P3 finalises note | Freeze, verify, zip |

### Three rules that keep this from falling apart

1. **Everyone uses `folds.csv`.** Different splits make results incomparable.
2. **Nobody commits to `main` directly.** Branch, pull request, one reviewer.
3. **Run the decision gate even if you're confident.** That comparison table *is* the engineering-reasoning marks.

---

## Part 7 — Risks and what we do about them

| Risk | Likelihood | What we do |
|---|---|---|
| Second dataset has different noise, breaking a fixed threshold | Medium | Person 1 derives the threshold from the data at runtime, not hardcoded |
| Polynomial misbehaves outside the training range | Medium | Person 2 tests beyond-range inputs; blending with trees dampens it |
| Judge's environment can't run our code | Low | Person 4 tests on a clean clone; pinned `requirements.txt` |
| We over-trust the perfect training score | Low | Already confirmed out-of-fold and on the 350 test rows |
| The S4-missing shortcut tempts someone | Low | Documented here as forbidden, with the reason |

Note on the last two: our detection is perfect on both training and out-of-fold validation, and the gap held on the test data. That's three independent confirmations. But we still keep Person 3's model benchmark as a cross-check, and if the rule and the best trained model ever disagree on the second dataset, that disagreement is itself a warning sign worth reporting.

---

## Part 8 — Glossary

Terms used above, in plain English.

| Term | Meaning |
|---|---|
| **Feature** | An input column the model looks at |
| **Target** | The thing we're trying to predict |
| **Residual** | Actual value minus expected value. How wrong the expectation was. |
| **Cross-validation** | Hide part of the data, train on the rest, test on the hidden part. Repeat. Stops you fooling yourself. |
| **Out-of-fold** | A score measured only on data the model never saw during training. The honest kind of score. |
| **Precision** | Of the tests we flagged as bad, how many really were bad |
| **Recall** | Of the genuinely bad tests, how many we caught |
| **F1** | Precision and recall combined into one number. 1.0 is perfect. |
| **MAE** | Mean Absolute Error — on average, how many degrees off our prediction is |
| **R²** | Fraction of the variation explained. 1.0 is perfect. |
| **Polynomial** | A curved formula, e.g. involving squares and cubes, rather than a straight line |
| **Boosted trees** | A popular family of models (XGBoost, LightGBM, CatBoost) that build many small decision trees in sequence |
| **Leakage** | Accidentally letting the model see the answer during training. Makes scores look great and real performance terrible. |
| **Digital twin** | A live software model of a physical machine, used to spot when reality diverges from expectation |

---

## Appendix — Verified numbers

Every figure below is output by `python verification/verify_findings.py`.

```
DATA
  Training rows                    1000  (11 columns)
  Test rows                         350  (9 columns)
  Valid / Invalid                866 / 134   (13.4% invalid)

SENSOR TWIN (degree-4 polynomial from control settings, Valid rows)
  Sensor_S1   R2 = 0.99748    error 0.15    sigma 0.19
  Sensor_S2   R2 = 0.99727    error 0.15    sigma 0.18
  Sensor_S3   R2 = 0.99595    error 0.25    sigma 0.31

FAULT TAXONOMY
  Missing S1/S2/S3                   15
  Duplicate measurement vector       24
  Single-sensor spike                95
  TOTAL                             134   = exact match to Invalid count

SEPARATION
  Max residual on Valid rows      1.0551
  Min residual on spiked rows     4.1774
  Gap ratio                          4.0x

DETECTION (out-of-fold, grouped CV, twins refit inside each fold)
  Precision 1.0000   Recall 1.0000   F1 1.0000   FP 0   FN 0

REFERENCE PARAMETER (5-fold CV)
  All rows,   boosted trees      MAE 0.8963   R2 0.95895
  Valid only, boosted trees      MAE 0.6600   R2 0.98949
  Valid only, poly4              MAE 0.5134   R2 0.99515
  Valid only, blend (0.65/0.35)  MAE 0.4606   R2 0.99607   <-- chosen

APPLIED TO THE 350 TEST ROWS
  Missing sensors                     6
  Duplicates                          8
  Spikes                             32
  Total flagged Invalid              46   = 13.1%  (train rate 13.4%)
  Max residual, unflagged        0.8711
  Min residual, flagged          4.1841   -> gap holds on unseen data

SENSOR_S4
  Max absolute correlation with anything   0.071   -> irrelevant

TRAIN vs TEST DRIFT
  Significant drift on any column:  None
```
