# π-thon — Methodology Note

**Team:** π-thon | **Task:** CPRI Hackathon Screening — validity classification and Reference Parameter regression | **Seed:** 42 | **Validation:** 5-fold StratifiedGroupKFold, duplicate-grouped, fixed in `folds.csv`

---

## 1. Approach

We modelled how the equipment physically behaves rather than fitting a black box to the labels. The dataset has 1,000 training rows — 866 Valid (86.6%) and 134 Invalid (13.4%). Two tasks run in parallel:

- **Classification** — detect Invalid measurements using a physics-informed digital twin whose residuals trigger deterministic fault rules. ML classifiers are benchmarked alongside it as evidence, not as the product.
- **Regression** — predict Reference_Parameter (range: 11.9–61.6, mean: 26.7) from the four operating conditions alone, trained exclusively on the 866 Valid rows to avoid corrupted targets. The shipped model is a weighted blend of a degree-4 polynomial Ridge and a HistGradientBoosting regressor, achieving **out-of-fold MAE 0.4578, RMSE 0.6575, R² 0.99625** on the contract folds (worst fold 0.5016).

Both tasks are evaluated on identical 5-fold splits stored in `folds.csv`. The group-aware split ensures no duplicate measurement vector straddles a fold boundary, making OOF metrics honest.

---

## 2. Parameters Considered Important

**Control settings (inputs):** Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min. These four drive equipment behaviour. Benchmark regression on operating inputs alone achieves R² = 0.961 (HistGBR), confirming they carry strong signal.

**Monitored sensors (outputs, not inputs):** Sensor_S1, S2, S3 are treated as measurements of equipment response, not causal drivers. The twin models the expected sensor readings given the operating conditions; deviations from those expectations are the anomaly signal.

**Sensor_S4 is excluded.** Max |ρ(S4, Reference_Parameter)| = 0.071, and including it in the ablation changes MAE by < 0.002. The exclusion is data-driven, not a manual choice.

Test_Duration_min is retained: removing it in the ablation study degrades regression MAE by approximately 0.12, despite its near-zero linear correlation with the target — a classic non-linear interaction captured by tree ensembles and polynomial expansion.

---

## 3. Method for Detecting Abnormal Data

**Digital twin architecture:** For each of S1, S2, S3 we fit a polynomial Ridge regression (degree 4, α = 1.0) on operating inputs using Valid training rows only. Predictions represent the physically expected sensor reading.

**Residuals → robust Z-score:** Each residual is normalised with a robust Z-score (median-centred, 1.4826 × MAD scaled) to handle non-Gaussian sensor noise without being pulled by the very spikes we are trying to detect.

**Adaptive threshold:** In each fold we measure the empirical gap between the highest Valid spike score and the lowest Invalid one, and place the threshold at its midpoint — so it self-calibrates rather than being tuned by hand. Across the five folds that gap spans 17.96–21.73, far wider than any plausible noise perturbation, so the decision is stable.

**Fault-rule precedence (deterministic, no ML):**
1. missing_sensor — any of S1, S2, S3 is NaN (15 cases)
2. duplicate_vector — 8-column fingerprint (4 dp, pipe-joined) appears > 1 time in the batch (24 cases)
3. sensor_spike — max robust-Z across S1/S2/S3 exceeds threshold (95 cases)
4. → **Valid**

**OOF result:** Precision, recall and F1 all 1.0 across the 5 folds. The twin separates a genuine high-load regime — all three sensors deviating consistently with the operating physics — from a sensor fault, where one deviates while the others stay on-curve.

---

## 4. Assumptions

- Valid-row Reference_Parameter values are engineer-verified ground truth; Invalid-row targets are corrupted and are excluded from regression training.
- Sensor faults affect one sensor at a time or appear as complete-row duplicates; the precedence rules are designed accordingly.
- The second (hidden) test dataset comes from the same equipment under similar operating conditions — no distributional shift. Verified: no train/test drift on any of the 8 measurement columns.
- Missing sensor values (15 cases total in training) are informative of invalidity, not missing at random.

---

## 5. Limitations

- **Extrapolation beyond the training hull.** At 1.5× the maximum training current and voltage the raw blend predicts 140.8 °C, which is not physical. Shipped predictions are clipped to the training range widened 20% ([1.99, 71.52] °C). No test row needed clipping; the guard exists for the second hidden dataset. Residuals outside the envelope (10–25 kV, 50–150 A) are also less well calibrated, so the spike rule is less reliable there.
- **The threshold assumes a comparable noise floor.** A sensor recalibration event on the live system would shift that floor and require re-deriving the threshold from fresh reference tests.
- **Duplicate detection is batch-scoped.** The rule needs the duplicated fingerprint to appear twice within the batch being scored; a lone duplicate falls through to the spike check.
- **We declined the missing-S4 shortcut.** Every training row with S4 missing is Valid, but treating that as an invalidity signal would conflate instrument absence with sensor fault and would not generalise.
- **The blend carries two models.** We accept that cost: OOF MAE 0.4578 against 0.5095 for the polynomial alone, a 10% gain on the criterion worth 35%. The polynomial alone is fitted inside the same artefact where interpretability matters more.

## 6. Digital Twin Deployment Steps

The twin is operational, not hypothetical. To run it live:

1. **Ingest** the four operating controls and the three sensor readings for each completed test.
2. **Evaluate the twin** — the polynomial Ridge models, fitted on the 866 Valid training rows, predict expected S1, S2, S3.
3. **Compute residuals as robust Z-scores**, using the training median and MAD per channel, stored alongside the model artefact.
4. **Compare to threshold** — `spike_score = max(Z_S1, Z_S2, Z_S3)`; above the fitted threshold, flag `sensor_spike`.
5. **Apply precedence** — missing values (`missing_sensor`) and fingerprint duplicates (`duplicate_vector`) are checked before the spike rule.
6. **Raise a named alarm** identifying the fault bucket and the specific suspect sensor, e.g. `sensor_spike: S2 (Z=18.4)`, so the operator knows where to look.
7. **Log for recalibration** — all scores and thresholds go to an audit log; quarterly, the threshold is re-derived from newly confirmed-Valid tests by the same empirical-gap algorithm.

**Measured false-alarm rate:** 0 false positives across all 866 Valid training rows. Noise stress tests show zero false positives at every magnitude tested, and full detection once added noise reaches roughly 20× the training noise floor.

---

*Benchmarks: src/benchmark.py | Shipped models: src/twin.py, src/regress.py | Seed: 42*
