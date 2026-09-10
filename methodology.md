# Power-Next: Methodology Note

**Project:** CPRI Hackathon Screening — Sensor Validity Classification & Reference Parameter Regression  
**Branch:** p3-benchmark | **Seed:** 42 | **Folds:** 5-fold StratifiedGroupKFold (folds.csv, committed once)

---

## 1. Approach

We modelled how the equipment physically behaves rather than fitting a black box to the labels. The dataset has 1,000 training rows — 866 Valid (86.6%) and 134 Invalid (13.4%). Two tasks run in parallel:

- **Classification** — detect Invalid measurements using a physics-informed digital twin whose residuals trigger deterministic fault rules. ML classifiers are benchmarked alongside it as evidence, not as the product.
- **Regression** — predict Reference_Parameter (range: 11.9–61.6, mean: 26.7) from operating conditions, trained exclusively on the 866 Valid rows to avoid corrupted targets.

Both tasks are evaluated on identical 5-fold splits stored in olds.csv. The group-aware split ensures no duplicate measurement vector straddles a fold boundary, making OOF metrics honest.

---

## 2. Parameters Considered Important

**Control settings (inputs):** Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min. These four drive equipment behaviour. Benchmark regression on operating inputs alone achieves R² = 0.961 (HistGBR), confirming they carry strong signal.

**Monitored sensors (outputs, not inputs):** Sensor_S1, S2, S3 are treated as measurements of equipment response, not causal drivers. The twin models the expected sensor readings given the operating conditions; deviations from those expectations are the anomaly signal.

**Sensor_S4 is excluded.** Verified correlation: max |ρ(S4, Reference_Parameter)| = 0.071. Including S4 in the ablation study changes MAE by < 0.002 — confirming it carries no predictive signal and adding it risks noise. The decision is data-driven, not manual.

Test_Duration_min is retained: removing it in the ablation study degrades regression MAE by approximately 0.12, despite its near-zero linear correlation with the target — a classic non-linear interaction captured by tree ensembles and polynomial expansion.

---

## 3. Method for Detecting Abnormal Data

**Digital twin architecture:** For each of S1, S2, S3 we fit a polynomial Ridge regression (degree 4, α = 1.0) on operating inputs using Valid training rows only. Predictions represent the physically expected sensor reading.

**Residuals → robust Z-score:** Each residual is normalised with a robust Z-score (median-centred, 1.4826 × MAD scaled) to handle non-Gaussian sensor noise without being pulled by the very spikes we are trying to detect.

**Adaptive threshold:** In each fold we compute the empirical gap between the upper boundary of Valid spike scores and the lower boundary of Invalid spike scores. The threshold is placed at the midpoint of this gap, making it self-calibrating. Across 5 folds the gap is 17.96–21.73 (mean: 18.93) and the threshold is 12.62–14.60 (mean: 12.92). The gap is large enough that small noise perturbations do not alter the decision.

**Fault-rule precedence (deterministic, no ML):**
1. missing_sensor — any of S1, S2, S3 is NaN (15 cases)
2. duplicate_vector — 8-column fingerprint (4 dp, pipe-joined) appears > 1 time in the batch (24 cases)
3. sensor_spike — max robust-Z across S1/S2/S3 exceeds threshold (95 cases)
4. → **Valid**

**OOF result:** Precision = 1.0, Recall = 1.0, F1 = 1.0 across all 5 folds. The twin distinguishes a genuine high-load regime (all three sensors deviate consistently with the expected operating physics) from a sensor fault (one sensor deviates while the others remain on-curve).

---

## 4. Assumptions

- Valid-row Reference_Parameter values are engineer-verified ground truth; Invalid-row targets are corrupted and are excluded from regression training.
- Sensor faults affect one sensor at a time or appear as complete-row duplicates; the precedence rules are designed accordingly.
- The second (hidden) test dataset comes from the same equipment under similar operating conditions — no distributional shift. Verified: no train/test drift on any of the 8 measurement columns.
- Missing sensor values (15 cases total in training) are informative of invalidity, not missing at random.

---

## 5. Limitations

- **Polynomial extrapolation:** The degree-4 Ridge model is calibrated to the training operating range (Voltage: 10–25 kV, Current: 50–150 A). Readings outside this envelope may have poorly calibrated residuals and could miss or falsely trigger the spike rule.
- **Threshold assumes comparable noise level:** The adaptive threshold is derived from training-data noise characteristics. A systematic sensor recalibration event on the live system (shifting the baseline noise floor) would require recomputing the threshold from fresh calibrated reference tests.
- **Batch duplicate detection:** The duplicate-vector rule requires the duplicated fingerprint to appear more than once in the prediction batch. A standalone duplicate submitted alone would not be flagged by this rule (it would fall through to the spike check).
- **We deliberately declined the missing-S4 shortcut:** Many rows with S4 NaN are Valid; using S4 missingness as a direct invalidity signal would conflate instrument absence with sensor fault, harming generalisation.
- **Blend ensemble in regression** outperforms single models (MAE 0.645) but adds deployment complexity. The polynomial model (degree 4, MAE 0.513 on Valid rows) is the recommended production model for interpretability.

---

## 6. Digital Twin Deployment Steps

The twin is fully operational, not hypothetical. Deployment on a live system:

1. **Ingest live readings** — receive the four operating controls (Voltage, Current, Temperature, Duration) and three sensor values (S1, S2, S3) for each completed test.
2. **Evaluate the twin** — apply the polynomial Ridge models fitted on all 866 Valid training rows to predict expected S1, S2, S3.
3. **Compute residuals → robust Z-scores** — using the training-set median and MAD for each sensor channel (stored with the model artefact).
4. **Compare to threshold** — spike_score = max(Z_S1, Z_S2, Z_S3). If spike_score > 12.77 (final fitted threshold), flag as sensor_spike.
5. **Apply precedence rules** — check for missing values (→ missing_sensor) and fingerprint duplicates (→ duplicate_vector) before the spike check.
6. **Raise a named alarm** — the fault bucket and the specific suspect sensor (the one with highest robust-Z) are logged and surfaced to the operator. Example: "sensor_spike: S2 (Z=18.4)".
7. **Log for recalibration** — all spike scores and thresholds are written to an audit log. Quarterly, the threshold is re-derived from new confirmed-Valid tests using the same empirical-gap algorithm.

**Measured false-alarm rate on training data:** 0 false positives across 866 Valid rows (FPR = 0.0%). Noise stress tests show the system reaches 100% detection at additive noise magnitude ≥ 20× the training noise floor, with monotonically non-decreasing detection rate and zero false positives at all magnitudes tested.

---

*Generated by src/benchmark.py | Branch: p3-benchmark | Seed: 42*
