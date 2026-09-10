# Power-Next: Methodology Note

**Project:** CPRI Hackathon Screening — Power Equipment Sensor Validity Classification & Reference Parameter Regression  
**Branch:** p3-benchmark | **Seed:** 42 | **Folds:** folds.csv (5-fold StratifiedGroupKFold, committed once)

---

## 1. Approach

The problem has two distinct sub-tasks:

**Task A — Validity Classification**: Determine whether each test measurement is Valid or Invalid. The dataset contains 1,000 training rows (866 Valid, 134 Invalid — 13.4% imbalance). Rather than treating this as a pure ML problem, we employ a **physics-informed digital twin** as the primary detector (P1), and benchmark ML classifiers as evidence for the leaderboard.

**Task B — Reference Parameter Regression**: Predict Reference_Parameter (range: 11.9–61.6, mean: 26.7) from the four operating inputs: Voltage, Current, Temperature, Duration. Sensor columns are deliberately excluded from regression features to avoid leakage — sensors are downstream measurements that the Reference_Parameter helps explain.

The benchmark uses the same olds.csv for both tasks, ensuring comparability across all four workstreams.

---

## 2. Anomaly Detection Method (Digital Twin — P1)

The P1 digital twin fits a **polynomial Ridge regression** (degree=4, alpha=1.0) for each of S1, S2, S3 using the four operating inputs on Valid training rows only. Residuals are normalised into **robust Z-scores** (median-centred, MAD-scaled) to handle non-Gaussian sensor noise.

**Fault-rule precedence** (deterministic, no ML):
1. missing_sensor — any of S1, S2, S3 is NaN
2. duplicate_vector — 8-column fingerprint (4 dp, pipe-joined) appears more than once in the prediction batch
3. sensor_spike — max robust-z across S1/S2/S3 exceeds adaptive threshold
4. 
one → **Valid**

**Threshold derivation**: In each fold, the empirical gap between the upper boundary of Valid scores and the lower boundary of Invalid scores is used. The threshold is placed at the midpoint of this gap, making it robust to dataset shift and requiring no manual tuning. OOF gap ≈ 18.2 (well-separated), final threshold ≈ 12.77.

**OOF result**: Precision=1.0, Recall=1.0, F1=1.0 across all 5 folds. Fault taxonomy: 15 missing_sensor, 24 duplicate_vector, 95 sensor_spike.

---

## 3. Key Model Parameters

| Model | Key Parameters |
|---|---|
| LogReg | max_iter=1000, class_weight=balanced |
| RF / ExtraTrees (clf) | 
_estimators=300, class_weight=balanced |
| HistGB | max_iter=300, class_weight=balanced |
| CatBoost (clf) | iterations=300, uto_class_weights=Balanced |
| XGBoost (clf) | 
_estimators=300, scale_pos_weight=866/134≈6.46 |
| LightGBM (clf) | 
_estimators=300, is_unbalance=True |
| Ridge (reg) | lpha=1.0 |
| ElasticNet | lpha=0.1, l1_ratio=0.5 |
| Huber | max_iter=500 |
| RF / ExtraTrees (reg) | 
_estimators=300 |
| HistGBR | max_iter=300 |
| CatBoost (reg) | iterations=300 |
| XGBoost (reg) | 
_estimators=300 |
| LightGBM (reg) | 
_estimators=300 |
| PolyRidge | degree=3, StandardScaler, Ridge(alpha=1.0) |
| Blend | Equal-weight mean of all non-linear regression OOF predictions |

All models use RANDOM_SEED=42. Class imbalance is handled explicitly in all classifiers.

---

## 4. Assumptions & Limitations

**Assumptions**:
- The 5-fold split in olds.csv is the canonical evaluation split; no model is allowed to regenerate it.
- Reference_Parameter is a continuous numeric target with no missing values (confirmed: 0 NaN).
- Sensor columns (S1–S4) are treated as measurement outcomes, not causal inputs for regression.
- Duplicate detection has **batch semantics** — a fingerprint must appear >1 time within the prediction batch.

**Limitations**:
- The digital twin's polynomial model may extrapolate poorly on operating inputs outside the training distribution.
- ML classifiers on 8 features with 13.4% imbalance are prone to precision/recall trade-offs — the leaderboard F1 scores must be read alongside recall to avoid optimistic precision inflation.
- The regression task uses only 4 operating inputs; adding sensor features (post-validity-filter) may improve R², but risks data leakage if applied naively.
- Blend is a simple equal-weight mean — Bayesian or stacking ensembles were not evaluated.

---

## 5. Digital Twin Section

The sensor digital twin is the cornerstone of the P1 workstream and serves as the **rule-based baseline** in the classification leaderboard. Unlike ML classifiers that learn statistical associations from labels, the twin:

- **Models the physics**: Each sensor is predicted from operating conditions via a polynomial Ridge model fitted exclusively on labelled-Valid data. This encodes the expected sensor behaviour under normal operation.
- **Detects deviation, not correlation**: The robust Z-score measures how far each measurement deviates from what physics predicts — not from historical average values.
- **Is fully interpretable**: Every prediction has a human-readable fault type and a continuous spike score. There are no black-box activations.
- **Generalises by design**: Because the threshold is derived from the empirical separation in each fold's training data, it adapts to changes in operating range without retraining.

The twin achieves F1=1.0 on 5-fold OOF — outperforming all ML classifiers on this dataset — while producing calibrated fault explanations that a domain engineer can directly audit.

---

*Generated by src/benchmark.py | Branch: p3-benchmark | Seed: 42*
