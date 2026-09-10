# P1: Sensor Twin & Invalid Detection — Walkthrough

## Branch & Commit
- **Branch:** `p1-twin`
- **Commit:** `0428cf50577326759f93e5d5f83049d53c08fc7d`
- **PR URL:** https://github.com/itsrajdeep/Power-Next/pull/new/p1-twin

---

## Summary

Implemented a deterministic sensor digital-twin model that detects Invalid test measurements using polynomial Ridge regression residuals and rule-based fault classification. The system achieves **perfect OOF classification** (Precision=1.0, Recall=1.0, F1=1.0) with the taxonomy emerging naturally from the rules — not hard-coded.

## Public API Implemented

| Function | Signature | Purpose |
|---|---|---|
| `fit_twin()` | `(train_df, *, alpha, degree, random_seed) → dict` | Fit polynomial Ridge twins for S1, S2, S3 on Valid rows |
| `sensor_residuals()` | `(df, twin_state) → DataFrame` | Compute residuals and robust-z scores |
| `predict_validity()` | `(df, twin_state) → DataFrame` | Predict with fault-type annotation |

## Files Changed

| File | Type | Description |
|---|---|---|
| `src/contract.py` | NEW | Shared cross-team contract with API signatures, constants, fingerprint |
| `src/__init__.py` | NEW | Package init |
| `src/p1_twin.py` | NEW | Core P1 implementation |
| `tests/test_p1.py` | NEW | 58 acceptance & unit tests |
| `tests/__init__.py` | NEW | Test package init |
| `folds.csv` | NEW | 5-fold stratified group-aware CV splits |
| `briefs/p1.md` | NEW | P1 brief |
| `run_p1.py` | NEW | Runner for OOF + final fit |
| `scripts/generate_folds.py` | NEW | Fold generation (run once) |
| `scripts/p1_charts.py` | NEW | Chart and stress test generator |
| `.gitignore` | MODIFIED | Added `.pytest_cache/` |
| `outputs/p1/*` | NEW | OOF predictions, metrics, charts, diagnostics |

---

## Test Results

```
python -m pytest tests/test_p1.py -v
```

**58 passed, 0 failed** in 8.57s

### Test Categories
| Category | Tests | Status |
|---|---|---|
| Contract & Scope | 4 | ✅ All pass |
| Fingerprint | 5 | ✅ All pass |
| Folds | 4 | ✅ All pass |
| Twin Fitting | 6 | ✅ All pass |
| Fault Rules | 12 | ✅ All pass |
| OOF Acceptance | 9 | ✅ All pass |
| Noise Stress | 4 | ✅ All pass |
| Threshold | 5 | ✅ All pass |
| Data Integrity | 9 | ✅ All pass |

---

## OOF Metrics

| Metric | Value |
|---|---|
| Precision (Invalid) | **1.0000** |
| Recall (Invalid) | **1.0000** |
| F1 (Invalid) | **1.0000** |

### Confusion Matrix

|  | Predicted Valid | Predicted Invalid |
|---|---|---|
| **Actual Valid** | TN=866 | FP=0 |
| **Actual Invalid** | FN=0 | TP=134 |

### Aggregate Fault Taxonomy

| Fault Type | Count |
|---|---|
| missing_sensor | **15** |
| duplicate_vector | **24** |
| sensor_spike | **95** |
| none (Valid) | **866** |
| **Total Invalid** | **134** |

---

## Per-Fold Thresholds

| Fold | Threshold | Lower | Upper | Gap |
|---|---|---|---|---|
| 0 | 12.6230 | 3.5247 | 21.7214 | 18.1968 |
| 1 | 12.8011 | 3.7229 | 21.8793 | 18.1564 |
| 2 | 12.9678 | 3.6647 | 22.2710 | 18.6062 |
| 3 | 12.6430 | 3.6651 | 21.6209 | 17.9558 |
| 4 | 14.6031 | 3.7393 | 25.4668 | 21.7275 |
| **Final** | **12.7738** | 3.6876 | 21.8600 | 18.1724 |

---

## Noise Stress Test

| Magnitude | Rows | Detected Invalid | Detection Rate |
|---|---|---|---|
| 0.0 | 50 | 0 | 0.000 |
| 0.5 | 50 | 0 | 0.000 |
| 1.0 | 50 | 4 | 0.080 |
| 2.0 | 50 | 21 | 0.420 |
| 5.0 | 50 | 49 | 0.980 |
| 10.0 | 50 | 49 | 0.980 |
| 20.0 | 50 | 50 | 1.000 |
| 50.0 | 50 | 50 | 1.000 |
| 100.0 | 50 | 50 | 1.000 |

Detection rate is monotonically non-decreasing with false positive rate = 0%.

---

## Integrity Verification

- ✅ `src/contract.py` hash unchanged: `7ec14063...`
- ✅ `folds.csv` hash unchanged: `2de96fe3...`
- ✅ No P2, P3, or P4 files created or modified
- ✅ No changes to `run.py`, regression code, benchmark code, or methodology

---

## Known Limitations

1. **Duplicate detection has batch semantics** — a row is only flagged as `duplicate_vector` if its fingerprint appears more than once within the prediction batch. The same row in a single-row batch would not be flagged.
2. **Extrapolation** — the polynomial Ridge model is fitted on the operating-input ranges present in training data. Predictions on inputs far outside this range may have poor residual calibration.
3. **MAD epsilon floor** — when all residuals for a sensor are identical (MAD=0), the robust-z score uses a small epsilon floor (1e-10), which produces very large z-scores for any deviation.
4. **S4 not modelled** — per specification, only S1, S2, S3 have twin models. S4 is used only in fingerprinting.
