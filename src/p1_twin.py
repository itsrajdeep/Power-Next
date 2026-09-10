"""
P1 – Sensor Digital Twin & Invalid Detection
=============================================

Implements the three public API functions from src/contract.py:

- fit_twin()       – fit polynomial Ridge twins on Valid training rows
- sensor_residuals() – compute raw residuals and robust-z scores
- predict_validity() – apply fault-rule precedence to classify rows

Also provides internal helpers for:
- Adaptive empirical-gap threshold derivation
- Out-of-fold evaluation
- Final fit and test prediction
"""

from __future__ import annotations

import math
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.contract import (
    RANDOM_SEED,
    COL_TEST_ID,
    COL_LABEL,
    COL_VOLTAGE,
    COL_CURRENT,
    COL_TEMP,
    COL_DURATION,
    COL_S1,
    COL_S2,
    COL_S3,
    COL_S4,
    OPERATING_COLS,
    CRITICAL_SENSOR_COLS,
    SENSOR_COLS,
    FINGERPRINT_COLS,
    LABEL_VALID,
    LABEL_INVALID,
    FAULT_NONE,
    FAULT_MISSING,
    FAULT_DUPLICATE,
    FAULT_SPIKE,
    fingerprint_series,
)


# ─── Epsilon floor for MAD ────────────────────────────────────────────────────
# When MAD is zero or negligibly small, we use this positive floor to avoid
# division by zero in robust-z score computation.  The value 1e-10 is chosen
# to be well below any realistic sensor noise while remaining numerically safe.
_MAD_EPSILON: float = 1e-10


# ─── Validation helpers ──────────────────────────────────────────────────────

def _validate_operating_cols(df: pd.DataFrame) -> None:
    """Raise ValueError if required operating columns are missing or malformed."""
    missing = [c for c in OPERATING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required operating columns: {missing}")

    # Nothing further to validate for empty DataFrames
    if len(df) == 0:
        return

    for col in OPERATING_COLS:
        series = df[col]
        # Check for non-numeric values (after NaN filtering)
        non_null = series.dropna()
        if len(non_null) > 0 and not pd.api.types.is_numeric_dtype(non_null):
            raise ValueError(
                f"Operating column '{col}' contains non-numeric values"
            )
        # Check for infinite values
        if len(non_null) > 0 and np.isinf(non_null.values.astype(float)).any():
            raise ValueError(
                f"Operating column '{col}' contains infinite values"
            )
        # Check for NaN / missing in operating inputs
        if series.isna().any():
            raise ValueError(
                f"Operating column '{col}' contains missing (NaN) values. "
                "Operating inputs must be complete."
            )



def _validate_sensor_cols(df: pd.DataFrame) -> None:
    """Raise ValueError if critical sensor columns are absent (NaN allowed)."""
    missing = [c for c in CRITICAL_SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")


# ─── Core twin fitting ───────────────────────────────────────────────────────

def _build_sensor_pipeline(
    alpha: float,
    degree: int,
    random_seed: int,
) -> Pipeline:
    """Build a single-sensor polynomial Ridge pipeline."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=random_seed)),
    ])


def _fit_residual_stats(
    residuals: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute robust residual center and scale from an array of residuals.

    center = median(residual)
    mad    = median(abs(residual - center))
    scale  = 1.4826 * max(mad, epsilon)

    Returns (center, scale).
    """
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    scale = 1.4826 * max(mad, _MAD_EPSILON)
    return center, scale


def _compute_spike_score(
    row_z: Dict[str, float],
) -> float:
    """
    Compute the spike score as the maximum available robust_z across sensors.
    """
    finite_z = [z for z in row_z.values() if np.isfinite(z)]
    if not finite_z:
        return float("nan")
    return max(finite_z)


# ─── Adaptive empirical-gap threshold ────────────────────────────────────────

def derive_threshold(
    spike_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Derive the adaptive empirical-gap threshold from a pool of spike scores.

    Eligible-gap procedure:
    1. Let n = number of finite candidate scores.
    2. min_upper_count = max(5, ceil(0.05 * n))
    3. Consider adjacent gaps only when:
       - the lower score is at or above the median-ranked observation
       - at least min_upper_count observations remain above the split
    4. Select the largest eligible adjacent gap
    5. If gaps tie, select the earliest eligible gap
    6. threshold = midpoint of the two gap endpoints
    7. Use spike_score > threshold (strict)

    Parameters
    ----------
    spike_scores : array
        Finite spike scores from the threshold pool.

    Returns
    -------
    dict with keys: lower, upper, gap, threshold

    Raises
    ------
    ValueError
        If there are too few scores or no positive eligible gap.
    """
    finite = spike_scores[np.isfinite(spike_scores)]
    finite = np.sort(finite)
    n = len(finite)

    if n < 3:
        raise ValueError(
            f"Too few finite spike scores ({n}) to derive a threshold. "
            "Need at least 3."
        )

    min_upper_count = max(5, math.ceil(0.05 * n))
    median_rank = n // 2  # index of median observation (0-indexed)

    best_gap = -1.0
    best_lower = float("nan")
    best_upper = float("nan")
    best_idx = -1

    for i in range(n - 1):
        lower_val = finite[i]
        upper_val = finite[i + 1]
        gap = upper_val - lower_val

        # Condition 1: lower score at or above median-ranked observation
        if i < median_rank:
            continue

        # Condition 2: at least min_upper_count observations above the split
        upper_count = n - (i + 1)
        if upper_count < min_upper_count:
            continue

        # Select largest eligible gap, earliest if tied
        if gap > best_gap:
            best_gap = gap
            best_lower = lower_val
            best_upper = upper_val
            best_idx = i

    if best_gap <= 0:
        raise ValueError(
            "No positive eligible gap found in spike scores. "
            f"n={n}, min_upper_count={min_upper_count}, median_rank={median_rank}"
        )

    threshold = (best_lower + best_upper) / 2.0

    return {
        "lower": float(best_lower),
        "upper": float(best_upper),
        "gap": float(best_gap),
        "threshold": float(threshold),
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def fit_twin(
    train_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    degree: int = 4,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Fit polynomial Ridge twins for S1, S2, S3 on Valid rows.

    Parameters
    ----------
    train_df : DataFrame
        Training data with OPERATING_COLS, CRITICAL_SENSOR_COLS, COL_LABEL.
    alpha : float
        Ridge regularisation strength.
    degree : int
        PolynomialFeatures degree.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Fitted model state with keys:
        - "models"    : dict mapping sensor name → fitted Pipeline
        - "centers"   : dict mapping sensor name → residual median
        - "scales"    : dict mapping sensor name → 1.4826 * MAD
        - "threshold" : float
        - "threshold_diagnostics" : dict with lower, upper, gap, threshold
        - "alpha"     : float
        - "degree"    : int
    """
    df = train_df.copy()

    # Validate inputs
    _validate_operating_cols(df[df[COL_LABEL] == LABEL_VALID])
    _validate_sensor_cols(df)

    if COL_LABEL not in df.columns:
        raise ValueError(f"Missing required column: {COL_LABEL}")

    # Filter to Valid rows only for fitting
    valid_mask = df[COL_LABEL] == LABEL_VALID
    valid_df = df[valid_mask].copy()

    if len(valid_df) == 0:
        raise ValueError("No Valid rows found for fitting.")

    # Fit twins for each critical sensor
    X_valid = valid_df[OPERATING_COLS].values
    models: Dict[str, Pipeline] = {}
    centers: Dict[str, float] = {}
    scales: Dict[str, float] = {}

    for sensor in CRITICAL_SENSOR_COLS:
        y_valid = valid_df[sensor].values

        # Valid rows should not have missing critical sensors, but verify
        if np.isnan(y_valid).any():
            raise ValueError(
                f"Valid training rows have NaN in {sensor}. "
                "This contradicts data integrity expectations."
            )

        pipeline = _build_sensor_pipeline(alpha, degree, random_seed)
        pipeline.fit(X_valid, y_valid)
        models[sensor] = pipeline

        # Compute residuals on valid training rows
        y_pred = pipeline.predict(X_valid)
        residuals = y_valid - y_pred
        center, scale = _fit_residual_stats(residuals)
        centers[sensor] = center
        scales[sensor] = scale

    # Compute spike scores for threshold derivation
    # Build threshold pool: exclude missing-S1/S2/S3 and duplicate rows
    # Missing rows
    missing_mask = df[CRITICAL_SENSOR_COLS].isna().any(axis=1)

    # Duplicate rows
    fps = fingerprint_series(df)
    dup_mask = fps.duplicated(keep=False)

    # Pool: rows that are neither missing nor duplicate
    pool_mask = ~missing_mask & ~dup_mask
    pool_df = df[pool_mask].copy()

    if len(pool_df) == 0:
        raise ValueError("No rows in the threshold pool after excluding "
                         "missing-sensor and duplicate rows.")

    # Compute spike scores for pool
    X_pool = pool_df[OPERATING_COLS].values
    pool_z: Dict[str, np.ndarray] = {}
    for sensor in CRITICAL_SENSOR_COLS:
        y_actual = pool_df[sensor].values
        y_pred = models[sensor].predict(X_pool)
        resid = y_actual - y_pred
        robust_z = np.abs(resid - centers[sensor]) / scales[sensor]
        pool_z[sensor] = robust_z

    spike_scores = np.max(
        np.column_stack([pool_z[s] for s in CRITICAL_SENSOR_COLS]),
        axis=1,
    )

    threshold_diag = derive_threshold(spike_scores)

    return {
        "models": models,
        "centers": centers,
        "scales": scales,
        "threshold": threshold_diag["threshold"],
        "threshold_diagnostics": threshold_diag,
        "alpha": alpha,
        "degree": degree,
    }


def sensor_residuals(
    df: pd.DataFrame,
    twin_state: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute residuals and robust-z scores for S1, S2, S3.

    Parameters
    ----------
    df : DataFrame
        Data with OPERATING_COLS and CRITICAL_SENSOR_COLS.
    twin_state : dict
        Fitted state from fit_twin().

    Returns
    -------
    DataFrame
        Copy of df with extra columns:
        - S1_expected, S2_expected, S3_expected
        - S1_residual, S2_residual, S3_residual  (signed: actual - expected)
        - S1_robust_z, S2_robust_z, S3_robust_z
        - spike_score (max robust_z across available sensors per row)
    """
    result = df.copy()
    _validate_operating_cols(df)

    models = twin_state["models"]
    centers = twin_state["centers"]
    scales_map = twin_state["scales"]

    X = df[OPERATING_COLS].values

    robust_z_cols: List[str] = []
    for sensor in CRITICAL_SENSOR_COLS:
        short = sensor.replace("Sensor_", "")  # S1, S2, S3

        # Predictions
        y_pred = models[sensor].predict(X)
        result[f"{short}_expected"] = y_pred

        # Raw signed residual: actual - expected
        y_actual = df[sensor].values.astype(float)
        residual = y_actual - y_pred
        result[f"{short}_residual"] = residual

        # Robust z-score
        robust_z = np.abs(residual - centers[sensor]) / scales_map[sensor]
        result[f"{short}_robust_z"] = robust_z
        robust_z_cols.append(f"{short}_robust_z")

    # Spike score: max of available robust_z values per row
    # For rows with missing sensors (NaN residual), robust_z will be NaN
    z_matrix = result[robust_z_cols].values
    # Use nanmax but keep NaN if all sensors are NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spike = np.nanmax(z_matrix, axis=1)
    # If all z values are NaN for a row, nanmax returns -inf or nan (depending on numpy)
    # Ensure it's NaN
    all_nan_mask = np.all(np.isnan(z_matrix), axis=1)
    spike[all_nan_mask] = np.nan
    result["spike_score"] = spike

    return result


def predict_validity(
    df: pd.DataFrame,
    twin_state: Dict[str, Any],
) -> pd.DataFrame:
    """
    Predict Validity_Label with fault-type annotation.

    Fault-rule precedence:
    1. missing_sensor  — any of S1, S2, S3 is NaN
    2. duplicate_vector — eight-column fingerprint duplicated in batch
    3. sensor_spike    — spike_score > threshold
    4. none            — Valid

    Duplicate detection has batch semantics: duplicated pairs are identified
    within the prediction batch using keep=False semantics, meaning both/all
    rows in a duplicated group are flagged. This remains stable after
    row-order shuffling.

    Parameters
    ----------
    df : DataFrame
        Data with OPERATING_COLS, SENSOR_COLS, COL_TEST_ID.
    twin_state : dict
        Fitted state from fit_twin().

    Returns
    -------
    DataFrame
        Columns: Test_ID, Validity_Label, fault_type
        Row order matches df.
    """
    work = df.copy()
    n = len(work)
    threshold = twin_state["threshold"]

    # Initialize outputs
    predictions = [LABEL_VALID] * n
    fault_types = [FAULT_NONE] * n

    # --- Rule 1: missing_sensor ---
    # Any of S1, S2, S3 is NaN
    missing_mask = work[CRITICAL_SENSOR_COLS].isna().any(axis=1).values

    # --- Rule 2: duplicate_vector ---
    # Compute fingerprints for the entire batch (batch semantics)
    # Need all FINGERPRINT_COLS present
    fp_cols_present = [c for c in FINGERPRINT_COLS if c in work.columns]
    if len(fp_cols_present) == len(FINGERPRINT_COLS):
        fps = fingerprint_series(work)
        dup_mask = fps.duplicated(keep=False).values
    else:
        dup_mask = np.zeros(n, dtype=bool)

    # --- Compute spike scores for non-missing rows ---
    # We need operating cols to be present and valid for twin computation
    _validate_operating_cols(work)

    models = twin_state["models"]
    centers = twin_state["centers"]
    scales_map = twin_state["scales"]
    X = work[OPERATING_COLS].values

    spike_scores = np.full(n, np.nan)
    for i in range(n):
        if missing_mask[i]:
            continue  # Skip spike computation for missing-sensor rows

        z_vals = []
        for sensor in CRITICAL_SENSOR_COLS:
            actual = work[sensor].iloc[i]
            if pd.isna(actual):
                continue
            pred = models[sensor].predict(X[i:i+1, :])[0]
            resid = float(actual) - pred
            z = abs(resid - centers[sensor]) / scales_map[sensor]
            z_vals.append(z)

        if z_vals:
            spike_scores[i] = max(z_vals)

    # --- Apply precedence ---
    for i in range(n):
        if missing_mask[i]:
            predictions[i] = LABEL_INVALID
            fault_types[i] = FAULT_MISSING
        elif dup_mask[i]:
            predictions[i] = LABEL_INVALID
            fault_types[i] = FAULT_DUPLICATE
        elif np.isfinite(spike_scores[i]) and spike_scores[i] > threshold:
            predictions[i] = LABEL_INVALID
            fault_types[i] = FAULT_SPIKE
        else:
            predictions[i] = LABEL_VALID
            fault_types[i] = FAULT_NONE

    return pd.DataFrame({
        COL_TEST_ID: work[COL_TEST_ID].values,
        COL_LABEL: predictions,
        "fault_type": fault_types,
    })


# ─── OOF evaluation ─────────────────────────────────────────────────────────

def run_oof_evaluation(
    train_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    degree: int = 4,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run 5-fold out-of-fold evaluation.

    Parameters
    ----------
    train_df : DataFrame
        Full training data.
    folds_df : DataFrame
        Fold assignments with columns [Test_ID, fold].
    alpha, degree, random_seed :
        Passed to fit_twin().

    Returns
    -------
    (oof_predictions, diagnostics)
        oof_predictions : DataFrame with Test_ID, Validity_Label, fault_type, fold
        diagnostics : dict with per-fold and aggregate metrics
    """
    # Validate fold-train join
    train_ids = set(train_df[COL_TEST_ID])
    fold_ids = set(folds_df[COL_TEST_ID])
    if train_ids != fold_ids:
        extra_in_folds = fold_ids - train_ids
        missing_in_folds = train_ids - fold_ids
        raise ValueError(
            f"Fold/train ID mismatch. "
            f"Extra in folds: {extra_in_folds}, Missing in folds: {missing_in_folds}"
        )
    if folds_df[COL_TEST_ID].duplicated().any():
        raise ValueError("Duplicate Test_IDs in folds.csv")

    # Merge
    merged = train_df.merge(folds_df[[COL_TEST_ID, "fold"]], on=COL_TEST_ID, how="left")
    assert merged["fold"].notna().all(), "Some training rows have no fold assignment"

    fold_ids_unique = sorted(merged["fold"].unique())
    assert len(fold_ids_unique) == 5, f"Expected 5 folds, got {len(fold_ids_unique)}"

    # Validate no fingerprint group crosses folds
    fps = fingerprint_series(merged)
    merged_check = merged.copy()
    merged_check["_fp"] = fps
    fp_folds = merged_check.groupby("_fp")["fold"].nunique()
    crossing = fp_folds[fp_folds > 1]
    if len(crossing) > 0:
        raise ValueError(
            f"Fingerprint groups cross folds: {crossing.index.tolist()[:5]}"
        )

    # OOF loop
    all_oof = []
    fold_diagnostics = {}

    for fold_val in fold_ids_unique:
        val_mask = merged["fold"] == fold_val
        train_mask = ~val_mask

        fold_train = merged[train_mask].copy()
        fold_val_df = merged[val_mask].copy()

        # Fit twin on fold training data
        twin_state = fit_twin(
            fold_train,
            alpha=alpha,
            degree=degree,
            random_seed=random_seed,
        )

        # Predict on validation fold
        preds = predict_validity(fold_val_df, twin_state)
        preds["fold"] = fold_val

        all_oof.append(preds)

        # Record fold diagnostics
        td = twin_state["threshold_diagnostics"]
        fold_diagnostics[int(fold_val)] = {
            "threshold": twin_state["threshold"],
            "lower": td["lower"],
            "upper": td["upper"],
            "gap": td["gap"],
            "n_val": len(fold_val_df),
        }

    oof_df = pd.concat(all_oof, ignore_index=True)

    # Validate every training row has exactly one OOF prediction
    assert len(oof_df) == len(train_df), (
        f"OOF has {len(oof_df)} rows, expected {len(train_df)}"
    )
    assert set(oof_df[COL_TEST_ID]) == set(train_df[COL_TEST_ID]), (
        "OOF Test_IDs don't match training Test_IDs"
    )

    # Compute metrics
    # Merge with actual labels
    actuals = train_df[[COL_TEST_ID, COL_LABEL]].rename(
        columns={COL_LABEL: "actual_label"}
    )
    eval_df = oof_df.merge(actuals, on=COL_TEST_ID)

    y_true = (eval_df["actual_label"] == LABEL_INVALID).astype(int)
    y_pred = (eval_df[COL_LABEL] == LABEL_INVALID).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Fault taxonomy
    taxonomy = oof_df["fault_type"].value_counts().to_dict()

    # Per-fold metrics
    per_fold_metrics = {}
    for fold_val in fold_ids_unique:
        fold_preds = eval_df[eval_df["fold"] == fold_val]
        yt = (fold_preds["actual_label"] == LABEL_INVALID).astype(int)
        yp = (fold_preds[COL_LABEL] == LABEL_INVALID).astype(int)
        f_tp = int(((yt == 1) & (yp == 1)).sum())
        f_fp = int(((yt == 0) & (yp == 1)).sum())
        f_fn = int(((yt == 1) & (yp == 0)).sum())
        f_tn = int(((yt == 0) & (yp == 0)).sum())
        f_prec = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0.0
        f_rec = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0.0
        f_f1 = (2 * f_prec * f_rec / (f_prec + f_rec)
                if (f_prec + f_rec) > 0 else 0.0)
        per_fold_metrics[int(fold_val)] = {
            "precision": f_prec,
            "recall": f_rec,
            "f1": f_f1,
            "tp": f_tp,
            "fp": f_fp,
            "fn": f_fn,
            "tn": f_tn,
        }

    diagnostics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "taxonomy": taxonomy,
        "per_fold_metrics": per_fold_metrics,
        "per_fold_threshold": fold_diagnostics,
    }

    return oof_df, diagnostics


# ─── Final fit and predict ───────────────────────────────────────────────────

def final_fit_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    degree: int = 4,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit final twin on all Valid training data and predict test data.

    Returns
    -------
    (test_predictions, twin_state)
    """
    twin_state = fit_twin(
        train_df,
        alpha=alpha,
        degree=degree,
        random_seed=random_seed,
    )
    test_preds = predict_validity(test_df, twin_state)
    return test_preds, twin_state
