"""
P1 compatibility API: Sensor Twin & Invalid Detection.

This module exposes the exact interface requested by the project brief while
delegating model fitting to the richer implementation in ``src.p1_twin``.

Verdict rules:
1. Invalid if any of Sensor_S1/Sensor_S2/Sensor_S3 is missing.
2. Invalid if the operating + S1/S2/S3 measurement vector is duplicated inside
   the dataframe being scored.
3. Invalid if the maximum absolute twin residual across available S1/S2/S3
   exceeds the derived threshold.
4. Valid otherwise.

Sensor_S4 is intentionally ignored so S4 values, S4 missingness, or removing the
S4 column entirely cannot change a verdict.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.contract import (
    RANDOM_SEED,
    COL_LABEL,
    OPERATING_COLS,
    CRITICAL_SENSOR_COLS,
    LABEL_VALID,
    LABEL_INVALID,
)
from src.p1_twin import fit_twin as _fit_state


_THRESHOLD_SIGMA_MULTIPLIER = 8.0
_SMALL_THRESHOLD_FLOOR = 1.0 + 0.1 + 1e-9


def _validate_twin(twin: Dict) -> None:
    """Validate the minimal model mapping required by the public API."""
    if not isinstance(twin, dict):
        raise TypeError("twin must be a dictionary keyed by critical sensor name.")

    missing = [sensor for sensor in CRITICAL_SENSOR_COLS if sensor not in twin]
    if missing:
        raise ValueError(f"Twin is missing models for: {missing}")

    invalid = [
        sensor
        for sensor in CRITICAL_SENSOR_COLS
        if not callable(getattr(twin[sensor], "predict", None))
    ]
    if invalid:
        raise TypeError(f"Twin models must expose predict(); invalid models: {invalid}")


def _operating_values(df: pd.DataFrame) -> np.ndarray:
    """Return finite operating inputs with consistent validation errors."""
    missing = [col for col in OPERATING_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing operating columns: {missing}")

    try:
        values = df[OPERATING_COLS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Operating columns must contain numeric values.") from exc

    if not np.isfinite(values).all():
        raise ValueError("Operating columns must contain only finite values.")
    return values


def _with_valid_label(train_valid: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Valid labels when the caller passes Valid rows only."""
    df = train_valid.copy()
    if COL_LABEL not in df.columns:
        df[COL_LABEL] = LABEL_VALID
    return df


def _duplicate_cols(df: pd.DataFrame) -> list[str]:
    """Columns used for duplicate detection; deliberately excludes Sensor_S4."""
    cols = OPERATING_COLS + CRITICAL_SENSOR_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for P1 duplicate detection: {missing}")
    return cols


def fit_twin(train_valid: pd.DataFrame) -> Dict:
    """Fit one polynomial Ridge sensor model per critical sensor.

    The brief passes Valid rows, but this function also tolerates the full
    training dataframe because ``src.p1_twin.fit_twin`` filters to Valid rows
    internally when ``Validity_Label`` is present.
    """
    state = _fit_state(
        _with_valid_label(train_valid),
        alpha=1.0,
        degree=4,
        random_seed=RANDOM_SEED,
    )
    return state["models"]


def sensor_residuals(df: pd.DataFrame, twin: Dict) -> pd.DataFrame:
    """Return S1/S2/S3 residuals, indexed like ``df``.

    Residuals are signed as ``actual - expected``. Missing sensor readings remain
    NaN, which lets rule one handle them explicitly in ``predict_validity``.
    """
    _validate_twin(twin)
    missing = [sensor for sensor in CRITICAL_SENSOR_COLS if sensor not in df.columns]
    if missing:
        raise ValueError(f"Missing critical sensor columns: {missing}")

    if len(df) == 0:
        columns = [sensor.replace("Sensor_", "") for sensor in CRITICAL_SENSOR_COLS]
        return pd.DataFrame(index=df.index, columns=columns, dtype=float)

    x_values = _operating_values(df)
    residual_cols = {}
    for sensor in CRITICAL_SENSOR_COLS:
        short = sensor.replace("Sensor_", "")
        expected = twin[sensor].predict(x_values)
        actual = df[sensor].to_numpy(dtype=float)
        residual_cols[short] = actual - expected
    return pd.DataFrame(residual_cols, index=df.index)


def derive_threshold(train_valid: pd.DataFrame, twin: Dict) -> float:
    """Compute a data-derived raw-residual spike threshold.

    The rule is ``max(8 * largest robust residual sigma, small_floor)`` using
    Valid-row residuals from the fitted twin. On the provided dataset this lands
    inside the project's required [1.1, 4.0] interval while leaving comfortable
    headroom above normal sensor noise.
    """
    df = _with_valid_label(train_valid)
    if COL_LABEL in df.columns:
        df = df[df[COL_LABEL] == LABEL_VALID].copy()
    if len(df) == 0:
        raise ValueError("No rows available to derive P1 threshold.")

    residual_df = sensor_residuals(df, twin)
    sigmas = []
    for col in residual_df.columns:
        values = residual_df[col].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        center = np.median(values)
        mad = np.median(np.abs(values - center))
        robust_sigma = 1.4826 * max(float(mad), 1e-10)
        sigmas.append(robust_sigma)

    if not sigmas:
        raise ValueError("No finite residuals available to derive P1 threshold.")

    threshold = max(_THRESHOLD_SIGMA_MULTIPLIER * max(sigmas), _SMALL_THRESHOLD_FLOOR)
    print(f"P1 derived raw-residual threshold: {threshold:.6f}")
    return float(threshold)


def predict_validity(df: pd.DataFrame, twin: Dict, threshold: float) -> pd.Series:
    """Return a Valid/Invalid Series with the same index and length as ``df``."""
    work = df.copy()
    if len(work) == 0:
        return pd.Series([], index=work.index, dtype=object, name=COL_LABEL)

    _validate_twin(twin)
    missing_cols = [sensor for sensor in CRITICAL_SENSOR_COLS if sensor not in work]
    if missing_cols:
        raise ValueError(f"Missing critical sensor columns: {missing_cols}")

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a positive finite number.") from exc
    if not np.isfinite(threshold_value) or threshold_value <= 0:
        raise ValueError("threshold must be a positive finite number.")

    missing_mask = work[CRITICAL_SENSOR_COLS].isna().any(axis=1)

    duplicate_frame = work[_duplicate_cols(work)].round(4)
    duplicate_mask = duplicate_frame.duplicated(keep=False)

    residual_df = sensor_residuals(work, twin).abs()
    max_abs_residual = residual_df.max(axis=1, skipna=True)
    spike_mask = max_abs_residual > threshold_value

    invalid_mask = missing_mask | duplicate_mask | spike_mask.fillna(False)
    labels = np.where(invalid_mask.to_numpy(), LABEL_INVALID, LABEL_VALID)
    return pd.Series(labels, index=work.index, name=COL_LABEL)
