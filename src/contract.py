"""
Power-Next Shared Contract
==========================
Public API surface shared across all four workstreams (P1–P4).
Do NOT modify this file after initial commit — all persons code against it.

Dataset path, column names, label constants, fingerprint function,
and P1/P2/P3/P4 function signatures live here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "dataset" / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
FOLDS_PATH = REPO_ROOT / "folds.csv"

# ─── Sheet names ──────────────────────────────────────────────────────────────
TRAIN_SHEET = "Training_Data"
TEST_SHEET = "Test_Data"
SAMPLE_SHEET = "Sample_Submission"

# ─── Column names ─────────────────────────────────────────────────────────────
COL_TEST_ID = "Test_ID"
COL_VOLTAGE = "Applied_Voltage_kV"
COL_CURRENT = "Load_Current_A"
COL_TEMP = "Ambient_Temperature_C"
COL_DURATION = "Test_Duration_min"
COL_S1 = "Sensor_S1"
COL_S2 = "Sensor_S2"
COL_S3 = "Sensor_S3"
COL_S4 = "Sensor_S4"
COL_REF = "Reference_Parameter"
COL_LABEL = "Validity_Label"
COL_FOLD = "fold"

OPERATING_COLS: List[str] = [COL_VOLTAGE, COL_CURRENT, COL_TEMP, COL_DURATION]
SENSOR_COLS: List[str] = [COL_S1, COL_S2, COL_S3, COL_S4]
CRITICAL_SENSOR_COLS: List[str] = [COL_S1, COL_S2, COL_S3]

FINGERPRINT_COLS: List[str] = [
    COL_VOLTAGE, COL_CURRENT, COL_TEMP, COL_DURATION,
    COL_S1, COL_S2, COL_S3, COL_S4,
]

# ─── Label constants ─────────────────────────────────────────────────────────
LABEL_VALID = "Valid"
LABEL_INVALID = "Invalid"

# ─── Fault-type constants ────────────────────────────────────────────────────
FAULT_NONE = "none"
FAULT_MISSING = "missing_sensor"
FAULT_DUPLICATE = "duplicate_vector"
FAULT_SPIKE = "sensor_spike"

# ─── Random seed ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Fingerprint function ────────────────────────────────────────────────────

def fingerprint(row_values: pd.Series | np.ndarray | list, decimals: int = 4) -> str:
    """
    Compute a duplicate-detection fingerprint for one row.

    Parameters
    ----------
    row_values : array-like
        Values for the eight FINGERPRINT_COLS in order.
    decimals : int
        Number of decimal places to round to.

    Returns
    -------
    str
        Pipe-delimited string of rounded values.
        NaN values are represented as the literal string ``'nan'``.
    """
    parts: List[str] = []
    for v in row_values:
        if isinstance(v, float) and math.isnan(v):
            parts.append("nan")
        elif pd.isna(v):
            parts.append("nan")
        else:
            parts.append(str(round(float(v), decimals)))
    return "|".join(parts)


def fingerprint_series(df: pd.DataFrame, cols: List[str] | None = None,
                       decimals: int = 4) -> pd.Series:
    """
    Compute fingerprint for every row in *df*.

    Parameters
    ----------
    df : DataFrame
        Must contain all ``FINGERPRINT_COLS``.
    cols : list[str] or None
        Override columns; defaults to ``FINGERPRINT_COLS``.
    decimals : int
        Rounding precision.

    Returns
    -------
    pd.Series
        String fingerprint per row, index-aligned with *df*.
    """
    use_cols = cols or FINGERPRINT_COLS
    return df[use_cols].apply(lambda r: fingerprint(r, decimals), axis=1)


# ─── Data loading helpers ────────────────────────────────────────────────────

def load_train(path: Path | str | None = None) -> pd.DataFrame:
    """Load Training_Data sheet, return a copy."""
    p = Path(path) if path else DATASET_PATH
    return pd.read_excel(p, sheet_name=TRAIN_SHEET)


def load_test(path: Path | str | None = None) -> pd.DataFrame:
    """Load Test_Data sheet, return a copy."""
    p = Path(path) if path else DATASET_PATH
    return pd.read_excel(p, sheet_name=TEST_SHEET)


def load_folds(path: Path | str | None = None) -> pd.DataFrame:
    """Load folds.csv (columns: Test_ID, fold)."""
    p = Path(path) if path else FOLDS_PATH
    return pd.read_csv(p)


# ─── P1 API: Sensor Twin & Invalid Detection ────────────────────────────────

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
        - ``"models"``    : dict mapping sensor name → fitted Pipeline
        - ``"centers"``   : dict mapping sensor name → residual median
        - ``"scales"``    : dict mapping sensor name → 1.4826 * MAD
        - ``"threshold"`` : float
        - ``"threshold_diagnostics"`` : dict with ``lower``, ``upper``, ``gap``, ``threshold``
        - ``"alpha"``     : float
        - ``"degree"``    : int
    """
    raise NotImplementedError("P1 must implement fit_twin()")


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
        Fitted state from ``fit_twin()``.

    Returns
    -------
    DataFrame
        Copy of *df* with extra columns:
        - ``S1_expected``, ``S2_expected``, ``S3_expected``
        - ``S1_residual``, ``S2_residual``, ``S3_residual``
        - ``S1_robust_z``, ``S2_robust_z``, ``S3_robust_z``
        - ``spike_score`` (max of robust_z across sensors per row)
    """
    raise NotImplementedError("P1 must implement sensor_residuals()")


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

    Parameters
    ----------
    df : DataFrame
        Data with OPERATING_COLS, SENSOR_COLS, COL_TEST_ID.
    twin_state : dict
        Fitted state from ``fit_twin()``.

    Returns
    -------
    DataFrame
        Columns: ``Test_ID``, ``Validity_Label``, ``fault_type``
        Row order matches *df*.
    """
    raise NotImplementedError("P1 must implement predict_validity()")
