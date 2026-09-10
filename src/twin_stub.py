"""
Stub implementation for Person 1 (Sensor Digital Twin & Invalid Detection).
Provides fallback stubs and baseline implementation until src/twin.py is integrated.
"""
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from src.contract import OPS, S3, MEASUREMENTS, is_duplicate


def fit_twin(train_valid: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits polynomial Ridge models for S1, S2, S3 from operating conditions.
    """
    X = train_valid[OPS].values
    twins = {}
    for s in S3:
        y = train_valid[s].values
        # degree-4 polynomial on 4 operating inputs achieves R^2 > 0.995
        model = make_pipeline(
            PolynomialFeatures(degree=4, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0)
        )
        model.fit(X, y)
        twins[s] = model
    return twins


def derive_threshold(train_valid: pd.DataFrame, twin: Optional[Dict[str, Any]] = None) -> float:
    """
    Derives the spike detection residual threshold.
    Based on verified findings, clean separation gap is 1.055 .. 4.177.
    """
    if twin is None:
        return 2.0

    X = train_valid[OPS].values
    max_res = 0.0
    for s in S3:
        if s in twin:
            preds = twin[s].predict(X)
            res = np.abs(train_valid[s].values - preds)
            valid_res = res[np.isfinite(res)]
            if len(valid_res) > 0:
                max_res = max(max_res, float(np.max(valid_res)))

    return max(2.0, max_res * 1.5)


def predict_validity(
    df: pd.DataFrame,
    twin: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None
) -> pd.Series:
    """
    Predicts 'Valid' or 'Invalid' for each record in df.
    Applies rule precedence:
    1. Missing critical sensors (S1, S2, S3) -> Invalid
    2. Duplicate measurement vectors -> Invalid
    3. Disconnected / all-zero sensors -> Invalid
    4. Sensor residuals exceeding dynamic threshold -> Invalid
    5. All other rows -> Valid
    """
    n = len(df)
    labels = np.array(["Valid"] * n, dtype=object)

    # 1. Missing readings in S1, S2, S3
    for s in S3:
        if s in df.columns:
            labels[df[s].isna().values] = "Invalid"

    # 2. Duplicate measurement vectors (scoped within df)
    meas_cols = [c for c in MEASUREMENTS if c in df.columns]
    if len(meas_cols) == len(MEASUREMENTS):
        dup_mask = is_duplicate(df).values
    elif meas_cols:
        dup_mask = df.duplicated(subset=meas_cols, keep=False).values
    else:
        dup_mask = np.zeros(n, dtype=bool)
    labels[dup_mask] = "Invalid"

    # 3. Disconnected / all-zero sensors
    present_s3 = [s for s in S3 if s in df.columns]
    if present_s3:
        all_zeros = (df[present_s3] == 0).all(axis=1).values
        labels[all_zeros] = "Invalid"

    # 4. Residual check against twin
    if twin is not None and threshold is not None:
        ops_present = [col for col in OPS if col in df.columns]
        if len(ops_present) == len(OPS):
            X = df[OPS].values
            for s in present_s3:
                if s in twin:
                    try:
                        preds = twin[s].predict(X)
                        residuals = np.abs(df[s].values - preds)
                        with np.errstate(invalid="ignore"):
                            spikes = (residuals > threshold) & ~np.isnan(residuals)
                        labels[spikes] = "Invalid"
                    except Exception:
                        pass

    return pd.Series(labels, index=df.index, name="Validity_Label")
