"""
Stub implementation for Person 2 (Reference Parameter Regression).
Provides fallback stubs and baseline implementation until src/regress.py is integrated.
"""
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from src.contract import OPS


def fit_reference_model(train_valid: pd.DataFrame) -> Any:
    """
    Fits baseline reference parameter regression model on Valid training rows.
    Uses degree-4 polynomial Ridge regression on operating conditions.
    """
    X = train_valid[OPS].values
    y = train_valid["Reference_Parameter"].values
    model = make_pipeline(
        PolynomialFeatures(degree=4, include_bias=False),
        StandardScaler(),
        Ridge(alpha=10.0)
    )
    model.fit(X, y)
    return model


def predict_reference(df: pd.DataFrame, model: Optional[Any] = None) -> pd.Series:
    """
    Predicts hot-spot Reference_Parameter temperature.
    If model is None, falls back to baseline 25.0 °C.
    """
    if model is None:
        return pd.Series([25.0] * len(df), index=df.index, name="Predicted_Reference_Parameter")

    ops_present = [c for c in OPS if c in df.columns]
    if len(ops_present) != len(OPS):
        return pd.Series([25.0] * len(df), index=df.index, name="Predicted_Reference_Parameter")

    X = df[OPS].values
    preds = model.predict(X)
    # Ensure no NaNs or infs, and clamp within sensible physical bounds
    preds = np.nan_to_num(preds, nan=25.0, posinf=150.0, neginf=20.0)
    return pd.Series(preds, index=df.index, name="Predicted_Reference_Parameter")
