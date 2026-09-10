"""
Finalization and Deliverables Module.
Owns submission validation, summary generation, attention scoring, and output writing.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from src.contract import OPS, S3, MEASUREMENTS, SUBMISSION_COLS

APPROACH_EXPLANATION = (
    "Our pipeline implements a physics-grounded digital twin. Operating conditions "
    "(voltage, current, ambient temperature, duration) predict expected thermal sensor "
    "responses (S1-S3). Tests with missing "
    "sensors, duplicate operating conditions, or sensor residuals exceeding dynamic "
    "empirical thresholds are classified as Invalid. Hot-spot reference temperatures "
    "are predicted by blending a degree-4 polynomial Ridge with gradient-boosted trees "
    "at a weight chosen out-of-fold, trained strictly on verified Valid records to "
    "eliminate corrupted targets and clipped to the observed physical range. "
    "Attention scoring prioritizes tests by "
    "combining invalidity status, sensor residual severity, missingness and "
    "temperature percentile rank for targeted engineering review."
)


def validate_submission(submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    """
    Strict validation of the submission dataframe against competition specifications.
    Fails loudly with descriptive error messages.
    """
    expected_rows = 350
    if len(submission_df) != expected_rows:
        raise ValueError(
            f"Submission row count mismatch: expected {expected_rows}, got {len(submission_df)}"
        )

    if list(submission_df.columns) != SUBMISSION_COLS:
        raise ValueError(
            f"Submission columns mismatch: expected {SUBMISSION_COLS}, got {list(submission_df.columns)}"
        )

    if len(sample_df) != expected_rows:
        raise ValueError(
            f"Sample submission row count unexpected: expected {expected_rows}, got {len(sample_df)}"
        )

    if not (submission_df["Test_ID"].values == sample_df["Test_ID"].values).all():
        mismatches = np.where(submission_df["Test_ID"].values != sample_df["Test_ID"].values)[0]
        raise ValueError(
            f"Test_ID alignment mismatch at {len(mismatches)} positions (first at index {mismatches[0]})"
        )

    preds = submission_df["Predicted_Reference_Parameter"]
    if preds.isna().any():
        nan_count = int(preds.isna().sum())
        raise ValueError(f"Found {nan_count} NaN values in Predicted_Reference_Parameter")

    if np.isinf(preds.values).any():
        inf_count = int(np.isinf(preds.values).sum())
        raise ValueError(f"Found {inf_count} infinite values in Predicted_Reference_Parameter")

    labels = submission_df["Validity_Label"]
    valid_labels = {"Valid", "Invalid"}
    actual_labels = set(labels.unique())
    invalid_labels = actual_labels - valid_labels
    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in Validity_Label: {invalid_labels}. Must only contain {valid_labels}"
        )


def compute_attention_scores(
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    twin: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Computes four normalized (0..1) attention components for each test record:
    1. c_invalid: Whether the record was flagged Invalid (1.0 or 0.0)
    2. c_residual: Severity of sensor residual relative to the threshold (0..1)
    3. c_missing: Count of missing measurements out of 8 possible (0..1)
    4. c_temp_percentile: Percentile rank of predicted Reference Parameter (0..1)

    Attention score is the unweighted mean of all 4 components.
    Returns (top_3_test_ids, diagnostics_dataframe).
    """
    n = len(submission_df)
    work = test_df.copy()

    # 1. Flagged Invalid (1.0 if Invalid, 0.0 if Valid)
    c_invalid = (submission_df["Validity_Label"] == "Invalid").astype(float).values

    # 2. Residual severity relative to threshold (0..1)
    c_residual = np.zeros(n, dtype=float)
    if twin is not None and threshold is not None and threshold > 0:
        ops_present = [col for col in OPS if col in work.columns]
        if len(ops_present) == len(OPS):
            X = work[OPS].values
            max_res = np.zeros(n, dtype=float)
            for s in S3:
                if s in work.columns and s in twin:
                    try:
                        preds = twin[s].predict(X)
                        res = np.abs(work[s].values - preds)
                        # Replace NaN residuals (e.g. from missing sensor) with 2x threshold
                        res = np.where(np.isnan(res), threshold * 2.0, res)
                        max_res = np.maximum(max_res, res)
                    except Exception:
                        pass
            # Normalize relative to threshold, capped at 1.0
            c_residual = np.clip(max_res / threshold, 0.0, 1.0)
    else:
        # If no twin/threshold provided, correlate residual severity with invalidity
        c_residual = c_invalid.copy()

    # 3. Missing-measurement count (normalized by total 8 measurement features)
    meas_cols = [c for c in MEASUREMENTS if c in work.columns]
    if meas_cols:
        missing_count = work[meas_cols].isna().sum(axis=1).values
        c_missing = np.clip(missing_count / float(len(MEASUREMENTS)), 0.0, 1.0)
    else:
        c_missing = np.zeros(n, dtype=float)

    # 4. Percentile rank of predicted Reference Parameter (0..1)
    preds = submission_df["Predicted_Reference_Parameter"].values
    if n > 1:
        # rank 0 to n-1, normalized to [0, 1]
        ranks = pd.Series(preds).rank(method="min").values - 1.0
        c_temp_percentile = ranks / float(n - 1)
    else:
        c_temp_percentile = np.zeros(n, dtype=float)

    # Equal weighting (25% each)
    attention_score = (c_invalid + c_residual + c_missing + c_temp_percentile) / 4.0

    diagnostics_df = pd.DataFrame({
        "Test_ID": submission_df["Test_ID"].values,
        "Predicted_Reference_Parameter": preds,
        "Validity_Label": submission_df["Validity_Label"].values,
        "c_invalid": np.round(c_invalid, 4),
        "c_residual": np.round(c_residual, 4),
        "c_missing": np.round(c_missing, 4),
        "c_temp_percentile": np.round(c_temp_percentile, 4),
        "attention_score": np.round(attention_score, 4),
    })

    # Sort descending by attention_score, then predicted temp, then Test_ID
    diagnostics_df = diagnostics_df.sort_values(
        by=["attention_score", "Predicted_Reference_Parameter", "Test_ID"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    diagnostics_df["rank"] = np.arange(1, len(diagnostics_df) + 1)

    top_3_ids = list(diagnostics_df.iloc[:3]["Test_ID"])

    return top_3_ids, diagnostics_df


def build_summary(
    submission_df: pd.DataFrame,
    attention_ids: List[str],
    explanation: str = APPROACH_EXPLANATION,
) -> Dict[str, Any]:
    """
    Builds the summary.json dictionary computing all values directly from final predictions.
    Enforces <= 100 words in approach_explanation.
    """
    word_count = len(explanation.strip().split())
    if word_count > 100:
        raise ValueError(
            f"Approach explanation exceeds 100 words limit: contains {word_count} words"
        )

    if len(attention_ids) != 3:
        raise ValueError(
            f"Expected exactly 3 attention Test IDs, got {len(attention_ids)}: {attention_ids}"
        )

    preds = submission_df["Predicted_Reference_Parameter"]
    abnormal_count = int((submission_df["Validity_Label"] == "Invalid").sum())

    summary = {
        "records_analysed": int(len(submission_df)),
        "abnormal_invalid_records": abnormal_count,
        "minimum_predicted_reference_parameter": float(round(float(preds.min()), 4)),
        "maximum_predicted_reference_parameter": float(round(float(preds.max()), 4)),
        "average_predicted_reference_parameter": float(round(float(preds.mean()), 4)),
        "highest_attention_test_ids": list(attention_ids),
        "approach_explanation": explanation.strip(),
    }
    return summary


def write_submission(submission_df: pd.DataFrame, output_path: Path) -> Path:
    """Writes submission to CSV without index column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path


def write_summary(summary_dict: Dict[str, Any], output_path: Path) -> Path:
    """Writes summary to formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    return output_path


def write_diagnostics(diagnostics_df: pd.DataFrame, output_path: Path) -> Path:
    """Writes attention diagnostics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_df.to_csv(output_path, index=False)
    return output_path
