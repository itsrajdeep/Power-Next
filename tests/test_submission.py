"""
Validation and Quality Assurance Test Suite for Deliverables.
Tests submission CSV, summary.json, and diagnostics compliance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.contract import load_data, SUBMISSION_COLS, MEASUREMENTS
from run import run_pipeline, DEFAULT_TEAM_NAME, DEFAULT_OUTPUT_DIR


@pytest.fixture(scope="session")
def pipeline_outputs():
    """Ensures pipeline is executed and returns deliverable filepaths."""
    team_name = "TestTeam"
    output_dir = Path("outputs/final")
    csv_path = output_dir / f"{team_name}.csv"
    summary_path = output_dir / "summary.json"
    diag_path = output_dir / "test_diagnostics.csv"

    # Run pipeline to ensure fresh outputs
    exit_code = run_pipeline(team_name=team_name, output_dir_str=str(output_dir), skip_charts=True)
    assert exit_code == 0, f"Pipeline execution failed with exit code {exit_code}"

    assert csv_path.exists(), f"Missing CSV: {csv_path}"
    assert summary_path.exists(), f"Missing summary: {summary_path}"
    assert diag_path.exists(), f"Missing diagnostics: {diag_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    submission_df = pd.read_csv(csv_path)
    diag_df = pd.read_csv(diag_path)
    _, test_df, sample_df = load_data()

    return {
        "submission_df": submission_df,
        "summary_data": summary_data,
        "diag_df": diag_df,
        "sample_df": sample_df,
        "test_df": test_df,
        "csv_path": csv_path,
    }


def test_submission_shape_and_columns(pipeline_outputs):
    """Submission must have exactly 350 rows and the exact required columns."""
    df = pipeline_outputs["submission_df"]
    assert df.shape == (350, 3), f"Expected shape (350, 3), got {df.shape}"
    assert list(df.columns) == SUBMISSION_COLS, (
        f"Columns mismatch. Expected {SUBMISSION_COLS}, got {list(df.columns)}"
    )


def test_no_index_column_written(pipeline_outputs):
    """CSV must not contain an unnamed pandas index column."""
    csv_path = pipeline_outputs["csv_path"]
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert first_line == ",".join(SUBMISSION_COLS), (
        f"CSV header contains extra/index column: {first_line}"
    )


def test_test_ids_match_sample_submission(pipeline_outputs):
    """Test IDs must match Sample_Submission sheet exactly in same order."""
    sub_df = pipeline_outputs["submission_df"]
    sample_df = pipeline_outputs["sample_df"]
    assert (sub_df["Test_ID"].values == sample_df["Test_ID"].values).all(), (
        "Test_ID order or values do not match Sample_Submission exactly."
    )


def test_no_duplicate_or_missing_test_ids(pipeline_outputs):
    """Ensure no duplicate Test IDs and all 350 test IDs are present."""
    sub_df = pipeline_outputs["submission_df"]
    assert len(sub_df["Test_ID"].unique()) == 350, "Duplicate Test_IDs found in submission."
    assert not sub_df["Test_ID"].isna().any(), "Missing/NaN Test_ID found in submission."


def test_predictions_no_nan_or_infinite(pipeline_outputs):
    """Predicted_Reference_Parameter must be finite with no NaN or infinity."""
    preds = pipeline_outputs["submission_df"]["Predicted_Reference_Parameter"]
    assert not preds.isna().any(), "Found NaN in Predicted_Reference_Parameter."
    assert np.isfinite(preds.values).all(), "Found infinite values in Predicted_Reference_Parameter."
    assert (preds.values > 0).all(), "Found negative or zero predicted temperatures."


def test_validity_labels_domain(pipeline_outputs):
    """Validity_Label must only contain strings 'Valid' or 'Invalid'."""
    labels = pipeline_outputs["submission_df"]["Validity_Label"]
    unique_labels = set(labels.unique())
    assert unique_labels.issubset({"Valid", "Invalid"}), (
        f"Unexpected labels found: {unique_labels - {'Valid', 'Invalid'}}"
    )


def test_summary_json_schema(pipeline_outputs):
    """summary.json must contain all seven keys with correct types."""
    summary = pipeline_outputs["summary_data"]
    required_keys = {
        "records_analysed": int,
        "abnormal_invalid_records": int,
        "minimum_predicted_reference_parameter": float,
        "maximum_predicted_reference_parameter": float,
        "average_predicted_reference_parameter": float,
        "highest_attention_test_ids": list,
        "approach_explanation": str,
    }
    for key, expected_type in required_keys.items():
        assert key in summary, f"summary.json is missing key '{key}'"
        assert isinstance(summary[key], expected_type), (
            f"Key '{key}' has type {type(summary[key])}, expected {expected_type}"
        )


def test_summary_statistics_agree_with_csv(pipeline_outputs):
    """Values in summary.json must match the submission CSV exactly."""
    summary = pipeline_outputs["summary_data"]
    sub_df = pipeline_outputs["submission_df"]
    preds = sub_df["Predicted_Reference_Parameter"]

    assert summary["records_analysed"] == len(sub_df)
    assert summary["abnormal_invalid_records"] == int((sub_df["Validity_Label"] == "Invalid").sum())
    assert summary["minimum_predicted_reference_parameter"] == pytest.approx(float(preds.min()), abs=1e-3)
    assert summary["maximum_predicted_reference_parameter"] == pytest.approx(float(preds.max()), abs=1e-3)
    assert summary["average_predicted_reference_parameter"] == pytest.approx(float(preds.mean()), abs=1e-3)


def test_approach_explanation_word_count(pipeline_outputs):
    """Approach explanation must be 100 words or fewer."""
    explanation = pipeline_outputs["summary_data"]["approach_explanation"]
    words = explanation.strip().split()
    assert len(words) <= 100, f"Explanation exceeds 100 words: count is {len(words)}"


def test_highest_attention_ids(pipeline_outputs):
    """summary.json must list exactly three attention IDs present in the test set."""
    summary = pipeline_outputs["summary_data"]
    attention_ids = summary["highest_attention_test_ids"]
    assert len(attention_ids) == 3, f"Expected 3 attention IDs, got {len(attention_ids)}"
    test_ids = set(pipeline_outputs["test_df"]["Test_ID"])
    for tid in attention_ids:
        assert tid in test_ids, f"Attention ID '{tid}' not present in test dataset."


def test_identical_measurement_vectors_receive_identical_outputs(pipeline_outputs):
    """Identical measurement vectors in the test set must receive identical outputs."""
    sub_df = pipeline_outputs["submission_df"]
    test_df = pipeline_outputs["test_df"]

    merged = test_df.merge(sub_df, on="Test_ID", how="left")
    # Identify duplicate measurement vectors in test_df
    dup_mask = merged.duplicated(subset=MEASUREMENTS, keep=False)
    if dup_mask.any():
        dup_rows = merged[dup_mask]
        for _, group in dup_rows.groupby(MEASUREMENTS):
            labels = group["Validity_Label"].unique()
            preds = group["Predicted_Reference_Parameter"].unique()
            assert len(labels) == 1, f"Conflicting labels for identical measurement vector: {labels}"
            assert len(preds) == 1, f"Conflicting predictions for identical measurement vector: {preds}"
