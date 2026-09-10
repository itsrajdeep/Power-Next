"""
Integration QA: Robustness Attack Test Suite.
Verifies the pipeline resilience against deliberately corrupted/edge-case test data.
Protects the 20% unseen-dataset evaluation criterion.
"""
import numpy as np
import pandas as pd
import pytest

from src.contract import load_data, valid_rows, SENSORS, MEASUREMENTS
from src.finalize import validate_submission

# Import lane modules or fallback stubs
try:
    from src.twin import fit_twin, derive_threshold, predict_validity
except ImportError:
    from src.twin_stub import fit_twin, derive_threshold, predict_validity

try:
    from src.regress import fit_reference_model, predict_reference
except ImportError:
    from src.regress_stub import fit_reference_model, predict_reference


@pytest.fixture(scope="module")
def fitted_models():
    """Fits sensor twin and regression models on Valid training data."""
    train, test, sample = load_data()
    train_valid = valid_rows(train)
    twin = fit_twin(train_valid)
    threshold = derive_threshold(train_valid, twin)
    model = fit_reference_model(train_valid)
    return {
        "twin": twin,
        "threshold": threshold,
        "model": model,
        "test": test,
        "sample": sample,
    }


def test_attack_1_randomly_nan_sensor_readings(fitted_models):
    """
    Attack: Randomly NaN 10% of sensor readings.
    Expectation: Runs without crashing, produces 350 valid rows.
    """
    test_df = fitted_models["test"].copy()
    twin = fitted_models["twin"]
    threshold = fitted_models["threshold"]
    model = fitted_models["model"]
    sample = fitted_models["sample"]

    rng = np.random.default_rng(42)
    for col in SENSORS:
        if col in test_df.columns:
            mask = rng.random(len(test_df)) < 0.10
            test_df.loc[mask, col] = np.nan

    labels = predict_validity(test_df, twin, threshold)
    preds = predict_reference(test_df, model)

    submission = pd.DataFrame({
        "Test_ID": test_df["Test_ID"],
        "Predicted_Reference_Parameter": preds,
        "Validity_Label": labels,
    })

    assert len(submission) == 350
    validate_submission(submission, sample)


def test_attack_2_drop_sensor_s4_entirely(fitted_models):
    """
    Attack: Drop the Sensor_S4 column entirely.
    Expectation: Runs without crashing, verdicts remain unchanged.
    """
    test_df = fitted_models["test"].copy()
    twin = fitted_models["twin"]
    threshold = fitted_models["threshold"]

    base_labels = predict_validity(test_df, twin, threshold)

    # Drop Sensor_S4
    corrupted_df = test_df.drop(columns=["Sensor_S4"])
    new_labels = predict_validity(corrupted_df, twin, threshold)

    assert (new_labels.values == base_labels.values).all(), (
        "Dropping irrelevant Sensor_S4 altered validity verdicts."
    )


def test_attack_3_duplicate_20_random_rows(fitted_models):
    """
    Attack: Duplicate 20 random rows within the test set.
    Expectation: Runs, and duplicate rows are flagged Invalid.
    """
    test_df = fitted_models["test"].copy()
    twin = fitted_models["twin"]
    threshold = fitted_models["threshold"]

    rng = np.random.default_rng(123)
    src_indices = rng.choice(len(test_df), size=20, replace=False)
    dst_indices = [i for i in range(len(test_df)) if i not in src_indices][:20]

    # Duplicate measurement vectors from src to dst
    meas_cols = [c for c in MEASUREMENTS if c in test_df.columns]
    test_df.loc[dst_indices, meas_cols] = test_df.loc[src_indices, meas_cols].values

    labels = predict_validity(test_df, twin, threshold)

    # Both source and destination duplicates must be flagged Invalid
    assert (labels.iloc[src_indices] == "Invalid").all(), "Source duplicate rows were not flagged Invalid."
    assert (labels.iloc[dst_indices] == "Invalid").all(), "Destination duplicate rows were not flagged Invalid."


def test_attack_4_multiply_currents_by_1_3(fitted_models):
    """
    Attack: Multiply all load currents by 1.3 (beyond training range).
    Expectation: Runs, predictions stay finite and sensible.
    """
    test_df = fitted_models["test"].copy()
    model = fitted_models["model"]

    test_df["Load_Current_A"] = test_df["Load_Current_A"] * 1.3
    preds = predict_reference(test_df, model)

    assert len(preds) == 350
    assert not preds.isna().any(), "NaN found under 1.3x current load."
    assert np.isfinite(preds.values).all(), "Non-finite predictions under 1.3x current load."
    assert (preds.values > 0).all(), "Negative temperatures predicted under 1.3x current load."
    assert (preds.values < 300).all(), "Unphysically extreme temperature predicted under 1.3x current load."


def test_attack_5_insert_all_zero_sensors(fitted_models):
    """
    Attack: Insert a test row of all-zero sensors.
    Expectation: Runs, flagged Invalid.
    """
    test_df = fitted_models["test"].copy()
    twin = fitted_models["twin"]
    threshold = fitted_models["threshold"]

    # Pick row 0 and zero out its sensors
    zero_idx = 0
    for col in SENSORS:
        if col in test_df.columns:
            test_df.loc[zero_idx, col] = 0.0

    labels = predict_validity(test_df, twin, threshold)
    assert labels.iloc[zero_idx] == "Invalid", "All-zero sensor row was not flagged Invalid."


def test_attack_6_shuffle_column_order(fitted_models):
    """
    Attack: Shuffle the column order of test data.
    Expectation: Runs, produces identical output.
    """
    test_df = fitted_models["test"].copy()
    twin = fitted_models["twin"]
    threshold = fitted_models["threshold"]
    model = fitted_models["model"]

    base_labels = predict_validity(test_df, twin, threshold)
    base_preds = predict_reference(test_df, model)

    # Permute columns
    rng = np.random.default_rng(999)
    shuffled_cols = list(test_df.columns)
    rng.shuffle(shuffled_cols)
    shuffled_df = test_df[shuffled_cols]

    new_labels = predict_validity(shuffled_df, twin, threshold)
    new_preds = predict_reference(shuffled_df, model)

    assert (new_labels.values == base_labels.values).all(), "Column shuffle altered validity labels."
    assert np.allclose(new_preds.values, base_preds.values, atol=1e-5), "Column shuffle altered predictions."
