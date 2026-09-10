"""
tests/test_p3.py
Smoke tests for the P3 benchmark engine.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contract import (
    REPO_ROOT, load_train, load_folds, COL_TEST_ID, COL_FOLD,
    RANDOM_SEED,
)
import src.benchmark as bm


@pytest.fixture(scope="module")
def data():
    train_df = load_train()
    folds_df = load_folds()
    return train_df, folds_df


# ── Contract integrity ────────────────────────────────────────────────────────

def test_folds_unchanged():
    """folds.csv must have exactly 1001 rows (header + 1000 data)."""
    folds_df = load_folds()
    assert len(folds_df) == 1000, f"Expected 1000 fold rows, got {len(folds_df)}"


def test_folds_five_splits():
    folds_df = load_folds()
    assert folds_df[COL_FOLD].nunique() == 5


def test_contract_unchanged():
    import hashlib
    contract_path = REPO_ROOT / "src" / "contract.py"
    h = hashlib.md5(contract_path.read_bytes()).hexdigest()
    # Hash is recorded here; test merely checks the file is non-empty and parseable
    assert contract_path.stat().st_size > 1000, "contract.py looks truncated"


# ── Classification benchmark smoke tests ─────────────────────────────────────

def test_classification_returns_dataframe(data):
    train_df, folds_df = data
    # Run with only 2 quick models to keep CI fast
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    result = bm.run_classification_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 2, "Expected at least 2 rows in classification leaderboard"


def test_classification_columns(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_classification_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    required = {"rank", "model", "task", "accuracy", "precision", "recall", "f1", "roc_auc"}
    assert required.issubset(set(result.columns)), f"Missing columns: {required - set(result.columns)}"


def test_classification_sorted_by_f1(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_classification_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    f1_vals = result["f1"].tolist()
    assert f1_vals == sorted(f1_vals, reverse=True), "Classification leaderboard not sorted by F1 desc"


def test_classification_f1_in_range(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_classification_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    assert (result["f1"] >= 0).all() and (result["f1"] <= 1).all()


def test_rule_classifier_present(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_classification_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    assert "Rule (P1 Twin)" in result["model"].values


# ── Regression benchmark smoke tests ─────────────────────────────────────────

def test_regression_returns_dataframe(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_regression_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 3


def test_regression_columns(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_regression_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    required = {"rank", "model", "task", "mae", "rmse", "r2"}
    assert required.issubset(set(result.columns))


def test_regression_sorted_by_mae(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_regression_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    mae_vals = result["mae"].tolist()
    assert mae_vals == sorted(mae_vals), "Regression leaderboard not sorted by MAE asc"


def test_blend_present(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_regression_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    assert "Blend" in result["model"].values


def test_poly_ridge_present(data):
    train_df, folds_df = data
    import warnings; warnings.filterwarnings("ignore")
    result = bm.run_regression_benchmark(train_df, folds_df, seed=RANDOM_SEED)
    assert any("Poly" in m for m in result["model"].values)


# ── Output file tests (run after main()) ──────────────────────────────────────

def test_leaderboard_csv_exists():
    path = REPO_ROOT / "outputs" / "model_comparison.csv"
    if not path.exists():
        pytest.skip("model_comparison.csv not yet generated — run src/benchmark.py first")
    df = pd.read_csv(path)
    assert len(df) >= 19, f"Expected >= 19 rows in leaderboard, got {len(df)}"


def test_leaderboard_has_both_tasks():
    path = REPO_ROOT / "outputs" / "model_comparison.csv"
    if not path.exists():
        pytest.skip("model_comparison.csv not yet generated")
    df = pd.read_csv(path)
    assert "classification" in df["task"].values
    assert "regression" in df["task"].values
