"""
SHARED CONTRACT — do not modify without telling the whole team.

Every lane imports from this module so that all four people are guaranteed to be
loading the same data, grouping duplicates the same way, and validating on the
same folds. Everything here is deterministic: any teammate running it gets a
byte-identical result, so it does not matter who runs it first.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "dataset" / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
FOLDS_CSV = ROOT / "folds.csv"

OPS = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
S3 = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]      # the three real sensors
S4 = "Sensor_S4"                                   # verified irrelevant, excluded from models
SENSORS = S3 + [S4]
MEASUREMENTS = OPS + SENSORS                       # the 8 columns that define a test

TARGET_REG = "Reference_Parameter"
TARGET_CLS = "Validity_Label"

SUBMISSION_COLS = ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data():
    """Return (train, test, sample_submission). Never mutates the workbook."""
    xls = pd.ExcelFile(XLSX)
    train = pd.read_excel(xls, "Training_Data")
    test = pd.read_excel(xls, "Test_Data")
    sample = pd.read_excel(xls, "Sample_Submission")
    return train, test, sample


def is_invalid(train: pd.DataFrame) -> pd.Series:
    """Binary target for classification: 1 = Invalid, 0 = Valid."""
    return (train[TARGET_CLS] == "Invalid").astype(int)


def valid_rows(train: pd.DataFrame) -> pd.DataFrame:
    """The 866 engineer-verified Valid rows. Train regression on these only."""
    return train[train[TARGET_CLS] == "Valid"].copy()


# ----------------------------------------------------------------------------
# Duplicate handling
# ----------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> pd.Series:
    """
    Identity of a test: the 8 measurement columns rounded to 4dp and joined.
    Two rows with the same fingerprint are the same physical measurement.
    NaN is mapped to a sentinel so missing values compare equal to each other.
    """
    return (df[MEASUREMENTS]
            .round(4)
            .fillna(-9e9)
            .astype(str)
            .agg("|".join, axis=1))


def duplicate_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group id per row. Duplicate rows share a group id."""
    return pd.factorize(fingerprint(df))[0]


def is_duplicate(df: pd.DataFrame) -> pd.Series:
    """True for every row whose measurement vector appears more than once.

    IMPORTANT: compute this within a single dataframe only. Never concatenate
    train and test before calling this.
    """
    return df.duplicated(subset=MEASUREMENTS, keep=False)


# ----------------------------------------------------------------------------
# Validation folds — the single source of truth for every experiment
# ----------------------------------------------------------------------------
def build_folds(train: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic 5-fold split, stratified on Validity_Label and grouped so that
    duplicate measurement vectors never straddle a fold boundary.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = is_invalid(train)
    groups = duplicate_groups(train)
    out = pd.DataFrame({"Test_ID": train["Test_ID"], "fold": -1})
    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(splitter.split(train, y, groups)):
        out.iloc[test_idx, out.columns.get_loc("fold")] = k
    assert (out.fold >= 0).all(), "every row must be assigned a fold"
    return out


def get_folds(train: pd.DataFrame = None) -> pd.DataFrame:
    """Load folds.csv, generating and saving it on first use."""
    if FOLDS_CSV.exists():
        return pd.read_csv(FOLDS_CSV)
    if train is None:
        train, _, _ = load_data()
    folds = build_folds(train)
    folds.to_csv(FOLDS_CSV, index=False)
    return folds


def fold_indices(train: pd.DataFrame):
    """Yield (train_idx, valid_idx) positional arrays for each of the 5 folds."""
    folds = get_folds(train)
    merged = train[["Test_ID"]].merge(folds, on="Test_ID", how="left")
    assert merged.fold.notna().all(), "Test_ID present in train but missing from folds.csv"
    f = merged.fold.values
    for k in range(N_FOLDS):
        yield np.where(f != k)[0], np.where(f == k)[0]


def assert_no_group_leakage(train: pd.DataFrame) -> None:
    """Fail loudly if any duplicate group is split across folds."""
    folds = get_folds(train)
    g = pd.DataFrame({"g": duplicate_groups(train),
                      "fold": train[["Test_ID"]].merge(folds, on="Test_ID", how="left").fold.values})
    bad = g.groupby("g").fold.nunique()
    assert (bad == 1).all(), f"{int((bad > 1).sum())} duplicate group(s) straddle a fold boundary"


if __name__ == "__main__":
    train, test, sample = load_data()
    folds = get_folds(train)
    assert_no_group_leakage(train)
    print(f"train={train.shape}  test={test.shape}  sample={sample.shape}")
    print(f"Valid={(train[TARGET_CLS] == 'Valid').sum()}  Invalid={is_invalid(train).sum()}")
    print(f"folds written to {FOLDS_CSV}")
    print(folds.fold.value_counts().sort_index().to_string())
    print("no duplicate group straddles a fold — contract OK")
