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


# ----------------------------------------------------------------------------
# COMPATIBILITY LAYER
#
# Person 1 developed against a differently-named version of this contract before
# the shared one was merged. Rather than rewrite ~700 lines of working, verified
# detector code, the names that version exposed are re-exported here as aliases.
#
# These are aliases, not second definitions: every one of them points at the
# canonical object above, so there is exactly one source of truth for the data
# path, the fold assignment and the duplicate grouping.
#
# Do not add new code against these names. New work uses the canonical names.
# ----------------------------------------------------------------------------
REPO_ROOT = ROOT
DATASET_PATH = XLSX
FOLDS_PATH = FOLDS_CSV

TRAIN_SHEET = "Training_Data"
TEST_SHEET = "Test_Data"
SAMPLE_SHEET = "Sample_Submission"

RANDOM_SEED = SEED

COL_TEST_ID = "Test_ID"
COL_VOLTAGE, COL_CURRENT, COL_TEMP, COL_DURATION = OPS
COL_S1, COL_S2, COL_S3 = S3
COL_S4 = S4
COL_REF = TARGET_REG
COL_LABEL = TARGET_CLS
COL_FOLD = "fold"

OPERATING_COLS = OPS
SENSOR_COLS = SENSORS
CRITICAL_SENSOR_COLS = S3
FINGERPRINT_COLS = MEASUREMENTS

LABEL_VALID = "Valid"
LABEL_INVALID = "Invalid"

FAULT_NONE = "none"
FAULT_MISSING = "missing_sensor"
FAULT_DUPLICATE = "duplicate_vector"
FAULT_SPIKE = "sensor_spike"


def fingerprint_row(row_values, decimals: int = 4) -> str:
    """Fingerprint of a single row, given its FINGERPRINT_COLS values in order.

    Row-wise counterpart of `fingerprint`, which works on a whole frame. The two
    use different NaN sentinels, so never compare a string from one against a
    string from the other. Within either, NaN compares equal to NaN.
    """
    parts = []
    for v in row_values:
        parts.append("nan" if pd.isna(v) else str(round(float(v), decimals)))
    return "|".join(parts)


def fingerprint_series(df: pd.DataFrame, cols=None, decimals: int = 4) -> pd.Series:
    """`fingerprint_row` applied to every row, index-aligned with *df*."""
    use_cols = cols or FINGERPRINT_COLS
    return df[use_cols].apply(lambda r: fingerprint_row(r, decimals), axis=1)


def load_train(path=None) -> pd.DataFrame:
    """Training_Data sheet only."""
    return pd.read_excel(Path(path) if path else XLSX, sheet_name=TRAIN_SHEET)


def load_test(path=None) -> pd.DataFrame:
    """Test_Data sheet only."""
    return pd.read_excel(Path(path) if path else XLSX, sheet_name=TEST_SHEET)


def load_folds(path=None) -> pd.DataFrame:
    """folds.csv as written by `get_folds` (columns: Test_ID, fold)."""
    return pd.read_csv(Path(path) if path else FOLDS_CSV)


if __name__ == "__main__":
    train, test, sample = load_data()
    folds = get_folds(train)
    assert_no_group_leakage(train)
    print(f"train={train.shape}  test={test.shape}  sample={sample.shape}")
    print(f"Valid={(train[TARGET_CLS] == 'Valid').sum()}  Invalid={is_invalid(train).sum()}")
    print(f"folds written to {FOLDS_CSV}")
    print(folds.fold.value_counts().sort_index().to_string())
    print("no duplicate group straddles a fold — contract OK")
