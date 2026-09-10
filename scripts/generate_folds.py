"""
Generate folds.csv for Power-Next cross-validation.

Stratified GroupKFold-like split:
- 5 folds
- Duplicate fingerprint groups stay together in one fold
- Stratified by Validity_Label
- Deterministic with RANDOM_SEED = 42
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.contract import (
    DATASET_PATH, TRAIN_SHEET, RANDOM_SEED,
    FINGERPRINT_COLS, COL_TEST_ID, COL_LABEL,
    fingerprint_series,
)

def main():
    train = pd.read_excel(DATASET_PATH, sheet_name=TRAIN_SHEET)
    assert len(train) == 1000, f"Expected 1000 rows, got {len(train)}"

    # Compute fingerprints
    fps = fingerprint_series(train)
    
    # Assign group IDs: rows with the same fingerprint get the same group
    fp_to_group = {}
    group_id = 0
    groups = []
    for fp in fps:
        if fp not in fp_to_group:
            fp_to_group[fp] = group_id
            group_id += 1
        groups.append(fp_to_group[fp])
    train["_group"] = groups
    
    # Get unique groups with their label (for stratification, use the label of the first row)
    group_df = train.groupby("_group").agg(
        label=(COL_LABEL, "first"),
        count=(COL_TEST_ID, "count"),
    ).reset_index()
    
    # Stratified split on groups
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    group_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(group_df, group_df["label"])):
        group_df.loc[val_idx, "fold"] = fold_idx
    
    # Map back to rows
    group_to_fold = dict(zip(group_df["_group"], group_df["fold"]))
    train["fold"] = train["_group"].map(group_to_fold)
    
    # Validate
    assert train["fold"].min() >= 0
    assert train["fold"].max() <= 4
    assert len(train["fold"].unique()) == 5
    assert train["fold"].notna().all()
    
    # Check no fingerprint group crosses folds
    fps_with_fold = train[["_group", "fold"]].drop_duplicates()
    groups_multi_fold = fps_with_fold.groupby("_group")["fold"].nunique()
    assert (groups_multi_fold == 1).all(), "Some fingerprint groups cross folds!"
    
    # Save
    folds_out = train[[COL_TEST_ID, "fold"]].copy()
    folds_out["fold"] = folds_out["fold"].astype(int)
    folds_out.to_csv("folds.csv", index=False)
    
    print(f"Generated folds.csv with {len(folds_out)} rows")
    print(f"Fold distribution:\n{folds_out['fold'].value_counts().sort_index()}")
    
    # Print label distribution per fold
    merged = train[[COL_TEST_ID, COL_LABEL, "fold"]]
    for f in range(5):
        subset = merged[merged["fold"] == f]
        print(f"Fold {f}: {len(subset)} rows, {dict(subset[COL_LABEL].value_counts())}")

if __name__ == "__main__":
    main()
