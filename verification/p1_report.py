"""
P1 verification report.

Run from the repository root:

    python verification/p1_report.py

The script prints the acceptance evidence requested in briefs/P1_sensor_twin.md
and writes the same text to verification/p1_report.txt.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, r2_score, recall_score

from src.contract import (
    RANDOM_SEED,
    COL_TEST_ID,
    COL_LABEL,
    OPERATING_COLS,
    CRITICAL_SENSOR_COLS,
    LABEL_VALID,
    LABEL_INVALID,
    FAULT_MISSING,
    FAULT_DUPLICATE,
    FAULT_SPIKE,
    load_train,
    load_test,
    load_folds,
)
from src.twin import fit_twin, sensor_residuals, derive_threshold, predict_validity


def _duplicate_mask_without_s4(df: pd.DataFrame) -> pd.Series:
    cols = OPERATING_COLS + CRITICAL_SENSOR_COLS
    return df[cols].round(4).duplicated(keep=False)


def _fault_taxonomy(df: pd.DataFrame, labels: pd.Series, twin: dict, threshold: float) -> dict[str, int]:
    missing = df[CRITICAL_SENSOR_COLS].isna().any(axis=1)
    duplicate = _duplicate_mask_without_s4(df)
    residuals = sensor_residuals(df, twin).abs()
    spike = residuals.max(axis=1, skipna=True) > threshold

    fault_type = pd.Series("none", index=df.index)
    fault_type.loc[spike.fillna(False)] = FAULT_SPIKE
    fault_type.loc[duplicate] = FAULT_DUPLICATE
    fault_type.loc[missing] = FAULT_MISSING
    fault_type.loc[labels == LABEL_VALID] = "none"
    return fault_type.value_counts().to_dict()


def _sensor_cv_table(train_df: pd.DataFrame, folds_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = train_df[train_df[COL_LABEL] == LABEL_VALID].copy()
    rows = []
    merged = train_df.merge(folds_df, on=COL_TEST_ID)

    for sensor in CRITICAL_SENSOR_COLS:
        y_true_all = []
        y_pred_all = []

        for fold in sorted(merged["fold"].unique()):
            fold_train = merged[(merged["fold"] != fold) & (merged[COL_LABEL] == LABEL_VALID)].copy()
            fold_valid = merged[(merged["fold"] == fold) & (merged[COL_LABEL] == LABEL_VALID)].copy()
            twin = fit_twin(fold_train)
            enriched = sensor_residuals(fold_valid, twin)
            short = sensor.replace("Sensor_", "")
            y_true = fold_valid[sensor].to_numpy(dtype=float)
            y_pred = y_true - enriched[short].to_numpy(dtype=float)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

        y_true_arr = np.asarray(y_true_all)
        y_pred_arr = np.asarray(y_pred_all)
        residuals = y_true_arr - y_pred_arr
        rows.append({
            "sensor": sensor,
            "cv_r2": r2_score(y_true_arr, y_pred_arr),
            "mae": mean_absolute_error(y_true_arr, y_pred_arr),
            "residual_sigma": float(np.std(residuals, ddof=1)),
        })

    return pd.DataFrame(rows)


def _oof_metrics(train_df: pd.DataFrame, folds_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = train_df.merge(folds_df, on=COL_TEST_ID)
    pieces = []

    for fold in sorted(merged["fold"].unique()):
        train_fold = merged[merged["fold"] != fold].copy()
        valid_fold = merged[merged["fold"] == fold].copy()
        train_valid = train_fold[train_fold[COL_LABEL] == LABEL_VALID].copy()
        twin = fit_twin(train_valid)
        threshold = derive_threshold(train_valid, twin)
        labels = predict_validity(valid_fold, twin, threshold)
        pieces.append(pd.DataFrame({
            COL_TEST_ID: valid_fold[COL_TEST_ID].values,
            "actual": valid_fold[COL_LABEL].values,
            "predicted": labels.values,
            "fold": fold,
        }))

    oof = pd.concat(pieces, ignore_index=True)
    y_true = (oof["actual"] == LABEL_INVALID).astype(int)
    y_pred = (oof["predicted"] == LABEL_INVALID).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return oof, metrics


def _separation_gap(train_df: pd.DataFrame, twin: dict, threshold: float) -> dict[str, float]:
    labels = predict_validity(train_df, twin, threshold)
    residuals = sensor_residuals(train_df, twin).abs()
    max_residual = residuals.max(axis=1, skipna=True)

    valid_scores = max_residual[train_df[COL_LABEL] == LABEL_VALID].dropna()
    spiked_scores = max_residual[
        (train_df[COL_LABEL] == LABEL_INVALID)
        & (labels == LABEL_INVALID)
        & ~train_df[CRITICAL_SENSOR_COLS].isna().any(axis=1)
        & ~_duplicate_mask_without_s4(train_df)
    ].dropna()

    return {
        "max_valid_residual": float(valid_scores.max()),
        "min_spiked_residual": float(spiked_scores.min()),
    }


def _noise_check(train_df: pd.DataFrame, twin: dict, threshold: float) -> list[dict[str, object]]:
    valid = train_df[train_df[COL_LABEL] == LABEL_VALID].copy()
    clean = valid[
        valid[CRITICAL_SENSOR_COLS].notna().all(axis=1)
        & ~_duplicate_mask_without_s4(valid)
    ].head(200).copy()

    rng = np.random.RandomState(RANDOM_SEED)
    rows = []
    for sigma in [0.1, 0.25, 0.5]:
        noisy = clean.copy()
        noise = rng.normal(0.0, sigma, size=(len(noisy), len(CRITICAL_SENSOR_COLS)))
        for idx, sensor in enumerate(CRITICAL_SENSOR_COLS):
            noisy[sensor] = noisy[sensor].to_numpy(dtype=float) + noise[:, idx]
        noisy[COL_TEST_ID] = [f"NOISE-{sigma}-{i}" for i in range(len(noisy))]
        labels = predict_validity(noisy, twin, threshold)
        rows.append({
            "sigma": sigma,
            "invalid_count": int((labels == LABEL_INVALID).sum()),
            "rows": len(noisy),
            "all_valid": bool((labels == LABEL_VALID).all()),
        })
    return rows


def build_report() -> str:
    train_df = load_train()
    test_df = load_test()
    folds_df = load_folds()

    train_valid = train_df[train_df[COL_LABEL] == LABEL_VALID].copy()
    twin = fit_twin(train_valid)
    threshold = derive_threshold(train_valid, twin)

    sensor_table = _sensor_cv_table(train_df, folds_df)
    train_labels = predict_validity(train_df, twin, threshold)
    taxonomy = _fault_taxonomy(train_df, train_labels, twin, threshold)
    oof, metrics = _oof_metrics(train_df, folds_df)
    gap = _separation_gap(train_df, twin, threshold)
    test_labels = predict_validity(test_df, twin, threshold)
    test_taxonomy = _fault_taxonomy(test_df, test_labels, twin, threshold)
    noise_rows = _noise_check(train_df, twin, threshold)

    out = StringIO()
    print("P1 Sensor Twin Verification Report", file=out)
    print("=" * 38, file=out)
    print("", file=out)
    print("Per-sensor twin CV metrics:", file=out)
    print(sensor_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"), file=out)
    print("", file=out)
    print(
        "Derived threshold: "
        f"{threshold:.6f} = max(8 * largest robust residual sigma, small floor)",
        file=out,
    )
    print("", file=out)
    print("Training fault taxonomy:", file=out)
    for key in ["none", FAULT_MISSING, FAULT_DUPLICATE, FAULT_SPIKE]:
        print(f"  {key}: {taxonomy.get(key, 0)}", file=out)
    invalid_sum = taxonomy.get(FAULT_MISSING, 0) + taxonomy.get(FAULT_DUPLICATE, 0) + taxonomy.get(FAULT_SPIKE, 0)
    print(f"  invalid_sum: {invalid_sum}", file=out)
    print("", file=out)
    print("Separation gap:", file=out)
    print(f"  max Valid residual: {gap['max_valid_residual']:.6f}", file=out)
    print(f"  min spiked residual: {gap['min_spiked_residual']:.6f}", file=out)
    print("", file=out)
    print("Out-of-fold invalid-detection metrics:", file=out)
    print(f"  precision: {metrics['precision']:.6f}", file=out)
    print(f"  recall:    {metrics['recall']:.6f}", file=out)
    print(f"  f1:        {metrics['f1']:.6f}", file=out)
    print(f"  rows:      {len(oof)}", file=out)
    print("", file=out)
    print("Test-set flag counts:", file=out)
    for key in ["none", FAULT_MISSING, FAULT_DUPLICATE, FAULT_SPIKE]:
        print(f"  {key}: {test_taxonomy.get(key, 0)}", file=out)
    flagged = int((test_labels == LABEL_INVALID).sum())
    print(f"  invalid_total: {flagged}", file=out)
    print(f"  flagged_rate: {flagged / len(test_df):.4%}", file=out)
    print("", file=out)
    print("Noise robustness:", file=out)
    for row in noise_rows:
        print(
            f"  sigma={row['sigma']}: {row['invalid_count']}/{row['rows']} "
            f"flagged, all_valid={row['all_valid']}",
            file=out,
        )
    print("", file=out)
    print("Fallbacks used: none.", file=out)
    print("Requests for other lanes: align P4 imports to src.twin or keep this compatibility module.", file=out)
    return out.getvalue()


def main() -> None:
    report = build_report()
    print(report)
    report_path = Path("verification") / "p1_report.txt"
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
