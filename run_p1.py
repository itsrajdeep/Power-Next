"""
P1 runner – OOF evaluation + final fit + test prediction + diagnostics.
Produces all outputs under outputs/p1/.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.contract import (
    DATASET_PATH, FOLDS_PATH, RANDOM_SEED,
    COL_TEST_ID, COL_LABEL,
    load_train, load_test, load_folds,
    LABEL_INVALID, LABEL_VALID,
)
from src.p1_twin import (
    fit_twin, sensor_residuals, predict_validity,
    run_oof_evaluation, final_fit_and_predict,
)


def main() -> None:
    output_dir = Path("outputs/p1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df = load_train()
    test_df = load_test()
    folds_df = load_folds()

    print(f"Training: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print(f"Folds: {len(folds_df)} rows")

    # ── OOF Evaluation ──────────────────────────────────────────────────
    print("\n=== OOF Evaluation ===")
    oof_df, diagnostics = run_oof_evaluation(
        train_df, folds_df,
        alpha=1.0,
        degree=4,
        random_seed=RANDOM_SEED,
    )

    # Save OOF predictions
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    print(f"OOF predictions saved ({len(oof_df)} rows)")

    # Print metrics
    print(f"\nPrecision (Invalid): {diagnostics['precision']:.4f}")
    print(f"Recall    (Invalid): {diagnostics['recall']:.4f}")
    print(f"F1        (Invalid): {diagnostics['f1']:.4f}")

    cm = diagnostics["confusion_matrix"]
    print(f"\nConfusion Matrix:")
    print(f"  TP={cm['tp']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TN={cm['tn']}")

    print(f"\nFault Taxonomy:")
    for fault, count in sorted(diagnostics["taxonomy"].items()):
        print(f"  {fault}: {count}")

    print(f"\nPer-fold Thresholds:")
    for fold, td in sorted(diagnostics["per_fold_threshold"].items()):
        print(f"  Fold {fold}: threshold={td['threshold']:.4f}, "
              f"lower={td['lower']:.4f}, upper={td['upper']:.4f}, "
              f"gap={td['gap']:.4f}")

    print(f"\nPer-fold Metrics:")
    for fold, fm in sorted(diagnostics["per_fold_metrics"].items()):
        print(f"  Fold {fold}: P={fm['precision']:.4f}, R={fm['recall']:.4f}, "
              f"F1={fm['f1']:.4f}, TP={fm['tp']}, FP={fm['fp']}, "
              f"FN={fm['fn']}, TN={fm['tn']}")

    # Save metrics JSON
    with open(output_dir / "oof_metrics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print("\nOOF metrics saved to outputs/p1/oof_metrics.json")

    # ── Final Fit ────────────────────────────────────────────────────────
    print("\n=== Final Fit ===")
    test_preds, final_state = final_fit_and_predict(
        train_df, test_df,
        alpha=1.0,
        degree=4,
        random_seed=RANDOM_SEED,
    )

    # Save test predictions
    test_preds.to_csv(output_dir / "test_predictions.csv", index=False)
    print(f"Test predictions saved ({len(test_preds)} rows)")

    # Final threshold diagnostics
    ftd = final_state["threshold_diagnostics"]
    print(f"\nFinal threshold: {final_state['threshold']:.4f}")
    print(f"  lower={ftd['lower']:.4f}, upper={ftd['upper']:.4f}, gap={ftd['gap']:.4f}")

    # Save threshold diagnostics
    with open(output_dir / "threshold_diagnostics.json", "w") as f:
        json.dump({
            "final": ftd,
            "per_fold": diagnostics["per_fold_threshold"],
        }, f, indent=2, default=str)

    # Test prediction taxonomy
    print(f"\nTest Prediction Taxonomy:")
    for fault, count in sorted(test_preds["fault_type"].value_counts().items()):
        print(f"  {fault}: {count}")

    print("\n=== P1 Complete ===")


if __name__ == "__main__":
    main()
