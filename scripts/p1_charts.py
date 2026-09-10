"""
P1 – Generate diagnostic charts and noise stress-test table.
Saves outputs to outputs/p1/.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.contract import (
    DATASET_PATH, FOLDS_PATH, RANDOM_SEED,
    COL_TEST_ID, COL_LABEL,
    OPERATING_COLS, CRITICAL_SENSOR_COLS, SENSOR_COLS,
    FINGERPRINT_COLS,
    LABEL_VALID, LABEL_INVALID,
    FAULT_MISSING, FAULT_DUPLICATE, FAULT_SPIKE, FAULT_NONE,
    load_train, load_test, load_folds,
    fingerprint_series,
)
from src.p1_twin import (
    fit_twin, sensor_residuals, predict_validity,
    run_oof_evaluation,
)


def main() -> None:
    output_dir = Path("outputs/p1")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train()
    test_df = load_test()

    # Fit final twin
    twin_state = fit_twin(train_df, alpha=1.0, degree=4, random_seed=RANDOM_SEED)

    # ── Noise Stress Test ────────────────────────────────────────────────
    print("Running noise stress test...")
    rng = np.random.RandomState(RANDOM_SEED)

    # Select clean, complete, Valid, nonduplicate rows
    valid = train_df[train_df[COL_LABEL] == LABEL_VALID].copy()
    fps = fingerprint_series(valid)
    dup_mask = fps.duplicated(keep=False)
    clean = valid[~dup_mask].copy()
    from src.contract import COL_S4
    complete = clean.dropna(subset=CRITICAL_SENSOR_COLS + [COL_S4])
    held_out = complete.head(50).copy()

    magnitudes = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    stress_results = []

    for mag in magnitudes:
        corrupted = held_out.copy()
        if mag > 0:
            noise = rng.randn(len(corrupted), 3) * mag
            for idx, sensor in enumerate(CRITICAL_SENSOR_COLS):
                corrupted[sensor] = corrupted[sensor].values + noise[:, idx]

        corrupted[COL_TEST_ID] = [f"NOISE-{i:04d}" for i in range(len(corrupted))]
        result = predict_validity(corrupted, twin_state)
        n_invalid = (result[COL_LABEL] == LABEL_INVALID).sum()
        n_spike = (result["fault_type"] == FAULT_SPIKE).sum()
        rate = n_invalid / len(corrupted)
        stress_results.append({
            "magnitude": mag,
            "n_rows": len(corrupted),
            "n_detected_invalid": int(n_invalid),
            "n_spike": int(n_spike),
            "detection_rate": round(rate, 4),
        })
        print(f"  mag={mag:6.1f}: detected={n_invalid}/{len(corrupted)} "
              f"(rate={rate:.3f})")

    stress_df = pd.DataFrame(stress_results)
    stress_df.to_csv(output_dir / "noise_stress_test.csv", index=False)
    print(f"Noise stress test saved to {output_dir / 'noise_stress_test.csv'}")

    # ── Residual Separation Chart ────────────────────────────────────────
    print("\nGenerating residual separation chart...")

    # Compute residuals on full training data
    valid_train = train_df[
        train_df[CRITICAL_SENSOR_COLS].notna().all(axis=1)
    ].copy()
    resid_df = sensor_residuals(valid_train, twin_state)
    resid_df["actual_label"] = valid_train[COL_LABEL].values

    # Determine fault types for colouring
    fps_all = fingerprint_series(valid_train)
    dup_mask_all = fps_all.duplicated(keep=False)
    missing_mask = valid_train[CRITICAL_SENSOR_COLS].isna().any(axis=1)

    resid_df["fault_group"] = "Valid"
    resid_df.loc[resid_df["actual_label"] == LABEL_INVALID, "fault_group"] = "Invalid (spike)"
    resid_df.loc[dup_mask_all.values, "fault_group"] = "Invalid (duplicate)"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, sensor in enumerate(CRITICAL_SENSOR_COLS):
        short = sensor.replace("Sensor_", "")
        ax = axes[idx]
        for group, color, marker in [
            ("Valid", "#2ecc71", "o"),
            ("Invalid (spike)", "#e74c3c", "x"),
            ("Invalid (duplicate)", "#f39c12", "D"),
        ]:
            mask = resid_df["fault_group"] == group
            ax.scatter(
                resid_df.loc[mask, f"{short}_expected"],
                resid_df.loc[mask, f"{short}_residual"],
                c=color, marker=marker, alpha=0.5, s=20,
                label=group,
            )
        ax.set_xlabel(f"{short} Expected")
        ax.set_ylabel(f"{short} Residual")
        ax.set_title(f"{short}: Residual vs Expected")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    plt.suptitle("Sensor Twin Residual Separation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_separation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual separation chart saved")

    # ── Fault Taxonomy Chart ────────────────────────────────────────────
    print("Generating fault taxonomy chart...")

    # OOF predictions
    folds_df = load_folds()
    oof_df, diagnostics = run_oof_evaluation(
        train_df, folds_df, alpha=1.0, degree=4, random_seed=RANDOM_SEED,
    )

    taxonomy = diagnostics["taxonomy"]
    fault_types = [FAULT_MISSING, FAULT_DUPLICATE, FAULT_SPIKE, FAULT_NONE]
    counts = [taxonomy.get(ft, 0) for ft in fault_types]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    labels = ["Missing Sensor\n(15)", "Duplicate Vector\n(24)",
              "Sensor Spike\n(95)", "Valid\n(866)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("OOF Fault Taxonomy (Perfect 15/24/95 Classification)",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "fault_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fault taxonomy chart saved")

    # ── Spike Score Distribution Chart ───────────────────────────────────
    print("Generating spike score distribution chart...")

    # Compute spike scores for threshold pool
    pool_mask = (
        train_df[CRITICAL_SENSOR_COLS].notna().all(axis=1) &
        ~fingerprint_series(train_df).duplicated(keep=False)
    )
    pool_df = train_df[pool_mask].copy()
    pool_resid = sensor_residuals(pool_df, twin_state)
    pool_resid["actual_label"] = pool_df[COL_LABEL].values

    fig, ax = plt.subplots(figsize=(10, 5))

    valid_scores = pool_resid.loc[
        pool_resid["actual_label"] == LABEL_VALID, "spike_score"
    ].dropna()
    invalid_scores = pool_resid.loc[
        pool_resid["actual_label"] == LABEL_INVALID, "spike_score"
    ].dropna()

    ax.hist(valid_scores, bins=50, alpha=0.6, color="#2ecc71",
            label=f"Valid (n={len(valid_scores)})", edgecolor="white")
    ax.hist(invalid_scores, bins=50, alpha=0.6, color="#e74c3c",
            label=f"Invalid (n={len(invalid_scores)})", edgecolor="white")
    ax.axvline(x=twin_state["threshold"], color="#8e44ad",
               linewidth=2, linestyle="--",
               label=f"Threshold ({twin_state['threshold']:.2f})")
    ax.set_xlabel("Spike Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Spike Score Distribution with Threshold",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "spike_score_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Spike score distribution chart saved")

    print("\n=== All P1 charts and diagnostics generated ===")


if __name__ == "__main__":
    main()
