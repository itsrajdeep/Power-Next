"""
Main Execution Pipeline for CPRI PowerNext-AI Hackathon.
Assembles sensor twin, regression models, deliverables generation, and QA validation.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# The team name contains a non-ASCII character and the default Windows console
# codepage (cp1252) cannot encode it, so an unconfigured console would kill the
# run on its first print. Force UTF-8 on the streams here rather than asking
# whoever runs this to set PYTHONIOENCODING first.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ----------------------------------------------------------------------------
# Top-level Configuration
# ----------------------------------------------------------------------------
DEFAULT_TEAM_NAME = "pi-thon"
DEFAULT_OUTPUT_DIR = "outputs/final"

# ----------------------------------------------------------------------------
# Import Lane Modules or Stubs
# ----------------------------------------------------------------------------
try:
    from src.twin import fit_twin, derive_threshold, predict_validity
    TWIN_STUB = False
except ImportError:
    from src.twin_stub import fit_twin, derive_threshold, predict_validity
    TWIN_STUB = True

try:
    from src.regress import fit_reference_model, predict_reference
    REGRESS_STUB = False
except ImportError:
    from src.regress_stub import fit_reference_model, predict_reference
    REGRESS_STUB = True

USING_STUBS = TWIN_STUB or REGRESS_STUB

from src.contract import load_data, valid_rows
from src.finalize import (
    validate_submission,
    compute_attention_scores,
    build_summary,
    write_submission,
    write_summary,
    write_diagnostics,
)


def generate_charts(
    submission_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    chart_dir: Path
) -> None:
    """Generates visual analytics charts for predictions and diagnostics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_dir.mkdir(parents=True, exist_ok=True)

        # Chart 1: Predicted Temperature Distribution by Validity Label
        plt.figure(figsize=(8, 5))
        valid_preds = submission_df.loc[
            submission_df["Validity_Label"] == "Valid", "Predicted_Reference_Parameter"
        ]
        invalid_preds = submission_df.loc[
            submission_df["Validity_Label"] == "Invalid", "Predicted_Reference_Parameter"
        ]

        plt.hist(
            valid_preds,
            bins=25,
            alpha=0.7,
            color="#2ecc71",
            edgecolor="black",
            label=f"Valid (n={len(valid_preds)})"
        )
        if len(invalid_preds) > 0:
            plt.hist(
                invalid_preds,
                bins=25,
                alpha=0.7,
                color="#e74c3c",
                edgecolor="black",
                label=f"Invalid (n={len(invalid_preds)})"
            )

        plt.title("Predicted Reference Parameter Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Reference Parameter (°C)", fontsize=11)
        plt.ylabel("Test Count", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(chart_dir / "final_predictions_distribution.png", dpi=150)
        plt.close()

        # Chart 2: Top 10 Attention Scores
        plt.figure(figsize=(9, 5))
        top_10 = diagnostics_df.head(10).iloc[::-1]  # reversed so rank 1 is at top
        bars = plt.barh(
            top_10["Test_ID"],
            top_10["attention_score"],
            color="#3498db",
            edgecolor="black"
        )
        plt.title("Top 10 Attention Score Tests (Human Review Priority)", fontsize=13, fontweight="bold")
        plt.xlabel("Normalized Attention Score [0..1]", fontsize=11)
        plt.ylabel("Test ID", fontsize=11)
        plt.xlim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5, axis="x")

        # Label bars
        for bar, val in zip(bars, top_10["attention_score"]):
            plt.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.3f}",
                va="center",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(chart_dir / "top_attention_scores.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"  [Warning] Chart generation failed: {e}")


def run_pipeline(
    team_name: str = DEFAULT_TEAM_NAME,
    output_dir_str: str = DEFAULT_OUTPUT_DIR,
    skip_charts: bool = False,
) -> int:
    """Executes the complete pipeline end to end."""
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"CPRI PowerNext-AI Pipeline — Team: {team_name}")
    print("=" * 76)

    if USING_STUBS:
        print("\n" + "!" * 76)
        print("  WARNING: PIPELINE RUNNING WITH PROVISIONAL STUBS!")
        if TWIN_STUB:
            print("    * src/twin.py not found -> using src/twin_stub.py")
        if REGRESS_STUB:
            print("    * src/regress.py not found -> using src/regress_stub.py")
        print("  Outputs written to disk are PROVISIONAL.")
        print("!" * 76 + "\n")

    # 1. Load Data
    print("[1/6] Loading datasets...")
    train, test, sample = load_data()
    print(f"      Train: {train.shape[0]} rows | Test: {test.shape[0]} rows | Sample: {sample.shape[0]} rows")

    # 2. Fit Models on Valid Training Records
    print("[2/6] Training sensor twin and regression models on Valid records...")
    train_valid = valid_rows(train)
    print(f"      Valid training records: {len(train_valid)} / {len(train)}")

    twin = fit_twin(train_valid)
    threshold = derive_threshold(train_valid, twin)
    print(f"      Sensor twin fitted. Empirical threshold: {threshold:.4f}")

    reg_model = fit_reference_model(train_valid)
    print("      Reference Parameter regression model fitted.")

    # 3. Predict on Test Data
    print("[3/6] Generating predictions on test dataset...")
    labels = predict_validity(test, twin, threshold)
    preds = predict_reference(test, reg_model)

    submission = pd.DataFrame({
        "Test_ID": test["Test_ID"],
        "Predicted_Reference_Parameter": preds,
        "Validity_Label": labels,
    })

    # 4. Strict Validation
    print("[4/6] Performing strict submission verification...")
    validate_submission(submission, sample)
    print("      Validation PASSED: exactly 350 rows, aligned Test_IDs, no NaN/inf, valid labels.")

    # 5. Attention Scoring and Summary Generation
    print("[5/6] Computing diagnostics, attention rankings, and summary metrics...")
    attention_ids, diagnostics_df = compute_attention_scores(test, submission, twin, threshold)
    summary_dict = build_summary(submission, attention_ids)

    # 6. Writing Deliverables
    print("[6/6] Writing final deliverables...")
    csv_file = output_dir / f"{team_name}.csv"
    summary_file = output_dir / "summary.json"
    diag_file = output_dir / "test_diagnostics.csv"

    write_submission(submission, csv_file)
    write_summary(summary_dict, summary_file)
    write_diagnostics(diagnostics_df, diag_file)

    if not skip_charts:
        chart_dir = Path("outputs/charts")
        print("      Rendering visualization charts...")
        generate_charts(submission, diagnostics_df, chart_dir)

    print("\n" + "=" * 76)
    print("PIPELINE EXECUTION COMPLETE — SUMMARY OF DELIVERABLES")
    print("=" * 76)
    print(f"  Submission CSV   : {csv_file} ({len(submission)} rows)")
    print(f"  Summary JSON     : {summary_file}")
    print(f"  Diagnostics CSV  : {diag_file} ({len(diagnostics_df)} rows)")
    print("-" * 76)
    print(f"  Analysed Records : {summary_dict['records_analysed']}")
    print(f"  Invalid Records  : {summary_dict['abnormal_invalid_records']} ({summary_dict['abnormal_invalid_records'] / len(submission) * 100:.1f}%)")
    print(f"  Min Temperature  : {summary_dict['minimum_predicted_reference_parameter']:.2f} °C")
    print(f"  Max Temperature  : {summary_dict['maximum_predicted_reference_parameter']:.2f} °C")
    print(f"  Mean Temperature : {summary_dict['average_predicted_reference_parameter']:.2f} °C")
    print(f"  Attention IDs    : {', '.join(summary_dict['highest_attention_test_ids'])}")
    print("=" * 76 + "\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CPRI PowerNext-AI Automated Hackathon Pipeline."
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default=DEFAULT_TEAM_NAME,
        help=f"Team name for output CSV filename (default: {DEFAULT_TEAM_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where deliverables will be written (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Skip generating chart images in outputs/charts/",
    )

    args = parser.parse_args()
    sys.exit(run_pipeline(args.team_name, args.output_dir, args.skip_charts))


if __name__ == "__main__":
    main()
