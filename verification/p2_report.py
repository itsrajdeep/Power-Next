"""
P2 verification report — Reference Parameter regression.

Prints the full candidate comparison, the chosen configuration, out-of-fold
error broken down by operating regime, and an extrapolation stress test.

    python verification/p2_report.py > verification/p2_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract import OPS, load_data, valid_rows          # noqa: E402
from src.regress import (                                     # noqa: E402
    fit_reference_model, predict_reference, evaluate,
)

RULE = "=" * 74


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def regime_breakdown(y: np.ndarray, pred: np.ndarray,
                     frame: pd.DataFrame) -> pd.DataFrame:
    """Absolute error by operating regime — where does the model struggle?"""
    err = np.abs(pred - y)
    rows = []

    for col, label in [("Load_Current_A", "current"),
                       ("Applied_Voltage_kV", "voltage")]:
        terciles = pd.qcut(frame[col], 3, labels=["low", "mid", "high"])
        for level in ["low", "mid", "high"]:
            mask = (terciles == level).to_numpy()
            rows.append({"regime": f"{label} {level}", "n": int(mask.sum()),
                         "MAE": float(err[mask].mean()),
                         "MaxAE": float(err[mask].max())})

    hot = (frame["Ambient_Temperature_C"] > 40).to_numpy()
    for mask, label in [(~hot, "ambient <= 40C"), (hot, "ambient > 40C")]:
        if mask.sum():
            rows.append({"regime": label, "n": int(mask.sum()),
                         "MAE": float(err[mask].mean()),
                         "MaxAE": float(err[mask].max())})

    return pd.DataFrame(rows)


def extrapolation_test(model: dict, train_valid: pd.DataFrame) -> pd.DataFrame:
    """Push current and voltage past anything seen in training.

    The degree-4 polynomial is the risk here. We check the predictions stay
    positive, stay finite, stay physically sensible in magnitude, and rise
    monotonically with current — more current must never mean a cooler hot spot.
    """
    base = {c: float(train_valid[c].median()) for c in OPS}
    rows = []
    for factor in [1.0, 1.2, 1.5]:
        probe = dict(base)
        probe["Load_Current_A"] = train_valid["Load_Current_A"].max() * factor
        probe["Applied_Voltage_kV"] = train_valid["Applied_Voltage_kV"].max() * factor
        frame = pd.DataFrame([probe])
        pred = float(predict_reference(frame, model).iloc[0])
        # Unclipped value too: the clip is what keeps the submission sane, but we
        # need to see what it is protecting us from.
        X = frame[model["features"]].to_numpy(dtype=float)
        raw = float(model["weight"] * model["poly"].predict(X)[0]
                    + (1.0 - model["weight"]) * model["trees"].predict(X)[0])
        rows.append({"factor": f"{factor:.1f}x max", **probe,
                     "raw": raw, "prediction": pred,
                     "clipped": raw != pred})
    return pd.DataFrame(rows)


def main() -> None:
    train, test, _ = load_data()
    train_valid = valid_rows(train)

    head("1. CANDIDATE COMPARISON (out-of-fold, contract folds)")
    table = evaluate(train_valid)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    model = fit_reference_model(train_valid)
    oof = model["oof"]
    blend = oof["blend"]

    head("2. CHOSEN CONFIGURATION")
    print(f"  polynomial degree : {model['degree']}")
    print(f"  ridge alpha       : {model['alpha']:g}")
    print(f"  blend weight      : {model['weight']:.2f}"
          f"  ({model['weight']:.2f} poly + {1 - model['weight']:.2f} trees)")
    print(f"  training rows     : {model['n_train']} (Valid only)")
    print(f"  model inputs      : {model['features']}")
    print(f"  sensors used      : none - by design")

    head("3. FINAL OUT-OF-FOLD PERFORMANCE")
    print(f"  MAE               : {blend['MAE']:.4f}")
    print(f"  RMSE              : {blend['RMSE']:.4f}")
    print(f"  R2                : {blend['R2']:.5f}")
    print(f"  median abs error  : {blend['MedAE']:.4f}")
    print(f"  max abs error     : {blend['MaxAE']:.4f}")
    print()
    for k, folds in [("poly ", oof["poly_per_fold_MAE"]),
                     ("trees", oof["trees_per_fold_MAE"]),
                     ("blend", oof["blend_per_fold_MAE"])]:
        print(f"  {k} per-fold MAE : " + "  ".join(f"{v:.4f}" for v in folds)
              + f"   (worst {max(folds):.4f})")
    print(f"\n  worst-fold MAE    : {oof['worst_fold_MAE']:.4f}   "
          f"spread across folds = {np.std(oof['blend_per_fold_MAE']):.4f}")

    head("4. ERROR BY OPERATING REGIME (out-of-fold)")
    print(regime_breakdown(oof["y"], oof["predictions"],
                           train_valid.reset_index(drop=True))
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    head("5. PREDICTIONS ON THE 350 TEST ROWS")
    pred = predict_reference(test, model)
    print(f"  n                 : {len(pred)}")
    print(f"  min               : {pred.min():.4f}")
    print(f"  max               : {pred.max():.4f}")
    print(f"  mean              : {pred.mean():.4f}")
    print(f"  clip range        : [{model['clip_lo']:.4f}, {model['clip_hi']:.4f}]")
    print(f"  clipped           : {model['n_clipped']} of {len(pred)}")
    print(f"  all finite        : {bool(np.isfinite(pred).all())}")

    head("6. EXTRAPOLATION STRESS TEST")
    ex = extrapolation_test(model, train_valid)
    print(ex.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    preds = ex["prediction"].to_numpy()
    raws = ex["raw"].to_numpy()
    # The clip deliberately saturates, so the shipped predictions can only be
    # required to be non-decreasing. The raw values must be strictly increasing:
    # that is what proves the polynomial itself has not turned over.
    monotone = bool(np.all(np.diff(preds) >= 0))
    raw_monotone = bool(np.all(np.diff(raws) > 0))
    print(f"\n  raw strictly increasing with current   : {raw_monotone}")
    print(f"  shipped non-decreasing (clip saturates): {monotone}")
    print(f"  all positive                          : {bool((preds > 0).all())}")
    print(f"  all finite                            : {bool(np.isfinite(preds).all())}")
    print(f"  within clip range                     : "
          f"{bool((preds >= model['clip_lo']).all() and (preds <= model['clip_hi']).all())}")

    head("7. ACCEPTANCE TESTS")
    checks = [
        ("out-of-fold MAE <= 0.55", blend["MAE"] <= 0.55, f"{blend['MAE']:.4f}"),
        ("out-of-fold R2 >= 0.994", blend["R2"] >= 0.994, f"{blend['R2']:.5f}"),
        ("worst fold MAE <= 0.70", oof["worst_fold_MAE"] <= 0.70,
         f"{oof['worst_fold_MAE']:.4f}"),
        ("trained on Valid rows only", model["n_train"] == 866, f"{model['n_train']} rows"),
        ("no sensor among inputs",
         not any(c.startswith("Sensor_") for c in model["features"]),
         str(model["features"])),
        ("all test predictions finite", bool(np.isfinite(pred).all()), "no NaN/inf"),
        ("test predictions within 10-70 C",
         bool(pred.min() >= 10 and pred.max() <= 70),
         f"{pred.min():.2f}..{pred.max():.2f}"),
        ("extrapolation stays positive", bool((preds > 0).all()),
         f"min {preds.min():.2f}"),
        ("extrapolation monotone in current", monotone and raw_monotone,
         f"raw {list(np.round(raws, 1))} -> shipped {list(np.round(preds, 1))}"),
        ("extrapolation bounded by clip",
         bool((preds <= model["clip_hi"]).all()),
         f"ceiling {model['clip_hi']:.2f}"),
    ]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:36s} {detail}")

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\n{RULE}")
    print("ALL ACCEPTANCE TESTS PASS" if not failed
          else f"FAILED: {', '.join(failed)}")
    print(RULE)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
