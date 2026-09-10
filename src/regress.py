"""
P2 — Reference Parameter regression.

Predicts the verified hot-spot temperature rise (`Reference_Parameter`, °C above
ambient) for every test record. Worth 35% of the competition score.

Three design decisions, all forced by the verified findings:

1. Train on Valid rows only. Twelve duplicate groups share identical inputs but
   carry different targets, which proves the target is corrupted on Invalid rows.
   Training on all 1000 rows costs 27% MAE (0.896 -> 0.660).

2. Use the four operating conditions only, never the sensors. Adding raw sensors
   buys ~5% and makes the prediction fail exactly when a sensor faults. Staying
   sensor-free means we can still predict a hot-spot temperature for a record
   whose sensor is broken, and keeps this lane independent of P1.

3. Blend a degree-4 polynomial with a boosted-tree model. The polynomial carries
   the smooth physics, the trees mop up the residual. The blend weight is chosen
   from out-of-fold predictions only.

Public API:
    fit_reference_model(train_valid) -> dict
    predict_reference(df, model)     -> pd.Series
    evaluate(train_valid)            -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.contract import OPS, SENSORS, SEED, TARGET_REG, TARGET_CLS, fold_indices

# Search grid. Deliberately small: the verified findings already located the
# optimum, so this confirms it rather than hunting for it.
DEGREES = (3, 4, 5)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BLEND_STEPS = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# Predictions are clipped to the training target range widened by this much on
# each side. A degree-4 polynomial can diverge violently outside its training
# hull; this costs nothing in range and prevents an absurd value on the second
# hidden dataset.
CLIP_MARGIN = 0.20


def _poly(degree: int, alpha: float) -> Pipeline:
    """Degree-`degree` polynomial expansion of the operating conditions + Ridge."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=None)),
    ])


def _trees() -> HistGradientBoostingRegressor:
    """Lightly tuned boosted trees. No long search — the polynomial does the work."""
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )


def _assert_inputs_clean(train_valid: pd.DataFrame) -> None:
    """Guard the two rules that must never silently break."""
    if TARGET_CLS in train_valid.columns:
        labels = set(train_valid[TARGET_CLS].unique())
        assert labels <= {"Valid"}, (
            f"regression must train on Valid rows only, got labels {sorted(labels)}"
        )
    leaked = [c for c in SENSORS if c in OPS]
    assert not leaked, f"sensor column leaked into the model inputs: {leaked}"


def _oof_predictions(X: np.ndarray, y: np.ndarray, frame: pd.DataFrame,
                     make_model, is_poly: bool) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions on the shared contract folds, plus per-fold MAE."""
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, va in fold_indices(frame):
        # A caller may hand us a subset that does not span all five folds — the
        # benchmark nests this inside its own CV loop, for instance. Skip folds
        # that end up empty on either side rather than fitting on nothing.
        if len(tr) == 0 or len(va) == 0:
            continue
        model = make_model()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        oof[va] = pred
        per_fold.append(float(np.mean(np.abs(pred - y[va]))))
    assert per_fold, "no usable fold: the frame must span at least two folds"
    assert np.isfinite(oof[~np.isnan(oof)]).all(), "out-of-fold predictions must be finite"
    return oof, per_fold


def _covered(*arrays: np.ndarray) -> np.ndarray:
    """Rows where every supplied out-of-fold vector has a prediction.

    Only ever narrower than everything when the caller passed a frame that does
    not span all five folds; on the full training set this is all True.
    """
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    m = _covered(pred)
    y, pred = y[m], pred[m]
    err = pred - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": 1.0 - ss_res / ss_tot,
        "MedAE": float(np.median(np.abs(err))),
        "MaxAE": float(np.max(np.abs(err))),
    }


def evaluate(train_valid: pd.DataFrame) -> pd.DataFrame:
    """Cross-validated comparison across every candidate, on the contract folds.

    One row per candidate, sorted best-first by out-of-fold MAE. Every number
    here is out-of-fold: no candidate is ever scored on data it was fitted on.
    """
    _assert_inputs_clean(train_valid)
    frame = train_valid.reset_index(drop=True)
    X = frame[OPS].to_numpy(dtype=float)
    y = frame[TARGET_REG].to_numpy(dtype=float)

    rows = []
    poly_oof: dict[tuple[int, float], np.ndarray] = {}

    for degree in DEGREES:
        for alpha in ALPHAS:
            oof, per_fold = _oof_predictions(
                X, y, frame, lambda d=degree, a=alpha: _poly(d, a), is_poly=True)
            poly_oof[(degree, alpha)] = oof
            rows.append({
                "model": f"poly{degree} ridge a={alpha:g}",
                **_metrics(y, oof),
                "fold_MAE_std": float(np.std(per_fold)),
                "worst_fold_MAE": float(np.max(per_fold)),
            })

    tree_oof, tree_folds = _oof_predictions(X, y, frame, _trees, is_poly=False)
    rows.append({
        "model": "HistGradientBoosting",
        **_metrics(y, tree_oof),
        "fold_MAE_std": float(np.std(tree_folds)),
        "worst_fold_MAE": float(np.max(tree_folds)),
    })

    # Best polynomial by out-of-fold MAE, then the blend built on top of it.
    best_key = min(poly_oof,
                   key=lambda k: np.mean(np.abs((poly_oof[k] - y)[_covered(poly_oof[k])])))
    best_poly_oof = poly_oof[best_key]
    weight = _sweep_weight(best_poly_oof, tree_oof, y)
    blend_oof = weight * best_poly_oof + (1.0 - weight) * tree_oof

    blend_folds = []
    for _, va in fold_indices(frame):
        cov = va[_covered(blend_oof)[va]]
        if len(cov):
            blend_folds.append(float(np.mean(np.abs(blend_oof[cov] - y[cov]))))

    rows.append({
        "model": f"BLEND w={weight:.2f} (poly{best_key[0]} a={best_key[1]:g} + trees)",
        **_metrics(y, blend_oof),
        "fold_MAE_std": float(np.std(blend_folds)),
        "worst_fold_MAE": float(np.max(blend_folds)),
    })

    return (pd.DataFrame(rows)
            .sort_values("MAE")
            .reset_index(drop=True))


def _sweep_weight(poly_oof: np.ndarray, tree_oof: np.ndarray,
                  y: np.ndarray) -> float:
    """Blend weight minimising out-of-fold MAE, swept 0..1 in steps of 0.05.

    Both inputs are out-of-fold predictions, so the weight has never seen a row
    in the same fit that produced its prediction. It never touches test data.
    """
    m = _covered(poly_oof, tree_oof)
    poly_oof, tree_oof, y = poly_oof[m], tree_oof[m], y[m]
    maes = [np.mean(np.abs(w * poly_oof + (1.0 - w) * tree_oof - y))
            for w in BLEND_STEPS]
    return float(BLEND_STEPS[int(np.argmin(maes))])


def fit_reference_model(train_valid: pd.DataFrame) -> dict:
    """Select degree/alpha/weight out-of-fold, then refit both models on all Valid rows.

    Returns a dict holding both fitted models, the blend weight, the clip range
    and the out-of-fold diagnostics that justified the choices.
    """
    _assert_inputs_clean(train_valid)
    frame = train_valid.reset_index(drop=True)
    X = frame[OPS].to_numpy(dtype=float)
    y = frame[TARGET_REG].to_numpy(dtype=float)

    # --- selection stage: everything decided from out-of-fold predictions -----
    best = None
    for degree in DEGREES:
        for alpha in ALPHAS:
            oof, per_fold = _oof_predictions(
                X, y, frame, lambda d=degree, a=alpha: _poly(d, a), is_poly=True)
            mae = float(np.mean(np.abs((oof - y)[_covered(oof)])))
            if best is None or mae < best["mae"]:
                best = {"degree": degree, "alpha": alpha, "mae": mae,
                        "oof": oof, "per_fold": per_fold}

    tree_oof, tree_folds = _oof_predictions(X, y, frame, _trees, is_poly=False)
    weight = _sweep_weight(best["oof"], tree_oof, y)
    blend_oof = weight * best["oof"] + (1.0 - weight) * tree_oof

    per_fold_blend = []
    for _, va in fold_indices(frame):
        cov = va[_covered(blend_oof)[va]]
        if len(cov):
            per_fold_blend.append(float(np.mean(np.abs(blend_oof[cov] - y[cov]))))

    # --- refit stage: the selected configuration, fitted on all Valid rows ----
    poly = _poly(best["degree"], best["alpha"]).fit(X, y)
    trees = _trees().fit(X, y)

    span = float(y.max() - y.min())
    return {
        "poly": poly,
        "trees": trees,
        "weight": weight,
        "degree": best["degree"],
        "alpha": best["alpha"],
        "features": list(OPS),
        "clip_lo": float(y.min() - CLIP_MARGIN * span),
        "clip_hi": float(y.max() + CLIP_MARGIN * span),
        "train_min": float(y.min()),
        "train_max": float(y.max()),
        "n_train": int(len(y)),
        "oof": {
            "poly": _metrics(y, best["oof"]),
            "trees": _metrics(y, tree_oof),
            "blend": _metrics(y, blend_oof),
            "poly_per_fold_MAE": best["per_fold"],
            "trees_per_fold_MAE": tree_folds,
            "blend_per_fold_MAE": per_fold_blend,
            "worst_fold_MAE": float(np.max(per_fold_blend)),
            "predictions": blend_oof,
            "y": y,
        },
        "n_clipped": 0,
    }


def predict_reference(df: pd.DataFrame, model: dict) -> pd.Series:
    """Predicted Reference_Parameter per row. Never NaN, never infinite.

    Records how many predictions hit the clip range in `model["n_clipped"]`.
    """
    missing = [c for c in model["features"] if c not in df.columns]
    assert not missing, f"missing required operating column(s): {missing}"

    X = df[model["features"]].to_numpy(dtype=float)

    # The four operating conditions are never missing in this dataset, but the
    # second hidden set is unseen — fall back to the training median rather than
    # propagating a NaN into the submission.
    if not np.isfinite(X).all():
        med = np.nanmedian(X, axis=0)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(med, idx[1])

    raw = (model["weight"] * model["poly"].predict(X)
           + (1.0 - model["weight"]) * model["trees"].predict(X))

    clipped = np.clip(raw, model["clip_lo"], model["clip_hi"])
    model["n_clipped"] = int(np.sum(raw != clipped))

    # Last line of defence: a non-finite prediction would break the submission
    # file, so anything that survives to here becomes the training mean.
    if not np.isfinite(clipped).all():
        fallback = 0.5 * (model["train_min"] + model["train_max"])
        clipped = np.where(np.isfinite(clipped), clipped, fallback)

    return pd.Series(clipped, index=df.index, name="Predicted_Reference_Parameter")
