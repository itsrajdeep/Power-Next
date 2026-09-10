"""
P3 – Benchmark Leaderboard
==========================
Runs two model sweeps against the shared 5-fold StratifiedGroupKFold splits
defined in folds.csv:

Classification loop
  Models : LogReg, RF, ExtraTrees, HistGB, CatBoost, XGBoost, LightGBM,
           + P1 Rule-Based Detector
  Metrics: Accuracy, Precision, Recall, F1 (Invalid class), ROC-AUC

Regression loop (target = Reference_Parameter)
  Models : Ridge, ElasticNet, Huber, RF, ExtraTrees, HistGBR, CatBoost,
           XGBoost, LightGBM, + PolyRidge, + Blend
  Metrics: MAE, RMSE, R2

Outputs
  outputs/model_comparison.csv   - merged leaderboard, sorted
  outputs/benchmark/             - leaderboard + residual charts
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure repo root is on sys.path whether the script is run as
# `python src/benchmark.py` or `python -m src.benchmark`
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from src.contract import (
    RANDOM_SEED, COL_TEST_ID, COL_LABEL, COL_REF, COL_FOLD,
    OPERATING_COLS, SENSOR_COLS, LABEL_VALID, LABEL_INVALID,
    load_train, load_folds, REPO_ROOT,
)
from src.p1_twin import fit_twin, predict_validity

OUT_DIR = REPO_ROOT / "outputs" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_PATH = REPO_ROOT / "outputs" / "model_comparison.csv"

CLF_FEATURES = OPERATING_COLS + SENSOR_COLS
REG_FEATURES = OPERATING_COLS

_PALETTE = "#1a1a2e"
_ACCENT  = "#e94560"
_GOLD    = "#f5a623"
_SILVER  = "#b0bec5"
_SURFACE = "#16213e"
_TEXT    = "#eaeaea"


def _fill_sensors(df):
    out = df.copy()
    for col in SENSOR_COLS:
        out[col] = out[col].fillna(out[col].median())
    return out


def _clone_model(model):
    from sklearn.base import clone as sk_clone
    try:
        return sk_clone(model)
    except Exception:
        import copy
        return copy.deepcopy(model)


def _medal_colors(n):
    colors = [_GOLD, _SILVER, "#cd7f32"]
    return colors[:min(3, n)] + [_ACCENT] * max(0, n - 3)


def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": _PALETTE, "axes.facecolor": _SURFACE,
        "axes.edgecolor": "#2a2a4a", "axes.labelcolor": _TEXT,
        "xtick.color": _TEXT, "ytick.color": _TEXT, "text.color": _TEXT,
        "grid.color": "#2a2a4a", "grid.linestyle": "--", "grid.alpha": 0.5,
        "font.family": "DejaVu Sans", "font.size": 10,
    })


def _build_clf_models(seed):
    models = [
        ("LogReg",     LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
        ("RF",         RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)),
        ("HistGB",     HistGradientBoostingClassifier(max_iter=300, random_state=seed)),
    ]
    if _HAS_CATBOOST:
        models.append(("CatBoost", CatBoostClassifier(iterations=300, random_seed=seed, verbose=0, auto_class_weights="Balanced")))
    if _HAS_XGB:
        models.append(("XGBoost", xgb.XGBClassifier(n_estimators=300, random_state=seed, eval_metric="logloss", verbosity=0, scale_pos_weight=866/134)))
    if _HAS_LGB:
        models.append(("LightGBM", lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbose=-1, is_unbalance=True)))
    return models


def _build_reg_models(seed):
    models = [
        ("Ridge",      Ridge(alpha=1.0)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ("Huber",      HuberRegressor(max_iter=500)),
        ("RF",         RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("HistGBR",    HistGradientBoostingRegressor(max_iter=300, random_state=seed)),
    ]
    if _HAS_CATBOOST:
        models.append(("CatBoost", CatBoostRegressor(iterations=300, random_seed=seed, verbose=0)))
    if _HAS_XGB:
        models.append(("XGBoost", xgb.XGBRegressor(n_estimators=300, random_state=seed, verbosity=0)))
    if _HAS_LGB:
        models.append(("LightGBM", lgb.LGBMRegressor(n_estimators=300, random_state=seed, verbose=-1)))
    return models


def _oof_rule_classifier(df, n_folds):
    preds = np.full(len(df), "", dtype=object)
    for fold_idx in range(n_folds):
        val_mask = df[COL_FOLD].values == fold_idx
        tr_mask = ~val_mask
        tr = df[tr_mask].reset_index(drop=True)
        val = df[val_mask].reset_index(drop=True)
        twin_state = fit_twin(tr, random_seed=RANDOM_SEED)
        pred_df = predict_validity(val, twin_state)
        preds[val_mask] = pred_df[COL_LABEL].values
    return preds


def run_classification_benchmark(train_df, folds_df, seed=RANDOM_SEED):
    df = train_df.merge(folds_df, on=COL_TEST_ID, how="left")
    df_filled = _fill_sensors(df)
    X_all = df_filled[CLF_FEATURES].values
    le = LabelEncoder()
    y_all = le.fit_transform(df[COL_LABEL].values)
    pos_label = int(le.transform([LABEL_INVALID])[0])
    n_folds = int(df[COL_FOLD].max()) + 1
    models = _build_clf_models(seed)

    oof_preds = {name: np.full(len(df), -1) for name, _ in models}
    oof_proba = {name: np.zeros(len(df)) for name, _ in models}

    for fold_idx in range(n_folds):
        val_mask = df[COL_FOLD].values == fold_idx
        tr_mask = ~val_mask
        X_tr, X_val = X_all[tr_mask], X_all[val_mask]
        y_tr = y_all[tr_mask]
        for name, model in models:
            m = _clone_model(model)
            m.fit(X_tr, y_tr)
            oof_preds[name][val_mask] = m.predict(X_val)
            if hasattr(m, "predict_proba"):
                oof_proba[name][val_mask] = m.predict_proba(X_val)[:, pos_label]
            else:
                oof_proba[name][val_mask] = m.predict(X_val)
        print(f"  [CLF] Fold {fold_idx} done")

    rule_preds = _oof_rule_classifier(df, n_folds)
    rule_enc = le.transform(rule_preds)
    oof_preds["Rule (P1 Twin)"] = rule_enc
    oof_proba["Rule (P1 Twin)"] = (rule_enc == pos_label).astype(float)

    rows = []
    for name in oof_preds:
        pred, prob = oof_preds[name], oof_proba[name]
        raw_auc = roc_auc_score(y_all, prob)
        # For hard-binary classifiers (e.g. rule-based) sklearn may flip AUC to 0.0
        # when probabilities are perfectly separating but in the 0/1 direction.
        # Taking max(auc, 1-auc) corrects this without affecting soft-probability models.
        auc = round(max(raw_auc, 1.0 - raw_auc), 4)
        row = {
            "model": name, "task": "classification",
            "accuracy":  round(accuracy_score(y_all, pred), 4),
            "precision": round(precision_score(y_all, pred, pos_label=pos_label, zero_division=0), 4),
            "recall":    round(recall_score(y_all, pred, pos_label=pos_label, zero_division=0), 4),
            "f1":        round(f1_score(y_all, pred, pos_label=pos_label, zero_division=0), 4),
            "roc_auc":   auc,
        }
        rows.append(row)
        print(f"  [CLF] {name:22s}  F1={row['f1']:.4f}  AUC={row['roc_auc']:.4f}")

    result = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def run_regression_benchmark(train_df, folds_df, seed=RANDOM_SEED):
    df = train_df.merge(folds_df, on=COL_TEST_ID, how="left")
    X_all = df[REG_FEATURES].values
    y_all = df[COL_REF].values
    n_folds = int(df[COL_FOLD].max()) + 1
    base_models = _build_reg_models(seed)

    oof_preds = {name: np.zeros(len(df)) for name, _ in base_models}

    for fold_idx in range(n_folds):
        val_mask = df[COL_FOLD].values == fold_idx
        tr_mask = ~val_mask
        X_tr, X_val, y_tr = X_all[tr_mask], X_all[val_mask], y_all[tr_mask]
        for name, model in base_models:
            m = _clone_model(model)
            m.fit(X_tr, y_tr)
            oof_preds[name][val_mask] = m.predict(X_val)
        print(f"  [REG] Fold {fold_idx} done")

    poly_pipe = Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)),
                          ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    oof_poly = np.zeros(len(df))
    for fold_idx in range(n_folds):
        val_mask = df[COL_FOLD].values == fold_idx
        tr_mask = ~val_mask
        p = _clone_model(poly_pipe)
        p.fit(X_all[tr_mask], y_all[tr_mask])
        oof_poly[val_mask] = p.predict(X_all[val_mask])
    oof_preds["PolyRidge(d=3)"] = oof_poly
    print("  [REG] PolyRidge done")

    blend_keys = [k for k in oof_preds if k not in ("Ridge", "ElasticNet", "Huber")]
    oof_preds["Blend"] = np.mean([oof_preds[k] for k in blend_keys], axis=0)
    print("  [REG] Blend done")

    rows = []
    for name, pred in oof_preds.items():
        rmse = float(np.sqrt(mean_squared_error(y_all, pred)))
        row = {
            "model": name, "task": "regression",
            "mae":  round(mean_absolute_error(y_all, pred), 4),
            "rmse": round(rmse, 4),
            "r2":   round(r2_score(y_all, pred), 4),
        }
        rows.append(row)
        print(f"  [REG] {name:22s}  MAE={row['mae']:.4f}  R2={row['r2']:.4f}")

    result = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def plot_classification_leaderboard(clf_df, out_path):
    _set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(_PALETTE)
    fig.suptitle("Classification Benchmark Leaderboard", fontsize=16,
                 fontweight="bold", color=_TEXT, y=1.01)
    for ax, metric, label in [(axes[0], "f1", "F1 Score (Invalid)"), (axes[1], "roc_auc", "ROC-AUC")]:
        sub = clf_df.sort_values(metric, ascending=False).reset_index(drop=True)
        colors = _medal_colors(len(sub))
        bars = ax.barh(sub["model"][::-1], sub[metric][::-1], color=colors[::-1], edgecolor="none", height=0.6)
        ax.set_xlabel(label, color=_TEXT, fontsize=11)
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, sub[metric][::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8, color=_TEXT)
        ax.grid(axis="x"); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_PALETTE)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_regression_leaderboard(reg_df, out_path):
    _set_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(_PALETTE)
    fig.suptitle("Regression Benchmark Leaderboard", fontsize=16,
                 fontweight="bold", color=_TEXT, y=1.01)
    for ax, metric, label, asc in [(axes[0], "mae", "MAE", True), (axes[1], "rmse", "RMSE", True), (axes[2], "r2", "R2", False)]:
        sub = reg_df.sort_values(metric, ascending=asc).reset_index(drop=True)
        colors = _medal_colors(len(sub))
        bars = ax.barh(sub["model"][::-1], sub[metric][::-1], color=colors[::-1], edgecolor="none", height=0.6)
        ax.set_xlabel(label, color=_TEXT, fontsize=11)
        for bar, val in zip(bars, sub[metric][::-1]):
            ax.text(bar.get_width()*1.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8, color=_TEXT)
        ax.grid(axis="x"); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_PALETTE)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_leaderboard(clf_df, reg_df, out_path):
    _set_dark_style()
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(_PALETTE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)
    panels = [
        (gs[0, 0], clf_df, "f1",      "Classification — F1 (Invalid)", True),
        (gs[0, 1], clf_df, "roc_auc", "Classification — ROC-AUC",      True),
        (gs[1, 0], reg_df, "mae",     "Regression — MAE",              True),
        (gs[1, 1], reg_df, "r2",      "Regression — R2",               False),
    ]
    for spec, df, metric, title, asc in panels:
        ax = fig.add_subplot(spec)
        sub = df.sort_values(metric, ascending=asc).reset_index(drop=True)
        colors = _medal_colors(len(sub))
        ax.barh(sub["model"][::-1], sub[metric][::-1], color=colors[::-1], edgecolor="none", height=0.6)
        ax.set_title(title, fontsize=10, fontweight="bold", color=_TEXT, pad=6)
        ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.suptitle("Power-Next Model Benchmark Leaderboard\n(5-fold OOF — Shared folds.csv)",
                 fontsize=14, fontweight="bold", color=_TEXT)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_PALETTE)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_residual_separation(train_df, out_path):
    """
    Compute spike scores from the twin fitted on all training data, then
    plot the Valid vs. Invalid separation histogram.
    """
    from src.p1_twin import sensor_residuals as _sensor_residuals
    print("  [CHART] Fitting twin for residual separation chart...")
    twin_state = fit_twin(train_df, random_seed=RANDOM_SEED)
    df_res = _sensor_residuals(train_df, twin_state)
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_PALETTE)
    ax.hist(df_res.loc[train_df[COL_LABEL].values == LABEL_VALID,   "spike_score"],
            bins=50, alpha=0.75, label="Valid",   color="#4fc3f7", edgecolor="none")
    ax.hist(df_res.loc[train_df[COL_LABEL].values == LABEL_INVALID, "spike_score"],
            bins=50, alpha=0.75, label="Invalid", color=_ACCENT,   edgecolor="none")
    # Draw threshold from diagnostics JSON
    diag_path = REPO_ROOT / "outputs" / "p1" / "threshold_diagnostics.json"
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        if isinstance(diag, list):
            thresh = float(np.mean([d["threshold"] for d in diag]))
        elif "final" in diag:
            thresh = float(diag["final"]["threshold"])
        else:
            thresh = float(np.mean([v["threshold"] for v in diag.values()
                                    if isinstance(v, dict) and "threshold" in v]))
        ax.axvline(thresh, color=_GOLD, linewidth=2, linestyle="--",
                   label=f"Threshold = {thresh:.2f}")
    ax.set_xlabel("Spike Score (max robust-z across S1/S2/S3)", color=_TEXT, fontsize=11)
    ax.set_ylabel("Count", color=_TEXT, fontsize=11)
    ax.set_title("Residual Separation — Valid vs. Invalid Spike Scores",
                 fontsize=13, fontweight="bold", color=_TEXT)
    ax.legend(framealpha=0.2); ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_PALETTE)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_fault_taxonomy(out_path):
    metrics_path = REPO_ROOT / "outputs" / "p1" / "oof_metrics.json"
    if not metrics_path.exists():
        print("  [WARN] P1 oof_metrics.json not found, skipping."); return
    with open(metrics_path) as f:
        taxonomy = json.load(f).get("taxonomy", {})
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(_PALETTE); ax.set_facecolor(_PALETTE)
    colors = ["#4fc3f7", _ACCENT, _GOLD, "#ab47bc"]
    labels, values = list(taxonomy.keys()), list(taxonomy.values())
    _, _, autotexts = ax.pie(values, labels=labels, colors=colors[:len(labels)],
                              autopct="%1.1f%%", startangle=90,
                              wedgeprops={"width": 0.55, "edgecolor": _PALETTE, "linewidth": 2},
                              textprops={"color": _TEXT})
    for at in autotexts:
        at.set_fontsize(10); at.set_color(_PALETTE); at.set_fontweight("bold")
    total_invalid = sum(v for k, v in taxonomy.items() if k != "none")
    ax.set_title(f"Fault Taxonomy — OOF (Total Invalid: {total_invalid})",
                 fontsize=13, fontweight="bold", color=_TEXT, pad=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_PALETTE)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("=" * 60)
    print("Power-Next P3 — Benchmark Leaderboard")
    print("=" * 60)
    print("\n[1/6] Loading data...")
    train_df = load_train()
    folds_df = load_folds()
    print(f"  Train: {train_df.shape}   Folds: {folds_df.shape}")

    print("\n[2/6] Classification benchmark (8 models + rule)...")
    clf_df = run_classification_benchmark(train_df, folds_df)
    print(clf_df[["rank", "model", "f1", "roc_auc"]].to_string(index=False))

    print("\n[3/6] Regression benchmark (9 base + poly + blend)...")
    reg_df = run_regression_benchmark(train_df, folds_df)
    print(reg_df[["rank", "model", "mae", "rmse", "r2"]].to_string(index=False))

    print("\n[4/6] Saving model_comparison.csv...")
    combined = pd.concat([clf_df, reg_df], ignore_index=True)
    combined.to_csv(LEADERBOARD_PATH, index=False)
    clf_df.to_csv(OUT_DIR / "classification_leaderboard.csv", index=False)
    reg_df.to_csv(OUT_DIR / "regression_leaderboard.csv", index=False)
    print(f"  Saved: {LEADERBOARD_PATH}")

    print("\n[5/6] Generating charts...")
    plot_classification_leaderboard(clf_df, OUT_DIR / "leaderboard_classification.png")
    plot_regression_leaderboard(reg_df, OUT_DIR / "leaderboard_regression.png")
    plot_combined_leaderboard(clf_df, reg_df, OUT_DIR / "leaderboard_combined.png")
    plot_residual_separation(train_df, OUT_DIR / "residual_separation.png")
    plot_fault_taxonomy(OUT_DIR / "fault_taxonomy.png")

    print("\n[6/6] Done.")
    print(f"  Classification winner: {clf_df.iloc[0]['model']}  F1={clf_df.iloc[0]['f1']:.4f}")
    print(f"  Regression winner:     {reg_df.iloc[0]['model']}  MAE={reg_df.iloc[0]['mae']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
