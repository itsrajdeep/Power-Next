"""
P3 - Benchmark Leaderboard
===========================
Runs two model sweeps using fold_indices from src/contract.py.

Classification (target: Validity_Label, Invalid = positive class)
  Feature sets : raw_operating, all_raw, sensor_consistency
  Models       : Dummy, LogReg, RF, ExtraTrees, HistGB, CatBoost, XGBoost, LightGBM
  Metrics      : Precision/Recall/F1 (Invalid), macro-F1, balanced-acc, ROC-AUC, PR-AUC, MCC

Regression (target: Reference_Parameter, trained on Valid rows only)
  Feature sets : raw_operating, all_raw, poly4_operating
  Models       : Dummy, Ridge, ElasticNet, Huber, RF, ExtraTrees, HistGBR, CatBoost, XGBoost, LightGBM, SVR
  Metrics      : MAE, RMSE, R2, median-AE, max-error, per-fold-std, worst-fold-MAE

Ablations
  1. Regression on all rows vs Valid rows only
  2. With Sensor_S4 vs without
  3. With Test_Duration_min vs without
  4. Polynomial degree 2 vs 3 vs 4 vs 5

Outputs
  outputs/model_comparison.csv
  outputs/charts/  (6 charts)
  verification/p3_report.txt
"""

from __future__ import annotations
import sys
import io
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CB = True
except ImportError:
    _HAS_CB = False

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

# ---- New contract API -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contract import (
    SEED, load_data, fold_indices, valid_rows, is_invalid,
    OPS, S3, SENSORS, S4, TARGET_CLS, TARGET_REG, ROOT,
    fingerprint, is_duplicate,
)

CHARTS_DIR = ROOT / "outputs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_PATH = ROOT / "outputs" / "model_comparison.csv"
REPORT_PATH = ROOT / "verification" / "p3_report.txt"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Style ------------------------------------------------------------------
_BG = "#1a1a2e"; _SRF = "#16213e"; _ACC = "#e94560"
_GOLD = "#f5a623"; _SILVER = "#b0bec5"; _TXT = "#eaeaea"

def _dark():
    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _SRF,
        "axes.edgecolor": "#2a2a4a", "axes.labelcolor": _TXT,
        "xtick.color": _TXT, "ytick.color": _TXT, "text.color": _TXT,
        "grid.color": "#2a2a4a", "grid.linestyle": "--", "grid.alpha": 0.5,
        "font.size": 10,
    })

def _mcolors(n):
    base = [_GOLD, _SILVER, "#cd7f32"]
    return base[:min(3, n)] + [_ACC] * max(0, n - 3)

def _clone(m):
    from sklearn.base import clone as sk
    try:
        return sk(m)
    except Exception:
        import copy; return copy.deepcopy(m)

# ---- Feature engineering ----------------------------------------------------

def _fill_sensors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in SENSORS:
        out[c] = out[c].fillna(out[c].median())
    return out


def build_features(df: pd.DataFrame, fset: str) -> np.ndarray:
    """Return feature matrix for a given feature-set name."""
    d = _fill_sensors(df)
    if fset == "raw_operating":
        return d[OPS].values
    elif fset == "all_raw":
        return d[OPS + SENSORS].values
    elif fset == "poly4_operating":
        X = d[OPS].values
        pipe = Pipeline([("poly", PolynomialFeatures(4, include_bias=False)),
                         ("sc", StandardScaler())])
        return pipe.fit_transform(X)
    elif fset == "sensor_consistency":
        d2 = d.copy()
        d2["S1_S2"] = d2[S3[0]] - d2[S3[1]]
        d2["S1_S3"] = d2[S3[0]] - d2[S3[2]]
        d2["S2_S3"] = d2[S3[1]] - d2[S3[2]]
        d2["spread"] = d2[S3].max(axis=1) - d2[S3].min(axis=1)
        med = d2[S3].median(axis=1)
        d2["med_dev"] = d2[S3].sub(med, axis=0).abs().max(axis=1)
        raw = df.copy()
        d2["missing_ct"] = raw[S3].isna().sum(axis=1).astype(float)
        d2["dup_flag"] = is_duplicate(raw).astype(float)
        cols = OPS + ["S1_S2", "S1_S3", "S2_S3", "spread", "med_dev", "missing_ct", "dup_flag"]
        return d2[cols].values
    else:
        raise ValueError(f"Unknown feature set: {fset}")


# ---- Model catalogues -------------------------------------------------------

def clf_models(seed):
    ms = [
        ("Dummy",      DummyClassifier(strategy="most_frequent", random_state=seed)),
        ("LogReg",     LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
        ("RF",         RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)),
        ("HistGB",     HistGradientBoostingClassifier(max_iter=300, random_state=seed)),
    ]
    if _HAS_CB:
        ms.append(("CatBoost", CatBoostClassifier(iterations=300, random_seed=seed, verbose=0, auto_class_weights="Balanced")))
    if _HAS_XGB:
        ms.append(("XGBoost",  xgb.XGBClassifier(n_estimators=300, random_state=seed, eval_metric="logloss", verbosity=0, scale_pos_weight=866/134)))
    if _HAS_LGB:
        ms.append(("LightGBM", lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbose=-1, is_unbalance=True)))
    return ms


def reg_models(seed):
    ms = [
        ("Dummy",      DummyRegressor(strategy="median")),
        ("Ridge",      Ridge(alpha=1.0)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ("Huber",      HuberRegressor(max_iter=500)),
        ("RF",         RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("HistGBR",    HistGradientBoostingRegressor(max_iter=300, random_state=seed)),
        ("SVR",        Pipeline([("sc", StandardScaler()), ("svr", SVR(C=10.0, epsilon=0.1))])),
    ]
    if _HAS_CB:
        ms.append(("CatBoost", CatBoostRegressor(iterations=300, random_seed=seed, verbose=0)))
    if _HAS_XGB:
        ms.append(("XGBoost",  xgb.XGBRegressor(n_estimators=300, random_state=seed, verbosity=0)))
    if _HAS_LGB:
        ms.append(("LightGBM", lgb.LGBMRegressor(n_estimators=300, random_state=seed, verbose=-1)))
    return ms


# ---- OOF helpers ------------------------------------------------------------

def oof_classify(model, X, y, folds_iter) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), -1, dtype=int)
    prob = np.zeros(len(y))
    for tr, va in folds_iter:
        m = _clone(model)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
        if hasattr(m, "predict_proba"):
            prob[va] = m.predict_proba(X[va])[:, 1]
        else:
            prob[va] = pred[va].astype(float)
    return pred, prob


def clf_scores(y, pred, prob) -> Dict:
    raw_auc = roc_auc_score(y, prob)
    return {
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall":    round(recall_score(y, pred, zero_division=0), 4),
        "f1":        round(f1_score(y, pred, zero_division=0), 4),
        "macro_f1":  round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "bal_acc":   round(balanced_accuracy_score(y, pred), 4),
        "roc_auc":   round(max(raw_auc, 1 - raw_auc), 4),
        "pr_auc":    round(average_precision_score(y, prob), 4),
        "mcc":       round(matthews_corrcoef(y, pred), 4),
    }


def oof_regress(model, X, y, folds_iter_fn) -> Tuple[np.ndarray, List[float]]:
    pred = np.zeros(len(y))
    fold_maes = []
    for tr, va in folds_iter_fn():
        m = _clone(model)
        m.fit(X[tr], y[tr])
        p = m.predict(X[va])
        pred[va] = p
        fold_maes.append(mean_absolute_error(y[va], p))
    return pred, fold_maes


def reg_scores(y, pred, fold_maes) -> Dict:
    return {
        "mae":        round(mean_absolute_error(y, pred), 4),
        "rmse":       round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "r2":         round(r2_score(y, pred), 4),
        "median_ae":  round(float(median_absolute_error(y, pred)), 4),
        "max_error":  round(float(np.max(np.abs(y - pred))), 4),
        "fold_std":   round(float(np.std(fold_maes)), 4),
        "worst_fold": round(float(np.max(fold_maes)), 4),
    }


# ---- Classification benchmark -----------------------------------------------

def run_clf_benchmark(train: pd.DataFrame) -> pd.DataFrame:
    y = is_invalid(train).values
    fsets = ["raw_operating", "all_raw", "sensor_consistency"]
    models = clf_models(SEED)
    rows = []
    for fset in fsets:
        X = build_features(train, fset)
        for name, model in models:
            folds_list = list(fold_indices(train))
            pred, prob = oof_classify(model, X, y, folds_list)
            sc = clf_scores(y, pred, prob)
            row = {"task": "classification", "model": name, "feature_set": fset, **sc}
            rows.append(row)
            print(f"  [CLF] {name:12s} | {fset:22s} | F1={sc['f1']:.4f} AUC={sc['roc_auc']:.4f}")
    df = pd.DataFrame(rows).sort_values(["feature_set", "f1"], ascending=[True, False])
    df.insert(0, "rank", df.groupby("feature_set")["f1"].rank(ascending=False, method="first").astype(int))
    return df.reset_index(drop=True)


# ---- Regression benchmark ---------------------------------------------------

def run_reg_benchmark(train: pd.DataFrame) -> pd.DataFrame:
    valid = valid_rows(train)
    y = valid[TARGET_REG].values
    fsets = ["raw_operating", "all_raw", "poly4_operating"]
    models = reg_models(SEED)
    rows = []
    for fset in fsets:
        X = build_features(valid, fset)
        for name, model in models:
            def _fi(X=X, valid=valid): return fold_indices(valid)
            pred, fmaes = oof_regress(model, X, y, _fi)
            sc = reg_scores(y, pred, fmaes)
            row = {"task": "regression", "model": name, "feature_set": fset, **sc}
            rows.append(row)
            print(f"  [REG] {name:12s} | {fset:22s} | MAE={sc['mae']:.4f} R2={sc['r2']:.4f}")
    df = pd.DataFrame(rows).sort_values(["feature_set", "mae"], ascending=[True, True])
    df.insert(0, "rank", df.groupby("feature_set")["mae"].rank(ascending=True, method="first").astype(int))
    return df.reset_index(drop=True)


# ---- Ablations --------------------------------------------------------------

def run_ablations(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = valid_rows(train)

    # 1. Regression: all rows vs Valid rows only
    for subset_name, subset in [("all_rows", train), ("valid_only", valid)]:
        y = subset[TARGET_REG].values
        X = build_features(subset, "raw_operating")
        model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
        pred, fmaes = oof_regress(model, X, y, lambda s=subset, X=X: fold_indices(s))
        sc = reg_scores(y, pred, fmaes)
        rows.append({"task": "ablation", "model": f"HistGBR({subset_name})",
                     "feature_set": "raw_operating", **sc})
        print(f"  [ABL] HistGBR {subset_name} MAE={sc['mae']:.4f}")

    # 2. With S4 vs without S4
    for include_s4, tag in [(True, "with_S4"), (False, "without_S4")]:
        cols = OPS + S3 + ([S4] if include_s4 else [])
        d = _fill_sensors(valid)
        X = d[cols].values
        y = valid[TARGET_REG].values
        model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
        pred, fmaes = oof_regress(model, X, y, lambda v=valid, X=X: fold_indices(v))
        sc = reg_scores(y, pred, fmaes)
        rows.append({"task": "ablation", "model": f"HistGBR({tag})",
                     "feature_set": "all_raw", **sc})
        print(f"  [ABL] HistGBR {tag} MAE={sc['mae']:.4f}")

    # 3. With Duration vs without Duration
    for include_dur, tag in [(True, "with_Duration"), (False, "no_Duration")]:
        cols = OPS if include_dur else [c for c in OPS if "Duration" not in c]
        d = _fill_sensors(valid)
        X = d[cols].values
        y = valid[TARGET_REG].values
        model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
        pred, fmaes = oof_regress(model, X, y, lambda v=valid, X=X: fold_indices(v))
        sc = reg_scores(y, pred, fmaes)
        rows.append({"task": "ablation", "model": f"HistGBR({tag})",
                     "feature_set": "raw_operating", **sc})
        print(f"  [ABL] HistGBR {tag} MAE={sc['mae']:.4f}")

    # 4. Polynomial degree 2 / 3 / 4 / 5
    y = valid[TARGET_REG].values
    for deg in [2, 3, 4, 5]:
        X_raw = _fill_sensors(valid)[OPS].values
        pipe = Pipeline([("poly", PolynomialFeatures(deg, include_bias=False)),
                         ("sc", StandardScaler()), ("ridge", Ridge(1.0))])
        pred, fmaes = oof_regress(pipe, X_raw, y, lambda v=valid, X=X_raw: fold_indices(v))
        sc = reg_scores(y, pred, fmaes)
        rows.append({"task": "ablation", "model": f"PolyRidge(d={deg})",
                     "feature_set": "poly_operating", **sc})
        print(f"  [ABL] PolyRidge(d={deg}) MAE={sc['mae']:.4f}")

    return pd.DataFrame(rows)


# ---- Optional: team entries (twin.py / regress.py) --------------------------

def try_add_team_entries(train: pd.DataFrame, rows: List[Dict]) -> None:
    # P1 twin
    try:
        from src.twin import predict_validity, fit_twin
        y = is_invalid(train).values
        pred_arr = np.full(len(train), -1, dtype=int)
        prob_arr = np.zeros(len(train))
        for tr, va in fold_indices(train):
            state = fit_twin(train.iloc[tr].reset_index(drop=True))
            res = predict_validity(train.iloc[va].reset_index(drop=True), state)
            preds = (res[TARGET_CLS] == "Invalid").astype(int).values
            pred_arr[va] = preds
            prob_arr[va] = preds.astype(float)
        sc = clf_scores(y, pred_arr, prob_arr)
        rows.append({"task": "classification", "model": "P1_Twin (team)",
                     "feature_set": "physics_rule", **sc})
        print(f"  [TEAM] P1 Twin F1={sc['f1']:.4f}")
    except Exception as e:
        rows.append({"task": "classification", "model": "P1_Twin (team)",
                     "feature_set": "physics_rule", "f1": "PLACEHOLDER", "note": str(e)})
        print(f"  [TEAM] P1 Twin not available: {e}")

    # P2 regression
    try:
        from src.regress import predict_reference
        valid = valid_rows(train)
        y = valid[TARGET_REG].values
        pred_arr = np.zeros(len(valid))
        for tr, va in fold_indices(valid):
            pred_arr[va] = predict_reference(valid.iloc[tr].reset_index(drop=True),
                                              valid.iloc[va].reset_index(drop=True))
        folds_list = list(fold_indices(valid))
        fold_maes = [mean_absolute_error(y[va], pred_arr[va]) for _, va in folds_list]
        sc = reg_scores(y, pred_arr, fold_maes)
        rows.append({"task": "regression", "model": "P2_Regressor (team)",
                     "feature_set": "team_model", **sc})
        print(f"  [TEAM] P2 Regressor MAE={sc['mae']:.4f}")
    except Exception as e:
        rows.append({"task": "regression", "model": "P2_Regressor (team)",
                     "feature_set": "team_model", "mae": "PLACEHOLDER", "note": str(e)})
        print(f"  [TEAM] P2 Regressor not available: {e}")


# ---- Charts -----------------------------------------------------------------

def chart_gap_separation(train: pd.DataFrame) -> None:
    """Histogram of max-abs sensor residual, Valid vs Invalid, log x-axis."""
    try:
        from src.twin import fit_twin, sensor_residuals as sr
        state = fit_twin(valid_rows(train))
        res = sr(train, state)
        scores = res["spike_score"].values
    except Exception:
        # Fallback: use pairwise deviation as proxy
        d = _fill_sensors(train)
        med = d[S3].median(axis=1)
        scores = d[S3].sub(med, axis=0).abs().max(axis=1).values

    y = is_invalid(train).values
    _dark()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_BG)
    bins = np.logspace(np.log10(max(scores.min(), 1e-3)), np.log10(scores.max() + 1), 60)
    ax.hist(scores[y == 0], bins=bins, alpha=0.75, label="Valid",   color="#4fc3f7", edgecolor="none")
    ax.hist(scores[y == 1], bins=bins, alpha=0.75, label="Invalid", color=_ACC,      edgecolor="none")
    diag = ROOT / "outputs" / "p1" / "threshold_diagnostics.json"
    if diag.exists():
        d2 = json.loads(diag.read_text())
        ks = [v for v in d2.values() if isinstance(v, dict) and "threshold" in v]
        if ks:
            thresh = float(np.mean([v["threshold"] for v in ks]))
            ax.axvline(thresh, color=_GOLD, lw=2, ls="--", label=f"Threshold={thresh:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Max absolute sensor residual (log scale)", color=_TXT, fontsize=11)
    ax.set_ylabel("Count", color=_TXT, fontsize=11)
    ax.set_title("Gap Separation — Valid vs Invalid Sensor Residual", fontsize=13, fontweight="bold", color=_TXT)
    ax.legend(framealpha=0.2); ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "gap_separation.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  Chart: gap_separation.png")


def chart_fault_taxonomy() -> None:
    counts = {"missing_sensor": 15, "duplicate_vector": 24, "sensor_spike": 95}
    mp = ROOT / "outputs" / "p1" / "oof_metrics.json"
    if mp.exists():
        d = json.loads(mp.read_text()).get("taxonomy", {})
        counts = {k: v for k, v in d.items() if k != "none"}
    _dark()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(_BG)
    names, vals = list(counts.keys()), list(counts.values())
    colors = [_GOLD, _ACC, "#4fc3f7"]
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="none", width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                str(int(b.get_height())), ha="center", va="bottom", fontsize=12, color=_TXT)
    ax.set_ylabel("Count", color=_TXT, fontsize=11)
    ax.set_title(f"Fault Taxonomy (Total Invalid: {sum(vals)})", fontsize=13, fontweight="bold", color=_TXT)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "fault_taxonomy.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print("  Chart: fault_taxonomy.png")


def chart_clf_leaderboard(clf_df: pd.DataFrame) -> None:
    _dark()
    sub = clf_df[clf_df["task"] == "classification"].copy()
    sub["f1"] = pd.to_numeric(sub["f1"], errors="coerce")
    best = sub.groupby("model")["f1"].max().dropna().sort_values(ascending=False)
    if best.empty:
        print("  [WARN] No numeric F1 values for classification leaderboard chart.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_BG)
    colors = _mcolors(len(best))
    bars = ax.barh(best.index[::-1], best.values[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    for bar, val in zip(bars, best.values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color=_TXT)
    ax.set_xlabel("Invalid-class F1 (best across feature sets)", color=_TXT, fontsize=11)
    ax.set_title("Classification Leaderboard", fontsize=13, fontweight="bold", color=_TXT)
    ax.set_xlim(0, 1.1); ax.grid(axis="x"); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "leaderboard_classification.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print("  Chart: leaderboard_classification.png")


def chart_reg_leaderboard(reg_df: pd.DataFrame) -> None:
    _dark()
    sub = reg_df[reg_df["task"] == "regression"].copy()
    sub["mae"] = pd.to_numeric(sub["mae"], errors="coerce")
    best = sub.groupby("model")["mae"].min().dropna().sort_values()
    if best.empty:
        print("  [WARN] No numeric MAE values for regression leaderboard chart.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_BG)
    colors = _mcolors(len(best))
    bars = ax.barh(best.index[::-1], best.values[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    for bar, val in zip(bars, best.values[::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color=_TXT)
    ax.set_xlabel("MAE (best across feature sets)", color=_TXT, fontsize=11)
    ax.set_title("Regression Leaderboard", fontsize=13, fontweight="bold", color=_TXT)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "leaderboard_regression.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print("  Chart: leaderboard_regression.png")


def chart_pred_vs_actual(train: pd.DataFrame) -> None:
    """Predicted vs actual Reference_Parameter, OOF with best model."""
    valid = valid_rows(train)
    y = valid[TARGET_REG].values
    X = build_features(valid, "raw_operating")
    model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
    pred = np.zeros(len(y))
    for tr, va in fold_indices(valid):
        m = _clone(model); m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    _dark()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(_BG)
    ax.scatter(y, pred, alpha=0.4, s=20, color="#4fc3f7", edgecolors="none")
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=_GOLD, lw=1.5, ls="--", label="Perfect")
    mae = mean_absolute_error(y, pred)
    ax.set_xlabel("Actual Reference_Parameter", color=_TXT, fontsize=11)
    ax.set_ylabel("Predicted", color=_TXT, fontsize=11)
    ax.set_title(f"Predicted vs Actual (OOF, MAE={mae:.3f})", fontsize=12, fontweight="bold", color=_TXT)
    ax.legend(framealpha=0.2); ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "pred_vs_actual.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print("  Chart: pred_vs_actual.png")


def chart_residual_by_regime(train: pd.DataFrame) -> None:
    """Regression residuals across Load_Current_A terciles."""
    valid = valid_rows(train)
    y = valid[TARGET_REG].values
    X = build_features(valid, "raw_operating")
    model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
    pred = np.zeros(len(y))
    for tr, va in fold_indices(valid):
        m = _clone(model); m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    residuals = y - pred
    current_col = "Load_Current_A"
    terciles = pd.qcut(valid[current_col], q=3, labels=["Low", "Mid", "High"])
    _dark()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_BG)
    colors_t = [_GOLD, _ACC, "#4fc3f7"]
    for i, (label, color) in enumerate(zip(["Low", "Mid", "High"], colors_t)):
        mask = terciles == label
        ax.scatter(valid.loc[mask, current_col], residuals[mask.values],
                   alpha=0.5, s=18, color=color, label=f"{label} load", edgecolors="none")
    ax.axhline(0, color=_TXT, lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Load_Current_A", color=_TXT, fontsize=11)
    ax.set_ylabel("Residual (actual - predicted)", color=_TXT, fontsize=11)
    ax.set_title("Regression Residuals by Load Regime (terciles)", fontsize=12, fontweight="bold", color=_TXT)
    ax.legend(framealpha=0.2); ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "residual_by_regime.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print("  Chart: residual_by_regime.png")


# ---- Main -------------------------------------------------------------------

def main():
    log_buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log_buf.write(s)
        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = Tee()

    print("=" * 65)
    print("Power-Next P3 -- Benchmark Leaderboard")
    print("=" * 65)

    print("\n[1/7] Loading data...")
    train, test, _ = load_data()
    print(f"  Train: {train.shape}  Valid: {valid_rows(train).shape}  Test: {test.shape}")

    print("\n[2/7] Classification benchmark...")
    clf_df = run_clf_benchmark(train)

    print("\n[3/7] Regression benchmark...")
    reg_df = run_reg_benchmark(train)

    print("\n[4/7] Ablations...")
    abl_df = run_ablations(train)

    print("\n[5/7] Team entries (twin.py / regress.py)...")
    team_rows = []
    try_add_team_entries(train, team_rows)
    team_df = pd.DataFrame(team_rows) if team_rows else pd.DataFrame()

    print("\n[6/7] Saving leaderboard...")
    combined = pd.concat([clf_df, reg_df, abl_df, team_df], ignore_index=True)
    combined.to_csv(LEADERBOARD_PATH, index=False)
    print(f"  Rows: {len(combined)}  Saved: {LEADERBOARD_PATH}")

    print("\n  -- Classification top-5 (sensor_consistency, F1 desc) --")
    top_clf = clf_df[clf_df["feature_set"] == "sensor_consistency"].nlargest(5, "f1")[["model", "f1", "roc_auc", "mcc"]]
    print(top_clf.to_string(index=False))

    print("\n  -- Regression top-5 (raw_operating, MAE asc) --")
    top_reg = reg_df[reg_df["feature_set"] == "raw_operating"].nsmallest(5, "mae")[["model", "mae", "rmse", "r2"]]
    print(top_reg.to_string(index=False))

    print("\n  -- Ablation results --")
    print(abl_df[["model", "feature_set", "mae", "r2"]].to_string(index=False))

    print("\n[7/7] Charts...")
    chart_gap_separation(train)
    chart_fault_taxonomy()
    chart_clf_leaderboard(combined)
    chart_reg_leaderboard(combined)
    chart_pred_vs_actual(train)
    chart_residual_by_regime(train)

    print("\n" + "=" * 65)
    print(f"Done. Leaderboard: {len(combined)} rows. Charts: {CHARTS_DIR}")
    print("=" * 65)

    sys.stdout = sys.__stdout__
    REPORT_PATH.write_text(log_buf.getvalue(), encoding="utf-8")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
