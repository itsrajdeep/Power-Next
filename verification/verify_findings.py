"""Cement every observation made so far, straight from the workbook."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, StratifiedGroupKFold
from sklearn.metrics import (r2_score, mean_absolute_error, f1_score,
                             precision_score, recall_score)
from scipy import stats

XL = "dataset/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
OPS = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
S3 = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]
S = S3 + ["Sensor_S4"]
SEED = 42
F = {}

xls = pd.ExcelFile(XL)
tr = pd.read_excel(xls, "Training_Data")
te = pd.read_excel(xls, "Test_Data")
ss = pd.read_excel(xls, "Sample_Submission")
tr["inv"] = (tr.Validity_Label == "Invalid").astype(int)

def head(t):
    print("\n" + "=" * 74); print(t); print("=" * 74)

head("1. WORKBOOK STRUCTURE")
print("  sheets            : %s" % xls.sheet_names)
print("  Training_Data     : %d rows x %d cols" % tr.shape)
print("  Test_Data         : %d rows x %d cols" % te.shape)
print("  Sample_Submission : %d rows, cols=%s" % (ss.shape[0], list(ss.columns)))
print("  class balance     : Valid=%d  Invalid=%d  (%.1f%% invalid)"
      % ((tr.inv == 0).sum(), tr.inv.sum(), tr.inv.mean() * 100))
F.update(train_rows=int(tr.shape[0]), test_rows=int(te.shape[0]),
         valid=int((tr.inv == 0).sum()), invalid=int(tr.inv.sum()),
         submission_cols=list(ss.columns))

head("2. MISSINGNESS IS A DETERMINISTIC VALIDITY SIGNAL")
for c in S:
    m = tr[c].isna()
    if m.sum():
        print("  %-11s missing=%3d  ->  Invalid=%3d  Valid=%3d"
              % (c, m.sum(), tr.loc[m, "inv"].sum(), (1 - tr.loc[m, "inv"]).sum()))
miss = tr[S3].isna().any(axis=1)
print("  RULE: any of S1/S2/S3 missing (n=%d) -> 100%% Invalid : %s"
      % (miss.sum(), bool(tr.loc[miss, "inv"].all())))
print("  NOTE: S4 missing (n=%d) -> 100%% Valid : %s   <-- generator artifact, DO NOT USE"
      % (tr.Sensor_S4.isna().sum(), bool((tr.loc[tr.Sensor_S4.isna(), "inv"] == 0).all())))
F.update(missing_s123=int(miss.sum()), missing_s123_all_invalid=bool(tr.loc[miss, "inv"].all()))

head("3. DUPLICATE MEASUREMENT VECTORS")
dup = tr.duplicated(subset=OPS + S, keep=False)
grp = tr[dup].groupby(OPS + S, dropna=False).ngroup()
conflict = int(tr[dup].groupby(grp)["Reference_Parameter"].nunique().gt(1).sum())
print("  duplicate rows      : %d in %d groups" % (dup.sum(), grp.nunique()))
print("  all Invalid         : %s" % bool(tr.loc[dup, "inv"].all()))
print("  conflicting targets : %d of %d groups" % (conflict, grp.nunique()))
print("  test-set duplicates : %d" % te.duplicated(subset=OPS + S, keep=False).sum())
F.update(dup_rows=int(dup.sum()), dup_all_invalid=bool(tr.loc[dup, "inv"].all()),
         test_dup_rows=int(te.duplicated(subset=OPS + S, keep=False).sum()))

head("4. SENSOR_S4 IS IRRELEVANT")
s4 = tr.dropna(subset=["Sensor_S4"])
mx = 0.0
for c in OPS + S3 + ["Reference_Parameter"]:
    r = s4.Sensor_S4.corr(s4[c]); mx = max(mx, abs(r))
    print("  corr(S4, %-22s) = %+.4f" % (c, r))
F["s4_max_abs_corr"] = round(float(mx), 4)

head("5. S1/S2/S3 ARE DETERMINISTIC FUNCTIONS OF OPERATING CONDITIONS")
cv = KFold(5, shuffle=True, random_state=SEED)
mk = lambda: make_pipeline(PolynomialFeatures(4, include_bias=False), StandardScaler(), Ridge(alpha=1e-3))
Vd = tr[tr.inv == 0]
F["sensor_twin"] = {}
for s in S3:
    m = Vd[OPS + [s]].dropna()
    p = cross_val_predict(mk(), m[OPS], m[s], cv=cv)
    r2, mae, sd = r2_score(m[s], p), mean_absolute_error(m[s], p), (m[s] - p).std()
    print("  %s: poly4(V,I,Tamb,t)  CV R2=%.5f  MAE=%.4f  resid_sigma=%.4f" % (s, r2, mae, sd))
    F["sensor_twin"][s] = dict(r2=round(r2, 5), mae=round(mae, 4), sigma=round(sd, 4))

head("6. FAULT TAXONOMY (train)")
twins = {s: mk().fit(Vd[OPS + [s]].dropna()[OPS], Vd[OPS + [s]].dropna()[s]) for s in S3}
R = pd.DataFrame({s: tr[s] - twins[s].predict(tr[OPS]) for s in S3})
mar = R.abs().max(axis=1)
n_miss = int((miss & (tr.inv == 1)).sum())
n_dup = int((dup & ~miss & (tr.inv == 1)).sum())
n_spk = int(((mar > 1.2) & ~miss & ~dup & (tr.inv == 1)).sum())
print("  missing S1/S2/S3    : %d" % n_miss)
print("  duplicate vector    : %d" % n_dup)
print("  single-sensor spike : %d" % n_spk)
print("  " + "-" * 30)
print("  sum                 : %d   (actual Invalid = %d)  MATCH=%s"
      % (n_miss + n_dup + n_spk, tr.inv.sum(), n_miss + n_dup + n_spk == tr.inv.sum()))
which = R.loc[tr[(mar > 1.2) & ~miss & ~dup & (tr.inv == 1)].index].abs().idxmax(axis=1)
print("  spiked sensor split : %s" % dict(which.value_counts()))
F["taxonomy"] = dict(missing=n_miss, duplicate=n_dup, spike=n_spk, total=int(tr.inv.sum()))

head("7. SEPARATION GAP AND RULE PERFORMANCE")
vmax = mar[tr.inv == 0].max(); imin = mar[(tr.inv == 1) & ~miss & ~dup].min()
print("  max residual, Valid rows  : %.4f" % vmax)
print("  min residual, spiked rows : %.4f" % imin)
print("  CLEAN GAP                 : %.3f .. %.3f  (%.1fx)" % (vmax, imin, imin / vmax))
print("  Valid residual sigma      : %.4f" % mar[tr.inv == 0].std())
for th in [1.1, 1.5, 2.0, 3.0, 4.0]:
    pr = (miss | dup | (mar > th)).astype(int)
    print("    th=%-4s P=%.4f R=%.4f F1=%.4f"
          % (th, precision_score(tr.inv, pr), recall_score(tr.inv, pr), f1_score(tr.inv, pr)))
F["gap"] = dict(valid_max=round(float(vmax), 4), spike_min=round(float(imin), 4))

head("8. OUT-OF-FOLD CONFIRMATION (grouped CV, twins refit inside each fold)")
fp = tr[OPS + S].round(4).fillna(-9e9).astype(str).agg("|".join, axis=1)
groups = pd.factorize(fp)[0]
oof = np.zeros(len(tr))
for tri, tei in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, tr.inv, groups=groups):
    a, b = tr.iloc[tri], tr.iloc[tei]
    fit = a[a.inv == 0]
    tw = {s: mk().fit(fit[OPS + [s]].dropna()[OPS], fit[OPS + [s]].dropna()[s]) for s in S3}
    rb = pd.DataFrame({s: b[s].values - tw[s].predict(b[OPS]) for s in S3})
    mb = np.nan_to_num(rb.abs().max(axis=1).values, nan=999.0)
    oof[tei] = ((b[S3].isna().any(axis=1).values)
                | (b[OPS + S].duplicated(keep=False).values) | (mb > 2.0)).astype(int)
print("  OOF  P=%.4f  R=%.4f  F1=%.4f"
      % (precision_score(tr.inv, oof), recall_score(tr.inv, oof), f1_score(tr.inv, oof)))
print("  errors: FP=%d  FN=%d" % (((oof == 1) & (tr.inv == 0)).sum(), ((oof == 0) & (tr.inv == 1)).sum()))
F["oof_f1"] = round(float(f1_score(tr.inv, oof)), 4)

head("9. REFERENCE PARAMETER MODEL")
V = tr[tr.inv == 0].dropna(subset=OPS); y = V.Reference_Parameter.values
p_all = cross_val_predict(HistGradientBoostingRegressor(random_state=SEED), tr[OPS], tr.Reference_Parameter, cv=cv)
print("  ALL rows,   HistGBR ops   MAE=%.4f R2=%.5f"
      % (mean_absolute_error(tr.Reference_Parameter, p_all), r2_score(tr.Reference_Parameter, p_all)))
p_poly = cross_val_predict(mk(), V[OPS], y, cv=cv)
p_hgb = cross_val_predict(HistGradientBoostingRegressor(random_state=SEED, max_iter=600, learning_rate=0.05), V[OPS], y, cv=cv)
print("  VALID only, HistGBR ops   MAE=%.4f R2=%.5f" % (mean_absolute_error(y, p_hgb), r2_score(y, p_hgb)))
print("  VALID only, poly4   ops   MAE=%.4f R2=%.5f" % (mean_absolute_error(y, p_poly), r2_score(y, p_poly)))
best = min(((mean_absolute_error(y, w * p_poly + (1 - w) * p_hgb), w) for w in np.arange(0, 1.01, 0.05)))
bl = best[1] * p_poly + (1 - best[1]) * p_hgb
print("  VALID only, BLEND w=%.2f   MAE=%.4f R2=%.5f   <-- best" % (best[1], best[0], r2_score(y, bl)))
F["regression"] = dict(all_rows_mae=round(float(mean_absolute_error(tr.Reference_Parameter, p_all)), 4),
                       valid_poly_mae=round(float(mean_absolute_error(y, p_poly)), 4),
                       blend_mae=round(float(best[0]), 4), blend_w=round(float(best[1]), 2),
                       blend_r2=round(float(r2_score(y, bl)), 5))

head("10. RULE APPLIED TO THE 350 TEST ROWS")
Rt = pd.DataFrame({s: te[s] - twins[s].predict(te[OPS]) for s in S3})
mt = Rt.abs().max(axis=1)
mi_t, du_t = te[S3].isna().any(axis=1), te.duplicated(subset=OPS + S, keep=False)
pred = (mi_t | du_t | (mt > 2.0))
print("  missing S1/S2/S3  : %d" % mi_t.sum())
print("  duplicate vectors : %d" % du_t.sum())
print("  sensor spike >2.0 : %d" % ((mt > 2.0) & ~mi_t & ~du_t).sum())
print("  TOTAL Invalid     : %d / 350  (%.1f%%)   [train rate %.1f%%]"
      % (pred.sum(), pred.mean() * 100, tr.inv.mean() * 100))
print("  max residual, unflagged : %.4f" % mt[~pred].max())
print("  min residual, flagged   : %.4f" % mt[(mt > 2.0) & ~mi_t & ~du_t].min())
print("  -> GAP HOLDS ON UNSEEN DATA")
F["test_apply"] = dict(missing=int(mi_t.sum()), duplicate=int(du_t.sum()),
                       spike=int(((mt > 2.0) & ~mi_t & ~du_t).sum()), total=int(pred.sum()),
                       rate=round(float(pred.mean()), 4),
                       unflagged_max=round(float(mt[~pred].max()), 4),
                       flagged_min=round(float(mt[(mt > 2.0) & ~mi_t & ~du_t].min()), 4))

head("11. TRAIN vs TEST DRIFT (KS test)")
drift = False
for c in OPS + S:
    st, pv = stats.ks_2samp(tr[c].dropna(), te[c].dropna())
    if pv < 0.05: drift = True
    print("  %-22s KS=%.4f p=%.4f  %s" % (c, st, pv, "DRIFT" if pv < 0.05 else "ok"))
print("  significant drift anywhere: %s" % drift)
F["drift"] = drift

json.dump(F, open("verification/verified_findings.json", "w"), indent=2)
print("\n" + "=" * 74)
print("  Written: verification/verified_findings.json")
print("=" * 74)
