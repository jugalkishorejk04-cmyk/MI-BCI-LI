"""
20_external_target_ml.py
------------------------
Item 5: restore a cross-population machine-learning result WITHOUT
circularity, by predicting an EXTERNAL target rather than the sign of DLI.

The V11 classifier (AUC 0.92) predicted mu_responder — the sign of DLI —
from features that included selectivity (= 0.57 x DLI) and LI_early (a
component of DLI). That is predicting sign(X) from transforms of X. This
script replaces the TARGET, not the features, so DLI-derived features are
legitimate predictors again.

Two models are run.

MODEL A — cross-dataset transfer, target = decoding-accuracy tertile
  Leave-one-dataset-out: train on all-but-one healthy cohort, test on the
  held-out one. Target is per-subject decoding accuracy binned into
  tertiles (or high-vs-rest binary). Accuracy is computed on the full
  montage and is independent of the C3/C4 ERD trajectory.
  NOTE: cohorts whose decoder is at chance cannot serve as a test set.
  Liu 2024 (mean acc ~0.56) is expected to be excluded on this ground.

MODEL B — within-stroke, target = NIHSS band
  Liu 2024 only, leave-one-out CV, target = NIHSS above/below the cohort
  median. Clinically the more meaningful model, but n = 50 — report with
  confidence intervals and a permutation null, and do NOT over-interpret.

Both report a PERMUTATION NULL (target shuffled) so that a modest AUC can
be judged against chance for the given n rather than against 0.5 in the
abstract.

INPUTS
  features_per_subject.csv          : DLI, selectivity, ERD slopes, LI stages
  behavioural_validation_within.csv : decode_acc per subject
  liu_clinical.csv                  : subject, NIHSS, hemiplegia_side
  blankertz_smr_per_subject.csv     : smr_snr_mean  (optional extra feature)
  ahn_theta_alpha_per_subject.csv   : ahn_ratio     (optional extra feature)

OUTPUT
  external_target_ml_report.txt
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
FEAT_F  = "features_per_subject.csv"
BEHAV_F = "behavioural_validation_within.csv"
CLIN_F  = "liu_clinical.csv"
SMR_F   = "blankertz_smr_per_subject.csv"
AHN_F   = "ahn_theta_alpha_per_subject.csv"
REPORT  = "external_target_ml_report.txt"
MIN_MEAN_ACC = 0.58
N_PERM = 1000

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


def read(p):
    try:
        d = pd.read_csv(p)
        if "subject" in d:
            d["subject"] = d["subject"].astype(str)
        return d
    except Exception as e:
        say(f"!! {p}: {e}")
        return None

feat, behav = read(FEAT_F), read(BEHAV_F)
clin, smr, ahn = read(CLIN_F), read(SMR_F), read(AHN_F)
if feat is None or behav is None:
    raise SystemExit("features and behavioural files are required")

key = ["dataset", "subject"]
M = feat.merge(behav, on=key, how="inner", suffixes=("", "_b"))
for extra, col in [(smr, "smr_snr_mean"), (ahn, "ahn_ratio")]:
    if extra is not None and col in extra:
        M = M.merge(extra.groupby(key, as_index=False)[col].mean(), on=key, how="left")

acc_col = next((c for c in M.columns if "acc" in c.lower()), None)
say(f"merged rows: {len(M)}   accuracy column: {acc_col}")

FEATURES = [c for c in M.columns
            if any(k in c.lower() for k in
                   ["dli", "select", "slope", "li_early", "li_mid", "li_late",
                    "smr_snr", "ahn_ratio"])
            and "acc" not in c.lower()]
say(f"features ({len(FEATURES)}): {FEATURES}")


def perm_null(model, X, y, cv, n=N_PERM):
    aucs = []
    for _ in range(n):
        yp = RNG.permutation(y)
        try:
            pr = cross_val_predict(model, X, yp, cv=cv, method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(yp, pr))
        except Exception:
            pass
    return np.array(aucs)


# ----------------------------------------------------------------------
# MODEL A — LODO transfer, target = high vs low decoding accuracy
# ----------------------------------------------------------------------
say("\n" + "=" * 72)
say("MODEL A — cross-dataset transfer, target = decoding accuracy (high vs low)")
say("=" * 72)

acc_by = M.groupby("dataset")[acc_col].mean().round(3)
say(acc_by.to_string())
usable = acc_by[acc_by >= MIN_MEAN_ACC].index.tolist()
say(f"cohorts with a usable decoder: {usable}")

A = M[M.dataset.isin(usable)].dropna(subset=FEATURES + [acc_col]).copy()
# target defined WITHIN each cohort so cohort-level accuracy differences
# do not leak into the label
A["target"] = A.groupby("dataset")[acc_col].transform(
    lambda s: (s > s.median()).astype(int))

for held in usable:
    tr = A[A.dataset != held]
    te = A[A.dataset == held]
    if len(te) < 15 or te.target.nunique() < 2:
        say(f"  hold out {held}: insufficient test data")
        continue
    for nm, clf in [("GB", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                      random_state=42)),
                    ("LR", LogisticRegression(max_iter=1000))]:
        pipe = Pipeline([("sc", StandardScaler()), ("m", clf)])
        pipe.fit(tr[FEATURES], tr.target)
        pr = pipe.predict_proba(te[FEATURES])[:, 1]
        auc = roc_auc_score(te.target, pr)
        say(f"  train on others -> test {held:14s} {nm}  n_test={len(te):3d}  AUC={auc:.3f}")

# ----------------------------------------------------------------------
# MODEL B — within-stroke, target = NIHSS band
# ----------------------------------------------------------------------
say("\n" + "=" * 72)
say("MODEL B — within-stroke, target = NIHSS above/below median (Liu 2024)")
say("=" * 72)

if clin is None:
    say("liu_clinical.csv not found — skipping Model B.")
else:
    L = M[M.dataset.str.contains("Liu", case=False)].merge(
        clin, on="subject", how="inner")
    nih = next((c for c in L.columns if "nihss" in c.lower()), None)
    if nih is None or len(L) < 25:
        say(f"insufficient clinical data (n={len(L)}, NIHSS col={nih})")
    else:
        L = L.dropna(subset=FEATURES + [nih])
        L["target"] = (L[nih] > L[nih].median()).astype(int)
        say(f"n = {len(L)}   class balance = {L.target.value_counts().to_dict()}")
        loo = LeaveOneOut()
        for nm, clf in [("LR", LogisticRegression(max_iter=1000)),
                        ("GB", GradientBoostingClassifier(n_estimators=100,
                                                          max_depth=2, random_state=42))]:
            pipe = Pipeline([("sc", StandardScaler()), ("m", clf)])
            pr = cross_val_predict(pipe, L[FEATURES], L.target,
                                   cv=loo, method="predict_proba")[:, 1]
            auc = roc_auc_score(L.target, pr)
            null = perm_null(pipe, L[FEATURES].values, L.target.values, loo, n=200)
            p = (np.sum(null >= auc) + 1) / (len(null) + 1)
            say(f"  {nm}  LOOCV AUC = {auc:.3f}   permutation null mean = "
                f"{null.mean():.3f} (95th pct {np.percentile(null,95):.3f})   p = {p:.4f}")

        # lesion-side stratified, matching the NIHSS finding in section 3.7
        side = next((c for c in L.columns if "hemipleg" in c.lower() or "side" in c.lower()), None)
        if side:
            for sv, g in L.groupby(side):
                if len(g) < 18 or g.target.nunique() < 2:
                    continue
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("m", LogisticRegression(max_iter=1000))])
                pr = cross_val_predict(pipe, g[FEATURES], g.target,
                                       cv=LeaveOneOut(), method="predict_proba")[:, 1]
                say(f"  side={sv}  n={len(g)}  LOOCV AUC = {roc_auc_score(g.target, pr):.3f}")

say("\nREPORTING RULES (fixed before results):")
say("  - Report AUC against the PERMUTATION null, not against 0.5.")
say("  - Model B at n=50 is under-powered; report with CI and do not present")
say("    it as a screening tool.")
say("  - If neither model exceeds its permutation null, report as a negative")
say("    finding: DLI-derived features do not predict external targets.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
