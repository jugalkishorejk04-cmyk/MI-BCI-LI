"""
24_modelB_unbiased_and_lee_coverage.py
--------------------------------------
Two jobs.

PART 1  Diagnose WHY features_per_subject.csv covers only 19 of Lee's 54
        subject-sessions for the ERD-slope columns. The previous diagnostic
        established that mu_ERD_slope_contra_T1 is the culprit (35/54 NaN)
        and that SMR is not. This part finds out why.

PART 2  Re-run Model B without the leave-one-out bias.

WHY MODEL B IS BEING RE-RUN
  The LOOCV permutation nulls came out centred at 0.309 (n=50) and 0.279
  (n=22) rather than 0.5. That is a known property of LOOCV, not a broken
  permutation: holding out one sample leaves the training set one member
  short in that sample's class, so the model leans away from it, held-out
  scores invert slightly, and null AUC falls below 0.5. The bias grows as n
  falls, which is exactly the pattern observed.

  The permutation p-values are therefore internally valid — but the OBSERVED
  AUCs are deflated by the same mechanism (T1 selectivity gives rho = -0.436
  against NIHSS, which should be an AUC near 0.72, not 0.646).

  Three estimators are reported side by side:
    (a) DIRECT univariate AUC — no model fitting, no CV, no bias. For a
        single feature this is the correct statistic; there is nothing to
        cross-validate. Null centres at 0.5.
    (b) STRATIFIED 5-FOLD CV — for multi-feature sets. Much smaller bias
        than LOOCV; null should sit near 0.5.
    (c) LOOCV — reported for continuity with the previous run.

  A pure-noise feature is also pushed through the LOOCV pipeline as a
  sanity check: if its null reproduces ~0.31, the bias explanation is
  confirmed and can be stated in the manuscript.

OUTPUT: modelB_unbiased_report.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
N_PERM = 2000
REPORT = "modelB_unbiased_report.txt"

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def rd(p):
    d = pd.read_csv(p)
    for c in ("subject", "session"):
        if c in d:
            d[c] = d[c].astype(str)
    return d

behav = rd("behavioural_validation_within.csv")
feat  = rd("features_per_subject.csv")
smr   = rd("blankertz_smr_per_subject.csv")
ahn   = rd("ahn_theta_alpha_per_subject.csv")
clin  = rd("liu_clinical.csv")

# ======================================================================
# PART 1 — why is Lee's ERD-slope coverage only 19/54?
# ======================================================================
say("=" * 74)
say("PART 1 — Lee ERD-slope coverage")
say("=" * 74)

fl = feat[feat.dataset.str.contains("Lee", case=False)] if "dataset" in feat else feat
bl = behav[behav.dataset.str.contains("Lee", case=False)]

say(f"features_per_subject.csv  Lee rows: {len(fl)}")
if "session" in fl:
    say(f"  sessions present: {sorted(fl.session.unique())}")
    say(f"  rows per session: {fl.groupby('session').size().to_dict()}")
say(f"  unique subjects: {fl.subject.nunique()}")
say(f"  subject id examples: {sorted(fl.subject.unique())[:8]}")

slope_col = next((c for c in fl.columns if "mu_ERD_slope_contra_T1" in c), None)
if slope_col:
    say(f"\n  {slope_col} NaN within the feature file itself: "
        f"{int(fl[slope_col].isna().sum())} / {len(fl)}")
    say("  -> if this is ~35, the gap is in FEATURE EXTRACTION, not the merge.")
    say("  -> if this is 0, the gap is a MERGE KEY mismatch (session or id format).")

say(f"\nbehavioural file Lee rows: {len(bl)}   unique subjects: {bl.subject.nunique()}")
if "session" in bl:
    say(f"  sessions present: {sorted(bl.session.unique())}")
    say(f"  rows per session: {bl.groupby('session').size().to_dict()}")
say(f"  subject id examples: {sorted(bl.subject.unique())[:8]}")

if "session" in fl and "session" in bl:
    fs, bs = set(fl.session.unique()), set(bl.session.unique())
    say(f"\nsession labels — features: {sorted(fs)}   behavioural: {sorted(bs)}")
    say(f"  overlap: {sorted(fs & bs)}")
fsub, bsub = set(fl.subject.unique()), set(bl.subject.unique())
say(f"subject overlap: {len(fsub & bsub)}   features-only: {len(fsub - bsub)}   "
    f"behav-only: {len(bsub - fsub)}")
if fsub - bsub:
    say(f"  features-only examples: {sorted(fsub - bsub)[:8]}")
if bsub - fsub:
    say(f"  behav-only examples: {sorted(bsub - fsub)[:8]}")

say("\nDECISION RULE:")
say("  - Coverage gap inside the feature file  -> re-extract Lee features, or")
say("    exclude Lee from the hierarchical table and say so.")
say("  - Merge-key mismatch                    -> fix the key; n returns to 54.")
say("  Either way, do NOT report the Lee Ahn increment (dR2 = 0.178) as n=54,")
say("  and do not impute the missing slopes.")

# ======================================================================
# PART 2 — Model B without LOOCV bias
# ======================================================================
say("\n" + "=" * 74)
say("PART 2 — Model B, three estimators")
say("=" * 74)

KEY = [k for k in ["dataset", "subject", "session"] if k in behav and k in smr]
M = behav.merge(smr[KEY + ["smr_snr_mean"]], on=KEY, how="left")
M = M.merge(ahn[KEY + ["ahn_ratio"]], on=KEY, how="left")
fkey = [k for k in KEY if k in feat.columns]
M = M.merge(feat, on=fkey, how="left", suffixes=("", "_f"))
S = M[M.dataset.str.contains("Liu", case=False)].drop_duplicates(subset=KEY).copy()
S = S.merge(clin, on="subject", how="inner")
say(f"stroke rows: {len(S)}   NIHSS median: {S.NIHSS.median()}")
S["target"] = (S.NIHSS > S.NIHSS.median()).astype(int)
say(f"class balance: {S.target.value_counts().to_dict()}")

SEL_T1 = next((c for c in S.columns if "mu_ERD_selectivity_T1" in c), None)
SEL_T2 = next((c for c in S.columns if "mu_ERD_selectivity_T2" in c), None)


def direct_auc(x, y):
    """Univariate AUC — no fitting, no CV, no bias. Sign-agnostic."""
    a = roc_auc_score(y, x)
    return max(a, 1 - a), a


def cv_auc(X, y, cv, C=1.0):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, C=C))])
    pr = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, pr)


def perm_direct(x, y, n=N_PERM):
    return np.array([direct_auc(x, RNG.permutation(y))[0] for _ in range(n)])


def perm_cv(X, y, cv, n=500, C=1.0):
    return np.array([cv_auc(X, RNG.permutation(y), cv, C) for _ in range(n)])


say("\n--- (a) DIRECT univariate AUC (recommended for single features) ---")
for label, col in [("T1 selectivity", SEL_T1), ("T2 selectivity", SEL_T2),
                   ("dLI", "dLI" if "dLI" in S else None)]:
    if not col or col not in S:
        continue
    d = S.dropna(subset=[col, "target"])
    auc, raw = direct_auc(d[col].values, d.target.values)
    null = perm_direct(d[col].values, d.target.values)
    p = (np.sum(null >= auc) + 1) / (len(null) + 1)
    rho, pr_ = stats.spearmanr(d[col], d.NIHSS)
    say(f"  {label:16s} n={len(d):3d}  AUC={auc:.3f} (raw {raw:.3f})  "
        f"null mean={null.mean():.3f} 95th={np.percentile(null,95):.3f}  p={p:.4f}")
    say(f"  {'':16s}      continuous: rho={rho:+.3f} p={pr_:.4f}")

say("\n--- (b) STRATIFIED 5-FOLD CV (multi-feature sets) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SETS = {
    "selectivity_both": [c for c in [SEL_T1, SEL_T2] if c],
    "sel_plus_dLI":     [c for c in [SEL_T1, SEL_T2, "dLI"] if c and c in S],
}
for name, cols in SETS.items():
    d = S.dropna(subset=cols + ["target"])
    if len(d) < 30:
        continue
    auc = cv_auc(d[cols].values, d.target.values, skf)
    null = perm_cv(d[cols].values, d.target.values, skf, n=500)
    p = (np.sum(null >= auc) + 1) / (len(null) + 1)
    say(f"  {name:20s} ({len(cols)}f, n={len(d)})  AUC={auc:.3f}  "
        f"null mean={null.mean():.3f}  p={p:.4f}")

say("\n--- (c) LOOCV, for continuity with the previous run ---")
if SEL_T1:
    d = S.dropna(subset=[SEL_T1, "target"])
    auc = cv_auc(d[[SEL_T1]].values, d.target.values, LeaveOneOut())
    null = perm_cv(d[[SEL_T1]].values, d.target.values, LeaveOneOut(), n=300)
    say(f"  T1 selectivity  AUC={auc:.3f}  null mean={null.mean():.3f}")

say("\n--- SANITY CHECK: pure noise through the LOOCV pipeline ---")
noise = RNG.normal(size=len(S))
auc_n = cv_auc(noise.reshape(-1, 1), S.target.values, LeaveOneOut())
null_n = perm_cv(noise.reshape(-1, 1), S.target.values, LeaveOneOut(), n=300)
say(f"  random feature: AUC={auc_n:.3f}   LOOCV null mean={null_n.mean():.3f}")
say("  If this null is ~0.30, the sub-0.5 centring is confirmed as LOOCV bias")
say("  and can be stated as such in the manuscript.")

say("\n--- lesion-side stratified, DIRECT AUC ---")
if SEL_T1 and "hemiplegia_side" in S:
    for sv, g in S.groupby("hemiplegia_side"):
        g = g.dropna(subset=[SEL_T1, "target"])
        if len(g) < 15 or g.target.nunique() < 2:
            continue
        auc, raw = direct_auc(g[SEL_T1].values, g.target.values)
        null = perm_direct(g[SEL_T1].values, g.target.values, n=2000)
        p = (np.sum(null >= auc) + 1) / (len(null) + 1)
        rho, pr_ = stats.spearmanr(g[SEL_T1], g.NIHSS)
        say(f"  {sv:8s} n={len(g):3d}  AUC={auc:.3f}  null={null.mean():.3f}  "
            f"p={p:.4f}   continuous rho={rho:+.3f} p={pr_:.4f}")

say("\nREPORTING:")
say("  - The CONTINUOUS Spearman correlation stays the primary clinical")
say("    result. The classifier is a secondary framing of the same signal.")
say("  - Report the DIRECT AUC with its permutation null; it needs no")
say("    bias caveat because no model is fitted.")
say("  - n=50 (22 right-hemiplegia) remains under-powered. No screening")
say("    claim follows from these numbers.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
