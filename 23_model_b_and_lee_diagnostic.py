"""
23_model_b_and_lee_diagnostic.py
--------------------------------
Replaces script 20's Model B with a computationally tractable version, and
diagnoses the Lee n=19 drop-out in the hierarchical table.

WHY THE OLD MODEL B WAS TOO SLOW
  It refitted the full LOOCV loop inside every permutation: 1000 permutations
  x 50 LOO folds = 50,000 model fits, with a GradientBoosting model among
  them. This version permutes the LABELS ONCE per permutation and reuses a
  single LOOCV pass per permutation with a cheap regularised linear model:
  200 permutations x 50 folds of L2 logistic regression finishes in seconds.

WHY THE MODEL IS DELIBERATELY SMALL
  n = 50 with ~14 features will overfit under LOOCV and produce an AUC that
  is mostly noise. Three nested feature sets are therefore run, smallest
  first, and every AUC is judged against its own permutation null rather
  than against 0.5.

PART 1  Lee coverage diagnostic — why does the hierarchical table drop to 19?
PART 2  Model B — within-stroke prediction of NIHSS band from dLI-derived
        features, with lesion-side stratification matching section 3.7.

INPUTS
  features_per_subject.csv, behavioural_validation_within.csv,
  blankertz_smr_per_subject.csv, ahn_theta_alpha_per_subject.csv,
  liu_clinical.csv   (subject, NIHSS, hemiplegia_side)

OUTPUT: model_b_report.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
N_PERM = 500
REPORT = "model_b_report.txt"

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

KEY = [k for k in ["dataset", "subject", "session"] if k in behav and k in smr]
M = behav.merge(smr[KEY + ["smr_snr_mean"]], on=KEY, how="left")
M = M.merge(ahn[KEY + ["ahn_ratio"]], on=KEY, how="left")
fkey = [k for k in KEY if k in feat.columns]
M = M.merge(feat, on=fkey, how="left", suffixes=("", "_f"))

# ======================================================================
# PART 1 — Lee coverage diagnostic
# ======================================================================
say("=" * 74)
say("PART 1 — why does Lee drop from 54 to 19 in the hierarchical table?")
say("=" * 74)

L = M[M.dataset.str.contains("Lee", case=False)].drop_duplicates(subset=KEY)
say(f"Lee rows after keyed merge and de-duplication: {len(L)}")

cand = [c for c in L.columns
        if "slope_contra" in c or c in ("smr_snr_mean", "ahn_ratio",
                                        "dLI", "decode_acc")]
say("\nNaN count per candidate regression column (Lee):")
for c in sorted(cand):
    n_na = int(L[c].isna().sum())
    say(f"  {c:34s} NaN={n_na:3d} / {len(L)}")

mu_slope = next((c for c in L.columns if "mu_ERD_slope_contra" in c), None)
core = [x for x in [mu_slope, "smr_snr_mean", "ahn_ratio", "dLI", "decode_acc"] if x]
say(f"\ncomplete cases across {core}: {int(L.dropna(subset=core).shape[0])}")

say("\nDropping one column at a time to find the culprit:")
for c in core:
    rest = [x for x in core if x != c]
    say(f"  without {c:34s} -> n = {int(L.dropna(subset=rest).shape[0])}")

if "smr_snr_mean" in L:
    say(f"\nLee SMR-SNR <= 1 (no detectable peak): "
        f"{int((L.smr_snr_mean <= 1).sum())} of {len(L)}")
    say(f"Lee SMR-SNR NaN: {int(L.smr_snr_mean.isna().sum())}")
say("\nIf SMR-SNR is the culprit, the fix is to FLOOR at 0 with a "
    "'no_peak_detected' flag rather than writing NaN, and to report how many "
    "subjects use the floor. Do not present the Lee Ahn increment as n=54.")

# ======================================================================
# PART 2 — Model B
# ======================================================================
say("\n" + "=" * 74)
say("PART 2 — Model B: within-stroke prediction of NIHSS band")
say("=" * 74)

S = M[M.dataset.str.contains("Liu", case=False)].drop_duplicates(subset=KEY).copy()
S["subject"] = S.subject.astype(str)
clin["subject"] = clin.subject.astype(str)
S = S.merge(clin, on="subject", how="inner")
say(f"merged stroke rows: {len(S)}  (expect 50)")
if len(S) < 40:
    say("!! subject IDs did not match — check whether the feature file uses "
        "zero-padded IDs ('01') and liu_clinical.csv uses '1'.")

nih = "NIHSS"
S["target"] = (S[nih] > S[nih].median()).astype(int)
say(f"NIHSS median = {S[nih].median()}   class balance = "
    f"{S.target.value_counts().to_dict()}")

SEL_T1 = next((c for c in S.columns if "mu_ERD_selectivity_T1" in c), None)
SEL_T2 = next((c for c in S.columns if "mu_ERD_selectivity_T2" in c), None)
SETS = {
    "S1_selectivity_T1_only":  [c for c in [SEL_T1] if c],
    "S2_selectivity_both":     [c for c in [SEL_T1, SEL_T2] if c],
    "S3_mu_dLI_selectivity":   [c for c in [SEL_T1, SEL_T2, "dLI"] if c],
    "S4_all_dLI_derived":      [c for c in S.columns
                                if any(k in c for k in
                                       ["ERD_slope_contra", "ERD_slope_bilateral",
                                        "ERD_selectivity", "dLI", "selectivity"])],
}

def loocv_auc(X, y, C=1.0):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, C=C))])
    pr = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return roc_auc_score(y, pr), pipe

def perm_null(X, y, n=N_PERM, C=1.0):
    aucs = np.empty(n)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, C=C))])
    for i in range(n):
        yp = RNG.permutation(y)
        pr = cross_val_predict(pipe, X, yp, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        aucs[i] = roc_auc_score(yp, pr)
    return aucs

for name, cols in SETS.items():
    cols = [c for c in cols if c in S.columns]
    if not cols:
        continue
    d = S.dropna(subset=cols + ["target"])
    if len(d) < 30 or d.target.nunique() < 2:
        say(f"\n{name}: insufficient data (n={len(d)})")
        continue
    X, y = d[cols].values, d.target.values
    C = 0.3 if len(cols) > 4 else 1.0        # stronger shrinkage when p is large
    auc, _ = loocv_auc(X, y, C)
    null = perm_null(X, y, C=C)
    p = (np.sum(null >= auc) + 1) / (len(null) + 1)
    say(f"\n{name}  ({len(cols)} features, n={len(d)}, C={C})")
    say(f"  LOOCV AUC = {auc:.3f}")
    say(f"  permutation null: mean {null.mean():.3f}, 95th pct "
        f"{np.percentile(null, 95):.3f}   p = {p:.4f}")
    if len(cols) > len(d) / 10:
        say(f"  NOTE: {len(cols)} features for n={len(d)} — over-parameterised; "
            f"treat as exploratory.")

# lesion-side stratification, matching section 3.7
side = "hemiplegia_side"
if side in S.columns and SEL_T1:
    say("\n--- lesion-side stratified (T1 selectivity only) ---")
    for sv, g in S.groupby(side):
        g = g.dropna(subset=[SEL_T1, "target"])
        if len(g) < 15 or g.target.nunique() < 2:
            say(f"  {sv}: n={len(g)} too small")
            continue
        auc, _ = loocv_auc(g[[SEL_T1]].values, g.target.values)
        null = perm_null(g[[SEL_T1]].values, g.target.values, n=300)
        p = (np.sum(null >= auc) + 1) / (len(null) + 1)
        say(f"  {sv:8s} n={len(g):3d}  AUC={auc:.3f}  null mean={null.mean():.3f}  p={p:.4f}")
        r, pr_ = stats.spearmanr(g[SEL_T1], g[nih])
        say(f"           continuous NIHSS: rho={r:+.3f} p={pr_:.4f}")

say("\nREPORTING RULES:")
say("  - Judge every AUC against its permutation null, not 0.5.")
say("  - n=50 is under-powered. Report with the null and do NOT present any")
say("    model here as a screening tool.")
say("  - The continuous Spearman correlation is the primary clinical result")
say("    (section 3.7); the classifier is a secondary framing of it.")
say("  - If no set clears its null, report as a negative finding: dLI-derived")
say("    features do not classify NIHSS band at this sample size.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
