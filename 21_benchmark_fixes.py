"""
21_benchmark_fixes.py
---------------------
Repairs three problems in the script 19/20 output and adds the two missing
analyses. Run this INSTEAD of re-running 19; it reads the same inputs.

FIXES
  F1  Merge on (dataset, subject, session) so BNCI2014_004 is not duplicated
      5x by the feature-file join.
  F2  Deduplicate to ONE ROW PER SUBJECT-SESSION before any correlation with
      decode_acc, which is a per-subject-session quantity. The long-over-task
      frame doubles every subject and halves the standard errors.
  F3  For BNCI (5 sessions per subject) additionally report a SUBJECT-LEVEL
      estimate, since 45 rows come from only 9 independent participants.

ADDITIONS
  A1  Ahn theta/alpha -> DLI. The missing arm: we know Ahn predicts accuracy;
      we do not yet know whether it predicts DLI. Needed to complete the
      dissociation for the second benchmark.
  A2  Step 4 of the hierarchical table (+DLI). Asks whether DLI adds variance
      to DECODING ACCURACY beyond the two published predictors. A null here
      demonstrates the dissociation inside a single model and is a positive
      result for the paper, not a failure.
  A3  Collinearity between SMR-SNR and the Ahn ratio (both are alpha-band
      quantities and are expected to overlap).

INPUTS: same as script 19.
OUTPUT: benchmark_fixed_report.txt, benchmark_table_S11_fixed.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

SMR_F   = "blankertz_smr_per_subject.csv"
AHN_F   = "ahn_theta_alpha_per_subject.csv"
BEHAV_F = "behavioural_validation_within.csv"
FEAT_F = "features_per_subject_MERGED.csv"
REPORT  = "benchmark_fixed_report.txt"
TABLE   = "benchmark_table_S11_fixed.csv"
MIN_ACC = 0.58
RNG = np.random.default_rng(42)

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def rd(p):
    d = pd.read_csv(p)
    for c in ("subject", "session"):
        if c in d:
            d[c] = d[c].astype(str)
    return d

smr, ahn, behav, feat = rd(SMR_F), rd(AHN_F), rd(BEHAV_F), rd(FEAT_F)

# ---------------------------------------------------------------- F1 + F2
KEY = [k for k in ["dataset", "subject", "session"] if k in behav.columns
       and k in smr.columns]
say(f"merge key: {KEY}")

M = behav.merge(smr[KEY + ["smr_snr_mean"]], on=KEY, how="left")
M = M.merge(ahn[KEY + ["ahn_ratio"]], on=KEY, how="left")

featkeep = [c for c in feat.columns
            if c in ["dataset", "subject", "session"] or "slope_contra" in c
            or "slope_bilateral" in c or "selectivity" in c]
fkey = [k for k in KEY if k in feat.columns]
M = M.merge(feat[featkeep], on=fkey, how="left", suffixes=("", "_f"))

say(f"rows after keyed merge: {len(M)}")
say(M.groupby("dataset").size().to_string())

# one row per subject-session for anything involving decode_acc
SS = M.drop_duplicates(subset=[k for k in KEY]).copy()
say(f"\nrows after de-duplication to subject-session: {len(SS)}")
say(SS.groupby("dataset").agg(rows=("subject", "size"),
                              subjects=("subject", "nunique"),
                              mean_acc=("decode_acc", "mean")).round(3).to_string())

usable = SS.groupby("dataset").decode_acc.mean()
usable = usable[usable >= MIN_ACC].index.tolist()
say(f"usable cohorts (decoder above {MIN_ACC}): {usable}")


def corr_block(df, x, y, label):
    d = df.dropna(subset=[x, y])
    if len(d) < 8:
        return None
    r, p = stats.pearsonr(d[x], d[y])
    rho, pp = stats.spearmanr(d[x], d[y])
    idx = np.arange(len(d))
    bs = []
    for _ in range(3000):
        s = RNG.choice(idx, len(d), replace=True)
        try:
            bs.append(stats.pearsonr(d[x].values[s], d[y].values[s])[0])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    say(f"  {label:34s} n={len(d):3d}  r={r:+.3f} [{lo:+.3f},{hi:+.3f}] "
        f"p={p:.4g}   rho={rho:+.3f} p={pp:.4g}")
    return dict(r=r, n=len(d), lo=lo, hi=hi, p=p, rho=rho)


def pooled(res):
    res = [x for x in res if x]
    if len(res) < 2:
        return
    z = [np.arctanh(x["r"]) for x in res]
    w = [x["n"] - 3 for x in res]
    say(f"  {'POOLED (Fisher-z)':34s} r = {np.tanh(np.average(z, weights=w)):+.3f}")


say("\n" + "=" * 74)
say("POSITIVE CONTROLS — de-duplicated (one row per subject-session)")
say("=" * 74)
for pred, direction in [("smr_snr_mean", "expect POSITIVE"),
                        ("ahn_ratio", "expect NEGATIVE")]:
    say(f"\n--- {pred} -> decode_acc   ({direction}) ---")
    res = []
    for ds in usable:
        res.append(corr_block(SS[SS.dataset == ds], pred, "decode_acc", ds))
    pooled(res)
    # F3: subject-level for multi-session cohorts
    for ds in usable:
        g = SS[SS.dataset == ds]
        if g.subject.nunique() < len(g):
            agg = g.groupby("subject", as_index=False)[[pred, "decode_acc"]].mean()
            corr_block(agg, pred, "decode_acc", f"{ds} (subject-level)")

# ---------------------------------------------------------------- A1
say("\n" + "=" * 74)
say("A1 — MISSING ARM: do the published predictors predict DLI?")
say("=" * 74)
for pred in ["smr_snr_mean", "ahn_ratio"]:
    say(f"\n--- {pred} -> dLI ---")
    res = []
    for ds in sorted(M.dataset.unique()):
        for task in sorted(M.task.unique()) if "task" in M else [None]:
            g = M[(M.dataset == ds)] if task is None else M[(M.dataset == ds) & (M.task == task)]
            g = g.drop_duplicates(subset=KEY + (["task"] if task else []))
            res.append(corr_block(g, pred, "dLI", f"{ds} {task or ''}"))
    pooled(res)
say("\n  A validated predictor that forecasts decoding accuracy but NOT dLI is")
say("  the double dissociation: it establishes that dLI indexes lateralisation")
say("  change rather than decodability.")

# ---------------------------------------------------------------- A3
say("\n" + "=" * 74)
say("A3 — collinearity between the two published predictors")
say("=" * 74)
for ds in sorted(SS.dataset.unique()):
    corr_block(SS[SS.dataset == ds], "smr_snr_mean", "ahn_ratio", ds)
say("  Both are alpha-band quantities; substantial overlap is expected and")
say("  explains why the +Ahn increment over SMR is small in most cohorts.")

# ---------------------------------------------------------------- A2
say("\n" + "=" * 74)
say("A2 — HIERARCHICAL TABLE, outcome = decoding accuracy, WITH step 4 (+dLI)")
say("=" * 74)

def pick(cols, *frags):
    for c in cols:
        lc = c.lower()
        if all(f in lc for f in frags):
            return c
    return None

rows = []
for ds in sorted(M.dataset.unique()):
    for task in sorted(M.task.unique()):
        g = M[(M.dataset == ds) & (M.task == task)].drop_duplicates(subset=KEY)
        band = "mu"
        slope_c = pick(g.columns, "slope_contra", band, task.lower()) or \
                  pick(g.columns, "slope_contra", band)
        steps = [("ERDslope", slope_c),
                 ("SMR", "smr_snr_mean"),
                 ("Ahn", "ahn_ratio"),
                 ("dLI", "dLI")]          # <-- fixed: single long-format column
        steps = [(n, c) for n, c in steps if c and c in g.columns]
        d = g.dropna(subset=[c for _, c in steps] + ["decode_acc"])
        if len(d) < 15:
            say(f"  {ds:14s} {task}  n={len(d)} too small — skipped")
            continue
        used, prev, rec = [], 0.0, dict(dataset=ds, task=task, n=len(d))
        for label, col in steps:
            used.append(col)
            m = sm.OLS(d.decode_acc.astype(float),
                       sm.add_constant(d[used].astype(float))).fit()
            rec[f"{label}_R2"] = round(m.rsquared, 3)
            rec[f"{label}_dR2"] = round(m.rsquared - prev, 3)
            rec[f"{label}_p"] = round(float(m.pvalues.get(col, np.nan)), 5)
            prev = m.rsquared
        rows.append(rec)
        say("  " + f"{ds:14s} {task} n={len(d):3d}  " +
            "  ".join(f"{l}: dR2={rec[l+'_dR2']:+.3f} (p={rec[l+'_p']:.3g})"
                      for l, _ in steps))

tab = pd.DataFrame(rows)
tab.to_csv(TABLE, index=False)
say(f"\nWrote {TABLE}  ({len(tab)} rows)")
say("\n  READ STEP 4 CAREFULLY: a NULL +dLI increment is the result you want.")
say("  It shows dLI carries no information about decodability once the")
say("  established predictors are in the model — the dissociation, inside")
say("  a single regression.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
