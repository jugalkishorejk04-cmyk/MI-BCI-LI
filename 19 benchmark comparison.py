"""
19_benchmark_comparison.py
--------------------------
Items 3 and 4 of the benchmark plan.

(3) POSITIVE CONTROLS — validate both published comparators against the
    outcome they were designed for (decoding accuracy):
        Blankertz SMR-SNR  -> decoding accuracy   (expect POSITIVE, r ~ 0.5)
        Ahn theta/alpha    -> decoding accuracy   (expect NEGATIVE)
    If these reproduce, the null against DLI becomes a DOUBLE DISSOCIATION
    rather than a possibly-attenuated null.

(4) COMPARATOR TABLE — hierarchical regression with an INDEPENDENT outcome:
        Step 1: contralateral ERD slope
        Step 2: + Blankertz SMR-SNR
        Step 3: + Ahn theta/alpha
        Step 4: + DLI  (does our measure add anything to decodability?)
    Outcome = decoding accuracy. No circularity: accuracy is computed on the
    full montage and is not derived from the C3/C4 ERD trajectory.

    Step 4 is informative either way. If DLI adds nothing, that SUPPORTS the
    dissociation claim (DLI indexes lateralisation, not decodability). If it
    adds variance, that is a positive finding.

INPUTS (edit paths to match your working directory)
  blankertz_smr_per_subject.csv      : dataset, subject, session, smr_snr_mean
  ahn_theta_alpha_per_subject.csv    : dataset, subject, session, ahn_ratio
  behavioural_validation_within.csv  : dataset, subject, decode_acc, dLI_*, sel_*
  <features file>                    : contralateral ERD slope per subject/task

OUTPUT
  benchmark_comparison_report.txt
  benchmark_comparison_table.csv     (Supplementary Table S11)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import itertools
import sys

SMR_F   = "blankertz_smr_per_subject.csv"
AHN_F   = "ahn_theta_alpha_per_subject.csv"
BEHAV_F = "behavioural_validation_within.csv"
FEAT_F  = "features_per_subject.csv"          # <-- your main feature file

REPORT  = "benchmark_comparison_report.txt"
TABLE   = "benchmark_comparison_table.csv"

# Exclude cohorts whose decoder is at/near chance — their accuracy is noise
# and cannot serve as a validation target. Liu is expected to fail this.
MIN_MEAN_ACC = 0.58

out = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.append(line)


def load(path, what):
    try:
        return pd.read_csv(path)
    except Exception as e:
        say(f"!! could not read {path} ({what}): {e}")
        return None


smr   = load(SMR_F, "Blankertz SMR")
ahn   = load(AHN_F, "Ahn theta/alpha")
behav = load(BEHAV_F, "decoding + DLI")
feat  = load(FEAT_F, "ERD slopes")

if behav is None or smr is None:
    say("Cannot proceed without decoding accuracy and SMR files.")
    sys.exit(1)

key = ["dataset", "subject"]
for df in (smr, ahn, behav, feat):
    if df is not None and "subject" in df:
        df["subject"] = df["subject"].astype(str)

# collapse SMR / Ahn to one row per subject (mean across sessions)
def collapse(df, col):
    if df is None or col not in df:
        return None
    return df.groupby(key, as_index=False)[col].mean()

smr_s = collapse(smr, "smr_snr_mean")
ahn_s = collapse(ahn, "ahn_ratio")

M = behav.copy()
if smr_s is not None:
    M = M.merge(smr_s, on=key, how="left")
if ahn_s is not None:
    M = M.merge(ahn_s, on=key, how="left")
if feat is not None:
    keep = [c for c in feat.columns if c in key or "slope_contra" in c or "dLI" in c or "select" in c]
    M = M.merge(feat[keep], on=key, how="left", suffixes=("", "_feat"))

say("=" * 72)
say("MERGED DATA")
say("=" * 72)
say(f"rows: {len(M)}   columns: {list(M.columns)}")
say("")

acc_col = next((c for c in M.columns if "acc" in c.lower()), None)
if acc_col is None:
    say("!! no decoding-accuracy column found — cannot run positive controls.")
    sys.exit(1)
say(f"using accuracy column: {acc_col}")

say("")
say("Mean decoding accuracy by dataset (cohorts below "
    f"{MIN_MEAN_ACC} excluded from validation):")
acc_by = M.groupby("dataset")[acc_col].agg(["size", "mean"]).round(3)
say(acc_by.to_string())
usable = acc_by[acc_by["mean"] >= MIN_MEAN_ACC].index.tolist()
say(f"usable cohorts: {usable}")

# ----------------------------------------------------------------------
# (3) POSITIVE CONTROLS
# ----------------------------------------------------------------------
say("")
say("=" * 72)
say("(3) POSITIVE CONTROLS — comparator vs the outcome it was designed for")
say("=" * 72)

controls = []
for pred, expect in [("smr_snr_mean", "positive"), ("ahn_ratio", "negative")]:
    if pred not in M.columns:
        say(f"\n{pred}: column absent, skipped")
        continue
    say(f"\n--- {pred}  (expected direction: {expect}) ---")
    for ds in usable:
        g = M[(M.dataset == ds)].dropna(subset=[pred, acc_col])
        if len(g) < 10:
            say(f"  {ds:14s} n={len(g)} too small")
            continue
        r, p = stats.pearsonr(g[pred], g[acc_col])
        rho, pp = stats.spearmanr(g[pred], g[acc_col])
        # bootstrap CI
        bs = [stats.pearsonr(*zip(*[(g[pred].values[i], g[acc_col].values[i])
              for i in np.random.randint(0, len(g), len(g))]))[0]
              for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        say(f"  {ds:14s} n={len(g):3d}  r={r:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}"
            f"   rho={rho:+.3f} p={pp:.4f}")
        controls.append(dict(comparator=pred, dataset=ds, n=len(g),
                             r=r, ci_lo=lo, ci_hi=hi, p=p, rho=rho))

    # Fisher-z random effects across usable cohorts
    sub = [c for c in controls if c["comparator"] == pred]
    if len(sub) >= 2:
        z = [np.arctanh(c["r"]) for c in sub]
        w = [c["n"] - 3 for c in sub]
        zbar = np.average(z, weights=w)
        say(f"  POOLED (Fisher-z, random effects): r = {np.tanh(zbar):+.3f}")

say("")
say("INTERPRETATION GATE:")
say("  If SMR->accuracy reproduces (r ~ +0.3 to +0.6) in at least one cohort,")
say("  the implementation is validated and the SMR->DLI null becomes a")
say("  DOUBLE DISSOCIATION. If it does not reproduce, the DLI null stays")
say("  caveated as possible attenuation and must be reported that way.")

# ----------------------------------------------------------------------
# (4) HIERARCHICAL COMPARATOR TABLE — outcome = decoding accuracy
# ----------------------------------------------------------------------
say("")
say("=" * 72)
say("(4) HIERARCHICAL REGRESSION — outcome: decoding accuracy (independent)")
say("=" * 72)

def find(cols, *frags):
    for c in cols:
        lc = c.lower()
        if all(f in lc for f in frags):
            return c
    return None

rows = []
for ds in M.dataset.unique():
    g = M[M.dataset == ds].copy()
    for task in ["T1", "T2"]:
        slope_c = find(g.columns, "slope_contra", task.lower()) or find(g.columns, "slope_contra")
        dli_c   = find(g.columns, "dli", task.lower())
        preds = [("ERD slope", slope_c),
                 ("+SMR", "smr_snr_mean" if "smr_snr_mean" in g else None),
                 ("+Ahn", "ahn_ratio" if "ahn_ratio" in g else None),
                 ("+DLI", dli_c)]
        preds = [(n, c) for n, c in preds if c is not None and c in g]
        if not preds:
            continue
        cols = [c for _, c in preds] + [acc_col]
        d = g.dropna(subset=cols)
        if len(d) < 15:
            continue

        prev_r2, step_out = 0.0, {}
        used = []
        for label, col in preds:
            used.append(col)
            X = sm.add_constant(d[used].astype(float))
            m = sm.OLS(d[acc_col].astype(float), X).fit()
            step_out[label] = dict(r2=m.rsquared, dr2=m.rsquared - prev_r2)
            prev_r2 = m.rsquared
        # F-change for the final step
        rows.append(dict(dataset=ds, task=task, n=len(d),
                         **{f"{k}_R2": round(v["r2"], 3) for k, v in step_out.items()},
                         **{f"{k}_dR2": round(v["dr2"], 3) for k, v in step_out.items()}))
        say(f"  {ds:14s} {task}  n={len(d):3d}  " +
            "  ".join(f"{k}: R2={v['r2']:.3f} (dR2={v['dr2']:+.3f})"
                      for k, v in step_out.items()))

tab = pd.DataFrame(rows)
tab.to_csv(TABLE, index=False)
say("")
say(f"Wrote {TABLE}  ({len(tab)} rows) -> Supplementary Table S11")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
