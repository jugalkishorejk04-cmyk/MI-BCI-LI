"""
25_denominator_test.py
----------------------
Tests the mechanistic account of why ERD selectivity tracks NIHSS
(rho = -0.436, p = 0.002) while dLI does not (rho = -0.234, p = 0.103),
despite the two being near-identical in healthy cohorts (rho = 0.99).

THE ACCOUNT
  selectivity = unnormalised differential slope, proportional to the
                slope of (|ERD_contra| - |ERD_ipsi|)
  dLI         = the same numerator divided by D = (|ERD_contra| + |ERD_ipsi|)

  In healthy brains D is comparatively stable across subjects, so
  normalisation costs little and the two measures track each other.
  In stroke, total ERD amplitude is itself pathological - reduced,
  asymmetric, lesion-dependent - so dividing by D injects that pathology
  into the outcome and attenuates the lateralisation signal.

TWO PREDICTIONS
  P1  D is MORE VARIABLE in stroke than in healthy cohorts
      (higher coefficient of variation in Liu 2024)
  P2  D is RELATED TO SEVERITY in stroke
      (D correlates with NIHSS)

  If both hold, the account is demonstrated. If P1 holds but P2 does not,
  the denominator is noisier in stroke but not severity-linked - the
  attenuation story still works, but the mechanism is dilution rather than
  contamination. If neither holds, the account is not supported and the
  divergence needs a different explanation (or should be reported without
  one).

TWO ROUTES TO D
  DIRECT   if features_per_subject.csv contains the stage-wise ERD
           magnitudes (|contra|, |ipsi| per stage), D is computed exactly.
  PROXY    otherwise, note that dLI / selectivity is proportional to 1/D.
           The per-subject ratio therefore gives an inverse proxy for D
           using data you already have, with no re-extraction. Subjects
           with near-zero selectivity are excluded, since the ratio is
           unstable there.

OUTPUT: denominator_test_report.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

REPORT = "denominator_test_report.txt"
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
clin  = rd("liu_clinical.csv")

KEY = [k for k in ["dataset", "subject", "session"] if k in behav]
fkey = [k for k in KEY if k in feat.columns]
M = behav.merge(feat, on=fkey, how="left", suffixes=("", "_f"))
M = M.drop_duplicates(subset=KEY + (["task"] if "task" in M else []))

# ---------------------------------------------------------------- route
say("=" * 72)
say("LOCATING THE DENOMINATOR D = |ERD_contra| + |ERD_ipsi|")
say("=" * 72)

mag_cols = [c for c in M.columns
            if ("erd" in c.lower())
            and any(k in c.lower() for k in ["contra", "ipsi"])
            and "slope" not in c.lower() and "select" not in c.lower()]
say(f"candidate raw-magnitude columns: {mag_cols if mag_cols else 'none found'}")

route = None
if len(mag_cols) >= 2:
    con = [c for c in mag_cols if "contra" in c.lower()]
    ips = [c for c in mag_cols if "ipsi" in c.lower()]
    if con and ips:
        M["D"] = M[con].abs().mean(axis=1) + M[ips].abs().mean(axis=1)
        route = "DIRECT"
        say("route: DIRECT — D computed from raw ERD magnitudes")

if route is None:
    say("route: PROXY — raw magnitudes not in the feature file.")
    say("  dLI / selectivity is proportional to 1/D, so 1/ratio is used as")
    say("  an inverse-free proxy for D. Subjects with |selectivity| below the")
    say("  10th percentile of their cohort are dropped (unstable ratio).")
    need = [c for c in ["dLI", "selectivity"] if c in M.columns]
    if len(need) < 2:
        say("!! neither route available — dLI and selectivity columns missing.")
        raise SystemExit(1)
    M = M[M.selectivity.notna() & M.dLI.notna()].copy()
    thr = M.groupby("dataset").selectivity.transform(
        lambda s: s.abs().quantile(0.10))
    n0 = len(M)
    M = M[M.selectivity.abs() > thr].copy()
    say(f"  dropped {n0 - len(M)} unstable-ratio rows; {len(M)} remain")
    M["ratio"] = M.dLI / M.selectivity
    # keep physically sensible ratios only
    M = M[(M.ratio > 0.05) & (M.ratio < 5.0)]
    M["D"] = 1.0 / M.ratio          # proportional to D, arbitrary units
    route = "PROXY"

# ---------------------------------------------------------------- P1
say("\n" + "=" * 72)
say("P1 — is D more variable in stroke than in healthy cohorts?")
say("=" * 72)
say(f"(route = {route}; with PROXY, D is in arbitrary units, so compare CV")
say(" and relative spread, not absolute values)")

rows = []
for ds, g in M.groupby("dataset"):
    d = g.D.dropna()
    if len(d) < 15:
        continue
    cv = d.std() / abs(d.mean()) if d.mean() != 0 else np.nan
    # robust alternative, insensitive to outliers
    rcv = stats.median_abs_deviation(d, scale="normal") / abs(np.median(d))
    rows.append(dict(dataset=ds, n=len(d), mean=d.mean(), sd=d.std(),
                     CV=cv, robust_CV=rcv,
                     IQR_over_median=(d.quantile(.75) - d.quantile(.25)) / abs(d.median())))
T = pd.DataFrame(rows)
say("")
say(T.round(3).to_string(index=False))

if len(T) > 1 and T.dataset.str.contains("Liu", case=False).any():
    liu = T[T.dataset.str.contains("Liu", case=False)].iloc[0]
    hea = T[~T.dataset.str.contains("Liu", case=False)]
    say(f"\nLiu 2024 CV        = {liu.CV:.3f}   robust CV = {liu.robust_CV:.3f}")
    say(f"healthy CV (mean)  = {hea.CV.mean():.3f}   range "
        f"{hea.CV.min():.3f}–{hea.CV.max():.3f}")
    say(f"healthy robust CV  = {hea.robust_CV.mean():.3f}")
    verdict = "SUPPORTED" if liu.CV > hea.CV.max() else (
              "PARTIAL" if liu.CV > hea.CV.mean() else "NOT SUPPORTED")
    say(f"\nP1 verdict: {verdict}")
    say("  SUPPORTED requires Liu's CV to exceed every healthy cohort's.")

    # formal test of variance heterogeneity
    groups = [g.D.dropna().values for _, g in M.groupby("dataset") if len(g) >= 15]
    if len(groups) > 1:
        W, pw = stats.levene(*groups, center="median")
        say(f"  Levene (Brown-Forsythe) across cohorts: W={W:.3f} p={pw:.4g}")

# ---------------------------------------------------------------- P2
say("\n" + "=" * 72)
say("P2 — is D related to stroke severity?")
say("=" * 72)

S = M[M.dataset.str.contains("Liu", case=False)].copy()
S = S.merge(clin, on="subject", how="inner")
say(f"stroke rows merged: {len(S)}")

if len(S) < 20:
    say("!! merge failed — check subject id formatting.")
else:
    for lab, g in [("all", S)] + list(S.groupby("hemiplegia_side")):
        g = g.dropna(subset=["D", "NIHSS"])
        if len(g) < 15:
            say(f"  {lab}: n={len(g)} too small")
            continue
        rho, p = stats.spearmanr(g.D, g.NIHSS)
        r, pp = stats.pearsonr(g.D, g.NIHSS)
        say(f"  {str(lab):8s} n={len(g):3d}  D vs NIHSS: rho={rho:+.3f} p={p:.4f}"
            f"   r={r:+.3f} p={pp:.4f}")

    # the comparison that matters: does D explain the selectivity/dLI gap?
    sel_c = next((c for c in S.columns if "mu_ERD_selectivity_T1" in c), None)
    if sel_c and "dLI" in S:
        g = S.dropna(subset=[sel_c, "dLI", "D", "NIHSS"])
        if len(g) >= 20:
            say("\n  Partial correlations with NIHSS, controlling for D:")
            for name, col in [("selectivity", sel_c), ("dLI", "dLI")]:
                # residualise both on D, then correlate
                res_x = g[col] - np.poly1d(np.polyfit(g.D, g[col], 1))(g.D)
                res_y = g.NIHSS - np.poly1d(np.polyfit(g.D, g.NIHSS, 1))(g.D)
                rho, p = stats.spearmanr(res_x, res_y)
                raw, praw = stats.spearmanr(g[col], g.NIHSS)
                say(f"    {name:12s} raw rho={raw:+.3f} (p={praw:.4f})  ->  "
                    f"partial rho={rho:+.3f} (p={p:.4f})")
            say("    If dLI's association strengthens once D is removed while")
            say("    selectivity's is unchanged, the denominator is what")
            say("    attenuates dLI — the account demonstrated directly.")

say("\n" + "=" * 72)
say("HOW TO REPORT")
say("=" * 72)
say("  P1 and P2 both supported -> state the mechanism in section 3.2 and")
say("    the clinical section; it converts an apparent inconsistency into a")
say("    prediction the data confirm.")
say("  P1 only -> report as dilution: D is noisier in stroke, attenuating")
say("    dLI, without being severity-linked itself.")
say("  Neither -> do NOT assert the mechanism. Report the selectivity/dLI")
say("    divergence as an observation and leave it unexplained.")
if route == "PROXY":
    say("\n  NOTE: this run used the PROXY route. State in the Methods that D")
    say("  was estimated from the dLI/selectivity ratio rather than measured")
    say("  directly, and treat the result as supporting rather than decisive.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
