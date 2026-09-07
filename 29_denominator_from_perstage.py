"""
29_denominator_from_perstage.py
-------------------------------
The denominator test, using the pipeline's own per-stage output.

features_per_epoch.csv already contains, per (dataset, subject, session,
task, stage): mu_ERD_C3, mu_ERD_Cz, mu_ERD_C4, LI_mu. These are the exact
values 03_compute_features.py used to build the summary table, so no
re-extraction is needed and the identity gate passes by construction.

Three prior attempts failed because they reimplemented compute_erd_db and
never matched it (r = 0.20, 0.20, 0.38). This reads the pipeline's numbers
instead.

DEFINITIONS, matching 03_compute_features.py lines 298-320:
    contra = C4 for left_hand (T1), C3 for right_hand (T2)
    c  = |mu_ERD_contra|      ip = |mu_ERD_ipsi|
    D  = c + ip               N  = c - ip
    bilateral = (c + ip)/2 = D/2

WHAT IS ALREADY SETTLED
    estimator hypothesis   mathematically void: with three equally spaced
                           stages, slope = (last - first)/2, so the slope
                           and endpoint-difference forms are the same
                           statistic up to a constant.
    amplitude DRIFT        null: dD = 4 x slope_bilateral, and that column
                           gives rho = +0.078, p = 0.59 against NIHSS.

WHAT THIS TESTS
    amplitude LEVEL (D_mean) - the last remaining candidate for why
    selectivity tracks NIHSS (rho = -0.436) more strongly than dLI
    (rho = -0.264).

OUTPUT: denominator_final_report.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

REPORT = "denominator_final_report.txt"
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


def rd(p):
    d = pd.read_csv(p)
    for c in ("subject", "session"):
        if c in d:
            d[c] = d[c].astype(str)
    return d

E = rd("features_per_epoch.csv")
say(f"features_per_epoch.csv: {len(E)} rows")
say(f"  columns: {list(E.columns)}")
say(f"  stages: {sorted(E.stage.unique())}")
say(f"  tasks:  {sorted(E.task.unique())}")
say(E.groupby('dataset').size().to_string())

# ---------------------------------------------- task-relative assignment
# 03_compute_features.py lines 298-299
def contra_ipsi(task):
    t = str(task).lower()
    if "left" in t or t == "t1":
        return "mu_ERD_C4", "mu_ERD_C3"      # left hand -> right hemisphere
    return "mu_ERD_C3", "mu_ERD_C4"

rows = []
for (ds, sub, sess, task), g in E.groupby(["dataset", "subject", "session", "task"]):
    cc, ic = contra_ipsi(task)
    if cc not in g or ic not in g:
        continue
    rec = dict(dataset=ds, subject=sub, session=sess, task=task)
    ok = True
    for st in ["early", "mid", "late"]:
        r = g[g.stage == st]
        if len(r) != 1:
            ok = False; break
        c, ip = abs(float(r[cc].iloc[0])), abs(float(r[ic].iloc[0]))
        if not (np.isfinite(c) and np.isfinite(ip)) or (c + ip) == 0:
            ok = False; break
        rec[f"c_{st}"], rec[f"i_{st}"] = c, ip
        rec[f"D_{st}"] = c + ip
        rec[f"N_{st}"] = c - ip
        rec[f"LI_{st}"] = (c - ip) / (c + ip)
    if not ok:
        continue
    rec["D_mean"] = np.mean([rec["D_early"], rec["D_mid"], rec["D_late"]])
    rec["dD"] = rec["D_late"] - rec["D_early"]
    x = np.array([0.0, 1.0, 2.0])
    yc = np.array([rec["c_early"], rec["c_mid"], rec["c_late"]])
    yb = np.array([rec["D_early"], rec["D_mid"], rec["D_late"]]) / 2.0
    rec["slope_contra"] = stats.linregress(x, yc)[0]
    rec["slope_bilateral"] = stats.linregress(x, yb)[0]
    rec["sel_derived"] = rec["slope_contra"] - rec["slope_bilateral"]
    rec["dLI_derived"] = rec["LI_late"] - rec["LI_early"]
    rows.append(rec)

P = pd.DataFrame(rows)
say(f"\nreconstructed: {len(P)} subject-session-task rows")

# ------------------------------------------------------------ identity gate
say("\n" + "=" * 70)
say("IDENTITY GATE")
say("=" * 70)
F = rd("features_per_subject.csv")
selc = next((c for c in F.columns if "mu_ERD_selectivity_T1" in c), None)
T1 = P[P.task.str.lower().str.contains("left|t1")]
j = T1.merge(F[["dataset", "subject", selc]], on=["dataset", "subject"], how="inner")
j = j.dropna(subset=["sel_derived", selc])
GATE = False
if len(j) > 30:
    r, _ = stats.pearsonr(j.sel_derived, j[selc])
    say(f"  sel_derived vs {selc}:  n={len(j)}  r={r:+.5f}")
    GATE = r > 0.95
    say("  GATE PASSED — reconstruction matches the pipeline." if GATE else
        "  GATE FAILED — check the task label mapping in contra_ipsi().")
else:
    say(f"  too few matched rows ({len(j)})")

# ---------------------------------------------------------------- the test
if GATE:
    clin = rd("liu_clinical.csv")
    G = T1.drop_duplicates(subset=["dataset", "subject", "session"])

    say("\n" + "=" * 70)
    say("D_mean by cohort")
    say("=" * 70)
    T = G.groupby("dataset").D_mean.agg(["count", "mean", "std"])
    T["CV"] = (T["std"] / T["mean"]).round(3)
    say(T.round(3).to_string())
    grps = [g.D_mean.dropna().values for _, g in G.groupby("dataset") if len(g) >= 15]
    if len(grps) > 1:
        W, pw = stats.levene(*grps, center="median")
        say(f"  Levene across cohorts: W={W:.3f} p={pw:.4g}")

    S = G[G.dataset.str.contains("Liu", case=False)].merge(clin, on="subject", how="inner")
    say(f"\nstroke merged: {len(S)}  (expect 50)")

    say("\n" + "=" * 70)
    say("THE LEVEL TEST")
    say("=" * 70)
    for lab, col in [("D_mean", "D_mean"), ("dD", "dD")]:
        d = S.dropna(subset=[col, "NIHSS"])
        rho, p = stats.spearmanr(d[col], d.NIHSS)
        say(f"  {lab:8s} vs NIHSS: rho={rho:+.3f} p={p:.4f}  n={len(d)}")

    say("\n  selectivity / dLI vs NIHSS, raw and partialled:")
    for nm, col in [("selectivity", "sel_derived"), ("dLI", "dLI_derived")]:
        d = S.dropna(subset=[col, "NIHSS", "D_mean", "dD"])
        raw, praw = stats.spearmanr(d[col], d.NIHSS)
        line = f"    {nm:12s} raw rho={raw:+.3f} (p={praw:.4f})"
        for ctrl in ["D_mean", "dD"]:
            rx = d[col] - np.poly1d(np.polyfit(d[ctrl], d[col], 1))(d[ctrl])
            ry = d.NIHSS - np.poly1d(np.polyfit(d[ctrl], d.NIHSS, 1))(d[ctrl])
            rr, pp = stats.spearmanr(rx, ry)
            line += f"  | {ctrl}-partial rho={rr:+.3f} (p={pp:.4f})"
        say(line)

    # lesion side, matching section 3.7
    if "hemiplegia_side" in S:
        say("\n  by lesion side:")
        for sv, g in S.groupby("hemiplegia_side"):
            g = g.dropna(subset=["sel_derived", "dLI_derived", "NIHSS", "D_mean"])
            if len(g) < 15:
                continue
            rs, ps = stats.spearmanr(g.sel_derived, g.NIHSS)
            rd_, pd_ = stats.spearmanr(g.dLI_derived, g.NIHSS)
            rD, pD = stats.spearmanr(g.D_mean, g.NIHSS)
            say(f"    {sv:8s} n={len(g):3d}  sel rho={rs:+.3f} (p={ps:.4f})   "
                f"dLI rho={rd_:+.3f} (p={pd_:.4f})   D_mean rho={rD:+.3f} (p={pD:.4f})")

    say("\n  DECISION (pre-committed):")
    say("   dLI strengthens materially while selectivity is unchanged")
    say("     -> amplitude LEVEL attenuates dLI. Report the mechanism.")
    say("   otherwise -> estimator void, drift null, level null. No mechanism")
    say("     remains. Report the sensitivity difference as an observation,")
    say("     and state that two candidate explanations were tested and")
    say("     excluded. That is itself worth reporting.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
