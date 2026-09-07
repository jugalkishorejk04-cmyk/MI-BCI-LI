"""
29b_denominator_from_perstage.py
--------------------------------
Fixes two reconstruction bugs in script 29, then runs the level test.

BUG 1 - CARTESIAN MERGE
  features_per_subject.csv has 274 rows: one per subject-SESSION
  (108 + 52 + 19 + 50 + 45). Script 29 merged on dataset+subject only, so
  each BNCI subject's 5 sessions on the left matched all 5 on the right:
  9 x 25 = 225 rows instead of 45. Total 108+52+19+50+225 = 454, which is
  exactly the n the gate reported. Most BNCI rows were pairing one
  session's selectivity against another session's. Fixed by including
  session in the key.

BUG 2 - TASK MAPPING UNVERIFIED
  Script 29 assumed T1 = left_hand. If the suffix maps the other way the
  comparison is across tasks. This version tests both and reports which
  matches.

Everything else is unchanged: the per-stage values come from the
pipeline's own features_per_epoch.csv, so no re-extraction is involved.

NOTE: features_per_epoch.csv holds Lee at 19 subjects (114 rows), the
partial extraction. That does not affect the Liu clinical test but does
mean Lee appears at n=19 in the cohort table.

OUTPUT: denominator_final_report_v2.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

REPORT = "denominator_final_report_v2.txt"
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
F = rd("features_per_subject.csv")
say(f"features_per_epoch.csv: {len(E)} rows")
say(f"features_per_subject.csv: {len(F)} rows   "
    f"session column present: {'session' in F.columns}")
if "session" in F:
    say(F.groupby('dataset').agg(rows=('subject','size'),
                                 subjects=('subject','nunique')).to_string())


def build(t1_is_left):
    """Reconstruct per-stage D, N, LI, selectivity and dLI."""
    rows = []
    for (ds, sub, sess, task), g in E.groupby(["dataset", "subject", "session", "task"]):
        left = "left" in str(task).lower()
        cc, ic = ("mu_ERD_C4", "mu_ERD_C3") if left else ("mu_ERD_C3", "mu_ERD_C4")
        rec = dict(dataset=ds, subject=sub, session=sess, task=task,
                   task_slot=("T1" if left == t1_is_left else "T2"))
        ok = True
        for st in ["early", "mid", "late"]:
            r = g[g.stage == st]
            if len(r) != 1:
                ok = False; break
            c, ip = abs(float(r[cc].iloc[0])), abs(float(r[ic].iloc[0]))
            if not (np.isfinite(c) and np.isfinite(ip)) or (c + ip) == 0:
                ok = False; break
            rec[f"c_{st}"], rec[f"D_{st}"] = c, c + ip
            rec[f"LI_{st}"] = (c - ip) / (c + ip)
        if not ok:
            continue
        x = np.array([0.0, 1.0, 2.0])
        yc = np.array([rec["c_early"], rec["c_mid"], rec["c_late"]])
        yb = np.array([rec["D_early"], rec["D_mid"], rec["D_late"]]) / 2.0
        rec["sel_derived"] = stats.linregress(x, yc)[0] - stats.linregress(x, yb)[0]
        rec["dLI_derived"] = rec["LI_late"] - rec["LI_early"]
        rec["D_mean"] = np.mean([rec["D_early"], rec["D_mid"], rec["D_late"]])
        rec["dD"] = rec["D_late"] - rec["D_early"]
        rows.append(rec)
    return pd.DataFrame(rows)


KEY = [k for k in ["dataset", "subject", "session"] if k in F.columns]
say(f"\nmerge key: {KEY}")
selc = next((c for c in F.columns if "mu_ERD_selectivity_T1" in c), None)

say("\n" + "=" * 70)
say("IDENTITY GATE — both task mappings")
say("=" * 70)
best, best_r, best_P = None, -1, None
for t1_left in (True, False):
    P = build(t1_left)
    j = (P[P.task_slot == "T1"]
         .merge(F[KEY + [selc]], on=KEY, how="inner")
         .dropna(subset=["sel_derived", selc]))
    if len(j) < 30:
        say(f"  T1 = {'left_hand' if t1_left else 'right_hand'}: only {len(j)} matched rows")
        continue
    r, _ = stats.pearsonr(j.sel_derived, j[selc])
    say(f"  T1 = {'left_hand ' if t1_left else 'right_hand'}   n={len(j):3d}  r={r:+.5f}")
    if r > best_r:
        best, best_r, best_P = t1_left, r, P

GATE = best_r > 0.95
say(f"\n  best: T1 = {'left_hand' if best else 'right_hand'}  r = {best_r:+.5f}")
say("  GATE PASSED — reconstruction matches the pipeline." if GATE else
    "  GATE FAILED — the remaining difference is elsewhere. Compare\n"
    "  compute_subject_summary() lines 297-345 against build() here,\n"
    "  paying attention to the x values passed to linregress and to any\n"
    "  filtering applied before the stage loop.")

if GATE:
    P = best_P
    clin = rd("liu_clinical.csv")
    G = P[P.task_slot == "T1"].drop_duplicates(subset=["dataset", "subject", "session"])

    say("\n" + "=" * 70)
    say("D_mean by cohort")
    say("=" * 70)
    T = G.groupby("dataset").D_mean.agg(["count", "mean", "std"])
    T["CV"] = T["std"] / T["mean"]
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

    if "hemiplegia_side" in S:
        say("\n  by lesion side:")
        for sv, g in S.groupby("hemiplegia_side"):
            g = g.dropna(subset=["sel_derived", "dLI_derived", "NIHSS", "D_mean"])
            if len(g) < 15:
                continue
            rs, ps = stats.spearmanr(g.sel_derived, g.NIHSS)
            rl, pl = stats.spearmanr(g.dLI_derived, g.NIHSS)
            rD, pD = stats.spearmanr(g.D_mean, g.NIHSS)
            say(f"    {sv:8s} n={len(g):3d}  sel rho={rs:+.3f} (p={ps:.4f})   "
                f"dLI rho={rl:+.3f} (p={pl:.4f})   D_mean rho={rD:+.3f} (p={pD:.4f})")

    say("\n  DECISION (pre-committed):")
    say("   dLI strengthens materially while selectivity is unchanged")
    say("     -> amplitude LEVEL attenuates dLI; report the mechanism.")
    say("   otherwise -> estimator void, drift null, level null. No mechanism")
    say("     remains: report the sensitivity difference as an observation,")
    say("     and note that two candidate explanations were tested and")
    say("     excluded.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
