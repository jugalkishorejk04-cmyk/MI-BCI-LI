"""
26_perstage_erd_denominator.py
------------------------------
Settles why ERD selectivity tracks NIHSS while dLI does not.

WHY THE EARLIER TESTS DID NOT SETTLE IT

  1. The "estimator vs normalisation" decomposition is mathematically void.
     With three equally spaced stages the OLS slope equals the endpoint
     difference over two, so:

         selectivity = (N_late - N_early) / 4        exactly,
         where N = |ERD_contra| - |ERD_ipsi|

     Slope and endpoint difference are the same statistic up to a constant.
     NORMALISATION IS THE ONLY DIFFERENCE between the two measures.

  2. The proxy denominator test was contaminated. It estimated
     D = selectivity / dLI, then partialled dLI on it - controlling dLI for
     a quantity with dLI in its denominator. The CV comparison inherits the
     same defect, since the ratio diverges wherever dLI approaches zero.

THE SHARPER HYPOTHESIS

     dLI = N_late / D_late  -  N_early / D_early

  The denominator is PER STAGE. If D were constant across stages, dLI would
  be selectivity rescaled. It is not - so dLI absorbs variance from how
  total ERD amplitude CHANGES across the session, not merely from its level.
  In stroke that drift is plausibly lesion-dependent and severity-related.

WHAT THIS SCRIPT DOES

  PART 1  Extract per-stage |ERD_contra| and |ERD_ipsi| (early / mid / late)
          for all five datasets. This is the quantity missing from
          features_per_subject.csv.
  PART 2  Verify selectivity = (N_late - N_early)/4 against the existing
          selectivity column. If this does not hold, the derivation above is
          wrong for this pipeline and everything downstream must be revisited.
  PART 3  The proper denominator test, with D measured rather than inferred:
            P1  is D more variable in stroke than in healthy cohorts?
            P2  is D, or its across-stage change dD, related to NIHSS?
            P3  partial correlations of selectivity and dLI with NIHSS,
                controlling for D_mean and for dD - uncontaminated this time.

CONFIG: the preprocessing block below must match 03_compute_features.py.
Check FMIN/FMAX, TMIN/TMAX, BASELINE and the stage split before running.

OUTPUT: perstage_erd_per_subject.csv, denominator_proper_report.txt
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
import mne
mne.set_log_level("ERROR")
from moabb.datasets import PhysionetMI, Cho2017, Lee2019_MI, BNCI2014_004, Liu2024
from moabb.paradigms import LeftRightImagery
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- CONFIG
SFREQ    = 160.0
FMIN, FMAX = 8.0, 13.0        # mu band - match 03_compute_features.py
TMIN, TMAX = 0.0, 3.0
BASE_END = 0.5                # baseline 0.0-0.5 s
REJECT_UV = 250e-6
OUT_FEAT = "perstage_erd_per_subject.csv"
REPORT   = "denominator_proper_report.txt"

DATASETS = [("PhysionetMI", PhysionetMI()), ("Cho2017", Cho2017()),
            ("BNCI2014_004", BNCI2014_004()), ("Lee2019_MI", Lee2019_MI()),
            ("Liu2024", Liu2024())]

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


def band_power(x, sf, lo, hi):
    nper = int(min(x.shape[-1], sf))
    f, pxx = signal.welch(x, fs=sf, nperseg=max(nper, 32), axis=-1)
    m = (f >= lo) & (f < hi)
    return np.trapezoid(pxx[..., m], f[m], axis=-1)


def erd_db(ep, ch_idx):
    """ERD in dB per trial for one channel: 10*log10(P_MI / P_base)."""
    X = ep.get_data()[:, ch_idx, :]
    nb = int(BASE_END * SFREQ)
    pb = band_power(X[:, :nb], SFREQ, FMIN, FMAX)
    pm = band_power(X[:, nb:], SFREQ, FMIN, FMAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(pm / pb)


rows = []
say("=" * 74)
say("PART 1 — per-stage ERD magnitude extraction")
say("=" * 74)

for name, ds in DATASETS:
    say(f"\n--- {name} ---")
    para = LeftRightImagery(fmin=FMIN, fmax=FMAX, resample=SFREQ,
                            tmin=TMIN, tmax=TMAX, baseline=None)
    try:
        subs = ds.subject_list
    except Exception as e:
        say(f"  cannot list subjects: {e}")
        continue

    for s in subs:
        try:
            ep, y, meta = para.get_data(ds, subjects=[s], return_epochs=True)
        except Exception as e:
            say(f"  subj {s}: load failed ({type(e).__name__})")
            continue
        chs = ep.ch_names
        if "C3" not in chs or "C4" not in chs:
            say(f"  subj {s}: C3/C4 absent — skipped")
            continue
        i3, i4 = chs.index("C3"), chs.index("C4")
        sessions = meta["session"].unique() if "session" in meta else ["0"]

        for sess in sessions:
            sm = (meta["session"].values == sess) if "session" in meta else np.ones(len(y), bool)
            for task, lab in [("T1", "left_hand"), ("T2", "right_hand")]:
                m = sm & (np.asarray(y) == lab)
                if m.sum() < 9:
                    continue
                sub = ep[np.where(m)[0]]
                # contralateral channel depends on the imagined hand
                ci, ii = (i4, i3) if lab == "left_hand" else (i3, i4)
                ec, ei = erd_db(sub, ci), erd_db(sub, ii)
                ok = np.isfinite(ec) & np.isfinite(ei)
                ec, ei = ec[ok], ei[ok]
                if len(ec) < 9:
                    continue
                # equal early / mid / late thirds, chronological
                thirds = np.array_split(np.arange(len(ec)), 3)
                rec = dict(dataset=name, subject=str(s), session=str(sess),
                           task=task, n_trials=len(ec))
                for st, idx in zip(["early", "mid", "late"], thirds):
                    c, i_ = np.abs(ec[idx]).mean(), np.abs(ei[idx]).mean()
                    rec[f"absC_{st}"] = c
                    rec[f"absI_{st}"] = i_
                    rec[f"N_{st}"] = c - i_          # numerator
                    rec[f"D_{st}"] = c + i_          # denominator
                    rec[f"LI_{st}"] = (c - i_) / (c + i_) if (c + i_) else np.nan
                rec["D_mean"] = np.mean([rec["D_early"], rec["D_mid"], rec["D_late"]])
                rec["dD"] = rec["D_late"] - rec["D_early"]
                rec["dD_rel"] = rec["dD"] / rec["D_mean"] if rec["D_mean"] else np.nan
                rec["sel_derived"] = (rec["N_late"] - rec["N_early"]) / 4.0
                rec["dLI_derived"] = rec["LI_late"] - rec["LI_early"]
                rows.append(rec)
        say(f"  subj {s}: ok")

F = pd.DataFrame(rows)
F.to_csv(OUT_FEAT, index=False)
say(f"\nWrote {OUT_FEAT}  ({len(F)} rows)")
say(F.groupby("dataset").size().to_string())

# ------------------------------------------------------------------ PART 2
say("\n" + "=" * 74)
say("PART 2 — verify selectivity = (N_late - N_early)/4")
say("=" * 74)
try:
    exist = pd.read_csv("features_per_subject.csv")
    exist["subject"] = exist.subject.astype(str)
    sel_c = next((c for c in exist.columns if "mu_ERD_selectivity_T1" in c), None)
    if sel_c:
        j = F[F.task == "T1"].merge(exist[["dataset", "subject", sel_c]],
                                    on=["dataset", "subject"], how="inner")
        j = j.dropna(subset=["sel_derived", sel_c])
        if len(j) > 20:
            r, p = stats.pearsonr(j.sel_derived, j[sel_c])
            slope = np.polyfit(j[sel_c], j.sel_derived, 1)[0]
            say(f"  n={len(j)}  r={r:+.4f}  slope={slope:.4f}")
            if r > 0.98:
                say("  IDENTITY CONFIRMED — the derivation holds for this pipeline.")
            else:
                say("  !! IDENTITY DOES NOT HOLD. The bilateral term or the stage")
                say("     split differs from the assumption. Investigate before")
                say("     interpreting anything below.")
except Exception as e:
    say(f"  could not verify against features_per_subject.csv: {e}")

# ------------------------------------------------------------------ PART 3
say("\n" + "=" * 74)
say("PART 3 — denominator test with D MEASURED, not inferred")
say("=" * 74)

G = F[F.task == "T1"].drop_duplicates(subset=["dataset", "subject", "session"])

say("\nP1 — variability of D across cohorts")
tab = []
for ds, g in G.groupby("dataset"):
    d = g.D_mean.dropna()
    if len(d) < 15:
        continue
    tab.append(dict(dataset=ds, n=len(d), mean=d.mean(), sd=d.std(),
                    CV=d.std() / abs(d.mean()),
                    robust_CV=stats.median_abs_deviation(d, scale="normal") / abs(np.median(d)),
                    dD_sd=g.dD.std()))
T = pd.DataFrame(tab)
say(T.round(3).to_string(index=False))
if T.dataset.str.contains("Liu").any():
    liu = T[T.dataset.str.contains("Liu")].iloc[0]
    hea = T[~T.dataset.str.contains("Liu")]
    say(f"\n  Liu CV={liu.CV:.3f}  healthy range {hea.CV.min():.3f}-{hea.CV.max():.3f}")
    say(f"  Liu dD_sd={liu.dD_sd:.3f}  healthy range "
        f"{hea.dD_sd.min():.3f}-{hea.dD_sd.max():.3f}")
    say("  P1 supported only if Liu exceeds EVERY healthy cohort.")
    grps = [g.D_mean.dropna().values for _, g in G.groupby("dataset") if len(g) >= 15]
    if len(grps) > 1:
        W, pw = stats.levene(*grps, center="median")
        say(f"  Levene across cohorts: W={W:.3f} p={pw:.4g}")

say("\nP2 / P3 — stroke cohort against NIHSS")
try:
    clin = pd.read_csv("liu_clinical.csv")
    clin["subject"] = clin.subject.astype(str)
    S = G[G.dataset.str.contains("Liu")].merge(clin, on="subject", how="inner")
    say(f"  merged: {len(S)} (expect 50)")

    for lab, col in [("D_mean", "D_mean"), ("dD (late-early)", "dD"),
                     ("dD relative", "dD_rel")]:
        d = S.dropna(subset=[col, "NIHSS"])
        rho, p = stats.spearmanr(d[col], d.NIHSS)
        say(f"    {lab:18s} vs NIHSS: rho={rho:+.3f} p={p:.4f}  n={len(d)}")

    say("\n  Selectivity and dLI vs NIHSS, raw and partialled on D:")
    for nm, col in [("selectivity", "sel_derived"), ("dLI", "dLI_derived")]:
        d = S.dropna(subset=[col, "NIHSS", "D_mean", "dD"])
        raw, praw = stats.spearmanr(d[col], d.NIHSS)
        line = f"    {nm:12s} raw rho={raw:+.3f} (p={praw:.4f})"
        for ctrl in ["D_mean", "dD"]:
            rx = d[col] - np.poly1d(np.polyfit(d[ctrl], d[col], 1))(d[ctrl])
            ry = d.NIHSS - np.poly1d(np.polyfit(d[ctrl], d.NIHSS, 1))(d[ctrl])
            rr, pp = stats.spearmanr(rx, ry)
            line += f"   | {ctrl}-partial rho={rr:+.3f} (p={pp:.4f})"
        say(line)

    say("\n  DECISION:")
    say("   dLI STRENGTHENS when partialled on D_mean or dD, selectivity")
    say("     unchanged  -> normalisation is what attenuates dLI. State the")
    say("     mechanism in section 3.2 and the clinical section.")
    say("   NEITHER changes  -> normalisation is NOT the explanation. Since")
    say("     the estimator hypothesis is mathematically void, no mechanism")
    say("     remains: report the divergence as an observation, unexplained.")
    say("   dD tracks NIHSS  -> the drift in total amplitude is itself")
    say("     severity-linked, which is the contamination account and is")
    say("     reportable in its own right.")
except Exception as e:
    say(f"  clinical merge failed: {e}")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
