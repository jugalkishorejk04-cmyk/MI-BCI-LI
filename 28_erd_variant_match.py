"""
28_erd_variant_match.py
-----------------------
Finds which ERD aggregation the pipeline uses, instead of guessing again.

THE DIAGNOSIS
  Scripts 26 and 27b both hit r = 0.20 against the pipeline's selectivity,
  and changing channels and filter order moved it by 0.001. So the mismatch
  is not a parameter. In the 27b output:

      mean |ERD_contra| = 4.306      mean |ERD_ipsi| = 4.285
      mean N_early      = 0.020      54% of subjects have |LI_early| < 0.1

  The lateralisation is gone. That is what happens when abs() is applied
  PER TRIAL before averaging: per-trial ERD scatters around zero, so
  mean(|x|) measures trial noise magnitude, near-identical in both
  hemispheres. The signal is in |mean(x)|.

  03_compute_features.py line 316 reads a SINGLE value per stage and then
  takes abs of it - so it aggregates first, rectifies second.

THREE CANDIDATE AGGREGATIONS
  A  mean over trials of |10log10(Pmi/Pbase)|        <- what 26/27b did
  B  |mean over trials of 10log10(Pmi/Pbase)|        <- log per trial, then average
  C  |10log10(mean(Pmi)/mean(Pbase))|                <- average power, then log

  B and C differ when trial power is skewed, which it usually is. The
  Methods text ("ERD_dB = 10 log10(P_MI / P_baseline), Welch's method",
  singular P) points to C, but this tests all three.

Each variant is scored against mu_ERD_selectivity_T1 from the feature file.
Whichever exceeds r = 0.95 is the pipeline's method, and the D-level test
then runs on that variant only.

OUTPUT: erd_variant_report.txt, perstage_variants.csv
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
import warnings
warnings.filterwarnings("ignore")

TRAPZ = getattr(np, "trapezoid", None) or np.trapz
REPORT, OUTCSV = "erd_variant_report.txt", "perstage_variants.csv"

FMIN, FMAX = 8.0, 13.0
LOAD_FMIN, LOAD_FMAX = 1.0, 45.0
TMIN, TMAX, SFREQ, BASE_END = 0.0, 3.0, 160.0, 0.5
CHANNELS = ["C3", "Cz", "C4"]

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

import mne
mne.set_log_level("ERROR")
from moabb.datasets import PhysionetMI, Cho2017, Lee2019_MI, BNCI2014_004, Liu2024
from moabb.paradigms import LeftRightImagery

DATASETS = [("PhysionetMI", PhysionetMI()), ("Cho2017", Cho2017()),
            ("BNCI2014_004", BNCI2014_004()), ("Lee2019_MI", Lee2019_MI()),
            ("Liu2024", Liu2024())]

B, A = signal.butter(2, [FMIN / (SFREQ / 2), FMAX / (SFREQ / 2)], btype="band")


def trial_power(x):
    """Per-trial band power: bandpass, Welch, integrate."""
    xf = signal.filtfilt(B, A, x, axis=-1)
    nper = int(min(xf.shape[-1], SFREQ))
    f, pxx = signal.welch(xf, fs=SFREQ, nperseg=max(nper, 32), axis=-1)
    m = (f >= FMIN) & (f < FMAX)
    return TRAPZ(pxx[..., m], f[m], axis=-1)


rows = []
say("--- extraction (three variants computed in one pass) ---")
for name, ds in DATASETS:
    para = LeftRightImagery(fmin=LOAD_FMIN, fmax=LOAD_FMAX, resample=SFREQ,
                            tmin=TMIN, tmax=TMAX, baseline=None, channels=CHANNELS)
    try:
        subs = ds.subject_list
    except Exception as e:
        say(f"  {name}: {e}"); continue
    nok = 0
    for s in subs:
        try:
            ep, y, meta = para.get_data(ds, subjects=[s], return_epochs=True)
        except Exception:
            continue
        if "C3" not in ep.ch_names or "C4" not in ep.ch_names:
            continue
        i3, i4 = ep.ch_names.index("C3"), ep.ch_names.index("C4")
        X = ep.get_data()
        nb = int(BASE_END * SFREQ)
        sess = meta["session"].values if "session" in meta else np.array(["0"] * len(y))
        for sv in np.unique(sess):
            for task, lab in [("T1", "left_hand"), ("T2", "right_hand")]:
                m = (sess == sv) & (np.asarray(y) == lab)
                if m.sum() < 9:
                    continue
                Xi = X[m]
                ci, ii = (i4, i3) if lab == "left_hand" else (i3, i4)
                rec = dict(dataset=name, subject=str(s), session=str(sv),
                           task=task, n_trials=int(m.sum()))
                ok_all = True
                store = {}
                for side, idx in [("C", ci), ("I", ii)]:
                    pb_ = trial_power(Xi[:, idx, :nb])
                    pm_ = trial_power(Xi[:, idx, nb:])
                    good = np.isfinite(pb_) & np.isfinite(pm_) & (pb_ > 0) & (pm_ > 0)
                    if good.sum() < 9:
                        ok_all = False; break
                    pb_, pm_ = pb_[good], pm_[good]
                    store[side] = (pb_, pm_)
                if not ok_all:
                    continue
                n = min(len(store["C"][0]), len(store["I"][0]))
                thirds = np.array_split(np.arange(n), 3)
                for st, idx in zip(["early", "mid", "late"], thirds):
                    for side in ("C", "I"):
                        pb_, pm_ = store[side]
                        db = 10 * np.log10(pm_[idx] / pb_[idx])
                        rec[f"A_{side}_{st}"] = np.abs(db).mean()
                        rec[f"B_{side}_{st}"] = abs(db.mean())
                        rec[f"C_{side}_{st}"] = abs(10 * np.log10(pm_[idx].mean() / pb_[idx].mean()))
                rows.append(rec)
        nok += 1
    say(f"  {name}: {nok} subjects")

P = pd.DataFrame(rows)

for V in ("A", "B", "C"):
    for st in ("early", "mid", "late"):
        c, i_ = P[f"{V}_C_{st}"], P[f"{V}_I_{st}"]
        P[f"{V}_N_{st}"] = c - i_
        P[f"{V}_D_{st}"] = c + i_
        P[f"{V}_LI_{st}"] = (c - i_) / (c + i_)
    P[f"{V}_sel"] = (P[f"{V}_N_late"] - P[f"{V}_N_early"]) / 4.0
    P[f"{V}_dLI"] = P[f"{V}_LI_late"] - P[f"{V}_LI_early"]
    P[f"{V}_Dmean"] = P[[f"{V}_D_early", f"{V}_D_mid", f"{V}_D_late"]].mean(axis=1)

P.to_csv(OUTCSV, index=False)
say(f"\nWrote {OUTCSV} ({len(P)} rows)")

# ---------------------------------------------------------------- scoring
say("\n" + "=" * 70)
say("WHICH VARIANT MATCHES THE PIPELINE?")
say("=" * 70)
F = pd.read_csv("features_per_subject.csv"); F["subject"] = F.subject.astype(str)
selc = next((c for c in F.columns if "mu_ERD_selectivity_T1" in c), None)
best, best_r = None, -1
for V in ("A", "B", "C"):
    j = (P[P.task == "T1"].merge(F[["dataset", "subject", selc]],
                                 on=["dataset", "subject"], how="inner")
         .dropna(subset=[f"{V}_sel", selc]))
    if len(j) < 30:
        continue
    r, _ = stats.pearsonr(j[f"{V}_sel"], j[selc])
    lab = {"A": "mean(|dB|)  per-trial rectify",
           "B": "|mean(dB)|  rectify after averaging",
           "C": "|dB(mean P)| average power then log"}[V]
    say(f"  {V}  {lab:38s} n={len(j):3d}  r={r:+.4f}")
    if r > best_r:
        best, best_r = V, r
say(f"\n  best: variant {best} at r = {best_r:+.4f}")
if best_r <= 0.95:
    say("  NONE MATCHES. The difference is elsewhere - most likely the stage")
    say("  split (are thirds taken over task-specific trials, or over all")
    say("  trials interleaved?) or an epoch-rejection step. Print the ERD")
    say("  function and the stage loop from 03_compute_features.py.")

# ---------------------------------------------------------- level test
if best_r > 0.95:
    say("\n" + "=" * 70)
    say(f"D-LEVEL TEST using variant {best}")
    say("=" * 70)
    clin = pd.read_csv("liu_clinical.csv"); clin["subject"] = clin.subject.astype(str)
    G = P[P.task == "T1"].drop_duplicates(subset=["dataset", "subject", "session"])
    T = G.groupby("dataset")[f"{best}_Dmean"].agg(["count", "mean", "std"])
    T["CV"] = T["std"] / T["mean"]
    say(T.round(3).to_string())

    S = G[G.dataset.str.contains("Liu")].merge(clin, on="subject", how="inner")
    say(f"\nstroke merged: {len(S)}")
    d = S.dropna(subset=[f"{best}_Dmean", "NIHSS"])
    rho, p = stats.spearmanr(d[f"{best}_Dmean"], d.NIHSS)
    say(f"  D_mean vs NIHSS: rho={rho:+.3f} p={p:.4f}  n={len(d)}")

    say("\n  selectivity / dLI vs NIHSS, raw and partialled on D_mean:")
    for nm, col in [("selectivity", f"{best}_sel"), ("dLI", f"{best}_dLI")]:
        d = S.dropna(subset=[col, "NIHSS", f"{best}_Dmean"])
        raw, praw = stats.spearmanr(d[col], d.NIHSS)
        D = d[f"{best}_Dmean"]
        rx = d[col] - np.poly1d(np.polyfit(D, d[col], 1))(D)
        ry = d.NIHSS - np.poly1d(np.polyfit(D, d.NIHSS, 1))(D)
        rr, pp = stats.spearmanr(rx, ry)
        say(f"    {nm:12s} raw rho={raw:+.3f} (p={praw:.4f})  ->  "
            f"D_mean-partial rho={rr:+.3f} (p={pp:.4f})")

    say("\n  DECISION (pre-committed):")
    say("   dLI strengthens materially, selectivity unchanged -> amplitude")
    say("     LEVEL is the mechanism; report it in section 3.2 and 3.4.")
    say("   otherwise -> drift arm null, estimator void, level null: no")
    say("     mechanism remains. Report the sensitivity difference as an")
    say("     observation and stop.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
