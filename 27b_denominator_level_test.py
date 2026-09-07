"""
27b_denominator_level_test.py
-----------------------------
Fixed re-run of Part C. Two corrections from 27:

  1. np.trapz was removed in NumPy 2.x -> np.trapezoid (with fallback).
  2. Part B parsed fmin=1, fmax=45 from 03_compute_features.py. That is the
     PARADIGM's broadband filter, not the mu band. The pipeline band-filters
     downstream, so FMIN/FMAX are now forced to 8-13 Hz.

  Also changed: the ERD computation now mirrors the Methods description -
  second-order Butterworth zero-phase bandpass FIRST, then Welch power -
  rather than integrating an unfiltered PSD. This should let the identity
  gate pass.

PART A is not repeated; it already ran and is settled:
    dD vs NIHSS                 rho = +0.078, p = 0.59   (drift arm null)
    selectivity dD-partial      rho = -0.473, p = 0.0005
    dLI         dD-partial      rho = -0.277, p = 0.0517

PART C tests the last remaining candidate: the LEVEL of total ERD
amplitude, D_mean.

OUTPUT: denominator_level_report_v2.txt, perstage_erd_matched.csv
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
import warnings
warnings.filterwarnings("ignore")

# NumPy 2.x compatibility
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

REPORT = "denominator_level_report_v2.txt"
OUTCSV = "perstage_erd_matched.csv"

# ---- CONFIG (matched to 03_compute_features.py) ----------------------
FMIN, FMAX = 8.0, 13.0        # mu band, applied AFTER broadband load
LOAD_FMIN, LOAD_FMAX = 1.0, 45.0   # what the paradigm itself uses
TMIN, TMAX = 0.0, 3.0
SFREQ = 160.0
BASE_END = 0.5
CHANNELS = ["C3", "Cz", "C4"]      # line 45 of 03_compute_features.py

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

say(f"config: load {LOAD_FMIN}-{LOAD_FMAX} Hz, analyse {FMIN}-{FMAX} Hz, "
    f"{TMIN}-{TMAX} s, sf={SFREQ}, baseline 0-{BASE_END} s, channels={CHANNELS}")

import mne
mne.set_log_level("ERROR")
from moabb.datasets import PhysionetMI, Cho2017, Lee2019_MI, BNCI2014_004, Liu2024
from moabb.paradigms import LeftRightImagery

DATASETS = [("PhysionetMI", PhysionetMI()), ("Cho2017", Cho2017()),
            ("BNCI2014_004", BNCI2014_004()), ("Lee2019_MI", Lee2019_MI()),
            ("Liu2024", Liu2024())]

# second-order Butterworth, zero-phase — as described in the Methods
B, A = signal.butter(2, [FMIN / (SFREQ / 2), FMAX / (SFREQ / 2)], btype="band")


def band_power(x):
    """Bandpass then Welch, integrated over the passband."""
    xf = signal.filtfilt(B, A, x, axis=-1)
    nper = int(min(xf.shape[-1], SFREQ))
    f, pxx = signal.welch(xf, fs=SFREQ, nperseg=max(nper, 32), axis=-1)
    m = (f >= FMIN) & (f < FMAX)
    return TRAPZ(pxx[..., m], f[m], axis=-1)


rows = []
say("\n--- extraction ---")
for name, ds in DATASETS:
    para = LeftRightImagery(fmin=LOAD_FMIN, fmax=LOAD_FMAX, resample=SFREQ,
                            tmin=TMIN, tmax=TMAX, baseline=None,
                            channels=CHANNELS)
    try:
        subs = ds.subject_list
    except Exception as e:
        say(f"  {name}: cannot list subjects ({e})"); continue
    nok = 0
    for s in subs:
        try:
            ep, y, meta = para.get_data(ds, subjects=[s], return_epochs=True)
        except Exception:
            continue
        chs = ep.ch_names
        if "C3" not in chs or "C4" not in chs:
            continue
        i3, i4 = chs.index("C3"), chs.index("C4")
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

                def erd(idx):
                    pb_ = band_power(Xi[:, idx, :nb])
                    pm_ = band_power(Xi[:, idx, nb:])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return 10 * np.log10(pm_ / pb_)

                ec, ei = erd(ci), erd(ii)
                ok = np.isfinite(ec) & np.isfinite(ei)
                ec, ei = ec[ok], ei[ok]
                if len(ec) < 9:
                    continue
                rec = dict(dataset=name, subject=str(s), session=str(sv), task=task,
                           n_trials=len(ec))
                for st, idx in zip(["early", "mid", "late"],
                                   np.array_split(np.arange(len(ec)), 3)):
                    c, i_ = np.abs(ec[idx]).mean(), np.abs(ei[idx]).mean()
                    rec[f"absC_{st}"], rec[f"absI_{st}"] = c, i_
                    rec[f"N_{st}"] = c - i_
                    rec[f"D_{st}"] = c + i_
                    rec[f"LI_{st}"] = (c - i_) / (c + i_) if (c + i_) else np.nan
                rec["D_mean"] = np.mean([rec["D_early"], rec["D_mid"], rec["D_late"]])
                rec["dD"] = rec["D_late"] - rec["D_early"]
                rec["sel_derived"] = (rec["N_late"] - rec["N_early"]) / 4.0
                rec["dLI_derived"] = rec["LI_late"] - rec["LI_early"]
                rows.append(rec)
        nok += 1
    say(f"  {name}: {nok} subjects")

P = pd.DataFrame(rows)
P.to_csv(OUTCSV, index=False)
say(f"\nWrote {OUTCSV} ({len(P)} rows)")

# ---------------------------------------------------------------- gate
say("\n" + "=" * 70)
say("IDENTITY GATE — sel_derived vs the pipeline's mu_ERD_selectivity_T1")
say("=" * 70)
GATE = False
clin = None
try:
    F = pd.read_csv("features_per_subject.csv"); F["subject"] = F.subject.astype(str)
    selc = next((c for c in F.columns if "mu_ERD_selectivity_T1" in c), None)
    j = (P[P.task == "T1"]
         .merge(F[["dataset", "subject", selc]], on=["dataset", "subject"], how="inner")
         .dropna(subset=["sel_derived", selc]))
    r, p = stats.pearsonr(j.sel_derived, j[selc])
    slope = np.polyfit(j[selc], j.sel_derived, 1)[0]
    say(f"  n={len(j)}  r={r:+.4f}  slope={slope:.4f}")
    GATE = r > 0.95
    say("  GATE PASSED — results below are interpretable." if GATE else
        "  GATE FAILED — preprocessing still differs. Do NOT interpret the\n"
        "  results below. Next step: compare the ERD function in\n"
        "  03_compute_features.py line by line against band_power() here.")
except Exception as e:
    say(f"  gate check failed: {e}")

try:
    clin = pd.read_csv("liu_clinical.csv"); clin["subject"] = clin.subject.astype(str)
except Exception as e:
    say(f"  clinical file unavailable: {e}")

# ------------------------------------------------------------ level test
if GATE and clin is not None:
    G = P[P.task == "T1"].drop_duplicates(subset=["dataset", "subject", "session"])

    say("\n" + "=" * 70)
    say("D_mean by cohort")
    say("=" * 70)
    T = G.groupby("dataset").D_mean.agg(["count", "mean", "std"])
    T["CV"] = T["std"] / T["mean"]
    say(T.round(3).to_string())

    S = G[G.dataset.str.contains("Liu")].merge(clin, on="subject", how="inner")
    say(f"\nstroke merged: {len(S)}")

    say("\n" + "=" * 70)
    say("THE LEVEL TEST")
    say("=" * 70)
    for lab, col in [("D_mean", "D_mean"), ("dD", "dD")]:
        d = S.dropna(subset=[col, "NIHSS"])
        rho, p = stats.spearmanr(d[col], d.NIHSS)
        say(f"  {lab:8s} vs NIHSS: rho={rho:+.3f} p={p:.4f}  n={len(d)}")

    say("\n  selectivity / dLI vs NIHSS, raw and partialled on D_mean:")
    for nm, col in [("selectivity", "sel_derived"), ("dLI", "dLI_derived")]:
        d = S.dropna(subset=[col, "NIHSS", "D_mean"])
        raw, praw = stats.spearmanr(d[col], d.NIHSS)
        rx = d[col] - np.poly1d(np.polyfit(d.D_mean, d[col], 1))(d.D_mean)
        ry = d.NIHSS - np.poly1d(np.polyfit(d.D_mean, d.NIHSS, 1))(d.D_mean)
        rr, pp = stats.spearmanr(rx, ry)
        say(f"    {nm:12s} raw rho={raw:+.3f} (p={praw:.4f})  ->  "
            f"D_mean-partial rho={rr:+.3f} (p={pp:.4f})")

    say("\n  DECISION (pre-committed):")
    say("   dLI strengthens materially while selectivity is unchanged")
    say("     -> the LEVEL of total ERD amplitude attenuates dLI. Report it.")
    say("   both move together, or neither moves")
    say("     -> with the drift arm null (PART A of script 27) and the")
    say("        estimator hypothesis mathematically void, no mechanism")
    say("        remains. Report the selectivity/dLI difference in sensitivity")
    say("        as an observation, without a mechanism.")
    say("\n  Note either way: selectivity rho = -0.436 (p = 0.002) and")
    say("  dLI rho = -0.264 (p = 0.064) point the SAME direction. The honest")
    say("  framing is a difference in sensitivity, not presence vs absence.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
