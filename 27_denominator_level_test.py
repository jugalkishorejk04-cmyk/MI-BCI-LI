"""
27_denominator_level_test.py
----------------------------
Re-runs the denominator test with preprocessing read directly from
03_compute_features.py, so the identity check in PART B can pass.

WHY SCRIPT 26 FAILED
  Line 45 of 03_compute_features.py restricts channels at the PARADIGM
  level: channels=["C3","Cz","C4"]. Script 26 loaded the full montage and
  picked C3/C4 afterwards. Different preprocessing path, different ERD
  values, so selectivity_derived correlated with the pipeline's selectivity
  at only r = 0.20. This script reads the config from your own source file
  instead of guessing.

WHAT IS ALREADY SETTLED (no computation needed)
  bilateral = (|contra| + |ipsi|) / 2 = D/2      [03_compute_features.py:320]
  With three equally spaced stages, slope = (last - first)/2, so

      dD = D_late - D_early = 4 x mu_ERD_slope_bilateral_T1

  That column was already tested against NIHSS in the section 3.7 confound
  analysis: rho = +0.078, p = 0.59. The AMPLITUDE-DRIFT arm of the
  hypothesis is therefore already answered, and it is null. PART A below
  re-derives it from your data so the equivalence is on the record.

WHAT REMAINS
  D_mean - the LEVEL of total ERD amplitude - cannot be recovered from
  slopes alone. PART C extracts it and tests whether it explains why
  selectivity tracks NIHSS while dLI does not.

OUTPUT: denominator_level_report.txt, perstage_erd_matched.csv
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats, signal
import warnings
warnings.filterwarnings("ignore")

REPORT = "denominator_level_report.txt"
OUTCSV = "perstage_erd_matched.csv"
SRC    = "03_compute_features.py"

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


# ======================================================================
# PART A — the drift arm, from existing columns only (no EEG processing)
# ======================================================================
say("=" * 74)
say("PART A — dD from existing columns (dD = 4 x slope_bilateral)")
say("=" * 74)

try:
    F = pd.read_csv("features_per_subject.csv")
    F["subject"] = F.subject.astype(str)
    clin = pd.read_csv("liu_clinical.csv")
    clin["subject"] = clin.subject.astype(str)
    S = F[F.dataset.str.contains("Liu", case=False)].merge(clin, on="subject", how="inner")
    say(f"stroke rows: {len(S)}")

    bil = next((c for c in S.columns if "mu_ERD_slope_bilateral_T1" in c), None)
    sel = next((c for c in S.columns if "mu_ERD_selectivity_T1" in c), None)
    dli = next((c for c in S.columns if c in ("dLI", "dLI_mu_T1", "ΔLI_mu_T1")), None)

    if bil:
        S["dD"] = 4.0 * S[bil]
        d = S.dropna(subset=["dD", "NIHSS"])
        rho, p = stats.spearmanr(d.dD, d.NIHSS)
        say(f"  dD vs NIHSS:            rho={rho:+.3f} p={p:.4f}  n={len(d)}")
        rb, pb = stats.spearmanr(d[bil], d.NIHSS)
        say(f"  slope_bilateral vs NIHSS: rho={rb:+.3f} p={pb:.4f}   (identical by construction)")
        say("  -> the amplitude-DRIFT arm is null; this reproduces the")
        say("     section 3.7 confound result and shows the two are the same test.")

    if sel and dli:
        d = S.dropna(subset=[sel, dli, "dD", "NIHSS"])
        if len(d) > 20:
            say("\n  Partial correlations with NIHSS, controlling for dD:")
            for nm, col in [("selectivity", sel), ("dLI", dli)]:
                raw, praw = stats.spearmanr(d[col], d.NIHSS)
                rx = d[col] - np.poly1d(np.polyfit(d.dD, d[col], 1))(d.dD)
                ry = d.NIHSS - np.poly1d(np.polyfit(d.dD, d.NIHSS, 1))(d.dD)
                rr, pp = stats.spearmanr(rx, ry)
                say(f"    {nm:12s} raw rho={raw:+.3f} (p={praw:.4f})  ->  "
                    f"dD-partial rho={rr:+.3f} (p={pp:.4f})")
except Exception as e:
    say(f"  PART A failed: {e}")


# ======================================================================
# PART B — read the pipeline's own preprocessing config
# ======================================================================
say("\n" + "=" * 74)
say("PART B — configuration read from " + SRC)
say("=" * 74)

cfg = dict(fmin=None, fmax=None, tmin=None, tmax=None, resample=None,
           channels=None, baseline_end=None, reject=None)
try:
    src = open(SRC, encoding="utf-8", errors="replace").read()
    pats = {
        "fmin": r"fmin\s*=\s*([\d.]+)", "fmax": r"fmax\s*=\s*([\d.]+)",
        "tmin": r"tmin\s*=\s*([\d.]+)", "tmax": r"tmax\s*=\s*([\d.]+)",
        "resample": r"resample\s*=\s*([\d.]+)",
        "channels": r"channels\s*=\s*(\[[^\]]*\])",
    }
    for k, pat in pats.items():
        m = re.search(pat, src)
        if m:
            cfg[k] = m.group(1)
    m = re.search(r"baseline[^\n]*?([\d.]+)\s*[,)]", src)
    if m:
        cfg["baseline_end"] = m.group(1)
    for k, v in cfg.items():
        say(f"  {k:14s} = {v}")
    say("\n  CHECK THESE against the source before trusting PART C. If any")
    say("  came back None, set it manually in the OVERRIDE block below.")
except Exception as e:
    say(f"  could not parse {SRC}: {e}")

# ---- OVERRIDE: set anything the parser missed -------------------------
FMIN = float(cfg["fmin"] or 8.0)
FMAX = float(cfg["fmax"] or 13.0)
TMIN = float(cfg["tmin"] or 0.0)
TMAX = float(cfg["tmax"] or 3.0)
SFREQ = float(cfg["resample"] or 160.0)
BASE_END = float(cfg["baseline_end"] or 0.5)
CHANNELS = ["C3", "Cz", "C4"]      # line 45 of 03_compute_features.py
say(f"\n  using: {FMIN}-{FMAX} Hz, {TMIN}-{TMAX} s, sf={SFREQ}, "
    f"baseline 0-{BASE_END} s, channels={CHANNELS}")


# ======================================================================
# PART C — extraction with matched config, identity gate, level test
# ======================================================================
say("\n" + "=" * 74)
say("PART C — per-stage extraction, identity gate, D-level test")
say("=" * 74)

import mne
mne.set_log_level("ERROR")
from moabb.datasets import PhysionetMI, Cho2017, Lee2019_MI, BNCI2014_004, Liu2024
from moabb.paradigms import LeftRightImagery

DATASETS = [("PhysionetMI", PhysionetMI()), ("Cho2017", Cho2017()),
            ("BNCI2014_004", BNCI2014_004()), ("Lee2019_MI", Lee2019_MI()),
            ("Liu2024", Liu2024())]


def bp(x, sf, lo, hi):
    nper = int(min(x.shape[-1], sf))
    f, pxx = signal.welch(x, fs=sf, nperseg=max(nper, 32), axis=-1)
    m = (f >= lo) & (f < hi)
    return np.trapz(pxx[..., m], f[m], axis=-1)


rows = []
for name, ds in DATASETS:
    para = LeftRightImagery(fmin=FMIN, fmax=FMAX, resample=SFREQ,
                            tmin=TMIN, tmax=TMAX, baseline=None,
                            channels=CHANNELS)          # <-- the fix
    try:
        subs = ds.subject_list
    except Exception as e:
        say(f"{name}: cannot list subjects ({e})"); continue
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
                    pb_ = bp(Xi[:, idx, :nb], SFREQ, FMIN, FMAX)
                    pm_ = bp(Xi[:, idx, nb:], SFREQ, FMIN, FMAX)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return 10 * np.log10(pm_ / pb_)
                ec, ei = erd(ci), erd(ii)
                ok = np.isfinite(ec) & np.isfinite(ei)
                ec, ei = ec[ok], ei[ok]
                if len(ec) < 9:
                    continue
                rec = dict(dataset=name, subject=str(s), session=str(sv), task=task)
                for st, idx in zip(["early", "mid", "late"], np.array_split(np.arange(len(ec)), 3)):
                    c, i_ = np.abs(ec[idx]).mean(), np.abs(ei[idx]).mean()
                    rec[f"N_{st}"] = c - i_
                    rec[f"D_{st}"] = c + i_
                    rec[f"LI_{st}"] = (c - i_) / (c + i_) if (c + i_) else np.nan
                rec["D_mean"] = np.mean([rec["D_early"], rec["D_mid"], rec["D_late"]])
                rec["dD"] = rec["D_late"] - rec["D_early"]
                rec["sel_derived"] = (rec["N_late"] - rec["N_early"]) / 4.0
                rec["dLI_derived"] = rec["LI_late"] - rec["LI_early"]
                rows.append(rec)
        nok += 1
    say(f"  {name}: {nok} subjects processed")

P = pd.DataFrame(rows)
P.to_csv(OUTCSV, index=False)
say(f"\nWrote {OUTCSV} ({len(P)} rows)")

# ---- identity gate ----------------------------------------------------
say("\nIDENTITY GATE — sel_derived vs the pipeline's selectivity")
try:
    F = pd.read_csv("features_per_subject.csv"); F["subject"] = F.subject.astype(str)
    sel = next((c for c in F.columns if "mu_ERD_selectivity_T1" in c), None)
    j = P[P.task == "T1"].merge(F[["dataset", "subject", sel]],
                                on=["dataset", "subject"], how="inner").dropna(subset=["sel_derived", sel])
    r, p = stats.pearsonr(j.sel_derived, j[sel])
    say(f"  n={len(j)}  r={r:+.4f}")
    GATE = r > 0.95
    say("  GATE PASSED — PART C results are interpretable." if GATE else
        "  GATE FAILED — config still differs. Do NOT interpret what follows;\n"
        "  compare the ERD function in 03_compute_features.py line by line.")
except Exception as e:
    GATE = False
    say(f"  gate check failed: {e}")

# ---- the level test ---------------------------------------------------
if GATE:
    G = P[P.task == "T1"].drop_duplicates(subset=["dataset", "subject", "session"])
    say("\nD_mean variability by cohort")
    T = G.groupby("dataset").D_mean.agg(["count", "mean", "std"])
    T["CV"] = T["std"] / T["mean"]
    say(T.round(3).to_string())

    S = G[G.dataset.str.contains("Liu")].merge(clin, on="subject", how="inner")
    say(f"\nstroke merged: {len(S)}")
    for lab, col in [("D_mean", "D_mean"), ("dD", "dD")]:
        d = S.dropna(subset=[col, "NIHSS"])
        rho, p = stats.spearmanr(d[col], d.NIHSS)
        say(f"  {lab:8s} vs NIHSS: rho={rho:+.3f} p={p:.4f}")

    say("\n  selectivity / dLI vs NIHSS, partialled on D_mean:")
    for nm, col in [("selectivity", "sel_derived"), ("dLI", "dLI_derived")]:
        d = S.dropna(subset=[col, "NIHSS", "D_mean"])
        raw, praw = stats.spearmanr(d[col], d.NIHSS)
        rx = d[col] - np.poly1d(np.polyfit(d.D_mean, d[col], 1))(d.D_mean)
        ry = d.NIHSS - np.poly1d(np.polyfit(d.D_mean, d.NIHSS, 1))(d.D_mean)
        rr, pp = stats.spearmanr(rx, ry)
        say(f"    {nm:12s} raw rho={raw:+.3f} (p={praw:.4f})  ->  "
            f"D_mean-partial rho={rr:+.3f} (p={pp:.4f})")

    say("\n  DECISION:")
    say("   dLI strengthens, selectivity unchanged -> the LEVEL of total ERD")
    say("     amplitude is what attenuates dLI. Report the mechanism.")
    say("   neither moves -> with the drift arm already null (PART A) and the")
    say("     estimator hypothesis void, no mechanism remains. Report the")
    say("     selectivity/dLI divergence as an observation, unexplained.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
