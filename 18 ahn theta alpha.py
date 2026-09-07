"""
18_ahn_theta_alpha.py
---------------------
Computes the Ahn et al. (2013) theta/alpha power ratio as a second published
comparator alongside the Blankertz SMR predictor.

Ahn M, Cho H, Ahn S, Jun SC. High Theta and Low Alpha Powers May Be Indicative
of BCI-Illiteracy in Motor Imagery. PLoS ONE 2013;8(11):e80886.

Ahn's finding: BCI-illiterate users show HIGH theta (4-8 Hz) and LOW alpha
(8-13 Hz) power at rest. The ratio theta/alpha should therefore correlate
NEGATIVELY with decoding accuracy.

This reuses the SAME rest segments already extracted for the Blankertz
predictor (script 12/17), so no new epoching is required — only a second
spectral summary of segments you already have.

OUTPUT: ahn_theta_alpha_per_subject.csv
  dataset, subject, session, rest_seconds, theta_pow, alpha_pow, ahn_ratio,
  rest_provenance
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import mne
mne.set_log_level("ERROR")

from moabb.datasets import PhysionetMI, Cho2017, Lee2019_MI, BNCI2014_004, Liu2024
from moabb.paradigms import LeftRightImagery

# ----------------------------------------------------------------------
# CONFIG — adapt these two to match your script 12/17 rest extraction
# ----------------------------------------------------------------------
OUT = "ahn_theta_alpha_per_subject.csv"
SFREQ = 160.0
THETA = (4.0, 8.0)
ALPHA = (8.0, 13.0)
# Channels: Ahn used a broad sensorimotor/parietal set. With the harmonised
# 3-channel set we use C3/Cz/C4; where the full montage is available the
# script will use all EEG channels and record which was used.
USE_FULL_MONTAGE = True

DATASETS = [
    ("PhysionetMI", PhysionetMI()),
    ("Cho2017",     Cho2017()),
    ("BNCI2014_004", BNCI2014_004()),
    ("Lee2019_MI",  Lee2019_MI()),
    ("Liu2024",     Liu2024()),
]


def rest_segment(epochs, dataset_name):
    """
    Return (data, provenance_string).

    IMPORTANT: replace the body of this function with the SAME rest-segment
    logic used in your Blankertz script, so the two comparators are computed
    on identical data. The fallback below concatenates pre-cue baseline
    windows, which is what the current Blankertz file used — keep it only if
    that is what you want to compare against.
    """
    X = epochs.get_data()                      # (n_trials, n_ch, n_times)
    n_base = int(0.5 * SFREQ)
    base = X[:, :, :n_base]                    # pre-cue baseline
    data = np.concatenate([base[i] for i in range(base.shape[0])], axis=-1)
    prov = "concatenated pre-cue baseline windows (0.0-0.5 s per trial)"
    return data, prov


def band_power(data, sf, lo, hi):
    """Welch PSD band power, averaged across channels."""
    nper = int(min(data.shape[-1], sf * 2))    # 2 s window -> 0.5 Hz resolution
    if nper < sf:                              # too short for a useful estimate
        nper = int(data.shape[-1])
    f, pxx = signal.welch(data, fs=sf, nperseg=nper, axis=-1)
    m = (f >= lo) & (f < hi)
    return float(np.mean(np.trapezoid(pxx[..., m], f[m], axis=-1)))


rows = []
for name, ds in DATASETS:
    print(f"\n=== {name} ===")
    para = LeftRightImagery(fmin=1, fmax=45, resample=SFREQ,
                            tmin=0, tmax=3, baseline=None)
    try:
        subjects = ds.subject_list
    except Exception as e:
        print(f"  cannot list subjects: {e}")
        continue

    for s in subjects:
        try:
            ep, y, meta = para.get_data(ds, subjects=[s], return_epochs=True)
        except Exception as e:
            print(f"  subj {s}: load failed ({e})")
            continue

        if not USE_FULL_MONTAGE:
            picks = [c for c in ["C3", "Cz", "C4"] if c in ep.ch_names]
            ep = ep.copy().pick(picks)

        sessions = meta["session"].unique() if "session" in meta else ["0"]
        for sess in sessions:
            if "session" in meta:
                idx = np.where(meta["session"].values == sess)[0]
                sub_ep = ep[idx]
            else:
                sub_ep = ep

            data, prov = rest_segment(sub_ep, name)
            dur = data.shape[-1] / SFREQ
            th = band_power(data, SFREQ, *THETA)
            al = band_power(data, SFREQ, *ALPHA)
            if al <= 0:
                ratio = np.nan
            else:
                ratio = th / al

            rows.append(dict(dataset=name, subject=s, session=str(sess),
                             rest_seconds=round(dur, 2),
                             n_channels=len(sub_ep.ch_names),
                             theta_pow=th, alpha_pow=al, ahn_ratio=ratio,
                             rest_provenance=prov))
        print(f"  subj {s}: ok")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nWrote {OUT}  ({len(df)} rows)")
print(df.groupby("dataset")[["ahn_ratio", "rest_seconds"]].describe().round(3))
