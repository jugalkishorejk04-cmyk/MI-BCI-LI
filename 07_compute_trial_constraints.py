"""
07_compute_trial_constraints.py — MEMORY-SAFE REWRITE
"""

import gc
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")

from scipy import stats
from scipy.signal import welch, hilbert, butter, filtfilt
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Liu2024
from moabb.paradigms import LeftRightImagery

PARADIGM = LeftRightImagery(
    fmin=1, fmax=45,
    channels=["C3", "Cz", "C4"],
    resample=160.0,
    tmin=0.0, tmax=3.0,
    baseline=None,
)

DOSE_LABELS = ["early", "mid", "late"]
SFREQ = 160.0
BASELINE_SAMPLES = (0, int(0.5 * SFREQ))
MI_SAMPLES = (int(0.5 * SFREQ), int(3.0 * SFREQ))

DATASETS = {
    "PhysionetMI":  PhysionetMI(),
    "BNCI2014_004": BNCI2014_004(),
    "Liu2024":      Liu2024(),
    "Cho2017":      Cho2017(),
}
DATASETS["PhysionetMI"].subject_list = [
    s for s in DATASETS["PhysionetMI"].subject_list if s != 88
]

PARTIAL_FILE      = "trial_constraints_partial.csv"
FINAL_STAGE_FILE  = "trial_constraints_per_stage.csv"
FINAL_SUBJECT_FILE = "trial_constraints_per_subject.csv"


def compute_single_trial_erd(epoch_data, sfreq, fmin, fmax):
    n_trials, n_ch, _ = epoch_data.shape
    b_start, b_end = BASELINE_SAMPLES
    m_start, m_end = MI_SAMPLES
    erd = np.full((n_trials, n_ch), np.nan)
    for t in range(n_trials):
        for ch in range(n_ch):
            base_seg = epoch_data[t, ch, b_start:b_end]
            mi_seg   = epoch_data[t, ch, m_start:m_end]
            if len(base_seg) < 8 or len(mi_seg) < 8:
                continue
            npb = min(len(base_seg), int(sfreq))
            f_b, psd_b = welch(base_seg, fs=sfreq, nperseg=npb)
            mask_b = (f_b >= fmin) & (f_b <= fmax)
            P_base = np.mean(psd_b[mask_b]) if mask_b.any() else np.nan
            npm = min(len(mi_seg), int(sfreq))
            f_m, psd_m = welch(mi_seg, fs=sfreq, nperseg=npm)
            mask_m = (f_m >= fmin) & (f_m <= fmax)
            P_mi = np.mean(psd_m[mask_m]) if mask_m.any() else np.nan
            if P_base and P_base > 0 and P_mi and P_mi > 0:
                erd[t, ch] = 10.0 * np.log10(P_mi / P_base)
    return erd


def compute_onset_latency(epoch_data, sfreq, fmin, fmax):
    n_trials, n_ch, _ = epoch_data.shape
    b_start, b_end = BASELINE_SAMPLES
    onset = np.full((n_trials, n_ch), np.nan)
    nyq = sfreq / 2.0
    fmax_safe = min(fmax, nyq - 1.0)
    try:
        b_coef, a_coef = butter(2, [fmin / nyq, fmax_safe / nyq], btype="band")
    except Exception:
        return onset
    mi_start   = int(0.5 * sfreq)
    min_consec = 3
    for t in range(n_trials):
        for ch in range(n_ch):
            sig = epoch_data[t, ch, :]
            try:
                filtered = filtfilt(b_coef, a_coef, sig)
            except Exception:
                continue
            envelope = np.abs(hilbert(filtered))
            base_env = envelope[b_start:b_end]
            if len(base_env) < 4:
                continue
            threshold = np.mean(base_env) - np.std(base_env)
            below = envelope[mi_start:] < threshold
            for s in range(len(below) - min_consec):
                if np.all(below[s:s + min_consec]):
                    onset[t, ch] = (mi_start + s) / sfreq
                    break
    return onset


def process_one_subject(sid, dataset_name, dataset, paradigm):
    try:
        epochs, y, meta = paradigm.get_data(
            dataset, subjects=[sid], return_epochs=True
        )
    except Exception as e:
        print(f"SKIP ({e})")
        gc.collect()
        return []

    sfreq    = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    data     = epochs.get_data().copy()
    del epochs
    gc.collect()

    sessions = meta["session"].values if "session" in meta.columns \
               else np.array(["session_0"] * len(y))

    rows = []
    for session_id in np.unique(sessions):
        for task in ["left_hand", "right_hand"]:
            task_short = "T1" if task == "left_hand" else "T2"
            contra_ch  = "C4" if task == "left_hand" else "C3"
            contra_idx = ch_names.index(contra_ch) if contra_ch in ch_names else -1

            mask = (y == task) & (sessions == session_id)
            if mask.sum() < 6:
                continue

            task_data = data[mask]
            n_trials  = len(task_data)
            bin_edges = np.linspace(0, n_trials, 4, dtype=int)

            for dose_i, dose_label in enumerate(DOSE_LABELS):
                s_idx = bin_edges[dose_i]
                e_idx = bin_edges[dose_i + 1]
                if e_idx - s_idx < 3:
                    continue

                bin_data = task_data[s_idx:e_idx]

                mu_erd   = compute_single_trial_erd(bin_data, sfreq, 8, 13)
                beta_erd = compute_single_trial_erd(bin_data, sfreq, 13, 30)

                mu_cv = beta_cv = np.nan
                if contra_idx >= 0:
                    v = mu_erd[:, contra_idx]
                    v = v[~np.isnan(v)]
                    if len(v) >= 3 and np.abs(np.mean(v)) > 1e-6:
                        mu_cv = np.std(v) / np.abs(np.mean(v))
                    vb = beta_erd[:, contra_idx]
                    vb = vb[~np.isnan(vb)]
                    if len(vb) >= 3 and np.abs(np.mean(vb)) > 1e-6:
                        beta_cv = np.std(vb) / np.abs(np.mean(vb))
                del mu_erd, beta_erd
                gc.collect()

                mu_onset   = compute_onset_latency(bin_data, sfreq, 8, 13)
                beta_onset = compute_onset_latency(bin_data, sfreq, 13, 30)

                mu_om = mu_osd = beta_om = beta_osd = np.nan
                if contra_idx >= 0:
                    vo = mu_onset[:, contra_idx]
                    vo = vo[~np.isnan(vo)]
                    if len(vo) >= 3:
                        mu_om  = np.mean(vo)
                        mu_osd = np.std(vo)
                    vob = beta_onset[:, contra_idx]
                    vob = vob[~np.isnan(vob)]
                    if len(vob) >= 3:
                        beta_om  = np.mean(vob)
                        beta_osd = np.std(vob)
                del mu_onset, beta_onset
                gc.collect()

                rows.append({
                    "dataset": dataset_name, "subject": sid,
                    "session": session_id,   "task": task_short,
                    "stage":   dose_label,   "n_trials": e_idx - s_idx,
                    "mu_ERD_CV_contra":        mu_cv,
                    "beta_ERD_CV_contra":      beta_cv,
                    "mu_onset_latency_mean":   mu_om,
                    "mu_onset_latency_SD":     mu_osd,
                    "beta_onset_latency_mean": beta_om,
                    "beta_onset_latency_SD":   beta_osd,
                })

    del data
    gc.collect()
    return rows


def summarise_constraints(df):
    rows = []
    for (ds, sid, sess), grp in df.groupby(["dataset", "subject", "session"]):
        row = {"dataset": ds, "subject": sid, "session": sess}
        for task in ["T1", "T2"]:
            tg = grp[grp["task"] == task]
            for metric in ["mu_ERD_CV_contra", "beta_ERD_CV_contra",
                           "mu_onset_latency_mean", "mu_onset_latency_SD",
                           "beta_onset_latency_mean", "beta_onset_latency_SD"]:
                vals = tg[metric].dropna()
                row[f"{metric}_mean_{task}"] = vals.mean() if len(vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 12: Trial-Level Constraints (memory-safe version)")
    print("  (ii)  Signal stability  — ERD CV")
    print("  (iii) Temporal precision — ERD onset latency SD")
    print("=" * 60)

    if os.path.exists(PARTIAL_FILE):
        existing      = pd.read_csv(PARTIAL_FILE)
        done_datasets = set(existing["dataset"].unique())
        all_data      = [existing]
        print(f"Resuming — already done: {done_datasets}")
    else:
        all_data      = []
        done_datasets = set()

    for ds_name, ds in DATASETS.items():
        if ds_name in done_datasets:
            print(f"\nSkipping {ds_name} (already complete)")
            continue

        print(f"\n{'─' * 50}")
        print(f"Processing: {ds_name} ({len(ds.subject_list)} subjects)")
        print(f"{'─' * 50}")

        ds_rows = []
        for sid in ds.subject_list:
            print(f"    Subject {sid}...", end=" ", flush=True)
            subject_rows = process_one_subject(sid, ds_name, ds, PARADIGM)
            if subject_rows:
                ds_rows.extend(subject_rows)
                print("OK")
            else:
                print("SKIP")
            gc.collect()

        if ds_rows:
            ds_df = pd.DataFrame(ds_rows)
            all_data.append(ds_df)
            pd.concat(all_data, ignore_index=True).to_csv(PARTIAL_FILE, index=False)
            print(f"  -> {len(ds_df)} rows saved to {PARTIAL_FILE}")

        gc.collect()

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(FINAL_STAGE_FILE, index=False)
        print(f"\n✓ {FINAL_STAGE_FILE} ({len(combined)} rows)")

        summary = summarise_constraints(combined)
        summary.to_csv(FINAL_SUBJECT_FILE, index=False)
        print(f"✓ {FINAL_SUBJECT_FILE} ({len(summary)} rows)")

        try:
            features = pd.read_csv("features_per_subject.csv")
            merged = features.merge(
                summary, on=["dataset", "subject", "session"], how="inner"
            )
            merged.to_csv("features_with_all_constraints.csv", index=False)
            print(f"✓ features_with_all_constraints.csv ({len(merged)} rows)")

            print(f"\n{'─' * 50}")
            print("CONSTRAINT-ΔLI CORRELATIONS:")
            for task in ["T1", "T2"]:
                dli = f"ΔLI_mu_{task}"
                for name, col in [
                    ("ERD CV",   f"mu_ERD_CV_contra_mean_{task}"),
                    ("Onset SD", f"mu_onset_latency_SD_mean_{task}"),
                ]:
                    mask = merged[dli].notna() & merged[col].notna()
                    if mask.sum() >= 10:
                        rho, p = stats.spearmanr(
                            merged.loc[mask, col],
                            merged.loc[mask, dli]
                        )
                        print(f"  {name} vs ΔLIμ {task}: "
                              f"ρ={rho:.4f}, p={p:.4f}, n={mask.sum()}")
        except FileNotFoundError:
            print("  features_per_subject.csv not found")

    print(f"\n{'=' * 60}")
    print("COMPLETE.")
    print(f"{'=' * 60}")