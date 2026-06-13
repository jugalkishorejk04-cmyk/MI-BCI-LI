"""
03_compute_features.py
======================
Unified feature extraction pipeline for all datasets.

One function — compute_features() — runs identically on every dataset
and produces a single DataFrame with identical columns:

  subject, dataset, session, task, stage,
  mu_ERD_C3, mu_ERD_Cz, mu_ERD_C4,
  beta_ERD_C3, beta_ERD_Cz, beta_ERD_C4,
  LI_mu, LI_beta (task-relative: positive = contralateral dominance)

Then computes per-subject summary metrics:
  ΔLI_mu_T1, ΔLI_mu_T2, ΔLI_beta_T1, ΔLI_beta_T2,
  μ_responder, β_responder, ERD_selectivity, etc.

Usage: python 03_compute_features.py

EXPECTED RUNTIME: 1-3 hours depending on your computer.
Output: features_per_epoch.csv and features_per_subject.csv in the
        current directory.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")

from scipy import stats
from scipy.signal import welch
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
from moabb.paradigms import LeftRightImagery


# ============================================================
# CONFIGURATION
# ============================================================
PARADIGM = LeftRightImagery(
    fmin=1, fmax=45,
    channels=["C3", "Cz", "C4"],
    resample=160.0,
    tmin=0.0, tmax=3.0,
    baseline=None,
)

# Baseline and MI windows (in seconds relative to epoch start)
BASELINE_WINDOW = (0.0, 0.5)     # first 0.5 s = pre-movement baseline
MI_WINDOW       = (0.5, 3.0)     # 0.5-3.0 s = motor imagery period

# Dose labels
DOSE_LABELS = ["early", "mid", "late"]

# Datasets
DATASETS = {
    "PhysionetMI":  PhysionetMI(),
    "Cho2017":      Cho2017(),
    "BNCI2014_004": BNCI2014_004(),
    "Lee2019_MI":   Lee2019_MI(),
    "Liu2024":      Liu2024(),
}
DATASETS["PhysionetMI"].subject_list = [
    s for s in DATASETS["PhysionetMI"].subject_list if s != 88
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_erd_db(epoch_data, sfreq, fmin, fmax, baseline_win, mi_win):
    """
    Compute ERD in dB for a single epoch.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)
    sfreq : float
    fmin, fmax : float — band limits
    baseline_win : tuple (t_start, t_end) in seconds
    mi_win : tuple (t_start, t_end) in seconds

    Returns
    -------
    erd_db : ndarray, shape (n_channels,) — ERD in dB per channel
    """
    n_ch, n_times = epoch_data.shape

    # Convert time windows to sample indices
    b_start = int(baseline_win[0] * sfreq)
    b_end   = int(baseline_win[1] * sfreq)
    m_start = int(mi_win[0] * sfreq)
    m_end   = int(mi_win[1] * sfreq)

    erd_db = np.full(n_ch, np.nan)

    for ch in range(n_ch):
        # Baseline PSD
        base_seg = epoch_data[ch, b_start:b_end]
        if len(base_seg) < 8:
            continue
        nperseg_b = min(len(base_seg), int(sfreq))
        f_b, psd_b = welch(base_seg, fs=sfreq, nperseg=nperseg_b)
        mask_b = (f_b >= fmin) & (f_b <= fmax)
        P_base = np.mean(psd_b[mask_b]) if mask_b.any() else np.nan

        # MI PSD
        mi_seg = epoch_data[ch, m_start:m_end]
        if len(mi_seg) < 8:
            continue
        nperseg_m = min(len(mi_seg), int(sfreq))
        f_m, psd_m = welch(mi_seg, fs=sfreq, nperseg=nperseg_m)
        mask_m = (f_m >= fmin) & (f_m <= fmax)
        P_mi = np.mean(psd_m[mask_m]) if mask_m.any() else np.nan

        # ERD in dB
        if P_base > 0 and P_mi > 0:
            erd_db[ch] = 10.0 * np.log10(P_mi / P_base)

    return erd_db


def compute_li_task_relative(erd_db_dict, task):
    """
    Task-relative Laterality Index.
    Positive = contralateral dominance (the correct neurophysiological direction).

    Parameters
    ----------
    erd_db_dict : dict with keys 'C3', 'C4' → ERD values
    task : str, 'left_hand' or 'right_hand'

    Returns
    -------
    li : float
    """
    c3 = abs(erd_db_dict.get("C3", np.nan))
    c4 = abs(erd_db_dict.get("C4", np.nan))

    if np.isnan(c3) or np.isnan(c4):
        return np.nan

    denom = c3 + c4
    if denom < 1e-12:
        return np.nan

    if task == "left_hand":
        # Contralateral = C4 (right hemisphere)
        return (c4 - c3) / denom
    else:
        # Contralateral = C3 (left hemisphere)
        return (c3 - c4) / denom


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def compute_features(dataset_name, dataset, paradigm, max_subjects=None):
    """
    Extract features for all subjects in a dataset.

    Returns a DataFrame with one row per (subject, session, task, dose_stage).
    """
    subjects = dataset.subject_list
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    all_rows = []

    for sid in subjects:
        print(f"    Subject {sid}...", end=" ", flush=True)

        try:
            epochs, y, meta = paradigm.get_data(
                dataset, subjects=[sid], return_epochs=True
            )
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        sfreq = epochs.info["sfreq"]
        ch_names = epochs.ch_names  # Should be ['C3', 'Cz', 'C4']
        data = epochs.get_data()    # (n_epochs, n_ch, n_times)

        # Determine session from metadata if available
        if "session" in meta.columns:
            sessions = meta["session"].values
        else:
            sessions = np.array(["session_0"] * len(y))

        # Process each (session × task) combination
        for session_id in np.unique(sessions):
            for task in ["left_hand", "right_hand"]:
                # Select epochs for this session + task
                mask = (y == task) & (sessions == session_id)
                if mask.sum() < 3:
                    continue

                task_data = data[mask]
                n_trials = len(task_data)

                # Dose binning: split into early/mid/late thirds
                idx = np.arange(n_trials)
                bin_edges = np.linspace(0, n_trials, 4, dtype=int)

                for dose_i, dose_label in enumerate(DOSE_LABELS):
                    start_idx = bin_edges[dose_i]
                    end_idx   = bin_edges[dose_i + 1]
                    if start_idx >= end_idx:
                        continue

                    bin_data = task_data[start_idx:end_idx]

                    # Compute mean ERD across epochs in this bin
                    mu_erds = []
                    beta_erds = []
                    for ep in bin_data:
                        mu_erds.append(
                            compute_erd_db(ep, sfreq, 8, 13, BASELINE_WINDOW, MI_WINDOW)
                        )
                        beta_erds.append(
                            compute_erd_db(ep, sfreq, 13, 30, BASELINE_WINDOW, MI_WINDOW)
                        )

                    mu_mean = np.nanmean(mu_erds, axis=0)
                    beta_mean = np.nanmean(beta_erds, axis=0)

                    # Build ERD dict
                    mu_dict = {ch: mu_mean[i] for i, ch in enumerate(ch_names)}
                    beta_dict = {ch: beta_mean[i] for i, ch in enumerate(ch_names)}

                    # Compute task-relative LI
                    li_mu = compute_li_task_relative(mu_dict, task)
                    li_beta = compute_li_task_relative(beta_dict, task)

                    row = {
                        "dataset":    dataset_name,
                        "subject":    sid,
                        "session":    session_id,
                        "task":       task,
                        "stage":      dose_label,
                        "n_trials":   end_idx - start_idx,
                        "mu_ERD_C3":  mu_dict.get("C3", np.nan),
                        "mu_ERD_Cz":  mu_dict.get("Cz", np.nan),
                        "mu_ERD_C4":  mu_dict.get("C4", np.nan),
                        "beta_ERD_C3": beta_dict.get("C3", np.nan),
                        "beta_ERD_Cz": beta_dict.get("Cz", np.nan),
                        "beta_ERD_C4": beta_dict.get("C4", np.nan),
                        "LI_mu":      li_mu,
                        "LI_beta":    li_beta,
                    }
                    all_rows.append(row)

        print("OK")

    return pd.DataFrame(all_rows)


def compute_subject_summary(df):
    """
    From the per-stage feature table, compute per-subject summary metrics:
    ΔLI, responder status, ERD slopes, ERD selectivity.
    """
    summary_rows = []

    for (ds, sid, sess), group in df.groupby(["dataset", "subject", "session"]):
        row = {"dataset": ds, "subject": sid, "session": sess}

        for task_label, task_short in [("left_hand", "T1"), ("right_hand", "T2")]:
            task_data = group[group["task"] == task_label]

            for band in ["mu", "beta"]:
                li_col = f"LI_{band}"

                # Get early and late LI
                early = task_data.loc[task_data["stage"] == "early", li_col]
                mid   = task_data.loc[task_data["stage"] == "mid",   li_col]
                late  = task_data.loc[task_data["stage"] == "late",  li_col]

                li_early = early.values[0] if len(early) > 0 else np.nan
                li_mid   = mid.values[0]   if len(mid) > 0   else np.nan
                li_late  = late.values[0]   if len(late) > 0  else np.nan

                # ΔLI = late - early
                dli = li_late - li_early if not (np.isnan(li_late) or np.isnan(li_early)) else np.nan

                row[f"LI_{band}_early_{task_short}"] = li_early
                row[f"LI_{band}_mid_{task_short}"]   = li_mid
                row[f"LI_{band}_late_{task_short}"]   = li_late
                row[f"ΔLI_{band}_{task_short}"]       = dli

            # ERD slopes (contralateral channel for this task)
            contra_ch = "C4" if task_label == "left_hand" else "C3"
            ipsi_ch   = "C3" if task_label == "left_hand" else "C4"

            for band in ["mu", "beta"]:
                erd_col_contra = f"{band}_ERD_{contra_ch}"
                erd_col_ipsi   = f"{band}_ERD_{ipsi_ch}"

                contra_vals = []
                ipsi_vals = []
                bilateral_vals = []

                for stage_i, stage_label in enumerate(DOSE_LABELS):
                    stage_data = task_data[task_data["stage"] == stage_label]
                    if len(stage_data) == 0:
                        contra_vals.append(np.nan)
                        ipsi_vals.append(np.nan)
                        bilateral_vals.append(np.nan)
                    else:
                        c = abs(stage_data[erd_col_contra].values[0])
                        ip = abs(stage_data[erd_col_ipsi].values[0])
                        contra_vals.append(c)
                        ipsi_vals.append(ip)
                        bilateral_vals.append((c + ip) / 2)

                x = np.array([1.0, 2.0, 3.0])

                # Contralateral ERD slope
                y_contra = np.array(contra_vals)
                if not np.any(np.isnan(y_contra)):
                    slope_c = stats.linregress(x, y_contra)[0]
                else:
                    slope_c = np.nan

                # Bilateral ERD slope
                y_bi = np.array(bilateral_vals)
                if not np.any(np.isnan(y_bi)):
                    slope_b = stats.linregress(x, y_bi)[0]
                else:
                    slope_b = np.nan

                # ERD selectivity = contra slope - bilateral slope
                selectivity = slope_c - slope_b if not (np.isnan(slope_c) or np.isnan(slope_b)) else np.nan

                row[f"{band}_ERD_slope_contra_{task_short}"]   = slope_c
                row[f"{band}_ERD_slope_bilateral_{task_short}"] = slope_b
                row[f"{band}_ERD_selectivity_{task_short}"]     = selectivity

        # Responder classification (ΔLI > 0 in at least one task)
        dli_mu_t1 = row.get("ΔLI_mu_T1", np.nan)
        dli_mu_t2 = row.get("ΔLI_mu_T2", np.nan)
        dli_beta_t1 = row.get("ΔLI_beta_T1", np.nan)
        dli_beta_t2 = row.get("ΔLI_beta_T2", np.nan)

        row["μ_responder"] = (
            (dli_mu_t1 > 0 if not np.isnan(dli_mu_t1) else False) or
            (dli_mu_t2 > 0 if not np.isnan(dli_mu_t2) else False)
        )
        row["β_responder"] = (
            (dli_beta_t1 > 0 if not np.isnan(dli_beta_t1) else False) or
            (dli_beta_t2 > 0 if not np.isnan(dli_beta_t2) else False)
        )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: Computing features across all datasets")
    print("=" * 60)

    all_features = []

    for name, ds in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"Processing: {name} ({len(ds.subject_list)} subjects)")
        print(f"{'─' * 50}")

        df = compute_features(name, ds, PARADIGM)

        if len(df) > 0:
            all_features.append(df)
            print(f"  → {len(df)} rows extracted")
        else:
            print(f"  → No data extracted (check errors above)")

    # Combine all datasets
    features = pd.concat(all_features, ignore_index=True)

    # Save per-epoch features
    features.to_csv("features_per_epoch.csv", index=False)
    print(f"\n✓ Saved features_per_epoch.csv ({len(features)} rows)")

    # Compute per-subject summaries
    print("\nComputing per-subject summaries...")
    summary = compute_subject_summary(features)
    summary.to_csv("features_per_subject.csv", index=False)
    print(f"✓ Saved features_per_subject.csv ({len(summary)} rows)")

    # Quick sanity check
    print(f"\n{'─' * 50}")
    print("SANITY CHECK:")
    print(f"{'─' * 50}")
    for ds_name in summary["dataset"].unique():
        ds_data = summary[summary["dataset"] == ds_name]
        n = len(ds_data)
        mu_resp = ds_data["μ_responder"].sum()
        print(f"  {ds_name}: {n} subjects, {mu_resp} μ-responders ({100*mu_resp/n:.1f}%)")

    print(f"\n{'=' * 60}")
    print("FEATURE EXTRACTION COMPLETE.")
    print("Next run: python 04_analyse_results.py")
    print(f"{'=' * 60}")
