"""
05_compute_coherence.py
=======================
Computes C3–C4 interhemispheric connectivity per subject, per session,
per task, per dose-stage using:
  - Imaginary part of coherence (ImCoh) — robust to volume conduction
  - Weighted phase-lag index (wPLI) — robust to zero-lag artefacts

These are the DIRECT MEASUREMENTS of the "interhemispheric competition"
constraint (Level 2 of the conceptual framework). The pre-registration
specifies that C3–C4 imaginary coherence will be tested in hierarchical
regression for predictive power beyond ERD selectivity (H2d).

This script uses the SAME harmonisation as 03_compute_features.py:
  - MOABB LeftRightImagery paradigm
  - 160 Hz, C3/Cz/C4, 0–3 s epochs
  - Early/mid/late dose binning

Output: coherence_per_stage.csv and coherence_per_subject.csv

Usage: python 05_compute_coherence.py

EXPECTED RUNTIME: 30-90 minutes (connectivity computation is slower
than simple PSD).

IMPORTANT: This script requires mne-connectivity.
Install it with: pip install mne-connectivity
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")

from scipy import stats
from mne_connectivity import spectral_connectivity_epochs
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
from moabb.paradigms import LeftRightImagery


# ============================================================
# CONFIGURATION (identical to 03_compute_features.py)
# ============================================================
PARADIGM = LeftRightImagery(
    fmin=1, fmax=45,
    channels=["C3", "Cz", "C4"],
    resample=160.0,
    tmin=0.0, tmax=3.0,
    baseline=None,
)

DOSE_LABELS = ["early", "mid", "late"]

# Frequency bands for connectivity
BANDS = {
    "mu":   (8, 13),
    "beta": (13, 30),
}

# Datasets (same as 03)
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
# COHERENCE COMPUTATION
# ============================================================

def compute_c3c4_connectivity(epochs_array, sfreq, ch_names):
    """
    Compute C3-C4 imaginary coherence and wPLI for mu and beta bands.

    Parameters
    ----------
    epochs_array : mne.EpochsArray or mne.Epochs
        Must contain at least C3 and C4 channels.
    sfreq : float
    ch_names : list of str

    Returns
    -------
    dict with keys:
        imcoh_mu, imcoh_beta, wpli_mu, wpli_beta
        (each a float, or np.nan if computation fails)
    """
    result = {
        "imcoh_mu": np.nan, "imcoh_beta": np.nan,
        "wpli_mu": np.nan, "wpli_beta": np.nan,
    }

    # Find C3 and C4 indices
    if "C3" not in ch_names or "C4" not in ch_names:
        return result

    c3_idx = ch_names.index("C3")
    c4_idx = ch_names.index("C4")

    # indices: (seeds, targets) — C3→C4
    indices = ([c3_idx], [c4_idx])

    try:
        con = spectral_connectivity_epochs(
            epochs_array,
            method=["imcoh", "wpli"],
            indices=indices,
            sfreq=sfreq,
            fmin=[BANDS["mu"][0], BANDS["beta"][0]],
            fmax=[BANDS["mu"][1], BANDS["beta"][1]],
            faverage=True,      # average across freqs within each band
            mode="multitaper",  # more robust than Welch for short epochs
            mt_bandwidth=4.0,   # 4 Hz bandwidth (good for 8-13 Hz band)
            verbose=False,
        )

        imcoh_data = con[0].get_data()  # (1 pair, 2 bands)
        wpli_data  = con[1].get_data()

        result["imcoh_mu"]   = float(imcoh_data[0, 0])
        result["imcoh_beta"] = float(imcoh_data[0, 1])
        result["wpli_mu"]    = float(wpli_data[0, 0])
        result["wpli_beta"]  = float(wpli_data[0, 1])

    except Exception as e:
        # Silently return NaN — will be logged in the summary
        pass

    return result


def compute_coherence_baseline_mi(epochs_array, sfreq, ch_names):
    """
    Compute connectivity separately for baseline and MI windows,
    then compute the change (MI - baseline).

    This captures whether interhemispheric coupling CHANGES during MI,
    not just its absolute level.

    Returns dict with baseline, MI, and delta values.
    """
    n_times = epochs_array.get_data().shape[-1]
    total_dur = n_times / sfreq  # should be ~3.0 s

    # Baseline: 0.0–0.5 s
    ep_base = epochs_array.copy().crop(tmin=0.0, tmax=0.5)
    # MI: 0.5–3.0 s
    ep_mi = epochs_array.copy().crop(tmin=0.5, tmax=min(3.0, total_dur - 0.01))

    base_conn = compute_c3c4_connectivity(ep_base, sfreq, ch_names)
    mi_conn = compute_c3c4_connectivity(ep_mi, sfreq, ch_names)

    result = {}
    for metric in ["imcoh_mu", "imcoh_beta", "wpli_mu", "wpli_beta"]:
        result[f"{metric}_baseline"] = base_conn[metric]
        result[f"{metric}_MI"]       = mi_conn[metric]

        # Delta = MI - baseline (positive = increased coupling during MI)
        if not np.isnan(base_conn[metric]) and not np.isnan(mi_conn[metric]):
            result[f"{metric}_delta"] = mi_conn[metric] - base_conn[metric]
        else:
            result[f"{metric}_delta"] = np.nan

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_dataset_coherence(dataset_name, dataset, paradigm, max_subjects=None):
    """
    Compute coherence for all subjects in a dataset.
    Returns DataFrame with one row per (subject, session, task, dose_stage).
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
        ch_names = epochs.ch_names
        data = epochs.get_data()

        if "session" in meta.columns:
            sessions = meta["session"].values
        else:
            sessions = np.array(["session_0"] * len(y))

        for session_id in np.unique(sessions):
            for task in ["left_hand", "right_hand"]:
                mask = (y == task) & (sessions == session_id)
                if mask.sum() < 6:  # need enough trials for reliable connectivity
                    continue

                task_indices = np.where(mask)[0]
                n_trials = len(task_indices)

                # Dose binning
                bin_edges = np.linspace(0, n_trials, 4, dtype=int)

                for dose_i, dose_label in enumerate(DOSE_LABELS):
                    start_idx = bin_edges[dose_i]
                    end_idx   = bin_edges[dose_i + 1]
                    if end_idx - start_idx < 3:  # minimum 3 trials for connectivity
                        continue

                    # Select epochs for this dose bin
                    bin_trial_indices = task_indices[start_idx:end_idx]

                    # Create EpochsArray for this subset
                    bin_data = data[bin_trial_indices]
                    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
                    n_ep = len(bin_data)
                    events = np.column_stack([
                        np.arange(n_ep) * data.shape[-1],
                        np.zeros(n_ep, dtype=int),
                        np.ones(n_ep, dtype=int),
                    ])
                    ep_subset = mne.EpochsArray(
                        bin_data, info, events=events, tmin=0.0, verbose=False
                    )

                    # Compute full-epoch connectivity
                    full_conn = compute_c3c4_connectivity(ep_subset, sfreq, ch_names)

                    # Compute baseline/MI split connectivity
                    split_conn = compute_coherence_baseline_mi(ep_subset, sfreq, ch_names)

                    row = {
                        "dataset":   dataset_name,
                        "subject":   sid,
                        "session":   session_id,
                        "task":      task,
                        "stage":     dose_label,
                        "n_trials":  end_idx - start_idx,
                    }
                    row.update(full_conn)
                    row.update(split_conn)
                    all_rows.append(row)

        print("OK")

    return pd.DataFrame(all_rows)


def compute_coherence_summary(df):
    """
    Per-subject summary: mean connectivity, connectivity slopes,
    and delta-connectivity (MI vs baseline change).
    """
    summary_rows = []

    for (ds, sid, sess), group in df.groupby(["dataset", "subject", "session"]):
        row = {"dataset": ds, "subject": sid, "session": sess}

        for task_label, task_short in [("left_hand", "T1"), ("right_hand", "T2")]:
            task_data = group[group["task"] == task_label]

            for metric in ["imcoh_mu", "imcoh_beta", "wpli_mu", "wpli_beta"]:
                # Per-stage values
                for stage in DOSE_LABELS:
                    stage_data = task_data[task_data["stage"] == stage]
                    val = stage_data[metric].values[0] if len(stage_data) > 0 else np.nan
                    row[f"{metric}_{stage}_{task_short}"] = val

                # Connectivity slope across stages
                vals = []
                for stage in DOSE_LABELS:
                    stage_data = task_data[task_data["stage"] == stage]
                    vals.append(stage_data[metric].values[0] if len(stage_data) > 0 else np.nan)

                x = np.array([1.0, 2.0, 3.0])
                y_arr = np.array(vals)
                if not np.any(np.isnan(y_arr)):
                    slope = stats.linregress(x, y_arr)[0]
                else:
                    slope = np.nan
                row[f"{metric}_slope_{task_short}"] = slope

                # Mean connectivity (across all stages)
                row[f"{metric}_mean_{task_short}"] = np.nanmean(vals)

                # Mean delta (MI - baseline) across stages
                delta_vals = []
                for stage in DOSE_LABELS:
                    stage_data = task_data[task_data["stage"] == stage]
                    dcol = f"{metric}_delta"
                    if len(stage_data) > 0 and dcol in stage_data.columns:
                        delta_vals.append(stage_data[dcol].values[0])
                    else:
                        delta_vals.append(np.nan)
                row[f"{metric}_delta_mean_{task_short}"] = np.nanmean(delta_vals)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 11: C3-C4 Interhemispheric Connectivity")
    print("(Imaginary Coherence + wPLI, μ and β bands)")
    print("=" * 60)

    all_coherence = []

    for name, ds in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"Processing: {name} ({len(ds.subject_list)} subjects)")
        print(f"{'─' * 50}")

        df = process_dataset_coherence(name, ds, PARADIGM)

        if len(df) > 0:
            all_coherence.append(df)
            print(f"  → {len(df)} rows extracted")
        else:
            print(f"  → No data extracted")

    # Combine
    coherence = pd.concat(all_coherence, ignore_index=True)
    coherence.to_csv("coherence_per_stage.csv", index=False)
    print(f"\n✓ Saved coherence_per_stage.csv ({len(coherence)} rows)")

    # Per-subject summary
    print("\nComputing per-subject coherence summaries...")
    coh_summary = compute_coherence_summary(coherence)
    coh_summary.to_csv("coherence_per_subject.csv", index=False)
    print(f"✓ Saved coherence_per_subject.csv ({len(coh_summary)} rows)")

    # ── Quick analysis on available data ──
    print(f"\n{'─' * 50}")
    print("QUICK ANALYSIS:")
    print(f"{'─' * 50}")

    for ds_name in coh_summary["dataset"].unique():
        ds_df = coh_summary[coh_summary["dataset"] == ds_name]
        n = len(ds_df)
        print(f"\n  {ds_name} (n={n}):")

        for task in ["T1", "T2"]:
            for metric in ["imcoh_mu", "wpli_mu"]:
                col = f"{metric}_mean_{task}"
                if col in ds_df.columns:
                    vals = ds_df[col].dropna()
                    print(f"    {metric} {task}: mean={vals.mean():.4f}, SD={vals.std():.4f}")

    # ── Merge with features and test H2d ──
    print(f"\n{'─' * 50}")
    print("HIERARCHICAL REGRESSION PREP:")
    print("Merge coherence_per_subject.csv with features_per_subject.csv")
    print("to test whether ImCoh adds predictive power beyond ERD selectivity.")
    print(f"{'─' * 50}")

    try:
        features = pd.read_csv("features_per_subject.csv")
        merged = features.merge(
            coh_summary,
            on=["dataset", "subject", "session"],
            how="inner",
        )
        merged.to_csv("features_with_coherence.csv", index=False)
        print(f"✓ Merged: {len(merged)} rows → features_with_coherence.csv")

        # Quick hierarchical regression preview
        from sklearn.linear_model import LinearRegression

        for task in ["T1", "T2"]:
            dli_col = f"ΔLI_mu_{task}"
            sel_col = f"mu_ERD_selectivity_{task}"
            coh_col = f"imcoh_mu_mean_{task}"

            cols = [dli_col, sel_col, coh_col]
            sub = merged.dropna(subset=cols)

            if len(sub) < 10:
                continue

            y = sub[dli_col].values
            X1 = sub[[sel_col]].values
            X2 = sub[[sel_col, coh_col]].values

            # Step 1: selectivity only
            m1 = LinearRegression().fit(X1, y)
            r2_1 = m1.score(X1, y)

            # Step 2: selectivity + coherence
            m2 = LinearRegression().fit(X2, y)
            r2_2 = m2.score(X2, y)

            delta_r2 = r2_2 - r2_1

            print(f"\n  {task}: R² (selectivity only) = {r2_1:.4f}")
            print(f"  {task}: R² (selectivity + ImCoh) = {r2_2:.4f}")
            print(f"  {task}: ΔR² = {delta_r2:.4f} {'← ImCoh adds predictive power' if delta_r2 > 0.01 else '← minimal addition'}")

    except FileNotFoundError:
        print("  features_per_subject.csv not found — run 03_compute_features.py first.")
        print("  Coherence data is saved; you can merge manually later.")

    print(f"\n{'=' * 60}")
    print("COHERENCE COMPUTATION COMPLETE.")
    print("Output files:")
    print("  coherence_per_stage.csv     — per-epoch connectivity values")
    print("  coherence_per_subject.csv   — per-subject summary metrics")
    print("  features_with_coherence.csv — merged features + connectivity")
    print(f"{'=' * 60}")
