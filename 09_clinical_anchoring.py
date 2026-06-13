"""
09_clinical_anchoring.py
========================
Correlates EEG-derived neuroplasticity metrics (ΔLI, ERD selectivity)
with clinical severity scores from Liu et al. 2024 stroke dataset.

Clinical scores available per subject:
  - NIHSS (National Institutes of Health Stroke Scale)
  - MBI (Modified Barthel Index)
  - mRS (modified Rankin Scale)
  - Age, sex, days post-stroke, paralysis side

This tests H4c: baseline ΔLI or early-session ERD selectivity
correlates with clinical severity at |ρ| ≥ 0.30.

BEFORE RUNNING: Download participants.tsv from the Liu 2024 figshare
repository (https://figshare.com/articles/dataset/21679035) and place
it in this folder.

Reads: features_per_subject.csv + participants.tsv
Output: clinical_anchoring_report.txt

Usage: python 09_clinical_anchoring.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import os, sys

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 17: Clinical Anchoring — Liu 2024 Stroke Cohort")
print("=" * 60)

# Load EEG features
try:
    feat = pd.read_csv("features_per_subject.csv")
    liu = feat[feat["dataset"] == "Liu2024"].copy()
    print(f"Loaded {len(liu)} Liu2024 subjects from features_per_subject.csv")
except FileNotFoundError:
    print("ERROR: features_per_subject.csv not found. Run 03 first.")
    sys.exit(1)

# Load clinical metadata
# Try several possible filenames
clinical_files = [
    "participants.tsv",
    "Liu2024_participants.tsv",
    "participants.csv",
]

clinical = None
for fname in clinical_files:
    if os.path.exists(fname):
        sep = "\t" if fname.endswith(".tsv") else ","
        clinical = pd.read_csv(fname, sep=sep)
        print(f"Loaded clinical metadata from {fname} ({len(clinical)} rows)")
        break

if clinical is None:
    print("\nERROR: Clinical metadata file not found.")
    print("Download participants.tsv from:")
    print("  https://figshare.com/articles/dataset/21679035")
    print("Place it in this folder and re-run.")
    sys.exit(1)

# Display available columns
print(f"\nClinical columns: {list(clinical.columns)}")

# ============================================================
# MERGE EEG + CLINICAL
# ============================================================
# Standardise subject IDs for merging
# Liu2024 in MOABB uses integer IDs (1-50)
# participants.tsv may use 'sub-01' format
if "participant_id" in clinical.columns:
    clinical["subject"] = clinical["participant_id"].str.extract(r"(\d+)").astype(int)
elif "subject" not in clinical.columns:
    # Try first column
    first_col = clinical.columns[0]
    try:
        clinical["subject"] = clinical[first_col].str.extract(r"(\d+)").astype(int)
    except Exception:
        clinical["subject"] = range(1, len(clinical) + 1)

merged = liu.merge(clinical, on="subject", how="inner")
print(f"Merged: {len(merged)} subjects with both EEG and clinical data")

# Identify clinical score columns
score_candidates = {
    "NIHSS": ["NIHSS", "nihss", "NIHSS_score", "nihss_score"],
    "MBI": ["MBI", "mbi", "MBI_score", "barthel"],
    "mRS": ["mRS", "mrs", "mRS_score", "rankin"],
    "age": ["age", "Age"],
    "days_post_stroke": ["days_post_stroke", "days_since_stroke", "onset_days"],
}

score_cols = {}
for score_name, candidates in score_candidates.items():
    for cand in candidates:
        if cand in merged.columns:
            score_cols[score_name] = cand
            break

print(f"\nIdentified clinical columns: {score_cols}")

report = open("clinical_anchoring_report.txt", "w", encoding="utf-8")
def log(msg):
    print(msg)
    report.write(msg + "\n")

# ============================================================
# CORRELATION ANALYSES
# ============================================================
log(f"\n{'=' * 60}")
log("CLINICAL CORRELATIONS")
log(f"{'=' * 60}")

eeg_metrics = {}
for task in ["T1", "T2"]:
    eeg_metrics[f"ΔLI_mu_{task}"] = f"ΔLI μ-band {task}"
    eeg_metrics[f"ΔLI_beta_{task}"] = f"ΔLI β-band {task}"
    eeg_metrics[f"mu_ERD_selectivity_{task}"] = f"ERD selectivity {task}"
    eeg_metrics[f"LI_mu_early_{task}"] = f"Baseline LI μ {task}"

for score_name, score_col in score_cols.items():
    log(f"\n{'─' * 40}")
    log(f"Clinical score: {score_name} ({score_col})")
    log(f"{'─' * 40}")

    score_vals = pd.to_numeric(merged[score_col], errors="coerce")
    n_valid = score_vals.notna().sum()
    log(f"  Valid values: {n_valid}, mean={score_vals.mean():.2f}, SD={score_vals.std():.2f}")

    for eeg_col, eeg_label in eeg_metrics.items():
        if eeg_col not in merged.columns:
            continue
        mask = score_vals.notna() & merged[eeg_col].notna()
        if mask.sum() < 5:
            continue

        rho, p = stats.spearmanr(score_vals[mask], merged.loc[mask, eeg_col])
        passes = abs(rho) >= 0.30 and p < 0.05
        log(f"  {eeg_label:30s}: ρ={rho:.4f}, p={p:.4f}, n={mask.sum()} {'✓ H4c' if passes else ''}")

# ============================================================
# STRATIFIED ANALYSIS BY HEMIPLEGIA SIDE
# ============================================================
log(f"\n{'=' * 60}")
log("STRATIFICATION BY HEMIPLEGIA SIDE")
log(f"{'=' * 60}")

side_candidates = ["paralysis_side", "hemiplegia", "affected_side", "lesion_side"]
side_col = None
for cand in side_candidates:
    if cand in merged.columns:
        side_col = cand
        break

if side_col:
    sides = merged[side_col].unique()
    log(f"Hemiplegia sides: {sides} (column: {side_col})")

    for side in sides:
        side_data = merged[merged[side_col] == side]
        n = len(side_data)
        if n < 3:
            continue

        log(f"\n  --- {side} hemiplegia (n={n}) ---")
        mu_resp = side_data["μ_responder"].sum()
        log(f"    μ-responder rate: {mu_resp}/{n} ({100*mu_resp/n:.1f}%)")

        for task in ["T1", "T2"]:
            col = f"ΔLI_mu_{task}"
            if col in side_data.columns:
                vals = side_data[col].dropna()
                log(f"    ΔLI μ {task}: mean={vals.mean():.4f}, SD={vals.std():.4f}")
else:
    log("  Hemiplegia side column not found in clinical data.")

# ============================================================
# SENSITIVITY ANALYSIS: EXCLUDING HIGH-ARTEFACT SUBJECTS
# ============================================================
log(f"\n{'=' * 60}")
log("SENSITIVITY ANALYSIS: Excluding high-artefact subjects")
log(f"{'=' * 60}")

artefact_subjects = [4, 5, 13, 14, 18, 24, 28, 33, 42, 43, 47, 48, 49]
clean = merged[~merged["subject"].isin(artefact_subjects)]
log(f"After exclusion: {len(clean)} subjects (removed {len(merged)-len(clean)})")

clean_resp = clean["μ_responder"].sum()
log(f"μ-responder rate (clean): {clean_resp}/{len(clean)} ({100*clean_resp/len(clean):.1f}%)")

for score_name, score_col in score_cols.items():
    score_vals = pd.to_numeric(clean[score_col], errors="coerce")
    for task in ["T1", "T2"]:
        dli_col = f"ΔLI_mu_{task}"
        mask = score_vals.notna() & clean[dli_col].notna()
        if mask.sum() >= 5:
            rho, p = stats.spearmanr(score_vals[mask], clean.loc[mask, dli_col])
            log(f"  {score_name} vs ΔLI μ {task} (clean): ρ={rho:.4f}, p={p:.4f}")

report.close()
print(f"\n✓ Saved clinical_anchoring_report.txt")
print("=" * 60)
