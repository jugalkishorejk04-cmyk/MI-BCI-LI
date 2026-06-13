"""
04_analyse_results.py
=====================
Cross-cohort analysis: tests whether the three main conclusions
replicate across independent datasets.

Reads: features_per_subject.csv (output from 03_compute_features.py)
Produces: analysis_report.txt (full statistical report)

Usage: python 04_analyse_results.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import sys

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 4: Cross-cohort replication analysis")
print("=" * 60)

try:
    df = pd.read_csv("features_per_subject.csv")
    print(f"\nLoaded {len(df)} subjects from features_per_subject.csv")
except FileNotFoundError:
    print("\nERROR: features_per_subject.csv not found.")
    print("Run 03_compute_features.py first.")
    sys.exit(1)

# Summary by dataset
print(f"\n{'─' * 50}")
print("DATASET OVERVIEW:")
for ds in df["dataset"].unique():
    n = len(df[df["dataset"] == ds])
    print(f"  {ds}: {n} subjects")

# Open report file
report = open("analysis_report.txt", "w", encoding="utf-8")
def log(msg):
    print(msg)
    report.write(msg + "\n")


# ============================================================
# CONCLUSION 1 REPLICATION: ΔLI as primary marker
# ============================================================
log(f"\n{'=' * 60}")
log("CONCLUSION 1: ΔLI as primary marker — per-dataset replication")
log(f"{'=' * 60}")

for ds in df["dataset"].unique():
    ds_df = df[df["dataset"] == ds]
    n = len(ds_df)
    log(f"\n--- {ds} (n={n}) ---")

    # Responder rates
    mu_resp = ds_df["μ_responder"].sum()
    beta_resp = ds_df["β_responder"].sum()
    combined = (ds_df["μ_responder"] & ds_df["β_responder"]).sum()
    non_resp = (~ds_df["μ_responder"] & ~ds_df["β_responder"]).sum()

    log(f"  μ-responders: {mu_resp}/{n} ({100*mu_resp/n:.1f}%)")
    log(f"  β-responders: {beta_resp}/{n} ({100*beta_resp/n:.1f}%)")
    log(f"  Combined μ+β: {combined}/{n} ({100*combined/n:.1f}%)")
    log(f"  Non-responders: {non_resp}/{n} ({100*non_resp/n:.1f}%)")

    # ΔLI descriptives
    for task in ["T1", "T2"]:
        for band in ["mu", "beta"]:
            col = f"ΔLI_{band}_{task}"
            if col in ds_df.columns:
                vals = ds_df[col].dropna()
                log(f"  Δ{band.upper()}_LI_{task}: mean={vals.mean():.4f}, SD={vals.std():.4f}, n={len(vals)}")

    # Mann-Whitney resp vs non-resp
    for task in ["T1", "T2"]:
        col = f"ΔLI_mu_{task}"
        if col not in ds_df.columns:
            continue
        resp = ds_df.loc[ds_df["μ_responder"], col].dropna()
        nonresp = ds_df.loc[~ds_df["μ_responder"], col].dropna()
        if len(resp) >= 3 and len(nonresp) >= 3:
            u, p = stats.mannwhitneyu(resp, nonresp, alternative="two-sided")
            pooled_sd = np.sqrt(
                ((len(resp)-1)*resp.std()**2 + (len(nonresp)-1)*nonresp.std()**2) /
                (len(resp)+len(nonresp)-2)
            )
            d = (resp.mean() - nonresp.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            log(f"  Mann-Whitney μΔLI_{task} resp vs non-resp: U={u:.0f}, p={p:.2e}, Cohen's d={d:.3f}")


# ============================================================
# CONCLUSION 2 REPLICATION: ERD selectivity constraint
# ============================================================
log(f"\n{'=' * 60}")
log("CONCLUSION 2: ERD selectivity — per-dataset replication")
log(f"{'=' * 60}")

for ds in df["dataset"].unique():
    ds_df = df[df["dataset"] == ds]
    log(f"\n--- {ds} ---")

    for task in ["T1", "T2"]:
        sel_col = f"mu_ERD_selectivity_{task}"
        dli_col = f"ΔLI_mu_{task}"
        contra_col = f"mu_ERD_slope_contra_{task}"
        bilat_col = f"mu_ERD_slope_bilateral_{task}"

        if sel_col not in ds_df.columns or dli_col not in ds_df.columns:
            continue

        mask = ds_df[sel_col].notna() & ds_df[dli_col].notna()
        x = ds_df.loc[mask, sel_col]
        y = ds_df.loc[mask, dli_col]

        if len(x) >= 5:
            rho, p = stats.spearmanr(x, y)
            log(f"  ERD selectivity vs ΔLIμ {task}: ρ={rho:.4f}, p={p:.2e}, n={len(x)}")

        # Also contralateral ERD slope alone
        mask2 = ds_df[contra_col].notna() & ds_df[dli_col].notna()
        x2 = ds_df.loc[mask2, contra_col]
        y2 = ds_df.loc[mask2, dli_col]
        if len(x2) >= 5:
            rho2, p2 = stats.spearmanr(x2, y2)
            log(f"  Contra ERD slope vs ΔLIμ {task}: ρ={rho2:.4f}, p={p2:.2e}")

        # Bilateral ERD slope
        mask3 = ds_df[bilat_col].notna() & ds_df[dli_col].notna()
        x3 = ds_df.loc[mask3, bilat_col]
        y3 = ds_df.loc[mask3, dli_col]
        if len(x3) >= 5:
            rho3, p3 = stats.spearmanr(x3, y3)
            log(f"  Bilateral ERD slope vs ΔLIμ {task}: ρ={rho3:.4f}, p={p3:.4f}")


# ============================================================
# CONCLUSION 3 REPLICATION: Three-phase dose-response
# ============================================================
log(f"\n{'=' * 60}")
log("CONCLUSION 3: Trajectory dynamics — per-dataset replication")
log(f"{'=' * 60}")

# Load per-epoch data for trajectory analysis
try:
    epoch_df = pd.read_csv("features_per_epoch.csv")
except FileNotFoundError:
    epoch_df = None
    log("  (features_per_epoch.csv not found — skipping trajectory analysis)")

for ds in df["dataset"].unique():
    ds_df = df[df["dataset"] == ds]
    log(f"\n--- {ds} ---")

    for task in ["T1", "T2"]:
        li_early = f"LI_mu_early_{task}"
        li_mid = f"LI_mu_mid_{task}"
        li_late = f"LI_mu_late_{task}"

        if li_early not in ds_df.columns:
            continue

        mask = ds_df[li_early].notna() & ds_df[li_mid].notna() & ds_df[li_late].notna()
        sub = ds_df[mask]

        if len(sub) < 3:
            continue

        # Trajectory stability
        em = sub[li_mid] - sub[li_early]
        ml = sub[li_late] - sub[li_mid]
        stable = ((em > 0) & (ml > 0)) | ((em < 0) & (ml < 0))
        n_stable = stable.sum()
        n_total = len(stable)
        log(f"  μ-band {task}: Stable={n_stable}/{n_total} ({100*n_stable/n_total:.1f}%), Non-monotonic={n_total-n_stable} ({100*(n_total-n_stable)/n_total:.1f}%)")

        # Threshold crossing among responders
        dli_col = f"ΔLI_mu_{task}"
        resp = sub[sub[dli_col] > 0] if dli_col in sub.columns else sub.iloc[0:0]
        if len(resp) >= 3:
            em_resp = resp[li_mid] - resp[li_early]
            first_early = (em_resp > 0).sum()
            log(f"    Threshold crossing (responders n={len(resp)}): {100*first_early/len(resp):.1f}% at early→mid")


# ============================================================
# CROSS-COHORT COMPARISON: Healthy vs Stroke
# ============================================================
log(f"\n{'=' * 60}")
log("CROSS-COHORT: Healthy vs Stroke comparison")
log(f"{'=' * 60}")

healthy_datasets = ["PhysionetMI", "Cho2017", "BNCI2014_004", "Lee2019_MI"]
stroke_datasets = ["Liu2024"]

healthy = df[df["dataset"].isin(healthy_datasets)]
stroke = df[df["dataset"].isin(stroke_datasets)]

log(f"\n  Healthy cohorts: {len(healthy)} subjects")
log(f"  Stroke cohorts:  {len(stroke)} subjects")

if len(healthy) > 0 and len(stroke) > 0:
    for task in ["T1", "T2"]:
        col = f"ΔLI_mu_{task}"
        h = healthy[col].dropna()
        s = stroke[col].dropna()
        if len(h) >= 3 and len(s) >= 3:
            u, p = stats.mannwhitneyu(h, s, alternative="two-sided")
            log(f"  ΔLIμ {task}: Healthy mean={h.mean():.4f} vs Stroke mean={s.mean():.4f}, Mann-Whitney p={p:.4f}")

    # Responder rates comparison
    h_resp = healthy["μ_responder"].mean()
    s_resp = stroke["μ_responder"].mean()
    log(f"\n  μ-responder rate: Healthy={100*h_resp:.1f}% vs Stroke={100*s_resp:.1f}%")

    h_nonresp = (~healthy["μ_responder"] & ~healthy["β_responder"]).mean()
    s_nonresp = (~stroke["μ_responder"] & ~stroke["β_responder"]).mean()
    log(f"  Non-responder rate: Healthy={100*h_nonresp:.1f}% vs Stroke={100*s_nonresp:.1f}%")


# ============================================================
# BCI-IV-2b SESSION-BY-SESSION ANALYSIS
# ============================================================
log(f"\n{'=' * 60}")
log("BCI-IV-2b: Session-by-session ΔLI dynamics")
log(f"{'=' * 60}")

bnci = df[df["dataset"] == "BNCI2014_004"]
if len(bnci) > 0:
    sessions = sorted(bnci["session"].unique())
    log(f"  Sessions available: {sessions}")

    for task in ["T1", "T2"]:
        col = f"ΔLI_mu_{task}"
        log(f"\n  {task}:")
        for sess in sessions:
            vals = bnci.loc[bnci["session"] == sess, col].dropna()
            if len(vals) > 0:
                log(f"    {sess}: mean ΔLIμ = {vals.mean():.4f}, SD = {vals.std():.4f}, n={len(vals)}")
else:
    log("  No BCI-IV-2b data available.")


# ============================================================
# SUMMARY
# ============================================================
log(f"\n{'=' * 60}")
log("ANALYSIS COMPLETE")
log(f"{'=' * 60}")
log("\nFull report saved to: analysis_report.txt")
log("Feature tables saved to: features_per_epoch.csv, features_per_subject.csv")

report.close()
print(f"\n✓ All outputs saved. You can now review analysis_report.txt")
