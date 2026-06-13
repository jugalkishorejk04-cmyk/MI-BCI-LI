"""
08_dose_response_models.py
==========================
Fits three dose-response models to BCI-IV-2b's 5-session ΔLI
trajectories per subject:

  (a) Linear:                ΔLI = a + b × session
  (b) Saturation-exponential: ΔLI = a × (1 − exp(−b × session))
  (c) Sigmoidal:             ΔLI = a / (1 + exp(−b × (session − c)))

Model selection by BIC. ΔBIC ≥ 2 favours the more complex model.

Reads: features_per_subject.csv
Output: dose_response_report.txt, dose_response_fits.csv

Usage: python 08_dose_response_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import sys

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 14: Non-Linear Dose-Response Model Comparison")
print("(BCI-IV-2b, 5 sessions across different days)")
print("=" * 60)

try:
    feat = pd.read_csv("features_per_subject.csv")
except FileNotFoundError:
    print("ERROR: features_per_subject.csv not found.")
    sys.exit(1)

bnci = feat[feat["dataset"] == "BNCI2014_004"]
if len(bnci) == 0:
    print("ERROR: No BCI-IV-2b data found.")
    sys.exit(1)

sessions = sorted(bnci["session"].unique())
n_sessions = len(sessions)
session_map = {s: i + 1 for i, s in enumerate(sessions)}

print(f"Sessions: {sessions}")
print(f"Subjects: {bnci['subject'].nunique()}")

report = open("dose_response_report.txt", "w", encoding="utf-8")
def log(msg):
    print(msg)
    report.write(msg + "\n")


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def linear_model(x, a, b):
    return a + b * x

def saturation_exp(x, a, b):
    return a * (1 - np.exp(-b * x))

def sigmoidal(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def compute_bic(n, k, rss):
    """BIC = n × ln(RSS/n) + k × ln(n)"""
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


# ============================================================
# FIT MODELS PER SUBJECT
# ============================================================

all_fits = []

for task in ["T1", "T2"]:
    dli_col = f"ΔLI_mu_{task}"
    log(f"\n{'─' * 50}")
    log(f"Task: {task}")
    log(f"{'─' * 50}")

    for sid in sorted(bnci["subject"].unique()):
        sub = bnci[bnci["subject"] == sid]

        # Build session × ΔLI vector
        x_vals = []
        y_vals = []
        for sess in sessions:
            row = sub[sub["session"] == sess]
            if len(row) > 0 and pd.notna(row[dli_col].values[0]):
                x_vals.append(session_map[sess])
                y_vals.append(row[dli_col].values[0])

        if len(x_vals) < 4:  # need at least 4 points for 3-param model
            continue

        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)
        n = len(x)

        results = {"subject": sid, "task": task, "n_sessions": n}

        # Fit linear (2 params)
        try:
            popt_lin, _ = curve_fit(linear_model, x, y, p0=[0, 0.1], maxfev=5000)
            y_pred_lin = linear_model(x, *popt_lin)
            rss_lin = np.sum((y - y_pred_lin) ** 2)
            bic_lin = compute_bic(n, 2, rss_lin)
            results["linear_a"] = popt_lin[0]
            results["linear_b"] = popt_lin[1]
            results["linear_rss"] = rss_lin
            results["linear_bic"] = bic_lin
        except Exception:
            bic_lin = np.inf
            results["linear_bic"] = np.inf

        # Fit saturation-exponential (2 params)
        try:
            popt_sat, _ = curve_fit(saturation_exp, x, y, p0=[0.5, 0.5],
                                     maxfev=5000, bounds=([-5, 0.01], [5, 10]))
            y_pred_sat = saturation_exp(x, *popt_sat)
            rss_sat = np.sum((y - y_pred_sat) ** 2)
            bic_sat = compute_bic(n, 2, rss_sat)
            results["satexp_a"] = popt_sat[0]
            results["satexp_b"] = popt_sat[1]
            results["satexp_rss"] = rss_sat
            results["satexp_bic"] = bic_sat
        except Exception:
            bic_sat = np.inf
            results["satexp_bic"] = np.inf

        # Fit sigmoidal (3 params)
        try:
            popt_sig, _ = curve_fit(sigmoidal, x, y, p0=[0.5, 1.0, 3.0],
                                     maxfev=5000, bounds=([-5, 0.01, 0.5], [5, 10, 6]))
            y_pred_sig = sigmoidal(x, *popt_sig)
            rss_sig = np.sum((y - y_pred_sig) ** 2)
            bic_sig = compute_bic(n, 3, rss_sig)
            results["sigmoid_a"] = popt_sig[0]
            results["sigmoid_b"] = popt_sig[1]
            results["sigmoid_c"] = popt_sig[2]
            results["sigmoid_rss"] = rss_sig
            results["sigmoid_bic"] = bic_sig
        except Exception:
            bic_sig = np.inf
            results["sigmoid_bic"] = np.inf

        # Best model by BIC
        bics = {"linear": bic_lin, "saturation_exp": bic_sat, "sigmoidal": bic_sig}
        best = min(bics, key=bics.get)
        results["best_model"] = best
        results["delta_bic_nonlin"] = bic_lin - min(bic_sat, bic_sig)

        all_fits.append(results)

        log(f"  S{sid}: BIC lin={bic_lin:.2f}, sat={bic_sat:.2f}, sig={bic_sig:.2f} → best={best} (ΔBIC={results['delta_bic_nonlin']:.2f})")


# ============================================================
# SUMMARY
# ============================================================
fits_df = pd.DataFrame(all_fits)
fits_df.to_csv("dose_response_fits.csv", index=False)

log(f"\n{'=' * 60}")
log("SUMMARY")
log(f"{'=' * 60}")

for task in ["T1", "T2"]:
    task_fits = fits_df[fits_df["task"] == task]
    n = len(task_fits)
    if n == 0:
        continue

    log(f"\n{task} (n={n} subjects):")

    for model in ["linear", "saturation_exp", "sigmoidal"]:
        count = (task_fits["best_model"] == model).sum()
        log(f"  Best model = {model}: {count}/{n} ({100*count/n:.1f}%)")

    # How many favour non-linear (ΔBIC ≥ 2)?
    favour_nonlin = (task_fits["delta_bic_nonlin"] >= 2).sum()
    favour_linear = (task_fits["delta_bic_nonlin"] <= -2).sum()
    equivocal = n - favour_nonlin - favour_linear
    log(f"  ΔBIC ≥ 2 (favour non-linear): {favour_nonlin}/{n} ({100*favour_nonlin/n:.1f}%)")
    log(f"  ΔBIC ≤ −2 (favour linear):    {favour_linear}/{n} ({100*favour_linear/n:.1f}%)")
    log(f"  Equivocal (|ΔBIC| < 2):       {equivocal}/{n} ({100*equivocal/n:.1f}%)")

log(f"\n{'─' * 60}")
log("INTERPRETATION:")
log(f"{'─' * 60}")
log("""
With n=9 subjects and 5 time points, individual model fits have
limited statistical power. The results should be interpreted as:

- If majority favour non-linear: CONSISTENT with the three-phase
  model, but not definitive evidence. Frame as hypothesis-supporting.

- If roughly equal split: the three-phase model is neither confirmed
  nor refuted. Frame as "individual trajectories are heterogeneous;
  denser sampling is needed."

- If majority favour linear: the three-phase model is not supported
  at the between-session timescale. Consider that within-session
  non-linearity (from PhysioNet) may not transfer to between-session.

In all cases, this analysis is EXPLORATORY per the pre-registration.
""")

report.close()
print(f"\n✓ Saved dose_response_report.txt and dose_response_fits.csv")
