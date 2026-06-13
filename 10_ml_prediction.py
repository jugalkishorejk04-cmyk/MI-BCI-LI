"""
10_ml_prediction.py
===================
Level 5: Predictive utility — can early-session features predict
late-session responder status?

Two generalization tests (per pre-registration):
  1. Cross-population: train on healthy pooled, test on stroke (Liu2024)
  2. Temporal: train on early-stage features, test on late-stage ΔLI

Models: Logistic regression and gradient boosting (LightGBM or sklearn).
Evaluation: ROC-AUC with nested 5-fold CV for within-dataset,
            leave-one-dataset-out for cross-dataset.

Reads: features_per_subject.csv
       (optionally) features_with_coherence.csv
       (optionally) features_with_all_constraints.csv
Output: ml_prediction_report.txt, ml_results.csv

Usage: python 10_ml_prediction.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import sys

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 18: ML Responder Prediction (Level 5)")
print("=" * 60)

# Try richest feature set first
for fname in ["features_with_all_constraints.csv",
              "features_with_coherence.csv",
              "features_per_subject.csv"]:
    try:
        df = pd.read_csv(fname)
        print(f"Loaded {fname} ({len(df)} rows, {len(df.columns)} columns)")
        break
    except FileNotFoundError:
        continue
else:
    print("ERROR: No feature file found.")
    sys.exit(1)

report = open("ml_prediction_report.txt", "w", encoding="utf-8")
def log(msg):
    print(msg)
    report.write(msg + "\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Use EARLY-stage features to predict responder status
# (which is defined by LATE-stage ΔLI)

feature_cols = []
for task in ["T1", "T2"]:
    for band in ["mu", "beta"]:
        col = f"LI_{band}_early_{task}"
        if col in df.columns:
            feature_cols.append(col)

    for metric in ["mu_ERD_selectivity", "mu_ERD_slope_contra",
                   "mu_ERD_slope_bilateral"]:
        col = f"{metric}_{task}"
        if col in df.columns:
            feature_cols.append(col)

    # Add coherence if available
    for coh_metric in ["imcoh_mu_mean", "wpli_mu_mean"]:
        col = f"{coh_metric}_{task}"
        if col in df.columns:
            feature_cols.append(col)

    # Add trial constraints if available
    for constraint in ["mu_ERD_CV_contra_mean", "mu_onset_latency_SD_mean"]:
        col = f"{constraint}_{task}"
        if col in df.columns:
            feature_cols.append(col)

log(f"\nFeature columns ({len(feature_cols)}):")
for c in feature_cols:
    log(f"  {c}")

target_col = "μ_responder"

# ============================================================
# WITHIN-DATASET CV (per dataset)
# ============================================================
log(f"\n{'=' * 60}")
log("1. WITHIN-DATASET CROSS-VALIDATION")
log(f"{'=' * 60}")

all_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    ]),
}

for ds_name in df["dataset"].unique():
    ds_df = df[df["dataset"] == ds_name].copy()

    # Drop rows with missing features or target
    available_feats = [c for c in feature_cols if c in ds_df.columns]
    clean = ds_df.dropna(subset=available_feats + [target_col])

    if len(clean) < 15:
        log(f"\n  {ds_name}: SKIP (only {len(clean)} complete cases)")
        continue

    X = clean[available_feats].values
    y = clean[target_col].astype(int).values

    # Check class balance
    pos_rate = y.mean()
    if pos_rate < 0.1 or pos_rate > 0.9:
        log(f"\n  {ds_name}: SKIP (class imbalance: {100*pos_rate:.1f}% positive)")
        continue

    log(f"\n  {ds_name} (n={len(clean)}, {int(y.sum())} resp, {len(y)-int(y.sum())} non-resp):")

    for model_name, pipeline in models.items():
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
            mean_auc = scores.mean()
            sd_auc = scores.std()
            log(f"    {model_name}: AUC = {mean_auc:.3f} ± {sd_auc:.3f}")

            all_results.append({
                "test": "within_dataset",
                "dataset": ds_name,
                "model": model_name,
                "auc_mean": mean_auc,
                "auc_sd": sd_auc,
                "n": len(clean),
            })
        except Exception as e:
            log(f"    {model_name}: FAILED ({e})")

# ============================================================
# CROSS-POPULATION: Train healthy → Test stroke
# ============================================================
log(f"\n{'=' * 60}")
log("2. CROSS-POPULATION GENERALIZATION")
log("   Train: all healthy pooled → Test: Liu2024 stroke")
log(f"{'=' * 60}")

healthy_ds = ["PhysionetMI", "Cho2017", "BNCI2014_004", "Lee2019_MI"]
healthy = df[df["dataset"].isin(healthy_ds)].copy()
stroke = df[df["dataset"] == "Liu2024"].copy()

available_feats = [c for c in feature_cols if c in healthy.columns and c in stroke.columns]
healthy_clean = healthy.dropna(subset=available_feats + [target_col])
stroke_clean = stroke.dropna(subset=available_feats + [target_col])

log(f"  Healthy training set: {len(healthy_clean)} subjects")
log(f"  Stroke test set: {len(stroke_clean)} subjects")

if len(healthy_clean) >= 20 and len(stroke_clean) >= 10:
    X_train = healthy_clean[available_feats].values
    y_train = healthy_clean[target_col].astype(int).values
    X_test = stroke_clean[available_feats].values
    y_test = stroke_clean[target_col].astype(int).values

    for model_name, pipeline in models.items():
        try:
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            y_pred = pipeline.predict(X_test)

            log(f"\n  {model_name}:")
            log(f"    AUC = {auc:.3f}")
            log(f"    {'PASSES' if auc >= 0.70 else 'BELOW'} pre-registered threshold (AUC ≥ 0.70)")

            # Classification report
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            log(f"    Accuracy = {acc:.3f}, F1 = {f1:.3f}")

            # Feature importances (for gradient boosting)
            if model_name == "GradientBoosting":
                clf = pipeline.named_steps["clf"]
                importances = clf.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                log(f"    Top features:")
                for rank in range(min(5, len(available_feats))):
                    idx = sorted_idx[rank]
                    log(f"      {rank+1}. {available_feats[idx]}: {importances[idx]:.4f}")

            all_results.append({
                "test": "cross_population",
                "dataset": "healthy→stroke",
                "model": model_name,
                "auc_mean": auc,
                "auc_sd": 0,
                "n": len(stroke_clean),
            })
        except Exception as e:
            log(f"  {model_name}: FAILED ({e})")
else:
    log("  Insufficient data for cross-population test.")

# ============================================================
# TEMPORAL GENERALIZATION: Early features → Late ΔLI
# ============================================================
log(f"\n{'=' * 60}")
log("3. TEMPORAL GENERALIZATION")
log("   Train on early-session features → predict late-session ΔLI")
log(f"{'=' * 60}")

# For datasets with multiple sessions (BCI-IV-2b)
bnci = df[df["dataset"] == "BNCI2014_004"]
sessions = sorted(bnci["session"].unique())

if len(sessions) >= 3:
    early_sessions = sessions[:2]
    late_sessions = sessions[3:]

    log(f"  Early sessions: {early_sessions}")
    log(f"  Late sessions: {late_sessions}")

    early_data = bnci[bnci["session"].isin(early_sessions)]
    late_data = bnci[bnci["session"].isin(late_sessions)]

    # Use early-session mean features to predict late-session responder status
    early_agg = early_data.groupby("subject")[available_feats + [target_col]].mean()
    late_agg = late_data.groupby("subject")[[target_col]].mean()

    # Responder in late sessions
    late_agg["late_responder"] = late_agg[target_col] > 0.5

    combined = early_agg.join(late_agg[["late_responder"]], how="inner").dropna()

    if len(combined) >= 6:
        X = combined[available_feats].values
        y = combined["late_responder"].astype(int).values

        log(f"  Combined: {len(combined)} subjects, {int(y.sum())} late-resp, {len(y)-int(y.sum())} late-non-resp")

        for model_name, pipeline in models.items():
            try:
                # Leave-one-out CV (small n)
                from sklearn.model_selection import LeaveOneOut
                loo = LeaveOneOut()
                y_probs = np.zeros(len(y))
                for train_idx, test_idx in loo.split(X, y):
                    pipeline.fit(X[train_idx], y[train_idx])
                    y_probs[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]

                auc = roc_auc_score(y, y_probs)
                log(f"  {model_name} (LOO-CV): AUC = {auc:.3f}")

                all_results.append({
                    "test": "temporal_generalization",
                    "dataset": "BCI-IV-2b",
                    "model": model_name,
                    "auc_mean": auc,
                    "auc_sd": 0,
                    "n": len(combined),
                })
            except Exception as e:
                log(f"  {model_name}: FAILED ({e})")
    else:
        log(f"  Insufficient complete cases ({len(combined)})")
else:
    log("  BCI-IV-2b multi-session data not available for temporal generalization.")

# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("ml_results.csv", index=False)

log(f"\n{'=' * 60}")
log("ML PREDICTION COMPLETE")
log(f"{'=' * 60}")
log(f"\nResults saved to: ml_results.csv")

report.close()
print(f"\n✓ Reports saved.")
