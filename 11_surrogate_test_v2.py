"""
11_surrogate_test_v2.py — Circularity control via subject-label permutation.
Tests whether the real (matched) selectivity-dLI correlation exceeds the
null distribution of mismatched (across-subject) pairings.
Reads precomputed per-subject features. No MOABB, no downloads.
"""
import numpy as np
import pandas as pd
from scipy import stats

N_PERM = 5000
rng = np.random.default_rng(42)

# Use the merged per-subject file. Adjust the filename if yours differs.
CANDIDATES = ["MERGED_features_per_subject.csv",
              "features_per_subject.csv"]
import os
src = next((f for f in CANDIDATES if os.path.exists(f)), None)
if src is None:
    raise SystemExit("No features_per_subject CSV found in this folder.")
print(f"Using: {src}")

feat = pd.read_csv(src)

# If Lee2019 is in a separate file, merge it (optional safety)
if os.path.exists("lee2019_features_per_subject.csv"):
    lee = pd.read_csv("lee2019_features_per_subject.csv")
    feat = pd.concat([feat[feat["dataset"] != "Lee2019_MI"], lee], ignore_index=True)
    print("Merged in lee2019_features_per_subject.csv")

rows = []
for ds in sorted(feat["dataset"].unique()):
    d = feat[feat["dataset"] == ds]
    for task in ["T1", "T2"]:
        sel_col = f"mu_ERD_selectivity_{task}"
        dli_col = f"ΔLI_mu_{task}"
        if sel_col not in d.columns or dli_col not in d.columns:
            continue
        sub = d[[sel_col, dli_col]].dropna()
        if len(sub) < 5:
            continue
        sel = sub[sel_col].to_numpy()
        dli = sub[dli_col].to_numpy()

        # Real (matched) correlation
        real_rho, real_p = stats.spearmanr(sel, dli)

        # Null: shuffle dLI across subjects, recorrelate
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            null[k] = stats.spearmanr(sel, rng.permutation(dli))[0]
        null = null[~np.isnan(null)]

        # Two-sided permutation p
        perm_p = (np.sum(np.abs(null) >= abs(real_rho)) + 1) / (len(null) + 1)
        pct = stats.percentileofscore(np.abs(null), abs(real_rho))

        print(f"{ds:14s} {task}: real rho={real_rho:.3f} (param p={real_p:.1e}), "
              f"null mean|rho|={np.mean(np.abs(null)):.3f}, "
              f"null 95th={np.percentile(np.abs(null),95):.3f}, "
              f"perm p={perm_p:.4f}, pct={pct:.1f}")

        rows.append({"dataset": ds, "task": task, "n": len(sub),
                     "real_rho": real_rho, "param_p": real_p,
                     "null_mean_abs": float(np.mean(np.abs(null))),
                     "null_95th": float(np.percentile(np.abs(null), 95)),
                     "perm_p": perm_p, "percentile": pct})

out = pd.DataFrame(rows)
out.to_csv("surrogate_results_v2.csv", index=False)
print("\nSaved surrogate_results_v2.csv")
print(out.to_string())