"""
22_model_a_decontaminated.py
----------------------------
Re-runs Model A from script 20 with three corrections.

C1  DE-DUPLICATION. Script 20 merged on (dataset, subject) without session,
    duplicating BNCI2014_004 fivefold, and the frame is long over task so
    every subject appears twice. Both inflate n and, worse, leak the same
    subject into both sides of a within-cohort median split.

C2  FEATURE DECONTAMINATION. The 19-feature set included smr_snr_mean and
    ahn_ratio, which predict decode_acc at r = +0.54 and -0.41. An AUC of
    ~0.70 may therefore be entirely the published benchmarks rather than the
    dLI-derived features. Three feature sets are now compared:
        SET A  dLI-derived only        <- the question actually being asked
        SET B  benchmarks only         <- the ceiling the benchmarks alone give
        SET C  both                    <- does A add anything over B?

C3  PERMUTATION NULL. Script 20 ran a null for Model B only. At these sample
    sizes an AUC of 0.6 is not obviously above chance; the null makes that
    judgeable.

OUTPUT: model_a_decontaminated_report.txt
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
FEAT_F, BEHAV_F = "features_per_subject_MERGED.csv", "behavioural_validation_within.csv"
SMR_F, AHN_F = "blankertz_smr_per_subject.csv", "ahn_theta_alpha_per_subject.csv"
REPORT = "model_a_decontaminated_report.txt"
MIN_ACC, N_PERM = 0.58, 500

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def rd(p):
    d = pd.read_csv(p)
    for c in ("subject", "session"):
        if c in d:
            d[c] = d[c].astype(str)
    return d

behav, feat, smr, ahn = rd(BEHAV_F), rd(FEAT_F), rd(SMR_F), rd(AHN_F)
KEY = [k for k in ["dataset", "subject", "session"] if k in behav and k in smr]

M = behav.merge(smr[KEY + ["smr_snr_mean"]], on=KEY, how="left")
M = M.merge(ahn[KEY + ["ahn_ratio"]], on=KEY, how="left")
fkey = [k for k in KEY if k in feat.columns]
M = M.merge(feat, on=fkey, how="left", suffixes=("", "_f"))
M = M.drop_duplicates(subset=KEY)                       # C1
say(f"rows after de-duplication: {len(M)}")
say(M.groupby("dataset").agg(rows=("subject", "size"),
                             subjects=("subject", "nunique"),
                             acc=("decode_acc", "mean")).round(3).to_string())

BENCH = [c for c in ["smr_snr_mean", "ahn_ratio"] if c in M]
DLI = [c for c in M.columns
       if any(k in c.lower() for k in
              ["dli", "select", "slope_contra", "slope_bilateral",
               "li_early", "li_mid", "li_late"])
       and c not in BENCH and "acc" not in c.lower()]
SETS = {"A_dLI_derived": DLI, "B_benchmarks": BENCH, "C_both": DLI + BENCH}
say(f"\nSET A ({len(DLI)}): {DLI}")
say(f"SET B ({len(BENCH)}): {BENCH}")

acc = M.groupby("dataset").decode_acc.mean()
usable = acc[acc >= MIN_ACC].index.tolist()
say(f"\nusable cohorts: {usable}")

A = M[M.dataset.isin(usable)].copy()
A["target"] = A.groupby("dataset").decode_acc.transform(
    lambda s: (s > s.median()).astype(int))

def run(tr, te, cols, clf):
    cols = [c for c in cols if c in tr and c in te]
    tr2 = tr.dropna(subset=cols + ["target"])
    te2 = te.dropna(subset=cols + ["target"])
    if len(tr2) < 30 or len(te2) < 15 or te2.target.nunique() < 2:
        return None, None
    pipe = Pipeline([("sc", StandardScaler()), ("m", clf)])
    pipe.fit(tr2[cols], tr2.target)
    pr = pipe.predict_proba(te2[cols])[:, 1]
    auc = roc_auc_score(te2.target, pr)
    null = []
    for _ in range(N_PERM):
        yp = RNG.permutation(tr2.target.values)
        p2 = Pipeline([("sc", StandardScaler()), ("m", clf)])
        try:
            p2.fit(tr2[cols], yp)
            null.append(roc_auc_score(te2.target, p2.predict_proba(te2[cols])[:, 1]))
        except Exception:
            pass
    return auc, np.array(null)

say("\n" + "=" * 74)
say("MODEL A — leave-one-dataset-out, target = decoding accuracy (high vs low)")
say("=" * 74)
for name, cols in SETS.items():
    say(f"\n### feature set {name} ###")
    aucs = []
    for held in usable:
        tr, te = A[A.dataset != held], A[A.dataset == held]
        for cn, clf in [("LR", LogisticRegression(max_iter=1000)),
                        ("GB", GradientBoostingClassifier(n_estimators=100,
                                                          max_depth=3, random_state=42))]:
            auc, null = run(tr, te, cols, clf)
            if auc is None:
                say(f"  {held:14s} {cn}  insufficient data")
                continue
            p = (np.sum(null >= auc) + 1) / (len(null) + 1)
            say(f"  test {held:14s} {cn}  n={len(te):3d}  AUC={auc:.3f}   "
                f"perm null mean={null.mean():.3f} (95th {np.percentile(null,95):.3f})  p={p:.4f}")
            if cn == "LR":
                aucs.append(auc)
    if aucs:
        say(f"  mean LR AUC across held-out cohorts: {np.mean(aucs):.3f}")

say("\nHOW TO READ THIS:")
say("  Compare SET A against SET B. If B alone reaches the same AUC as C,")
say("  the transfer result is carried by the published benchmarks and the")
say("  dLI-derived features add nothing — report it that way.")
say("  If A clears its permutation null on its own, dLI-derived features do")
say("  predict an external target across cohorts, and that is the")
say("  non-circular replacement for the V11 classifier.")
say("  Judge every AUC against its permutation null, not against 0.5.")

with open(REPORT, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {REPORT}")
