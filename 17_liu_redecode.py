"""
17_liu_redecode.py - Re-decode Liu2024 on the FULL 0-4s window (scripts 15/16 used
0.5-3.0s, which the window scan showed was suboptimal). Establishes Liu's fair
best-case decoding accuracy, so the paper can state "at chance under an optimised
window" rather than "under a window later found suboptimal".
"""
import warnings, gc
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mne; mne.set_log_level("ERROR")
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from moabb.datasets import Liu2024
from moabb.paradigms import LeftRightImagery

p = LeftRightImagery(fmin=8, fmax=30, resample=160.0, tmin=0, tmax=4, baseline=None)
ds = Liu2024()
rows=[]
for sid in ds.subject_list:
    try:
        ep,y,meta = p.get_data(ds, subjects=[sid], return_epochs=True)
    except Exception as e:
        print(f"subj {sid} skip ({e})", flush=True); continue
    X=ep.get_data(); yb=(y=='right_hand').astype(int)
    if np.min(np.bincount(yb))<5: 
        print(f"subj {sid} skip (imbalance)", flush=True); continue
    # full 0-4s window
    clf=Pipeline([('csp',CSP(6,reg='ledoit_wolf',log=True)),('lda',LinearDiscriminantAnalysis())])
    acc=cross_val_score(clf,X,yb,cv=5).mean()
    rows.append({"subject":sid,"acc_full_0_4s":acc})
    print(f"subj {sid}: acc={acc:.3f}", flush=True)
    del X; gc.collect()

df=pd.DataFrame(rows)
df.to_csv("liu_redecode_full_window.csv",index=False)
print(f"\nLiu full-window (0-4s) decoding, n={len(df)}")
print(f"  mean={df['acc_full_0_4s'].mean():.3f}  median={df['acc_full_0_4s'].median():.3f}")
print(f"  %>0.60={100*(df['acc_full_0_4s']>0.6).mean():.0f}%  %>0.70={100*(df['acc_full_0_4s']>0.7).mean():.0f}%")
print(f"  (compare: 0.5-3.0s window gave mean 0.496)")
