"""
12_blankertz_smr_v2.py - Blankertz et al. (2010) SMR predictor, corrected.
Fix vs v1: uses a LONGER rest segment (whole pre-cue baseline concatenated across
trials, >= several seconds) so the 1/f fit is stable. Fixed mu band 8-13 Hz.

Blankertz R, Sannelli C, Halder S, et al. Neurophysiological predictor of
SMR-based BCI performance. NeuroImage. 2010;51(4):1303-1309.

Output: blankertz_smr_per_subject.csv
"""
import gc, os, warnings
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mne; mne.set_log_level("ERROR")
from scipy.signal import welch
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
from moabb.paradigms import LeftRightImagery

SF = 160.0
MU = (8.0, 13.0)
NOISE = (3.0, 40.0)
PARADIGM = LeftRightImagery(fmin=1, fmax=45, channels=["C3","Cz","C4"],
                            resample=160.0, tmin=0.0, tmax=3.0, baseline=None)
DATASETS = {"PhysionetMI":PhysionetMI(),"Cho2017":Cho2017(),
            "BNCI2014_004":BNCI2014_004(),"Lee2019_MI":Lee2019_MI(),"Liu2024":Liu2024()}
DATASETS["PhysionetMI"].subject_list=[s for s in DATASETS["PhysionetMI"].subject_list if s!=88]

def smr_snr(sig):
    """dB height of mu peak over a log-log 1/f fit. Needs an adequately long sig."""
    if len(sig) < int(SF*2):           # require >= 2 s of data
        return np.nan
    f, pxx = welch(sig, fs=SF, nperseg=int(SF*2))
    ok = (f>=NOISE[0])&(f<=NOISE[1])&(pxx>0)
    if ok.sum() < 10: return np.nan
    lf, lp = np.log10(f[ok]), np.log10(pxx[ok])
    nonmu = ~((f[ok]>=MU[0])&(f[ok]<=MU[1]))
    if nonmu.sum() < 6: return np.nan
    b = np.polyfit(lf[nonmu], lp[nonmu], 1)
    mu = (f>=MU[0])&(f<=MU[1])&(pxx>0)
    if mu.sum() < 2: return np.nan
    resid = 10*np.log10(pxx[mu]) - 10*np.polyval(b, np.log10(f[mu]))
    return float(np.max(resid))

rows=[]
for name, ds in DATASETS.items():
    print(f"\n{'='*40}\n{name}\n{'='*40}", flush=True)
    for sid in ds.subject_list:
        try:
            ep,y,meta = PARADIGM.get_data(ds, subjects=[sid], return_epochs=True)
        except Exception as e:
            print(f"  subj {sid} skip ({e})", flush=True); gc.collect(); continue
        ch=ep.ch_names
        if "C3" not in ch or "C4" not in ch:
            del ep; gc.collect(); continue
        i3,i4=ch.index("C3"),ch.index("C4")
        data=ep.get_data().copy(); del ep; gc.collect()
        sess=meta["session"].values if "session" in meta.columns else np.array(["0"]*len(y))
        base_n=int(0.5*SF)   # per-trial baseline samples
        for s_id in np.unique(sess):
            m=(sess==s_id); d=data[m]
            if m.sum()<5: continue
            # CONCATENATE baseline windows across ALL trials -> long rest segment
            c3=np.concatenate([d[t,i3,:base_n] for t in range(len(d))])
            c4=np.concatenate([d[t,i4,:base_n] for t in range(len(d))])
            s3,s4=smr_snr(c3),smr_snr(c4)
            rows.append({"dataset":name,"subject":sid,"session":str(s_id),
                         "smr_snr_C3":s3,"smr_snr_C4":s4,
                         "smr_snr_mean":np.nanmean([s3,s4]),
                         "rest_seconds":round(len(c3)/SF,1)})
        del data; gc.collect()
        print(f"  subj {sid} done", flush=True)

out=pd.DataFrame(rows)
out.to_csv("blankertz_smr_per_subject.csv", index=False)
print(f"\nSaved ({len(out)} rows). Rest-segment length (s):")
print(out.groupby('dataset')['rest_seconds'].mean().round(1))
print("SMR-SNR by dataset:")
print(out.groupby('dataset')['smr_snr_mean'].agg(['mean','std','count']).round(2))
