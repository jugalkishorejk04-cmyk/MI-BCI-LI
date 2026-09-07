"""
13_shu_benchmark_v2.py - Shu et al. (2018) LI/CAS benchmark, corrected.

Fixes vs v1:
  - LI/CAS from strictly-positive BAND POWER (bounded LI in [-1,1])   [bug #2,#3]
  - mu-band selection constrained to 7-13 Hz with peak test + fixed fallback [bug #4]
  - CSP+LDA accuracy on the FULL montage, not C3/Cz/C4                 [bug #5]
  - flexible channel names so Liu2024 (FC3/CP3) resolves               [bug #1]
  - self-validation: prints fallback rate, LI-bound check, per-dataset accuracy

Shu X et al. Front Neurosci. 2018;12:93. doi:10.3389/fnins.2018.00093

Output: shu_benchmark_per_subject.csv
"""
import gc, os, warnings
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mne; mne.set_log_level("ERROR")
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
from moabb.paradigms import LeftRightImagery
from bci_benchmark_common import (SFREQ, bandpower, pick_mu_band,
                                  laterality_index, cortical_activation_strength,
                                  csp_lda_accuracy)

# FULL montage (no channel restriction) so CSP has spatial filters + Shu channels exist
PARADIGM = LeftRightImagery(fmin=1, fmax=45, resample=160.0, tmin=0.0, tmax=4.0, baseline=None)
DATASETS = {"PhysionetMI":PhysionetMI(),"Cho2017":Cho2017(),
            "BNCI2014_004":BNCI2014_004(),"Lee2019_MI":Lee2019_MI(),"Liu2024":Liu2024()}
DATASETS["PhysionetMI"].subject_list=[s for s in DATASETS["PhysionetMI"].subject_list if s!=88]

# flexible: multiple naming conventions per hemisphere
LEFT_CH  = ["FC5","FC3","FC1","C3","CP5","CP3","CP1"]
RIGHT_CH = ["FC6","FC4","FC2","C4","CP6","CP4","CP2"]

def load_liu_paretic():
    for fn in ["participants.tsv","1778869948884_participants.tsv"]:
        if os.path.exists(fn):
            clin=pd.read_csv(fn,sep="\t")
            col=[c for c in clin.columns if 'aralysis' in c or 'aretic' in c]
            idc=[c for c in clin.columns if 'articipant' in c.lower() or c.lower()=='subject']
            if col and idc:
                m={}
                for _,r in clin.iterrows():
                    sid=int(''.join(ch for ch in str(r[idc[0]]) if ch.isdigit()))
                    m[sid]=str(r[col[0]]).strip().lower()
                return m
    return {}
liu_paretic=load_liu_paretic()

rows=[]; fb_count=0; total=0
for name, ds in DATASETS.items():
    print(f"\n{'='*44}\n{name}\n{'='*44}", flush=True)
    for sid in ds.subject_list:
        try:
            ep,y,meta=PARADIGM.get_data(ds, subjects=[sid], return_epochs=True)
        except Exception as e:
            print(f"  subj {sid} skip ({e})", flush=True); gc.collect(); continue
        ch=ep.ch_names
        L=[c for c in LEFT_CH if c in ch]; R=[c for c in RIGHT_CH if c in ch]
        if "C3" not in ch or "C4" not in ch or len(L)<2 or len(R)<2:
            print(f"  subj {sid} skip (montage L={len(L)} R={len(R)})", flush=True)
            del ep; gc.collect(); continue
        data=ep.get_data().copy(); labels=np.asarray(y); del ep; gc.collect()
        sess=meta["session"].values if "session" in meta.columns else np.array(["0"]*len(labels))
        i3,i4=ch.index("C3"),ch.index("C4")
        iL=[ch.index(c) for c in L]; iR=[ch.index(c) for c in R]
        base_n=int(1.0*SFREQ); t0,t1=int(1.0*SFREQ),int(4.0*SFREQ)

        classes=sorted(np.unique(labels))
        is_left=lambda l: str(l).lower().startswith('left')
        left_lbls=[c for c in classes if is_left(c)]
        paretic_label=left_lbls[0] if left_lbls else classes[0]

        for s_id in np.unique(sess):
            m=(sess==s_id); d=data[m]; lab=labels[m]
            if len(d)<12: continue
            p_idx=np.where(lab==paretic_label)[0][:5]
            if len(p_idx)<3: continue

            # individual mu band from C3/C4 over the paretic-hand trials (task window)
            c3sig=np.concatenate([d[t,i3,t0:t1] for t in p_idx])
            c4sig=np.concatenate([d[t,i4,t0:t1] for t in p_idx])
            lo,hi,fb=pick_mu_band(c3sig,c4sig)
            total+=1; fb_count+=int(fb)

            # band POWER per hemisphere (strictly positive) -> bounded LI
            def hemi_power(idx_list):
                vals=[]
                for t in p_idx:
                    for ci in idx_list:
                        bp=bandpower(d[t,ci,t0:t1],lo,hi)
                        if bp==bp and bp>0: vals.append(bp)
                return np.mean(vals) if vals else np.nan
            # contralateral to LEFT hand = RIGHT hemisphere
            bp_contra=hemi_power(iR); bp_ipsi=hemi_power(iL)
            shu_LI=laterality_index(bp_contra,bp_ipsi)
            shu_CAS=cortical_activation_strength(bp_contra,bp_ipsi)

            # two-class decoding on remaining trials, FULL montage
            keep=np.ones(len(d),bool); keep[p_idx]=False
            other=[c for c in classes if c!=paretic_label]
            if other:
                o_idx=np.where(lab==other[0])[0][:5]; keep[o_idx]=False
            Xacc=d[keep][:,:,t0:t1]
            yb=np.array([0 if l==paretic_label else 1 for l in lab[keep]])
            acc=csp_lda_accuracy(Xacc,yb,lo,hi)

            rows.append({"dataset":name,"subject":sid,"session":str(s_id),
                         "shu_LI":shu_LI,"shu_CAS":shu_CAS,"bci_acc_2class":acc,
                         "band_lo":lo,"band_hi":hi,"band_fallback":int(fb)})
        del data; gc.collect()
        print(f"  subj {sid} done", flush=True)

out=pd.DataFrame(rows)
out.to_csv("shu_benchmark_per_subject.csv", index=False)

# ---- self-validation report ----
print(f"\nSaved ({len(out)} rows)")
print(f"Datasets present: {sorted(out['dataset'].unique())}")
oob=((out['shu_LI']<-1)|(out['shu_LI']>1)).sum()
print(f"LI-bound check: {oob} subjects outside [-1,1]  (should be 0)")
print(f"Band fallback used: {fb_count}/{total} = {100*fb_count/max(total,1):.0f}%")
print("Mean 2-class accuracy by dataset (PhysioNet target ~0.65):")
print(out.groupby('dataset')['bci_acc_2class'].mean().round(3))
print("Mean LI / CAS by dataset:")
print(out.groupby('dataset')[['shu_LI','shu_CAS']].mean().round(3))
