"""
16_lee_acrossday_validation.py
==============================
The CLEAN test: does session-1 ΔLI / selectivity predict the CHANGE in decoding
accuracy from session 1 to session 2, across 54 Lee subjects?

Across-days => session-2 accuracy is a separate recording on a different day, so
within-session drift cannot carry across. This is the drift-free version of the
biomarker test, at n=54.

Phase consistency (the review's wrinkle): accuracy for BOTH sessions is computed
from the SAME phase. We use EEG_MI_train (offline) for both, stated explicitly.

Reads raw .mat (both sessions), full montage CSP+LDA per session.
Output: lee_acrossday.csv, lee_acrossday_report.txt
"""
import os, glob, warnings, gc
os.environ["OMP_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.signal import welch, resample_poly
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mne; mne.set_log_level("ERROR")
from mne.decoding import CSP

SF_RAW=1000.0; SF=160.0
BASE=(0,int(0.5*SF)); MI=(int(0.5*SF),int(3.0*SF)); STAGES=3; MU=(8.0,13.0)
EPOCH_S=3.0
PHASE="EEG_MI_train"   # SAME phase both sessions (offline)
BASE_DIR=r"F:\MI_BCI_Pipeline\F-\MI_BCI_Pipeline\F-\mne_data\MNE-lee2019-mi-data"

def erd_db(seg):
    b=seg[BASE[0]:BASE[1]]; m=seg[MI[0]:MI[1]]
    if len(b)<8 or len(m)<8: return np.nan
    fb,pb=welch(b,fs=SF,nperseg=min(len(b),int(SF)))
    fm,pm=welch(m,fs=SF,nperseg=min(len(m),int(SF)))
    mb=(fb>=MU[0])&(fb<=MU[1]); mm=(fm>=MU[0])&(fm<=MU[1])
    Pb=np.mean(pb[mb]) if mb.any() else np.nan
    Pm=np.mean(pm[mm]) if mm.any() else np.nan
    return 10*np.log10(Pm/Pb) if (Pb and Pb>0 and Pm and Pm>0) else np.nan

def sel_dli(c,i):
    n=len(c)
    if n<STAGES*2: return np.nan,np.nan
    e=np.linspace(0,n,STAGES+1,dtype=int); cs,is_=[],[]
    for s in range(STAGES):
        cc=c[e[s]:e[s+1]]; ii=i[e[s]:e[s+1]]; cc=cc[~np.isnan(cc)]; ii=ii[~np.isnan(ii)]
        if len(cc)<1 or len(ii)<1: return np.nan,np.nan
        cs.append(np.mean(np.abs(cc))); is_.append(np.mean(np.abs(ii)))
    cs=np.array(cs); is_=np.array(is_); x=np.array([1.,2.,3.])
    sel=np.polyfit(x,cs,1)[0]-np.polyfit(x,(cs+is_)/2,1)[0]
    le=(cs[0]-is_[0])/(cs[0]+is_[0]) if (cs[0]+is_[0])>0 else np.nan
    ll=(cs[2]-is_[2])/(cs[2]+is_[2]) if (cs[2]+is_[2])>0 else np.nan
    return sel, ll-le

def load_phase(matfile, phase):
    d=sio.loadmat(matfile,squeeze_me=True,struct_as_record=False)
    if phase not in d: return None,None,None
    s=d[phase]; chan=list(s.chan)
    idx3=[chan.index(c) for c in ["C3","Cz","C4"]]
    x=s.x; t=np.atleast_1d(s.t).astype(int); y=np.atleast_1d(s.y_dec).astype(int)
    segR=int(EPOCH_S*SF_RAW)
    X3=[]; Xfull=[]; ys=[]
    for oi,on in enumerate(t):
        if on+segR>x.shape[0]: continue
        seg=x[on:on+segR,:]                       # full montage
        seg=resample_poly(seg,int(SF),int(SF_RAW),axis=0)
        Xfull.append(seg.T); X3.append(seg[:,idx3].T); ys.append(y[oi])
    return np.array(X3),np.array(Xfull),np.array(ys)

def decode_acc(Xfull,y):
    y=np.array([0 if v==2 else 1 for v in y])  # 2=left->0,1=right->1
    if np.min(np.bincount(y))<6: return np.nan
    Xb=mne.filter.filter_data(Xfull[:,:,int(0.5*SF):int(3.0*SF)].astype(float),SF,MU[0],MU[1],verbose=False)
    skf=StratifiedKFold(5,shuffle=True,random_state=42); accs=[]
    for tr,te in skf.split(Xb,y):
        try:
            csp=CSP(n_components=6,reg='ledoit_wolf',log=True)
            Z=csp.fit_transform(Xb[tr],y[tr]); Zt=csp.transform(Xb[te])
            lda=LinearDiscriminantAnalysis().fit(Z,y[tr]); accs.append(lda.score(Zt,y[te]))
        except Exception: continue
    return np.mean(accs) if accs else np.nan

rows=[]
for sid in range(1,55):
    rec={"subject":sid}
    good=True
    accs={}
    for sess in [1,2]:
        pat=os.path.join(BASE_DIR,"**",f"session{sess}",f"s{sid}",f"sess0{sess}_subj{sid:02d}_EEG_MI.mat")
        f=glob.glob(pat,recursive=True)
        if not f: good=False; break
        X3,Xfull,y=load_phase(f[0],PHASE)
        if X3 is None: good=False; break
        accs[sess]=decode_acc(Xfull,y)
        if sess==1:
            # session-1 ΔLI/selectivity from C3/C4 (label 2=left contra=C4; 1=right contra=C3)
            for lbl,tag,ci,ii in [(2,"T1",2,0),(1,"T2",0,2)]:
                m=(y==lbl)
                if m.sum()<STAGES*2: continue
                d=X3[m]
                c=np.array([erd_db(d[t,ci,:]) for t in range(len(d))])
                i=np.array([erd_db(d[t,ii,:]) for t in range(len(d))])
                sel,dli=sel_dli(c,i)
                rec[f"sel_{tag}_s1"]=sel; rec[f"dLI_{tag}_s1"]=dli
        del X3,Xfull; gc.collect()
    if good and 1 in accs and 2 in accs:
        rec["acc_s1"]=accs[1]; rec["acc_s2"]=accs[2]
        rec["acc_gain"]=accs[2]-accs[1]
        rows.append(rec); print(f"subj {sid} done",flush=True)
    else:
        print(f"subj {sid} skip",flush=True)

df=pd.DataFrame(rows); df.to_csv("lee_acrossday.csv",index=False)
lines=[]; W=lambda s:(lines.append(s),print(s,flush=True))
W(f"\nLEE ACROSS-DAYS (n={len(df)}): does S1 metric predict S1->S2 accuracy gain?\n")
W(f"Phase used (both sessions): {PHASE}")

# QC: how many survive accuracy filter in BOTH sessions
EXCL=0.55
both_ok=(df["acc_s1"]>EXCL)&(df["acc_s2"]>EXCL)
W(f"Subjects with acc>{EXCL} in BOTH sessions: {both_ok.sum()}/{len(df)}")
W(f"Mean acc S1={df['acc_s1'].mean():.3f}  S2={df['acc_s2'].mean():.3f}  gain={df['acc_gain'].mean():+.3f}")

# Reliability floor: SE of accuracy gain, and split-half ceiling estimate
# gain SE ~ sqrt(2)*SE_acc; SE_acc ~ sqrt(p(1-p)/ntrials). Report as context.
W(f"Accuracy gain SD across subjects: {df['acc_gain'].std():.3f}")
W(f"(Context: if most of this is estimation noise, correlations are attenuated.)\n")

def report_block(sub, label):
    W(f"--- {label} (n={len(sub)}) ---")
    for metric in ["dLI","selectivity"]:
        for tag in ["T1","T2"]:
            col=f"{'dLI' if metric=='dLI' else 'sel'}_{tag}_s1"
            if col in sub:
                d=sub[[col,"acc_gain"]].dropna()
                if len(d)>=8:
                    r,p=stats.spearmanr(d[col],d["acc_gain"])
                    W(f"  {metric} {tag}: rho={r:+.3f} p={p:.3f} n={len(d)}")
    W("")

# BOTH ways: unfiltered (all subjects) and filtered (acc>EXCL both sessions)
report_block(df, "UNFILTERED (all subjects)")
report_block(df[both_ok], f"FILTERED (acc>{EXCL} both sessions)")
W("If the correlation only appears in the filtered sample, note it explicitly -")
W("the exclusion conditions on BCI operability, which may correlate with the predictor.")

open("lee_acrossday_report.txt","w").write("\n".join(lines))
W("\nSaved lee_acrossday.csv + report")
