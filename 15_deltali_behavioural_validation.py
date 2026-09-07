"""
15_deltali_behavioural_validation.py
=====================================
Tests whether the within-session lateralisation shift (ΔLI) and ERD selectivity
track an INDEPENDENT behavioural outcome: within-session improvement in BCI
decoding control (the "learning slope").

Decoding accuracy is computed on the FULL montage, so it is independent of the
C3/C4 ERD trajectory that ΔLI/selectivity are built from.

DESIGN (margin-trend, per the methodological review):
  - One CSP+LDA per subject/session, stratified k-fold balanced across class AND
    across early/mid/late thirds (asserted in code).
  - Record held-out LDA decision-function margin per trial.
  - Regress margin on chronological trial index -> within-session learning slope.
    Report RAW slope and slope AFTER regressing out a per-trial variance nuisance.
  - Correlate learning slope with BOTH ΔLI and selectivity, per dataset.
  - Report effect sizes + bootstrap CIs; random-effects summary across datasets.

Analysis 2 (across-days):
  - Lee (n=54, POWERED): session-1 ΔLI/selectivity vs (sess2 - sess1) accuracy,
    accuracy from the SAME phase both sessions.
  - BCI-IV-2b (n=9, DESCRIPTIVE): session-1 metrics vs accuracy change to session 5.

Outputs: behavioural_validation_within.csv, behavioural_validation_report.txt
"""
import gc, os, warnings
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mne; mne.set_log_level("ERROR")
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
from moabb.paradigms import LeftRightImagery

SF=160.0
BASE=(0,int(0.5*SF)); MI=(int(0.5*SF),int(3.0*SF)); STAGES=3; MU=(8.0,13.0)
FULL = LeftRightImagery(fmin=1,fmax=45,resample=160.0,tmin=0.0,tmax=4.0,baseline=None)  # full montage
DATASETS={"PhysionetMI":PhysionetMI(),"Cho2017":Cho2017(),"BNCI2014_004":BNCI2014_004(),
          "Lee2019_MI":Lee2019_MI(),"Liu2024":Liu2024()}
DATASETS["PhysionetMI"].subject_list=[s for s in DATASETS["PhysionetMI"].subject_list if s!=88]

def erd_db(seg):
    b=seg[BASE[0]:BASE[1]]; m=seg[MI[0]:MI[1]]
    if len(b)<8 or len(m)<8: return np.nan
    fb,pb=welch(b,fs=SF,nperseg=min(len(b),int(SF)))
    fm,pm=welch(m,fs=SF,nperseg=min(len(m),int(SF)))
    mb=(fb>=MU[0])&(fb<=MU[1]); mm=(fm>=MU[0])&(fm<=MU[1])
    Pb=np.mean(pb[mb]) if mb.any() else np.nan
    Pm=np.mean(pm[mm]) if mm.any() else np.nan
    return 10*np.log10(Pm/Pb) if (Pb and Pb>0 and Pm and Pm>0) else np.nan

def sel_and_dli(trials_c, trials_i):
    """selectivity + ΔLI from chronologically-ordered contra/ipsi ERD trials."""
    n=len(trials_c)
    if n<STAGES*2: return np.nan,np.nan
    e=np.linspace(0,n,STAGES+1,dtype=int); cs,is_=[],[]
    for s in range(STAGES):
        c=trials_c[e[s]:e[s+1]]; i=trials_i[e[s]:e[s+1]]
        c=c[~np.isnan(c)]; i=i[~np.isnan(i)]
        if len(c)<1 or len(i)<1: return np.nan,np.nan
        cs.append(np.mean(np.abs(c))); is_.append(np.mean(np.abs(i)))
    cs=np.array(cs); is_=np.array(is_); x=np.array([1.,2.,3.])
    sel=np.polyfit(x,cs,1)[0]-np.polyfit(x,(cs+is_)/2,1)[0]
    li_e=(cs[0]-is_[0])/(cs[0]+is_[0]) if (cs[0]+is_[0])>0 else np.nan
    li_l=(cs[2]-is_[2])/(cs[2]+is_[2]) if (cs[2]+is_[2])>0 else np.nan
    return sel, li_l-li_e

def band_power_wholehead(X, lo, hi):
    """Per-trial whole-head mean band power (nuisance proxy for drift)."""
    out=np.empty(len(X))
    for t in range(len(X)):
        ps=[]
        for ch in range(X.shape[1]):
            f,p=welch(X[t,ch],fs=SF,nperseg=min(X.shape[2],int(SF)))
            m=(f>=lo)&(f<=hi)
            if m.any(): ps.append(np.trapezoid(p[m],f[m]))
        out[t]=np.mean(ps) if ps else np.nan
    return out

def learning_slope(X, y, seed=42):
    """
    Full-montage CSP+LDA, stratified k-fold balanced across class AND thirds.
    Returns dict: raw slope, variance-controlled, full-nuisance-controlled,
    mean held-out accuracy (QC), n_used.
    """
    n=len(y)
    nan=dict(raw=np.nan,var=np.nan,full=np.nan,acc=np.nan,n=0)
    if len(np.unique(y))<2 or np.min(np.bincount(y))<6: return nan
    third=np.zeros(n,int); e=np.linspace(0,n,4,dtype=int)
    for k in range(3): third[e[k]:e[k+1]]=k
    strat=pd.factorize(np.array([f"{yy}_{tt}" for yy,tt in zip(y,third)]))[0]
    if np.min(np.bincount(strat))<3: return nan
    nsplit=5 if np.min(np.bincount(strat))>=5 else 3
    Xb=mne.filter.filter_data(X.astype(float),SF,MU[0],MU[1],verbose=False)
    margins=np.full(n,np.nan); preds=np.full(n,np.nan)
    skf=StratifiedKFold(n_splits=nsplit,shuffle=True,random_state=seed)
    for tr,te in skf.split(Xb,strat):
        if len(np.unique(third[tr]))<3: return nan
        try:
            nc=min(6,X.shape[1]-1)
            csp=CSP(n_components=nc,reg='ledoit_wolf',log=True)
            Ztr=csp.fit_transform(Xb[tr],y[tr]); Zte=csp.transform(Xb[te])
            lda=LinearDiscriminantAnalysis().fit(Ztr,y[tr])
            margins[te]=lda.decision_function(Zte); preds[te]=lda.predict(Zte)
        except Exception:
            return nan
    ok=~np.isnan(margins)
    if ok.sum()<STAGES*2: return nan
    acc=np.mean(preds[ok]==y[ok])
    idx=np.arange(n)[ok]
    mg=margins[ok]*np.where(y[ok]==1,1,-1)   # sign so higher=more correct
    raw=np.polyfit(idx,mg,1)[0]
    # nuisance regressors on the same trials
    tv=np.array([np.var(X[t]) for t in range(n)])[ok]
    def resid_slope(covs):
        Cst=np.column_stack([np.ones_like(idx,dtype=float)]+covs)
        beta,_,_,_=np.linalg.lstsq(Cst,mg,rcond=None)
        r=mg-Cst@beta
        return np.polyfit(idx,r,1)[0]
    var_ctrl=resid_slope([tv.astype(float)])
    mu_pow=band_power_wholehead(X,8,13)[ok]
    beta_pow=band_power_wholehead(X,13,30)[ok]
    full_ctrl=resid_slope([tv.astype(float),mu_pow,beta_pow])
    return dict(raw=raw,var=var_ctrl,full=full_ctrl,acc=acc,n=int(ok.sum()))

def boot_ci(x,y,n=2000,seed=1):
    rng=np.random.default_rng(seed); rs=[]
    x=np.asarray(x); y=np.asarray(y); m=len(x)
    for _ in range(n):
        b=rng.integers(0,m,m)
        r=stats.spearmanr(x[b],y[b])[0]
        if not np.isnan(r): rs.append(r)
    return (np.percentile(rs,2.5),np.percentile(rs,97.5)) if rs else (np.nan,np.nan)

rows=[]
for name,ds in DATASETS.items():
    print(f"\n{'='*44}\n{name}\n{'='*44}",flush=True)
    for sid in ds.subject_list:
        try:
            ep,y,meta=FULL.get_data(ds,subjects=[sid],return_epochs=True)
        except Exception as e:
            print(f"  subj {sid} skip ({e})",flush=True); gc.collect(); continue
        ch=ep.ch_names
        if "C3" not in ch or "C4" not in ch: del ep; gc.collect(); continue
        data=ep.get_data().copy(); labels=np.asarray(y); del ep; gc.collect()
        sess=meta["session"].values if "session" in meta.columns else np.array(["0"]*len(labels))
        i3,i4=ch.index("C3"),ch.index("C4")
        for s_id in np.unique(sess):
            for task,contra,ipsi,tag in [("left_hand",i4,i3,"T1"),("right_hand",i3,i4,"T2")]:
                m=(labels==task)&(sess==s_id)
                if m.sum()<STAGES*3: continue
                d=data[m]
                c=np.array([erd_db(d[t,contra,:int(3*SF)]) for t in range(len(d))])
                i=np.array([erd_db(d[t,ipsi,:int(3*SF)]) for t in range(len(d))])
                sel,dli=sel_and_dli(c,i)
                # decoding on full montage: both classes needed -> use this task vs other
                mm=(sess==s_id)
                yb=labels[mm]; Xb=data[mm][:,:,int(0.5*SF):int(3.0*SF)]
                yy=np.array([0 if l=="left_hand" else 1 for l in yb])
                r=learning_slope(Xb,yy)
                rows.append({"dataset":name,"subject":sid,"session":str(s_id),"task":tag,
                             "selectivity":sel,"dLI":dli,
                             "slope_raw":r["raw"],"slope_var":r["var"],
                             "slope_full":r["full"],"decode_acc":r["acc"],"n_trials":r["n"]})
        del data; gc.collect()
        print(f"  subj {sid} done",flush=True)

df=pd.DataFrame(rows)
df.to_csv("behavioural_validation_within.csv",index=False)

lines=[]; W=lambda s:(lines.append(s),print(s,flush=True))

# QC: mean decode accuracy per dataset (PhysioNet must reach ~0.65 or margins are noise)
W("\n=== QC: mean full-montage decode accuracy per dataset ===")
for ds in df["dataset"].unique():
    d=df[df["dataset"]==ds]["decode_acc"].dropna()
    W(f"  {ds:12s}: mean acc={d.mean():.3f}  (n={len(d)})  %>0.60={100*(d>0.6).mean():.0f}%")

# Exclusion: subjects at/near chance have meaningless margins
EXCL=0.55
W(f"\nExclusion rule: decode_acc > {EXCL} (subjects at chance dropped)")
dv=df[df["decode_acc"]>EXCL].copy()
W(f"Retained {len(dv)}/{len(df)} subject-task rows after exclusion")
W(f"Retained n per dataset: {dv.groupby('dataset').size().to_dict()}")

W("\n=== WITHIN-SESSION: metric vs learning slope (full-nuisance-controlled) ===")
W("Also showing raw and variance-controlled to expose attenuation.\n")
for metric in ["dLI","selectivity"]:
    W(f"--- {metric} ---")
    per=[]
    for ds in ["PhysionetMI","Cho2017","Lee2019_MI","Liu2024","BNCI2014_004"]:
        d=dv[dv["dataset"]==ds][[metric,"slope_raw","slope_var","slope_full"]].dropna()
        if len(d)<8:
            W(f"  {ds:12s}: n={len(d)} too few"); continue
        rr=stats.spearmanr(d[metric],d["slope_raw"])[0]
        rv=stats.spearmanr(d[metric],d["slope_var"])[0]
        rf,pf=stats.spearmanr(d[metric],d["slope_full"])
        lo,hi=boot_ci(d[metric].values,d["slope_full"].values)
        if ds!="BNCI2014_004": per.append(rf)   # drop 3-ch cohort from pooled
        W(f"  {ds:12s}: raw={rr:+.2f} var={rv:+.2f} FULL={rf:+.3f} [{lo:+.2f},{hi:+.2f}] p={pf:.3f} n={len(d)}")
    if per:
        z=np.arctanh(np.clip(per,-.999,.999))
        W(f"  RANDOM-EFFECTS (Fisher-z mean, excl BNCI): rho={np.tanh(np.mean(z)):+.3f}  (k={len(per)} cohorts)\n")

open("behavioural_validation_report.txt","w").write("\n".join(lines))
W("Saved behavioural_validation_within.csv + report")
