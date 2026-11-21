"""Joint robust camera-state filtering and smoothing; callback data only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def psd(a, floor=1e-6):
    a=np.nan_to_num((a+a.T)*.5)
    w,v=np.linalg.eigh(a)
    return (v*np.maximum(w,floor))@v.T


def fit_parameters(train, units, weekday, same_day_weight=3.):
    n=len(units)
    scales=np.array([max(1.,float(train.loc[train.unit.eq(u),'people'].median())) for u in units])
    matrices=[]; means=[]; weights=[]; differences=[]; second=[]
    for _,g in train.groupby('recording',sort=True):
        p=g.pivot_table(index='minute_of_day',columns='unit',values='people').reindex(columns=units).sort_index()
        grid=np.arange(int(p.index.min()),int(p.index.max())+1)
        p=p.reindex(grid)
        z=np.arcsinh(p.to_numpy()/scales)
        mean=np.nanmean(z,axis=0)
        means.append(mean)
        weights.append(same_day_weight if str(g.weekday.iloc[0])==str(weekday) else 1.)
        matrices.append(z-mean)
        differences.append(np.diff(z,axis=0))
        second.append(z[2:]-2*z[1:-1]+z[:-2])
    means=np.asarray(means); weights=np.asarray(weights)[:,None]
    mu=np.nansum(means*weights,axis=0)/np.maximum(1.,np.sum(np.isfinite(means)*weights,axis=0))
    x=np.concatenate(matrices)
    cov=pd.DataFrame(x).cov(min_periods=12).to_numpy()
    cov=psd(cov,1e-5)
    sd=np.sqrt(np.maximum(np.diag(cov),1e-5))
    corr=cov/(sd[:,None]*sd[None,:])
    corr=psd(.9*corr+.1*np.eye(n),.03)
    # For iid Gaussian observation noise, median second-difference squared is
    # 0.4549364 * 6 * observation variance. Fit only historical observations.
    r=np.nanmedian(np.concatenate(second)**2,axis=0)/(6*.4549364)
    r=np.where(np.isfinite(r),r,np.nanmedian(r))
    r=np.clip(r,1e-4,.5)
    dif=np.concatenate(differences)
    innovation=pd.DataFrame(dif).cov(min_periods=12).to_numpy()
    innovation=np.nan_to_num(innovation)
    innovation=psd(innovation-2*np.diag(r),1e-5)
    return scales,mu,cov,corr,r,innovation


def reconstruct(train,released,targets,qscale=.2,velocity=.0,noise_scale=1.,robust=0.,same_day_weight=3.,covariance='level',reversion=0.):
    units=sorted(set(train.unit))
    if not units or released.empty:
        return {}
    n=len(units)
    scales,mu,cov,corr,r,innovation=fit_parameters(train,units,targets.weekday.iloc[0],same_day_weight)
    lo=int(min(released.minute_of_day.min(),targets.now_minute_of_day.min() if targets.now_minute_of_day.notna().any() else targets.minute_of_day.min()))
    hi=int(max(released.minute_of_day.max(),targets.minute_of_day.max()))
    times=np.arange(lo,hi+1)
    obs=released.loc[released.unit.isin(units)].pivot_table(index='minute_of_day',columns='unit',values='people').reindex(index=times,columns=units)
    z=np.arcsinh(obs.to_numpy()/scales)-mu
    if covariance=='innovation':
        spatial=innovation
    else:
        spatial=np.sqrt(r)[:,None]*corr*np.sqrt(r)[None,:]
    if reversion>0:
        d=n;F=(1-reversion)*np.eye(n)
        Q=(1-(1-reversion)**2)*qscale*cov
        P=qscale*cov
    elif velocity>0:
        d=2*n
        F=np.block([[np.eye(n),np.eye(n)],[np.zeros((n,n)),velocity*np.eye(n)]])
        Q=np.block([[qscale*spatial,np.zeros((n,n))],[np.zeros((n,n)),qscale*.03*spatial]])
        P=np.block([[cov*10.,np.zeros((n,n))],[np.zeros((n,n)),spatial]])
    else:
        d=n;F=np.eye(n);Q=qscale*spatial;P=cov*10.
    Q=psd(Q,1e-8);P=psd(P,1e-5)
    state=np.zeros(d);filtered=[];covs=[];priors=[];prior_covs=[]
    for row in z:
        prior=F@state;Pc=F@P@F.T+Q
        good=np.flatnonzero(np.isfinite(row))
        if len(good):
            residual=row[good]-prior[good]
            local_r=noise_scale*r[good]
            if robust:
                # Student-like robust observation weighting, based only on this
                # observed innovation and the causal prior uncertainty.
                standardized=residual**2/np.maximum(np.diag(Pc)[good]+local_r,1e-6)
                local_r=local_r*np.maximum(1.,standardized/robust)
            S=Pc[np.ix_(good,good)]+np.diag(local_r)
            gain=cho_solve(cho_factor(S,lower=True,check_finite=False),Pc[good],check_finite=False).T
            state=prior+gain@residual
            P=Pc-gain@Pc[good]
            P=(P+P.T)*.5
        else:
            state=prior;P=Pc
        filtered.append(state.copy());covs.append(P.copy());priors.append(prior);prior_covs.append(Pc)
    filtered=np.asarray(filtered);smooth=filtered.copy()
    if targets.kind.isin(['hidden','missing']).any():
        for t in range(len(times)-2,-1,-1):
            J=cho_solve(cho_factor(prior_covs[t+1],lower=True,check_finite=False),F@covs[t],check_finite=False).T
            smooth[t]=filtered[t]+J@(smooth[t+1]-priors[t+1])
    lookup={u:i for i,u in enumerate(units)}
    powers={0:np.eye(d)}
    out={}
    for target in targets.itertuples(index=False):
        j=lookup.get(target.unit)
        if j is None:
            continue
        if target.kind in ('forecast','hidden-future'):
            if not np.isfinite(target.now_minute_of_day):
                continue
            origin=int(target.now_minute_of_day)-lo
            if origin<0 or origin>=len(times):
                continue
            h=max(0,int(target.minute_of_day-target.now_minute_of_day))
            if h not in powers:
                powers[h]=np.linalg.matrix_power(F,h)
            latent=(powers[h]@filtered[origin])[j]
        else:
            pos=int(target.minute_of_day)-lo
            if pos<0 or pos>=len(times):
                continue
            latent=smooth[pos,j]
        out[target.id]=max(0.,float(scales[j]*np.sinh(np.clip(latent+mu[j],-8.,8.))))
    return out


def predict_fold(train,released,targets,baseline,config=None):
    config=dict(config or {})
    blend=float(config.pop('blend',.25))
    kinds=config.pop('kinds',('forecast','hidden','hidden-future','missing'))
    target=targets.loc[targets.kind.isin(kinds)]
    result=baseline.copy()
    if target.empty:
        return result
    pred=reconstruct(train,released,target,**config)
    take=result.id.isin(pred)
    result.loc[take,'people']=(1-blend)*result.loc[take,'people']+blend*result.loc[take,'id'].map(pred)
    return result
