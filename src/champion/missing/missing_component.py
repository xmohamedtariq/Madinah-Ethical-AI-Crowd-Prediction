"""Missing-only boundary-anchored reconstruction using callback data alone."""
import numpy as np
import pandas as pd

_KERNEL_CONFIG = {"method": "level", "radius": 50.0, "ridge": 0.25, "blend": 0.5,
          "inner": True, "peer_weight": 0.7, "time_length": 10.0}


def _interp(times, values, query, max_gap=3.0):
    times, values, query = np.asarray(times), np.asarray(values), np.asarray(query)
    pos = np.searchsorted(times, query)
    out = np.full(query.shape, np.nan, dtype=float)
    ok = pos < len(times)
    exact = np.zeros(query.shape, bool)
    exact[ok] = times[pos[ok]] == query[ok]
    out[exact] = values[pos[exact]]
    inside = (~exact) & (pos > 0) & (pos < len(times))
    ix = np.flatnonzero(inside)
    if len(ix):
        lo, hi = pos[ix] - 1, pos[ix]
        good = times[hi] - times[lo] <= max_gap
        ix, lo, hi = ix[good], lo[good], hi[good]
        frac = (query[ix] - times[lo]) / (times[hi] - times[lo])
        out[ix] = values[lo] * (1 - frac) + values[hi] * frac
    return out


def _delta_one(released, camera, query, config):
    """Return linear bridge and learned departure; unsupported positions are NaN."""
    own = released.loc[released.unit.eq(camera)].sort_values("minute_of_day")
    ot = own.minute_of_day.to_numpy(float)
    oy = own.people.to_numpy(float)
    query = np.asarray(query, dtype=float)
    linear = np.full(len(query), np.nan)
    delta = np.zeros(len(query))
    if len(ot) < 16:
        return linear, delta
    insertion = np.searchsorted(ot, query)
    peers = {u: (g.minute_of_day.to_numpy(float), g.people.to_numpy(float))
             for u, g in released.sort_values("minute_of_day").groupby("unit") if u != camera}
    for right in np.unique(insertion):
        mask = insertion == right
        if right == 0 or right >= len(ot):
            continue
        a, b = ot[right-1], ot[right]
        if b-a > 90:
            continue
        q = query[mask]
        fraction = (q-a)/(b-a)
        linear[mask] = oy[right-1]*(1-fraction)+oy[right]*fraction
        mid=(a+b)/2
        near=np.abs(ot-mid) <= config["radius"]
        t,y=ot[near],oy[near]
        if len(t)<14:
            continue
        # Detrending prevents a shared broad rise from dominating shape fits.
        trend=np.column_stack([np.ones(len(t)),(t-mid)/30])
        yr=y-trend@np.linalg.lstsq(trend,y,rcond=None)[0]
        possibilities=[]
        query_all=np.r_[a,q,b]
        for unit,(pt,py) in peers.items():
            xt=_interp(pt,py,t)
            xq=_interp(pt,py,query_all)
            valid=np.isfinite(xt)
            if valid.sum()<14 or not np.isfinite(xq).all():
                continue
            xres=xt[valid]-trend[valid]@np.linalg.lstsq(trend[valid],xt[valid],rcond=None)[0]
            corr=np.corrcoef(xres,yr[valid])[0,1] if np.std(xres)>1e-5 and np.std(yr[valid])>1e-5 else 0
            if np.isfinite(corr):
                possibilities.append((abs(corr),xt,xq))
        possibilities.sort(key=lambda z:z[0],reverse=True)
        possibilities=possibilities[:4]
        while possibilities:
            X=np.column_stack([z[1] for z in possibilities])
            good=np.isfinite(X).all(axis=1)
            if good.sum()>=14:
                break
            possibilities.pop()
        if not possibilities:
            continue
        X=X[good]; yy=y[good]; tt=t[good]
        Q=np.column_stack([z[2] for z in possibilities])
        if config["method"] in ("level","difference"):
            if config["method"]=="difference":
                dt=np.diff(tt)
                usable=(dt>0)&(dt<=3)
                XX=np.diff(X,axis=0)[usable]/dt[usable,None]
                Y=np.diff(yy)[usable]/dt[usable]
                weights=np.exp(-np.abs(tt[1:][usable]-mid)/config["radius"])
            else:
                design=np.column_stack([np.ones(len(tt)),(tt-mid)/30])
                XX=X-design@np.linalg.lstsq(design,X,rcond=None)[0]
                Y=yy-design@np.linalg.lstsq(design,yy,rcond=None)[0]
                weights=np.exp(-np.abs(tt-mid)/config["radius"])
            if len(Y)<10:
                continue
            scale=np.maximum(np.std(XX,axis=0),1.)
            Z=XX/scale
            A=Z.T@(weights[:,None]*Z)+np.eye(Z.shape[1])*len(Y)*config["ridge"]
            beta=np.linalg.solve(A,Z.T@(weights*Y))/scale
            innovation=Q[1:-1]-((1-fraction[:,None])*Q[0]+fraction[:,None]*Q[-1])
            correction=innovation@beta
        else:
            # Conditional local kernel with temporal and multivariate peer similarity.
            scale=np.maximum(np.std(X,axis=0),3.)
            XX=X/scale; QQ=Q/scale
            lt=config["time_length"];rho=config["peer_weight"]
            kd=lambda u,v:np.exp(-.5*np.mean((u[:,None,:]-v[None,:,:])**2,axis=2))
            K=(1-rho)*np.exp(-.5*((tt[:,None]-tt[None,:])/lt)**2)+rho*kd(XX,XX)
            k=(1-rho)*np.exp(-.5*((query_all[:,None]-tt[None,:])/lt)**2)+rho*kd(QQ,XX)
            design=np.column_stack([np.ones(len(tt)),(tt-mid)/30])
            residual=yy-design@np.linalg.lstsq(design,yy,rcond=None)[0]
            values=k@np.linalg.solve(K+np.eye(len(tt))*config["ridge"],residual)
            correction=values[1:-1]-(1-fraction)*values[0]-fraction*values[-1]
        cap=max(10.,0.6*max(oy[right-1],oy[right]))
        delta[mask]=np.clip(correction,-cap,cap)
    return linear,delta


def _inner_alpha(released, target_group, config):
    """Select correction strength using removed contiguous observed blocks only."""
    errors={a:[] for a in [0.,0.25,0.5,0.75,1.]}
    for camera, targets in target_group.groupby("unit"):
        own=released.loc[released.unit.eq(camera)].sort_values("minute_of_day")
        if len(own)<55:
            continue
        times=own.minute_of_day.to_numpy(float)
        ins=np.searchsorted(times,targets.minute_of_day.to_numpy(float))
        valid=(ins>0)&(ins<len(times))
        lengths=times[ins[valid]]-times[ins[valid]-1]-1
        length=int(np.clip(np.median(lengths),3,20)) if len(lengths) else 7
        for start in np.unique(np.linspace(15,len(own)-length-15,3).astype(int)):
            block=own.iloc[start:start+length]
            if len(block)!=length or block.minute_of_day.max()-block.minute_of_day.min()>length+2:
                continue
            # Remove labels before EVERY feature and preprocessing computation.
            permitted=released.drop(index=block.index)
            lin,delta=_delta_one(permitted,camera,block.minute_of_day.to_numpy(float),config)
            good=np.isfinite(lin)
            if good.sum()<max(2,length//2):
                continue
            truth=block.people.to_numpy(float)[good]
            denominator=max(float(np.mean(np.abs(lin[good]-truth))),1.)
            for a in errors:
                errors[a].append(float(np.mean(np.abs(np.maximum(0,lin[good]+a*delta[good])-truth)))/denominator)
    if not errors[0.]:
        return 0.
    scores={a:float(np.mean(v)) for a,v in errors.items()}
    best=min(scores,key=scores.get)
    # Demand a small inner gain to avoid selecting a correction on numerical noise.
    return best if scores[best]<scores[0.]*.995 else 0.


def _predict_kernel(train,released,targets,baseline):
    # train is permitted but not needed by these same-recording missing models.
    result=baseline.copy(deep=True)
    missing=targets.loc[targets.kind.eq("missing")]
    if missing.empty:
        return result
    predictions=result.set_index("id").people.to_dict()
    for recording,group in missing.groupby("recording"):
        observed=released.loc[released.recording.eq(recording)].copy()
        alpha=_inner_alpha(observed,group,_KERNEL_CONFIG) if _KERNEL_CONFIG.get("inner",True) else _KERNEL_CONFIG.get("alpha",.5)
        if alpha<=0:
            continue
        for camera,part in group.groupby("unit"):
            linear,delta=_delta_one(observed,camera,part.minute_of_day.to_numpy(float),_KERNEL_CONFIG)
            good=np.isfinite(linear)
            for index,row in enumerate(part.itertuples(index=False)):
                if good[index]:
                    anchor=predictions[row.id] if _KERNEL_CONFIG.get("anchor","linear")=="baseline" else linear[index]
                    proposed=max(0.,float(anchor+alpha*delta[index]))
                    predictions[row.id]=(1-_KERNEL_CONFIG["blend"])*predictions[row.id]+_KERNEL_CONFIG["blend"]*proposed
    result["people"]=result.id.map(predictions)
    return result

_KERNEL_CONFIG = {'method': 'kernel', 'radius': 65.0, 'ridge': 0.15, 'blend': 0.35, 'inner': True, 'peer_weight': 0.25, 'time_length': 15.0, 'anchor': 'baseline'}


"""Gap-shape regression trained only from permitted, contiguously masked rows."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

_GAP_CONFIG = dict(source="all", grouping="pooled", peers=True, loss="absolute_error",
              blend=.4, leaves=12, ridge=False, additive=False)


def interp(t, y, q, maxgap=3.):
    q = np.asarray(q, float)
    j = np.searchsorted(t, q)
    z = np.full(q.shape, np.nan)
    valid = j < len(t)
    exact = np.zeros(q.shape, bool)
    exact[valid] = t[j[valid]] == q[valid]
    z[exact] = y[j[exact]]
    ok = (~exact) & (j > 0) & (j < len(t))
    ix = np.flatnonzero(ok)
    if len(ix):
        a, b = j[ix]-1, j[ix]
        good = t[b]-t[a] <= maxgap
        ix, a, b = ix[good], a[good], b[good]
        f = (q[ix]-t[a])/(t[b]-t[a])
        z[ix] = (1-f)*y[a]+f*y[b]
    return z


def bank(frame):
    return {u:(g.minute_of_day.to_numpy(float),g.people.to_numpy(float))
            for u,g in frame.sort_values("minute_of_day").groupby("unit")}


def features(data, unit, q, units, own=None):
    t,y = data[unit] if own is None else own
    q=np.asarray(q,float);j=np.searchsorted(t,q)
    valid=(j>0)&(j<len(t))
    jj=np.clip(j,1,len(t)-1);a=t[jj-1];b=t[jj]
    ya=y[jj-1];yb=y[jj];dt=b-a;f=(q-a)/dt
    linear=(1-f)*ya+f*yb
    scale=np.maximum(10.,(np.abs(ya)+np.abs(yb))/2)
    # This scale is computed from observed boundaries, never the pseudo labels.
    X=[f,dt/20.,np.log1p(np.maximum(linear,0)),(yb-ya)/scale,
       (yb-ya)/dt/scale,(q-900)/300.,np.full(len(q),units.index(unit)/max(1,len(units)-1))]
    for distance in [1,3,6,10]:
        li=np.maximum(0,jj-1-distance);ri=np.minimum(len(t)-1,jj+distance)
        left=(ya-y[li])/np.maximum(a-t[li],1.)/scale
        right=(y[ri]-yb)/np.maximum(t[ri]-b,1.)/scale
        X.extend([left,right,(a-t[li])/20.,(t[ri]-b)/20.])
        # Cubic Hermite correction is only a covariate; learners see raw-MAE labels.
        hermite=(f**3-2*f*f+f)*dt*left+(f**3-f*f)*dt*right
        X.append(hermite)
    signed=[];strength=[]
    for peer in units:
        if peer==unit or peer not in data:
            X.extend([np.full(len(q),np.nan) for _ in range(5)])
            continue
        pt,py=data[peer]
        pq=interp(pt,py,q);pa=interp(pt,py,a);pb=interp(pt,py,b)
        pscale=np.maximum(10.,(np.abs(pa)+np.abs(pb))/2)
        pin=(pq-((1-f)*pa+f*pb))/pscale
        slope=(pb-pa)/pscale
        beta=np.zeros(len(q));corr=np.zeros(len(q));support=np.zeros(len(q))
        for right in np.unique(jj):
            m=jj==right;mid=(t[right-1]+t[right])/2
            nearby=np.abs(t-mid)<=60
            tt=t[nearby];yy=y[nearby];xx=interp(pt,py,tt)
            ok=np.isfinite(xx)
            if ok.sum()<12:
                continue
            tt=tt[ok];yy=yy[ok];xx=xx[ok]
            D=np.column_stack([np.ones(len(tt)),(tt-mid)/30])
            yr=yy-D@np.linalg.lstsq(D,yy,rcond=None)[0]
            xr=xx-D@np.linalg.lstsq(D,xx,rcond=None)[0]
            covariance=np.dot(yr,xr)
            v=np.dot(xr,xr)
            beta[m]=covariance/(v+25*len(tt))
            corr[m]=covariance/np.sqrt(max(v*np.dot(yr,yr),1e-12))
            support[m]=len(tt)/120.
        correction=beta*pscale*pin/scale
        X.extend([pin,slope,corr,correction,support])
        signed.append(correction);strength.append(np.abs(corr))
    if signed:
        s=np.stack(signed,axis=1);w=np.stack(strength,axis=1)
        ok=np.isfinite(s)
        denominator=np.maximum(np.sum(w*ok,axis=1),.1)
        X.extend([np.sum(np.nan_to_num(s)*w,axis=1)/denominator,
                  np.max(w*ok,axis=1),np.sum(ok,axis=1)/max(1,len(units)-1)])
    else:
        X.extend([np.zeros(len(q)) for _ in range(3)])
    return np.column_stack(X),linear,scale,valid&(dt<=90)


def dataset(train,released,units,configuration):
    records=[]
    if configuration["source"]=="all":
        records.extend((str(r),bank(g)) for r,g in train.groupby("recording"))
    records.extend((str(r),bank(g)) for r,g in released.groupby("recording"))
    XX=[];YY=[];WW=[];UU=[]
    for recording,data in records:
        for unit in units:
            if unit not in data:
                continue
            t,y=data[unit]
            if len(t)<35:
                continue
            for length in [5,10,16]:
                for start in range(8,len(t)-10,15):
                    hidden=(t>=t[start])&(t<t[start]+length)
                    if hidden.sum()<max(3,length//2):
                        continue
                    keep=~hidden
                    q=t[hidden]
                    x,linear,scale,valid=features(data,unit,q,units,(t[keep],y[keep]))
                    if not valid.all():
                        continue
                    # Every feature was built after the entire pseudo block was removed.
                    XX.append(x);YY.append((y[hidden]-linear)/scale)
                    WW.append(scale);UU.extend([unit]*len(q))
    return np.concatenate(XX),np.concatenate(YY),np.concatenate(WW),np.array(UU)


def _predict_gapshape(train,released,targets,baseline):
    result=baseline.copy(deep=True)
    target=targets.loc[targets.kind.eq("missing")]
    if target.empty:
        return result
    units=sorted(set(released.unit)&set(train.unit))
    if len(units)<2:
        return result
    x,y,w,uu=dataset(train,released,units,_GAP_CONFIG)
    if len(y)<200:
        return result
    # If peer features are disabled, retain only own-boundary and slope covariates.
    columns=slice(None) if _GAP_CONFIG["peers"] else slice(0,27)
    predictions=result.set_index("id").people.to_dict()
    for key,part in target.groupby("unit" if _GAP_CONFIG["grouping"]=="camera" else lambda i:0):
        keep=uu==key if _GAP_CONFIG["grouping"]=="camera" else np.ones(len(y),bool)
        if keep.sum()<150:
            continue
        if _GAP_CONFIG["ridge"]:
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            model=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=200.))
            model.fit(x[keep,columns],y[keep],ridge__sample_weight=w[keep])
        else:
            model=HistGradientBoostingRegressor(loss=_GAP_CONFIG["loss"],max_iter=150,
                  learning_rate=.05,max_leaf_nodes=_GAP_CONFIG["leaves"],l2_regularization=8.,
                  min_samples_leaf=35,early_stopping=False,random_state=19)
            model.fit(x[keep,columns],y[keep],sample_weight=w[keep])
        for recording,g in part.groupby("recording"):
            data=bank(released.loc[released.recording.eq(recording)])
            for unit,pp in g.groupby("unit"):
                if unit not in data or unit not in units or len(data[unit][0])<15:
                    continue
                xx,linear,scale,valid=features(data,unit,pp.minute_of_day.to_numpy(float),units)
                residual=np.clip(model.predict(xx[:,columns]),-.6,.6)*scale
                for index,row in enumerate(pp.itertuples(index=False)):
                    if valid[index]:
                        anchor=predictions[row.id] if _GAP_CONFIG["additive"] else linear[index]
                        proposed=max(0.,float(anchor+residual[index]))
                        predictions[row.id]=(1-_GAP_CONFIG["blend"])*predictions[row.id]+_GAP_CONFIG["blend"]*proposed
    result["people"]=result.id.map(predictions)
    return result

_GAP_CONFIG = {'source': 'all', 'grouping': 'camera', 'peers': True, 'loss': 'absolute_error', 'blend': 0.4, 'leaves': 12, 'ridge': False, 'additive': False}



ENSEMBLE_WEIGHTS = (1., 1.)
def predict_fold(train, released, targets, baseline):
    result = baseline.copy(deep=True)
    if not targets.kind.eq("missing").any():
        return result
    k = _predict_kernel(train, released, targets, baseline).set_index("id").people
    g = _predict_gapshape(train, released, targets, baseline).set_index("id").people
    b = result.people.to_numpy(float)
    values = b + ENSEMBLE_WEIGHTS[0]*(result.id.map(k).to_numpy()-b) + ENSEMBLE_WEIGHTS[1]*(result.id.map(g).to_numpy()-b)
    result["people"] = np.maximum(0., values)
    return result

ENSEMBLE_WEIGHTS = (1.0, 1.0)
