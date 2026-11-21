"""Forecast challenger: causal features, matching visible-camera sets and masked views.

Only the four callback arguments contain modeling data. No evaluator/data file reads.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from threadpoolctl import threadpool_limits

DEFAULT = dict(model='hgb', target='weighted_sqrt', mask='masked', blend=.10,
               iterations=180, leaves=15, l2=20., peer=True)


class History:
    def __init__(self, rows, units):
        self.units = tuple(units)
        self.series = {}
        self.cache = {}
        for unit, group in rows.groupby('unit', sort=False):
            g = group.sort_values('minute_of_day')
            self.series[unit] = (g.minute_of_day.to_numpy(float), g.people.to_numpy(float))

    def state(self, unit, now):
        key = (unit, int(now))
        if key in self.cache:
            return self.cache[key]
        if unit not in self.series:
            return None
        t, y = self.series[unit]
        end = np.searchsorted(t, now, side='right')
        t, y = t[:end], y[:end]
        if len(y) < 3:
            return None
        last = float(y[-1]); scale = np.sqrt(last + 10.)
        f = {'last': last, 'loglast': np.log1p(last), 'age': float(now-t[-1]),
             'history_minutes': float(now-t[0]), 'nobs': len(y)}
        for w in (3, 7, 15, 30, 60):
            ix = t >= now-w
            tt, yy = t[ix], y[ix]
            if len(yy) == 0:
                tt, yy = t[-1:], y[-1:]
            f[f'n{w}'] = len(yy)
            f[f'coverage{w}'] = len(yy)/(w+1.)
            f[f'mean{w}'] = float(np.mean(yy))
            f[f'median{w}'] = float(np.median(yy))
            f[f'std{w}'] = float(np.std(yy))
            f[f'diffmean{w}'] = (last-f[f'mean{w}'])/scale
            if len(yy) >= 2:
                n = min(3, max(1,len(yy)//2))
                dt = np.median(tt[-n:])-np.median(tt[:n])
                slope = (np.median(yy[-n:])-np.median(yy[:n]))/max(1.,dt)
                f[f'slope{w}'] = float(slope)
                f[f'ratevol{w}'] = float(np.median(np.abs(np.diff(yy))/np.sqrt(np.maximum(1.,np.diff(tt)))))
            else:
                f[f'slope{w}'] = f[f'ratevol{w}'] = 0.
        for w in (1,3,5,10,20):
            f[f'delta{w}'] = float(last-np.interp(now-w,t,y))
        self.cache[key] = f
        return f

    def features(self, unit, now, h, weekday, peer=True):
        own = self.state(unit,now)
        if own is None:
            return None
        f = dict(own)
        scale = np.sqrt(own['last']+10.)
        f.update(now=float(now), target_minute=float(now+h), horizon=float(h),
                 friday=float(str(weekday)=='Fri'), cam=float(ord(unit[-1])-65),
                 sin_time=np.sin((now+h)*np.pi/720),cos_time=np.cos((now+h)*np.pi/720),
                 damped_trend=own['slope7']*5*(1-np.exp(-h/5)),
                 trend_h=own['slope15']*h)
        levels=[];rates=[]
        if peer:
            for pu in self.units:
                ps = None if pu==unit else self.state(pu,now)
                valid = ps is not None and ps['age'] <= 10
                f['peer_'+pu+'_level'] = ps['last'] if valid else np.nan
                f['peer_'+pu+'_relative'] = ps['last']/(own['last']+10.) if valid else np.nan
                f['peer_'+pu+'_slope'] = ps['slope7'] if valid else np.nan
                f['peer_'+pu+'_age'] = ps['age'] if valid else 60.
                if valid:
                    levels.append(ps['last']);rates.append(ps['slope7'])
        f['peer_n']=len(levels)
        f['peer_mean']=float(np.mean(levels)) if levels else 0.
        f['peer_trend']=float(np.mean(rates)) if rates else 0.
        f['relative_plaza']=(own['last']+5.)/(f['peer_mean']+5.)
        return f


def training_rows(train, released, targets, cfg):
    # Mask *metadata* is known task structure. Counts in released are not used to fit.
    visible = sorted(released.unit.unique())
    mask_positions = {u:set(g.minute_of_day.astype(int)) for u,g in released.groupby('unit')}
    query_origins = sorted(targets.loc[targets.kind.eq('forecast'),'now_minute_of_day'].dropna().astype(int).unique())
    features=[];ys=[];weights=[]
    for rec, raw in train.groupby('recording',sort=False):
        raw=raw[raw.unit.isin(visible)].copy()
        if not len(raw):continue
        weekday=str(raw.weekday.iloc[0])
        exact = raw[[int(m) in mask_positions.get(u,set()) for u,m in zip(raw.unit,raw.minute_of_day)]]
        views=[('dense',raw)] if cfg['mask']=='dense' else [('dense',raw),('masked',exact)]
        if cfg['mask']=='masked':views=[('masked',exact)]
        for view_name,view in views:
            if len(view)<30:continue
            hist=History(view,visible)
            for unit,truth in raw.groupby('unit',sort=False):
                truth=truth.sort_values('minute_of_day')
                obs=dict(zip(truth.minute_of_day.astype(int),truth.people.astype(float)))
                times=truth.minute_of_day.to_numpy(int)
                if len(times)<24:continue
                orig = sorted(set(times[12:-5:10]).union(t for t in query_origins if times[0]+10<=t<=times[-1]-1))
                for now in orig:
                    for h in (1,3,5,8,12,15):
                        if now+h not in obs:continue
                        f=hist.features(unit,now,h,weekday,cfg['peer'])
                        if f is None or f['age']>20:continue
                        features.append(f);ys.append(obs[now+h])
                        weights.append(1. if view_name=='dense' else 2.)
    X=pd.DataFrame(features)
    return X,np.asarray(ys,float),np.asarray(weights,float),visible


def predict_config(train,released,targets,baseline,cfg):
    output=baseline[['id','people']].copy()
    query=targets[targets.kind.eq('forecast')].copy()
    if query.empty:return output
    with threadpool_limits(limits=1):
        X,y,view_weights,visible=training_rows(train,released,targets,cfg)
        if len(X)<100:return output
        last=X['last'].to_numpy()
        if cfg['target']=='raw':scale=np.ones(len(y))
        else:scale=np.sqrt(last+10.)
        target=(y-last)/scale
        weight=view_weights*(scale if cfg['target']=='weighted_sqrt' else 1.)
        if cfg['model']=='extra':
            model=ExtraTreesRegressor(n_estimators=140,max_features=.8,min_samples_leaf=18,
                                      n_jobs=1,random_state=7331)
            fill=X.median().fillna(0);Xfit=X.fillna(fill)
        else:
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=cfg['iterations'],
                   max_leaf_nodes=cfg['leaves'],max_depth=7,learning_rate=.055,l2_regularization=cfg['l2'],
                   min_samples_leaf=35,early_stopping=False,random_state=7331)
            Xfit=X
        model.fit(Xfit,target,sample_weight=weight)
        hist=History(released,visible);rows=[];ids=[];horizons=[]
        supported=set(train.unit.unique())
        for r in query.itertuples(index=False):
            if r.unit not in supported:continue
            f=hist.features(r.unit,int(r.now_minute_of_day),int(r.horizon),r.weekday,cfg['peer'])
            if f is None or f['age']>20:continue
            rows.append(f);ids.append(r.id);horizons.append(int(r.horizon))
        if not rows:return output
        val=pd.DataFrame(rows).reindex(columns=X.columns)
        valscale=np.ones(len(val)) if cfg['target']=='raw' else np.sqrt(val['last'].to_numpy()+10.)
        pred=np.maximum(0.,val['last'].to_numpy()+valscale*model.predict(val.fillna(fill) if cfg['model']=='extra' else val))
        base=output.set_index('id').people
        blend=cfg['blend']
        if isinstance(blend,str):
            blend=np.where(np.asarray(horizons)>=8,.35,.1)
        proposed=(1-blend)*base.loc[ids].to_numpy()+blend*pred
        changes=dict(zip(ids,proposed))
        output['people']=[changes.get(i,p) for i,p in zip(output.id,output.people)]
        print('FORECAST_FIT',len(X),'rows',len(ids),'predictions',cfg,flush=True)
    return output


def predict_fold(train,released,targets,baseline):
    return predict_config(train,released,targets,baseline,DEFAULT)

