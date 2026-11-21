"""Madinah AutoGluon V10: causal trajectory analog forecasting experiment.

A local shadow backtest ONLY. No Kaggle submission generation or upload.
Uses Sep 11 to freeze group choices, Sep 12 as a reused shadow check.
All historical analog curves have earlier day than the forecast target.
Current-recording measurements are limited to each target's `now`.
Not an independent validation and not Kaggle's custom metric.

Usage:
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v10_trajectory.py --mode audit
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v10_trajectory.py --mode dry-run
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v10_trajectory.py --mode validate --time-limit 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import madinah_autogluon_v1 as v1
import madinah_autogluon_v2 as v2
import madinah_autogluon_v6_augmented as v6
import madinah_autogluon_v7_step_change as v7

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
MODEL_ROOT = ROOT / 'models' / 'ag_v10_trajectory'
V7_FILE = REPORTS / 'ag_v7_cv_predictions.csv'
V7_SUMMARY = REPORTS / 'ag_v7_summary.json'
KEYS = ['recording', 'day', 'unit', 'minute_of_day', 'kind']
FOCUS = ('cam_E', 'cam_A', 'cam_F')
ALPHAS = (.25, .50, .75, 1.0)
SHIFTS = (-8., -4., 0., 4., 8.)
FEATURES = [
    'tr_add', 'tr_ratio', 'tr_affine', 'tr_median', 'tr_hist_future',
    'tr_delta', 'tr_slope', 'tr_num_curves', 'tr_num_matches',
    'tr_best_loss', 'tr_best_shift', 'tr_current_last', 'tr_last_age',
    'tr_delta15', 'tr_reference', 'tr_relative_error',
]


def metric(y, p):
    a, b = np.asarray(y, float), np.asarray(p, float)
    if not len(a):
        return math.nan
    return float(np.mean(np.abs(a - b)))


def gain(old, new):
    return float(100 * (old-new)/old) if old > 1e-12 else 0.


def bounded(arr):
    return np.clip(np.asarray(arr, float), 0, 100000.)


def get_reference():
    if not V7_FILE.is_file() or not V7_SUMMARY.is_file():
        raise FileNotFoundError('Missing V7 CV results; run V7 validation first.')
    df = pd.read_csv(V7_FILE, low_memory=False)
    needed = set(KEYS + ['people', 'v7_accepted'])
    if needed.difference(df.columns) or df.duplicated(KEYS).any() or df.empty:
        raise ValueError('Bad V7 checkpoint: missing columns, duplicates, or empty.')
    df['day'] = df['day'].astype(str)
    for col in ('people', 'v7_accepted'):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError('Invalid ' + col)
    summary = json.loads(V7_SUMMARY.read_text(encoding='utf-8'))
    actual = metric(df.people, df.v7_accepted)
    if int(summary.get('rows',-1)) != len(df) or not np.isclose(actual, float(summary.get('accepted_mae',-1)), atol=2e-4, rtol=0):
        raise ValueError(f'V7 checkpoint mismatch: n={len(df)} mae={actual:.6f}')
    days = sorted(df.day.unique())
    if len(days) != 2:
        raise ValueError('Expected Sep 11 / Sep 12 V7 folds')
    return df, days


def interp_at(times, values, q, max_nearest=10., max_bracket=18.):
    """Query a PRIOR RELEASED day's curve; two-sided historical interpolation is legal."""
    if len(times) == 0 or not np.isfinite(q):
        return math.nan
    j = int(np.searchsorted(times, q, side='left'))
    if j < len(times) and abs(times[j]-q) < 1e-7:
        return float(values[j])
    if 0 < j < len(times) and times[j]-times[j-1] <= max_bracket:
        w = (q-times[j-1])/(times[j]-times[j-1])
        return float(values[j-1]*(1-w)+values[j]*w)
    candidates = [(abs(times[k]-q), float(values[k])) for k in (j-1, j) if 0 <= k < len(times)]
    if candidates:
        d, val = min(candidates, key=lambda item:item[0])
        if d <= max_nearest:
            return val
    return math.nan


def curve_index(history):
    out = defaultdict(list)
    for (day, rec, cam), group in history.groupby(['day','recording','unit'],sort=False, observed=True):
        grp=group.dropna(subset=['minute_of_day','people']).sort_values('minute_of_day')
        if grp.empty:
            continue
        grp=grp.drop_duplicates('minute_of_day',keep='last')
        ts=grp.minute_of_day.to_numpy(float)
        val=grp.people.to_numpy(float)
        out[str(cam)].append((str(day), str(rec), str(grp.weekday.iloc[0]), ts, val))
    for cam in out:
        out[cam].sort(key=lambda tup:(tup[0],tup[1]))
    return out


def current_idx(visible):
    return v1.obs_index(visible)


def trajectory_features(targets, visible, history, anchor, *, shifts=SHIFTS):
    """For forecast only: match a prior-day full curve to the released current prefix.

    Never reads the target's true `people`. Requires earlier-day analogs and
    current-recording observations strictly at or before each target's `now`.
    """
    history_curves=curve_index(history)
    current=current_idx(visible)
    out=[]
    for i, row in enumerate(targets.itertuples(index=False)):
        base=float(anchor[i])
        if str(row.kind) != 'forecast':
            out.append({f:math.nan for f in FEATURES})
            continue
        day, rec, cam = str(row.day), str(row.recording), str(row.unit)
        now=float(row.now_minute_of_day)
        target=float(row.minute_of_day)
        if not np.isfinite(now) or now > target + 1e-9:
            raise ValueError('Forecast cutoff later than target')
        pair=current.get(rec,{}).get(cam)
        if pair is None:
            out.append({f:math.nan for f in FEATURES})
            continue
        t_all, y_all = pair
        cut=int(np.searchsorted(t_all, now, side='right'))
        ts=t_all[:cut]
        ys=y_all[:cut]
        if not len(ts) or now-ts[-1] > 20.:
            out.append({f:math.nan for f in FEATURES})
            continue
        last=float(ys[-1]); age=float(now-ts[-1])
        recent=(ts >= now-35.)
        tr=ts[recent]; yr=ys[recent]
        if len(tr)>8:
            idx=np.linspace(0,len(tr)-1,8,dtype=int)
            tr=tr[idx]; yr=yr[idx]
        # Similarity uses only the current recording's prefix up to `now`.
        matches=[]
        for hday, hrec, weekday, hts, hys in history_curves.get(cam,[]):
            if hday >= day:  # Excludes own day and every later day.
                continue
            weekday_bonus = 0.82 if weekday == str(row.weekday) else 1.35
            try:
                delta_days = (pd.Timestamp(day)-pd.Timestamp(hday)).days
            except (TypeError, ValueError):
                delta_days=7
            recency=1.+max(0,delta_days-7)*.005
            for shift in shifts:
                hn=interp_at(hts,hys,now+shift)
                ht=interp_at(hts,hys,target+shift)
                if not np.isfinite(hn) or not np.isfinite(ht):
                    continue
                hist_recent=np.asarray([interp_at(hts,hys,t+shift) for t in tr],float)
                usable=np.isfinite(hist_recent)
                if usable.sum()<3:
                    continue
                yy=yr[usable]; xx=hist_recent[usable]
                level_shift=float(np.median(yy-xx))
                # Robust 1D affine correction; slope bounded to protect stability.
                med_x=float(np.median(xx)); med_y=float(np.median(yy))
                dx=xx-med_x; dy=yy-med_y
                norm=float(np.dot(dx,dx))
                scale=float(np.clip(np.dot(dx,dy)/(norm+3.),.3,2.3))
                intercept=med_y-scale*med_x
                fit=intercept+scale*xx
                relative_loss=float(np.median(np.abs(yy-fit))/(8.+max(last,0.)))
                relative_loss+=float(abs(shift))*.003
                # Analog predictions transfer shape, not the absolute prior-day level.
                change=float(np.clip(ht-hn,-160.,160.))
                additive=max(0., last+change)
                multiplicative=max(0.,last*((ht+4.)/max(4.,hn+4.)))
                affine=max(0.,intercept+scale*ht)
                score=float((relative_loss+.08)*weekday_bonus*recency)
                matches.append((score,additive,multiplicative,affine,float(ht),change,shift,(hday,hrec)))
        if not matches:
            out.append({f:math.nan for f in FEATURES})
            continue
        matches.sort(key=lambda z:z[0])
        # One best shift per historical recording: avoid pseudo-replication.
        top=[]
        selected_keys=set()
        for item in matches:
            if len(top)>=4:
                break
            if item[7] not in selected_keys:
                top.append(item)
                selected_keys.add(item[7])
        weights=np.array([1./(e[0]+.2)**2 for e in top],float)
        weights/=weights.sum()
        add=float(np.dot(weights,[e[1] for e in top]))
        ratio=float(np.dot(weights,[e[2] for e in top]))
        aff=float(np.dot(weights,[e[3] for e in top]))
        hist_future=float(np.dot(weights,[e[4] for e in top]))
        delta=float(np.dot(weights,[e[5] for e in top]))
        slope=float(np.clip(np.median(np.diff(ys[-5:])/np.maximum(1.,np.diff(ts[-5:]))) if len(ys)>=5 else 0.,-4.,4.))
        pred_median=float(np.median([add,ratio,aff]))
        res={
            'tr_add':add,'tr_ratio':ratio,'tr_affine':aff,'tr_median':pred_median,
            'tr_hist_future':hist_future, 'tr_delta':delta,'tr_slope':slope,
            'tr_num_curves':float(len({t[0] for t in history_curves.get(cam,[]) if t[0]<day})),
            'tr_num_matches':float(len(matches)),'tr_best_loss':float(top[0][0]),
            'tr_best_shift':float(top[0][6]),'tr_current_last':last,'tr_last_age':age,
            'tr_delta15':float(ys[-1]-ys[max(0,len(ys)-6)]),
            'tr_reference':base,'tr_relative_error':float((add-base)/(abs(base)+8.)),
        }
        for key in res:
            if not np.isfinite(res[key]):
                res[key]=math.nan
        out.append(res)
    return pd.DataFrame(out,columns=FEATURES)


def candidate_trajectory(feat, base):
    """Shadow proposals are defined without using ground-truth labels."""
    out={}
    base=np.asarray(base,float)
    for name, key in [('add','tr_add'),('ratio','tr_ratio'),('affine','tr_affine'),('median','tr_median')]:
        alt=pd.to_numeric(feat[key],errors='coerce').to_numpy(float)
        out['trajectory_'+name]=bounded(np.where(np.isfinite(alt),alt,base))
    av=feat['tr_add'].to_numpy(float)
    af=feat['tr_affine'].to_numpy(float)
    output=np.where(np.isfinite(av)&np.isfinite(af),.5*np.nan_to_num(av)+.5*np.nan_to_num(af),
                    np.where(np.isfinite(av),av,np.where(np.isfinite(af),af,base)))
    out['trajectory_dual']=bounded(output)
    return out


def align_validation(ref, day, train, test_targets, cameras):
    hist=train.loc[train.day.astype(str).lt(day)].copy()
    held=train.loc[train.day.astype(str).eq(day)].copy()
    withheld, visible, nmatched, nrequested=v1.prepare_validation(held,test_targets)
    if withheld.empty:
        raise ValueError('No targets matched test hole schedule for '+day)
    joined=withheld[KEYS+['people']].merge(
        ref.loc[ref.day.eq(day),KEYS+['people','v7_accepted']],
        on=KEYS,how='left',validate='one_to_one',suffixes=('', '_v7'),sort=False)
    if len(joined)!=len(withheld) or joined['v7_accepted'].isna().any():
        raise ValueError('V7 CV key set differs; cannot compare')
    if not np.allclose(joined.people.to_numpy(float),joined.people_v7.to_numpy(float)):
        raise ValueError('Truth does not match locked V7 checkpoint')
    X, anchor=v2.engineer(withheld,visible,hist,cameras)
    X=v7.v7_features(v6.improved_features(X))
    T=trajectory_features(withheld,visible,hist,anchor)
    if len(T)!=len(X):
        raise AssertionError('Trajectory features misaligned')
    X=pd.concat([X.reset_index(drop=True),T.reset_index(drop=True)],axis=1)
    return withheld,X,joined['v7_accepted'].to_numpy(float),int(nmatched),int(nrequested),hist,visible


def training_batches(train, targets, date, cameras, seed, random_augments=1):
    """Create causally valid forecast pseudo-targets on historical days.

    Critical: every pseudo-day uses only earlier recorded days as history,
    not full historical set. Pseudo labels are withheld from current context.
    """
    blocks=[]
    counter={'days':[], 'rows':0}
    hist_days=sorted(set(train.loc[train.day.astype(str).lt(date),'day'].astype(str)))
    for k, pseudo_day in enumerate(hist_days):
        prior=train.loc[train.day.astype(str).lt(pseudo_day)].copy()
        if prior.empty:
            continue
        current=train.loc[train.day.astype(str).eq(pseudo_day)].copy()
        if current.empty:continue
        batches=[]
        tpl,vis,_,_=v1.prepare_validation(current,targets)
        if not tpl.empty:batches.append((tpl,vis))
        for extra in range(random_augments):
            # v1 requires >=100 masked rows; avoid a predictable small-sample failure.
            if len(current)*.38<130:continue
            p,v=v1.generate_train_holes(current,targets,seed=seed+k*17+extra,frac=.38)
            p,v=v6.mask_invalid_random_pseudo(p,v)
            if not p.empty:batches.append((p,v))
        day_rows=0
        for p,v in batches:
            p=p.loc[p.kind.astype(str).eq('forecast')].copy()
            if len(p)<20:continue
            X,anchor=v2.engineer(p,v,prior,cameras)
            X=v7.v7_features(v6.improved_features(X))
            T=trajectory_features(p,v,prior,anchor)
            X=pd.concat([X.reset_index(drop=True),T.reset_index(drop=True)],axis=1)
            # Rows with a usable causal trajectory, and ONLY focus cameras.
            good=(p.unit.astype(str).isin(FOCUS).to_numpy()
                  & np.isfinite(T['tr_add'].to_numpy(float)))
            if not good.any():continue
            X=X.loc[good].reset_index(drop=True)
            labels=p.people.to_numpy(float)[good]
            anchor=T['tr_add'].to_numpy(float)[good]
            with np.errstate(divide='ignore',invalid='ignore'):
                log_delta=np.log1p(labels)-np.log1p(anchor)
            if len(X) and np.isfinite(log_delta).all():
                blocks.append((X,log_delta))
                day_rows+=len(X)
        counter['days'].append({'day':pseudo_day,'training_forecast_rows':day_rows})
    if not blocks:
        return None,None,counter
    cols=blocks[0][0].columns
    X=pd.concat([x[cols] for x,_ in blocks],ignore_index=True)
    y=np.concatenate([y for _,y in blocks])
    counter['rows']=len(y)
    return X,y,counter


def model_predict(Xtr,ytr,Xva,path,args):
    from autogluon.tabular import TabularPredictor
    if path.exists():
        raise FileExistsError('Refusing to overwrite AutoGluon model '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    t=Xtr.copy();label='__v10_log_delta__';t[label]=ytr
    model=TabularPredictor(label=label,problem_type='regression',eval_metric='mean_absolute_error',path=str(path)).fit(
        train_data=t,
        presets='medium_quality',
        time_limit=args.time_limit,
        num_cpus=max(1,min(args.cpus,os.cpu_count() or 1)),num_gpus=0,
        num_bag_folds=args.bag_folds,num_bag_sets=1,num_stack_levels=0,
        dynamic_stacking=False,
        hyperparameters={
            'GBM':[{'num_leaves':16,'n_estimators':420,'learning_rate':.04},
                   {'num_leaves':31,'n_estimators':420,'learning_rate':.035}],
            'XGB':[{'max_depth':3,'n_estimators':350,'learning_rate':.045}],
            'RF':[{}],
        },
    )
    if set(Xtr.columns)!=set(Xva.columns):
        raise ValueError('Model feature schema mismatch')
    arr=np.asarray(model.predict(Xva[Xtr.columns]),float)
    if not np.isfinite(arr).all():
        raise ValueError('AutoGluon returned invalid predictions')
    return arr,model.model_best


def reconcile_candidate_columns(fold_rows):
    """Align fold prediction columns without inventing a model forecast.

    AutoGluon can be unavailable in an early fold when its pseudo-target count
    falls below the training threshold. A later fold can then introduce
    candidate_auto_* columns that do not exist in the earlier fold. Pandas
    concat would fill those with NaN, which previously crashed V10.

    Missing *columns* for an unfitted AutoGluon head inherit the V7 prediction.
    Nonfinite *values* in an existing column remain hard errors: silently
    replacing those would mask invalid model output.
    """
    if not fold_rows:
        raise ValueError('No V10 folds to reconcile')
    names = sorted({name for fold in fold_rows for name in fold.columns
                    if name.startswith('candidate_')})
    details=[]
    for fold in fold_rows:
        if 'v7_accepted' not in fold.columns:
            raise ValueError('Cannot reconcile candidates without V7 baseline')
        base=fold['v7_accepted'].to_numpy(dtype=float)
        if not np.isfinite(base).all():
            raise ValueError('V7 baseline has nonfinite values')
        missing=[]
        for name in names:
            if name not in fold.columns:
                if not name.startswith('candidate_auto_'):
                    raise ValueError(f'Unexpected missing candidate {name}; refusing blind fill')
                fold[name]=base.copy()
                missing.append(name)
            arr=pd.to_numeric(fold[name],errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                n_bad=int((~np.isfinite(arr)).sum())
                raise ValueError(f'{name} has {n_bad} nonfinite predictions within a fold')
        day=str(fold['day'].iloc[0]) if not fold.empty else 'empty'
        details.append({'day':day,'rows':int(len(fold)),
                        'auto_candidates_fallback_to_v7':missing})
    return details


def select_on_early(full,days,min_rows=60,min_gain=6.,min_win=.60,block_minutes=20):
    """Freeze camera-specific choices from Sep 11 alone, then inspect Sep 12."""
    y=full.people.to_numpy(float)
    base=full.v7_accepted.to_numpy(float)
    cams=full.unit.astype(str).to_numpy()
    kinds=full.kind.astype(str).to_numpy()
    d=full.day.astype(str).to_numpy()
    minute=full.minute_of_day.to_numpy(float)
    early=d==days[0]
    late=d==days[1]
    proposal=base.copy()
    candidates=[x for x in full.columns if x.startswith('candidate_')]
    decisions=[]
    for cam in FOCUS:
        group=(cams==cam)&(kinds=='forecast')
        fit=group&early
        item={'camera':cam,'n_early':int(fit.sum()),'n_late':int((group&late).sum()),
              'selected':'v7_accepted','alpha':0.,'early_gain_pct':0.}
        if fit.sum()<min_rows:
            item['reason']='too_few_early_rows';decisions.append(item);continue
        e0=np.abs(y[fit]-base[fit])
        b=np.floor(minute[fit]/block_minutes).astype(int)
        possibilities=[]
        for name in candidates:
            cand=full[name].to_numpy(float)
            if not np.isfinite(cand).all():
                raise ValueError('Candidate has nonfinite values')
            for alpha in ALPHAS:
                p=bounded((1-alpha)*base[fit]+alpha*cand[fit])
                err=np.abs(y[fit]-p)
                g=gain(float(e0.mean()),float(err.mean()))
                blocks=[float(np.mean(e0[b==bb]-err[b==bb])) for bb in np.unique(b) if np.sum(b==bb)>=3]
                wins=sum(diff>0 for diff in blocks)
                possibilities.append((g,wins,len(blocks),name,alpha))
        possibilities.sort(key=lambda z:(z[0],z[1],-z[4]),reverse=True)
        if not possibilities:
            decisions.append(item);continue
        p,w,n,name,alpha=possibilities[0]
        item.update({'best_early_candidate':name,'best_gain_pct':p,
                     'winning_blocks':w,'blocks':n})
        if p>=min_gain and n>=4 and w>=4 and w/n>=min_win:
            cand=full[name].to_numpy(float)
            proposal[group]=bounded((1-alpha)*base[group]+alpha*cand[group])
            item.update({'selected':name,'alpha':alpha,'early_gain_pct':p})
        decisions.append(item)
    late_base=metric(y[late],base[late]);late_new=metric(y[late],proposal[late])
    severe=[]
    for cam in FOCUS:
        mask=late&(kinds=='forecast')&(cams==cam)
        if mask.sum()<40:continue
        old=metric(y[mask],base[mask]);new=metric(y[mask],proposal[mask])
        if new>old*1.15:
            severe.append({'camera':cam,'n':int(mask.sum()),'old':old,'new':new})
    gate='PASS_SHADOW_ONLY' if (late_new<=.98*late_base and not severe and any(v['selected']!='v7_accepted' for v in decisions)) else 'KEEP_V7'
    return proposal,decisions,gate,severe


def sentinel_test(withheld,visible,hist,anchor):
    forecast=withheld.loc[withheld.kind.astype(str).eq('forecast')]
    if forecast.empty:return 'NO_FORECAST'
    r=forecast.iloc[[0]].copy()
    now=float(r.now_minute_of_day.iloc[0])
    rec=str(r.recording.iloc[0])
    x=visible.copy()
    future=x.recording.astype(str).eq(rec)&x.minute_of_day.astype(float).gt(now)
    if not future.any():return 'NO_FUTURE_CONTEXT'
    selected=int(r.index[0])
    one_anchor=np.asarray([float(np.asarray(anchor,float)[selected])])
    a=trajectory_features(r,visible,hist,one_anchor)
    x.loc[future,'people']=99999.
    b=trajectory_features(r,x,hist,one_anchor)
    if not np.allclose(a.select_dtypes('number'),b.select_dtypes('number'),equal_nan=True,rtol=0,atol=1e-9):
        raise AssertionError('Future observation changed trajectory features')
    return 'PASS_FUTURE_INVARIANCE'


def run(args):
    ref,days=get_reference()
    tr,te,sub,targets,context=v1.read_data()
    cams=sorted(set(tr.unit.dropna().astype(str)).union(te.unit.dropna().astype(str)))
    if args.mode=='audit':
        all_targets=len(targets)
        new_cams=sorted(set(targets.unit.astype(str))-set(tr.unit.astype(str)))
        n_new=int(targets.unit.astype(str).isin(new_cams).sum())
        errors=ref.assign(abs_error=(ref.people-ref.v7_accepted).abs())
        high=errors.loc[errors.kind.eq('forecast')&errors.unit.isin(FOCUS),'abs_error'].sum()
        total=errors.abs_error.sum()
        print('V10 AUDIT. V7 checkpoint rows:',len(ref),'MAE people:',round(metric(ref.people,ref.v7_accepted),6))
        print('Focus forecast (E/A/F) share of absolute error: %.2f%%'%(100*high/total))
        print('Total official target IDs:',all_targets,'new-camera targets:',n_new,'cameras:',new_cams)
        print('No training or Kaggle submission. 0.400 custom score NOT measured.')
        return
    if args.mode=='validate' and shutil.disk_usage(ROOT).free<5_000_000_000:
        raise RuntimeError('Disk free <5GB; clean old unneeded model checkpoints before training')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    rows=[];folds=[]
    for day in days:
        val,X,baseline,nmatch,nreq,hist,visible=align_validation(ref,day,tr,targets,cams)
        anchor=pd.to_numeric(X['causal_reference'],errors='coerce').to_numpy(float)
        check=sentinel_test(val,visible,hist,anchor)
        z=val[KEYS+['people']].reset_index(drop=True).copy()
        z['v7_accepted']=baseline
        for name,pred in candidate_trajectory(X,baseline).items():
            z['candidate_'+name]=pred
        trained=False;model='SKIPPED'
        if args.mode=='validate':
            Xtr,ytr,stats=training_batches(tr,targets,day,cams,args.seed,args.augments)
            print('V10 train forecast pseudo rows:',stats)
            if Xtr is not None and len(Xtr)>=args.min_auto_rows:
                focus=val.kind.astype(str).eq('forecast').to_numpy()&val.unit.astype(str).isin(FOCUS).to_numpy()
                usable=focus&np.isfinite(X['tr_add'].to_numpy(float))
                if usable.any():
                    predlog,model=model_predict(Xtr,ytr,X.loc[usable].reset_index(drop=True),MODEL_ROOT/stamp/day/'log_delta',args)
                    projected=bounded(np.expm1(np.log1p(X.loc[usable,'tr_add'].to_numpy(float))+np.clip(predlog,-2.,2.)))
                    q=baseline.copy();q[usable]=projected
                    z['candidate_auto_delta']=q
                    z['candidate_auto_blend']=bounded(.5*baseline+.5*q)
                    trained=True
            print('V10 fold',day,': AutoGluon trained=',trained,'best=',model)
        rows.append(z)
        folds.append({'day':day,'n':len(z),'requested':nreq,'matched':nmatch,'coverage_pct':100*nmatch/max(1,nreq),
                      'sentinel':check,'auto_trained':trained,'auto_model':model,
                      'v7_mae':metric(z.people,z.v7_accepted)})
        print(f'{day} | n={len(z)}/{nreq} | V7 MAE={folds[-1]["v7_mae"]:.4f} | future={check}')
    candidate_fallbacks=reconcile_candidate_columns(rows)
    for info in candidate_fallbacks:
        if info['auto_candidates_fallback_to_v7']:
            print('V10 legitimate fallback on',info['day'],':',
                  ', '.join(info['auto_candidates_fallback_to_v7']),
                  '(model was not fitted on this fold)')
    full=pd.concat(rows,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if len(full)!=len(ref) or full.duplicated(KEYS).any():
        raise ValueError('V10 target key mismatch relative to V7')
    proposed,policies,gate,severe=select_on_early(full,days,args.min_group_rows,args.min_early_gain,args.min_block_win,args.block_minutes)
    full['v10_proposed']=proposed
    full['v10_accepted']=proposed if gate=='PASS_SHADOW_ONLY' else full['v7_accepted'].to_numpy(float)
    early=full.day.astype(str).eq(days[0]).to_numpy();late=~early
    y=full.people.to_numpy(float);b=full.v7_accepted.to_numpy(float)
    result={
        'description':'AutoGluon V10: earlier-day trajectory analogs and optional log-delta AutoML',
        'warning':'Exploratory reused Sep 11/12 checks. NOT independent or custom Kaggle score. No hidden test labels.',
        'days':days,'n_rows':len(full),'model_folds':folds,
        'candidate_column_alignment':candidate_fallbacks,
        'focus_cameras':list(FOCUS),'selection_day':days[0],'shadow_day':days[1],
        'selection_early':{'v7_mae':metric(y[early],b[early]),'v10_proposed':metric(y[early],proposed[early])},
        'shadow_later':{'v7_mae':metric(y[late],b[late]),'v10_proposed':metric(y[late],proposed[late]),
                        'gain_pct':gain(metric(y[late],b[late]),metric(y[late],proposed[late]))},
        'reused_combined':{'v7_mae':metric(y,b),'v10_proposed':metric(y,proposed),
                           'gain_pct':gain(metric(y,b),metric(y,proposed))},
        'early_frozen_policy':policies,'severe_late_regressions':severe,
        'verdict':gate,'official_kaggle_score_verified':False,'submission_created':False,
        'limitations':['Sep 12 reused during prior V1-V9 tuning',
                       'CV lacks new camera targets',
                       'No inference/submission produced',
                       'Custom competition scoring formula not established'],
        'parameters':vars(args),
    }
    REPORTS.mkdir(exist_ok=True)
    summary_name = 'ag_v10_summary.json' if args.mode=='validate' else 'ag_v10_dry_run_summary.json'
    (REPORTS/summary_name).write_text(json.dumps(result,indent=2,allow_nan=False,default=str),encoding='utf-8')
    predictions_name = 'ag_v10_cv_predictions.csv' if args.mode=='validate' else 'ag_v10_dry_run_cv_predictions.csv'
    full.to_csv(REPORTS/predictions_name,index=False)
    camera_rows=[]
    for (d,k,cam), group in full.groupby(['day','kind','unit'],sort=True,observed=True):
        camera_rows.append({'day':str(d),'kind':str(k),'camera':str(cam),'n':int(len(group)),
            'v7_mae':metric(group.people,group.v7_accepted),
            'v10_proposed_mae':metric(group.people,group.v10_proposed),
            'v10_accepted_mae':metric(group.people,group.v10_accepted)})
    camera_name = 'ag_v10_by_camera.csv' if args.mode=='validate' else 'ag_v10_dry_run_by_camera.csv'
    pd.DataFrame(camera_rows).to_csv(REPORTS/camera_name,index=False)
    print('\nV10:',gate,'| V7 all=%.6f | V10 proposed=%.6f | shadow gain=%+.2f%%'%(
        result['reused_combined']['v7_mae'],result['reused_combined']['v10_proposed'],result['shadow_later']['gain_pct']))
    print('Saved',summary_name,'; no Kaggle submission created.')


def main():
    p=argparse.ArgumentParser(description='Madinah AutoGluon V10 trajectory causal shadow test')
    p.add_argument('--mode',choices=['audit','dry-run','validate'],default='audit')
    p.add_argument('--time-limit',type=int,default=120)
    p.add_argument('--cpus',type=int,default=6)
    p.add_argument('--bag-folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--augments',type=int,default=1)
    p.add_argument('--min-auto-rows',type=int,default=120)
    p.add_argument('--min-group-rows',type=int,default=60)
    p.add_argument('--min-early-gain',type=float,default=6.)
    p.add_argument('--min-block-win',type=float,default=.60)
    p.add_argument('--block-minutes',type=int,default=20)
    args=p.parse_args()
    if args.time_limit<10 or args.cpus<1 or args.bag_folds not in (0,2,3,4,5) or args.augments not in (0,1,2) or not (0<=args.min_block_win<=1):
        p.error('Invalid training parameters')
    if args.mode!='validate':
        run(args)
        return
    # A validation failure must never leave a stale dry-run JSON looking like
    # the result of the attempted training run.
    REPORTS.mkdir(parents=True,exist_ok=True)
    summary=REPORTS/'ag_v10_summary.json'
    if summary.is_file():
        shutil.copy2(summary, REPORTS/'ag_v10_summary_previous.json')
        summary.unlink()
    status=REPORTS/'ag_v10_run_status.json'
    started=datetime.now().isoformat(timespec='seconds')
    status.write_text(json.dumps({'state':'RUNNING','mode':'validate',
                                  'started_at':started},indent=2),encoding='utf-8')
    try:
        run(args)
    except Exception as exc:
        status.write_text(json.dumps({'state':'FAILED','mode':'validate',
                                      'started_at':started,
                                      'error_type':type(exc).__name__,
                                      'error':str(exc)},indent=2),encoding='utf-8')
        raise
    status.write_text(json.dumps({'state':'SUCCESS','mode':'validate',
                                  'started_at':started,
                                  'result_file':'ag_v10_summary.json'},indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
