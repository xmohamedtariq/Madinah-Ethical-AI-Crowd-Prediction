"""Madinah Ethical AI AutoGluon V7 - strict shadow experiment.

What is different from V6:
- Additional point-in-time, scale/shift/trend features from V2's causal inputs.
- A direct causal analog forecaster, not another ensemble of the same V6 models.
- Two supervised AutoGluon forecast heads: absolute count and log-ratio to
  a historical/causal reference. All targets are prior released-day pseudo-holes.
- Conservative group rules selected on Sep 11, evaluated on Sep 12.

IMPORTANT:
- V2/V4/V6 have previously used Sep 11 and Sep 12 for model/policy tuning.
  Sep 12 is therefore a reused SHADOW set, not an unbiased independent test.
- The metric printed is MAE in people counts, not Kaggle's custom score.
- This script NEVER creates or uploads a Kaggle submission.
- It NEVER reads private target values from the official test file.

Requires in the same directory: madinah_autogluon_v1.py, v2.py,
madinah_autogluon_v5_meta.py, v6_augmented.py, and V6 report files.

Example PowerShell:
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v7_step_change.py --mode audit
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v7_step_change.py --mode dry-run
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v7_step_change.py --mode validate --profile balanced --time-limit 120 --bag-folds 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import madinah_autogluon_v1 as v1
import madinah_autogluon_v2 as v2
import madinah_autogluon_v6_augmented as v6

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models' / 'ag_v7'
BASE_FILE = REPORTS / 'ag_v6_cv_predictions.csv'
SUMMARY_FILE = REPORTS / 'ag_v6_summary.json'
KEYS = ['recording', 'day', 'unit', 'minute_of_day', 'kind']
FOCUS = [('forecast', 'cam_E'), ('forecast', 'cam_A'),
         ('forecast', 'cam_F'), ('missing', 'cam_A'), ('missing', 'cam_E')]
ALPHAS = (0.25, 0.50, 0.75, 1.0)


def safe_num(df, key):
    if key not in df:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[key], errors='coerce').to_numpy(float)


def clean(a, fallback):
    arr = np.asarray(a, dtype=float)
    ref = np.asarray(fallback, dtype=float)
    return np.where(np.isfinite(arr), arr, ref)


def bounded(a, lo=0.0, hi=100000.0):
    return np.clip(np.asarray(a, dtype=float), lo, hi)


def mae(y, pred):
    return float(np.abs(np.asarray(y, float) - np.asarray(pred, float)).mean())


def gain(base, cand):
    return 100.0 * (base - cand) / base if base > 0 else 0.0


def v7_features(X):
    """Pure transformations of target-safe V2/V6 features; never read labels."""
    X = X.copy()
    forbidden = ('people', '__target_v7__', '__log_v7__', '__residual_v7__')
    if any(c in X.columns for c in forbidden):
        raise ValueError('Forbidden target column in model features')
    cur = clean(safe_num(X, 'own_mean5'), safe_num(X, 'own_last'))
    prior_t = safe_num(X, 'hist_target_mean')
    prior_now = safe_num(X, 'hist_asof_mean')
    slope = clean(safe_num(X, 'own_slope15'), safe_num(X, 'own_slope2'))
    h = np.maximum(0., clean(safe_num(X, 'horizon'), np.zeros(len(X))))
    age = np.maximum(0., clean(safe_num(X, 'own_age'), np.full(len(X), 999.)))
    rule = safe_num(X, 'causal_reference')
    cross = safe_num(X, 'cross_last_median')
    interp = safe_num(X, 'own_interp')
    own30 = safe_num(X, 'own_mean30')
    time = safe_num(X, 'minute_of_day')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        scale = bounded((cur+3.0)/(prior_now+3.0), 0.20, 3.0)
        drift = bounded(slope, -2.0, 2.0) * np.minimum(h, 20.)
        reliability = np.exp(-age/25.) * np.exp(-h/45.)
        ratio_delta = (cur-prior_now) / (np.abs(prior_now)+8.)
        own_to_cross = (cur+3.)/(cross+3.)
        frac_hist = (prior_t+3.)/(prior_now+3.)
    added = {
        'v7_profile_ratio': scale,
        'v7_profile_delta': cur-prior_now,
        'v7_historical_shape': prior_t-prior_now,
        'v7_drift_horizon': drift,
        'v7_asof_reliability': reliability,
        'v7_relative_level': ratio_delta,
        'v7_own_to_cross': own_to_cross,
        'v7_hist_target_ratio': frac_hist,
        'v7_short_vs_long': cur-own30,
        'v7_rule_vs_hist': rule-prior_t,
        'v7_interp_vs_rule': interp-rule,
        'v7_observation_age': np.minimum(age, 240.),
        'v7_horizon_sqrt': np.sqrt(np.maximum(h, 0.)),
        'v7_time_5min': np.floor(time/5.),
        'v7_cross_vs_hist': cross-prior_now,
        'v7_scaled_shape': (prior_t-prior_now)*scale,
        'v7_dynamic_trend': drift*reliability,
        'v7_now_level': cur,
    }
    for name, values in added.items():
        a = np.asarray(values, dtype=float)
        a[~np.isfinite(a)] = np.nan
        X[name] = a
    if X.columns.duplicated().any():
        raise ValueError('Duplicate feature names')
    return X


def causal_rules(X, anchor):
    """Point-in-time analog candidates. No target truth, no future reads."""
    n = len(X)
    anchor = bounded(anchor)
    kind = X['kind'].astype(str).to_numpy()
    future = (kind == 'forecast')
    missing = (kind == 'missing')
    own = clean(safe_num(X, 'own_mean5'), safe_num(X, 'own_last'))
    last = clean(safe_num(X, 'own_last'), own)
    prior_t = safe_num(X, 'hist_target_mean')
    prior_now = safe_num(X, 'hist_asof_mean')
    horizon = bounded(clean(safe_num(X, 'horizon'), np.zeros(n)), 0., 180.)
    age = bounded(clean(safe_num(X, 'own_age'), np.full(n, 999.)), 0., 999.)
    slope = clean(safe_num(X, 'own_slope15'), safe_num(X, 'own_slope2'))
    interp = safe_num(X, 'own_interp')
    next_gap = safe_num(X, 'own_gap_after')
    prev_gap = safe_num(X, 'own_gap_before')
    cross = safe_num(X, 'cross_at_target_median')

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        confidence = np.exp(-age/20.) * np.exp(-horizon/35.)
        shape_delta = prior_t-prior_now
        ratios = np.clip((own+4.)/(prior_now+4.), 0.25, 3.0)
        ratio_strength = np.minimum(.85, confidence)
        scaled = prior_t * (1. + ratio_strength*(ratios-1.))
        shifted = prior_t + confidence*(own-prior_now)
        persistence = own + np.clip(slope, -1.6, 1.6)*horizon/(1.+horizon/12.)
        momentum = own + .65*shape_delta + np.clip(slope,-1.0,1.0)*np.minimum(horizon, 12.)*.25
        adaptive = .35*clean(scaled, anchor) + .35*clean(shifted, anchor) + .30*clean(persistence, anchor)
    variants = {
        'analog_shape': np.where(future, clean(adaptive, anchor), anchor),
        'analog_scaled': np.where(future, clean(scaled, anchor), anchor),
        'analog_shifted': np.where(future, clean(shifted, anchor), anchor),
        'analog_persistence': np.where(future, clean(persistence, anchor), anchor),
        'analog_momentum': np.where(future, clean(momentum, anchor), anchor),
    }
    # Legal two-sided interpolation ONLY for 'missing' holes, never forecasts.
    interp_safe = (
        missing & np.isfinite(interp) & np.isfinite(prev_gap) & np.isfinite(next_gap)
        & (prev_gap > 0) & (next_gap > 0) & (prev_gap <= 35) & (next_gap <= 35)
    )
    interpolated = np.where(interp_safe, interp, anchor)
    cross_available = missing & np.isfinite(cross)
    variants['missing_interpolation'] = np.where(missing, interpolated, anchor)
    variants['missing_cross_blend'] = np.where(missing & cross_available, .65*interpolated+.35*cross, anchor)
    for name, p in variants.items():
        variants[name] = bounded(clean(p, anchor))
    return variants


def load_v6():
    if not BASE_FILE.is_file() or not SUMMARY_FILE.is_file():
        raise FileNotFoundError('Missing V6 reports. Run V6 validate first: ' + str(BASE_FILE))
    df = pd.read_csv(BASE_FILE)
    required = set(KEYS+['people','v6_shadow'])
    if required - set(df.columns):
        raise ValueError('Missing V6 CV fields: ' + str(required-set(df.columns)))
    if df.empty or df.duplicated(KEYS).any():
        raise ValueError('Empty or duplicate V6 CV rows')
    days = sorted(df['day'].astype(str).unique())
    if len(days) != 2:
        raise ValueError('Expected exactly two V6 holdout days')
    for key in ['people', 'v6_shadow']:
        df[key] = pd.to_numeric(df[key], errors='raise')
        if not np.isfinite(df[key]).all() or (df[key] < 0).any():
            raise ValueError('Invalid V6 ' + key)
    summary = json.loads(SUMMARY_FILE.read_text(encoding='utf-8'))
    actual = mae(df['people'], df['v6_shadow'])
    expected = summary.get('combined_mae', {}).get('v6')
    if expected is None or not np.isclose(actual, float(expected), atol=2e-5, rtol=0):
        raise ValueError('V6 checkpoint score mismatch: actual=' + str(actual))
    if summary.get('comparison_rows') is not None and len(df) != int(summary['comparison_rows']):
        raise ValueError('V6 checkpoint row count mismatch')
    print(f'V6 checkpoint: n={len(df)} days={days} MAE={actual:.6f}')
    return df, days, actual


def align_holdout(day_frame, history, targets, cameras, v6_df):
    withheld, visible, matched, requested = v1.prepare_validation(day_frame, targets)
    if withheld.empty:
        raise ValueError('No validation matches')
    allowed, anchor = v2.engineer(withheld, visible, history, cameras)
    X = v7_features(v6.improved_features(allowed))
    target = withheld[KEYS+['people']].merge(
        v6_df[KEYS+['people','v6_shadow']], on=KEYS, how='left',
        validate='one_to_one', suffixes=('', '_saved'), sort=False,
    )
    if len(target) != len(withheld) or target['v6_shadow'].isna().any():
        raise ValueError('V7 targets are not identical to V6 validation targets')
    if not np.allclose(target['people'],target['people_saved'],atol=1e-8,rtol=0):
        raise ValueError('Validation truth changed compared to V6')
    if len(X) != len(target):
        raise ValueError('Target-feature row alignment error')
    return target, X, np.asarray(anchor, float), int(matched), int(requested)


def train_features(history, targets, cameras, args, fold_seed):
    X, y, anchor, stats = v6.prepare_training(history,targets,cameras,fold_seed,args.augments,args.random_fraction)
    X = v7_features(X)
    if len(X)!=len(y) or not np.isfinite(y).all() or not np.isfinite(anchor).all():
        raise ValueError('Invalid generated training examples')
    return X, np.asarray(y,float), np.asarray(anchor,float), stats


def model_hyperparams(profile):
    if profile == 'fast':
        return {'GBM':[{'num_leaves':20, 'learning_rate':.045,'n_estimators':360}],
                'XGB':[{'max_depth':4,'n_estimators':350,'learning_rate':.05}],
                'RF':[{}]}
    return {
        'GBM':[{'num_leaves':16,'n_estimators':600,'learning_rate':.04},
               {'num_leaves':31,'n_estimators':650,'learning_rate':.035,'extra_trees':True}],
        'XGB':[{'max_depth':3,'n_estimators':500,'learning_rate':.045,'subsample':.85},
               {'max_depth':5,'n_estimators':400,'learning_rate':.045,'subsample':.85}],
        'CAT':[{'iterations':400,'depth':5,'learning_rate':.045,'verbose':False}],
        'RF':[{}],
    }


def fit_predict(X_train, y_train, X_val, path, args, label):
    from autogluon.tabular import TabularPredictor
    if path.exists():
        raise FileExistsError(f'Cannot overwrite existing AutoGluon models: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if label in X_train.columns or label in X_val.columns:
        raise ValueError('Reserved target name in features')
    t = X_train.copy()
    t[label] = np.asarray(y_train, float)
    if not np.isfinite(t[label]).all():
        raise ValueError('Non-finite label')
    model = TabularPredictor(
        label=label, path=str(path), eval_metric='mean_absolute_error', problem_type='regression'
    ).fit(
        train_data=t,
        presets='medium_quality',
        time_limit=args.time_limit,
        hyperparameters=model_hyperparams(args.profile),
        num_cpus=max(1,min(args.cpus,os.cpu_count() or 1)),
        num_gpus=0,
        num_bag_folds=args.bag_folds,
        num_bag_sets=1,
        num_stack_levels=0,
        dynamic_stacking=False,
    )
    result = np.asarray(model.predict(X_val),float)
    if len(result)!=len(X_val) or not np.isfinite(result).all():
        raise ValueError('Invalid AutoGluon forecast predictions')
    return result, model.model_best


def collect_candidates(vaX, trX, tr_y, tr_anchor, val_anchor, baseline, model_dir, args):
    rules = causal_rules(vaX,val_anchor)
    out = {name: v for name,v in rules.items()}
    out['v6_baseline'] = np.asarray(baseline, float)
    f_train = trX['kind'].astype(str).eq('forecast').to_numpy()
    f_val = vaX['kind'].astype(str).eq('forecast').to_numpy()
    info = {'train_forecast':int(f_train.sum()), 'validation_forecast':int(f_val.sum()), 'models':{}}
    if f_train.sum() < 150 or not f_val.any():
        print('Too few forecast pseudo-targets for V7 AutoGluon heads')
        return out, info
    Xt = trX.loc[f_train].reset_index(drop=True)
    Xv = vaX.loc[f_val].reset_index(drop=True)
    labels = tr_y[f_train]
    anchors = tr_anchor[f_train]
    # Robust multiplicative target models shifts across recordings.
    with np.errstate(divide='ignore',invalid='ignore'):
        log_ratio = np.log1p(labels) - np.log1p(anchors)
    if not np.isfinite(log_ratio).all():
        raise ValueError('Invalid training log-ratio label')
    log_ratio = np.clip(log_ratio, -3., 3.)
    forecast_anchor = np.asarray(val_anchor,float)[f_val]
    log_hat, log_best = fit_predict(Xt, log_ratio, Xv, model_dir/'log_ratio', args, '__log_v7__')
    log_count = np.expm1(np.log1p(np.maximum(forecast_anchor,0.)) + np.clip(log_hat,-2.5,2.5))
    out['ag_log_ratio'] = np.asarray(baseline,float).copy()
    out['ag_log_ratio'][f_val] = bounded(log_count)
    info['models']['log_ratio'] = log_best
    direct, direct_best = fit_predict(Xt, labels, Xv, model_dir/'direct', args, '__target_v7__')
    out['ag_direct'] = np.asarray(baseline,float).copy()
    out['ag_direct'][f_val] = bounded(direct)
    out['ag_dual'] = np.asarray(baseline,float).copy()
    out['ag_dual'][f_val] = .5*(out['ag_direct'][f_val]+out['ag_log_ratio'][f_val])
    info['models']['direct'] = direct_best
    for name,p in out.items():
        if not np.isfinite(p).all() or (p<0).any():
            raise ValueError(f'Bad candidate {name}')
    return out,info


def policy_from_early(full, days, args):
    y = full['people'].to_numpy(float)
    base = full['v6_shadow'].to_numpy(float)
    early = full['day'].astype(str).eq(days[0]).to_numpy()
    late = full['day'].astype(str).eq(days[1]).to_numpy()
    kind = full['kind'].astype(str).to_numpy()
    camera = full['unit'].astype(str).to_numpy()
    mins = full['minute_of_day'].to_numpy(float)
    proposals = base.copy()
    choices=[]
    candidates = sorted([c for c in full.columns if c.startswith('candidate_')])
    for k,c in FOCUS:
        g = (kind==k)&(camera==c)
        training_group = g&early
        n = int(training_group.sum())
        item={'kind':k,'unit':c,'n_early':n,'n_later':int((g&late).sum()),
              'selected':'v6_shadow','alpha':0.0,'gain_pct_early':0.0}
        if n < (args.min_rows if k=='forecast' else args.min_missing_rows):
            item['reason']='insufficient historical validation coverage'
            choices.append(item)
            continue
        b0=np.abs(y[training_group]-base[training_group]); prev=float(b0.mean())
        blocks=(mins[training_group]//args.block_minutes).astype(int)
        options=[]
        for name in candidates:
            alt=full[name].to_numpy(float)
            for alpha in ALPHAS:
                p=bounded((1-alpha)*base[training_group]+alpha*alt[training_group])
                e=np.abs(y[training_group]-p)
                pct=gain(prev,float(e.mean()))
                diffs=[float((b0[blocks==block]-e[blocks==block]).mean())
                       for block in np.unique(blocks) if (blocks==block).sum()>=3]
                win=sum(d>0 for d in diffs)
                options.append((pct,win,len(diffs),name,alpha))
        options.sort(key=lambda z:(z[0],z[1],-z[4]),reverse=True)
        best=options[0]
        pct,win,total,name,alpha=best
        item.update({'best_candidate':name,'best_early_gain_pct':float(pct),
                     'positive_blocks':int(win),'total_blocks':int(total)})
        if pct >= args.min_early_gain and total >= 4 and win >= 4 and win/max(1,total)>=args.min_block_win:
            all_alt=full[name].to_numpy(float)
            proposals[g]=bounded((1-alpha)*base[g]+alpha*all_alt[g])
            item.update({'selected':name,'alpha':alpha,'gain_pct_early':float(pct)})
        choices.append(item)
    if not np.isfinite(proposals).all():
        raise ValueError('Nonfinite proposed predictions')
    late_baseline=mae(y[late],base[late]);late_candidate=mae(y[late],proposals[late])
    severe=[]
    for k in sorted(set(kind)):
        mask=(kind==k)&late
        if mask.sum()<40:continue
        old=mae(y[mask],base[mask]);new=mae(y[mask],proposals[mask])
        if new>old*1.10:
            severe.append({'kind':k,'n':int(mask.sum()),'before':old,'after':new})
    accepted = (any(p['selected']!='v6_shadow' for p in choices)
                and late_candidate <= .98*late_baseline and not severe)
    return proposals,choices,accepted,severe


def point_in_time_sentinel_test(validation, context, hist, cameras):
    """Changing a future observation cannot change forecast target features."""
    future=validation.loc[validation['kind'].astype(str).isin(['forecast','hidden-future'])]
    if future.empty:return 'SKIPPED_NO_FUTURE_TARGET'
    test=future.iloc[[0]].copy()
    rec=str(test.iloc[0]['recording'])
    cutoff=float(test.iloc[0]['now_minute_of_day'])
    altered=context.copy()
    mask=altered['recording'].astype(str).eq(rec)&(pd.to_numeric(altered['minute_of_day'],errors='coerce')>cutoff)
    if not mask.any():return 'SKIPPED_NO_LATER_CONTEXT'
    before, ref0=v2.engineer(test,context,hist,cameras)
    altered.loc[mask,'people']=99999.
    after, ref1=v2.engineer(test,altered,hist,cameras)
    if before.shape != after.shape or not np.allclose(ref0,ref1,rtol=0,atol=1e-9):
        raise AssertionError('Future observations changed historical reference')
    for col in before.columns:
        a,b=before[col],after[col]
        if pd.api.types.is_numeric_dtype(a):
            if not np.allclose(pd.to_numeric(a,errors='coerce'),pd.to_numeric(b,errors='coerce'),
                               rtol=0,atol=1e-9,equal_nan=True):
                raise AssertionError('Future-leakage feature: '+col)
        elif not a.equals(b):
            raise AssertionError('Future-leakage categorical feature: '+col)
    return 'PASS_FUTURE_CONTEXT_INVARIANT'


def run(args):
    baseline,days,baseline_mae=load_v6()
    if args.mode=='audit':
        print('V7 audit passed. Baseline locked, no training, no submission.')
        print(f'Required {args.target_gain:.1f}% reduction target: {baseline_mae*(1-args.target_gain/100):.6f} MAE people')
        return
    tr,te,sample,targets,context=v1.read_data()
    all_cams=sorted(set(tr['unit'].dropna().astype(str)).union(te['unit'].dropna().astype(str)))
    tr['day']=tr['day'].astype(str)
    if args.mode=='validate' and shutil.disk_usage(ROOT).free < 5_000_000_000:
        raise RuntimeError('Fewer than 5GB disk free; free disk space before fitting new models')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    all_results=[]
    fold_stats=[]
    safety=[]
    for day in days:
        hist=tr.loc[tr['day'].lt(day)].copy()
        held=tr.loc[tr['day'].eq(day)].copy()
        if hist.empty or held.empty:
            raise ValueError('Invalid chronological fold: '+day)
        va_targets,visible,n_matched,n_requested = v1.prepare_validation(held,targets)
        check=point_in_time_sentinel_test(va_targets,visible,hist,all_cams)
        safety.append({'day':day,'future_invariance':check})
        Xtrain, ytrain, atrain, train_summary = train_features(hist,targets,all_cams,args,args.seed+int(day[-2:]))
        va, Xval, aval, matched, requested = align_holdout(held,hist,targets,all_cams,baseline)
        print(f'\nV7 fold {day}: train {len(Xtrain)}, validation {matched}/{requested} ({matched/max(1,requested):.1%}), features {len(Xtrain.columns)}, safety {check}')
        if set(Xtrain.columns)!=set(Xval.columns):
            raise ValueError('Training and inference feature names differ')
        Xval = Xval[Xtrain.columns]
        if args.mode == 'dry-run':
            continue
        raw,best=collect_candidates(Xval,Xtrain,ytrain,atrain,aval,
                                    va['v6_shadow'].to_numpy(float),MODELS/stamp/day,args)
        day_frame=va[KEYS+['people','v6_shadow']].copy()
        for name,pred in raw.items():
            if name=='v6_baseline':continue
            day_frame['candidate_'+name]=pred
        all_results.append(day_frame)
        fold_stats.append({'day':day,'n_validation':len(va),'requested':requested,
                           'coverage':matched/max(1,requested), 'training':train_summary,
                           'heads':best, 'future_invariance':check})
    if args.mode=='dry-run':
        print('\nV7 DRY-RUN PASSED: no training and no Kaggle submission.')
        return
    full=pd.concat(all_results,ignore_index=True)
    if len(full)!=len(baseline) or full.duplicated(KEYS).any():
        raise ValueError('V7 results do not match V6 comparison universe')
    full=full.sort_values(KEYS).reset_index(drop=True)
    y=full['people'].to_numpy(float)
    base=full['v6_shadow'].to_numpy(float)
    proposed,policies,passed,regressions=policy_from_early(full,days,args)
    # Shadow candidates are never silently adopted if later-date gate fails.
    full['v7_shadow_proposal']=proposed
    full['v7_accepted']=proposed if passed else base
    late=full['day'].astype(str).eq(days[1]).to_numpy()
    early=~late
    reports=[]
    for (d,k,c),group in full.groupby(['day','kind','unit'],sort=True):
        rows=group.index.to_numpy()
        reports.append({'day':d,'kind':k,'unit':c,'n':len(rows),
                        'v6_mae':mae(y[rows],base[rows]),
                        'v7_proposed_mae':mae(y[rows],proposed[rows]),
                        'v7_accepted_mae':mae(y[rows],full['v7_accepted'].to_numpy(float)[rows])})
    table=pd.DataFrame(reports)
    kind_rows=[]
    for (d,k), group in full.groupby(['day','kind'],sort=True):
        inds=group.index.to_numpy()
        kind_rows.append({'day':d,'kind':k,'n':len(inds),
                          'v6_mae':mae(y[inds],base[inds]),
                          'v7_proposed_mae':mae(y[inds],proposed[inds]),
                          'v7_accepted_mae':mae(y[inds],full['v7_accepted'].to_numpy(float)[inds])})
    kinds=pd.DataFrame(kind_rows)
    summary={
        'baseline':f'AutoGluon V6, same {len(full)} matched validation rows',
        'warning':'People-count MAE, NOT Kaggle custom score; Sep 12 previously reused for V2/V4/V5/V6 tuning, so this is NOT independent final validation.',
        'days':days,'rows':len(full),'safety_checks':safety,'folds':fold_stats,
        'v6_mae':baseline_mae, 'target_gain_percent':args.target_gain,
        'target_mae':baseline_mae*(1-args.target_gain/100),
        'proposed_mae':mae(y,proposed),
        'proposed_gain_pct':gain(baseline_mae,mae(y,proposed)),
        'accepted_mae':mae(y,full['v7_accepted']),
        'accepted_gain_pct':gain(baseline_mae,mae(y,full['v7_accepted'])),
        'later_v6_mae':mae(y[late],base[late]),
        'later_proposed_mae':mae(y[late],proposed[late]),
        'later_gain_pct':gain(mae(y[late],base[late]),mae(y[late],proposed[late])),
        'early_v6_mae':mae(y[early],base[early]),
        'early_proposed_mae':mae(y[early],proposed[early]),
        'policy':policies,'severe_later_regressions':regressions,
        'gate':'PASS_SHADOW_DIAGNOSTIC_ONLY' if passed else 'KEEP_V6',
        'target_55_percent_reduction_observed':bool(passed and mae(y,proposed)<=baseline_mae*(1-args.target_gain/100)),
        'official_competition_score_verified':False,
        'submission_created':False,
    }
    REPORTS.mkdir(parents=True,exist_ok=True)
    full.to_csv(REPORTS/'ag_v7_cv_predictions.csv',index=False)
    table.to_csv(REPORTS/'ag_v7_by_camera.csv',index=False)
    kinds.to_csv(REPORTS/'ag_v7_per_kind.csv',index=False)
    (REPORTS/'ag_v7_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print('\n=== AUTOGLUON V7 SHADOW RESULT ===')
    print('V6 MAE: %.6f | V7 proposed: %.6f | proposed gain: %+.2f%%' % (
        baseline_mae,summary['proposed_mae'],summary['proposed_gain_pct']))
    print('Later V6: %.6f | later V7: %.6f | later gain: %+.2f%%' % (
        summary['later_v6_mae'],summary['later_proposed_mae'],summary['later_gain_pct']))
    print('55%% target: %.6f MAE | observed: %s' % (summary['target_mae'],summary['target_55_percent_reduction_observed']))
    print('VERDICT:',summary['gate'])
    print('Saved reports/ag_v7_summary.json and CV breakdowns; NO Kaggle submission created.')


def main():
    p=argparse.ArgumentParser(description='Madinah AutoGluon V7 - strict shadow evaluation, no submissions')
    p.add_argument('--mode',choices=['audit','dry-run','validate'],default='audit')
    p.add_argument('--profile',choices=['fast','balanced'],default='fast')
    p.add_argument('--time-limit',type=int,default=120)
    p.add_argument('--bag-folds',type=int,default=3)
    p.add_argument('--cpus',type=int,default=6)
    p.add_argument('--augments',type=int,default=2)
    p.add_argument('--random-fraction',type=float,default=.35)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--target-gain',type=float,default=55.)
    p.add_argument('--min-rows',type=int,default=65)
    p.add_argument('--min-missing-rows',type=int,default=25)
    p.add_argument('--min-early-gain',type=float,default=7.)
    p.add_argument('--min-block-win',type=float,default=.60)
    p.add_argument('--block-minutes',type=int,default=20)
    a=p.parse_args()
    if not (0<a.target_gain<100) or not (0<=a.augments<=5) or not (.1<=a.random_fraction<=.6) or a.time_limit<10 or a.cpus<1 or a.bag_folds not in (0,2,3,4,5):
        p.error('Invalid target gain, training budget, augmentation or bagging setting')
    run(a)


if __name__=='__main__':
    main()
