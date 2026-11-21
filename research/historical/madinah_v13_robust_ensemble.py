"""Madinah V13: point-in-time two-loss-weighted forecast ensemble.

This research script requires madinah_autogluon_v12_research.py and the official data
in the same project. It never submits to Kaggle and never estimates a private score.

WARNING: Sep 11/Sep 12 have already been repeatedly used in prior model selection.
All MAE measurements here are retrospective EXPLORATORY diagnostics, not unbiased CV.

Purpose: keep original V7 predictions intact, train two LightGBM models on historical
pre-`now` snapshots only, blend their forecasts, check per-camera regressions, and
export forecast-only ID predictions for a possible future manually reviewed overlay.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import madinah_autogluon_v12_research as prev
import madinah_autogluon_v1 as v1

KEYS = ['recording', 'day', 'unit', 'minute_of_day']
FOCUS = {'cam_E', 'cam_A', 'cam_F'}
NEW = {'cam_I', 'cam_J', 'cam_K', 'cam_L', 'cam_M'}
BLEND_MODELS = 0.25   # 25% importance-corrected model, 75% original L1 model
PRIMARY_ALPHA = 0.50  # Blend into V7 for diagnostics or user-supplied base predictions
ALTERNATE_ALPHA = 0.60 # Exploratory more aggressive, NOT automatically accepted
BASELINE_MAE = 15.007488956008123


def model_prediction(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame,
                     weight_power: float) -> np.ndarray:
    """Labels influence fitting only. Features are built as-of-now upstream."""
    if set(Xtr.columns) != set(Xva.columns):
        raise AssertionError('Training/inference feature mismatch')
    Xva = Xva[Xtr.columns]
    a = pd.to_numeric(Xtr.anchor, errors='raise').to_numpy(float)
    av = pd.to_numeric(Xva.anchor, errors='raise').to_numpy(float)
    ytr = np.asarray(ytr, float)
    if (not np.isfinite(ytr).all() or not np.isfinite(a).all()
            or not np.isfinite(av).all() or np.any(a < 0) or np.any(av < 0)):
        raise ValueError('Invalid anchors or targets')
    yt = (ytr - a) / np.maximum(10., a)
    w = (np.maximum(10., a) / 80.) ** weight_power
    w = w / np.mean(w)
    model = LGBMRegressor(
        objective='regression_l1', n_estimators=290, learning_rate=.038,
        num_leaves=17, max_depth=-1, min_child_samples=45,
        colsample_bytree=.85, subsample=.9, subsample_freq=1,
        reg_lambda=3., verbose=-1, n_jobs=6, random_state=47,
    )
    # The raw camera name is not a model feature: important for unseen cameras.
    xtrain = Xtr.drop(columns=['cam']).replace([np.inf, -np.inf], np.nan)
    xval = Xva.drop(columns=['cam']).replace([np.inf, -np.inf], np.nan)
    model.fit(xtrain, yt, sample_weight=w)
    p = model.predict(xval)
    result = av + np.clip(p, -1.5, 2.) * np.maximum(10., av)
    result = np.clip(result, 0., 100000.)
    if not np.isfinite(result).all():
        raise AssertionError('Non-finite model output')
    return result


def two_head_predict(Xtr, ytr, Xva):
    direct = model_prediction(Xtr, ytr, Xva, 0.0)
    weighted = model_prediction(Xtr, ytr, Xva, .35)
    ens = (1. - BLEND_MODELS) * direct + BLEND_MODELS * weighted
    return direct, weighted, ens


def blend_existing(base, overlay, alpha=PRIMARY_ALPHA):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha outside [0,1]')
    b = np.asarray(base, dtype=float)
    o = np.asarray(overlay, dtype=float)
    if b.shape != o.shape or not np.isfinite(b).all() or not np.isfinite(o).all():
        raise ValueError('Blending requires aligned finite arrays')
    return np.clip((1.-alpha)*b+alpha*o, 0., 100000.)


def build_cv_fold(day):
    train = prev.tr[prev.tr.day < day].copy()
    held = prev.tr[prev.tr.day == day].copy()
    va, visible, matched, requested = v1.prepare_validation(held, prev.targets)
    va = va.loc[va.kind == 'forecast'].copy()
    if len(va) == 0:
        raise AssertionError(f'No simulated forecast targets for {day}')
    synth = prev.training_synthetic(train, prev.targets, every=10)
    Xtr, ytr, _ = prev.make_rows(synth, prev.Store(train), prev.Profile(train), True)
    Xva, _, Mva = prev.make_rows(va, prev.Store(visible), prev.Profile(train), False)
    if len(Mva) != len(va):
        raise AssertionError('Forecast target coverage dropped')
    # previous V12 validation used 137 features including raw cam (not passed to model).
    if set(Xtr.columns) != set(Xva.columns):
        raise AssertionError('Feature schema changed')
    return Xtr, ytr, Xva[Xtr.columns], Mva, int(requested)


def camera_report(full, prediction_col):
    foc = full.loc[full.kind == 'forecast'].copy()
    result=[]
    for (day, cam), grp in foc.groupby(['day','unit']):
        be=float(abs(grp.people - grp.v7_accepted).mean())
        af=float(abs(grp.people - grp[prediction_col]).mean())
        result.append({'day':str(day),'camera':str(cam),'n':len(grp),
                       'v7_mae':be,'candidate_mae':af,
                       'gain_pct':100*(be-af)/be if be else 0.})
    return pd.DataFrame(result)


def validate():
    prev.checkpoint_audit()
    prev.future_invariance_sentinel()
    rows=[]
    folds=[]
    for day in ('2026-09-11','2026-09-12'):
        start = time.time()
        Xtr, ytr, Xva, meta, requested = build_cv_fold(day)
        p0,pw,ens = two_head_predict(Xtr,ytr,Xva)
        meta=meta.copy()
        meta['v13_direct']=p0
        meta['v13_weighted']=pw
        meta['v13_ensemble']=ens
        rows.append(meta)
        folds.append({'day':day,'n_train_synthetic':len(Xtr),
                      'n_val_forecast':len(meta),'n_requested_targets':requested,
                      'training_time_s':round(time.time()-start,2)})
        print(day, 'synthetic_rows',len(Xtr),'forecast_val',len(meta),
              'training_seconds',round(time.time()-start,1),flush=True)
    foc=pd.concat(rows,ignore_index=True)
    if len(foc)!=2036 or foc.duplicated(KEYS).any():
        raise AssertionError('Historical forecast checkpoint changed')
    full=prev.old.merge(foc[KEYS+['v13_direct','v13_weighted','v13_ensemble']],
                        on=KEYS,how='left',validate='one_to_one')
    if len(full)!=3719 or int(full.v13_ensemble.notna().sum())!=2036:
        raise AssertionError('V7 overlap not exact')
    base=float(abs(full.people-full.v7_accepted).mean())
    if abs(base-BASELINE_MAE)>1e-5:
        raise AssertionError('Unexpected V7 baseline')
    alpha_report={}
    for alpha in (PRIMARY_ALPHA, ALTERNATE_ALPHA):
        field=f'v13_blend_{int(alpha*100)}'
        full[field]=full.v7_accepted.to_numpy(float)
        mask=full.v13_ensemble.notna()
        full.loc[mask,field]=blend_existing(full.loc[mask,'v7_accepted'],
                                           full.loc[mask,'v13_ensemble'],alpha)
        cam=camera_report(full,field)
        day_stats=[]
        for day,g in full.groupby('day'):
            b=float(abs(g.people-g.v7_accepted).mean())
            m=float(abs(g.people-g[field]).mean())
            day_stats.append({'day':str(day),'n':len(g),'v7_mae':b,'v13_mae':m,
                              'gain_pct':100*(b-m)/b})
        score=float(abs(full.people-full[field]).mean())
        worst=float(cam.gain_pct.min())
        n_severe=int((cam.gain_pct < -5).sum())
        alpha_report[str(alpha)]={
            'mae_people':score,'gain_pct_vs_v7':100*(base-score)/base,
            'by_day':day_stats,'worst_camera_day_gain_pct':worst,
            'camera_day_regressions_beyond_5_pct':n_severe,
            'provisional_gate':('PASS_EXPLORATORY_ONLY'
                                if all(s['gain_pct']>0 for s in day_stats)
                                and n_severe==0 else 'REJECT_FOR_DEPLOYMENT'),
        }
        (HERE/'reports').mkdir(exist_ok=True)
        cam.to_csv(HERE/'reports'/f'ag_v13_by_camera_alpha{int(alpha*100)}.csv',index=False)
    summary={'name':'V13 count-aware dual-head LightGBM research',
             'created_from':'real official training and V7 CV predictions',
             'validation_integrity':'REUSED_NON_INDEPENDENT',
             'warning':'These dates have been repeatedly tuned in V1-V12; neither gain nor provisional gate is an unbiased estimate. No official Kaggle metric.',
             'v7_mae_people':base, 'v12_lightgbm_primary_mae':14.254747343926903,
             'model_mixture_importance_power':.35,
             'model_mixture_importance_weight':BLEND_MODELS,
             'folds':folds,'alphas':alpha_report,
             'known_unseen_test_camera_targets':int(prev.targets.unit.isin(NEW).sum()),
             'all_4457_test_predictions_completed':False,
             'kaggle_submitted':False,'official_score_verified':False}
    rep=HERE/'reports'
    full.to_csv(rep/'ag_v13_cv_predictions.csv',index=False)
    (rep/'ag_v13_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('=== V13 EXPERIMENT: NOT INDEPENDENT ===')
    for key,val in alpha_report.items():
        print('alpha',key,'MAE',round(val['mae_people'],5),'gain%',round(val['gain_pct_vs_v7'],2),
              'worst camera/day gain%',round(val['worst_camera_day_gain_pct'],2),
              'decision',val['provisional_gate'])
    print('NO Kaggle submission; no official 0.400 score measured.')
    return summary


def forecast_test(base_submission=None, alpha=PRIMARY_ALPHA):
    prev.checkpoint_audit()
    prev.future_invariance_sentinel()
    hist=prev.tr.copy()
    synth=prev.training_synthetic(hist, prev.targets, every=10)
    Xtr,ytr,_=prev.make_rows(synth,prev.Store(hist),prev.Profile(hist),True)
    only=prev.targets.loc[prev.targets.kind=='forecast'].copy()
    visible=prev.test.loc[prev.test.id.isna() & prev.test.people.notna()].copy()
    Xva,_,meta=prev.make_rows(only,prev.Store(visible),prev.Profile(hist),False)
    if len(meta)!=2660:raise AssertionError(f'Expected 2660 targets, got {len(meta)}')
    _,_,p=two_head_predict(Xtr,ytr,Xva[Xtr.columns])
    meta=meta.merge(only[KEYS+['id']],on=KEYS,validate='one_to_one')
    meta['prediction_v13_standalone']=p
    if len(meta)!=2660 or meta.id.duplicated().any():
        raise AssertionError('Official forecast target count or ID conflict')
    rep=HERE/'reports';rep.mkdir(exist_ok=True)
    path=rep/'ag_v13_forecast_only_predictions.csv'
    meta[['id','unit','prediction_v13_standalone']].to_csv(path,index=False)
    print('Created forecast-only candidate (NOT Kaggle submission):',path)
    print('Note: new cameras I-M were not part of historical validation; local zero-shot score is unknown.')
    if base_submission:
        base=pd.read_csv(base_submission)
        if set(base.columns)!={'id','people'} or len(base)!=4457 or base.id.duplicated().any() \
            or set(base.id.astype(str))!=set(prev.sample.id.astype(str)):
            raise ValueError('Full base submission ID/column verification failed')
        base.people=pd.to_numeric(base.people,errors='raise')
        if not np.isfinite(base.people).all():
            raise ValueError('Nonfinite base submission')
        overlay=base.merge(meta[['id','unit','prediction_v13_standalone']],on='id',how='left',validate='one_to_one')
        mask=overlay.prediction_v13_standalone.notna() & (~overlay.unit.isin(NEW))
        # IMPORTANT: before manually using this file the full base must be legal
        # under forecast as-of-now rules; we cannot verify base provenance here.
        overlay.loc[mask,'people']=blend_existing(overlay.loc[mask,'people'],
            overlay.loc[mask,'prediction_v13_standalone'],alpha)
        dest=HERE/'submissions'/f'submission_v13_overlay_alpha{int(alpha*100)}_REVIEW_ONLY.csv'
        dest.parent.mkdir(exist_ok=True)
        overlay[['id','people']].to_csv(dest,index=False)
        print('OPTIONAL FULL LOCAL OVERLAY - NOT SUBMITTED:',dest,'patched old-camera forecast rows',int(mask.sum()))
        print('BASE PROVENANCE NOT VERIFIED; NOT automatically safe to submit.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('audit','validate','forecast-test'),default='audit')
    ap.add_argument('--base-submission',default=None)
    ap.add_argument('--alpha',type=float,default=PRIMARY_ALPHA)
    args=ap.parse_args()
    if args.mode=='audit':prev.checkpoint_audit();prev.future_invariance_sentinel()
    elif args.mode=='validate':validate()
    else:forecast_test(args.base_submission,args.alpha)

if __name__=='__main__':main()
