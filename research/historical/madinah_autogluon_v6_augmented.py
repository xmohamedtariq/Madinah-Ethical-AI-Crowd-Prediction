"""Madinah Ethical AI / AutoGluon V6: target-shaped + augmented supervised experiment.

This is an EXPERIMENT, not a Kaggle scorer or a submission generator.
- Uses officially released historical observations and target schedule.
- Maintains whole-day chronological HOLDOUT: Sep 11, then Sep 12.
- Forecast and hidden-future features are evaluated at or before each target's `now`.
- Trains with test-template-shaped pseudo-targets PLUS masked, random pseudo-targets.
- Adds scale/shift features derived ONLY from V2's causal features.
- Reproduces the frozen V4 baseline, freezes any local group selection on the
  early holdout, and shows its performance on the later holdout.
- Sep 12 is NOT an independent fresh test: earlier V2/V4/V5 experiments saw it.
- Does NOT create or upload a Kaggle submission or promise score <= 0.400.

Depends on madinah_autogluon_v1.py, v2.py and v5_meta.py IN THE SAME FOLDER.
Requires V2 out-of-date CV report and V4 frozen_camera report.

PowerShell:
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v6_augmented.py --mode audit
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v6_augmented.py --mode dry-run
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v6_augmented.py --mode validate --time-limit 180 --specialist-time-limit 90
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
import madinah_autogluon_v5_meta as v5

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models' / 'ag_v6'
KEYS = ['recording', 'day', 'unit', 'minute_of_day', 'kind']
FOCUS = [('forecast', 'cam_E'), ('forecast', 'cam_A'), ('forecast', 'cam_F'),
         ('missing', 'cam_A'), ('missing', 'cam_E')]
ALPHAS = (0.25, 0.50, 0.75)
CANDIDATES = ('v6_pooled', 'v6_forecast')


def mae(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if not len(y):
        return math.nan
    return float(np.mean(np.abs(y - p)))


def gain_pct(before, after):
    return 100 * (before-after)/before if before > 0 and np.isfinite(after) else 0.0


def safe_num(df, name):
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)


def ratio(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = (a + 2.) / (b + 2.)
    z[~np.isfinite(z)] = np.nan
    return np.clip(z, .05, 10.)


def improved_features(X):
    """Pure transforms of V2's already-legal, per-target features."""
    X = X.copy()
    last = safe_num(X, 'own_last')
    mean5 = safe_num(X, 'own_mean5')
    mean30 = safe_num(X, 'own_mean30')
    prior_t = safe_num(X, 'hist_target_mean')
    prior_now = safe_num(X, 'hist_asof_mean')
    cross_now = safe_num(X, 'cross_last_median')
    anchor = safe_num(X, 'causal_reference')
    rule = safe_num(X, 'rule_hist_delta')
    slope = safe_num(X, 'own_slope15')
    horizon = safe_num(X, 'horizon')
    age = safe_num(X, 'own_age')
    mins = safe_num(X, 'minute_of_day')

    X['v6_same_day_scale'] = ratio(last, prior_now)
    X['v6_cross_scale'] = ratio(last, cross_now)
    X['v6_profile_shift'] = prior_t - prior_now
    X['v6_current_vs_prior'] = last - prior_now
    X['v6_rule_vs_profile'] = rule - prior_t
    X['v6_anchor_relative'] = ratio(anchor, prior_t)
    X['v6_recent_scale'] = ratio(mean5, mean30)
    X['v6_cross_relative'] = ratio(cross_now, prior_now)
    X['v6_horizon_trend'] = np.clip(slope, -2., 2.) * np.clip(horizon, 0., 30.)
    X['v6_observation_gap'] = np.clip(age, 0., 180.)
    X['v6_horizon_5'] = np.clip(horizon / 5., 0., 6.)
    X['v6_time_bucket'] = np.floor(mins / 10.)
    X['v6_own_vs_cross'] = last - cross_now
    X['v6_rule_vs_anchor'] = rule - anchor
    X['v6_hist_gradient'] = ratio(prior_t, prior_now)
    # Numeric models in AutoGluon accept NaN; infinite values are never allowed.
    for col in X.columns:
        if col.startswith('v6_'):
            a = pd.to_numeric(X[col], errors='raise').to_numpy(float)
            X[col] = np.where(np.isfinite(a), a, np.nan)
    if any(c in X.columns for c in ['people', '__target_v6__', '__residual_v6__']):
        raise RuntimeError('Target label leaked into model features')
    return X


def load_baseline():
    df, v4_pred, days, early, later, summary = v5.load_and_freeze(
        REPORTS / 'ag_v2_cv_predictions.csv', REPORTS / 'ag_v4_frozen_camera_summary.json'
    )
    if len(days) != 2 or np.isnan(v4_pred).any():
        raise ValueError('V4 baseline is not properly reproducible')
    if df.duplicated(KEYS).any():
        raise ValueError('Duplicate baseline targets')
    return df, v4_pred, days, early, later, summary


def mask_invalid_random_pseudo(pseudo, visible):
    """Discard forecast targets whose simulated now precedes recording start."""
    future = pseudo['kind'].astype(str).isin(['forecast', 'hidden-future'])
    times = pd.to_numeric(pseudo['now_minute_of_day'], errors='coerce')
    starts = pseudo['recording'].astype(str).map(v1.rec_start).astype(float)
    keep = ~future | (times.ge(starts) & times.notna() & starts.notna())
    return pseudo.loc[keep].copy(), visible


def prepare_training(history, targets, cameras, seed, augments, frac):
    from collections import Counter
    batches = []
    counters = Counter()
    # V2's already tested test-template geometry in historical recordings.
    tpl, tpl_visible, note = v2.synthetic_training(history, targets, seed, mode='template')
    if len(tpl):
        f, anchor = v2.engineer(tpl, tpl_visible, history, cameras)
        f = improved_features(f)
        batches.append((f, tpl['people'].to_numpy(float), anchor))
        counters['template_rows'] += len(tpl)
    for offset in range(augments):
        p, visible = v1.generate_train_holes(history, targets, seed=seed+101+offset, frac=frac)
        p, visible = mask_invalid_random_pseudo(p, visible)
        if len(p) < 80:
            continue
        f, anchor = v2.engineer(p, visible, history, cameras)
        f = improved_features(f)
        batches.append((f, p['people'].to_numpy(float), anchor))
        counters['random_rows'] += len(p)
        counters['random_scenarios'] += 1
    if not batches:
        raise ValueError('No training pseudo-targets were generated')
    X = pd.concat([x for x, _, _ in batches], ignore_index=True)
    y = np.concatenate([y for _, y, _ in batches])
    anchor = np.concatenate([p for _, _, p in batches])
    assert len(X) == len(y) == len(anchor)
    if not np.isfinite(y).all() or not np.isfinite(anchor).all() or (y < 0).any():
        raise ValueError('Nonfinite training target or anchor')
    counters['training_rows'] = len(y)
    counters['feature_count'] = len(X.columns)
    return X, y, anchor, dict(counters)


def prepare_holdout(history, day_frame, test_targets, cameras, baseline):
    withheld, visible, matched, requested = v1.prepare_validation(day_frame, test_targets)
    if withheld.empty:
        raise ValueError('No validation targets matched the template')
    if withheld.duplicated(KEYS).any():
        raise ValueError('Duplicate validation target')
    baseline_day = baseline.loc[baseline['day'].astype(str).eq(str(day_frame['day'].iloc[0])), KEYS+['v4_anchor']]
    joined = withheld.merge(baseline_day, how='left', on=KEYS, validate='one_to_one', sort=False)
    if joined['v4_anchor'].isna().any() or len(joined) != len(withheld):
        raise ValueError('V4 baseline does not align with V6 validation targets')
    feats, rules = v2.engineer(withheld, visible, history, cameras)
    feats = improved_features(feats)
    return withheld, feats, rules, joined['v4_anchor'].to_numpy(float), matched, requested


def hyperparameters(profile):
    if profile == 'fast':
        return {'GBM': [{'num_leaves': 31, 'n_estimators': 450, 'learning_rate': .04}],
                'XGB': [{'max_depth': 4, 'n_estimators': 420, 'learning_rate': .05}],
                'RF': [{}]}
    return {
        'GBM': [{'num_leaves': 31, 'n_estimators': 650, 'learning_rate': .035},
                {'num_leaves': 20, 'n_estimators': 750, 'learning_rate': .04, 'extra_trees': True}],
        'XGB': [{'max_depth': 4, 'n_estimators': 600, 'learning_rate': .045,
                 'subsample': .85, 'colsample_bytree': .85}],
        'CAT': [{'iterations': 380, 'depth': 5, 'learning_rate': .055, 'verbose': False}],
        'RF': [{}],
        'XT': [{}],
    }


def fit_predict(trX, labels, vaX, path, seconds, args, profile):
    from autogluon.tabular import TabularPredictor
    if path.exists():
        raise FileExistsError(f'Will not overwrite model: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    train = trX.copy()
    train['__target_v6__'] = np.asarray(labels, float)
    if not np.isfinite(train['__target_v6__']).all():
        raise ValueError('Training targets are not finite')
    predictor = TabularPredictor(
        label='__target_v6__', problem_type='regression',
        eval_metric='mean_absolute_error', path=str(path)
    ).fit(
        train_data=train,
        time_limit=seconds,
        presets='medium_quality',
        hyperparameters=hyperparameters(profile),
        num_bag_folds=args.bag_folds,
        num_bag_sets=1,
        num_stack_levels=0,
        dynamic_stacking=False,
        num_cpus=max(1, min(args.cpus, os.cpu_count() or 1)),
        num_gpus=0,
    )
    p = np.asarray(predictor.predict(vaX), dtype=float)
    if len(p) != len(vaX) or not np.isfinite(p).all():
        raise ValueError('AutoGluon returned invalid predictions')
    return p, predictor.model_best


def test_coverage(day_df, test_targets):
    withheld, _, count, requested = v1.prepare_validation(day_df, test_targets)
    return {'requested': int(requested), 'matched': int(count), 'coverage_pct': 100*count/max(1,requested),
            'held_out_targets': int(len(withheld))}


def run(args):
    if not (0 <= args.augments <= 5) or not (.1 <= args.random_fraction <= .6):
        raise ValueError('Augments must be 0..5, random fraction .1..6')
    if args.bag_folds not in (0, 2, 3, 4, 5):
        raise ValueError('bag folds must be 0 or between 2 and 5')
    tr, te, sample, targets, context = v1.read_data()
    baseline_df, v4_pred, days, early, later, v4 = load_baseline()
    baseline = baseline_df[KEYS].copy()
    baseline['v4_anchor'] = v4_pred
    cams = sorted(set(tr['unit'].astype(str)).union(te['unit'].dropna().astype(str)))
    tr['day'] = tr['day'].astype(str)
    if args.mode == 'audit':
        print('=== V6 CHECKPOINT (NO TRAINING) ===')
        print('Reproduced V4 MAE:', round(mae(baseline_df['people'], v4_pred), 6))
        print('Heldout days:', days, '| baseline rows:', len(baseline_df))
        print('Train cameras:', tr['unit'].nunique(), '| all cameras:', len(cams))
        for day in days:
            day_frame = tr.loc[tr['day'].eq(day)].copy()
            print(day, test_coverage(day_frame, targets))
        print('No new Kaggle score, no submission, no training.')
        return
    if args.mode == 'validate' and shutil.disk_usage(ROOT).free < 5_000_000_000:
        raise RuntimeError('Fewer than 5 GB free; cannot safely train new models. Free disk space first.')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    model_dir = MODELS / run_id
    reports = []
    details = []
    for day in days:
        history = tr.loc[tr['day'].lt(day)].copy()
        holdout = tr.loc[tr['day'].eq(day)].copy()
        if history.empty or holdout.empty or set(history['day']) & set(holdout['day']):
            raise ValueError('Invalid forward-date fold')
        print(f'\n=== V6 FORWARD-DATE HOLDOUT: {day} ===')
        Xtr, ytr, anchtr, counters = prepare_training(history, targets, cams, args.seed,
                                                     args.augments, args.random_fraction)
        val, Xva, anchva, v4_day, matched, requested = prepare_holdout(
            history, holdout, targets, cams, baseline
        )
        kinds = val['kind'].astype(str).to_numpy()
        print(f"Train: {len(Xtr)} rows; {len(Xtr.columns)} features; template={counters.get('template_rows',0)}; "
              f"random={counters.get('random_rows',0)}")
        print(f'Validation: {len(val)}/{requested} ({100*matched/max(1,requested):.1f}%)')
        if args.mode == 'dry-run':
            print('Leakage-safety preflight only; no model trained.')
            continue
        base_dir = model_dir / day
        pp, best_pool = fit_predict(Xtr, ytr, Xva, base_dir/'pooled', args.time_limit, args, args.profile)
        pooled = np.maximum(0., pp)
        special = pooled.copy()
        ftr = Xtr['kind'].astype(str).eq('forecast').to_numpy()
        fval = kinds == 'forecast'
        best_special = None
        if args.specialist_time_limit > 0 and ftr.sum() >= 100 and fval.any():
            print(f'Fitting forecast residual specialist: {int(ftr.sum())} pseudo-targets')
            xp, best_special = fit_predict(
                Xtr.loc[ftr].reset_index(drop=True), ytr[ftr]-anchtr[ftr],
                Xva.loc[fval].reset_index(drop=True),
                base_dir/'forecast_residual', args.specialist_time_limit, args, args.profile
            )
            special[fval] = np.maximum(0., anchva[fval] + np.clip(xp, -100., 100.))
        day_frame = val[KEYS+['weekday', 'horizon', 'people']].reset_index(drop=True).copy()
        day_frame['v4_anchor'] = v4_day
        day_frame['v6_pooled'] = pooled
        day_frame['v6_forecast'] = special
        if not np.isfinite(day_frame[['v4_anchor','v6_pooled','v6_forecast']].to_numpy(float)).all():
            raise ValueError('Nonfinite validation predictions')
        details.append(day_frame)
        report = {'day':day, 'train':counters, 'n_val':len(val), 'requested':requested,
                  'coverage_pct':round(100*matched/max(1,requested),2),
                  'v4_mae':mae(day_frame['people'],day_frame['v4_anchor']),
                  'v6_pooled_mae':mae(day_frame['people'],day_frame['v6_pooled']),
                  'v6_forecast_mae':mae(day_frame['people'],day_frame['v6_forecast']),
                  'best_pooled_model':best_pool, 'best_specialist_model':best_special}
        reports.append(report)
        print(f"V4={report['v4_mae']:.4f}, pooled={report['v6_pooled_mae']:.4f}, "
              f"forecast-specialist={report['v6_forecast_mae']:.4f}")
    if args.mode == 'dry-run':
        print('\nV6 DRY-RUN PASS; no models trained, no Kaggle CSV created.')
        return

    full = pd.concat(details, ignore_index=True)
    if len(full) != len(baseline_df):
        raise ValueError('V6 validation coverage does not equal V4; comparison unsafe')
    full = full.sort_values(KEYS).reset_index(drop=True)
    b = full['v4_anchor'].to_numpy(float)
    y = full['people'].to_numpy(float)
    day = full['day'].astype(str).to_numpy()
    unit = full['unit'].astype(str).to_numpy()
    kind = full['kind'].astype(str).to_numpy()
    minute = full['minute_of_day'].to_numpy(float)
    mask_early = day == days[0]
    mask_late = day == days[1]
    proposed = b.copy()
    picks = []
    for kind_req, camera in FOCUS:
        g = (kind==kind_req) & (unit==camera)
        on_early = g & mask_early
        row = {'kind':kind_req, 'unit':camera, 'early_n':int(on_early.sum()),
               'later_n':int((g & mask_late).sum()),
               'selection':'v4', 'alpha':0.0, 'early_gain_pct':0.0}
        if on_early.sum() >= args.min_group_rows:
            curr = np.abs(y[on_early] - b[on_early])
            variants = []
            for name in CANDIDATES:
                base_alt = full[name].to_numpy(float)
                for alpha in ALPHAS:
                    cand = np.maximum(0., (1-alpha)*b[on_early] + alpha*base_alt[on_early])
                    new = np.abs(y[on_early] - cand)
                    imp = gain_pct(float(np.mean(curr)), float(np.mean(new)))
                    minute_blocks = (minute[on_early]//20).astype(int)
                    diffs = [float(np.mean(curr[minute_blocks==k]-new[minute_blocks==k]))
                             for k in np.unique(minute_blocks) if (minute_blocks==k).sum() >= 3]
                    positive = int(sum(x > 0 for x in diffs))
                    variants.append((imp, -alpha, name, alpha, positive, len(diffs)))
            variants.sort(reverse=True)
            imp, _, name, alpha, positive, total = variants[0]
            row.update({'best_early_candidate':name, 'best_early_gain_pct':float(imp),
                        'positive_blocks':positive, 'blocks':total})
            if imp >= args.min_gain_pct and total >= 4 and positive >= 4 and positive/total >= .6:
                proposed[g] = np.maximum(0., (1-alpha)*b[g] + alpha*full[name].to_numpy(float)[g])
                row.update({'selection':name, 'alpha':float(alpha), 'early_gain_pct':float(imp)})
        picks.append(row)

    full['v6_shadow'] = proposed
    breakdown=[]
    for (d,k,c), frame in full.groupby(['day','kind','unit'], sort=True):
        yy=frame['people'].to_numpy(float)
        before=mae(yy, frame['v4_anchor'])
        after=mae(yy, frame['v6_shadow'])
        breakdown.append({'day':d,'kind':k,'unit':c,'n':len(frame),'v4_mae':before,
                          'v6_mae':after,'gain_percent':gain_pct(before,after)})
    by_camera=pd.DataFrame(breakdown)
    by_camera.to_csv(REPORTS/'ag_v6_by_camera.csv',index=False)
    grp=[]
    for (d,k), frame in full.groupby(['day','kind'],sort=True):
        yy=frame['people'].to_numpy(float)
        before=mae(yy,frame['v4_anchor'])
        after=mae(yy,frame['v6_shadow'])
        grp.append({'day':d,'kind':k,'n':len(frame),'v4_mae':before,'v6_mae':after,
                    'gain_percent':gain_pct(before,after)})
    by_kind=pd.DataFrame(grp)
    by_kind.to_csv(REPORTS/'ag_v6_per_kind.csv',index=False)
    full.to_csv(REPORTS/'ag_v6_cv_predictions.csv',index=False)
    late_by_kind=by_kind[by_kind['day'].eq(days[1])]
    bad=late_by_kind[(late_by_kind['n']>=40)&(late_by_kind['v6_mae']>1.10*late_by_kind['v4_mae'])]
    early_v4=mae(y[mask_early],b[mask_early])
    early_v6=mae(y[mask_early],proposed[mask_early])
    late_v4=mae(y[mask_late],b[mask_late])
    late_v6=mae(y[mask_late],proposed[mask_late])
    all_v4=mae(y,b)
    all_v6=mae(y,proposed)
    passed=late_v6 <= late_v4*.98 and bad.empty and any(p['selection']!='v4' for p in picks)
    summary={
        'description':'V6 augmented AutoGluon experiment and early-frozen camera gate',
        'warning':'Exploratory only: V2/V4 selected using both dates and Sep 12 has been reused in prior experiments. Not an unbiased Kaggle score.',
        'days':days, 'folds':reports, 'group_policies':picks,
        'early_mae':{'v4':early_v4,'v6':early_v6,'gain_pct':gain_pct(early_v4,early_v6)},
        'later_mae':{'v4':late_v4,'v6':late_v6,'gain_pct':gain_pct(late_v4,late_v6)},
        'combined_mae':{'v4':all_v4,'v6':all_v6,'gain_pct':gain_pct(all_v4,all_v6)},
        'severe_later_kind_regressions':bad.to_dict('records'),
        'verdict':'PASS_SHADOW_DIAGNOSTIC_ONLY' if passed else 'KEEP_V4',
        'parameters':vars(args),
        'comparison_rows':len(full),
        'competition_score_verified':False,
        'submission_created':False,
        'model_dir':str(model_dir),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS/'ag_v6_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print('\n=== V6 AUGMENTED AUTOGLUON SUMMARY ===')
    for name in ('early_mae','later_mae','combined_mae'):
        r=summary[name]
        print(f"{name:14}: V4={r['v4']:.4f}, V6={r['v6']:.4f}, gain={r['gain_pct']:+.2f}%")
    for p in picks:
        print(f" {p['kind']:15} {p['unit']:6}: {p['selection']} alpha={p['alpha']}")
    print('VERDICT:',summary['verdict'])
    print('Saved: reports/ag_v6_summary.json, ag_v6_by_camera.csv, ag_v6_per_kind.csv, ag_v6_cv_predictions.csv')
    print('No Kaggle CSV created or uploaded. MAE is not official Kaggle score.')


def main():
    p=argparse.ArgumentParser(description='Madinah AutoGluon V6 augmented forward-date experiment')
    p.add_argument('--mode',choices=('audit','dry-run','validate'),default='audit')
    p.add_argument('--profile',choices=('fast','balanced'),default='balanced')
    p.add_argument('--time-limit',type=int,default=180)
    p.add_argument('--specialist-time-limit',type=int,default=90)
    p.add_argument('--bag-folds',type=int,default=3)
    p.add_argument('--cpus',type=int,default=6)
    p.add_argument('--augments',type=int,default=2)
    p.add_argument('--random-fraction',type=float,default=.35)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-group-rows',type=int,default=60)
    p.add_argument('--min-gain-pct',type=float,default=5.0)
    args=p.parse_args()
    if args.time_limit<10 or args.specialist_time_limit<0 or args.cpus<1 or args.min_group_rows<10:
        p.error('Invalid training limits or group minimum')
    run(args)


if __name__=='__main__':
    main()
