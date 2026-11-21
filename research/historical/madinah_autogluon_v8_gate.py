"""Madinah AutoGluon V8: conservative chronological prediction-error meta gate.

Uses cached V7 outer-holdout predictions, without rerunning heavy V1-V7 training.
Fits an AutoGluon residual model using Sep 11 only. Its strength is selected
on the trailing chronological portion of Sep 11 using a model fit on the
preceding portion. After freezing that strength, refits on Sep 11 and observes
its behavior on Sep 12, a REUSED SHADOW set (not a fresh blind holdout).

IMPORTANT: V7 baseline itself was tuned on Sep 11, and Sep 12 has been examined
in V1-V7. All results are exploratory; do not claim unbiased improvement.
This code does NOT access official test labels, generate submission CSVs, or
submit to Kaggle. The reported metric is MAE in people, NOT Kaggle's metric.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models' / 'ag_v8_gate'
INPUT_CSV = REPORTS / 'ag_v7_cv_predictions.csv'
INPUT_JSON = REPORTS / 'ag_v7_summary.json'
KEYS = ['recording', 'day', 'unit', 'minute_of_day', 'kind']
BASE = 'v7_accepted'
PRED_PREFIX = 'candidate_'
CORRECTION_LABEL = '__meta_residual_v8__'


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def pct_gain(old, new):
    return 100.0 * (old - new) / old if old > 1e-12 else 0.0


def read_v7():
    if not INPUT_CSV.exists() or not INPUT_JSON.exists():
        raise FileNotFoundError('Run V7 validation first; missing: ' + str(INPUT_CSV))
    data = pd.read_csv(INPUT_CSV, low_memory=False)
    required = set(KEYS + ['people', BASE, 'v6_shadow'])
    if required - set(data):
        raise ValueError('Missing V7 fields: ' + str(sorted(required - set(data))))
    if not any(c.startswith(PRED_PREFIX) for c in data):
        raise ValueError('No V7 candidate columns were found')
    if data.empty or data.duplicated(KEYS).any():
        raise ValueError('V7 rows empty or duplicate outer-holdout keys')
    data['day'] = data['day'].astype(str)
    days = sorted(data['day'].unique())
    if len(days) != 2 or days != ['2026-09-11', '2026-09-12']:
        raise ValueError('Expected V7 September 11 and 12 chronological holdouts')
    vals = pd.to_numeric(data['people'], errors='coerce').to_numpy(float)
    base = pd.to_numeric(data[BASE], errors='coerce').to_numpy(float)
    if not (np.isfinite(vals).all() and np.isfinite(base).all() and (vals >= 0).all() and (base >= 0).all()):
        raise ValueError('Invalid held-out truth or V7 prediction')
    summary = json.loads(INPUT_JSON.read_text(encoding='utf-8'))
    expected = summary.get('accepted_mae', summary.get('proposed_mae'))
    if expected is None or not np.isclose(mae(vals, base), float(expected), atol=2e-4, rtol=0):
        raise ValueError('V7 summary and predictions disagree; do not proceed')
    if summary.get('rows') != len(data):
        raise ValueError('V7 row count mismatch')
    data = data.sort_values(KEYS).reset_index(drop=True)
    return data, summary, days


def features(df):
    """Only use causal PREDICTIONS and public identifiers, never labels.

    All candidate_* values were made by V7 from causal features. There is no
    label target, same-day ground truth, future context, or date in model X.
    """
    candidates = sorted(c for c in df if c.startswith(PRED_PREFIX))
    if not candidates:
        raise ValueError('No input candidates')
    arr = df[candidates].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    base = pd.to_numeric(df[BASE], errors='coerce').to_numpy(float)
    if not np.isfinite(base).all():
        raise ValueError('Invalid baseline')
    arr = np.where(np.isfinite(arr), arr, base[:, None])
    arr = np.maximum(arr, 0.0)
    diff = arr - base[:, None]
    X = pd.DataFrame(index=range(len(df)))
    X['unit'] = df['unit'].astype(str).to_numpy()
    X['kind'] = df['kind'].astype(str).to_numpy()
    minute = pd.to_numeric(df['minute_of_day'], errors='raise').to_numpy(float)
    X['minute_of_day'] = minute
    X['minute_sin'] = np.sin(2*np.pi*minute/1440.0)
    X['minute_cos'] = np.cos(2*np.pi*minute/1440.0)
    X['base_prediction'] = base
    X['base_log'] = np.log1p(np.maximum(base, 0.0))
    X['v6_diff'] = pd.to_numeric(df['v6_shadow'], errors='coerce').fillna(pd.Series(base)).to_numpy(float) - base
    X['expert_mean_diff'] = diff.mean(axis=1)
    X['expert_median_diff'] = np.median(diff, axis=1)
    X['expert_std'] = diff.std(axis=1)
    X['expert_max_diff'] = diff.max(axis=1)
    X['expert_min_diff'] = diff.min(axis=1)
    X['expert_range'] = np.ptp(arr, axis=1)
    X['expert_robust_scale'] = X['expert_range'].to_numpy() / (base + 10.)
    X['expert_above_count'] = (diff > 0).sum(axis=1)
    X['expert_below_count'] = (diff < 0).sum(axis=1)
    for c, column in zip(candidates, diff.T):
        short = c[len(PRED_PREFIX):]
        X['expert_'+short] = np.clip(column, -300.0, 300.0)
    # Never accidentally include a labeled value as a feature.
    if {'people', 'day', 'recording', 'id', CORRECTION_LABEL} & set(X):
        raise ValueError('A label/date/identifier slipped into V8 features')
    if X.columns.duplicated().any() or np.isinf(X.select_dtypes(include='number').to_numpy()).any():
        raise ValueError('Bad meta feature matrix')
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def chronological_split(early, fraction=0.65):
    minutes = np.sort(early['minute_of_day'].astype(float).unique())
    if len(minutes) < 10:
        raise ValueError('Too few distinct early-day minutes for calibration')
    cut_idx = max(1, min(len(minutes)-2, int(np.floor(len(minutes)*fraction))-1))
    cut = float(minutes[cut_idx])
    train = early['minute_of_day'].astype(float).to_numpy() <= cut
    cal = ~train
    min_train = max(20, min(50, int(len(early)*0.35)))
    min_cal = max(15, min(30, int(len(early)*0.25)))
    if train.sum() < min_train or cal.sum() < min_cal:
        raise ValueError('Not enough rows in chronological early-day split')
    return train, cal, cut


def clipped_add(base, correction, alpha):
    """Limit the possible damage from severe out-of-regime meta errors."""
    b = np.asarray(base, float)
    c = np.asarray(correction, float)
    lim = np.maximum(12.0, 0.55*np.maximum(b, 0))
    return np.maximum(0.0, b + float(alpha)*np.clip(c, -lim, lim))


def frozen_policy(early_cal, cal_correction, args):
    """Choose camera/type alpha from Sep 11 calibration only."""
    baseline = early_cal[BASE].to_numpy(float)
    y = early_cal['people'].to_numpy(float)
    minute = early_cal['minute_of_day'].to_numpy(float)
    kinds = early_cal['kind'].astype(str).to_numpy()
    cams = early_cal['unit'].astype(str).to_numpy()
    groups = []
    alpha_by_group = {}
    pool = (0.0, 0.15, 0.25, 0.4, 0.6, 0.8)
    for k, c in sorted(set(zip(kinds, cams))):
        mask = (kinds==k) & (cams==c)
        n = int(mask.sum())
        entry = {'kind':k,'unit':c,'calibration_rows':n,'alpha':0.0}
        if n < args.min_calibration_rows:
            entry['reason'] = 'too_few_calibration_rows'
            groups.append(entry)
            continue
        old_abs = np.abs(y[mask]-baseline[mask]); old = float(old_abs.mean())
        best = (0.0, 0.0, 0, 0, 0.0)
        block = (minute[mask]//args.block_minutes).astype(int)
        for a in pool:
            p = clipped_add(baseline[mask], cal_correction[mask], a)
            new_abs = np.abs(y[mask]-p)
            g = pct_gain(old, float(new_abs.mean()))
            wins = 0
            count = 0
            for b in np.unique(block):
                seg = (block == b)
                if seg.sum() < 2:
                    continue
                count += 1
                wins += int((old_abs[seg]-new_abs[seg]).mean()>0)
            option=(g, a, wins, count, float(new_abs.mean()))
            if option[0] > best[0]+1e-8:
                best = option
        g,a,wins,count,new = best
        entry.update({'gain_pct_on_calibration':round(float(g),4), 'winning_blocks':wins,'blocks':count,
                      'baseline_calibration_mae':old,'candidate_calibration_mae':new})
        accepted = a>0 and g >= args.min_calibration_gain and count >= 2 and wins/max(count,1) >= args.min_block_win
        if accepted:
            entry['alpha'] = float(a)
            alpha_by_group[(k,c)] = float(a)
        groups.append(entry)
    return alpha_by_group, groups


def apply_policy(frame, correction, policy):
    base = frame[BASE].to_numpy(float)
    out = base.copy()
    cam = frame['unit'].astype(str).to_numpy()
    kind = frame['kind'].astype(str).to_numpy()
    for (k,c),alpha in policy.items():
        m=(kind==k)&(cam==c)
        out[m]=clipped_add(base[m],correction[m],alpha)
    if not np.isfinite(out).all() or (out<0).any():
        raise ValueError('Invalid V8 predictions')
    return out


def fit_model(X, residual, args, output_path):
    from autogluon.tabular import TabularPredictor
    if output_path.exists():
        raise FileExistsError('Refusing to overwrite model: '+str(output_path))
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite residual label')
    if args.profile=='fast':
        hyper={'GBM':[{'learning_rate':0.04,'num_leaves':12,'n_estimators':300}],
               'XGB':[{'learning_rate':0.04,'max_depth':3,'n_estimators':300}], 'RF':[{}]}
    else:
        hyper={'GBM':[{'learning_rate':0.04,'num_leaves':12,'n_estimators':350},
                      {'learning_rate':0.035,'num_leaves':20,'n_estimators':500}],
               'XGB':[{'learning_rate':0.045,'max_depth':3,'n_estimators':400}],
               'CAT':[{'iterations':350,'depth':4,'learning_rate':0.05,'verbose':False}],
               'RF':[{}]}
    t=X.copy()
    t[CORRECTION_LABEL]=np.asarray(residual,float)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    predictor=TabularPredictor(label=CORRECTION_LABEL,problem_type='regression',
                eval_metric='mean_absolute_error',path=str(output_path)).fit(
        train_data=t,presets='medium_quality',hyperparameters=hyper,
        num_cpus=min(args.cpus,os.cpu_count() or 1),num_gpus=0,
        time_limit=args.time_limit,num_bag_folds=0,
        num_stack_levels=0,dynamic_stacking=False)
    return predictor


def grouped_report(frame, candidate, group_cols):
    z=frame.copy()
    z['v8_candidate']=candidate
    rows=[]
    for k,part in z.groupby(group_cols,sort=True):
        if not isinstance(k,tuple): k=(k,)
        rows.append(dict(zip(group_cols,k),n=len(part),
            v7_mae=mae(part.people,part[BASE]),
            v8_mae=mae(part.people,part.v8_candidate),
            improvement_pct=pct_gain(mae(part.people,part[BASE]),mae(part.people,part.v8_candidate))))
    return pd.DataFrame(rows)


def run(args):
    df, report, days = read_v7()
    early=df[df.day.eq(days[0])].copy().reset_index(drop=True)
    late=df[df.day.eq(days[1])].copy().reset_index(drop=True)
    print('=== V8 FROZEN META-GATE (NO KAGGLE SUBMISSION) ===')
    print('V7 baseline MAE: %.6f on %d matched rows' % (mae(df.people,df[BASE]),len(df)))
    print('Early %s: %d rows | Later %s: %d rows' % (days[0],len(early),days[1],len(late)))
    if args.mode=='audit':
        for k,group in df.groupby('kind'):
            print('%-14s %4d rows | V7 MAE %.4f' % (k,len(group),mae(group.people,group[BASE])))
        print('AUDIT PASS; NO TRAINING, NO Kaggle CSV.')
        return
    tr_mask,cal_mask,cut=chronological_split(early,args.calibration_fraction)
    Xe=features(early)
    Xl=features(late)
    if list(Xe.columns)!=list(Xl.columns):
        raise ValueError('Train/test meta feature mismatch')
    print('Chronological early-day train=%d, calibration=%d; cutoff minute=%d; features=%d' %
          (tr_mask.sum(),cal_mask.sum(),int(cut),len(Xe.columns)))
    print('WARNING: Earlier V7 policy selected on Sep 11; Sep 12 reused. Shadow test only.')
    if args.mode=='dry-run':
        print('DRY-RUN PASS: no training or Kaggle CSV.')
        return
    if shutil.disk_usage(ROOT).free < 4_000_000_000:
        raise RuntimeError('Less than 4 GB free disk; avoid further AutoGluon models.')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    train_y=(early['people']-early[BASE]).to_numpy(float)
    model_cal=fit_model(Xe.loc[tr_mask].reset_index(drop=True),train_y[tr_mask],args,MODELS/stamp/'early_partial')
    pred_cal=np.asarray(model_cal.predict(Xe.loc[cal_mask].reset_index(drop=True)),float)
    if len(pred_cal)!=int(cal_mask.sum()) or not np.isfinite(pred_cal).all():
        raise ValueError('Invalid calibration correction')
    policy,selection=frozen_policy(early.loc[cal_mask].reset_index(drop=True),pred_cal,args)
    print('Selected correction strengths from Sep 11 calibration ONLY:')
    if policy:
        for (k,c),alpha in sorted(policy.items()):
            print('  %-14s %-7s alpha=%.2f' % (k,c,alpha))
    else:
        print('  None: retain V7')
    # Refit exclusively on all of Sep 11. Sep 12 labels are never read by fit.
    if policy:
        model_final=fit_model(Xe,train_y,args,MODELS/stamp/'early_full')
        pred_late=np.asarray(model_final.predict(Xl),float)
        if not np.isfinite(pred_late).all():
            raise ValueError('Invalid late-day corrections')
    else:
        # No calibrated correction was accepted; avoid wasting disk and time.
        pred_late=np.zeros(len(late),dtype=float)
    new_late=apply_policy(late,pred_late,policy)
    old_mae=mae(late.people,late[BASE]); new_mae=mae(late.people,new_late)
    per_kind=grouped_report(late,new_late,['kind'])
    severe=per_kind.loc[(per_kind.n>=40)&(per_kind.v8_mae>per_kind.v7_mae*1.10),'kind'].tolist()
    passed=bool(policy and pct_gain(old_mae,new_mae)>=args.min_later_gain and not severe)
    verdict='PASS_REUSED_SHADOW_DIAGNOSTIC_ONLY' if passed else 'KEEP_V7'
    # We deliberately DO NOT compute any in-sample combined V8 MAE.
    # Early labels were used to train the meta model, so it would be misleading.
    result={
        'method':'Sep11-only AutoGluon residual meta gate on V7 out-of-fold candidate diversity',
        'warning':'Sep11 V7 baseline policy was tuned on Sep11; Sep12 was reused in V1-V7. NOT independent validation. MAE people is NOT Kaggle metric.',
        'rows_v7_total':len(df),'selection_train_rows':int(tr_mask.sum()),
        'selection_calibration_rows':int(cal_mask.sum()),'shadow_later_rows':len(late),
        'days':days,'alpha_selected_from':'Sep11 trailing chronological calibration only',
        'selected_policy':selection,'later_v7_mae_people':old_mae,'later_v8_mae_people':new_mae,
        'later_gain_percent':pct_gain(old_mae,new_mae),'later_serious_regressions':severe,
        'v6_full_cv_mae_people':report.get('v6_mae'),
        'original_55pct_from_v6_target_mae_people':report.get('target_mae'),
        'note_on_targets':'Full combined V8 MAE not computed: Sep11 truth trains the meta model. Later score cannot be compared numerically to an all-days target.',
        'verdict':verdict,'competition_score_verified':False,'kaggle_submission_created':False,
    }
    REPORTS.mkdir(parents=True,exist_ok=True)
    (REPORTS/'ag_v8_meta_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    per_kind.to_csv(REPORTS/'ag_v8_meta_per_kind.csv',index=False)
    grouped_report(late,new_late,['kind','unit']).to_csv(REPORTS/'ag_v8_meta_by_camera.csv',index=False)
    late_out=late[KEYS+['people',BASE]].copy()
    late_out['v8_candidate']=new_late
    late_out.to_csv(REPORTS/'ag_v8_late_shadow_predictions.csv',index=False)
    print('Later-day MAE: V7=%.6f | V8=%.6f | gain=%+.2f%%' %
          (old_mae,new_mae,pct_gain(old_mae,new_mae)))
    print('VERDICT:',verdict)
    print('Saved reports/ag_v8_meta_summary.json, per-kind and per-camera diagnostics.')
    print('NO Kaggle submission created. This is NOT a fresh independent holdout.')


def main():
    parser=argparse.ArgumentParser(description='V8 AutoGluon residual gate; no Kaggle submission')
    parser.add_argument('--mode',choices=['audit','dry-run','validate'],default='audit')
    parser.add_argument('--profile',choices=['fast','balanced'],default='fast')
    parser.add_argument('--time-limit',type=int,default=90)
    parser.add_argument('--cpus',type=int,default=6)
    parser.add_argument('--calibration-fraction',type=float,default=0.65)
    parser.add_argument('--min-calibration-rows',type=int,default=12)
    parser.add_argument('--min-calibration-gain',type=float,default=5.0)
    parser.add_argument('--min-later-gain',type=float,default=2.0)
    parser.add_argument('--min-block-win',type=float,default=0.55)
    parser.add_argument('--block-minutes',type=int,default=15)
    a=parser.parse_args()
    if a.time_limit<10 or a.cpus<1 or not (0.5<=a.calibration_fraction<=0.8) or a.min_calibration_rows<5 or not (0<=a.min_block_win<=1):
        parser.error('Invalid training/calibration settings')
    run(a)


if __name__=='__main__':
    main()
