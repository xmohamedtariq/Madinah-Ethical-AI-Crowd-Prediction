"""Madinah Ethical AI -- AutoGluon V5 chronological residual shadow trial.

Uses *saved* V2 out-of-day predictions and the V4 frozen policy. Does not
refit V2, inspect official test targets, generate a Kaggle CSV, or submit.

Two-stage experiment:
  1. Train a low-capacity AutoGluon residual model on the EARLY part of Sep 11.
     Choose conservative corrections on the LATE part of Sep 11, by camera.
  2. Refit on all Sep 11 forecast rows and evaluate once on Sep 12.

Caveats: V2's earlier selection had seen both holdout dates, and several shadow
experiments have now been evaluated on Sep 12. Therefore this is *not* a truly
independent estimate of real Kaggle performance. Never claim 0.400 from MAE.

PowerShell:
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v5_meta.py --mode audit
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v5_meta.py --mode validate --time-limit 120
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

ROOT = Path(__file__).resolve().parent
FOCUS = ('cam_E', 'cam_A', 'cam_F')
REQUIRED = {'recording', 'day', 'unit', 'minute_of_day', 'kind', 'people', 'horizon',
            'sp_rule_50', 'pooled', 'specialists', 'rules_only'}
GRID = (0.0, 0.2, 0.35, 0.5)


def mae(y, p):
    if not len(y):
        return math.nan
    return float(np.abs(np.asarray(y, float) - np.asarray(p, float)).mean())


def load_and_freeze(pred_csv: Path, v4_json: Path):
    if not pred_csv.is_file() or not v4_json.is_file():
        raise FileNotFoundError('V2 CV predictions and V4 summary are required; run V2 and V4 first')
    df = pd.read_csv(pred_csv)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f'V2 predictions missing columns: {missing}')
    if df.empty or df.duplicated(['recording', 'unit', 'minute_of_day', 'kind']).any():
        raise ValueError('Empty or duplicate CV rows')
    for col in ['day', 'unit', 'kind']:
        if df[col].isna().any():
            raise ValueError(f'Null {col}')
    for col in ('people', 'sp_rule_50', 'pooled', 'specialists', 'rules_only'):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'Invalid values in {col}')
    df['minute_of_day'] = pd.to_numeric(df['minute_of_day'], errors='raise')
    df['horizon'] = pd.to_numeric(df['horizon'], errors='coerce').fillna(0.)
    future = df['kind'].isin(['forecast', 'hidden-future'])
    if (df.loc[future, 'horizon'] < 0).any() or not np.isfinite(df['horizon']).all():
        raise ValueError('Invalid forecast horizon')
    days = sorted(df['day'].astype(str).unique())
    v4 = json.loads(v4_json.read_text(encoding='utf-8'))
    if len(days) != 2 or days != v4.get('days'):
        raise ValueError(f'V2/V4 holdout day mismatch: {days} vs {v4.get("days")}')
    if not v4.get('focus_groups'):
        raise ValueError('No frozen V4 policy found')
    v4_pred = df['sp_rule_50'].to_numpy(float).copy()
    pools = {
        'pooled': df['pooled'].to_numpy(float),
        'rules_only': df['rules_only'].to_numpy(float),
        'specialists': df['specialists'].to_numpy(float),
    }
    pools['half_pooled_specialists'] = .5 * (pools['pooled'] + pools['specialists'])
    pools['half_specialists_rules'] = .5 * (pools['specialists'] + pools['rules_only'])
    weight = v4.get('parameters', {}).get('soft_weight')
    if weight is None or not 0 < float(weight) <= 1:
        raise ValueError('Invalid V4 soft_weight')
    kind = df['kind'].astype(str).to_numpy()
    unit = df['unit'].astype(str).to_numpy()
    seen = set()
    for rule in v4['focus_groups']:
        key = (rule['kind'], rule['unit'])
        if key in seen:
            raise ValueError('Duplicate V4 policy group')
        seen.add(key)
        name = rule['selected']
        if name == 'sp_rule_50':
            continue
        if name not in pools:
            raise ValueError(f'Unrecognized frozen V4 source: {name}')
        mask = (kind == key[0]) & (unit == key[1])
        v4_pred[mask] = (1-float(weight))*v4_pred[mask] + float(weight)*pools[name][mask]
    early_mask = df['day'].eq(days[0]).to_numpy()
    late_mask = df['day'].eq(days[1]).to_numpy()
    y = df['people'].to_numpy(float)
    for tag, mask in [('early', early_mask), ('later', late_mask), ('combined', np.ones(len(df), bool))]:
        given = v4.get(tag, {})
        computed = mae(y[mask], v4_pred[mask])
        if int(given.get('n', -1)) != int(mask.sum()) or not math.isclose(computed, given['v4_mae_people'], rel_tol=0, abs_tol=2e-5):
            raise ValueError(f'Cannot reproduce V4 {tag} MAE: {computed:.6f}, expected {given.get("v4_mae_people")}')
    return df, v4_pred, days, early_mask, late_mask, v4


def meta_features(df: pd.DataFrame, anchor: np.ndarray):
    # Only prediction outputs and known metadata; NEVER include people, date,
    # recording, future observations, or a derived target label as a feature.
    minute = df['minute_of_day'].to_numpy(float)
    h = df['horizon'].to_numpy(float)
    base = df['sp_rule_50'].to_numpy(float)
    pooled = df['pooled'].to_numpy(float)
    sp = df['specialists'].to_numpy(float)
    rule = df['rules_only'].to_numpy(float)
    pstack = np.column_stack([base, pooled, sp, rule, anchor])
    X = pd.DataFrame({
        'unit': df['unit'].astype(str).to_numpy(),
        'kind': df['kind'].astype(str).to_numpy(),
        'minute': minute,
        'minute_sin': np.sin(2 * np.pi * minute / 1440),
        'minute_cos': np.cos(2 * np.pi * minute / 1440),
        'horizon': h,
        'horizon_short': (h <= 5).astype(int),
        'horizon_mid': ((h > 5) & (h <= 10)).astype(int),
        'v4_pred': anchor,
        'v2_pred': base,
        'pooled_pred': pooled,
        'specialist_pred': sp,
        'rules_pred': rule,
        'pool_minus_v4': pooled - anchor,
        'sp_minus_rule': sp - rule,
        'pool_minus_rule': pooled - rule,
        'sp_minus_v4': sp - anchor,
        'v4_minus_rule': anchor - rule,
        'pred_range': np.ptp(pstack, axis=1),
        'pred_std': pstack.std(axis=1),
        'level_log1p': np.log1p(np.maximum(anchor, 0.)),
    })
    assert 'people' not in X.columns and 'day' not in X.columns and 'recording' not in X.columns
    return X


def make_chrono_masks(df: pd.DataFrame, eligible: np.ndarray, train_frac=0.68):
    """Hold out later time across ALL cameras in each early-day recording.

    Unlike a per-camera split, no label from a minute *after* the cutoff in
    another camera can enter the early chronological training partition.
    """
    train = np.zeros(len(df), dtype=bool)
    valid = np.zeros(len(df), dtype=bool)
    for _, part in df.loc[eligible].groupby('recording', sort=True):
        minutes = np.sort(part['minute_of_day'].unique())
        if len(minutes) < 24:
            raise ValueError('Not enough unique early forecast minutes for chronological split')
        cut_idx = min(max(10, int(np.floor(len(minutes) * train_frac))), len(minutes)-8)
        cutoff = minutes[cut_idx - 1]
        inds = part.index.to_numpy()
        times = part['minute_of_day'].to_numpy(float)
        train[inds[times <= cutoff]] = True
        valid[inds[times > cutoff]] = True
    if (train & valid).any() or not train.any() or not valid.any():
        raise ValueError('Chronological split failed')
    return train, valid


def fit_meta(X: pd.DataFrame, yres: np.ndarray, mask: np.ndarray, folder: Path, seconds: int, cpus: int):
    from autogluon.tabular import TabularPredictor
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        raise FileExistsError(f'Refusing overwrite: {folder}')
    d = X.loc[mask].reset_index(drop=True).copy()
    d['__residual__'] = yres[mask]
    if len(d) < 80 or not np.isfinite(d['__residual__']).all():
        raise ValueError('Not enough finite early forecast data for residual meta model')
    hps = {
        'GBM': [{'learning_rate': .045, 'num_leaves': 12, 'n_estimators': 350}],
        'XGB': [{'max_depth': 3, 'n_estimators': 360, 'learning_rate': .045,
                 'subsample': .9, 'colsample_bytree': .9}],
        'RF': [{}],
    }
    return TabularPredictor(
        label='__residual__', problem_type='regression',
        eval_metric='mean_absolute_error', path=str(folder)
    ).fit(train_data=d, presets='medium_quality',
          hyperparameters=hps, time_limit=seconds,
          num_bag_folds=0, num_gpus=0, num_cpus=min(cpus, os.cpu_count() or 1))


def predict_residual(model, X: pd.DataFrame, mask: np.ndarray, anchor: np.ndarray, max_correction: float):
    if not mask.any():
        return np.zeros(0)
    p = np.asarray(model.predict(X.loc[mask]), dtype=float)
    if not np.isfinite(p).all() or len(p) != int(mask.sum()):
        raise ValueError('Invalid meta residual predictions')
    cap = np.minimum(max_correction, 8. + 0.20 * np.maximum(0, anchor[mask]))
    return np.clip(p, -cap, cap)


def choose_early_alphas(df, y, anchor, valid, delta, min_rows, min_gain):
    sel = {}
    details = []
    unit = df['unit'].astype(str).to_numpy()
    minute = df['minute_of_day'].to_numpy(float)
    for cam in FOCUS:
        mask = valid & (unit == cam)
        count = int(mask.sum())
        base = mae(y[mask], anchor[mask])
        detail = {'camera': cam, 'early_val_rows': count,
                  'early_val_v4_mae': None if not np.isfinite(base) else base,
                  'alpha': 0.0, 'early_gain_percent': 0.,
                  'positive_blocks': 0, 'total_blocks': 0}
        if count >= min_rows and base > 0:
            errs = []
            base_abs = np.abs(y[mask] - anchor[mask])
            for alpha in GRID:
                cand = np.maximum(0., anchor[mask] + alpha*delta[mask])
                gain = 100. * (base - mae(y[mask], cand))/base
                new_abs = np.abs(y[mask] - cand)
                blocks = (minute[mask]//15).astype(int)
                block_diff = [float(np.mean(base_abs[blocks==k] - new_abs[blocks==k]))
                              for k in np.unique(blocks) if int(np.sum(blocks==k))>=3]
                positive = sum(g > 0 for g in block_diff)
                errs.append((gain, -alpha, alpha, positive, len(block_diff)))
            errs.sort(reverse=True)
            gain, _, alpha, positive, total = errs[0]
            detail.update({'best_early_gain_percent': float(gain),
                           'positive_blocks': int(positive), 'total_blocks': int(total)})
            if alpha > 0 and gain >= min_gain and total >= 2 and positive/total >= .60:
                sel[cam] = float(alpha)
                detail['alpha'] = float(alpha)
                detail['early_gain_percent'] = float(gain)
        details.append(detail)
    return sel, details


def run(args):
    df, v4_pred, days, early, later, v4 = load_and_freeze(args.input, args.v4_summary)
    y = df['people'].to_numpy(float)
    kinds = df['kind'].astype(str).to_numpy()
    cameras = df['unit'].astype(str).to_numpy()
    early_forecast = early & (kinds == 'forecast')
    print('=== AUTOGLUON V5 CHRONOLOGICAL META DIAGNOSTIC ===')
    print(f'V4 reproduced on {len(df)} rows: early={mae(y[early],v4_pred[early]):.4f}; '
          f'late={mae(y[later],v4_pred[later]):.4f}; combined={mae(y,v4_pred):.4f}')
    print(f'Early forecast rows: {int(early_forecast.sum())}. Focus: {", ".join(FOCUS)}')
    if args.mode == 'audit':
        print('AUDIT PASS: no training, no Kaggle CSV produced.')
        return None
    if shutil.disk_usage(ROOT).free/(1024**3) < 4:
        raise RuntimeError('Free at least 4 GB of disk space before fitting AutoGluon')
    X = meta_features(df, v4_pred)
    train, valid = make_chrono_masks(df, early_forecast)
    print(f'Early chronological residual split: fit={int(train.sum())} calibrate={int(valid.sum())}')
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = args.models / tag
    residuals = y - v4_pred
    first_model = fit_meta(X, residuals, train, folder/'early_chrono',
                           args.time_limit, args.cpus)
    val_delta = np.zeros(len(df), dtype=float)
    val_delta[valid] = predict_residual(first_model, X, valid, v4_pred, args.max_correction)
    alphas, choice_details = choose_early_alphas(
        df, y, v4_pred, valid, val_delta, args.min_rows, args.min_gain_percent)
    print('Frozen alpha decisions, determined only from Sep 11 chronological calibration:')
    for row in choice_details:
        print(f"  {row['camera']:6} n={row['early_val_rows']:3} "
              f"alpha={row['alpha']:.2f} gain={row['early_gain_percent']:+.2f}%")
    if not alphas:
        print('No early-calibrated camera correction passed. KEEP_V4.')
        summary = {
            'verdict': 'KEEP_V4', 'reason': 'no early-chronological correction passed',
            'days': days, 'selection_details': choice_details,
            'v4_late_mae_people': mae(y[later],v4_pred[later]),
            'v5_late_mae_people': None, 'v5_late_gain_percent': None,
            'competition_score_verified': False, 'submission_created': False,
            'warning': 'V2/V4 were previously selected using these folds; not independent Kaggle performance.',
        }
        args.reports.mkdir(parents=True, exist_ok=True)
        (args.reports/'ag_v5_meta_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    final_model = fit_meta(X, residuals, early_forecast, folder/'all_early_forecast',
                           args.time_limit, args.cpus)
    late_delta = np.zeros(len(df), dtype=float)
    late_focus = later & (kinds == 'forecast') & np.isin(cameras, list(alphas))
    late_delta[late_focus] = predict_residual(final_model, X, late_focus, v4_pred, args.max_correction)
    candidate = v4_pred.copy()
    for cam, alpha in alphas.items():
        m = late_focus & (cameras == cam)
        candidate[m] = np.maximum(0.0, v4_pred[m] + alpha*late_delta[m])
    # Avoid presenting in-sample early predictions as a purported combined CV.
    rows = []
    for (kind, unit), part in df.loc[later].groupby(['kind', 'unit'], sort=True):
        indices = part.index.to_numpy()
        score_v4 = mae(y[indices], v4_pred[indices])
        score_v5 = mae(y[indices], candidate[indices])
        rows.append({'kind': kind, 'unit': unit, 'n': len(indices),
                     'v4_later_mae_people': score_v4, 'v5_later_mae_people': score_v5,
                     'gain_percent': 100*(score_v4-score_v5)/score_v4 if score_v4 > 0 else 0.,
                     'correction_alpha': alphas.get(unit, 0.) if kind=='forecast' else 0.})
    table = pd.DataFrame(rows)
    impacted = table[table['correction_alpha'].gt(0) & table['n'].ge(40)]
    severe = impacted[impacted['v5_later_mae_people'] > 1.10*impacted['v4_later_mae_people']]
    v4_later = mae(y[later], v4_pred[later])
    v5_later = mae(y[later], candidate[later])
    gain = 100*(v4_later - v5_later)/v4_later
    # The only promotion criterion uses the *later* day; still exploratory due to prior reuse.
    verdict = ('PASS_SHADOW_DIAGNOSTIC_ONLY' if gain>=2 and severe.empty else 'KEEP_V4')
    summary = {
        'verdict': verdict, 'days': days,
        'early_chrono_fit_rows': int(train.sum()),
        'early_chrono_calibration_rows': int(valid.sum()),
        'early_total_forecast_rows': int(early_forecast.sum()),
        'later_rows': int(later.sum()),
        'policy': 'Forecast-only AutoGluon residual; early-only chronological alpha selection; V4 unchanged otherwise',
        'selected_alpha_by_camera': alphas,
        'selection_details': choice_details,
        'v4_late_mae_people': v4_later,
        'v5_late_mae_people': v5_later,
        'v5_late_gain_percent': gain,
        'later_regressions_over_10pct': severe.to_dict('records'),
        'models_folder': str(folder),
        'competition_score_verified': False,
        'submission_created': False,
        'warning': 'V2/V4 were previously selected using these two folds and the Sep 12 shadow set has been reused. Not independent or official Kaggle score. MAE is in people.',
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports/'ag_v5_meta_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    table.sort_values(['kind', 'v4_later_mae_people'], ascending=[True, False]).to_csv(
        args.reports/'ag_v5_meta_later_by_camera.csv', index=False)
    out = df.loc[later, ['recording', 'day', 'unit', 'minute_of_day', 'kind', 'people']].copy()
    out['v4_pred'] = v4_pred[later]
    out['v5_pred'] = candidate[later]
    out.to_csv(args.reports/'ag_v5_meta_later_predictions.csv', index=False)
    print(f'Later-day V4 MAE: {v4_later:.6f} people')
    print(f'Later-day V5 MAE: {v5_later:.6f} people; gain: {gain:+.2f}%')
    print('Verdict:', verdict)
    print('Wrote reports/ag_v5_meta_summary.json and diagnostics. NO Kaggle submission.')
    return summary


def main():
    p = argparse.ArgumentParser(description='AutoGluon V5: forecast residual shadow on V4 baseline; never submits')
    p.add_argument('--mode', choices=['audit','validate'], default='audit')
    p.add_argument('--input', type=Path, default=ROOT/'reports'/'ag_v2_cv_predictions.csv')
    p.add_argument('--v4-summary', type=Path, default=ROOT/'reports'/'ag_v4_frozen_camera_summary.json')
    p.add_argument('--reports', type=Path, default=ROOT/'reports')
    p.add_argument('--models', type=Path, default=ROOT/'models'/'ag_v5_meta')
    p.add_argument('--time-limit', type=int, default=120)
    p.add_argument('--cpus', type=int, default=6)
    p.add_argument('--max-correction', type=float, default=30)
    p.add_argument('--min-rows', type=int, default=30)
    p.add_argument('--min-gain-percent', type=float, default=3.)
    args = p.parse_args()
    if args.time_limit < 30 or args.cpus < 1 or args.max_correction <= 0 or args.min_rows < 12 or args.min_gain_percent < 0:
        p.error('Invalid limits')
    run(args)


if __name__ == '__main__':
    main()
