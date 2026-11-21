"""
frozen_protocol.py - Fixed, unchangeable validation protocol for Madinah Ethical AI.
Provides:
1. Frozen validation split on 20260911_1100 (Friday) and 20260912_1100 (Saturday).
2. Frozen synthetic gaps matching exact test geometry and random seeds.
3. Frozen reference baseline predictions.
4. Exact metric calculation matching Kaggle public/private split definition:
   Score = 0.5 * Score(Fold1) + 0.5 * Score(Fold2)
         = 0.25 * sum_k [ 0.5 * (Ratio_k(Fold1) + Ratio_k(Fold2)) ]
"""
import sys
import os
import hashlib
import json
import pandas as pd
import numpy as np

WORKSPACE_DIR = r'C:\Users\USER\Desktop\Madinah-AI-Fresh'
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data', 'official')
PROTO_CACHE_DIR = os.path.join(WORKSPACE_DIR, 'frozen_protocol_cache')
os.makedirs(PROTO_CACHE_DIR, exist_ok=True)

def generate_frozen_folds(random_seed=2026):
    """
    Generates and caches the exact, immutable synthetic gaps for Fold 1 and Fold 2.
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    folds = {}
    
    for fold_rec, hu_list in [('20260911_1100', ['cam_B', 'cam_G']), ('20260912_1100', ['cam_D'])]:
        rng = np.random.RandomState(random_seed)
        val_data = train_df[train_df['recording'] == fold_rec].copy()
        train_context = train_df[train_df['recording'] != fold_rec].copy()
        
        all_minutes = sorted(val_data['minute_of_day'].unique())
        all_units = sorted(val_data['unit'].unique())
        vis_units = [u for u in all_units if u not in hu_list]
        
        target_nows = [682, 736, 787, 843, 896, 955, 1009, 1063, 1113, 1166, 1225]
        actual_nows = sorted(list(set([min(all_minutes, key=lambda x: abs(x - tn)) for tn in target_nows])))
        
        gap_rows = []
        ref_preds = {}
        masked = {}
        
        # 1. Forecast & Hidden-future
        for now_m in actual_nows:
            now_row = val_data[val_data['minute_of_day'] == now_m].iloc[0]
            now_str = now_row['minute']
            
            for h in range(1, 16):
                m = now_m + h
                # Visible units: forecast
                for u in vis_units:
                    u_m = val_data[(val_data['unit'] == u) & (val_data['minute_of_day'] == m)]
                    if len(u_m) > 0:
                        y_true = u_m.iloc[0]['people']
                        gid = f"{fold_rec}_{u}_{m:04d}"
                        gap_rows.append({
                            'id': gid,
                            'recording': fold_rec,
                            'day': u_m.iloc[0]['day'],
                            'weekday': u_m.iloc[0]['weekday'],
                            'unit': u,
                            'minute': u_m.iloc[0]['minute'],
                            'minute_of_day': m,
                            'people_true': y_true,
                            'kind': 'forecast',
                            'now': now_str,
                            'now_minute_of_day': float(now_m),
                            'horizon': float(h)
                        })
                        masked[(u, m)] = 'forecast'
                        u_now = val_data[(val_data['unit'] == u) & (val_data['minute_of_day'] == now_m)]
                        ref_preds[gid] = u_now.iloc[0]['people'] if len(u_now) > 0 else val_data[val_data['unit'] == u]['people'].median()
                        
                # Hidden units: hidden-future
                for hu in hu_list:
                    hu_m = val_data[(val_data['unit'] == hu) & (val_data['minute_of_day'] == m)]
                    if len(hu_m) > 0:
                        y_true = hu_m.iloc[0]['people']
                        gid = f"{fold_rec}_{hu}_{m:04d}"
                        gap_rows.append({
                            'id': gid,
                            'recording': fold_rec,
                            'day': hu_m.iloc[0]['day'],
                            'weekday': hu_m.iloc[0]['weekday'],
                            'unit': hu,
                            'minute': hu_m.iloc[0]['minute'],
                            'minute_of_day': m,
                            'people_true': y_true,
                            'kind': 'hidden-future',
                            'now': now_str,
                            'now_minute_of_day': float(now_m),
                            'horizon': float(h)
                        })
                        masked[(hu, m)] = 'hidden-future'
                        
            # Blackout for 30 minutes after now
            for h in range(1, 31):
                m = now_m + h
                for u in all_units:
                    if (u, m) not in masked:
                        masked[(u, m)] = 'blackout'
                        
        # 2. Missing gaps
        avail_starts = [m for m in all_minutes if m not in [n + h for n in actual_nows for h in range(0, 35)] and m < max(all_minutes) - 25]
        for _ in range(26):
            u = rng.choice(vis_units)
            start_m = rng.choice(avail_starts)
            gap_len = rng.randint(5, 18)
            end_m = start_m + gap_len - 1
            if not any((u, m) in masked for m in range(start_m - 1, end_m + 2)):
                for m in range(start_m, end_m + 1):
                    u_m = val_data[(val_data['unit'] == u) & (val_data['minute_of_day'] == m)]
                    if len(u_m) > 0:
                        gid = f"{fold_rec}_{u}_{m:04d}"
                        gap_rows.append({
                            'id': gid,
                            'recording': fold_rec,
                            'day': u_m.iloc[0]['day'],
                            'weekday': u_m.iloc[0]['weekday'],
                            'unit': u,
                            'minute': u_m.iloc[0]['minute'],
                            'minute_of_day': m,
                            'people_true': u_m.iloc[0]['people'],
                            'kind': 'missing',
                            'now': np.nan,
                            'now_minute_of_day': np.nan,
                            'horizon': np.nan
                        })
                        masked[(u, m)] = 'missing'
                        
        # 3. Hidden gaps
        for hu in hu_list:
            hu_rows = val_data[val_data['unit'] == hu]
            for idx, row in hu_rows.iterrows():
                m = row['minute_of_day']
                if (hu, m) not in masked and not any(m in range(n + 1, n + 31) for n in actual_nows):
                    gid = f"{fold_rec}_{hu}_{m:04d}"
                    gap_rows.append({
                        'id': gid,
                        'recording': fold_rec,
                        'day': row['day'],
                        'weekday': row['weekday'],
                        'unit': hu,
                        'minute': row['minute'],
                        'minute_of_day': m,
                        'people_true': row['people'],
                        'kind': 'hidden',
                        'now': np.nan,
                        'now_minute_of_day': np.nan,
                        'horizon': np.nan
                    })
                    masked[(hu, m)] = 'hidden'
                    
        val_gaps = pd.DataFrame(gap_rows)
        val_released = val_data[~val_data.apply(lambda r: (r['unit'], r['minute_of_day']) in masked, axis=1)].copy()
        
        # Missing reference: linear interpolation
        for (rec, u), grp in val_gaps[val_gaps['kind'] == 'missing'].groupby(['recording', 'unit']):
            u_all = val_data[val_data['unit'] == u].sort_values('minute_of_day').copy()
            u_all.loc[u_all['minute_of_day'].isin(grp['minute_of_day']), 'people'] = np.nan
            interp = u_all.set_index('minute_of_day')['people'].interpolate(method='linear')
            for idx, r in grp.iterrows():
                ref_preds[r['id']] = interp.loc[r['minute_of_day']]
                
        # Hidden & Hidden-future reference
        piv_tr = train_context.pivot(index=['recording', 'minute_of_day'], columns='unit', values='people').dropna()
        for hu in hu_list:
            scales = {vu: (piv_tr[hu] / piv_tr[vu]).median() for vu in vis_units}
            best_vu = min(vis_units, key=lambda vu: (piv_tr[hu] - scales[vu] * piv_tr[vu]).abs().mean())
            best_scale = scales[best_vu]
            
            # Hidden
            for idx, r in val_gaps[(val_gaps['unit'] == hu) & (val_gaps['kind'] == 'hidden')].iterrows():
                m = r['minute_of_day']
                src_row = val_released[(val_released['unit'] == best_vu) & (val_released['minute_of_day'] == m)]
                ref_preds[r['id']] = round(src_row.iloc[0]['people'] * best_scale, 3) if len(src_row) > 0 else piv_tr[hu].median()
                
            # Hidden-future
            for idx, r in val_gaps[(val_gaps['unit'] == hu) & (val_gaps['kind'] == 'hidden-future')].iterrows():
                now_m = int(r['now_minute_of_day'])
                src_now = val_released[(val_released['unit'] == best_vu) & (val_released['minute_of_day'] == now_m)]
                ref_preds[r['id']] = round(src_now.iloc[0]['people'] * best_scale, 3) if len(src_now) > 0 else piv_tr[hu].median()
                
        val_gaps['people_ref'] = val_gaps['id'].map(ref_preds)
        
        folds[fold_rec] = {
            'train_context': train_context,
            'val_released': val_released,
            'val_gaps': val_gaps,
            'hidden_units': hu_list,
            'visible_units': vis_units
        }
    return folds

def evaluate_predictions_under_protocol(folds, all_preds):
    """
    Computes official split-average metric across Fold 1 and Fold 2.
    """
    fold_scores = {}
    task_ratios_by_fold = {}
    
    for fold_rec, f_data in folds.items():
        gaps = f_data['val_gaps']
        f_res = {}
        r_list = []
        for k in ['forecast', 'hidden', 'hidden-future', 'missing']:
            k_df = gaps[gaps['kind'] == k]
            pred_vals = k_df['id'].map(all_preds)
            if pred_vals.isna().any():
                n_miss = pred_vals.isna().sum()
                raise ValueError(f"Fold {fold_rec} task {k} has {n_miss} missing predictions!")
            mae_model = (pred_vals - k_df['people_true']).abs().mean()
            mae_ref = (k_df['people_ref'] - k_df['people_true']).abs().mean()
            ratio = mae_model / mae_ref if mae_ref > 0 else 1.0
            f_res[k] = {
                'count': len(k_df),
                'mae_model': mae_model,
                'mae_ref': mae_ref,
                'ratio': ratio
            }
            r_list.append(ratio)
        f_res['score'] = float(np.mean(r_list))
        fold_scores[fold_rec] = f_res
        task_ratios_by_fold[fold_rec] = {k: f_res[k]['ratio'] for k in ['forecast', 'hidden', 'hidden-future', 'missing']}
        
    # Aggregate across folds (mean of Fold 1 and Fold 2)
    mean_task_ratios = {}
    for k in ['forecast', 'hidden', 'hidden-future', 'missing']:
        mean_task_ratios[k] = 0.5 * (task_ratios_by_fold['20260911_1100'][k] + task_ratios_by_fold['20260912_1100'][k])
        
    aggregate_score = float(np.mean(list(mean_task_ratios.values())))
    
    summary = {
        'aggregate_score': aggregate_score,
        'mean_task_ratios': mean_task_ratios,
        'fold_1_score': fold_scores['20260911_1100']['score'],
        'fold_2_score': fold_scores['20260912_1100']['score'],
        'details': fold_scores
    }
    return summary
