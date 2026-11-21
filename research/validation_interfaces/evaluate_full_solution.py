"""
evaluate_full_solution.py - End-to-end evaluation of the complete multi-task solution across all validation folds.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from src.validation import create_synthetic_validation_split, compute_competition_metric
from src.exp_task_forecast import extract_forecast_features_for_point, build_training_dataset_for_forecast
from src.exp_task_hidden_features import extract_features

train_df = pd.read_csv(r'C:\Users\USER\Desktop\Madinah-AI-Fresh\data\official\train.csv')

def run_full_pipeline(train_context, val_released, val_gaps, fold_name="Validation"):
    """
    Executes the complete multi-task solution strictly adhering to all context and leakage rules.
    """
    preds = {}
    
    # ----------------------------------------------------
    # TASK 1: MISSING (Linear Interpolation across gap)
    # ----------------------------------------------------
    ms_gaps = val_gaps[val_gaps['kind'] == 'missing']
    for (rec, u), grp in ms_gaps.groupby(['recording', 'unit']):
        u_pts = val_released[(val_released['recording'] == rec) & (val_released['unit'] == u)].sort_values('minute_of_day')
        gap_min = grp['minute_of_day'].min()
        gap_max = grp['minute_of_day'].max()
        
        before = u_pts[u_pts['minute_of_day'] < gap_min]
        after = u_pts[u_pts['minute_of_day'] > gap_max]
        
        if len(before) > 0 and len(after) > 0:
            m_before = before.iloc[-1]['minute_of_day']
            y_before = before.iloc[-1]['people']
            m_after = after.iloc[0]['minute_of_day']
            y_after = after.iloc[0]['people']
            for idx, r in grp.iterrows():
                m = r['minute_of_day']
                alpha = (m - m_before) / (m_after - m_before)
                preds[r['id']] = max(0.0, y_before + alpha * (y_after - y_before))
        else:
            for idx, r in grp.iterrows():
                preds[r['id']] = r['people_ref']
                
    # ----------------------------------------------------
    # TASK 2: HIDDEN (Multi-camera Ridge + LightGBM)
    # ----------------------------------------------------
    hd_gaps = val_gaps[val_gaps['kind'] == 'hidden']
    piv_tr = train_context.pivot(index=['recording', 'minute_of_day'], columns='unit', values='people').dropna()
    piv_val = val_released.pivot(index='minute_of_day', columns='unit', values='people')
    
    hidden_models = {}
    
    for hu in val_gaps[val_gaps['kind'].isin(['hidden', 'hidden-future'])]['unit'].unique():
        vis_cams = [c for c in piv_val.columns if c != hu and c in piv_tr.columns]
        
        X_tr = extract_features(piv_tr, vis_cams, piv_tr.index)
        y_tr = piv_tr[hu].copy()
        
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        
        r_reg = Ridge(alpha=50.0, fit_intercept=True)
        r_reg.fit(X_tr_s, y_tr)
        
        l_reg = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=12,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        l_reg.fit(X_tr, y_tr)
        
        hidden_models[hu] = {
            'vis_cams': vis_cams,
            'scaler': scaler,
            'ridge': r_reg,
            'lgb': l_reg
        }
        
    for idx, r in hd_gaps.iterrows():
        hu = r['unit']
        m = r['minute_of_day']
        hm = hidden_models[hu]
        X_val_all = extract_features(piv_val, hm['vis_cams'], piv_val.index)
        if m in X_val_all.index:
            x_row = X_val_all.loc[[m]]
            x_row_s = hm['scaler'].transform(x_row)
            p_r = hm['ridge'].predict(x_row_s)[0]
            p_l = hm['lgb'].predict(x_row)[0]
            preds[r['id']] = max(0.0, 0.5 * p_r + 0.5 * p_l)
        else:
            preds[r['id']] = r['people_ref']

    # ----------------------------------------------------
    # TASK 3: FORECAST (LightGBM L1 Model)
    # ----------------------------------------------------
    print(f"[{fold_name}] Training Forecast LightGBM L1 Model...")
    X_train_fc, y_train_fc = build_training_dataset_for_forecast(train_context)
    y_res_fc = y_train_fc - X_train_fc['y_now']
    
    fc_lgb = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    fc_lgb.fit(X_train_fc, y_res_fc)
    
    fc_gaps = val_gaps[val_gaps['kind'] == 'forecast']
    for idx, r in fc_gaps.iterrows():
        u = r['unit']
        now_m = int(r['now_minute_of_day'])
        h = int(r['horizon'])
        
        feat = extract_forecast_features_for_point(val_released, u, now_m, h)
        if feat is not None:
            feat_df = pd.DataFrame([feat])[X_train_fc.columns]
            pred_res = fc_lgb.predict(feat_df)[0]
            preds[r['id']] = max(0.0, feat['y_now'] + pred_res)
        else:
            preds[r['id']] = r['people_ref']
            
    # ----------------------------------------------------
    # TASK 4: HIDDEN-FUTURE (Reconstruct + Forecast Blend)
    # ----------------------------------------------------
    hf_gaps = val_gaps[val_gaps['kind'] == 'hidden-future']
    # Build augmented val_released with reconstructed hidden cameras up to now
    val_rel_augmented = val_released.copy()
    for hu, hm in hidden_models.items():
        X_val_all = extract_features(piv_val, hm['vis_cams'], piv_val.index)
        x_val_s = hm['scaler'].transform(X_val_all)
        p_r = hm['ridge'].predict(x_val_s)
        p_l = hm['lgb'].predict(X_val_all)
        pred_hu_all = np.maximum(0.0, 0.5 * p_r + 0.5 * p_l)
        
        recon_df = pd.DataFrame({
            'recording': val_released.iloc[0]['recording'],
            'day': val_released.iloc[0]['day'],
            'weekday': val_released.iloc[0]['weekday'],
            'unit': hu,
            'minute': [f"{int(m//60):02d}:{int(m%60):02d}" for m in piv_val.index],
            'minute_of_day': piv_val.index,
            'people': pred_hu_all
        })
        val_rel_augmented = pd.concat([val_rel_augmented, recon_df], ignore_index=True)
        
    for idx, r in hf_gaps.iterrows():
        hu = r['unit']
        now_m = int(r['now_minute_of_day'])
        h = int(r['horizon'])
        
        # Approach A: Forecast from reconstructed history
        feat_a = extract_forecast_features_for_point(val_rel_augmented, hu, now_m, h)
        if feat_a is not None:
            feat_a_df = pd.DataFrame([feat_a])[X_train_fc.columns]
            pred_res_a = fc_lgb.predict(feat_a_df)[0]
            val_a = max(0.0, feat_a['y_now'] + pred_res_a)
        else:
            val_a = r['people_ref']
            
        # Approach B: Scale from best visible reference camera's forecast
        vis_cams = hidden_models[hu]['vis_cams']
        scales = {vu: (piv_tr[hu] / piv_tr[vu]).median() for vu in vis_cams}
        best_vu = min(vis_cams, key=lambda vu: (piv_tr[hu] - scales[vu] * piv_tr[vu]).abs().mean())
        best_scale = scales[best_vu]
        
        feat_vu = extract_forecast_features_for_point(val_released, best_vu, now_m, h)
        if feat_vu is not None:
            feat_vu_df = pd.DataFrame([feat_vu])[X_train_fc.columns]
            pred_res_vu = fc_lgb.predict(feat_vu_df)[0]
            val_b = max(0.0, (feat_vu['y_now'] + pred_res_vu) * best_scale)
        else:
            val_b = r['people_ref']
            
        preds[r['id']] = 0.5 * val_a + 0.5 * val_b
        
    return preds

def main():
    print("=========================================================")
    print("  EVALUATING COMPLETE INDEPENDENT SOLUTION ON VALIDATION ")
    print("=========================================================")
    
    all_fold_results = {}
    for fold_rec in ['20260911_1100', '20260912_1100']:
        print(f"\n>>> Running Validation Fold: {fold_rec}")
        tr_ctx, val_rel, val_gaps = create_synthetic_validation_split(train_df, fold_rec)
        
        fold_preds = run_full_pipeline(tr_ctx, val_rel, val_gaps, fold_name=fold_rec)
        res = compute_competition_metric(val_gaps, fold_preds)
        all_fold_results[fold_rec] = res
        
        print(f"\n--- Results on Fold {fold_rec} ---")
        print(f"Overall Competition Score: {res['overall_score']:.5f} (Reference: 1.00000)")
        for k in ['forecast', 'hidden', 'hidden-future', 'missing']:
            k_info = res[k]
            print(f"  {k:15s}: count={k_info['count']:4d}, MAE_ref={k_info['mae_ref']:8.3f}, MAE_model={k_info['mae_model']:8.3f}, Ratio={k_info['ratio']:.5f}")
            
    print("\n=========================================================")
    print("  FINAL CROSS-VALIDATION SUMMARY (BOTH FOLDS COMBINED)")
    print("=========================================================")
    avg_score = np.mean([all_fold_results[f]['overall_score'] for f in all_fold_results])
    print(f"AVERAGE CV SCORE: {avg_score:.5f}")
    for k in ['forecast', 'hidden', 'hidden-future', 'missing']:
        avg_r = np.mean([all_fold_results[f][k]['ratio'] for f in all_fold_results])
        print(f"  {k:15s} Mean Ratio: {avg_r:.5f}")

if __name__ == '__main__':
    main()
