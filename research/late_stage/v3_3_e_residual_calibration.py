#!/usr/bin/env python3
"""V3.3: E-camera pseudo-gap cross-date residual calibration (RESEARCH ONLY).

Runs the unchanged, verified selected champion callback on predeclared, legal
pseudo Missing masks. Trains one frozen, ridge-regularized, clipped correction
using OTHER-date pseudo labels, then tests on held-out-date pseudo labels.
It NEVER writes any Kaggle submission or modifies selected predictions/models.

Requirements: Python 3.10+, numpy, pandas, and v3_2_missing_loro.py in the
same directory. For actual results, the project must contain:
    verified_605700/cloud_research/selected/candidate.py

Run:
    python v3_3_e_residual_calibration.py --root .
Smoke-test (NOT an ML result):
    python v3_3_e_residual_calibration.py --root . --smoke-linear --skip-frozen
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v3_2_missing_loro as v2

RECORDINGS = ("20260911_1100", "20260912_1100")
CAMERA = "cam_E"
# Obtained from user's successful V3.2 run. These are identity gates, not model params.
EXPECTED_SOURCE_SHA = "836f54e615089b67099d7b6d2d685765bc999b96bd14f2599982c095fbffa930"
EXPECTED_TRAIN_SHA = "3210a0b33212a60a888977848ec09f5f89bc0954bfd3b88e61fb9d840537f6f6"
EXPECTED_MISSING_DIAG_SHA = "8b2a4455db2dacb93ba9fc00914b2f97bc69a9dc607ba2a8712141c1ebb55379"
EXPECTED_TASK_COMP_SHA = "2b8a968d63641f49b5ebac27a274ab1f3006fd8d9ba0bb94e62a6edba82e7409"
CANDIDATE_STRATA = (("fast", 8), ("fast", 12), ("control", 7))
REGION_COUNT = 4
GUARD_MIN = 7
# One fixed, predeclared family. These are NOT chosen from frozen error stats.
RIDGE_LAMBDA = 2.0
RIDGE_SHRINK = 0.50
MAX_CORRECTION = 3.0
TARGET_SCORE = 0.600
MIN_FAST_BLOCKS_PER_DATE = 4
SEED = 33117


def sha(path: Path) -> str:
    return v2.digest(path)


def safe_float(x: Any) -> float | None:
    z = float(x)
    return z if np.isfinite(z) else None


def stable(rec: str, bucket: str, block: v2.Block) -> str:
    return hashlib.sha256(f"{SEED}|{rec}|{bucket}|{block.camera}|{block.start}|{block.length}".encode()).hexdigest()


def load_exclusion_keys(path: Path) -> tuple[set[tuple[str, str, int]], dict]:
    """Only read target IDs and timestamps; NO frozen gold/selected scores."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen ID mask manifest: {path}")
    if sha(path) != EXPECTED_MISSING_DIAG_SHA:
        raise RuntimeError("Frozen mask diagnostic SHA changed. Do not reuse old mask exclusions.")
    df = pd.read_csv(path, usecols=["id", "recording", "camera", "minute_of_day"])
    if len(df) != 358 or df.id.duplicated().any():
        raise ValueError("Expected 358 unique frozen Missing IDs")
    keys = set(zip(df.recording.astype(str), df.camera.astype(str), df.minute_of_day.astype(int)))
    return keys, {"excluded_mask_rows": len(keys), "id_manifest_sha256": sha(path)}


def choose_blocks(full: pd.DataFrame, rec: str,
                  forbidden: set[tuple[str, str, int]]) -> tuple[list[v2.Block], list[dict]]:
    """Use only endpoint slope, timestamps, and frozen ID exclusions.

    No masked interior values are accessed for the choice. A deterministic
    hash chooses windows; balancing by clock quarter and pattern limits skew.
    """
    pool = v2.enumerate_candidate_blocks(full, rec)
    chosen: list[v2.Block] = []
    descriptions: list[dict] = []
    lo, hi = int(full.minute_of_day.min()), int(full.minute_of_day.max()) + 1
    forbidden_e = {m for r, c, m in forbidden if r == rec and c == CAMERA}
    for region in range(REGION_COUNT):
        qlo = lo + region * (hi-lo) // REGION_COUNT
        qhi = lo + (region+1)*(hi-lo) // REGION_COUNT
        for pattern, length in CANDIDATE_STRATA:
            possible = pool.get((CAMERA, pattern, length), [])
            possible = [b for b in possible if qlo <= b.start < qhi]
            if pattern == "control":
                possible = [b for b in possible if b.slope >= -0.5]
            # Frozen target cells must be outside the whole synthetic window's
            # guard region; these labels may not be part of pseudo targets.
            possible = [b for b in possible if not any(m in forbidden_e for m in
                         range(b.start - GUARD_MIN, b.end + GUARD_MIN + 1))]
            for b in sorted(possible, key=lambda b: stable(rec, f"{region}|{pattern}|{length}", b)):
                if any(not (b.end + GUARD_MIN < old.start or
                            b.start - GUARD_MIN > old.end) for old in chosen):
                    continue
                chosen.append(b)
                descriptions.append({"recording": rec, "camera": CAMERA,
                                     "block_start": b.start, "block_end": b.end,
                                     "block_length": b.length,
                                     "endpoint_slope": b.slope, "flow_pattern": pattern,
                                     "region": region})
                break
    chosen.sort(key=lambda b: b.start)
    if sum(b.slope <= -2 for b in chosen) < MIN_FAST_BLOCKS_PER_DATE:
        raise RuntimeError(f"Too few E fast falling pseudo gaps for {rec}")
    if not chosen:
        raise RuntimeError(f"No E pseudo gaps for {rec}")
    # Each panel has 2 fast / 1 control per alternating regions; panels are disjoint.
    return chosen, descriptions


def panels_from_blocks(blocks: list[v2.Block]) -> dict[str, list[v2.Block]]:
    # Deterministic, time-distributed panel assignment. Last panel can be smaller.
    panels: dict[str, list[v2.Block]] = {"P0": [], "P1": []}
    for idx, b in enumerate(sorted(blocks, key=lambda a: a.start)):
        panels[f"P{idx % 2}"].append(b)
    return panels


def pseudo_features(data: pd.DataFrame, rec: str, blocks: list[v2.Block],
                    forbidden: set[tuple[str, str, int]], pred_fn) -> pd.DataFrame:
    full = data.loc[data.recording.eq(rec)].copy()
    train_ctx = data.loc[data.recording.ne(rec)].copy()
    released, targets, gold, baseline = v2.make_pseudogap_inputs(full, blocks)
    # Remove all 358 known frozen Missing cells for this date from the same-
    # recording released context. These cells are NEVER treated as observations
    # for pseudo gap inference (avoids in-fold frozen-label leakage).
    exclude_this_recording = {(c, m) for r, c, m in forbidden if r == rec}
    rm = np.array([(str(c), int(m)) in exclude_this_recording for c,m in
                   zip(released.unit, released.minute_of_day)], dtype=bool)
    released = released.loc[~rm].copy()
    if any((rec, str(c), int(m)) in forbidden for c,m in
           zip(targets.unit, targets.minute_of_day)):
        raise AssertionError("Pseudo targets overlap original frozen Missing labels")
    # Recompute linear reference from truly released counts; the E pseudo
    # endpoints are buffered away from original frozen E gaps.
    obs = released.loc[released.unit.eq(CAMERA)].set_index('minute_of_day').people.sort_index()
    grid = obs.reindex(sorted(set(obs.index) | set(targets.minute_of_day.astype(int))))
    interpolated = grid.interpolate(method='linear', limit_area='inside')
    baseline = pd.DataFrame({'id': targets.id, 'people': targets.minute_of_day.map(interpolated)})
    if baseline.people.isna().any():
        raise RuntimeError(f"{rec}: legal baseline has NaNs")
    if any((rec, str(c), int(m)) in forbidden for c,m in zip(released.unit,released.minute_of_day)):
        raise AssertionError("Frozen Missing labels were not fully removed from released context")
    # Spy check. Only metadata is passed to the model; gold joined AFTER inference.
    assert not {'people','truth','people_true','people_ref'}.intersection(targets.columns)
    print(f"{rec}: {len(blocks)} blocks, {len(targets)} E pseudo targets; invoking champion...", flush=True)
    preds = v2.get_predictions(pred_fn, train_ctx, released, targets, baseline)
    feats = v2.build_features(released, targets, baseline, preds)
    scored = v2.add_truth_after_prediction(feats, gold)
    mapping = {(b.start, b.length): f"{rec}|{b.start}|{b.length}" for b in blocks}
    scored['block_id'] = [mapping[(int(r.block_start), int(r.block_length))] for r in scored.itertuples()]
    scored['panel'] = None
    assert len(scored) == len(targets) and (scored.camera == CAMERA).all()
    return scored


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Strict label-free feature allowlist. Beta learns ONLY from other-date gold."""
    x = df.loc[:, v2.FEATURE_COLUMNS].copy()
    t = x.gap_position_fraction.to_numpy(float)
    L = x.block_length.to_numpy(float) + 1.0
    slope = x.endpoint_slope_per_min.to_numpy(float)
    left = x.left_count.to_numpy(float)
    right = x.right_count.to_numpy(float)
    before = x.local_prev_mean.to_numpy(float)
    after = x.local_next_mean.to_numpy(float)
    pred = x.selected_prediction.to_numpy(float)
    ref = x.reference_prediction.to_numpy(float)
    shape = 4.0 * t * (1.0 - t)
    mleft = (left - before) / 2.0
    mright = (after - right) / 2.0
    curvature = L*((mleft-slope)*t*(1-t)**2 - (mright-slope)*t*t*(1-t))
    # All numeric scales are fixed before any pseudo label scoring.
    z = np.column_stack([
        shape,
        shape * np.clip(slope / 12.0, -4.0, 0.0),
        shape * np.clip(curvature / 25.0, -4.0, 4.0),
        shape * np.clip((ref - pred) / 20.0, -3.0, 3.0),
        shape * (2.0*t-1.0),
        shape * np.clip((mleft-mright) / 15.0, -3.0, 3.0),
    ])
    gate = (x.camera.eq(CAMERA).to_numpy() & (slope <= -2.0)
            & (x.block_length.to_numpy(float) >= 5)
            & (x.block_length.to_numpy(float) <= 15)
            & np.isfinite(z).all(axis=1))
    if not np.isfinite(z).all():
        raise AssertionError("Invalid feature matrix")
    return z, gate


def fit_beta(training: pd.DataFrame) -> np.ndarray:
    z, gate = feature_matrix(training)
    if int(gate.sum()) < 20:
        raise RuntimeError(f"Only {gate.sum()} E eligible training examples; cannot fit")
    r = (training.truth - training.selected_prediction).to_numpy(float)
    y = np.clip(r[gate], -15.0, 15.0)
    blocks = training.block_id.astype(str).to_numpy()[gate]
    counts = pd.Series(blocks).value_counts().to_dict()
    # Every gap contributes equal *total* weight, instead of long gaps
    # dominating the regression just because they contain more minutes.
    weights = np.array([1.0/counts[b] for b in blocks], dtype=float)
    weights /= weights.sum()
    xx = z[gate]
    xtwx = (xx * weights[:,None]).T @ xx
    xtwy = (xx * weights[:,None]).T @ y
    coeff = np.linalg.solve(xtwx + RIDGE_LAMBDA*np.eye(xx.shape[1]), xtwy)
    if not np.isfinite(coeff).all():
        raise AssertionError("Non-finite ridge coefficients")
    return coeff


def adjust(features: pd.DataFrame, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z, gate = feature_matrix(features)
    p = features.selected_prediction.to_numpy(float)
    q = p.copy()
    delta = np.clip(RIDGE_SHRINK * z @ beta, -MAX_CORRECTION, MAX_CORRECTION)
    q[gate] = np.maximum(0.,p[gate]+delta[gate])
    if not np.isfinite(q).all() or not np.array_equal(q[~gate],p[~gate]):
        raise AssertionError("Correction changed noneligible predictions or produced invalid values")
    return q,gate


def score(scored: pd.DataFrame, q: np.ndarray) -> dict[str,float]:
    y=scored.truth.to_numpy(float)
    p=scored.selected_prediction.to_numpy(float)
    ref=scored.reference_prediction.to_numpy(float)
    mae_ref=float(np.abs(ref-y).mean())
    if mae_ref <=0:raise AssertionError('Reference denominator must be positive')
    mae_p=float(np.abs(p-y).mean())
    mae_q=float(np.abs(q-y).mean())
    return {"n":len(scored),"ref_mae":mae_ref,"champion_mae":mae_p,
            "candidate_mae":mae_q,"champion_ratio":mae_p/mae_ref,
            "candidate_ratio":mae_q/mae_ref,"delta_ratio":(mae_q-mae_p)/mae_ref,
            "changed":int((q!=p).sum())}


def group_stats(rec: str, df: pd.DataFrame, q: np.ndarray) -> list[dict]:
    d=df.copy()
    d['candidate']=q
    d['err_old']=abs(d.selected_prediction-d.truth)
    d['err_new']=abs(d.candidate-d.truth)
    d['regime']=np.where(d.endpoint_slope_per_min<=-2,'fast_falling','control')
    d['gap_duration']=d.block_length.map(lambda x:'short_5_9' if x<=9 else 'long_10_15')
    rows=[]
    den=float(abs(d.reference_prediction-d.truth).mean()); n=len(d)
    for col in ['regime','gap_duration','panel','block_start']:
        for key,sub in d.groupby(col,dropna=False):
            rows.append({'recording':rec,'dimension':col,'group':str(key),'rows':len(sub),
                'mae_old':float(sub.err_old.mean()),'mae_new':float(sub.err_new.mean()),
                'delta_ratio_contribution':float((sub.err_new-sub.err_old).sum()/(n*den)),
                'changed':int((sub.candidate!=sub.selected_prediction).sum())})
    return rows


def frozen_preview(root:Path, beta:np.ndarray) -> dict:
    inp=root/'reports'/'breakthrough_v2'
    diagnostic=inp/'missing_target_diagnostics.csv'
    tasks=inp/'per_task_score_comparisons.csv'
    if not (diagnostic.is_file() and tasks.is_file()):
        return {'status':'SKIPPED_FROZEN_DIAGNOSTICS_MISSING'}
    if sha(diagnostic)!=EXPECTED_MISSING_DIAG_SHA or sha(tasks)!=EXPECTED_TASK_COMP_SHA:
        raise RuntimeError("Frozen diagnostics/score-table SHA mismatch")
    d=pd.read_csv(diagnostic)
    comp=pd.read_csv(tasks)
    cr=comp.loc[comp.model.eq('verified_champion')]
    if len(cr)!=8 or cr.duplicated(['recording','task']).any():
        raise AssertionError('Champion eight task contributions missing')
    base=float(cr.ratio.sum()/8)
    if abs(base-v2.CHAMPION_SCORE)>1e-12:
        raise RuntimeError('The original frozen champion score does not reconstruct exactly')
    features=d.loc[:,v2.FEATURE_COLUMNS].copy()
    q,gate=adjust(features,beta)
    # Label-poison test: add wrong truths, ensure predictions stay bitwise stable.
    poison=features.copy()
    poison['truth']=np.linspace(9e8,-9e8,len(poison))
    poison['abs_error']=np.arange(len(poison))
    q_poison,_=adjust(poison,beta)
    if not np.array_equal(q_poison,q):
        raise AssertionError('Frozen label-invariance failed')
    p=d.selected_prediction.to_numpy(float)
    gold=d.truth.to_numpy(float)
    nrec=d.groupby('recording').size().to_dict()
    factors=np.array([1/(8*nrec[r]*ref) for r,ref in zip(d.recording,d.task_reference_mae)],float)
    existing=abs(p-gold)*factors
    if np.max(abs(existing-d.normalized_full_score_contribution.to_numpy(float)))>1e-10:
        raise AssertionError('Frozen champion contributions mismatch')
    deltas=(abs(q-gold)-abs(p-gold))*factors
    return {
        'status':'RETROSPECTIVE_REUSED_FROZEN_NOT_INDEPENDENT_NOT_PROMOTABLE',
        'score':base+float(deltas.sum()),'delta':float(deltas.sum()),
        'reaches_0_600_retrospective_only':bool(base+deltas.sum()<TARGET_SCORE),
        'gated_E':int(gate.sum()),'changed_E':int((q!=p).sum()),
        'fold_deltas':{rec:float(deltas[d.recording.eq(rec)].sum()) for rec in RECORDINGS},
        'camera_E_full_score_delta':float(deltas.sum()),
        'champion_reconstructed':base
    }


def run(root:Path, output:Path|None=None, smoke_linear:bool=False, skip_frozen:bool=False) -> dict:
    root=root.resolve()
    output=(output or (root/'reports'/('v3_3_smoke' if smoke_linear else 'v3_3'))).resolve()
    v2.assert_output_dir_safe(root,output)
    train,train_path=v2.read_train(root)
    if not smoke_linear and sha(train_path)!=EXPECTED_TRAIN_SHA:
        raise RuntimeError('Official train.csv sha256 differs from verified V3.2 run. Re-audit identity.')
    pred_fn,identity=v2.load_champion(root,smoke_linear)
    if not smoke_linear and identity['sha256']!=EXPECTED_SOURCE_SHA:
        raise RuntimeError('Verified selected candidate.py changed since V3.2. Stop before testing.')
    manifest=root/'reports'/'breakthrough_v2'/'missing_target_diagnostics.csv'
    forbidden,maskmeta=load_exclusion_keys(manifest)
    print('V3.3: E-camera research; NO submission or champion modification',flush=True)
    print('CHAMPION:',identity['kind'],flush=True)
    frames={};cover=[]
    for rec in RECORDINGS:
        full=train.loc[train.recording.eq(rec)].copy()
        blocks, descriptions=choose_blocks(full,rec,forbidden)
        panels=panels_from_blocks(blocks)
        cover.extend(descriptions)
        parts=[]
        for panel,blist in panels.items():
            if not blist:continue
            f=pseudo_features(train,rec,blist,forbidden,pred_fn)
            f['panel']=panel
            parts.append(f)
        df=pd.concat(parts,ignore_index=True)
        if df.id.duplicated().any():
            raise AssertionError('Pseudo panels contain overlapping target IDs')
        frames[rec]=df
    # Fixed, predeclared leave-one-recording-out test on two days.
    outer=[]; groups=[]; betas={}
    for holdout in RECORDINGS:
        training_rec=next(x for x in RECORDINGS if x!=holdout)
        beta=fit_beta(frames[training_rec])
        eval_df=frames[holdout]
        q,gate=adjust(eval_df,beta)
        decoy=eval_df.copy()
        decoy['truth']=1e10 + np.arange(len(decoy), dtype=float)
        if not np.array_equal(q, adjust(decoy,beta)[0]):
            raise AssertionError('Held-out prediction changed when truth labels were poisoned')
        res=score(eval_df,q)
        betas[holdout]=beta
        outer.append({'holdout':holdout,'training':training_rec,'gated':int(gate.sum()),
                      'alpha_family':'SHRUNK_WEIGHTED_E_RIDGE_FIXED',**res})
        groups.extend(group_stats(holdout,eval_df,q))
    # The outer recording test is evidence only, never final-selection truth.
    successful=all(v['delta_ratio']<-0.001 and v['changed']>0 for v in outer)
    mean_delta=float(np.mean([v['delta_ratio'] for v in outer]))
    max_harm=float(max(v['delta_ratio'] for v in outer))
    test_supported=bool(successful and mean_delta<=-0.005 and max_harm<0)
    all_scored=pd.concat([frames[r] for r in RECORDINGS],ignore_index=True)
    final_beta=fit_beta(all_scored)
    conf={
        'kind':'V3_3_E_RESIDUAL_RIDGE_RESEARCH_NOT_DEPLOYED',
        'features':['temporal_shape','endpoint_slope','curvature','reference_minus_champion',
                    'gap_skew','local_momentum_disagreement'],
        'coefficients':[float(x) for x in final_beta],
        'ridge_lambda':RIDGE_LAMBDA,'shrink':RIDGE_SHRINK,'cap':MAX_CORRECTION,
        'gating':'camera==cam_E and endpoint_slope<=-2 and length 5-15',
        'training_source':'two historically reused labeled recordings, pseudo gaps excluding frozen Missing IDs',
        'passes_two_date_pseudo_guardrail':test_supported,
        'never_promote_automatically':True,
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/'v3_3_frozen_config.json').write_text(json.dumps(conf,indent=2),encoding='utf-8')
    conf_sha=sha(output/'v3_3_frozen_config.json')
    # ONLY NOW read frozen true values, and only for a retrospective preview.
    preview=({'status':'SKIPPED_BY_USER_OR_SMOKE'} if skip_frozen or smoke_linear
             else frozen_preview(root,final_beta))
    # A source or dataset modifying callback is NOT acceptable.
    if not smoke_linear and (sha(train_path)!=EXPECTED_TRAIN_SHA or
        sha(Path(identity['path']))!=EXPECTED_SOURCE_SHA):
        raise RuntimeError('Source/train modified during the research run')
    result={
        'experiment':'V3_3_CAMERA_E_PSEUDOGAP_RESIDUAL_CALIBRATION',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'status':'SMOKE_STUB_NOT_SCIENTIFIC' if smoke_linear else 'RESEARCH_ONLY_NO_SUBMISSION',
        'champion_reference_score':v2.CHAMPION_SCORE,
        'kaggle_public_best_user_report':0.729,
        'target_score':TARGET_SCORE,
        'champion':identity,
        'train_sha256':sha(train_path),
        'frozen_mask_manifest':maskmeta,
        'protocol':{'seed':SEED,'strata':CANDIDATE_STRATA,'quarters':REGION_COUNT,
                    'mask_guard_minutes':GUARD_MIN,
                    'outer_protocol':'recording-disjoint E pseudo gaps, exactly two dates, 2 panels per date',
                    'frozen_masks_excluded_from_released':True,
                    'frozen_labels_used_in_fit':False,
                    'selection_after_original_frozen_error_analysis':True,
                    'old_records_cam_E_support':0},
        'pseudo_target_count_by_date':{r:len(frames[r]) for r in RECORDINGS},
        'pseudo_fast_count_by_date':{r:int((frames[r].endpoint_slope_per_min<=-2).sum()) for r in RECORDINGS},
        'outer_holdout_results':outer,
        'guardrail':{'pass':test_supported,'outer_wins':int(sum(x['delta_ratio']<-1e-8 for x in outer)),
                     'mean_missing_ratio_delta':mean_delta,'max_harm_delta':max_harm},
        'frozen_config_sha256':conf_sha,
        'frozen_preview':preview,
        'decision':'NO_AUTOMATIC_PROMOTION_UNTOUCHED_E_RECORDINGS_UNAVAILABLE',
        'limitations':[
            'Both dates were already used in prior research; E pseudo mask LOSO is not independent final validation.',
            'Frozen original Missing labels were excluded from pseudo-target windows and released features, but other official frozen task masks are not fully reproduced.',
            'E exists only in the Friday/Saturday recordings: two recording-level folds are insufficient for reliable transfer.',
            'The frozen preview is retrospective and cannot alone prove a better Kaggle score.',
            'This script contains no submission generation, integration, or Kaggle API code.',
        ]}
    (output/'v3_3_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(outer).to_csv(output/'v3_3_outer_dates.csv',index=False)
    pd.DataFrame(cover).to_csv(output/'v3_3_pseudogap_coverage.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'v3_3_group_results.csv',index=False)
    lines=[
        '# V3.3 E-camera residual calibration — research-only decision','',
        '**Champion preserved at 0.6057000521183546. No submission generated.**','',
        f"Champion source: `{identity['kind']}`. The two-date pseudo-gap guardrail passed: **{test_supported}**.",
        f"Mean E pseudo Missing ratio delta: **{mean_delta:+.6f}** (negative is improvement).",'',
        '| Held-out date | Pseudo rows | Gated | Changed | Champion MAE | Candidate MAE | Delta ratio |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for r in outer:
        lines.append(f"| {r['holdout']} | {r['n']} | {r['gated']} | {r['changed']} | {r['champion_mae']:.5f} | {r['candidate_mae']:.5f} | {r['delta_ratio']:+.6f} |")
    lines+=['','## Frozen replay (development only)','',
            f"- Status: `{preview.get('status')}`",
            f"- Estimated full local score: `{preview.get('score', 'not run')}`",
            f"- Frozen score delta: `{preview.get('delta', 'not run')}`",'',
            '**No promotion**: camera E has only two historically reused labeled recording dates; this test is neither untouched nor official.',
            'The historical original frozen-task masks are only partially represented by the Missing ID exclusions.',
            'Never treat the retrospective score as an independently verified improvement or as a Kaggle result.',
    ]
    (output/'v3_3_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Outer guardrail:',test_supported,'mean delta:',mean_delta,flush=True)
    print('Frozen preview:',preview,flush=True)
    print('Output:',output,flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--smoke-linear',action='store_true',help='Mock linear callback; for code QA only')
    ap.add_argument('--skip-frozen',action='store_true')
    args=ap.parse_args()
    run(args.root,args.output,args.smoke_linear,args.skip_frozen)

if __name__=='__main__':
    main()
