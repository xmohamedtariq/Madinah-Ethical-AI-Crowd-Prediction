#!/usr/bin/env python3
"""V3.4: causal shadow-backtest residual calibration for the Forecast task.

RESEARCH ONLY. Requires the exact verified champion predict_fold callback. It:
- never edits the champion, predicts only pseudo tasks, or generates submissions;
- uses only historical training recordings and released values <= each forecast origin;
- estimates recent champion error from an earlier, legally simulated forecast;
- freezes a correction strength using older recordings BEFORE evaluating 9/11-9/12;
- refuses to claim an official or an independent competition score.

Run from project root:
  python v3_4_shadow_forecast.py --root .
Smoke test only, NOT a machine learning result:
  python v3_4_shadow_forecast.py --root . --smoke-persistence

Requires v3_2_missing_loro.py in the same directory, plus numpy and pandas.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd

import v3_2_missing_loro as core

EXPECTED_TRAIN_SHA = '3210a0b33212a60a888977848ec09f5f89bc0954bfd3b88e61fb9d840537f6f6'
EXPECTED_CHAMP_SHA = '836f54e615089b67099d7b6d2d685765bc999b96bd14f2599982c095fbffa930'
CHAMPION_SCORE = 0.6057000521183546
TARGET_SCORE = 0.600
# Earliest recordings have zero or one available prior recording. Reserve those for history.
EARLY_TRAIN = ('20260905_1124', '20260905_1900')
LATER_PSEUDO = ('20260911_1100', '20260912_1100')
HIDDEN_BY_FOLD = {
    '20260911_1100': {'cam_B', 'cam_G'},
    '20260912_1100': {'cam_D'},
}
# Camera E is absent from early training dates. It is not an eligible correction target.
COMMON_CAMERAS = {'cam_A','cam_B','cam_C','cam_D','cam_F','cam_G','cam_H'}
ALPHA_GRID = (0.0, 0.10, 0.20, 0.35)
SHADOW_LOOKBACK = 12
SHADOW_HORIZONS = (3, 6, 9, 12)
MAIN_HORIZONS = tuple(range(1,16))
PER_DATE_ANCHORS = 4
MIN_ELIGIBLE_CAMERAS = 3
CORRECTION_CAP_PEOPLE = 5.0
MIN_OLD_RECORDINGS = 2
SELECT_MEAN_RATIO_GAIN = 0.004
SELECT_MAX_OLD_RATIO_HARM = 0.003
SEED = 34026


def hms(minute: int) -> str:
    return f'{minute//60:02d}:{minute%60:02d}'


def stable_key(s: str) -> str:
    return hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest()


def rec_is_before(a: str, b: str) -> bool:
    """Recording names are ISO-like YYYYMMDD_HHMM and lexically ordered."""
    return a < b


def load_inputs(root: Path, smoke: bool):
    data, data_path = core.read_train(root)
    if not smoke and core.digest(data_path) != EXPECTED_TRAIN_SHA:
        raise RuntimeError('Train.csv does not match the actual V3.3 run. Stop.')
    model, identity = core.load_champion(root, smoke)
    if not smoke and identity.get('sha256') != EXPECTED_CHAMP_SHA:
        raise RuntimeError('Selected champion source SHA differs from V3.3. Stop.')
    return data, data_path, model, identity


def eligible_origins(full: pd.DataFrame, rec: str) -> tuple[list[tuple[int,list[str]]], dict]:
    hidden = HIDDEN_BY_FOLD.get(rec,set())
    allowed = sorted(COMMON_CAMERAS - hidden)
    by_cam = {c: set(part.minute_of_day.to_numpy(int)) for c,part in full.groupby('unit') if c in allowed}
    if not by_cam:
        raise RuntimeError(f'{rec}: no eligible visible cameras')
    lo,hi = int(full.minute_of_day.min()),int(full.minute_of_day.max())
    all_candidates = []
    # Conditions concern time availability only; we do NOT inspect any future count VALUES.
    for now in range(lo+30,hi-15):
        must = set([now,now-SHADOW_LOOKBACK,now-9,now-6,now-3] + [now+h for h in MAIN_HORIZONS])
        cams = [c for c,T in by_cam.items() if must.issubset(T)]
        if len(cams)>=MIN_ELIGIBLE_CAMERAS:
            all_candidates.append((now,cams))
    if not all_candidates:
        raise RuntimeError(f'{rec}: no contiguous forecast origins')
    # Fixed quantile anchors with 15-minute minimum separation; no labels or errors used.
    ix = np.rint(np.linspace(0,len(all_candidates)-1,PER_DATE_ANCHORS+2)[1:-1]).astype(int)
    picks=[]
    for i in ix:
        n,c = all_candidates[int(i)]
        if all(abs(n - earlier)>15 for earlier,_ in picks):
            picks.append((n,c))
    if len(picks)<2:
        raise RuntimeError(f'{rec}: fewer than two separated origins')
    meta={'total_eligible_origins':len(all_candidates), 'selected':[
        {'now':int(n),'cameras':list(c)} for n,c in picks],
        'excluded_hidden_cameras':sorted(hidden), 'never_correct_camera_E':True}
    return picks,meta


def make_targets(full: pd.DataFrame, now: int, cams: list[str],
                 horizons: tuple[int,...]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Make target metadata from time and camera identities only. Never put labels in it.
    index = full.set_index(['unit','minute_of_day'])
    target_rows=[]; base_rows=[]; gold_rows=[]
    for c in cams:
        now_key=(c,now)
        if now_key not in index.index:
            raise RuntimeError(f'Missing observed start: {now_key}')
        base=float(index.loc[now_key,'people'])
        for h in horizons:
            minute=now+h
            row=index.loc[(c,minute)]
            tid=f'{full.recording.iloc[0]}_{c}_{minute:04d}'
            target_rows.append({'id':tid,'recording':str(full.recording.iloc[0]),
                'day':str(row.day),'weekday':str(row.weekday),'unit':c,
                'minute':hms(minute),'minute_of_day':int(minute),'kind':'forecast',
                'now':hms(now),'now_minute_of_day':float(now),'horizon':float(h)})
            base_rows.append({'id':tid,'people':base})
            gold_rows.append({'id':tid,'truth':float(row.people),
                              'camera':c,'horizon':h,'now':now})
    return pd.DataFrame(target_rows),pd.DataFrame(base_rows),pd.DataFrame(gold_rows)


def call_at_origin(data: pd.DataFrame, rec: str, origin: int, cams: list[str],
                   horizons: tuple[int,...], model: Callable) -> pd.DataFrame:
    full = data.loc[data.recording.eq(rec)].copy()
    train = data.loc[data.recording.map(lambda r:rec_is_before(str(r),rec))].copy()
    if train.recording.nunique() < 1:
        raise RuntimeError(f'{rec}: no earlier training recordings')
    if not (train.recording.astype(str) < rec).all():
        raise AssertionError('Temporal train split invalid')
    # Hide the designated whole-recording hidden cameras in later pseudo folds.
    hidden = HIDDEN_BY_FOLD.get(rec,set())
    released = full.loc[(full.minute_of_day<=origin) & (~full.unit.isin(hidden))].copy()
    if len(released)==0 or released.minute_of_day.max()>origin:
        raise AssertionError('Future rows in released context')
    targets, baseline, gold = make_targets(full,origin,cams,horizons)
    for f in ('people','truth','people_true','people_ref'):
        if f in targets.columns:
            raise AssertionError(f'Label leak in metadata targets: {f}')
    if (targets.minute_of_day <= origin).any():
        raise AssertionError('Forecast target must be strictly after origin')
    if set(targets.id)!=set(baseline.id):
        raise AssertionError('Invalid baseline-target identity')
    pred = core.get_predictions(model,train,released,targets,baseline)
    # Gold is looked up and joined ONLY after all predictions are complete.
    scored=targets[['id','unit','horizon','now_minute_of_day']].merge(
        baseline.rename(columns={'people':'reference_prediction'}),on='id',validate='one_to_one').merge(
        pred,on='id',validate='one_to_one').merge(
        gold[['id','truth']],on='id',validate='one_to_one')
    scored['recording']=rec
    if not np.isfinite(scored[['reference_prediction','selected_prediction','truth']].to_numpy(float)).all():
        raise AssertionError('Invalid scored values')
    return scored


def future_poison_audit(data: pd.DataFrame, model: Callable, rec: str,
                        now: int, cams: list[str]) -> dict:
    """Mutate ALL withheld rows after now and prove current-origin predictions unchanged."""
    before = call_at_origin(data, rec, now, cams, (1, 5, 10, 15), model)
    poisoned = data.copy()
    is_future = poisoned.recording.eq(rec) & poisoned.minute_of_day.gt(now)
    poisoned.loc[is_future, 'people'] = (
        1e7 + np.arange(int(is_future.sum()), dtype=float))
    after = call_at_origin(poisoned, rec, now, cams, (1, 5, 10, 15), model)
    if list(before.id) != list(after.id):
        raise AssertionError('Causal test reordered or dropped target ids')
    gap = float(np.max(abs(before.selected_prediction.to_numpy(float)
                               -after.selected_prediction.to_numpy(float))))
    if gap > 1e-9:
        raise AssertionError(f'Forbidden use of future values: max delta {gap:.12f}')
    return {'recording':rec, 'now':now, 'future_values_mutated':int(is_future.sum()),
            'max_pred_change':gap, 'passed':True}


def shadow_bias(shadow: pd.DataFrame, now: int) -> dict[str,float]:
    if not (shadow.now_minute_of_day==now-SHADOW_LOOKBACK).all():
        raise AssertionError('Wrong shadow origin')
    weights={3.0:1.0,6.0:1.0,9.0:1.5,12.0:2.0}
    biases={}
    for c,part in shadow.groupby('unit'):
        resid=np.clip(part.truth.to_numpy(float)-part.selected_prediction.to_numpy(float),-20,20)
        w=np.array([weights[float(h)] for h in part.horizon],float)
        biases[c]=float(np.average(resid,weights=w))
    return biases


def build_pseudo(data: pd.DataFrame, model: Callable, rec: str):
    full=data.loc[data.recording.eq(rec)].copy()
    picks,meta=eligible_origins(full,rec)
    pieces=[]
    for i,(now,cams) in enumerate(picks, start=1):
        print(f'  {rec} origin {i}/{len(picks)} {hms(now)} {len(cams)} cams: main + shadow',flush=True)
        main=call_at_origin(data,rec,now,cams,MAIN_HORIZONS,model)
        sh=call_at_origin(data,rec,now-SHADOW_LOOKBACK,cams,SHADOW_HORIZONS,model)
        b=shadow_bias(sh,now)
        main['shadow_bias']=[b[c] for c in main.unit]
        main['time_now']=now
        main['anchor_id']=f'{rec}|{now}'
        # Safely apply no truth-based selection. This truth-invariance test is exact.
        shadow_copy=main.copy()
        shadow_copy['truth']=1e10+np.arange(len(shadow_copy),dtype=float)
        if not np.array_equal(correction(main,0.2),correction(shadow_copy,0.2)):
            raise AssertionError('Candidate uses true future labels')
        pieces.append(main)
    results=pd.concat(pieces,ignore_index=True)
    return results,meta


def correction(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    p=frame.selected_prediction.to_numpy(float)
    h=frame.horizon.to_numpy(float)
    bias=frame.shadow_bias.to_numpy(float)
    # Recent forecast residual should fade over longer horizons.
    applied=np.clip(alpha * bias * np.exp(-h/18.0),
                    -CORRECTION_CAP_PEOPLE,CORRECTION_CAP_PEOPLE)
    out=np.maximum(0.,p+applied)
    if not np.isfinite(out).all() or (alpha==0 and not np.array_equal(out,p)):
        raise AssertionError('Invalid correction')
    return out


def stats(d: pd.DataFrame, a: float) -> dict:
    pred=correction(d,a)
    tru=d.truth.to_numpy(float)
    old=d.selected_prediction.to_numpy(float)
    ref=d.reference_prediction.to_numpy(float)
    ref_mae=float(np.mean(abs(ref-tru)))
    old_mae=float(np.mean(abs(old-tru)))
    new_mae=float(np.mean(abs(pred-tru)))
    if ref_mae <= 1e-10:
        raise RuntimeError('Zero pseudo persistence baseline error')
    return {'rows':len(d),'alpha':float(a),'reference_mae':ref_mae,
            'champion_mae':old_mae,'candidate_mae':new_mae,
            'champion_ratio':old_mae/ref_mae,'candidate_ratio':new_mae/ref_mae,
            'delta_ratio':(new_mae-old_mae)/ref_mae,
            'changed':int(np.sum(abs(pred-old)>1e-12))}


def train_selector(early: dict[str,pd.DataFrame]) -> tuple[float,list[dict]]:
    results=[]
    for a in ALPHA_GRID:
        per=[stats(df,a) for df in early.values()]
        delta=[v['delta_ratio'] for v in per]
        wins=sum(v < -1e-10 for v in delta)
        mean=float(np.mean(delta))
        worst=float(max(delta))
        supported=(a>0 and len(per)>=MIN_OLD_RECORDINGS and
            wins>=2 and mean <= -SELECT_MEAN_RATIO_GAIN and
            worst <= SELECT_MAX_OLD_RATIO_HARM)
        results.append({'alpha':a,'mean_delta':mean,'worst_delta':worst,
            'wins':wins,'n_recordings':len(per),'supported':supported,
            'per_recording':{k:stats(d,a)['delta_ratio'] for k,d in early.items()}})
    admissible=[x for x in results if x['supported']]
    selected=min(admissible,key=lambda v:v['mean_delta'])['alpha'] if admissible else 0.
    return selected,results


def details(df: pd.DataFrame, alpha: float) -> list[dict]:
    x=df.copy()
    x['candidate']=correction(x,alpha)
    x['horizon_band']=pd.cut(x.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
    x['signed_bias_band']=pd.cut(x.shadow_bias,[-np.inf,-5,-1,1,5,np.inf],
                                labels=['strong_negative','negative','near_zero','positive','strong_positive'])
    result=[]
    for dim in ('unit','horizon_band','signed_bias_band','anchor_id'):
        for key,group in x.groupby(dim,observed=True,dropna=False):
            result.append({'recording':str(df.recording.iloc[0]),'dimension':dim,'group':str(key),
                'rows':len(group),'champion_mae':float(abs(group.selected_prediction-group.truth).mean()),
                'candidate_mae':float(abs(group.candidate-group.truth).mean()),
                'delta_abs_error_sum':float((abs(group.candidate-group.truth)-abs(group.selected_prediction-group.truth)).sum())})
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--smoke-persistence',action='store_true',help='Software test only; NOT a score')
    a=p.parse_args()
    root=a.root.resolve()
    out=(a.output_dir or root/'reports'/('v3_4_smoke' if a.smoke_persistence else 'v3_4')).resolve()
    core.assert_output_dir_safe(root,out)
    data,source,model,identity=load_inputs(root,a.smoke_persistence)
    orig_train_sha=core.digest(source)
    source_path=Path(identity['path']) if identity.get('path') else None
    orig_model_sha=core.digest(source_path) if source_path else None
    print('V3.4 shadow forecast calibration (RESEARCH ONLY; NO Kaggle submission)',flush=True)
    print('Champion source:',identity['kind'],flush=True)
    early={}; coverage={}
    for rec in EARLY_TRAIN:
        # Training only from strictly earlier RECORDINGS. The earliest evaluated
        # old recording may have just one predecessor; errors are diagnostic.
        if sum(r<rec for r in data.recording.unique())<1:
            continue
        d,info=build_pseudo(data,model,rec)
        early[rec]=d;coverage[rec]=info
    if len(early)<MIN_OLD_RECORDINGS:
        raise RuntimeError('Insufficient old chronological recordings for calibration')
    selected,alpha_grid=train_selector(early)
    out.mkdir(parents=True,exist_ok=True)
    conf={'experiment':'V3_4_SHADOW_FORECAST_CALIBRATION',
          'selected_alpha':selected,'alpha_grid':list(ALPHA_GRID),
          'selection_source':'ONLY_OLD_RECORDINGS_WITH_PAST_ONLY_TRAIN',
          'correction_cap_people':CORRECTION_CAP_PEOPLE,
          'shadow_lookback_minutes':SHADOW_LOOKBACK,
          'shadow_horizons':list(SHADOW_HORIZONS),
          'forecast_horizons':list(MAIN_HORIZONS),
          'calibration_confident':bool(selected>0),
          'NO_AUTOMATIC_PROMOTION':True,'NO_SUBMISSION':True}
    config_file=out/'v3_4_frozen_config.json'
    config_file.write_text(json.dumps(conf,indent=2),encoding='utf8')
    config_sha=core.digest(config_file)
    print('Selected alpha from old dates ONLY:',selected,'(frozen before later evaluation)',flush=True)
    later={}; causal_audits=[]
    for rec in LATER_PSEUDO:
        d,info=build_pseudo(data,model,rec)
        later[rec]=d;coverage[rec]=info
        anchor=info['selected'][0]
        print('  Running future-value poison audit on',rec,flush=True)
        causal_audits.append(future_poison_audit(data,model,rec,int(anchor['now']),anchor['cameras']))
    per_rec=[]; groups=[]
    for rec,d in {**early,**later}.items():
        res=stats(d,selected)
        per_rec.append({'recording':rec,'stage':'OLDER_CALIBRATION' if rec in early else 'LATER_PSEUDO_REUSED_DATE',**res})
        groups.extend(details(d,selected))
    later_deltas=[row['delta_ratio'] for row in per_rec if row['recording'] in later]
    both_improve=bool(selected>0 and len(later_deltas)==2 and max(later_deltas)<0 and np.mean(later_deltas)<-0.005)
    source_sha_final=core.digest(source)
    if source_sha_final!=orig_train_sha or (source_path and core.digest(source_path)!=orig_model_sha):
        raise RuntimeError('Source or train modified during experiment; STOP')
    conclusion='REJECT_OR_ABSTAIN_NO_AUTOMATIC_PROMOTION'
    if both_improve:
        conclusion='FOLLOWUP_REPLAY_REQUIRED_NOT_PROMOTED'
    summary={
        'experiment':'V3_4_CAUSAL_SHADOW_FORECAST_RESIDUAL',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'status':'SMOKE_ONLY_NOT_SCIENTIFIC' if a.smoke_persistence else 'RESEARCH_ONLY_NO_SUBMISSION',
        'champion_reference_full_frozen_score_NOT_REPLAYED':CHAMPION_SCORE,
        'target_full_score':TARGET_SCORE,'new_verified_full_score':None,
        'champion_identity':identity,'train_sha256':orig_train_sha,
        'source_sha_after':core.digest(source_path) if source_path else None,
        'rules':{'train_recordings_strictly_earlier_than_target':True,
                 'released_and_reference_only_at_or_before_now':True,
                 'shadow_targets_future_of_shadow_origin_but_observed_by_main_origin':True,
                 'removed_hidden_cameras_on_later_pseudo':True,
                 'camera_E_never_corrected':True,
                 'no_truth_in_champion_inputs':True,
                 'no_submit_files_or_kaggle_api':True},
        'coverage':coverage,'future_poison_audits':causal_audits,'older_alpha_scan':alpha_grid,
        'frozen_config_hash':config_sha,'selected_alpha':selected,
        'per_recording':per_rec,'later_both_improve':bool(both_improve),
        'decision':conclusion,
        'limitations':[
            'The 9/11 and 9/12 dates have been reused across many previous experiments; they are not untouched independent validation.',
            'The pseudo-forecast masks do NOT match all four official frozen tasks or official reference denominators.',
            'Forecast-only proxy ratios cannot be added to 0.605700 to claim a new competition score.',
            'Official public/private Kaggle improvement is not demonstrated.',
            'Older recordings are short and may have too little context for representative champion predictions.',
            'Even a positive shadow result requires exact frozen protocol replay and further generalization evidence.',
        ]}
    (out/'v3_4_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    pd.DataFrame(per_rec).to_csv(out/'v3_4_per_recording.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'v3_4_group_metrics.csv',index=False)
    (out/'v3_4_decision.md').write_text(
        '# V3.4 shadow forecast research verdict\n\n'
        f'**{conclusion}**. Champion stays at **{CHAMPION_SCORE:.13f}**. '
        'No submission generated or uploaded.\n\n'
        f'Frozen alpha from only older recordings: **{selected:.3f}**. '
        f'Later pseudo-date guard: **{both_improve}**.\n\n'
        '| Recording | Stage | Rows | Champion MAE | Candidate MAE | Delta ratio |\n'
        '|---|---|---:|---:|---:|---:|\n' + '\n'.join(
            f"| {r['recording']} | {r['stage']} | {r['rows']} | {r['champion_mae']:.5f} | {r['candidate_mae']:.5f} | {r['delta_ratio']:+.6f} |" for r in per_rec)
        + '\n\nThese are historical pseudo-forecast tests, not the exact frozen multi-task score or a Kaggle score.\n',
        encoding='utf8')
    print('Research reports:',out,flush=True)
    print('Decision:',conclusion,'Selected alpha:',selected,flush=True)
    print('Champion preserved. SUBMISSION_GENERATED=False. NEW_VERIFIED_SCORE=None',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as ex:
        print(f'V3.4 STOPPED without submission: {type(ex).__name__}: {ex}',file=sys.stderr)
        raise SystemExit(2)
