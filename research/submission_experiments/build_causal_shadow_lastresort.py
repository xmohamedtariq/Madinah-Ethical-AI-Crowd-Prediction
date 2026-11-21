from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path('/mnt/data')
BASE = ROOT / 'Madinah_LastShot_Experimental.csv'
TEST = ROOT / 'working_test.csv'
SAMPLE = ROOT / 'working_sample_submission.csv'
SHADOW = ROOT / 'final_shadow_checks.csv'
OUTPUT = ROOT / 'Madinah_ShadowValidated_LastResort.csv'
REPORT = ROOT / 'Madinah_ShadowValidated_LastResort_Audit.json'

base = pd.read_csv(BASE, float_precision='round_trip')
test = pd.read_csv(TEST)
sample = pd.read_csv(SAMPLE)
assert list(base.columns) == ['id', 'people']
assert len(base) == len(sample) == 4457 and base.id.equals(sample.id)
assert base.id.is_unique and np.isfinite(base.people.to_numpy()).all()
targets = test.loc[test.id.notna()].copy()
assert len(targets) == 4457 and targets.id.is_unique
assert targets.people.isna().all(), 'Unexpected released truth for an official test target.'
rows = base.merge(targets[['id','recording','kind','unit','now_minute_of_day','minute_of_day']], on='id', how='left', validate='one_to_one')
assert rows.kind.notna().all()
original = rows.people.to_numpy(float).copy()
mask = rows.kind.eq('forecast') & rows.unit.isin(['cam_I','cam_J','cam_K','cam_L','cam_M'])
audit_groups = []
for (rec, cam, now), g in rows.loc[mask].groupby(['recording','unit','now_minute_of_day']):
    observed = test[(test.recording == rec) & (test.unit == cam) & test.people.notna() & (test.minute_of_day <= now)].sort_values('minute_of_day')
    if len(observed) < 3 or now - float(observed.minute_of_day.iloc[-1]) > 1:
        audit_groups.append({'recording':rec,'camera':cam,'now':float(now),'n':len(g),'changed':False,'reason':'unavailable recent history'})
        continue
    last = float(observed.people.iloc[-1])
    prev3 = float(observed.people.iloc[-3:].mean())
    prev5 = float(observed.people.iloc[-min(5,len(observed)):].mean())
    # Both rules use released observations at or before origin only.
    # Selected via pre-origin historical shadow tests, not withheld truth.
    method = 'mean5' if cam == 'cam_M' else 'reversal25'
    level = prev5 if method == 'mean5' else max(0.0, last - 0.25*(last-prev3))
    rows.loc[g.index,'people'] = max(0.0, level)
    audit_groups.append({'recording':rec,'camera':cam,'now':float(now),'n':len(g),'changed':True,'method':method,'level':level,'last':last,'mean3':prev3,'mean5':prev5})

out = rows[['id','people']].copy()
assert np.isfinite(out.people.to_numpy()).all() and (out.people >= 0).all()
assert len(out) == 4457 and out.id.equals(sample.id) and out.id.is_unique and list(out.columns) == ['id','people']
out.to_csv(OUTPUT,index=False,float_format='%.8f')
back = pd.read_csv(OUTPUT)
assert back.id.equals(sample.id) and np.isfinite(back.people.to_numpy()).all() and (back.people >= 0).all()
diff = abs(back.people.to_numpy() - original)
assert (diff[~mask.to_numpy()] < 5.01e-8).all()
shadow = pd.read_csv(SHADOW)
shadow['use'] = np.where(shadow.cam.eq('cam_M'), shadow.method.eq('mean5'), shadow.method.eq('naive_reversal'))
chosen = shadow.loc[shadow.use, ['rec','cam','realnow','pseudo_now','h','truth','pred','ae']].copy()
last = shadow.loc[shadow.method.eq('last'),['rec','cam','realnow','pseudo_now','h','ae']].copy()
paired = chosen.merge(last,on=['rec','cam','realnow','pseudo_now','h'],suffixes=('_alternative','_last'),validate='one_to_one')
assert len(paired) == 2921
means = paired.groupby('rec')[['ae_alternative','ae_last']].mean().reset_index()
report = {
    'candidate':OUTPUT.name, 'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    'rows':len(back), 'changed_from_LastShot':int((diff>1e-6).sum()),
    'changed_by_kind': rows.assign(changed=diff>1e-6).groupby('kind').changed.sum().astype(int).to_dict(),
    'all_nonforecast_and_oldcamera_rows_unchanged':True,
    'selected_rules':{'cam_I':'reversal25','cam_J':'reversal25','cam_K':'reversal25','cam_L':'reversal25','cam_M':'mean5'},
    'released_only_shadow': {
       'n_paired_pseudo_targets':len(paired),
       'last_mean_absolute_error':float(paired.ae_last.mean()),
       'candidate_mean_absolute_error':float(paired.ae_alternative.mean()),
       'per_recording':means.to_dict('records'),
       'note':'Shadow origins from the two actual test recordings, earlier than actual test prediction origins. Correlated pseudo-targets; this is a diagnostic, not an independently sampled validation set. Baseline comparison is carry-forward, not official champion.',
    },
    'historical_development_local_score':0.5927808585814978,
    'historical_development_score_provenance':'Inherited unchanged from LastShot on all historical cameras; validation has no I-M test camera labels. Heavily reused development folds; not independent.',
    'score_0_400_verified':False, 'kaggle_score':None, 'submitted':False,
    'caution':'No valid evidence for score 0.400 or lower. This file changes unseen camera forecasting, with possible Kaggle regression versus Champion. No withheld test labels accessed.',
    'origin_group_details':audit_groups,
}
REPORT.write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'origin_group_details'},indent=2))
