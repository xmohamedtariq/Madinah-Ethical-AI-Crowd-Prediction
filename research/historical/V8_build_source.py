#!/usr/bin/env python3
"""Conservative development candidate: median17 on hidden outputs only.

Input is the already-audited V7 CSV, not unknown test labels. Nonhidden tasks
are copied byte-for-byte. Hidden uses a centered, noncausal neighborhood, as
permitted by the competition's hidden task (not applied to forecast/HF).
"""
from pathlib import Path
import hashlib, json, io, csv
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

BASE=Path('/mnt/data/recovered/submission_v7_masked_foundation_temporal.csv')
DATA=Path('/mnt/data/work/v2/data/official')
OUT=Path('/mnt/data/submission_v8_hidden_median17.csv')
REPORT=Path('/mnt/data/v8_validation_report.json')
V7_SHA='1c6984fec7d3ec3674fa0135b8b08e3e01394ecf5bacdf319ea22faaa6a998e8'
DATA_SHA={'train.csv':'3210a0b33212a60a888977848ec09f5f89bc0954bfd3b88e61fb9d840537f6f6',
          'test.csv':'6f2c6400b02964f3f37b912a5d6dbe085b28e7070a8094bdfad918e57766e6b9',
          'sample_submission.csv':'22e1d6a23de7095c18be2e1e2d8dfbccaaa4010e647feee608e10f16ee798471'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE)==V7_SHA, 'V7 source integrity failed'
for file,h in DATA_SHA.items():assert sha(DATA/file)==h, f'{file} hash mismatch'
base=pd.read_csv(BASE,float_precision='round_trip')
sample=pd.read_csv(DATA/'sample_submission.csv')
test=pd.read_csv(DATA/'test.csv')
assert len(base)==len(sample)==4457
assert base.id.equals(sample.id) and list(base.columns)==['id','people']
assert base.id.is_unique and base.people.notna().all() and np.isfinite(base.people).all() and base.people.ge(0).all()
gaps=test.loc[test.id.notna()].copy()
assert len(gaps)==4457 and gaps.id.is_unique
assert set(base.id)==set(gaps.id)
assert gaps.kind.isin(['forecast','hidden','hidden-future','missing']).all()
hid=gaps[gaps.kind.eq('hidden')].copy()
assert len(hid)==849
pred=dict(zip(base.id,base.people))
replacements={}
for (_,unit),g in hid.groupby(['recording','unit'],sort=False):
    g=g.sort_values('minute_of_day')
    t=g.minute_of_day.to_numpy();y=g.id.map(pred).to_numpy(dtype=float)
    bounds=np.r_[0,np.where(np.diff(t)>1)[0]+1,len(g)]
    for i,j in zip(bounds[:-1],bounds[1:]):
        # Match the historical validation transform. Don't join distant gaps.
        yy=y[i:j]
        sm=median_filter(yy,size=17,mode='nearest') if len(yy)>=3 else yy
        for k,v in zip(g.id.iloc[i:j],sm):replacements[k]=float(max(v,0.0))
assert len(replacements)==849
lines=BASE.read_bytes().splitlines(keepends=True)
new=[lines[0]];unchanged=0
for row in lines[1:]:
    ident=next(csv.reader([row.decode('utf-8').rstrip('\r\n')]))[0]
    if ident not in replacements:
        new.append(row);unchanged+=1
    else:
        ending='\r\n' if row.endswith(b'\r\n') else '\n'
        buf=io.StringIO(newline='')
        csv.writer(buf,lineterminator=ending).writerow([ident,format(replacements[ident],'.17g')])
        new.append(buf.getvalue().encode('utf-8'))
OUT.write_bytes(b''.join(new))
output=pd.read_csv(OUT,float_precision='round_trip')
assert output.id.equals(sample.id) and len(output)==4457 and output.id.is_unique
assert list(output.columns)==['id','people']
assert output.people.notna().all() and np.isfinite(output.people).all() and output.people.ge(0).all()
unchanged_actual=(output.people.to_numpy()==base.people.to_numpy())
by_kind={}
for kind,g in gaps.groupby('kind'):
    idx=g.id
    a=base.set_index('id').loc[idx,'people'].to_numpy();b=output.set_index('id').loc[idx,'people'].to_numpy()
    changed=int(np.sum(a!=b))
    by_kind[kind]={'rows':len(g),'changed_predictions':changed,'max_abs_change':float(np.max(np.abs(a-b)))}
    if kind!='hidden':assert changed==0, f'leakage risk, changed {kind}'
assert unchanged==3608
assert sum(v['rows'] for v in by_kind.values())==4457
assert set(replacements)==set(hid.id)
report={
    'candidate':'V8 experimental hidden-only median17',
    'input_baseline':'V7 verified candidate',
    'input_sha256':V7_SHA,
    'output_sha256':sha(OUT),'rows':len(output),'columns':['id','people'],
    'exact_sample_ids_and_order':True,'unique_ids':True,'finite_nonnegative':True,
    'future_target_count':int(gaps.kind.isin(['forecast','hidden-future']).sum()),
    'forecast_and_hidden_future_byte_identical_to_V7':True,
    'nonhidden_rows_exact_byte_identical_to_V7':unchanged,
    'by_kind':by_kind,
    'method':'17-sample centered median applied separately to contiguous hidden-camera prediction runs, never future-task outputs or future input rows',
    'matched_development_score_v6':0.6829470361179171,
    'matched_development_score_v7':0.6491889954416621,
    'matched_development_score_v8':0.647335,
    'score_calibration':'local, reconstructed reference and previously inspected development days; not Kaggle official',
    'public_score':None,'target_public_score':0.400,'goal_verified':False,
    'known_limitations':'Only two dated masked development recordings and no fresh holdout; V7 and smoothing selection inspected same development dates. Public/private Kaggle score unknown.'
}
REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
