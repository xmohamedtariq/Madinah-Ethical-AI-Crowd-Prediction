#!/usr/bin/env python3
"""Reproducible V10 experimental temporal-only competition submission.

MODIFIED TASKS ONLY:
  hidden: 75% run-wise 17-point median + 25% run-wise poly2 Savgol (width<=31)
  hidden-future: separate each camera/origin and apply Savgol poly2 (width<=15)
UNMODIFIED: forecast, missing; copied byte-for-byte from verified V7.

The hidden-future operation uses ONLY predictions at one forecast origin,
not observed people after now. Smoothing predicted horizons is causal because
all constituent values were already produced using inputs available at now.

The local score remains development-only; no independent Kaggle evaluation.
"""
from pathlib import Path
import csv,io,hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'recovered/submission_v7_masked_foundation_temporal.csv'
DATA=ROOT/'work/v2/data/official'
OUT=ROOT/'submission_v10_temporal_ensemble.csv'
REPORT=ROOT/'V10_integrity.json'
BASE_SHA='1c6984fec7d3ec3674fa0135b8b08e3e01394ecf5bacdf319ea22faaa6a998e8'
DATA_SHA={'train.csv':'3210a0b33212a60a888977848ec09f5f89bc0954bfd3b88e61fb9d840537f6f6',
          'test.csv':'6f2c6400b02964f3f37b912a5d6dbe085b28e7070a8094bdfad918e57766e6b9',
          'sample_submission.csv':'22e1d6a23de7095c18be2e1e2d8dfbccaaa4010e647feee608e10f16ee798471'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def trend_savgol(y,maxwidth):
 n=len(y)
 if n<5:return y.copy()
 width=min(maxwidth,n if n%2 else n-1)
 return savgol_filter(y,window_length=width,polyorder=2,mode='interp')
def fix_hidden(g):
 g=g.sort_values('minute_of_day');t=g.minute_of_day.to_numpy(float);raw=g.baseline.to_numpy(float);sm=raw.copy()
 breaks=np.r_[0,np.where(np.diff(t)>1)[0]+1,len(g)]
 for a,b in zip(breaks[:-1],breaks[1:]):
  v=raw[a:b]
  if len(v)<3:continue
  med=median_filter(v,size=17,mode='nearest')
  sg=trend_savgol(v,31)
  sm[a:b]=np.maximum(0.,.75*med+.25*sg)
 return dict(zip(g.id,sm))
def fix_future(g):
 # Group is one CAMERA and one NOW only, and contains ONLY inferred targets.
 # No observed rows or actual target values enter this function.
 g=g.sort_values('horizon');v=g.baseline.to_numpy(float)
 smooth=np.maximum(0.,trend_savgol(v,15))
 return dict(zip(g.id,smooth))

def main():
 assert sha(BASE)==BASE_SHA
 for f,h in DATA_SHA.items():assert sha(DATA/f)==h,('Official data changed',f)
 sample=pd.read_csv(DATA/'sample_submission.csv');test=pd.read_csv(DATA/'test.csv')
 base=pd.read_csv(BASE,float_precision='round_trip')
 assert len(sample)==len(base)==4457
 assert list(base.columns)==list(sample.columns)==['id','people'] and base.id.equals(sample.id) and base.id.is_unique
 assert base.people.notna().all() and np.isfinite(base.people).all() and base.people.ge(0).all()
 targets=test.loc[test.id.notna()].copy()
 assert len(targets)==4457 and targets.id.is_unique and set(targets.id)==set(sample.id)
 assert targets.kind.isin(['forecast','hidden','hidden-future','missing']).all()
 assert targets[targets.kind.eq('hidden')].shape[0]==849
 assert targets[targets.kind.eq('hidden-future')].shape[0]==484
 targets['baseline']=targets.id.map(dict(zip(base.id,base.people)))
 assert targets.baseline.notna().all()
 replacements={}
 for _,g in targets.loc[targets.kind.eq('hidden')].groupby(['recording','unit'],sort=False):
  replacements.update(fix_hidden(g))
 future=targets.loc[targets.kind.eq('hidden-future')]
 assert future.now.notna().all() and future.now_minute_of_day.notna().all() and future.horizon.notna().all()
 assert (future.minute_of_day>future.now_minute_of_day).all()
 for _,g in future.groupby(['recording','unit','now_minute_of_day'],sort=False):
  assert g.now_minute_of_day.nunique()==1
  assert (g.minute_of_day-g.now_minute_of_day == g.horizon).all()
  assert g.id.is_unique and g.horizon.is_unique
  replacements.update(fix_future(g))
 assert len(replacements)==849+484
 assert set(replacements)==set(targets[targets.kind.isin(['hidden','hidden-future'])].id)
 lines=BASE.read_bytes().splitlines(keepends=True);out=[lines[0]];untouched=0
 for line in lines[1:]:
  ident=next(csv.reader([line.decode('utf8').rstrip('\r\n')]))[0]
  if ident not in replacements:
   out.append(line);untouched+=1
  else:
   ending='\r\n' if line.endswith(b'\r\n') else '\n';buf=io.StringIO(newline='')
   csv.writer(buf,lineterminator=ending).writerow([ident,format(replacements[ident],'.17g')])
   out.append(buf.getvalue().encode('utf8'))
 assert untouched==3124
 OUT.write_bytes(b''.join(out))
 new=pd.read_csv(OUT,float_precision='round_trip')
 assert new.id.equals(sample.id) and new.id.is_unique and list(new.columns)==['id','people']
 assert new.people.notna().all() and np.isfinite(new.people).all() and new.people.ge(0).all()
 aa=base.set_index('id');bb=new.set_index('id');stats={}
 for kind,g in targets.groupby('kind'):
  idx=g.id.to_numpy();dif=bb.loc[idx,'people'].to_numpy()-aa.loc[idx,'people'].to_numpy()
  count=int(np.count_nonzero(dif))
  if kind in ['forecast','missing']:assert count==0
  stats[kind]={'total_rows':len(g),'changed_predictions':count,'max_absolute_change':float(np.max(np.abs(dif)))}
 assert sum(x['total_rows'] for x in stats.values())==4457
 meta={'file':OUT.name,'sha256':sha(OUT),'base_sha256':BASE_SHA,'rows':4457,
   'columns':['id','people'],'exact_id_order':True,'unique_ids':True,'finite_nonnegative':True,
   'unmodified_rows_byte_identical':untouched,'forecast_rows_byte_identical_to_base':2660,
   'missing_rows_byte_identical_to_base':464,'hidden_rows_input_source':'only V7 predictions',
   'hidden_future_input_source':'only V7 origin-specific causal predictions',
   'test_targets_used_for_training_or_smoothing':False,
   'hidden_formula':'0.75 median17+0.25 SavitzkyGolay31 on V7 hidden by contiguous camera run',
   'hidden_future_formula':'SavitzkyGolay poly2 width<=15 on V7 hidden-future, within one unit+now',
   'public_score':None,'score_is_local_only':True,
   'by_kind':stats,'warning':'Parameters were selected using previously inspected development recordings; this is not independent out-of-sample validation or 99% improvement.'}
 REPORT.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
