from pathlib import Path
import pandas as pd,numpy as np, json,hashlib
ROOT=Path('/mnt/data')
base=pd.read_csv(ROOT/'Madinah_Best_Per_Category_Combined.csv'); test=pd.read_csv(ROOT/'working_test.csv');sample=pd.read_csv(ROOT/'working_sample_submission.csv'); val=pd.read_csv(ROOT/'val_with_truth.csv')
assert len(base)==len(sample)==4457 and list(base.columns)==['id','people'] and base.id.equals(sample.id)
assert len(test[test.id.notna()])==4457
meta=test.loc[test.id.notna(), ['id','recording','unit','kind','minute_of_day']]
assert meta.id.is_unique
val_mod=val.copy();mod=base.merge(meta,on='id',validate='one_to_one',how='left');assert mod.kind.notna().all()
refs={'20260911_1100':{'forecast':42.62255128205128,'hidden':4.878997311803499,'hidden-future':5.757955489227114,'missing':8.60094304493806},'20260912_1100':{'forecast':20.571955454545456,'hidden':7.543435829951615,'hidden-future':9.900914687194568,'missing':5.569755984973631}}
# Two levels: conservative and higher-risk; no claim of independent gain.
additional={
 ('Fri','cam_A','forecast'):-2.5,
 ('Fri','cam_E','forecast'):-2.0,
 ('Fri','cam_F','forecast'):-1.0,
 ('Sat','cam_A','forecast'):+2.0,
 ('Sat','cam_E','forecast'):+1.8,
 ('Sat','cam_F','forecast'):+0.5,
 ('Fri','cam_B','hidden'): +0.5,
 ('Fri','cam_G','hidden'): +0.1,
 ('Sat','cam_D','hidden'):+0.15,
 ('Fri','cam_B','hidden-future'):+0.7,
 ('Fri','cam_G','hidden-future'):+0.2,
 ('Sat','cam_D','hidden-future'):-0.6,
}

def xform(df, rec_col='recording', kind_col='kind',unit_col='unit'):
 z=df.copy(); z['daytoken']=np.where(z[rec_col].astype(str).str.startswith(('20260911','20260918')),'Fri','Sat')
 z['offset']=0.0
 for (day,cam,kind),shift in additional.items():
  mask=z.daytoken.eq(day)&z[unit_col].eq(cam)&z[kind_col].eq(kind)
  z.loc[mask,'offset']=shift
 z['new_people']=np.maximum(0,z.people+z.offset)
 # centered smoothing *only* for task Hidden (not Hidden-future, Forecast, Missing).
 # never mix recordings/cameras or non-adjacent minutes; uses only model output, no withheld truth.
 for (rec,unit),g in z[z.kind.eq('hidden')].groupby([rec_col,unit_col]):
  s=g.sort_values('minute_of_day');mins=s.minute_of_day.to_numpy(int); begin=0
  for end in list(np.where(np.diff(mins)>1)[0]+1)+[len(s)]:
   segment=s.iloc[begin:end]
   if len(segment)>=3:
    smooth=segment.new_people.rolling(17,center=True,min_periods=1).mean().to_numpy()
    z.loc[segment.index,'new_people']=0.85*segment.new_people.to_numpy()+0.15*smooth
   begin=end
 return z

mod=xform(mod)
# Reconstructed local scoring: on reused, selected historical folds, optimistic.
val_mod['baseline']=val_mod.people
# incorporate previous LastChance calibration already embedded in real test candidate
basic={('Fri','cam_B','hidden'):0.5,('Fri','cam_G','hidden'):0.25,('Sat','cam_D','hidden'):-1.25,('Fri','cam_B','hidden-future'):0.5,('Fri','cam_G','hidden-future'):0.25,('Sat','cam_D','hidden-future'):-2.0,('Fri','cam_H','missing'):-1.0,('Sat','cam_H','missing'):-1.0}
val_mod['daytoken']=np.where(val_mod.recording.str.startswith('20260911'),'Fri','Sat')
for (day,unit,kind),offset in basic.items():
 mask=val_mod.daytoken.eq(day)&val_mod.unit.eq(unit)&val_mod.kind.eq(kind)
 val_mod.loc[mask,'people']=np.maximum(0,val_mod.loc[mask,'people']+offset)
local_previous=val_mod.copy()
val_mod=xform(val_mod)

def score(df):
 contrib=[]
 for (rec,kind),g in df.groupby(['recording','kind']):
  mae=float(np.mean(np.abs(g.new_people-g.people_truth))) if 'new_people' in g else float(np.mean(np.abs(g.people-g.people_truth)))
  contrib.append({'recording':rec,'kind':kind,'n':len(g),'mae':mae,'contribution':mae/refs[rec][kind]/8})
 return pd.DataFrame(contrib)
prior=score(local_previous);new=score(val_mod)
result=prior.merge(new,on=['recording','kind','n'],suffixes=('_prior','_new'))
result['score_change']=result.contribution_new-result.contribution_prior
out=ROOT/'Madinah_LastShot_Experimental.csv';output=pd.DataFrame({'id':mod.id,'people':mod.new_people});output.to_csv(out,index=False,float_format='%.8f')
check=pd.read_csv(out)
assert len(check)==4457 and list(check.columns)==['id','people'] and check.id.equals(sample.id) and check.id.is_unique and np.isfinite(check.people).all() and check.people.ge(0).all()
assert set(check.id)==set(test.loc[test.id.notna(),'id'])
changed=np.abs(check.people.to_numpy()-base.people.to_numpy())>1e-6
meta['changed']=changed
print('changes:',meta.groupby('kind').changed.agg(['sum','count']).to_string())
print('reused local prior',prior.contribution.sum(),'new',new.contribution.sum())
print(result.to_string(index=False))
print('NEW:',out,'sha:',hashlib.sha256(out.read_bytes()).hexdigest())
report={'name':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'row_count':4457,'valid':True,'changes':int(changed.sum()),'changes_by_kind':meta.groupby('kind').changed.sum().astype(int).to_dict(),'previous_retrospective':float(prior.contribution.sum()),'new_retrospective':float(new.contribution.sum()),'per_task_recording':result.to_dict(orient='records'),'adjustments':{'|'.join(k):v for k,v in additional.items()},'caveat':'High-risk experimental unsubmitted output. Both scores are on heavily reused in-sample historic dev recordings; not official Kaggle and not independent confirmation. No demonstrated route to 0.400. Forecast offsets are week/weekday-specific and may not transfer. Only prediction-derived smoothing on Hidden. No test target truth used.'}
(ROOT/'Madinah_LastShot_Experimental_Audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
