from pathlib import Path
import hashlib, json
import pandas as pd
import numpy as np
root=Path('/mnt/data')
paths={
 'sample':root/'working_sample_submission.csv',
 'test':root/'working_test.csv',
 'champion':root/'Champion_Production_DryRun_605700.csv',
 'calibrated':root/'Madinah_LastChance_Calibrated_Challenge.csv',
 'prior_official':root/'files/submission_madinah_codex605700_experimental.csv',
}
for k,v in paths.items():
 if not v.is_file(): raise RuntimeError(f'Missing input {k}: {v}')
sample=pd.read_csv(paths['sample'])
test=pd.read_csv(paths['test'])
champ=pd.read_csv(paths['champion'])
cal=pd.read_csv(paths['calibrated'])
prior=pd.read_csv(paths['prior_official'])
assert list(sample.columns)==list(champ.columns)==list(cal.columns)==['id','people']
assert len(sample)==len(champ)==len(cal)==len(prior)==4457
assert sample.id.is_unique and champ.id.is_unique and cal.id.is_unique and prior.id.is_unique
assert set(sample.id)==set(champ.id)==set(cal.id)==set(prior.id)
targets=test.loc[test.id.notna(),['id','kind']].copy()
assert len(targets)==4457 and targets.id.is_unique and targets.kind.notna().all()
assert set(targets.id)==set(sample.id)
for c in (sample,champ,cal,prior):
 assert np.isfinite(c.people).all() and (c.people>=0).all()
kind=sample[['id']].merge(targets,on='id',how='left',validate='one_to_one')
champ=sample[['id']].merge(champ,on='id',how='left',validate='one_to_one')
cal=sample[['id']].merge(cal,on='id',how='left',validate='one_to_one')
prior=sample[['id']].merge(prior,on='id',how='left',validate='one_to_one')
use_champ=kind.kind.eq('forecast')
values=np.where(use_champ,champ.people.to_numpy(dtype=float),cal.people.to_numpy(dtype=float))
assert np.isfinite(values).all() and (values>=0).all()
out=root/'Madinah_Best_Per_Category_Combined.csv'
pd.DataFrame({'id':sample.id,'people':values}).to_csv(out,index=False,float_format='%.8f')
result=pd.read_csv(out)
assert result.columns.tolist()==['id','people'] and result.id.equals(sample.id)
assert result.id.is_unique and len(result)==4457 and np.isfinite(result.people).all() and (result.people>=0).all()
max_lastchance_diff=float(np.abs(result.people.to_numpy()-cal.people.to_numpy()).max())
max_prior_diff=float(np.abs(result.people.to_numpy()-prior.people.to_numpy()).max())
changed_by_kind={}
for task in ['forecast','hidden','hidden-future','missing']:
 mask=kind.kind.eq(task).to_numpy()
 diff=np.abs(result.people.to_numpy()[mask]-prior.people.to_numpy()[mask])
 changed_by_kind[task]={
  'rows':int(mask.sum()),
  'source':'Champion 0.605700' if task=='forecast' else 'LastChance calibration based on Champion',
  'changed_from_previous_official_0_729':int((diff>1e-6).sum()),
 }
summary={
 'name':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'row_count':len(result),'unique_ids':result.id.nunique(),'matches_sample_order':True,
 'changed_vs_prior_official_0_729':int((np.abs(result.people.to_numpy()-prior.people.to_numpy())>1e-6).sum()),
 'max_abs_difference_vs_lastchance':max_lastchance_diff,
 'max_abs_difference_vs_prior':max_prior_diff,
 'per_kind':changed_by_kind,
 'retrospective_local_selected_0_594289':'in-sample/reused folds only; not independently verified',
 'public_kaggle_score':'UNTESTED',
 'warning':'This composite is numerically identical to the previously generated LastChance CSV. Do not submit again if LastChance was already submitted.',
}
(root/'Madinah_Best_Per_Category_Audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
