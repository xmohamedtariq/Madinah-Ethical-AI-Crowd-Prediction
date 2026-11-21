"""Standalone last-chance Madinah submission builder.
All calibration offsets are fixed based on 2026-09-11/12 reused development labels,
not on hidden competition truths. Research-only, no Kaggle upload.
"""
from pathlib import Path
import numpy as np, pandas as pd, hashlib, json
root=Path('/mnt/data')
base=pd.read_csv(root/'Champion_Production_DryRun_605700.csv')
test=pd.read_csv(root/'working_test.csv')
sample=pd.read_csv(root/'working_sample_submission.csv')
assert base.columns.tolist()==['id','people'] and sample.columns.tolist()==['id','people']
assert base.id.is_unique and len(base)==len(sample)==4457
assert base.id.tolist()==sample.id.tolist()
targets=test[test.id.notna()].copy();assert len(targets)==4457 and targets.id.is_unique
assert not targets.people.notna().any()
assert set(base.id)==set(targets.id)
mod=base.merge(targets[['id','recording','unit','kind']],on='id',how='left',validate='one_to_one')
# Extra conservative vs full in-sample mean-bias correction; fixed, no use of test target labels.
# Apply only to matching recurring camera/weekdays and kind.
corrections={
('20260918_1100','cam_B','hidden'):0.50,
('20260918_1100','cam_G','hidden'):0.25,
('20260919_1100','cam_D','hidden'):-1.25,
('20260918_1100','cam_B','hidden-future'):0.50,
('20260918_1100','cam_G','hidden-future'):0.25,
('20260919_1100','cam_D','hidden-future'):-2.00,
('20260918_1100','cam_H','missing'):-1.00,
('20260919_1100','cam_H','missing'):-1.00,
}
mod['delta']=np.zeros(len(mod),dtype=float)
for (rec,cam,kind),delta in corrections.items():
 mask=mod.recording.eq(rec)&mod.unit.eq(cam)&mod.kind.eq(kind)
 if not mask.any():
  print('WARNING unused calibration',rec,cam,kind)
 mod.loc[mask,'delta']=delta
mod['new_people']=np.maximum(0,mod.people.astype(float)+mod.delta)
output=mod[['id','new_people']].rename(columns={'new_people':'people'})
out=root/'Madinah_LastChance_Calibrated_Challenge.csv'
output.to_csv(out,index=False,float_format='%.8f')
check=pd.read_csv(out)
assert len(check)==4457 and check.columns.tolist()==['id','people'] and check.id.tolist()==sample.id.tolist()
assert check.id.is_unique and np.isfinite(check.people).all() and (check.people>=0).all()
assert (check.people-pd.read_csv(root/'Champion_Production_DryRun_605700.csv').people).abs().gt(1e-6).sum()>0
print('Output:',out,'size=',out.stat().st_size,'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
print('Per kind changed',mod[mod.delta.ne(0)].groupby(['recording','kind','unit']).agg(rows=('delta','size'),offset=('delta','first')).to_string())
print('Changed',int((mod.new_people-mod.people).abs().gt(1e-6).sum()),'of 4457')
# Retrospective eval: NOT independent, label source repeated in candidate selection.
val=pd.read_csv(root/'val_with_truth.csv')
refs={'20260911_1100':{'forecast':42.62255128205128,'hidden':4.878997311803499,'hidden-future':5.757955489227114,'missing':8.60094304493806},'20260912_1100':{'forecast':20.571955454545456,'hidden':7.543435829951615,'hidden-future':9.900914687194568,'missing':5.569755984973631}}
base_score=0.0;modified_score=0.0;grp=[]
maprec={'20260911_1100':'20260918_1100','20260912_1100':'20260919_1100'}
for (rec,kind),g in val.groupby(['recording','kind']):
 change=pd.Series(0.,index=g.index)
 for (rr,cam,kk),x in corrections.items():
  if rr==maprec[rec] and kk==kind:
   change.loc[g[g.unit.eq(cam)].index]=x
 pred=(g.people+change).clip(lower=0)
 a=float(np.mean(np.abs(g.people-g.people_truth))); b=float(np.mean(np.abs(pred-g.people_truth)))
 ra=a/refs[rec][kind]; rb=b/refs[rec][kind]
 base_score+=ra/8;modified_score+=rb/8
 grp.append(dict(recording=rec,kind=kind,n=len(g),mae_champion=a,mae_candidate=b,contribution_change=(rb-ra)/8))
print('Local retrospective baseline',base_score,'candidate',modified_score,'delta',modified_score-base_score)
print(pd.DataFrame(grp).to_string(index=False))
report={
'file':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
'row_count':len(check),'id_match_sample':True,'nonfinite':0,'negative':0,
'changed_count':int((mod.new_people-mod.people).abs().gt(1e-6).sum()),
'calibrations':{f'{a}|{b}|{c}':d for (a,b,c),d in corrections.items()},
'retrospective_local_baseline':base_score,'retrospective_local_candidate':modified_score,
'caveat':'Offsets derived from same reused September 11/12 development labels used to assess them. This is IN-SAMPLE, optimistic, not independently validated; official Kaggle performance unknown.',
'upload_status':'Not uploaded to Kaggle'}
(root/'Madinah_LastChance_Calibrated_Audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
