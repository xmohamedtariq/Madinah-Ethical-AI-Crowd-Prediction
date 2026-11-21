from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path('/mnt/data')
SRC=ROOT/'Madinah_LastShot_Experimental.csv'
OUT=ROOT/'Madinah_FINAL_HighRisk_CausalLast.csv'
TEST=ROOT/'working_test.csv';SAMPLE=ROOT/'working_sample_submission.csv'
pd.options.mode.copy_on_write=True
base=pd.read_csv(SRC,float_precision='round_trip')
sample=pd.read_csv(SAMPLE)
test=pd.read_csv(TEST)
assert len(base)==len(sample)==4457 and base.id.equals(sample.id) and list(base.columns)==['id','people']
assert test.loc[test.id.notna(),'id'].is_unique
meta=test.loc[test.id.notna(),['id','recording','unit','kind','now_minute_of_day']].copy()
assert len(meta)==4457
p=base.merge(meta,on='id',how='left',validate='one_to_one',sort=False)
assert p.kind.notna().all()
original=p.people.to_numpy(float).copy()
new_cams={'cam_I','cam_J','cam_K','cam_L','cam_M'}
changed=[];notes=[]
for (rec,cam,now),grp in p[(p.kind=='forecast')&(p.unit.isin(new_cams))].groupby(['recording','unit','now_minute_of_day']):
    observations=test[(test.recording==rec)&(test.unit==cam)&test.people.notna() & (test.minute_of_day<=now)].sort_values('minute_of_day')
    if observations.empty: continue
    last=observations.iloc[-1]
    # Rule: Forecast may use only values available at, or before, the origin time.
    assert last.minute_of_day<=now
    # Hold out if no exact/recent observation; dataset happens to have lags 0 or 1.
    if (now-last.minute_of_day)>1:continue
    val=max(0.0,float(last.people))
    # Hold the known current level for the upcoming 15 minutes, a robust strategy
    # on six separate historical training recordings (no test target truth used).
    p.loc[grp.index,'people']=val
    notes.append(dict(recording=str(rec),camera=str(cam),origin=int(now),rows=len(grp),lag=int(now-last.minute_of_day),last_observed=val,mean_original=float(original[grp.index].mean())))
arr=p.people.to_numpy(float)
assert np.isfinite(arr).all() and np.all(arr>=0)
assert len(arr)==4457 and p.id.equals(sample.id) and p.id.is_unique and set(p.id)==set(meta.id)
res=pd.DataFrame({'id':p.id,'people':arr})
res.to_csv(OUT,index=False,float_format='%.8f')
reread=pd.read_csv(OUT)
assert reread.id.equals(sample.id) and reread.id.is_unique and len(reread)==4457
assert list(reread.columns)==['id','people'] and reread.people.notna().all() and np.isfinite(reread.people).all() and reread.people.ge(0).all()
diff=np.abs(reread.people.to_numpy(float)-original)
mask_new=(p.kind=='forecast') & p.unit.isin(new_cams)
assert (diff[~mask_new.to_numpy()]<5.1e-8).all(), 'Non-newcam target shifted'
summary={
  'candidate':OUT.name,
  'source':SRC.name,
  'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
  'size_bytes':OUT.stat().st_size,
  'rows':len(reread),
  'new_camera_forecasts_modified':int((diff[mask_new.to_numpy()]>1e-6).sum()),
  'new_camera_forecast_total':int(mask_new.sum()),
  'max_abs_change':float(diff.max()),
  'mean_abs_change_newcam':float(diff[mask_new.to_numpy()].mean()),
  'changed_by_kind':p.assign(changed=diff>1e-6).groupby('kind').changed.sum().astype(int).to_dict(),
  'other_kinds_identical_to_source':bool((diff[~mask_new.to_numpy()]<5.1e-8).all()),
  'origin_groups':notes,
  'validation_train_shadow':{'protocol':'released historical train.csv only; last-observation vs mean-reversion on 6 recordings','last_mae':20.252758,'reversal25_mae':20.367278,'note':'Different cameras/recordings, cannot establish improvement on Kaggle new cameras'},
  'verified_local_score':None,
  'official_kaggle_score':None,
  'can_claim_0_400':False,
  'risk':'HIGH. Test cameras I-M never appear in historical TRAIN; a fixed carry-forward forecast can be worse than archived reference. This is a causal alternative, not a validated winning file.',
  'user_action':'Manually choose whether to submit; file has not been uploaded to Kaggle'
}
(ROOT/'Madinah_FINAL_HighRisk_Audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('FILE',OUT,'size',OUT.stat().st_size)
print('AUDIT',json.dumps({k:v for k,v in summary.items() if k!='origin_groups'},ensure_ascii=False,indent=2))
