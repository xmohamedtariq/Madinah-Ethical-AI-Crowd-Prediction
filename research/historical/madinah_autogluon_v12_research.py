"""V12 strictly-as-of-now supervised forecast experiment, no Kaggle upload.

Use target label ONLY for fitting or evaluation. Feature snapshots never examine values after now.
CV: train earlier days, validate Sep 11 and Sep 12 with test-mimicking withheld rows.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys, time, json, argparse
from collections import defaultdict
from lightgbm import LGBMRegressor

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import madinah_autogluon_v1 as v1

P=Path(__file__).resolve().parent

def input_file(name, *, report=False):
 candidates = ([P/'reports'/name, P/'madinah_v12_work'/name, P/name] if report else
               [P/'data'/'official'/name, P/'madinah_v12_work'/name, P/name])
 for cand in candidates:
  if cand.is_file():return cand
 raise FileNotFoundError(f'{name} not found; expected in '+', '.join(map(str,candidates)))

tr=pd.read_csv(input_file('train.csv'))
test=pd.read_csv(input_file('test.csv'))
sample=pd.read_csv(input_file('sample_submission.csv'))
old=pd.read_csv(input_file('ag_v7_cv_predictions.csv',report=True))
if len(tr)!=13443 or len(test)!=8611 or len(sample)!=4457:
 raise ValueError('Official input row counts differ from the audited 2026-11-19 research snapshot; inspect data before use')
if len(old)!=3719:
 raise ValueError('Expected V7 CV checkpoint with exactly 3719 rows')

targets=test[test.id.notna()].copy()
ALL_KEYS=['recording','unit','minute_of_day']

class Store:
 def __init__(self,frame):
  self.by={}
  for (r,c),g in frame.groupby(['recording','unit'],sort=False):
   g=g.sort_values('minute_of_day').drop_duplicates('minute_of_day')
   self.by[(str(r),str(c))]=(g.minute_of_day.to_numpy(float),g.people.to_numpy(float))
  self.cams=defaultdict(list)
  for rec,cam in self.by:self.cams[rec].append(cam)
 def pull(self, rec,cam,now):
  ts,vs=self.by.get((rec,cam),(np.array([]),np.array([])))
  ii=np.searchsorted(ts,now,side='right')
  return ts[:ii],vs[:ii]

def series_feat(ts,vs,now):
 f={}
 if len(vs)==0:
  return {'last':np.nan,'mean3':np.nan,'age':999.,'n_recent20':0.,'slope10':np.nan,'std10':np.nan,'relative10':np.nan}
 last=float(vs[-1]); f['last']=last;f['age']=float(now-ts[-1]);f['loglast']=np.log1p(last)
 for w in (3,5,10,15,25,40,60):
  mask=ts>=now-w
  y=vs[mask];t=ts[mask]
  f[f'n{w}']=len(y)
  f[f'mean{w}']=float(np.mean(y)) if len(y) else np.nan
  f[f'median{w}']=float(np.median(y)) if len(y) else np.nan
  f[f'std{w}']=float(np.std(y)) if len(y)>=2 else np.nan
  f[f'slope{w}']=float(np.polyfit(t-t[-1],y,1)[0]) if len(y)>=4 and t[-1]>t[0]+2 else np.nan
  f[f'range{w}']=float(np.ptp(y)) if len(y)>=2 else np.nan
  f[f'relative{w}']=(last-f[f'mean{w}'])/max(5., f[f'mean{w}']) if len(y) else np.nan
  # robust two sided back-slope using recent and earlier half
  if len(y)>=6:
   half=(t[-1]+t[0])/2
   a=y[t<=half];b=y[t>half]
   f[f'halfdiff{w}']=(float(b.mean()-a.mean())/max(5.,last)) if len(a) and len(b) else np.nan
  else: f[f'halfdiff{w}']=np.nan
 for lag in (1,2,3,5,8,10,15,20,30):
  j=np.searchsorted(ts,now-lag,side='right')-1
  if j>=0 and now-lag-ts[j]<=2:
   v=float(vs[j]); f[f'lag{lag}']=v;f[f'delta{lag}']=last-v;f[f'ratio{lag}']=(last+5)/(v+5)
  else:
   f[f'lag{lag}']=np.nan;f[f'delta{lag}']=np.nan;f[f'ratio{lag}']=np.nan
 return f

class Profile:
 def __init__(self,frame):
  self.data=defaultdict(list)
  for (day,cam),g in frame.groupby(['day','unit']):
   g=g.sort_values('minute_of_day');t=g.minute_of_day.to_numpy(float);v=g.people.to_numpy(float)
   wk=str(g.weekday.iloc[0]);self.data[str(cam)].append((str(day),wk,t,v))
 def get(self,day,cam,wk,now,tar):
  arr=[[],[],[],[]]
  for d,w,t,v in self.data.get(cam,[]):
   if d==day:continue
   i=np.searchsorted(t,now);j=np.searchsorted(t,tar)
   if i>=len(t) or j>=len(t) or abs(t[i]-now)>2 or abs(t[j]-tar)>2:continue
   if w==wk:arr[0].append(float(v[i]));arr[1].append(float(v[j]))
   arr[2].append(float(v[i]));arr[3].append(float(v[j]))
  def m(x):return float(np.median(x)) if x else np.nan
  return m(arr[0]),m(arr[1]),m(arr[2]),m(arr[3])

def snapshot(store, rec,cam,now):
 ts,vs=store.pull(rec,cam,now)
 f=series_feat(ts,vs,now)
 f={('own_'+key):val for key,val in f.items()}
 ff=[]
 for ocam in store.cams.get(rec,[]):
  if ocam==cam:continue
  t,y=store.pull(rec,ocam,now)
  if not len(y) or now-t[-1]>15:continue
  s=series_feat(t,y,now)
  ff.append(s)
 f['cross_n']=len(ff)
 for key in ('last','slope5','slope10','slope25','relative5','relative10','relative25','halfdiff10','halfdiff25','std10'):
  vals=np.array([x.get(key,np.nan) for x in ff],float);vals=vals[np.isfinite(vals)]
  f['cross_'+key+'_median']=float(np.median(vals)) if len(vals) else np.nan
  f['cross_'+key+'_mean']=float(np.mean(vals)) if len(vals) else np.nan
 f['cross_big_rel10']=float(np.median([x['relative10'] for x in ff if x.get('last',0)>45 and np.isfinite(x.get('relative10',np.nan))])) if any(x.get('last',0)>45 and np.isfinite(x.get('relative10',np.nan)) for x in ff) else np.nan
 f['now']=float(now);f['now_sin']=np.sin(2*np.pi*now/1440);f['now_cos']=np.cos(2*np.pi*now/1440)
 f['t_from_11']=float(now-660)
 return f

def make_rows(rows,store,prof,training):
 output=[];ys=[];meta=[]
 # Avoid copying row-specific labels into feature snapshots
 for (rec,cam,now),g in rows.groupby(['recording','unit','now_minute_of_day'],sort=False,dropna=False):
  now=float(now);rec=str(rec);cam=str(cam)
  if not np.isfinite(now):continue
  f0=snapshot(store,rec,cam,now)
  if not np.isfinite(f0['own_last']) or f0['own_age']>45:continue
  for r in g.itertuples(index=False):
   tar=float(r.minute_of_day);h=tar-now
   if h<1 or h>15.01:continue
   f=dict(f0)
   f.update({'horizon':h,'target_minute':tar, 'weekday_saturday':int(str(r.weekday)=='Sat'),
             'cam':cam,'is_new_cam':int(cam in ('cam_I','cam_J','cam_K','cam_L','cam_M'))})
   same_now,same_t,any_now,any_t=prof.get(str(r.day),cam,str(r.weekday),now,tar)
   f['prior_now_same']=same_now;f['prior_target_same']=same_t
   f['prior_now_any']=any_now;f['prior_target_any']=any_t
   f['prior_slope_same']=(same_t-same_now)/max(1,h) if np.isfinite(same_t) and np.isfinite(same_now) else np.nan
   f['prior_slope_any']=(any_t-any_now)/max(1,h) if np.isfinite(any_t) and np.isfinite(any_now) else np.nan
   last=float(f0['own_last']);f['anchor']=last
   for fld in ('slope5','slope10','slope25','slope40'):
    slope=f0.get('own_'+fld,np.nan)
    f['trend_'+fld+'_target']=float(last+np.clip(slope,-3,3)*h) if np.isfinite(slope) else np.nan
   for name,a,b in [('same',same_now,same_t),('any',any_now,any_t)]:
    f[f'prior_{name}_pred_ratio']=(last+5)*(b+5)/(a+5)-5 if np.isfinite(a) and np.isfinite(b) else np.nan
    f[f'prior_{name}_pred_add']=last+(b-a) if np.isfinite(a) and np.isfinite(b) else np.nan
   f['own_vs_cross']=last/f0['cross_last_median'] if np.isfinite(f0['cross_last_median']) and f0['cross_last_median']>3 else np.nan
   # normalised slope helps transfer from high cameras to unknown cameras
   for name in ('own_slope5','own_slope10','own_slope15','own_slope25','own_slope40'):
    v=f0.get(name,np.nan)
    f[name+'_fraction']=v/max(12,last) if np.isfinite(v) else np.nan
   output.append(f)
   if training:ys.append(float(r.people))
   meta.append((str(r.recording),str(r.day),cam,int(tar),str(getattr(r,'kind','forecast')),float(getattr(r,'people',np.nan))))
 X=pd.DataFrame(output);Y=np.array(ys) if training else None
 M=pd.DataFrame(meta,columns=['recording','day','unit','minute_of_day','kind','people'])
 return X,Y,M

def training_synthetic(historical, template_targets, every=10, seed=42):
 # points after `now` are used ONLY as targets/labels, never inside snapshots.
 now_by_wk={wk:np.sort(x['now_minute_of_day'].unique()) for wk,x in template_targets[template_targets.kind=='forecast'].groupby('weekday')}
 rr=[]
 for (rec,cam),g in historical.groupby(['recording','unit'],sort=False):
  g=g.sort_values('minute_of_day');ts=g.minute_of_day.to_numpy(int);tt=g.set_index('minute_of_day')['people'].to_dict();day=str(g.day.iloc[0]);wk=str(g.weekday.iloc[0]);
  if len(ts)<25:continue
  lo=ts.min()+8;hi=ts.max()-1
  nows=set(range(int(max(lo,660)),int(min(hi,1250)),every))
  # historical events can start early morning; sample part of full recording too but limited
  if wk in now_by_wk:
   for base in now_by_wk[wk]:
    for offset in (-8,-4,0,4,8):nows.add(int(base+offset))
  for now in sorted(nows):
   if now<lo or now>hi:continue
   for h in range(1,16):
    tar=now+h
    if tar in tt:
     rr.append((str(rec),day,wk,str(cam),tar,now,h,float(tt[tar]),'forecast'))
 return pd.DataFrame(rr,columns=['recording','day','weekday','unit','minute_of_day','now_minute_of_day','horizon','people','kind'])

def train_predict(Xtr,ytr,Xva,method,seed=47,engine="lightgbm",model_path=None,time_limit=180):
 Xtr=Xtr.copy();Xva=Xva.copy()
 # We don't want a raw cam name categorization for unseen cam_E / new cameras.
 Xtr=Xtr.drop(columns=['cam']);Xva=Xva.drop(columns=['cam'])
 anchor_train=Xtr.anchor.to_numpy(float);anchor_val=Xva.anchor.to_numpy(float)
 train_features=Xtr.replace([np.inf,-np.inf],np.nan)
 val_features=Xva.replace([np.inf,-np.inf],np.nan)
 # two targets ensure same model transfers camera scale appropriately
 if method=='logratio':
  y=np.log1p(ytr)-np.log1p(anchor_train)
 elif method=='relative':
  y=(ytr-anchor_train)/np.maximum(10.,anchor_train)
 else: y=ytr
 kw=dict(n_estimators=290,learning_rate=.038,num_leaves=17,max_depth=-1,min_child_samples=45,
  colsample_bytree=.85,subsample=.9,subsample_freq=1, reg_lambda=3.0,verbose=-1,
  n_jobs=6,random_state=seed)
 if engine=='lightgbm':
  model=LGBMRegressor(objective='regression_l1',**kw)
  model.fit(train_features,y)
  p=model.predict(val_features)
 elif engine=='autogluon':
  from autogluon.tabular import TabularPredictor
  if model_path is None:raise ValueError('Explicit model_path is required for AutoGluon')
  if Path(model_path).exists():raise FileExistsError(f'Refusing to overwrite {model_path}')
  from warnings import warn
  warn('AutoGluon engine was not available for local V12 research. Its CV metrics are not the LightGBM metrics in the included report.')
  data=train_features.copy();data['__v12_target__']=y
  model=TabularPredictor(label='__v12_target__',problem_type='regression',eval_metric='mean_absolute_error',path=str(model_path))
  model.fit(data,presets='medium_quality',time_limit=int(time_limit),num_bag_folds=3,
            hyperparameters={'GBM':[{'num_leaves':17,'learning_rate':.038,'n_estimators':290}],
                             'XGB':[{'max_depth':4,'n_estimators':300,'learning_rate':.045}],
                             'RF':[{}]},verbosity=2)
  p=model.predict(val_features).to_numpy(dtype=float)
 else:raise ValueError('Unknown engine: '+str(engine))
 if method=='logratio': p=np.expm1(np.log1p(anchor_val)+np.clip(p,-1.5,1.5))
 elif method=='relative':p=anchor_val+np.clip(p,-1.5,2.0)*np.maximum(10.,anchor_val)
 return np.clip(p,0,100000),model


def checkpoint_audit():
 if not {'id','recording','unit','day','weekday','minute_of_day','people','kind','now_minute_of_day','horizon'}<=set(test.columns):
  raise ValueError('Official test schema mismatch')
 if set(sample.id.astype(str)) != set(test.loc[test.id.notna(),'id'].astype(str)):
  raise ValueError('Sample ID mismatch')
 if old.duplicated(ALL_KEYS).any() or old.v7_accepted.isna().any():
  raise ValueError('Duplicate or invalid V7 checkpoint rows')
 value=float(np.abs(old.people-old.v7_accepted).mean())
 if not np.isclose(value,15.007488956008123,atol=1e-5,rtol=0):
  raise ValueError(f'V7 checkpoint MAE mismatch: {value}')
 print('AUDIT: train',len(tr),'test_targets',len(sample),'V7_CV_rows',len(old),'V7_MAE',f'{value:.6f}')
 print('Target types:',targets.groupby('kind').size().to_dict())
 print('New-camera targets:',int(targets.unit.isin(['cam_I','cam_J','cam_K','cam_L','cam_M']).sum()))
 return value

def future_invariance_sentinel():
 """Modify every observation AFTER a forecast cutoff and re-check point-in-time snapshot."""
 context=test.loc[test.id.isna() & test.people.notna()].copy()
 focus=targets[(targets.kind=='forecast') & (targets.unit=='cam_E')].head(1)
 if focus.empty:raise AssertionError('Missing forecast target for safety sentinel')
 r=focus.iloc[0]
 rec=str(r.recording);cam=str(r.unit);now=float(r.now_minute_of_day)
 before=snapshot(Store(context),rec,cam,now)
 changed=context.copy()
 mask=(changed.recording.astype(str)==rec)&(changed.minute_of_day>now)
 if not mask.any(): raise AssertionError('Future sentinel requires later visible rows')
 changed.loc[mask,'people']+=1_000_000
 after=snapshot(Store(changed),rec,cam,now)
 if before.keys()!=after.keys():raise AssertionError('Feature columns changed after modifying the future')
 for key in before:
  a=before[key];b=after[key]
  if not (np.isclose(a,b,atol=1e-12,rtol=0,equal_nan=True)):
   raise AssertionError('Future leakage detected in feature: '+key)
 print('SAFETY PASS: altering future observations leaves forecast features unchanged')
 return True

def ci_group_bootstrap(full, alpha=.25, seed=42, B=1200):
 """Exploratory paired block bootstrap. NOT an independent CV confidence interval."""
 foc=full[full.kind=='forecast'].copy()
 foc['weekday']=np.where(foc.day.astype(str).str.endswith('11'),'Fri','Sat')
 tmp=targets[targets.kind=='forecast'][['weekday','unit','minute_of_day','now_minute_of_day']].drop_duplicates(['weekday','unit','minute_of_day'])
 foc=foc.merge(tmp,on=['weekday','unit','minute_of_day'],validate='many_to_one')
 foc['err_base']=abs(foc.people-foc.v7_accepted)
 foc['err_new']=abs(foc.people-foc.v7_accepted*(1-alpha)-foc.v12_relative*alpha)
 block=foc.groupby(['day','unit','now_minute_of_day'],sort=False).agg(before=('err_base','sum'),after=('err_new','sum'),n=('people','size')).reset_index()
 base=block.before.to_numpy(float);new=block.after.to_numpy(float)
 rng=np.random.default_rng(seed)
 inds=rng.integers(0,len(block),size=(B,len(block)))
 delta=(base[inds]-new[inds]).sum(axis=1)/base[inds].sum(axis=1)*100
 return {'clusters':int(len(block)),'sampled_blocks':int(B),'approx_percentile_05':float(np.quantile(delta,.05)),
         'approx_percentile_95':float(np.quantile(delta,.95)),
         'limitations':'Only within existing reused validation days, ignores model-selection bias and between-day uncertainty'}

def cv_validate(args):
 checkpoint_audit();future_invariance_sentinel()
 all_out=[];fold_report=[]
 for day in ['2026-09-11','2026-09-12']:
  train=tr[tr.day<day].copy();held=tr[tr.day==day].copy()
  va,visible,matched,requested=v1.prepare_validation(held,targets)
  va=va[va.kind=='forecast'].copy()
  start=time.time()
  synth=training_synthetic(train,targets,every=10)
  Xtr,ytr,_=make_rows(synth,Store(train),Profile(train),True)
  Xva,_,Mva=make_rows(va,Store(visible),Profile(train),False)
  if len(Mva)!=len(va): raise AssertionError('Forecast validation coverage decreased')
  if set(Xtr.columns)!=set(Xva.columns):raise AssertionError('Feature schema mismatch')
  Xva=Xva[Xtr.columns]
  model_path=P/'models'/'ag_v12_research'/day
  pred,_=train_predict(Xtr,ytr,Xva,'relative',engine=args.engine,model_path=model_path,time_limit=args.time_limit)
  Mva['v12_relative']=pred
  out=Mva.merge(old[ALL_KEYS+['v7_accepted']],on=ALL_KEYS,how='left',validate='one_to_one')
  if out.v7_accepted.isna().any():raise AssertionError('V7 coverage mismatch')
  all_out.append(out)
  print('Fold',day,'train_samples',len(Xtr),'forecast_targets',len(out),'time',f'{time.time()-start:.1f}s',flush=True)
  fold_report.append({'day':day,'train_samples':int(len(Xtr)),'forecast_targets':int(len(out)),
                      'baseline_forecast_mae':float(np.abs(out.people-out.v7_accepted).mean()),
                      'standalone_forecast_mae':float(np.abs(out.people-out.v12_relative).mean())})
 forecast=pd.concat(all_out,ignore_index=True)
 if len(forecast)!=2036:raise AssertionError('Expected 2036 historical forecast rows')
 both=old.merge(forecast[ALL_KEYS+['v12_relative']],on=ALL_KEYS,how='left',validate='one_to_one')
 if int(both.v12_relative.notna().sum())!=2036:raise AssertionError('Expected 2036 forecast matches')
 base_mae=float(abs(both.people-both.v7_accepted).mean())
 blend_cases={};camera_reports=[]
 for alpha in (.25,.35):
  key=f'blend_{int(alpha*100)}'
  both[key]=both.v7_accepted.where(both.v12_relative.isna(),
                                   both.v7_accepted*(1-alpha)+both.v12_relative.fillna(both.v7_accepted)*alpha)
  value=float(abs(both.people-both[key]).mean())
  by_day=[]
  for day,g in both.groupby('day'):
   before=float(abs(g.people-g.v7_accepted).mean());after=float(abs(g.people-g[key]).mean())
   by_day.append({'day':str(day),'before_mae':before,'after_mae':after,
                  'gain_pct':100*(before-after)/before})
  blend_cases[key]={'full_n':int(len(both)),'full_mae_people':value,'full_gain_pct':100*(base_mae-value)/base_mae,
                     'days':by_day,'forecast_mae_people':float(abs(forecast.people-(forecast.v7_accepted*(1-alpha)+forecast.v12_relative*alpha)).mean())}
 for (day,cam),g in forecast.groupby(['day','unit']):
  rec={'day':day,'camera':cam,'n':len(g),'baseline_mae':float(abs(g.people-g.v7_accepted).mean())}
  for alpha in (.25,.35):
   pp=g.v7_accepted*(1-alpha)+g.v12_relative*alpha
   rec[f'blend_{int(alpha*100)}_mae']=float(abs(g.people-pp).mean())
   rec[f'blend_{int(alpha*100)}_gain_pct']=100*(rec['baseline_mae']-rec[f'blend_{int(alpha*100)}_mae'])/rec['baseline_mae']
  camera_reports.append(rec)
 camdf=pd.DataFrame(camera_reports)
 selected=blend_cases['blend_25']
 verdict=('SHADOW_IMPROVED_NOT_INDEPENDENT' if all(x['gain_pct']>0 for x in selected['days'])
          and all(x>0 for x in camdf.blend_25_gain_pct) else 'KEEP_V7')
 try: uncertainty=ci_group_bootstrap(forecast,.25)
 except Exception as exc: uncertainty={'error':str(exc)}
 summary={'version':'V12 actual-data research, point-in-time normalized LightGBM',
          'warning':'Exploratory reused Sep 11 / Sep 12 validation; V7 policy used these dates. Not unbiased validation or official Kaggle score.',
          'engine':args.engine,'rows_total':int(len(both)),'forecast_rows':int(len(forecast)),
          'v7_baseline_mae_people':base_mae,'folds':fold_report,'blends':blend_cases,
          'conservative_fixed_blend':0.25,'selection_verdict':verdict,
          'target_55pct_from_v7_mae':float(base_mae*.45),'achieved_55pct':bool(selected['full_gain_pct']>=55),
          'known_new_camera_targets_unvalidated':int(targets.unit.isin(['cam_I','cam_J','cam_K','cam_L','cam_M']).sum()),
          'covered_test_target_fraction':float(len(both)/len(sample)),
          'boostrap_block_assessment':uncertainty,'future_invariance':'PASS',
          'official_kaggle_metric':None,'official_kaggle_score_verified':False,
          'no_kaggle_submission':True,'no_test_truth_read':True}
 report_dir=P/'reports';report_dir.mkdir(exist_ok=True)
 suffix='' if args.engine=='lightgbm' else '_autogluon'
 both.to_csv(report_dir/f'ag_v12_cv_predictions{suffix}.csv',index=False)
 camdf.to_csv(report_dir/f'ag_v12_by_camera{suffix}.csv',index=False)
 (report_dir/f'ag_v12_research_summary{suffix}.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 print('=== V12 RESEARCH CV RESULT ===')
 print('V7 MAE=%.6f | V12 blend .25 MAE=%.6f, gain=%.2f%% | blend .35 MAE=%.6f gain=%.2f%%' % (
  base_mae,blend_cases['blend_25']['full_mae_people'],blend_cases['blend_25']['full_gain_pct'],
  blend_cases['blend_35']['full_mae_people'],blend_cases['blend_35']['full_gain_pct']))
 print('Verdict:',verdict)
 print('No Kaggle submission, no official-score claim. Reports saved.')

def full_test_forecast(args):
 checkpoint_audit();future_invariance_sentinel()
 train=tr.copy()
 hist_store=Store(train);hist_profile=Profile(train)
 synth=training_synthetic(train,targets,every=10)
 Xtr,ytr,_=make_rows(synth,hist_store,hist_profile,True)
 context=test.loc[test.id.isna() & test.people.notna()].copy()
 only=targets[targets.kind=='forecast'].copy()
 Xte,_,Mte=make_rows(only,Store(context),hist_profile,False)
 if len(Mte)!=len(only):raise AssertionError(f'No current-camera context for {len(only)-len(Mte)} forecast targets')
 Xte=Xte[Xtr.columns]
 pred,_=train_predict(Xtr,ytr,Xte,'relative',engine=args.engine,
                      model_path=P/'models'/'ag_v12_research'/'official_test',time_limit=args.time_limit)
 Mte['people_v12_standalone']=pred
 keys=ALL_KEYS
 exported=Mte.merge(only[keys+['id']],on=keys,how='inner',validate='one_to_one')
 if len(exported)!=2660 or exported.id.duplicated().any() or not np.isfinite(exported.people_v12_standalone).all():
  raise AssertionError('Forecast candidate ID/finite check failed')
 outdir=P/'reports';outdir.mkdir(exist_ok=True)
 suffix='' if args.engine=='lightgbm' else '_autogluon'
 dest=outdir/f'ag_v12_forecast_only_predictions{suffix}.csv'
 exported[['id','people_v12_standalone']].to_csv(dest,index=False)
 print('Created 2660 forecast-only predictions:',dest)
 print('NOT a complete Kaggle submission. These forecast predictions were NOT blended with an external deployed baseline.')
 if args.base_submission:
  base=pd.read_csv(args.base_submission)
  if set(base.columns)!=set(['id','people']) or len(base)!=len(sample) or base.id.duplicated().any() or set(base.id)!=set(sample.id):
   raise ValueError('Base submission columns, ID set, count or duplicates invalid')
  base['people']=pd.to_numeric(base.people,errors='raise')
  if not np.isfinite(base.people.to_numpy(float)).all():raise ValueError('Invalid base submission predictions')
  patch=exported[['id','people_v12_standalone','unit']]
  base=base.merge(patch,on='id',how='left',validate='one_to_one')
  mask=base.people_v12_standalone.notna()
  if not args.enable_new_cameras:
   # These five units have zero historical training labels; no local zero-shot CV exists.
   mask &= ~base.unit.fillna('').isin(['cam_I','cam_J','cam_K','cam_L','cam_M'])
  alpha=float(args.blend)
  if not 0<=alpha<=1:raise ValueError('Blend weight must be [0,1]')
  base.loc[mask,'people']=(1-alpha)*base.loc[mask,'people']+alpha*base.loc[mask,'people_v12_standalone']
  base=base[['id','people']]
  if base.people.isna().any() or len(base)!=4457:raise AssertionError('Invalid local overlay')
  final=P/'submissions'/f'submission_v12_overlay_alpha{int(alpha*100)}{suffix}_local_only.csv'
  final.parent.mkdir(exist_ok=True)
  base.to_csv(final,index=False)
  print('Created local candidate:',final,'patched forecast rows',int(mask.sum()),'; NOT UPLOADED to Kaggle')
  print('New camera overlays enabled:',bool(args.enable_new_cameras),'(default is False due to missing cold-start CV)')

def main():
 parser=argparse.ArgumentParser(description='V12 point-in-time causal crowd count research; NEVER submits to Kaggle')
 parser.add_argument('--mode',choices=['audit','validate','forecast-test'],default='audit')
 parser.add_argument('--engine',choices=['lightgbm','autogluon'],default='lightgbm')
 parser.add_argument('--time-limit',type=int,default=180)
 parser.add_argument('--base-submission',type=str,default=None,
                     help='Optional pre-existing full valid submission used to overlay forecast predictions locally')
 parser.add_argument('--enable-new-cameras',action='store_true',help='Opt in to overlay on unseen cameras I-M (not locally validated; risky)')
 parser.add_argument('--blend',type=float,default=.25,
                     help='When overlaying forecast predictions on an existing submission; default 0.25')
 args=parser.parse_args()
 if args.mode=='audit':checkpoint_audit();future_invariance_sentinel()
 elif args.mode=='validate':cv_validate(args)
 elif args.mode=='forecast-test':full_test_forecast(args)

if __name__=='__main__':main()
