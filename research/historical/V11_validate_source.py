#!/usr/bin/env python3
"""Audit V11 against unchanged legacy and matched fold labels.

CAUTION: These validation days were reused in V6..V10 optimization.
Not an independent holdout and cannot establish future Kaggle results.
"""
import json, hashlib
from pathlib import Path
import numpy as np,pandas as pd
from V10_build_source import fix_hidden,fix_future
from V11_build_source import second_pass_hidden,generate,sha256
ROOT=Path(__file__).resolve().parent
REPORTS=ROOT/'exp/reports/v7_hidden_temporal'
TASKS=['forecast','hidden','hidden-future','missing']

def metric(df,pred):
    r={}
    for kind,g in df.groupby('kind'):
        sel=g.index.to_numpy();y=g.people_true.to_numpy(); ref=g.people_ref.to_numpy()
        mae=float(np.mean(np.abs(pred[sel]-y)))
        denom=float(np.mean(np.abs(ref-y)))
        r[kind]={'count':len(g),'mae':mae,'ref_mae':denom,'ratio':mae/denom}
    assert set(r)==set(TASKS)
    return {'aggregate':float(np.mean([r[k]['ratio'] for k in TASKS])),'tasks':r}

def compare():
    folds=[]
    for proto in ['matched','legacy']:
        for rec in ['20260911_1100','20260912_1100']:
            df=pd.read_csv(REPORTS/f'{proto}_{rec}_predictions.csv').reset_index(drop=True)
            v10=df.prediction.to_numpy(float).copy()
            for _,g in df[df.kind.eq('hidden')].groupby(['recording','unit']):
                x=g.copy();x['baseline']=x.prediction
                v10[g.index]=g.id.map(fix_hidden(x))
            for _,g in df[df.kind.eq('hidden-future')].groupby(['recording','unit','now_minute_of_day']):
                x=g.copy();x['baseline']=x.prediction
                v10[g.index]=g.id.map(fix_future(x))
            v11=v10.copy()
            for _,g in df[df.kind.eq('hidden')].groupby(['recording','unit']):
                vx=dict(zip(df.id,v10))
                v11[g.index]=g.id.map(second_pass_hidden(g,vx,passes=2)).to_numpy(float)
            k=df.kind.eq('hidden').to_numpy()
            assert np.array_equal(v10[~k],v11[~k])
            m10=metric(df,v10);m11=metric(df,v11)
            folds.append({'protocol':proto,'recording':rec,'V10':m10,'V11':m11,
                'delta_V11_minus_V10':m11['aggregate']-m10['aggregate'],
                'improved':bool(m11['aggregate']<m10['aggregate'])})
    mean={}
    for name in ['V10','V11']:
        mean[name]=float(np.mean([x[name]['aggregate'] for x in folds if x['protocol']=='matched']))
        mean[f'{name}_legacy']=float(np.mean([x[name]['aggregate'] for x in folds if x['protocol']=='legacy']))
    mean['relative_reduction_vs_V10']=1-mean['V11']/mean['V10']
    mean['relative_reduction_vs_V6']=1-mean['V11']/0.682947036
    # Pre-defined conservative gate: improved matched Friday+Saturday AND both legacy dates.
    mean['all_four_reused_comparisons_improved']=bool(all(x['improved'] for x in folds))
    report={'summary':mean,'folds':folds,'labelled_matched_coverage':3719,'total_test_targets':4457,
        'hidden_test_targets':849,'training_recordings_in_bench':['20260904_1105','20260905_0426','20260905_1124','20260905_1900','20260911_1100','20260912_1100'],
        'critical_limitation':'Repeatedly tuned on same two dated validation recordings; no independent out-of-sample score.',
        'public_leaderboard_unverified':True}
    (ROOT/'V11_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    info=generate();res=compare()
    assert res['summary']['all_four_reused_comparisons_improved']
    print(json.dumps({'candidate_sha256':info['candidate_sha256'],'changed_predictions':info['changed_predictions_total'],'summary':res['summary'],'fold_deltas':[(x['protocol'],x['recording'],x['delta_V11_minus_V10']) for x in res['folds']]},indent=2))
