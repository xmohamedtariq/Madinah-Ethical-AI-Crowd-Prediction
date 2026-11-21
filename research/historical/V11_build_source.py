#!/usr/bin/env python3
"""Build a causal-safe V11 candidate from frozen V10 predictions.

Only `hidden` (NOT hidden-future) predictions are modified. Hidden is
permitted to use released data on both sides of the target; this code uses
only the V10 predictions. Forecast and hidden-future rows remain byte-exact.

The smoothing operation was selected by exploratory comparisons on
reused development folds; it has NOT been tested independently or on Kaggle.
"""
from pathlib import Path
import hashlib, csv, io, json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'submission_v10_temporal_ensemble.csv'
TARGET=ROOT/'submission_v11_hidden_secondpass.csv'
DATA=ROOT/'work/v2/data/official'
META=ROOT/'V11_integrity.json'
EXPECTED_SOURCE_SHA='9e1237ac258cca66b3ee4ad5cd3916a41c12c163dcf81c878aa0be9701171efc'
EXPECTED_DATA_SHA={
    'train.csv':'3210a0b33212a60a888977848ec09f5f89bc0954bfd3b88e61fb9d840537f6f6',
    'test.csv':'6f2c6400b02964f3f37b912a5d6dbe085b28e7070a8094bdfad918e57766e6b9',
    'sample_submission.csv':'22e1d6a23de7095c18be2e1e2d8dfbccaaa4010e647feee608e10f16ee798471',
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def second_pass_hidden(group, base_values, passes=2):
    """Two low-pass passes within contiguous target runs, no gap-crossing.

    All inputs are already-computed estimates for the same camera/day.
    This function must not be applied to any future (forecast/HF) tasks.
    """
    g=group.sort_values('minute_of_day')
    ts=g.minute_of_day.to_numpy(dtype=int)
    vv=g.id.map(base_values).to_numpy(dtype=float)
    if not np.isfinite(vv).all() or len(vv)!=len(g):
        raise ValueError('Missing or non-finite base predictions')
    result=vv.copy()
    breaks=np.r_[0,np.flatnonzero(np.diff(ts)>1)+1,len(ts)]
    for start,end in zip(breaks[:-1],breaks[1:]):
        n=end-start
        width=min(31,n if n%2 else n-1)
        if width<5:continue
        smooth=vv[start:end]
        for _ in range(passes):
            smooth=savgol_filter(smooth,window_length=width,polyorder=2,mode='interp')
        result[start:end]=np.maximum(0.,smooth)
    return dict(zip(g.id,result))

def generate():
    if sha256(SOURCE)!=EXPECTED_SOURCE_SHA:
        raise ValueError('V10 source checksum mismatch')
    for name,dig in EXPECTED_DATA_SHA.items():
        if sha256(DATA/name)!=dig:
            raise ValueError('Official data checksum mismatch '+name)
    sample=pd.read_csv(DATA/'sample_submission.csv')
    test=pd.read_csv(DATA/'test.csv')
    orig=pd.read_csv(SOURCE,float_precision='round_trip')
    assert list(orig.columns)==['id','people'] and list(sample.columns)==['id','people']
    assert len(orig)==len(sample)==4457 and orig.id.equals(sample.id) and orig.id.is_unique
    assert np.isfinite(orig.people).all() and orig.people.ge(0).all()
    masked=test[test.id.notna()].copy()
    assert len(masked)==4457 and masked.id.is_unique and set(masked.id)==set(sample.id)
    assert masked.people.isna().all()
    assert masked.kind.value_counts().to_dict()=={'forecast':2660,'hidden':849,'hidden-future':484,'missing':464}
    values=dict(zip(orig.id,orig.people))
    altered={}
    for _,g in masked.loc[masked.kind.eq('hidden')].groupby(['recording','unit'],sort=False):
        altered.update(second_pass_hidden(g,values))
    assert len(altered)==849
    assert set(altered)==set(masked.loc[masked.kind.eq('hidden'),'id'])
    # Preserve untouched source lines byte-for-byte (including all future task rows).
    source_lines=SOURCE.read_bytes().splitlines(keepends=True)
    output=[source_lines[0]];unchanged=0
    for line in source_lines[1:]:
        ident=next(csv.reader([line.decode('utf-8').rstrip('\r\n')]))[0]
        if ident in altered:
            eol='\r\n' if line.endswith(b'\r\n') else '\n'
            buffer=io.StringIO(newline='')
            csv.writer(buffer,lineterminator=eol).writerow([ident,format(float(altered[ident]),'.17g')])
            output.append(buffer.getvalue().encode('utf-8'))
        else:
            output.append(line);unchanged+=1
    assert unchanged==3608
    TARGET.write_bytes(b''.join(output))
    got=pd.read_csv(TARGET,float_precision='round_trip')
    assert got.id.equals(sample.id) and got.id.is_unique
    assert list(got.columns)==['id','people'] and len(got)==4457
    assert np.isfinite(got.people).all() and (got.people>=0).all()
    check=masked[['id','kind']].merge(orig.rename(columns={'people':'old'}),on='id').merge(got.rename(columns={'people':'new'}),on='id')
    dif=check.old.to_numpy()!=check.new.to_numpy()
    changes=check.loc[dif].groupby('kind').size().to_dict()
    assert set(changes).issubset({'hidden'})
    assert ((check.kind!='hidden')&(check.old!=check.new)).sum()==0
    if not np.any(dif):raise AssertionError('Transformation made no difference')
    info={
        'candidate':TARGET.name,'candidate_sha256':sha256(TARGET),
        'parent':SOURCE.name,'parent_sha256':EXPECTED_SOURCE_SHA,
        'row_count':len(got),'column_names':['id','people'],
        'matches_official_sample_ids_and_order':True,'unique_ids':True,
        'all_values_finite_nonnegative':True,
        'changed_predictions_total':int(dif.sum()),
        'changes_by_kind':changes,
        'unchanged_source_lines':unchanged,
        'forecast_rows_untouched':2660,'hidden_future_rows_untouched':484,'missing_rows_untouched':464,
        'operation':'two Savitzky-Golay poly2 passes with odd window up to 31, per contiguous hidden camera run',
        'features_used_for_new_operation':'V10 hidden estimates only',
        'post_now_observations_used_for_future_prediction':False,
        'public_kaggle_score':None,
        'evaluation_status':'DEVELOPMENT_ONLY_REUSED_RECORDINGS',
        'independent_validation':False,
        'target_0_400_verified':False,
    }
    META.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    return info

if __name__=='__main__':
    print(json.dumps(generate(),indent=2))
