#!/usr/bin/env python3
r"""Madinah competition: prevent repeating an effectively identical submission.
Read-only. Does not generate submissions, modify source data, or call Kaggle.

PowerShell example:
.\.venv\Scripts\python.exe .\last_three_attempts_guard.py --reference .\best_0729.csv --candidate .\new_candidate.csv --test .\verified_605700\official_data\test.csv
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(path, label):
    frame=pd.read_csv(path)
    if frame.columns.tolist()!=['id','people']:
        raise ValueError(f'{label}: expected exactly id,people; got {list(frame)}')
    if frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError(f'{label}: missing or duplicate IDs')
    vals=pd.to_numeric(frame.people, errors='coerce').to_numpy(float)
    if np.any(~np.isfinite(vals)) or np.any(vals<0):
        raise ValueError(f'{label}: nonfinite or negative predictions')
    return frame


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',required=True,type=Path)
    ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--test',type=Path,help='Optional official test.csv to break down changed rows by task')
    ap.add_argument('--tolerance',type=float,default=1e-6,help='Absolute numerical equality threshold')
    args=ap.parse_args()
    if args.tolerance<=0: ap.error('tolerance must be positive')
    a=load(args.reference, 'REFERENCE');b=load(args.candidate, 'CANDIDATE')
    if len(a)!=4457 or len(b)!=4457 or set(a.id)!=set(b.id):
        raise ValueError('Candidates must have identical IDs and exactly 4,457 entries')
    if not a.id.equals(b.id):
        b=b.set_index('id').loc[a.id].reset_index()
    diff=b.people.to_numpy(float)-a.people.to_numpy(float)
    changed=np.abs(diff)>args.tolerance
    out={'reference':str(args.reference),'candidate':str(args.candidate),
         'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
         'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
         'id_match':True,'row_count':len(a),'tolerance':args.tolerance,
         'effectively_changed':int(changed.sum()),'effectively_unchanged':int((~changed).sum()),
         'max_abs_difference':float(np.max(np.abs(diff))),'mean_abs_difference':float(np.mean(np.abs(diff))),
         'rmse_difference':float(np.sqrt(np.mean(diff**2))),
         'verdict':'EFFECTIVELY_IDENTICAL_DO_NOT_REUPLOAD' if not np.any(changed) else 'DISTINCT_CANDIDATE_NOT_ACCURACY_VALIDATED'}
    if args.test:
        test=pd.read_csv(args.test)
        targets=test.loc[test.id.notna(),['id','recording','unit','kind']].copy()
        if targets.id.duplicated().any() or set(targets.id)!=set(a.id):
            raise ValueError('Test target IDs do not match submission IDs')
        targets['delta']=targets.id.map(dict(zip(a.id,diff)))
        targets['changed']=np.abs(targets.delta)>args.tolerance
        out['changed_by_recording_and_kind']=[{'recording':rec,'kind':kind,'targets':int(len(g)),
            'changed':int(g.changed.sum()),'max_abs_difference':float(np.max(abs(g.delta))),
            'mean_abs_difference':float(np.mean(abs(g.delta)))}
            for (rec,kind),g in targets.groupby(['recording','kind'])]
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 3 if out['verdict']=='EFFECTIVELY_IDENTICAL_DO_NOT_REUPLOAD' else 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print(f'VALIDATION_ERROR: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(2)
