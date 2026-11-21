"""Reproduce selected Madinah model on its official test inputs with a support gate.

Dry-run only: never calls Kaggle, never modifies champion source or inputs.
The output is a candidate prediction file requiring human approval before any upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path, help='Directory containing candidate.py and forecast/, hidden/, missing/')
    ap.add_argument('--train', required=True, type=Path)
    ap.add_argument('--test', required=True, type=Path)
    ap.add_argument('--sample', required=True, type=Path, help='Official id/order template')
    ap.add_argument('--baseline', required=True, type=Path, help='Archived production reference (NOT the official sample)')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--audit', required=True, type=Path)
    args = ap.parse_args()

    for p in [args.source / 'candidate.py', args.train, args.test, args.sample, args.baseline]:
        if not p.exists():
            ap.error(f'Missing required file: {p}')
    if args.output.resolve() in {p.resolve() for p in [args.train, args.test, args.sample, args.baseline]}:
        ap.error('Output must never overwrite input.')
    if args.output.exists() or args.audit.exists():
        ap.error('Refusing to overwrite an existing dry-run or audit. Choose fresh paths.')

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    baseline = pd.read_csv(args.baseline)
    expected_target_columns = {'id', 'people', 'kind', 'recording', 'unit', 'minute_of_day'}
    assert expected_target_columns <= set(test), f'Bad test columns: {expected_target_columns - set(test)}'
    assert list(sample.columns) == ['id', 'people'], f'Unexpected sample columns: {list(sample.columns)}'
    assert list(baseline.columns) == ['id', 'people'] and baseline.id.is_unique and set(baseline.id) == set(sample.id), 'Baseline IDs or columns differ from official sample'
    assert baseline.people.notna().all() and np.isfinite(baseline.people).all() and (baseline.people >= 0).all()
    assert sample.id.is_unique and sample.id.notna().all() and sample.people.notna().all()
    assert np.isfinite(sample.people).all() and (sample.people >= 0).all()
    assert test.id.notna().sum() == len(sample) == 4457, 'Competition target count differs from expected archive count'
    assert test.loc[test.id.notna(), 'id'].is_unique
    assert set(sample.id) == set(test.id.dropna())
    assert test.loc[test.id.notna(), 'people'].isna().all(), 'Test has labels for target IDs'
    assert test.loc[test.id.isna(), 'people'].notna().all(), 'Released rows have unexpected missing counts'
    target = test.loc[test.id.notna()].copy()
    released = test.loc[test.id.isna()].copy()
    assert target.kind.notna().all()

    known = set(train.unit.unique())
    permitted = target.loc[target.unit.isin(known)].drop(columns=['people']).copy()
    unsupported = target.loc[~target.unit.isin(known)].copy()
    assert len(permitted) + len(unsupported) == len(sample)
    # Restrict evidence to historical camera population to avoid introducing
    # completely unseen peer features into a model validated on known cameras.
    permitted_context = released.loc[released.unit.isin(known)].copy()
    initial = baseline.set_index('id').loc[permitted.id, ['people']].reset_index()

    sys.path.insert(0, str(args.source.resolve()))
    from candidate import predict_fold

    print(f'TARGETS={len(target)} SUPPORTED={len(permitted)} UNSUPPORTED={len(unsupported)}', flush=True)
    print('KNOWN_CAMERA_UNITS=' + ','.join(sorted(known)), flush=True)
    print('UNSUPPORTED_UNIT_BREAKDOWN=' + json.dumps(unsupported.unit.value_counts().to_dict()), flush=True)
    start = time.monotonic()
    result = predict_fold(train, permitted_context, permitted, initial)
    elapsed = round(time.monotonic() - start, 3)
    assert list(result.columns) == ['id', 'people']
    assert result.id.is_unique and set(result.id) == set(permitted.id)
    result = result.set_index('id').people
    assert np.isfinite(result.to_numpy(float)).all() and (result >= 0).all()
    output = sample[['id']].copy()
    output['people'] = output.id.map(baseline.set_index('id').people)
    eligible = output.id.isin(permitted.id)
    output.loc[eligible, 'people'] = output.loc[eligible, 'id'].map(result).to_numpy()
    assert output.shape == (4457, 2)
    assert output.id.equals(sample.id)
    assert output.people.notna().all()
    assert np.isfinite(output.people).all() and (output.people >= 0).all()
    original = baseline.set_index('id').people
    final = output.set_index('id').people
    absdiff = (final - original).abs()
    changed = absdiff > 1e-10
    assert not changed.loc[unsupported.id].any(), 'UNSUPPORTED CAMERA CHANGED'
    assert not changed.loc[target[target.kind == 'hidden'].id].any(), 'HIDDEN CHANGED (champion baseline must be frozen)'
    diag = target[['id', 'recording', 'kind', 'unit']].copy()
    diag['changed'] = diag.id.map(changed)
    diag['absolute_change'] = diag.id.map(absdiff)
    diag['source_supported'] = diag.unit.isin(known)
    by_kind = {}
    for kind, g in diag.groupby('kind', sort=True):
        by_kind[kind] = {
            'target_count': int(len(g)),
            'changed_count': int(g.changed.sum()),
            'unsupported_count': int((~g.source_supported).sum()),
            'max_absolute_change': round(float(g.absolute_change.max()), 6),
            'mean_absolute_change': round(float(g.absolute_change.mean()), 6),
        }
    by_rec = {}
    for rec, g in diag.groupby('recording', sort=True):
        by_rec[rec] = {'count':int(len(g)), 'changed':int(g.changed.sum()),
                       'unsupported':int((~g.source_supported).sum())}
    audit = {
        'status': 'PRODUCTION_DRY_RUN_VALID_FORMAT_NOT_OFFICIAL_KAGGLE_SCORE',
        'human_approval_required': True,
        'submitted_to_kaggle': False,
        'official_score': None,
        'note': 'Baseline is archived V13 production reference, not the official sample. No guarantee this beats the established official 0.729. Archived historical validation is not a live-test estimate.',
        'input_hashes': {k:sha(p) for k,p in [('train',args.train),('test',args.test),('sample',args.sample),('archived_baseline',args.baseline),('candidate_py',args.source/'candidate.py')]},
        'elapsed_seconds': elapsed,
        'targets':int(len(target)),
        'supported':int(len(permitted)),
        'unsupported':int(len(unsupported)),
        'unsupported_camera_ids':sorted(unsupported.unit.unique()),
        'changed':int(changed.sum()),
        'unchanged':int((~changed).sum()),
        'overall_max_absolute_change':round(float(absdiff.max()),6),
        'overall_mean_absolute_change':round(float(absdiff.mean()),6),
        'by_kind':by_kind,
        'by_recording':by_rec,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output,index=False, float_format='%.17g')
    args.audit.write_text(json.dumps(audit,indent=2,ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['status','elapsed_seconds','targets','supported','unsupported','changed','unchanged','by_kind']},indent=2), flush=True)
    print(f'WROTE_DRY_RUN={args.output}',flush=True)
    print(f'WROTE_AUDIT={args.audit}',flush=True)

if __name__ == '__main__':
    main()
