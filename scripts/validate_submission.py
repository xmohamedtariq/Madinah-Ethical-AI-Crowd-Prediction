"""Validate a prediction CSV against an official Kaggle submission template.

Does not require training labels or upload anything.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def validate(sample_path: Path, prediction_path: Path) -> dict:
    sample = pd.read_csv(sample_path, dtype={'id':str})
    submission = pd.read_csv(prediction_path, dtype={'id':str})
    assert list(sample.columns) == ['id','people'], f'Sample columns: {list(sample.columns)}'
    assert list(submission.columns) == ['id','people'], f'Submission columns: {list(submission.columns)}'
    assert len(submission) == len(sample), 'Row count differs from sample'
    assert submission.id.notna().all() and submission.id.is_unique, 'IDs are missing or duplicated'
    assert submission.id.equals(sample.id), 'Submission IDs or row order do not match sample'
    vals = pd.to_numeric(submission.people, errors='coerce').to_numpy(dtype=float)
    assert np.isfinite(vals).all(), 'Nonfinite or missing prediction value'
    assert (vals >= 0).all(), 'Negative count prediction'
    return {'status':'PASS', 'rows':len(vals),'unique_ids':len(submission.id.unique()),
            'min_prediction':float(np.min(vals)), 'max_prediction':float(np.max(vals))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample', type=Path, required=True)
    ap.add_argument('--submission', type=Path, required=True)
    args = ap.parse_args()
    print(validate(args.sample,args.submission))

if __name__ == '__main__':
    main()
