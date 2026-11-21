"""Recording-disjoint-trained local-linear-trend RTS gap reconstruction."""
from functools import lru_cache
import numpy as np
import pandas as pd

_CONFIGS = [(r, ql, qs) for r in (0.05, 0.2, 0.5) for ql in (0.005, 0.05) for qs in (0.0001, 0.005)]


def _smooth(values, config):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if finite.sum() < 5:
        return values.copy()
    diff = np.diff(values[finite])
    med = np.median(diff)
    scale = 1.4826 * np.median(np.abs(diff-med))
    variance = max(scale**2, 0.25)
    r_mult, ql_mult, qs_mult = config
    observation_var = max(r_mult * variance, 1e-4)
    Q = variance * np.array([[ql_mult + qs_mult/3, qs_mult/2], [qs_mult/2, qs_mult]])
    F = np.array([[1., 1.], [0., 1.]])
    H = np.array([[1., 0.]])
    n = len(values)
    means = np.zeros((n, 2)); covs = np.zeros((n, 2, 2))
    filtered_m = np.zeros_like(means); filtered_p = np.zeros_like(covs)
    first = int(np.flatnonzero(finite)[0])
    m = np.array([values[first], 0.])
    P = np.diag([variance*10, variance*10])
    I = np.eye(2)
    for i in range(n):
        if i:
            m = F @ m
            P = F @ P @ F.T + Q
        means[i], covs[i] = m, P
        if finite[i]:
            S = float(H @ P @ H.T + observation_var)
            K = (P @ H.T).ravel() / S
            m = m + K * (values[i] - float(H @ m))
            P = (I - np.outer(K, H.ravel())) @ P
        filtered_m[i], filtered_p[i] = m, P
    smooth_m = filtered_m.copy(); smooth_p = filtered_p.copy()
    for i in range(n-2, -1, -1):
        C = filtered_p[i] @ F.T @ np.linalg.pinv(covs[i+1])
        smooth_m[i] = filtered_m[i] + C @ (smooth_m[i+1] - means[i+1])
        smooth_p[i] = filtered_p[i] + C @ (smooth_p[i+1] - covs[i+1]) @ C.T
    return smooth_m[:, 0]


def _choose_config(train, camera):
    trials = []
    for config in _CONFIGS:
        errors = []
        for _, rec in train.loc[train.unit.eq(camera)].groupby("recording", sort=True):
            s = rec.groupby("minute_of_day", sort=True).people.mean()
            t = s.index.to_numpy(int); y = s.to_numpy(float)
            if len(y) < 80:
                continue
            lengths = (6, 10, 16)
            for length in lengths:
                max_start = len(y) - length - 2
                if max_start < 10:
                    continue
                starts = np.unique(np.linspace(5, max_start, 4).astype(int))
                for start in starts:
                    stop = start + length
                    if t[stop-1] - t[start] > length + 2:
                        continue
                    masked = y.copy(); masked[start:stop] = np.nan
                    pred = _smooth(masked, config)[start:stop]
                    errors.extend(np.abs(pred-y[start:stop]).tolist())
        trials.append((float(np.mean(errors)) if errors else np.inf, config))
    return min(trials, key=lambda x: x[0])[1] if trials else _CONFIGS[4]


def predict_missing(train, released, targets, baseline):
    out = baseline.set_index("id").people.astype(float).copy()
    missing = targets.loc[targets.kind.eq("missing")]
    for recording, qrec in missing.groupby("recording", sort=True):
        context = released.loc[released.recording.eq(recording)]
        for camera, query in qrec.groupby("unit", sort=True):
            observed = context.loc[context.unit.eq(camera)].groupby("minute_of_day", sort=True).people.mean()
            if len(observed) < 8:
                continue
            low, high = int(observed.index.min()), int(observed.index.max())
            grid = pd.Series(np.nan, index=np.arange(low, high+1), dtype=float)
            grid.loc[observed.index.astype(int)] = observed.to_numpy(float)
            config = _choose_config(train, camera)
            smooth = _smooth(grid.to_numpy(float), config)
            positions = query.minute_of_day.to_numpy(int)-low
            valid = (positions >= 0) & (positions < len(smooth))
            out.loc[query.id.to_numpy()[valid]] = np.maximum(0., smooth[positions[valid]])
    return out.rename("people").reset_index()
