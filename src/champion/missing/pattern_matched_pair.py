"""Two-scale bridge with inner validation under the actual peer outage pattern."""
from __future__ import annotations

import numpy as np

from .paired_bridge import _predict_gap


def _comparison(values, own_column, left, right, queries):
    # Conservative inner control using both exact boundaries and nearby means.
    own = values[:, own_column]
    fraction = (queries-left)/(right-left)
    exact = (1-fraction)*own[left] + fraction*own[right]
    a = own[max(0,left-3):left+1]
    b = own[right:min(len(own), right+4)]
    medians = (1-fraction)*np.nanmedian(a) + fraction*np.nanmedian(b)
    return .65*exact + .35*medians


def _validate(values, column, left, right, actual_level):
    """Fit no release-derived weight before masking all pseudo-target cells."""
    length = right-left
    columns = np.arange(values.shape[1])
    peers = columns[columns != column]
    template = ~np.isfinite(values[left:right+1, peers])
    own = values[:, column]
    exact = np.flatnonzero(np.isfinite(own))
    candidate = exact[::5]
    errors = []
    sample_weight = []
    for start in candidate:
        stop = start+length
        if stop >= len(own) or not np.isfinite(own[stop]):
            continue
        interior = np.arange(start+1,stop)
        query = interior[np.isfinite(own[interior])]
        if len(query) < max(3., (length-1)*.55):
            continue
        masked = values.copy()
        masked[interior,column] = np.nan
        peer_pattern = masked[start:stop+1, peers].copy()
        peer_pattern[template] = np.nan
        masked[start:stop+1, peers] = peer_pattern
        _, estimates, strengths = _predict_gap(masked, column, masked, start, stop, query)
        # This cutoff selects strongly connected observations rather than trying
        # a new unconstrained regression for every camera and recording.
        valid = np.isfinite(estimates) & (strengths >= .14)
        if valid.sum() < max(2., .30*len(query)):
            continue
        base = _comparison(masked, column, start, stop, query)
        base = base[valid]
        pred = estimates[valid]
        truth = own[query][valid]
        normalizer = max(np.mean(abs(base-truth)), np.mean(truth)*.06, 1.)
        levels = (own[start]+own[stop])/2.
        contextual = np.exp(-.5*(np.log((levels+3.)/(actual_level+3.))/.9)**2)
        sample_weight.append(contextual)
        errors.append([np.mean(abs(truth-((1-f)*base+f*pred)))/normalizer
                       for f in [0.,.25,.50,.75]])
    if len(errors) < 4:
        return 0.
    weighted = np.average(errors, weights=sample_weight, axis=0)
    choice = int(np.argmin(weighted))
    return [0.,.25,.50,.75][choice] if weighted[choice] < .98*weighted[0] else 0.


def _predictions(released, targets):
    forecasts = {}
    for recording, group in targets.loc[targets.kind.eq("missing")].groupby("recording"):
        observations = released.loc[released.recording.eq(recording)]
        if observations.empty:
            continue
        grid = observations.pivot_table(index="minute_of_day",columns="unit",values="people")
        origin, latest = int(grid.index.min()), int(grid.index.max())
        values = grid.reindex(np.arange(origin,latest+1)).to_numpy(float)
        units = {name: position for position, name in enumerate(grid.columns)}
        for unit, query in group.groupby("unit"):
            if unit not in units:
                continue
            col = units[unit]
            times = np.flatnonzero(np.isfinite(values[:,col]))
            absolute = query.minute_of_day.to_numpy(int)-origin
            insertions = np.searchsorted(times,absolute)
            gaps = [i for i in np.unique(insertions)
                    if 0 < i < len(times) and 3 <= times[i]-times[i-1] <= 25]
            for i in gaps:
                left, right = times[i-1], times[i]
                active = insertions==i
                _, estimates, confidence = _predict_gap(values, col, values, left, right,
                                                         absolute[active])
                supported = np.isfinite(estimates) & (confidence >= .14)
                if not supported.any():
                    continue
                current_level = (values[left,col]+values[right,col])/2.
                weight = _validate(values,col,left,right,current_level)
                if weight <= 0.:
                    continue
                ids = query.id.to_numpy()[active][supported]
                for id_, estimate in zip(ids,estimates[supported]):
                    forecasts[id_] = float(estimate), weight
    return forecasts


def predict_fold(train,released,targets,baseline,reference=None):
    result = (baseline if reference is None else reference).copy(deep=True)
    estimates = _predictions(released,targets)
    if estimates:
        p = dict(zip(result.id,result.people))
        for id_,(count,weight) in estimates.items():
            p[id_] = (1-weight)*p[id_] + weight*count
        result["people"] = result.id.map(p)
    return result
