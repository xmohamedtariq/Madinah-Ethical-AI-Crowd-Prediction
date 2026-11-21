"""A two-sided, softly aligned historical event-shape prior for missing rows."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

_CLOCK = np.arange(24 * 60)
_SHIFTS = np.arange(-12., 13.)
_SPEEDS = np.array([.85, 1., 1.15])
_WARP = np.asarray(np.meshgrid(_SHIFTS, _SPEEDS)).reshape(2, -1).T


def _bank(frame, units):
    p = frame.pivot_table(index="minute_of_day", columns="unit", values="people")
    v = p.reindex(index=_CLOCK, columns=units).to_numpy(float)
    return np.sqrt(np.maximum(0., v) + 1.)


def _smooth(values, sigma=2.):
    good = np.isfinite(values)
    v = gaussian_filter1d(np.where(good, values, 0.), sigma, axis=0, mode="constant")
    count = gaussian_filter1d(good.astype(float), sigma, axis=0, mode="constant")
    return np.divide(v, count, out=np.full_like(v, np.nan), where=count > .35)


def _data_grids(train, released, units):
    historical = [_smooth(_bank(group, units)) for _, group in train.groupby("recording")]
    current = _bank(released, units)
    return historical, current


def _local_design(magnitude, time, historical_curve):
    # Remove each trace's weighted local linear trend before aligning waveform.
    # This makes the timing estimate depend on curvature and pulses rather than
    # an absolute count scale that can change between recordings.
    w = np.exp(-.5 * (time/24.)**2)
    x = time
    center = np.average(x, weights=w)
    dx = x-center
    ybar = np.average(magnitude, weights=w)
    y_slope = np.sum(w * dx * (magnitude-ybar)) / max(np.sum(w * dx**2), 1e-8)
    residual = magnitude - (ybar + dx*y_slope)
    zbar = (historical_curve*w[None, :]).sum(axis=1) / w.sum()
    z_slope = ((historical_curve-zbar[:, None])*w[None, :]) @ dx / max(np.sum(w*dx**2), 1e-8)
    shapes = historical_curve - zbar[:, None] - z_slope[:, None]*dx
    return w, residual, shapes


def _estimate(historical, current, unit, left, right, query):
    mid = (left+right)/2.
    lower, upper = max(0, left-20), min(len(current), right+21)
    around = np.arange(lower, upper)
    raw = current[:, unit]
    if (not np.isfinite(raw[left] + raw[right])) or len(around) < 12:
        return None
    fraction = (query-left) / (right-left)
    own_lin = (1-fraction)*raw[left] + fraction*raw[right]
    raw_observed = np.flatnonzero(np.isfinite(raw) & (abs(_CLOCK-mid) <= 42.))
    if len(raw_observed) < 10 or len(raw_observed[raw_observed <= left]) < 3 or len(raw_observed[raw_observed >= right]) < 3:
        return None
    # Smooth only permitted observations on each side, including for synthetic
    # gaps. Original masked counts are unavailable to this function.
    smooth = _smooth(current[lower:upper], sigma=1.2)
    core_q = np.r_[left, query, right]
    likelihoods = []
    corrections = []
    for donor in historical:
        core_lookup = mid + _WARP[:, :1] + _WARP[:, 1:] * (core_q-mid)
        ix0 = np.floor(core_lookup).astype(int)
        ix0 = np.clip(ix0, 0, len(donor)-2)
        f = np.clip(core_lookup-ix0, 0., 1.)
        aligned_core = (1-f)*donor[ix0, unit] + f*donor[ix0+1, unit]
        eligible = np.isfinite(aligned_core).all(axis=1)
        if not eligible.any():
            continue
        peer_loss = np.zeros(len(_WARP))
        peer_count = np.zeros(len(_WARP))
        own_slope = np.ones(len(_WARP))
        own_variability = 0.
        for peer in range(current.shape[1]):
            positions = around[np.isfinite(current[around, peer])]
            if len(positions) < 10:
                continue
            if sum(positions < mid) < 3 or sum(positions > mid) < 3:
                continue
            values = smooth[positions-lower, peer]
            grids = mid + _WARP[:, :1] + _WARP[:, 1:]*(positions-mid)
            index0 = np.clip(np.floor(grids).astype(int), 0, len(donor)-2)
            eta = np.clip(grids-index0, 0., 1.)
            template = (1-eta)*donor[index0, peer] + eta*donor[index0+1, peer]
            usable = np.isfinite(template).all(axis=1) & eligible
            if not usable.any():
                continue
            w, residual, shapes = _local_design(values, positions-mid, template[usable])
            amplitude = np.sqrt(np.average(residual**2, weights=w))
            if amplitude < .12:
                continue
            loading = np.average(values, weights=w) / np.maximum((template[usable]*w).sum(axis=1)/w.sum(), .5)
            floor = .1 + .3*amplitude
            deviation = residual[None, :] - shapes*loading[:, None]
            loss = np.sqrt((deviation**2*w).sum(axis=1)/w.sum())/(floor + amplitude)
            peer_weight = 3. if peer == unit else 1.
            peer_loss[usable] += peer_weight*np.minimum(loss, 4.)
            peer_count[usable] += peer_weight
            if peer == unit:
                own_variability = amplitude
                own_slope[usable] = np.clip(loading, .4, 2.5)
        eligible &= peer_count >= 3. 
        if own_variability < .12 or not eligible.any():
            continue
        data_loss = peer_loss[eligible]/peer_count[eligible]
        warp = _WARP[eligible]
        regularizer = .12*abs(warp[:, 0]/12.) + .10*abs(warp[:, 1]-1.)/.15
        uncertainty = data_loss+regularizer
        core = aligned_core[eligible]
        linear = core[:, :1]*(1-fraction)+core[:, -1:]*fraction
        curvature = own_slope[eligible, None]*(core[:, 1:-1]-linear)
        likelihoods.append(uncertainty)
        corrections.append(curvature)
    if not likelihoods:
        return None
    losses = np.concatenate(likelihoods)
    adjustments = np.concatenate(corrections)
    # Keep influence local to well-aligned warps without selecting a single
    # exact donor offset on noisy outside-gap observations.
    best = np.argpartition(losses, min(9, len(losses)-1))[:10]
    weight = np.exp(-8.*(losses[best]-losses[best].min()))
    correction = np.average(adjustments[best], weights=weight, axis=0)
    uncertainty = np.average((adjustments[best]-correction)**2, weights=weight, axis=0)
    confidence = np.clip(1.2 - losses[best].min(), 0., 1.)
    confidence *= np.exp(-np.sqrt(np.mean(uncertainty)) / .7)
    capped = np.clip(correction, -.3*own_lin, .3*own_lin)
    return np.maximum(0., own_lin**2-1.), capped, float(confidence)


def _alpha(historical, values, column, lengths):
    present = np.isfinite(values[:, column])
    results = []
    for i, left in enumerate(np.flatnonzero(present)[::11]):
        gap = lengths[i%len(lengths)]
        right = left+gap
        if right >= len(present) or not present[right]:
            continue
        blocked = np.arange(left+1,right)
        queries = blocked[present[blocked]]
        if len(queries) < max(3, (gap-1)*.6):
            continue
        copy = values.copy()
        copy[blocked,column] = np.nan
        features = _estimate(historical, copy, column, left, right, queries)
        if features is None:
            continue
        base, residual, confidence = features
        target = values[queries,column]**2 - 1.
        normalizer = max(np.mean(abs(base-target)), np.mean(abs(target))*.06, 1.)
        candidates = [np.maximum(0., (np.sqrt(base+1.)+confidence*f*residual)**2-1.)
                      for f in [0., .5, 1., 1.5]]
        results.append([np.mean(abs(x-target))/normalizer for x in candidates])
    if len(results) < 4:
        return .5
    errors = np.mean(results, axis=0)
    best = int(np.argmin(errors))
    return [0., .5, 1., 1.5][best] if errors[best] < errors[0]*.99 else 0.


def _predictions(train, released, targets):
    estimates = {}
    units = sorted(set(train.unit))
    lookup = {unit: i for i,unit in enumerate(units)}
    target_groups = targets.loc[targets.kind.eq("missing")].groupby("recording")
    for recording, group in target_groups:
        observations = released.loc[released.recording.eq(recording)]
        if observations.empty:
            continue
        historical, current = _data_grids(train, observations, units)
        for unit, queries in group.groupby("unit"):
            if unit not in lookup:
                continue
            col = lookup[unit]
            t = np.flatnonzero(np.isfinite(current[:,col]))
            q = queries.minute_of_day.to_numpy(int)
            index = np.searchsorted(t, q)
            gaps = [j for j in np.unique(index) if 0 < j < len(t) and 4 <= t[j]-t[j-1] <= 22]
            if not gaps:
                continue
            lengths = [t[j]-t[j-1] for j in gaps]
            strength = _alpha(historical,current,col,lengths)
            if strength <= 0.:
                continue
            for j in gaps:
                take = index == j
                features = _estimate(historical,current,col,t[j-1],t[j],q[take])
                if features is None:
                    continue
                _, offset, confidence = features
                estimates.update({id_: float(x) for id_, x in
                                  zip(queries.id.to_numpy()[take], offset*strength*confidence)})
    return estimates


def predict_fold(train, released, targets, baseline, reference=None):
    result = (baseline if reference is None else reference).copy(deep=True)
    if not targets.kind.eq("missing").any() or train.empty:
        return result
    deltas = _predictions(train, released, targets)
    if deltas:
        offset = result.id.map(deltas).fillna(0.).to_numpy(float)
        transformed = np.sqrt(result.people.to_numpy(float)+1.)
        active = offset != 0.
        result.loc[active, "people"] = np.maximum(0., (transformed[active]+offset[active])**2-1.)
    return result
