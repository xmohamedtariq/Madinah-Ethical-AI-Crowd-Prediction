"""Two-scale paired-camera bridges for missing blocks with a released peer."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _smooth(values, sigma=2.):
    present = np.isfinite(values)
    weighted = gaussian_filter1d(np.where(present, values, 0.), sigma, mode="constant")
    mass = gaussian_filter1d(present.astype(float), sigma, mode="constant")
    return np.divide(weighted, mass, out=np.full(values.shape, np.nan), where=mass > .12)


def _near_interpolation(values):
    """Use exact peer observations, or a hole bounded within three minutes."""
    values = values.copy()
    indices = np.flatnonzero(np.isfinite(values))
    for left, right in zip(indices[:-1], indices[1:]):
        gap = right - left
        if 1 < gap <= 3:
            f = np.arange(1, gap) / gap
            values[left+1:right] = values[left] * (1-f) + values[right] * f
    return values


def _winsor(values):
    limit = max(np.quantile(np.abs(values), .95), 1e-3)
    return np.clip(values, -limit, limit)


def _fit_pair(own, peer, own_smooth, peer_smooth, center, local_radius):
    both = np.flatnonzero(np.isfinite(own) & np.isfinite(peer))
    if len(both) < 28:
        return None
    consecutive = both[1:][np.diff(both) == 1]
    if len(consecutive) < 25:
        return None
    dy = _winsor(own[consecutive] - own[consecutive-1])
    dx = _winsor(peer[consecutive] - peer[consecutive-1])
    slope = np.dot(dx, dy) / max(np.dot(dx, dx), 1e-4)
    fast_corr = np.dot(dx, dy) / max(np.linalg.norm(dx) * np.linalg.norm(dy), 1e-4)
    y = own_smooth[both]
    x = peer_smooth[both]
    slow_corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 1e-5 and np.std(y) > 1e-5 else 0.
    if fast_corr < .3 or slow_corr < .65:
        return None

    curve = (both - center) / local_radius
    weight = np.exp(-.5 * curve**2)
    y_center = np.average(y, weights=weight)
    x_center = np.average(x, weights=weight)
    global_scale = np.std(x)
    xs = (x - x_center) / global_scale
    design = np.column_stack([xs, np.ones(len(both)), curve])
    global_slope = np.cov(x, y)[0, 1] / max(np.var(x, ddof=1), 1e-4)
    # Allow slow drift in the camera-pair offset. Regularize the slow slope
    # towards the observed full-recording relationship when local contrast is low.
    penalty = np.diag([max(weight.sum() * .03, .4), 0., .4])
    prior = np.array([global_slope * global_scale, 0., 0.])
    weighted = design.T * weight
    local_beta = np.linalg.solve(weighted @ design + penalty,
                                 weighted @ (y - y_center) + penalty @ prior)
    slow_slope = np.clip(local_beta[0] / global_scale, .4 * global_slope, 1.6 * global_slope)
    local_diff_weights = np.exp(-.5 * ((consecutive - center) / local_radius)**2)
    fast_slope = (np.dot(local_diff_weights * dx, dy) + 35 * np.mean(dx**2) * slope)
    fast_slope /= max(np.dot(local_diff_weights * dx, dx) + 35 * np.mean(dx**2), 1e-4)
    return fast_corr, slow_corr, slow_slope, fast_slope


def _edge_residual(remaining, peer_component, boundary, direction):
    indices = boundary + direction * np.arange(7)
    indices = indices[(indices >= 0) & (indices < len(remaining))]
    usable = np.isfinite(remaining[indices]) & np.isfinite(peer_component[indices])
    indices = indices[usable]
    if not len(indices):
        return np.nan
    weights = np.exp(-abs(indices-boundary) / 2.)
    values = remaining[indices] - peer_component[indices]
    if len(values) >= 4:
        middle = np.median(values)
        span = np.maximum(np.median(np.abs(values - middle)) * 3., 2.)
        values = np.clip(values, middle - span, middle + span)
    return np.average(values, weights=weights)


def _predict_gap(values, column, observations, left, right, queries, radius=42.):
    own = observations[:, column]
    own_smooth = _smooth(own)
    fraction = (queries - left) / (right - left)
    comparison = (1 - fraction) * own[left] + fraction * own[right]
    predictions = []
    weights = []
    for peer_column in range(values.shape[1]):
        if peer_column == column:
            continue
        peer = values[:, peer_column]
        peer_smooth = _smooth(peer)
        query_peer = _near_interpolation(peer)
        if not np.any(np.isfinite(query_peer[queries])):
            continue
        parameters = _fit_pair(own, peer, own_smooth, peer_smooth, (left+right)/2., radius)
        if parameters is None:
            continue
        fast_corr, slow_corr, slow_slope, fast_slope = parameters
        peer_component = slow_slope * peer_smooth + fast_slope * (query_peer - peer_smooth)
        a = _edge_residual(own, peer_component, left, -1)
        b = _edge_residual(own, peer_component, right, 1)
        if not np.isfinite(a+b):
            continue
        proposed = peer_component[queries] + (1-fraction)*a + fraction*b
        reliability = fast_corr**4 * slow_corr**2
        directly_seen = np.isfinite(peer[queries])
        predictions.append(proposed)
        weights.append(reliability * np.where(directly_seen, 1., .55))
    if not predictions:
        return comparison, np.full(len(queries), np.nan), np.zeros(len(queries))
    predictions = np.asarray(predictions)
    weights = np.asarray(weights)
    present = np.isfinite(predictions)
    total = (weights*present).sum(axis=0)
    combined = (weights*np.where(present, predictions, 0.)).sum(axis=0)
    estimate = np.divide(combined, total, out=np.full(len(total), np.nan), where=total > 0.)
    return comparison, np.maximum(0., estimate), total


def _calibrate(values, column, observed_positions, gap_lengths):
    """Remove each entire pseudo-gap before fitting offsets, slopes or weights."""
    measures = []
    own = values[:, column]
    n = len(own)
    centers = np.arange(10, n - 10, 11)
    for index, center in enumerate(centers):
        gap = gap_lengths[index % len(gap_lengths)]
        left = center - gap // 2
        right = left + gap
        if not (0 <= left < right < n and np.isfinite(own[left] + own[right])):
            continue
        interval = np.arange(left+1, right)
        queries = interval[observed_positions[interval]]
        if len(queries) < max(3., (right-left)/2.):
            continue
        fit = values.copy()
        fit[interval, column] = np.nan
        comparison, estimate, reliability = _predict_gap(fit, column, fit, left, right, queries)
        supported = np.isfinite(estimate)
        if supported.sum() < len(queries) / 2:
            continue
        y = own[queries][supported]
        manual = comparison[supported]
        prediction = estimate[supported]
        # Normalize within block, then score blocks equally; a surge in one
        # historical block must not choose the weight for the whole recording.
        scale = max(np.mean(abs(y-manual)), np.mean(abs(y)) * .06, 1.)
        measures.append([np.mean(abs(y - ((1-f)*manual + f*prediction))) / scale
                         for f in [0., .35, .65, .90]])
    if len(measures) < 4:
        return 0.
    errors = np.mean(measures, axis=0)
    best = int(np.argmin(errors))
    if errors[best] >= errors[0] * .98:
        return 0.
    return [0., .35, .65, .90][best]


def paired_predictions(released, targets):
    estimates = {}
    for recording, target_group in targets.loc[targets.kind.eq("missing")].groupby("recording"):
        context = released.loc[released.recording.eq(recording)]
        if context.empty:
            continue
        frame = context.pivot_table(index="minute_of_day", columns="unit", values="people")
        lo, hi = int(frame.index.min()), int(frame.index.max())
        values = frame.reindex(np.arange(lo, hi+1)).to_numpy(float)
        units = {unit: position for position, unit in enumerate(frame.columns)}
        for unit, queries in target_group.groupby("unit"):
            if unit not in units:
                continue
            column = units[unit]
            observed = np.isfinite(values[:, column])
            times = np.flatnonzero(observed)
            q = queries.minute_of_day.to_numpy(int) - lo
            insertion = np.searchsorted(times, q)
            valid = (insertion > 0) & (insertion < len(times))
            groups = [i for i in np.unique(insertion[valid]) if 3 <= times[i]-times[i-1] <= 30]
            if not groups:
                continue
            gap_lengths = [int(times[i] - times[i-1]) for i in groups]
            weight = _calibrate(values, column, observed, gap_lengths)
            if weight <= 0.:
                continue
            for i in groups:
                mask = insertion == i
                comparison, estimate, reliability = _predict_gap(values, column, values,
                                                                  times[i-1], times[i], q[mask])
                chosen_ids = queries.id.to_numpy()[mask]
                for id_, y, strength in zip(chosen_ids, estimate, reliability):
                    if np.isfinite(y):
                        estimates[id_] = float(y), float(weight * min(1., strength / .15))
    return estimates


def predict_fold(train, released, targets, baseline, reference=None):
    result = (baseline if reference is None else reference).copy(deep=True)
    estimates = paired_predictions(released, targets)
    if estimates:
        prediction = dict(zip(result.id, result.people))
        for id_, (value, weight) in estimates.items():
            prediction[id_] = (1-weight) * prediction[id_] + weight * value
        result["people"] = result.id.map(prediction)
    return result
