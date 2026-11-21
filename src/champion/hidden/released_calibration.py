"""Use causal visible-camera proxy forecasts to gate hidden-camera innovations."""

import numpy as np
import pandas as pd

from .innovations import model_prediction
from .lagged_transfer import Observations


def _relatedness(train, anchor, peers):
    """Historical short-window change agreement, after within-day centering."""
    corrs = {}
    values = {p: [] for p in peers}
    count_levels = {p: [] for p in [anchor, *peers]}
    for _, group in train.groupby("recording", sort=True):
        z = group.pivot_table(index="minute_of_day", columns="unit", values="people")
        if anchor not in z:
            continue
        r = np.log(z + 4)
        for p in count_levels:
            if p in r:
                count_levels[p].append(float(r[p].median()))
        d = r.subtract(r.rolling(5, min_periods=2).mean())
        for p in peers:
            if p in d and d[[anchor, p]].dropna().shape[0] >= 20:
                values[p].append(float(d[anchor].corr(d[p])))
    base_level = np.median(count_levels[anchor]) if count_levels[anchor] else np.nan
    for p in peers:
        center = np.median(count_levels[p]) if count_levels[p] else np.nan
        corr = np.nanmedian(values[p]) if values[p] else 0.0
        level_similarity = np.exp(-.5 * abs(base_level - center))
        corrs[p] = (.25 + max(0, corr) ** 1.25) * level_similarity
    return corrs


def _pseudo_rows(context, stride=4):
    units = sorted(context.unit.unique())
    pieces = []
    for p in units:
        z = context[context.unit == p].groupby("minute_of_day", sort=True).people.mean()
        weeks = context.loc[context.unit == p, "weekday"].iloc[0]
        times = z.index.to_numpy(int)
        log = np.log(z.to_numpy(float) + 4)
        observed = Observations(times, log)
        for h in (2, 5, 8, 11, 14):
            now = times[times % stride == 0]
            future = now + h
            anchor = observed.average(now - 2, now)
            actual = observed.average(future, future)
            # Support is local to the pseudo-example. A recording-wide
            # row-count gate would let future rows enable earlier examples.
            good = np.isfinite(anchor) & np.isfinite(actual)
            if good.any():
                pieces.append(pd.DataFrame({
                    "unit": p,
                    "weekday": weeks,
                    "minute_of_day": future[good],
                    "now_minute_of_day": now[good],
                    "horizon": h,
                    "level": anchor[good],
                    "change": actual[good] - anchor[good],
                }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _proxies(train, context):
    history = _pseudo_rows(context)
    results = []
    if not len(history):
        return history
    for unit, q in history.groupby("unit", sort=True):
        if not train.unit.eq(unit).any():
            continue
        # A visible proxy is fully hidden from its own prediction model.
        based_on = context.loc[context.unit != unit]
        pred = model_prediction(train, based_on, q, causal=True, model_type="ridge")
        results.append(q.assign(predicted=pred))
    if not results:
        return history.iloc[0:0]
    return pd.concat(results, ignore_index=True).dropna(subset=["predicted"])


def _calibrate(model, horizon, cutoff, proxies, weights, adapt):
    prior = model
    if proxies.empty:
        return 0.18 * prior
    data = proxies[proxies.minute_of_day <= cutoff]
    if len(data) < 12:
        return 0.18 * prior
    age = cutoff - data.minute_of_day.to_numpy(float)
    offset = horizon - data.horizon.to_numpy(float)
    pred = data.predicted.to_numpy(float)
    obs = data.change.to_numpy(float)
    weight = data.unit.map(weights).to_numpy(float) * np.exp(-age / 47 - 0.5 * (offset / 4.5) ** 2)
    scale = np.sum(weight)
    if scale < 0.8:
        return 0.18 * prior
    # As an initial check, allow current-day proxy evidence to shut off
    # deterioration against persistence before correcting any hidden series.
    gain = np.average(np.abs(obs) - np.abs(obs - pred), weights=weight)
    bias = 0.0
    slope = 1.0
    if adapt:
        bias = np.clip(np.average(obs - pred, weights=weight), -0.15, 0.15)
        direction = np.sum(weight * pred * (obs - bias))
        slope = np.clip((0.045 + direction) / (0.045 + np.sum(weight * pred * pred)), 0.0, 1.25)
        gain = np.average(np.abs(obs) - np.abs(obs - (slope * pred + bias)), weights=weight)
    strength = np.clip(.12 + 5 * gain, 0.0, .50)
    return float(strength * (slope * prior + bias))


def predict_profile(train, released, targets, baseline, adapt):
    out = baseline.set_index("id").people.copy()
    for rec, queries in targets[targets.kind == "hidden-future"].groupby("recording", sort=True):
        context = released[released.recording == rec]
        pseudo = _proxies(train, context)
        historic_units = sorted(train.unit.unique())
        for unit, q in queries.groupby("unit", sort=True):
            strengths = _relatedness(train, unit, [p for p in historic_units if p != unit])
            pred = model_prediction(train, context, q, causal=True, model_type="ridge")
            q = q.assign(new=pred)
            q = q[np.isfinite(q.new)]
            if not len(q):
                continue
            gain = np.array([
                _calibrate(r.new, r.horizon, r.now_minute_of_day, pseudo, strengths, adapt)
                for r in q.itertuples(index=False)
            ])
            original = out.loc[q.id].to_numpy(float)
            out.loc[q.id] = np.maximum(0, np.exp(np.log(original + 4) + gain) - 4)
    return out.rename("people").reset_index()
