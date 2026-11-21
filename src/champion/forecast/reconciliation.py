"""Forecast post-processing based on prefix-stable cross-camera count ratios."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .calendar_transfer import SETTINGS as CALENDAR_SETTINGS
from .calendar_transfer import calendar_origin, historical_bank, pivot


PRIOR = 5.0
SETTINGS = (0.25, 0.55, 1.0)


def robust_slope(x, y, weights, prior=0., penalty=.02):
    w = weights / weights.sum()
    cx = x - np.dot(w, x)
    cy = y - np.dot(w, y)
    return np.clip((np.dot(w, cx * cy) + penalty * prior) / (np.dot(w, cx * cx) + penalty), -.8, .8)


def pair_relationships(prefix, now, cameras, history):
    edges = []
    for u, v in combinations(cameras, 2):
        xy = prefix.loc[now-80:now, [u, v]].dropna()
        r = xy.loc[now-15:now]
        if len(r) < 5 or r.index[-1] < now - 3:
            continue
        xp, yp = np.log(r[u].to_numpy() + PRIOR), np.log(r[v].to_numpy() + PRIOR)
        d = xp - yp
        pooled = (xp + yp) * .5
        w = np.exp((r.index.to_numpy() - now) / 7.)
        # A fit learned on an allowed historical recording acts as a weak prior.
        old = history.get((u, v))
        if old is None:
            if len(xy) < 14:
                continue
            delta = np.log(xy + PRIOR).diff()
            if delta[u].corr(delta[v]) < .55:
                continue
            old = 0.0
        slope = robust_slope(pooled, d, w, old, .04)
        anchor = np.average(pooled, weights=w)
        recent_ratio = np.average(d, weights=w)
        residual = d - recent_ratio - slope * (pooled - anchor)
        ratio_noise = np.sqrt(np.average(residual**2, weights=w))
        if ratio_noise > .18:
            continue
        stability = np.clip((.18 - ratio_noise) / .12, 0., 1.)
        edges.append((u, v, recent_ratio, slope, anchor, stability))
    return edges


def historical_relationships(bank, cameras):
    relationships = {}
    for u, v in combinations(cameras, 2):
        all_x = []
        all_y = []
        scores = []
        for _, _, raw, _ in bank.values():
            if u not in raw or v not in raw:
                continue
            xy = np.log(raw[[u, v]] + PRIOR)
            d1 = xy.diff().dropna()
            if len(d1) < 45:
                continue
            d6 = (xy - xy.shift(6)).dropna()
            scores.append((len(d1), d1[u].corr(d1[v])))
            all_x.extend(d6.mean(axis=1))
            all_y.extend(d6[u] - d6[v])
        if not scores:
            continue
        n, corr = np.asarray(scores).T
        if np.average(corr, weights=n) < .32:
            continue
        x = np.asarray(all_x)
        y = np.asarray(all_y)
        relationships[(u, v)] = robust_slope(x, y, np.ones(len(x)), penalty=.015) if len(x) else 0.
    return relationships


def reconcile(forecast, edges, strength):
    result = forecast.copy()
    for idx, row in forecast.iterrows():
        values = row.dropna()
        cameras = list(values.index)
        if len(cameras) < 2:
            continue
        coord = np.log(values.to_numpy() + PRIOR)
        rel = {u: j for j,u in enumerate(cameras)}
        # Weight the base by approximate count precision on the logarithmic scale.
        precision = np.clip(np.sqrt(values.to_numpy() + PRIOR)/8., .5, 2.)
        system = np.diag(precision)
        rhs = np.zeros(len(coord))
        for u, v, anchor_ratio, slope, anchor_level, stability in edges:
            if u not in rel or v not in rel:
                continue
            j, k = rel[u], rel[v]
            pooled = (coord[j] + coord[k]) * .5
            desired = anchor_ratio + slope * (pooled - anchor_level)
            offset = np.clip(desired-(coord[j]-coord[k]), -.25, .25)
            contrast = np.zeros(len(coord))
            contrast[j] = 1
            contrast[k] = -1
            w = strength * stability
            system += w * np.outer(contrast, contrast)
            rhs += w * contrast * offset
        correction = np.linalg.solve(system, rhs)
        result.loc[idx, cameras] = np.maximum(np.exp(coord + correction)-PRIOR, 0.)
    return result


def choose_strength(bank, cameras):
    """Select projection strength on masked causal pseudo-origins, with a zero option."""
    scores = {w: [] for w in (0., *SETTINGS)}
    for day, weekday, raw, _ in bank.values():
        eligible = {k: v for k,v in bank.items() if v[0] != day}
        available = sorted(set(raw.columns) & set(cameras))
        edges_from_train = historical_relationships(eligible, available)
        positions = raw.index[raw.notna().any(axis=1)]
        for now in range(int(positions[0])+26, int(positions[-1])-14, 17):
            prefix = raw.loc[:now, available].copy()
            removed = prefix.index[(now-prefix.index > 8) & (prefix.index % 4 != 0)]
            prefix.loc[removed] = np.nan
            edges = pair_relationships(prefix, now, available, edges_from_train)
            if not edges:
                continue
            indices = np.arange(1,16)
            path = calendar_origin(eligible, prefix, now, weekday, CALENDAR_SETTINGS["long_clock"], available)
            last = prefix.iloc[-9:].ffill().iloc[-1].to_numpy()
            past = prefix.iloc[-9:].mean().to_numpy()
            naive = .75 * last + .25 * past
            # Mix seasonal and anchored pseudo-forecasts. True future counts are excluded.
            proxy = pd.DataFrame(.60 * path.to_numpy() + .40 * naive, index=indices, columns=available)
            proxy = proxy.fillna(pd.DataFrame(np.tile(naive, (15, 1)), index=indices, columns=available))
            true = raw.loc[now+1:now+15, available].to_numpy()
            common = np.isfinite(proxy.to_numpy()) & np.isfinite(true)
            if common.sum() < 30:
                continue
            loss = np.abs(proxy.to_numpy()-true)[common].mean()
            for strength in scores:
                see = reconcile(proxy, edges, strength).to_numpy()
                scores[strength].append(np.abs(see-true)[common].mean() / max(loss, 1.))
    averages = {w: np.mean(loss) for w, loss in scores.items() if loss}
    choice = min(averages, key=averages.get) if averages else 0.
    print("RECONCILIATION_INNER", choice, averages, flush=True)
    return choice


def predict_config(train, released, targets, baseline, settings=None):
    result = baseline[["id", "people"]].copy()
    query = targets.loc[targets.kind.eq("forecast")].merge(result, on="id", validate="one_to_one")
    if query.empty:
        return result
    bank = historical_bank(train)
    cameras = sorted(query.unit.unique())
    history = historical_relationships(bank, cameras)
    strength = choose_strength(bank, cameras) if settings is None else float(settings)
    if strength <= 0:
        return result
    adjustments = {}
    for (recording, now), group in query.groupby(["recording", "now_minute_of_day"], sort=False):
        now = int(now)
        prefix = pivot(released.loc[released.recording.eq(recording) & (released.minute_of_day <= now)])
        path = group.pivot_table(index="horizon", columns="unit", values="people")
        cameras = sorted(set(path.columns)&set(prefix.columns))
        edges = pair_relationships(prefix, now, cameras, history)
        if edges:
            updated = reconcile(path, edges, strength)
            for row in group.itertuples(index=False):
                adjustments[row.id] = updated.at[row.horizon, row.unit]
    print("RECONCILIATION_ROWS", len(adjustments), len(query), flush=True)
    updates = result.id.map(adjustments)
    result["people"] = updates.fillna(result.people)
    return result


def predict_fold(train, released, targets, baseline):
    return predict_config(train, released, targets, baseline)
