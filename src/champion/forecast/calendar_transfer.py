"""Causal forecast using nearby clock-time templates and a shared phase estimate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


SETTINGS = {
    "short_clock": (9, 2.5, 12.0, 0.0),
    "long_clock": (30, 3.5, 18.0, 0.0),
    "scaled": (22, 3.5, 18.0, 0.65),
}
OFFSETS = np.arange(-8, 9)
FUTURE = np.arange(1, 16)


def pivot(rows):
    return rows.pivot_table(index="minute_of_day", columns="unit", values="people").reindex(range(1440))


def historical_bank(train):
    bank = {}
    for rec, rows in train.groupby("recording", sort=False):
        raw = pivot(rows)
        original = raw.to_numpy()
        template = np.sqrt(np.maximum(original, 0) + 4.)
        good = np.isfinite(template)
        smooth = gaussian_filter1d(np.where(good, template, 0), 1.2, axis=0)
        confidence = gaussian_filter1d(good.astype(float), 1.2, axis=0)
        smooth = np.divide(smooth, confidence, out=np.full_like(smooth, np.nan), where=confidence >= .40)
        bank[rec] = (rows.day.iloc[0], rows.weekday.iloc[0], raw, pd.DataFrame(smooth, columns=raw.columns))
    return bank


def fit_source(values, sample_times, prefix, cfg):
    window, shift_scale, residual_scale, variance_weight = cfg
    delta = sample_times[-1] - sample_times
    weight = np.exp(-delta / (3. + .28 * window))[:, None]
    weight *= np.where(delta <= window, 1., 0.)[:, None]

    candidate = values[sample_times[None, :] + OFFSETS[:, None], :]
    actual = np.broadcast_to(prefix, candidate.shape)
    observed = np.isfinite(actual) & np.isfinite(candidate)
    wc = observed * weight[None, :, :]
    n = wc.sum(axis=1)
    safe_n = np.maximum(n, 1e-9)
    mu_x = (np.where(observed, candidate, 0) * wc).sum(axis=1) / safe_n
    mu_y = (np.where(observed, actual, 0) * wc).sum(axis=1) / safe_n
    xc = np.where(observed, candidate - mu_x[:, None, :], 0)
    yc = np.where(observed, actual - mu_y[:, None, :], 0)
    sx = (wc * xc**2).sum(axis=1) / safe_n
    xy = (wc * xc * yc).sum(axis=1) / safe_n
    # Gain 1 is a fixed prior for innovations on a variance-stabilized count scale.
    gain = np.clip(1 + variance_weight * (xy - sx) / (sx + 1.6), .50, 1.8)
    mismatch = (np.minimum((yc - gain[:, None, :] * xc) ** 2, 12.) * wc).sum(axis=1) / safe_n
    enough = (observed.sum(axis=1) >= 4) & (np.max(np.where(observed, sample_times[None, :, None], -1000), axis=1) >= sample_times[-1] - 5)
    importance = np.minimum(mu_y, 22.) * enough
    ncam = enough.sum(axis=1)
    err = np.divide((mismatch * importance).sum(axis=1), importance.sum(axis=1),
                    out=np.full(len(OFFSETS), np.inf), where=ncam >= 2)
    err /= .85
    err += (OFFSETS / shift_scale) ** 2 / 2.

    anchor_weight = np.exp(-delta / 1.7)[None, :, None] * observed
    desired = np.where(observed, yc - gain[:, None, :] * xc, 0)
    short_residual = (anchor_weight * desired).sum(axis=1) / np.maximum(anchor_weight.sum(axis=1), 1e-9)
    future_index = sample_times[-1] + OFFSETS[:, None] + FUTURE[None, :]
    future_x = values[future_index, :]
    prediction = (mu_y[:, None, :]
                  + gain[:, None, :] * (future_x - mu_x[:, None, :])
                  + short_residual[:, None, :] * np.exp(-FUTURE[None, :, None]/residual_scale))
    return err, np.where(enough[:, None, :], np.maximum(0, prediction)**2 - 4., np.nan)


def calendar_origin(bank, prefix, now, weekday, cfg, columns):
    window, _, _, _ = cfg
    if now - window - max(OFFSETS) < 0 or now + 15 + max(OFFSETS) >= 1440:
        return pd.DataFrame(np.nan, index=FUTURE, columns=columns)
    minute = np.arange(now-window, now+1)
    values = np.sqrt(np.maximum(prefix.reindex(index=minute,columns=columns).to_numpy(), 0) + 4.)
    all_costs = []
    all_preds = []
    for _, source_weekday, _, source in bank.values():
        source_values = source.reindex(columns=columns).to_numpy()
        if np.isfinite(source_values[now-window:now+16]).sum() < 15:
            continue
        cost, pred = fit_source(source_values, minute, values, cfg)
        if source_weekday != weekday:
            cost += .9
        all_costs.append(cost)
        all_preds.append(pred)
    if not all_costs:
        return pd.DataFrame(np.nan, index=FUTURE, columns=columns)
    costs = np.concatenate(all_costs)
    forecasts = np.concatenate(all_preds, axis=0)
    if not np.isfinite(costs).any():
        return pd.DataFrame(np.nan, index=FUTURE, columns=columns)
    weights = np.exp(-(costs - np.min(costs)))[:, None, None] * np.isfinite(forecasts)
    den = weights.sum(axis=0)
    result = np.divide(np.nansum(weights * forecasts, axis=0), den,
                       out=np.full(den.shape, np.nan), where=den>1e-6)
    return pd.DataFrame(np.maximum(result, 0), index=FUTURE, columns=columns)


def choose_settings(bank, target_cameras):
    """Hold out complete days and mimic the short released prefix at each origin."""
    history = {}
    for name, conf in SETTINGS.items():
        errors = []
        for day, weekday, raw, _ in bank.values():
            eligible = {key: value for key, value in bank.items() if value[0] != day}
            if not eligible:
                continue
            positions = raw.index[raw.notna().any(axis=1)]
            cameras = sorted(set(raw.columns) & set(target_cameras))
            if len(cameras) < 3:
                continue
            for now in range(int(positions[0])+25, int(positions[-1])-14, 17):
                prefix = raw.loc[:now, cameras].copy()
                # Short dense observation segment plus older sparse anchors.
                removed = prefix.index[(now-prefix.index > 8) & (prefix.index % 4 != 0)]
                prefix.loc[removed] = np.nan
                prediction = calendar_origin(eligible, prefix, now, weekday, conf, cameras).to_numpy()
                true = raw.loc[now+1:now+15, cameras].to_numpy()
                last = prefix.iloc[-9:].ffill().iloc[-1].to_numpy()
                both = np.isfinite(prediction) & np.isfinite(true) & np.isfinite(last)
                if both.sum() >= 30:
                    e = np.abs(prediction - true)[both].mean()
                    b = np.abs(last[None, :] - true)[both].mean()
                    errors.append(e / max(b, 1))
        if errors:
            history[name] = float(np.mean(errors))
    choice = min(history, key=history.get) if history else "short_clock"
    print("CALENDAR_INNER", choice, history, flush=True)
    return choice


def predict_config(train, released, targets, baseline, blend=.45, settings=None):
    result = baseline[["id", "people"]].copy()
    query = targets.loc[targets.kind.eq("forecast")]
    if query.empty:
        return result
    bank = historical_bank(train)
    target_cameras = sorted(query.unit.unique())
    if settings is None:
        settings = choose_settings(bank, target_cameras)
    conf = SETTINGS[settings]
    changes = {}
    for (recording, now), group in query.groupby(["recording", "now_minute_of_day"], sort=False):
        now = int(now)
        released_prefix = released.loc[(released.recording.eq(recording)) & (released.minute_of_day <= now)]
        columns = sorted(released_prefix.unit.unique())
        if len(columns) < 2:
            continue
        prefix = pivot(released_prefix)
        p = calendar_origin(bank, prefix, now, group.weekday.iloc[0], conf, columns)
        for row in group.itertuples(index=False):
            lead = int(row.horizon)
            if row.unit in p and lead in p.index and np.isfinite(value := p.at[lead,row.unit]):
                changes[row.id] = value
    vals = result.id.map(changes)
    result["people"] = result.people + (blend * (vals - result.people)).fillna(0.)
    print("CALENDAR_COVERAGE", len(changes), len(query), flush=True)
    return result


def predict_fold(train, released, targets, baseline):
    return predict_config(train, released, targets, baseline)
