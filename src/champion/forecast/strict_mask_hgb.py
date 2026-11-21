"""HGB forecast with historical release masks independent of current future rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


from . import frozen_forecast as forecast


def synthetic_masks(query, mode):
    """Use target geometry and fixed assumptions, never released values or timestamps."""
    units = sorted(query.unit.unique())
    origins = query.now_minute_of_day.dropna().astype(int).unique()
    minutes = np.arange(1440)
    dense = ((origins[:, None] - minutes[None, :] >= 0)
             & (origins[:, None] - minutes[None, :] <= 8)).any(axis=0)
    future_blocks = ((minutes[None, :] - origins[:, None] >= 1)
                     & (minutes[None, :] - origins[:, None] <= 15)).any(axis=0)
    rng = np.random.default_rng(12481)
    masks = []
    for unit in units:
        if mode == "random":
            keep = (rng.random(1440) < .33) & ~future_blocks
            keep |= dense & (rng.random(1440) < .88)
        else:
            keep = ~future_blocks & (minutes % 4 == 0)
            keep |= dense
        masks.append(pd.DataFrame({"unit": unit, "minute_of_day": minutes[keep]}))
    return pd.concat(masks, ignore_index=True)



def tiled_masks(prefix, query, now):
    """Extend only prefix mask patterns to other clock times in historical recordings."""
    rng = np.random.default_rng(55177)
    origins = np.sort(query.now_minute_of_day.dropna().astype(int).unique())
    past_origins = origins[origins <= now]
    first = max(0, int(origins[0])-30)
    end = min(1440, int(origins[-1])+43)
    clock = np.arange(now+1, end)
    to_cover = origins[np.minimum(np.searchsorted(origins, clock), len(origins)-1)] - clock
    age = np.clip(to_cover, -45, 65)
    masks = []
    for unit, rows in prefix.groupby("unit", sort=True):
        seen = np.zeros(1440, dtype=bool)
        seen[rows.minute_of_day.astype(int)] = True
        sample = past_origins[:,None] - age[None,:]
        sample_ok = (sample >= first) & (sample <= now)
        hits = seen[np.clip(sample, 0, 1439)] & sample_ok
        # The expected prior geometry uses a dense eight-minute block followed by a hole.
        prior = np.where((age >= 0) & (age <= 8), .86,
                         np.where((age >= 39) & (age <= 52), .08, .33))
        expected = (hits.sum(axis=0) + 1.2 * prior) / (sample_ok.sum(axis=0) + 1.2)
        synthetic = clock[rng.random(len(clock)) < expected]
        minutes = np.r_[np.flatnonzero(seen), synthetic]
        masks.append(pd.DataFrame({"unit": unit, "minute_of_day": minutes}))
    return pd.concat(masks, ignore_index=True) if masks else prefix

def train_model(train, mask, query):
    cfg = forecast.DEFAULT
    X, y, view_weights, visible = forecast.training_rows(train, mask, query, cfg)
    if len(X) < 100:
        return None
    last = X["last"].to_numpy()
    scale = np.sqrt(last + 10.)
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=cfg["iterations"], max_leaf_nodes=cfg["leaves"],
        max_depth=7, learning_rate=.055, l2_regularization=cfg["l2"],
        min_samples_leaf=35, early_stopping=False, random_state=7331,
    )
    model.fit(X, (y-last)/scale, sample_weight=view_weights*scale)
    return model, X.columns, visible, len(X)


def predict_origin(trained, prefix, rows):
    if trained is None:
        return {}
    model, columns, visible, _ = trained
    history = forecast.History(prefix, visible)
    features = []
    ids = []
    for row in rows.itertuples(index=False):
        f = history.features(row.unit, int(row.now_minute_of_day), int(row.horizon), row.weekday, True)
        if f is None or f["age"] > 20:
            continue
        features.append(f)
        ids.append(row.id)
    if not features:
        return {}
    val = pd.DataFrame(features).reindex(columns=columns)
    last = val["last"].to_numpy()
    preds = np.maximum(0., last + np.sqrt(last+10.) * model.predict(val))
    return dict(zip(ids, preds))


def predict_config(train, released, targets, baseline, blend=.25, settings="grid"):
    result = baseline[["id", "people"]].copy()
    query = targets[targets.kind.eq("forecast")]
    if query.empty:
        return result
    changes = {}
    n_rows = []
    with threadpool_limits(limits=1):
        if settings not in {"prefix", "prefix_local", "tiled"}:
            trained = train_model(train, synthetic_masks(query, settings), query)
            if trained is not None:
                n_rows.append(trained[3])
        for (recording, now), group in query.groupby(["recording", "now_minute_of_day"], sort=False):
            prefix = released.loc[(released.recording.eq(recording)) & (released.minute_of_day <= now)]
            if settings in {"prefix", "prefix_local", "tiled"}:
                scope = group if settings == "prefix_local" else query
                mask = tiled_masks(prefix, scope, int(now)) if settings == "tiled" else prefix
                trained = train_model(train, mask, scope)
                if trained is not None:
                    n_rows.append(trained[3])
            changes.update(predict_origin(trained, prefix, group))
    values = result.id.map(changes)
    result["people"] += (blend * (values - result.people)).fillna(0.)
    print("STRICT_HGB", settings, "train_n", n_rows, "predict_n", len(changes), flush=True)
    return result


def predict_fold(train, released, targets, baseline):
    return predict_config(train, released, targets, baseline)
