"""Transfer short-timescale innovations while retaining the frozen count levels."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .lagged_transfer import (
    Observations, featurize, masked_history, observation_bank, peer_presence,
)


def center_delta(bank, peers, t, now, weekday, causal):
    features = featurize(bank, peers, t, now, weekday, causal)
    extra = []
    for name in peers:
        if name not in bank:
            extra.extend(np.full(len(t), np.nan) for _ in range(5))
            continue
        series = bank[name]
        cut = now if causal else t
        center, age = series.near(cut, cut, causal)
        log = lambda v: np.log(v + 4)
        if causal:
            # Relative positions reveal a fast jump without giving a stale value
            # from before the previous common blind interval the same weight.
            front = series.average(cut - 2, cut)
            back = series.average(cut - 7, cut - 4)
            prior = series.average(cut - 18, cut - 11)
            extra.extend((log(center) - log(front), log(front) - log(back),
                          log(back) - log(prior), log(front) - log(prior), age))
        else:
            background = series.average(t - 9, t + 9)
            lead, _ = series.near(t + 2, t, False)
            lag, _ = series.near(t - 4, t, False)
            lag2, _ = series.near(t - 8, t, False)
            extra.extend((log(center) - log(background), log(lead) - log(background),
                          log(lag) - log(background), log(lag2) - log(background), age))
    return np.column_stack([features, *extra])


def examples(rec, unit, peers, causal):
    own = rec[rec.unit == unit].groupby("minute_of_day", sort=True).people.mean()
    if len(own) < 30:
        return None
    times = own.index.to_numpy(int)
    counts = own.to_numpy(float)
    log = np.log(counts + 4)
    arr = Observations(times, log)
    features = []
    values = []
    weights = []
    weekday = rec.weekday.iloc[0]
    for shift in (0, 8, 16, 24, 32, 40, 48):
        phase = (times - shift) % 55
        bank = observation_bank(masked_history(rec, unit, peers, shift), peers)
        if causal:
            use = (phase >= 25) & (phase <= 39)
            t = times[use]
            now = t - phase[use] + 24
            center = arr.average(now - 2, now)
            confidence = np.sqrt(t - now)
        else:
            use = (phase <= 24) & (phase % 2 == 0)
            t = times[use]
            now = t
            center = arr.average(t - 9, t + 9)
            confidence = np.ones(len(t))
        if not len(t):
            continue
        X = center_delta(bank, peers, t, now, weekday, causal)
        y = (log[use] - center) / confidence
        last_col = 3 + np.arange(len(peers)) * 10
        support = np.isfinite(y) & (np.isfinite(X[:, last_col]).sum(axis=1) >= 2)
        features.append(X[support])
        values.append(y[support])
        weights.append((counts[use][support] + 4) * confidence[support])
    if not features or not sum(map(len, features)):
        return None
    return np.vstack(features), np.concatenate(values), np.concatenate(weights)


def model_prediction(train, context, group, causal, model_type):
    unit = group.unit.iloc[0]
    weekday = group.weekday.iloc[0]
    t = group.minute_of_day.to_numpy(int)
    now = group.now_minute_of_day.to_numpy(int) if causal else t
    all_peers = [u for u in sorted(train.unit.unique()) if u != unit]
    presence = peer_presence(context, all_peers, t, now, causal)
    answer = np.full(len(group), np.nan)
    for pattern in np.unique(presence, axis=0):
        peers = [u for u, p in zip(all_peers, pattern) if p]
        if len(peers) < 2:
            continue
        rows = (presence == pattern).all(axis=1)
        Xq = center_delta(observation_bank(context, peers), peers, t[rows], now[rows], weekday, causal)
        parts = []
        for _, rec in train.groupby("recording", sort=True):
            part = examples(rec, unit, peers, causal)
            if part is None:
                continue
            X, y, w = part
            importance = (2.5 if rec.weekday.iloc[0] == weekday else 1.0) / len(y) ** 0.30
            parts.append((X, y, w * importance))
        if len(parts) < 2:
            continue
        X, y, w = [np.concatenate(v) for v in zip(*parts)]
        w /= w.mean()
        if model_type == "ridge":
            start = 3 + 10 * len(peers)
            if causal:
                # In the linear model, slopes alone have reduced domain shift.
                cols = [2] + [start + 5 * j + k for j in range(len(peers)) for k in range(4)]
            else:
                cols = [start + 5 * j + k for j in range(len(peers)) for k in range(4)]
            X = X[:, cols]
            Xq = Xq[:, cols]
            imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            scale = StandardScaler(with_mean=False)
            X = scale.fit_transform(imputer.fit_transform(X))
            Xq = scale.transform(imputer.transform(Xq))
            model = Ridge(alpha=225)
        else:
            model = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=110,
                max_leaf_nodes=12, max_depth=4, min_samples_leaf=40,
                l2_regularization=17, early_stopping=False, random_state=827,
            )
        model.fit(X, y, sample_weight=w)
        val = model.predict(Xq)
        if causal:
            val *= np.sqrt(t[rows] - now[rows])
        answer[rows] = np.clip(val, -0.42, 0.42)
    return answer


def predict_profile(train, released, targets, baseline, model_type="ridge", strength=0.45,
                    hidden=True, future=True, base_anchor=True):
    result = baseline.set_index("id").people.copy()
    kinds = ([] if not hidden else ["hidden"]) + ([] if not future else ["hidden-future"])
    for (rec, unit, kind), group in targets[targets.kind.isin(kinds)].groupby(["recording", "unit", "kind"]):
        causal = kind == "hidden-future"
        context = released[released.recording == rec]
        innovation = model_prediction(train, context, group, causal, model_type)
        valid = np.isfinite(innovation)
        if not valid.any():
            continue
        original = result.loc[group.id].to_numpy(float)
        log = np.log(original + 4)
        t = group.minute_of_day.to_numpy(float)
        if causal:
            origin = group.now_minute_of_day.to_numpy(float)
            if base_anchor:
                # Use only the baseline's earliest published same-origin
                # prediction as the common count-level anchor. All horizons
                # in that group depend on the same causal observation prefix.
                order = np.argsort(t, kind="stable")
                anchor = log[np.array([order[np.flatnonzero(origin[order] == p)[0]] for p in origin])]
                if len(group):
                    innovation -= log - anchor
        else:
            ix = np.argsort(t)
            smooth = Observations(t[ix], log[ix]).average(t - 9, t + 9)
            if base_anchor:
                innovation -= log - smooth
        corrected = np.maximum(np.exp(log[valid] + strength * innovation[valid]) - 4, 0.0)
        result.loc[group.id[valid]] = corrected
    return result.rename("people").reset_index()


def predict_fold(train, released, targets, baseline):
    return predict_profile(train, released, targets, baseline)
