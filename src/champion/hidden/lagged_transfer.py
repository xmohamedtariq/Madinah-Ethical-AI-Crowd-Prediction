"""Predict fully hidden cameras from peer histories under shared blind intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Observations:
    minute: np.ndarray
    count: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "prefix", np.r_[0.0, self.count.cumsum()])

    def average(self, left, right):
        start = np.searchsorted(self.minute, left, side="left")
        end = np.searchsorted(self.minute, right, side="right")
        size = np.where(right >= left, end - start, 0)
        out = np.full(len(left), np.nan)
        valid = size > 0
        out[valid] = (self.prefix[end[valid]] - self.prefix[start[valid]]) / size[valid]
        return out

    def near(self, timestamp, cutoff, causal):
        size = np.searchsorted(self.minute, cutoff, side="right") if causal else np.full(len(timestamp), len(self.minute))
        left = np.minimum(np.searchsorted(self.minute, timestamp, side="right"), size) - 1
        right = left + 1
        il = np.clip(left, 0, len(self.minute) - 1)
        ir = np.clip(right, 0, len(self.minute) - 1)
        dt_l = np.where(left >= 0, np.abs(self.minute[il] - timestamp), np.inf)
        dt_r = np.where(right < size, np.abs(self.minute[ir] - timestamp), np.inf)
        age = np.minimum(dt_l, dt_r)
        nearest = np.where(dt_l <= dt_r, self.count[il], self.count[ir])
        span = np.maximum(self.minute[ir] - self.minute[il], 1)
        mix = np.clip((timestamp - self.minute[il]) / span, 0, 1)
        interp = (1 - mix) * self.count[il] + mix * self.count[ir]
        value = np.where((left >= 0) & (right < size) & (span <= 15), interp, nearest)
        return np.where(age <= 16, value, np.nan), np.minimum(age, 24)


def observation_bank(data, peers):
    bank = {}
    for unit, z in data.groupby("unit", sort=False):
        if unit in peers and len(z):
            z = z.groupby("minute_of_day", sort=True).people.mean()
            bank[unit] = Observations(z.index.to_numpy(float), z.to_numpy(float))
    return bank


def featurize(bank, peers, t, now, weekday, causal):
    X = [t / 60, np.full(len(t), weekday == "Fri"), t - now]
    for unit in peers:
        if unit not in bank:
            X.extend(np.full(len(t), np.nan) for _ in range(10))
            continue
        v = bank[unit]
        end = now if causal else t
        last, age = v.near(end, end, causal)
        lag, lag_age = v.near(end - 4, end, causal)
        if causal:
            recent = v.average(end - 5, end)
            mid = v.average(end - 13, end - 6)
            slow = v.average(end - 26, end - 14)
            long = v.average(end - 75, end - 27)
            before = v.average(end - 150, end - 76)
        else:
            recent = v.average(t - 3, t + 3)
            mid = v.average(t - 10, t - 4)
            slow = v.average(t - 18, t - 11)
            long = v.average(t - 32, t + 32)
            before = v.average(t + 4, t + 10)

        log = lambda v: np.log2(v + 4)
        X.extend((log(last), log(lag), log(recent), log(long), log(recent)-log(mid),
                  log(mid)-log(slow), log(recent)-log(long), log(before)-log(long), age, lag_age))
    return np.column_stack(X)


def peer_presence(released, peers, t, now, causal):
    visible = np.full((len(t), len(peers)), False)
    for i, u in enumerate(peers):
        ts = np.sort(released.loc[released.unit == u, "minute_of_day"].unique())
        if len(ts):
            visible[:, i] = np.searchsorted(ts, now, side="right") > 0 if causal else True
    return visible


def masked_history(rec, target, peers, shift):
    """Mask an entire target camera and synchronize 30-minute holes in the peers."""
    z = rec[rec.unit.isin(peers)]
    phase = (z.minute_of_day - shift) % 55
    z = z[phase < 25]
    if not len(z):
        return z
    # Drop rare camera-specific intervals in addition to the shared query grid.
    omit_first = (z.minute_of_day + z.unit.map({u: i * 7 for i, u in enumerate(peers)})) % 103 < 6
    return z[~omit_first]


def training_samples(rec, unit, peers, causal):
    own = rec[rec.unit == unit].sort_values("minute_of_day")
    if len(own) < 30:
        return []
    times = own.minute_of_day.to_numpy(int)
    values = own.people.to_numpy(float)
    weekday = own.weekday.iloc[0]
    blocks = []
    # Spread pseudo-origins through the timeline. Masks are independent of values.
    for shift in (0, 8, 16, 24, 32, 40, 48):
        mask = masked_history(rec, unit, peers, shift)
        bank = observation_bank(mask, peers)
        if causal:
            t_mod = (times - shift) % 55
            use = (t_mod >= 25) & (t_mod <= 39)
            t = times[use]
            now = t - t_mod[use] + 24
        else:
            t_mod = (times - shift) % 55
            use = (t_mod <= 24) & (t_mod % 3 == 0)
            t = times[use]
            now = t
        if len(t):
            X = featurize(bank, peers, t, now, weekday, causal)
            cols = np.array([3 + 10 * i for i in range(len(peers))], int)
            has_peers = np.isfinite(X[:, cols]).sum(axis=1) >= 2
            blocks.append((X[has_peers], values[use][has_peers]))
    return blocks


def fit_prediction(X, y, weights, Xq, model_name):
    transform = model_name.endswith("_log")
    if transform:
        weights = weights * (y + 4)
        y = np.log2(y + 4)
    weights = weights / np.mean(weights)
    if model_name.startswith("ridge"):
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        scale = StandardScaler()
        X = scale.fit_transform(imputer.fit_transform(X))
        Xq = scale.transform(imputer.transform(Xq))
        model = Ridge(alpha=100)
    else:
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.055, max_iter=135,
            max_leaf_nodes=14, max_depth=5, min_samples_leaf=30,
            l2_regularization=13, early_stopping=False, random_state=467,
        )
    model.fit(X, y, sample_weight=weights)
    pred = model.predict(Xq)
    return np.maximum(0.0, 2**pred - 4 if transform else pred)


def _join_examples(groups, weekday, recency):
    X = np.vstack([g[0] for g in groups])
    y = np.concatenate([g[1] for g in groups])
    weights = np.concatenate([
        np.full(len(g[1]), (2.5 if g[2] == weekday else 1.0) * (len(g[1]) / 150) ** recency / max(len(g[1]), 1))
        for g in groups
    ])
    return X, y, weights


def _estimate(train, released, q, models, causal, inner):
    yday = q.weekday.iloc[0]
    us = q.unit.iloc[0]
    t = q.minute_of_day.to_numpy(int)
    now = q.now_minute_of_day.to_numpy(int) if causal else t
    all_peers = [u for u in sorted(train.unit.unique()) if u != us]
    visibility = peer_presence(released, all_peers, t, now, causal)
    values = np.full((len(q), len(models)), np.nan)
    training_records = list(train.groupby("recording", sort=True))

    for visible in np.unique(visibility, axis=0):
        peers = [p for p, keep in zip(all_peers, visible) if keep]
        if len(peers) < 2:
            continue
        use = (visibility == visible).all(axis=1)
        bank = observation_bank(released, peers)
        Xq = featurize(bank, peers, t[use], now[use], yday, causal)
        groups = []
        for _, z in training_records:
            samples = training_samples(z, us, peers, causal)
            samples = [(X, y) for X, y in samples if len(y)]
            if samples:
                X = np.vstack([g[0] for g in samples])
                y = np.concatenate([g[1] for g in samples])
                groups.append((X, y, z.weekday.iloc[0], z.day.iloc[0]))
        if len(groups) < 2:
            continue
        if inner:
            inner_losses = np.zeros(len(models))
            days = sorted({g[3] for g in groups})
            for day in days:
                held = [g for g in groups if g[3] == day]
                kept = [g for g in groups if g[3] != day]
                if not held or not kept:
                    continue
                X, y, w = _join_examples(kept, held[0][2], 0.7)
                vx, vy, _ = _join_examples(held, held[0][2], 0.7)
                for j, m in enumerate(models):
                    pred = fit_prediction(X, y, w, vx, m)
                    inner_losses[j] += np.mean(np.abs(pred - vy)) / len(days)
            print("INNER", ("forecast" if causal else "hidden"), us, len(peers), dict(zip(models, inner_losses.round(3))), flush=True)
        X, y, w = _join_examples(groups, yday, 0.7)
        for i, m in enumerate(models):
            pred = fit_prediction(X, y, w, Xq, m)
            values[use, i] = pred
    return values


PROFILES = {
    "lagged": (("hgb_log",), 0.45, False),
    "lagged_full": (("hgb_log",), 1.0, False),
    "lagged_small": (("hgb_log",), 0.20, False),
    "ridge": (("ridge_log",), 0.45, False),
    "ensemble": (("hgb_log", "ridge_log"), 0.45, True),
}


def predict_profile(train, released, targets, baseline, profile):
    models, blend, inner = PROFILES[profile]
    out = baseline.set_index("id").people.copy()
    q = targets[targets.kind.isin(("hidden", "hidden-future"))]
    for (recording, unit, kind), group in q.groupby(["recording", "unit", "kind"], sort=True):
        causal = kind == "hidden-future"
        # Other recordings on the same day only enter with explicit chronological
        # context. This transfer uses the query recording and no outside counts.
        context = released[released.recording == recording]
        candidates = _estimate(train, context, group, models, causal, inner)
        available = np.isfinite(candidates).all(axis=1)
        pred = candidates[available].mean(axis=1)
        ids = group.id[available]
        out.loc[ids] = (1 - blend) * out.loc[ids] + blend * pred
    return out.rename("people").reset_index()


def predict_fold(train, released, targets, baseline):
    return predict_profile(train, released, targets, baseline, "lagged")
