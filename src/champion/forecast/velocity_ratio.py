"""Camera projection with a supervised relative response to forecasted motion."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import reconciliation as ratio
from .calendar_transfer import SETTINGS as CALENDAR_SETTINGS
from .calendar_transfer import calendar_origin, historical_bank, pivot


def fit_velocity(bank, cameras, relationships):
    result = {}
    for cam_u, cam_v in combinations(cameras, 2):
        pair = (cam_u, cam_v)
        if pair not in relationships:
            continue
        designs = []
        outcomes = []
        for _, _, raw, _ in bank.values():
            if cam_u not in raw or cam_v not in raw:
                continue
            x = np.log(raw[[cam_u,cam_v]] + ratio.PRIOR)
            level = x.mean(axis=1, skipna=False)
            velocity = (level - level.shift(4)) / 4
            contrast = x[cam_u] - x[cam_v]
            frame = pd.concat([level, velocity, contrast], axis=1)
            for lag in (4, 10):
                change = frame.diff(lag).dropna().to_numpy()
                if len(change):
                    designs.append(change[:, :2])
                    outcomes.append(change[:, 2])
        if not designs:
            continue
        x = np.concatenate(designs)
        y = np.concatenate(outcomes)
        coefficient = np.zeros(2)
        for _ in range(5):
            residual = y - x @ coefficient
            w = np.minimum(1., .12 / np.maximum(.01, abs(residual)))
            penalty = np.diag([12., .9])
            rhs = x.T @ (w*y) + penalty @ np.array([relationships[pair], 0.])
            coefficient = np.linalg.solve(x.T @ (w[:,None]*x)+penalty, rhs)
        result[pair] = float(np.clip(coefficient[1], -2., 2.))
    return result


def velocity_projection(path, prefix, now, base_edges, dynamics, strength):
    anchors = {}
    history = np.log(prefix.loc[now-10:now] + ratio.PRIOR)
    level = np.log(path + ratio.PRIOR)
    for u, v, *_ in base_edges:
        cont = history[[u,v]].mean(axis=1,skipna=False)
        timeline = pd.concat([cont, level[[u,v]].mean(axis=1,skipna=False).set_axis(level.index+now)])
        timeline = timeline.reindex(range(now-10, now+16))
        velocity = (timeline - timeline.shift(4))/4
        old = velocity.loc[:now].dropna().tail(5)
        if len(old) >= 2:
            anchors[(u, v)] = (old.median(),velocity)
    all_parts = []
    for h, base_row in path.iterrows():
        adapted = []
        for u,v,anchor_ratio,slope,anchor_level,stability in base_edges:
            beta = dynamics.get((u,v), 0.)
            history_velocity, velocity = anchors.get((u,v), (0., None))
            change = 0.
            if velocity is not None and np.isfinite(velocity.at[now+h]):
                change = np.clip(beta*(velocity.at[now+h]-history_velocity),-.12,.12)
            adapted.append((u,v,anchor_ratio+change,slope,anchor_level,stability))
        all_parts.append(ratio.reconcile(base_row.to_frame().T,adapted,strength))
    return pd.concat(all_parts)


def choose_strength(bank, cameras):
    scores = {0.: [], .55: [], 1.: []}
    for day, weekday, raw, _ in bank.values():
        allowed = {k:v for k,v in bank.items() if v[0] != day}
        available = sorted(set(raw.columns) & set(cameras))
        history = ratio.historical_relationships(allowed, available)
        dynamics = fit_velocity(allowed, available, history)
        positions = raw.index[raw.notna().any(axis=1)]
        for now in range(int(positions[0])+25, int(positions[-1])-14, 17):
            current = raw.loc[:now, available].copy()
            current.loc[current.index[(now-current.index > 8) & (current.index % 4 != 0)]] = np.nan
            edges = ratio.pair_relationships(current, now, available, history)
            if not edges:
                continue
            path = calendar_origin(allowed,current,now,weekday,CALENDAR_SETTINGS["long_clock"],available)
            naive = .75*current.iloc[-9:].ffill().iloc[-1] + .25*current.iloc[-9:].mean()
            forecast = (.60*path + .40*naive).fillna(pd.DataFrame(
                np.tile(naive.to_numpy(), (15,1)),index=path.index,columns=available))
            true = raw.loc[now+1:now+15,available].to_numpy()
            mask = np.isfinite(forecast.to_numpy()) & np.isfinite(true)
            if mask.sum()<30:
                continue
            base_loss = np.abs(forecast.to_numpy()-true)[mask].mean()
            for strength in scores:
                if strength == 0:
                    output = forecast
                else:
                    output = velocity_projection(forecast,current,now,edges,dynamics,strength)
                scores[strength].append(np.abs(output.to_numpy()-true)[mask].mean()/max(base_loss,1.))
    averages = {x: np.mean(y) for x,y in scores.items() if y}
    choice = min(averages,key=averages.get) if averages else 0.
    print("VELOCITY_RATIO_INNER",choice,averages,flush=True)
    return choice


def predict_config(train, released, targets, baseline, settings="v21", blend=.25):
    if settings == "tiled":
        from . import strict_mask_hgb
        baseline = strict_mask_hgb.predict_config(
            train,released,targets,baseline,blend=blend,settings="tiled"
        )
    result = baseline[["id","people"]].copy()
    query = targets.loc[targets.kind.eq("forecast")].merge(result,on="id",validate="one_to_one")
    if query.empty:
        return result
    cameras = sorted(query.unit.unique())
    bank = historical_bank(train)
    strength = choose_strength(bank,cameras)
    if strength <= 0:
        return result
    history = ratio.historical_relationships(bank,cameras)
    dynamics = fit_velocity(bank,cameras,history)
    updates = {}
    for (recording,now), group in query.groupby(["recording","now_minute_of_day"]):
        now = int(now)
        view = released.loc[released.recording.eq(recording) & (released.minute_of_day <= now)]
        prefix = pivot(view)
        path = group.pivot_table(index="horizon",columns="unit",values="people")
        current = sorted(set(path.columns)&set(prefix.columns))
        edges = ratio.pair_relationships(prefix,now,current,history)
        if edges:
            updated=velocity_projection(path,prefix,now,edges,dynamics,strength)
            for row in group.itertuples(index=False):
                updates[row.id] = updated.at[row.horizon,row.unit]
    values = result.id.map(updates)
    result["people"] = values.fillna(result.people)
    print("VELOCITY_RATIO_ROWS",len(updates),len(query),flush=True)
    return result


def predict_fold(train,released,targets,baseline):
    return predict_config(train,released,targets,baseline)
