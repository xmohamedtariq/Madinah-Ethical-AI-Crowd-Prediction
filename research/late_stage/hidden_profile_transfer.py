"""Whole-recording nearest-profile transfer for a fully hidden camera."""
import numpy as np
import pandas as pd


def _robust_scale(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 1.0
    q25, q75 = np.quantile(x, [.25, .75])
    return max(float((q75-q25)/1.349), .25)


def _profile_distance(current, donor, peers):
    scores = []
    for peer in peers:
        if peer not in current or peer not in donor:
            continue
        a = np.log1p(current[peer].clip(lower=0))
        b = np.log1p(donor[peer].clip(lower=0))
        aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if len(aligned) < 25:
            continue
        ac = aligned.a - aligned.a.median()
        bc = aligned.b - aligned.b.median()
        sa, sb = _robust_scale(ac), _robust_scale(bc)
        scores.append(float(np.median(np.abs(ac/sa-bc/sb))))
    return float(np.median(scores)) if scores else np.inf


def predict_hidden(train, released, targets, baseline, shrink=0.5, neighbors=3):
    result = baseline.set_index("id").people.astype(float).copy()
    hidden = targets.loc[targets.kind.eq("hidden")]
    for recording, query_rec in hidden.groupby("recording", sort=True):
        context = released.loc[released.recording.eq(recording)]
        current = context.pivot_table(index="minute_of_day", columns="unit", values="people", aggfunc="mean").sort_index()
        for camera, query in query_rec.groupby("unit", sort=True):
            peers = [u for u in sorted(context.unit.unique()) if u != camera]
            donors = []
            for donor_recording, frame in train.groupby("recording", sort=True):
                if donor_recording == recording or not frame.unit.eq(camera).any():
                    continue
                donor = frame.pivot_table(index="minute_of_day", columns="unit", values="people", aggfunc="mean").sort_index()
                if camera not in donor or donor[camera].notna().sum() < 40:
                    continue
                distance = _profile_distance(current, donor, peers)
                if np.isfinite(distance):
                    same_weekday = float(frame.weekday.iloc[0] == query.weekday.iloc[0])
                    donors.append((distance - 0.15*same_weekday, donor))
            if not donors:
                continue
            donors.sort(key=lambda x: x[0])
            donors = donors[:neighbors]
            distances = np.asarray([d for d, _ in donors], float)
            weights = np.exp(-(distances-distances.min())/.35)
            weights /= weights.sum()
            predictions = []
            for (_, donor), weight in zip(donors, weights):
                log_y = np.log1p(donor[camera].clip(lower=0))
                correction = []
                for minute in query.minute_of_day.astype(int):
                    available = [p for p in peers if p in current and p in donor
                                 and minute in current.index and minute in donor.index
                                 and np.isfinite(current.at[minute, p]) and np.isfinite(donor.at[minute, p])]
                    if available:
                        shifts = [np.log1p(max(current.at[minute,p],0))-
                                  np.log1p(max(donor.at[minute,p],0)) for p in available]
                        correction.append(float(np.median(shifts)))
                    else:
                        correction.append(0.0)
                donor_y = log_y.reindex(query.minute_of_day.astype(int)).to_numpy(float)
                delta = np.asarray(correction, float)
                predictions.append((weight, donor_y, delta))
            query_pred = np.full(len(query), 0.0)
            total = np.zeros(len(query))
            for weight, donor_y, delta in predictions:
                valid = np.isfinite(donor_y)
                proposal = np.expm1(donor_y + shrink*delta)
                query_pred[valid] += weight*np.maximum(0., proposal[valid])
                total[valid] += weight
            valid = total > 0
            values = result.loc[query.id].to_numpy(float)
            values[valid] = query_pred[valid]/total[valid]
            result.loc[query.id] = values
    return result.rename("people").reset_index()
