"""Local development composite, using only the four evaluator callback inputs."""

import numpy as np
from threadpoolctl import threadpool_limits

from forecast.velocity_ratio import predict_config as predict_forecast
from hidden.released_calibration import predict_profile as predict_proxy_hidden_future
from missing.frozen_missing import predict_fold as predict_frozen_missing
from missing.pattern_event import predict_fold as improve_missing


def branch(train, released, targets, baseline, kind):
    if kind == "forecast":
        return predict_forecast(train, released, targets, baseline, settings="tiled")
    if kind == "hidden-future":
        return predict_proxy_hidden_future(train, released, targets, baseline, adapt=False)
    starting = predict_frozen_missing(train, released, targets, baseline)
    return improve_missing(train, released, targets, baseline, reference=starting)


def predict_fold(train, released, targets, baseline):
    result = baseline.set_index("id").people.astype(float).copy()
    targets_by_id = targets.set_index("id")
    if (not result.index.is_unique or not targets_by_id.index.is_unique
            or result.index.hasnans or set(result.index) != set(targets_by_id.index)):
        raise ValueError("Baseline and target IDs must be unique and match exactly.")

    with threadpool_limits(limits=1):
        for recording, recording_targets in targets.groupby("recording", sort=False):
            context = released.loc[released.recording.eq(recording)]
            for kind in ("forecast", "hidden-future", "missing"):
                query = recording_targets.loc[recording_targets.kind.eq(kind)]
                if query.empty:
                    continue
                initial = baseline.set_index("id").loc[query.id, ["people"]].reset_index()
                predicted = branch(train, context, query, initial, kind)
                values = predicted.set_index("id").people
                if not values.index.is_unique or set(values.index) != set(query.id):
                    raise ValueError(f"Unexpected IDs from {kind} in {recording}.")
                result.loc[query.id] = values.reindex(query.id).to_numpy(float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Predictions must be finite and nonnegative.")
    return result.rename("people").reset_index()
