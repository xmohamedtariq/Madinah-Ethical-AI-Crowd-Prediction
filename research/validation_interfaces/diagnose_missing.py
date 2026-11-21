"""Feature and score-contribution audit of every frozen Missing target."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "verified_605700"
FOLDS = PKG / "research_kit" / "audit" / "folds"
PRED = pd.read_csv(PKG / "independent_replay" / "verified_local" / "validation_predictions.csv")
ROWS, REFS = [], {}
for fold in sorted(FOLDS.iterdir()):
    target = pd.read_csv(fold / "targets.csv")
    truth = pd.read_csv(fold / "gold.csv")
    released = pd.read_csv(fold / "released.csv")
    m = target[target.kind.eq("missing")].merge(truth, on="id").merge(
        PRED[["id", "people"]].rename(columns={"people": "prediction"}), on="id")
    ref = float(np.mean(np.abs(m.people_ref - m.people_true)))
    n = len(m)
    for camera, gaps in m.groupby("unit", sort=True):
        times = released.loc[(released.recording.eq(fold.name)) & released.unit.eq(camera)].sort_values("minute_of_day")
        observed_t = times.minute_of_day.to_numpy(int)
        observed_y = times.people.to_numpy(float)
        peer = released.loc[(released.recording.eq(fold.name)) & ~released.unit.eq(camera)]
        peer_by_time = peer.groupby("minute_of_day").people.agg(["median", "mean"])
        all_gap_times = np.sort(gaps.minute_of_day.to_numpy(int))
        blocks, start = [], 0
        for i in range(1, len(all_gap_times)+1):
            if i == len(all_gap_times) or all_gap_times[i] != all_gap_times[i-1] + 1:
                blocks.append(all_gap_times[start:i])
                start = i
        for block in blocks:
            left_ix = np.searchsorted(observed_t, block[0], side="left") - 1
            right_ix = np.searchsorted(observed_t, block[-1], side="right")
            left_t = observed_t[left_ix] if left_ix >= 0 else np.nan
            right_t = observed_t[right_ix] if right_ix < len(observed_t) else np.nan
            left_y = observed_y[left_ix] if left_ix >= 0 else np.nan
            right_y = observed_y[right_ix] if right_ix < len(observed_y) else np.nan
            left_gap, right_gap = (block[0]-left_t if np.isfinite(left_t) else np.nan), (right_t-block[-1] if np.isfinite(right_t) else np.nan)
            slope = ((right_y-left_y)/(right_t-left_t) if np.isfinite(left_y+right_y+left_t+right_t) else np.nan)
            if not np.isfinite(slope):
                flow = "one_sided_or_unbounded"
            elif slope > 1.0:
                flow = "rising_fast"
            elif slope > 0.15:
                flow = "rising_slow"
            elif slope < -1.0:
                flow = "falling_fast"
            elif slope < -0.15:
                flow = "falling_slow"
            else:
                flow = "flat_endpoints"
            for pos, minute in enumerate(block, start=1):
                row = gaps.loc[gaps.minute_of_day.eq(minute)].iloc[0]
                loc = np.searchsorted(observed_t, minute)
                own_prev = observed_y[max(0, loc-3):loc]
                own_next = observed_y[loc:min(len(observed_y), loc+3)]
                pm = peer_by_time.loc[minute, "median"] if minute in peer_by_time.index else np.nan
                pred_error = abs(float(row.prediction) - float(row.people_true))
                ROWS.append({
                    "id": row.id, "recording": fold.name, "camera": camera,
                    "minute_of_day": minute, "block_start": int(block[0]), "block_length": len(block),
                    "gap_position": pos, "gap_position_fraction": pos/(len(block)+1),
                    "left_neighbor_distance": left_gap, "right_neighbor_distance": right_gap,
                    "left_count": left_y, "right_count": right_y, "endpoint_slope_per_min": slope,
                    "flow_pattern": flow,
                    "local_prev_mean": float(np.mean(own_prev)) if len(own_prev) else np.nan,
                    "local_next_mean": float(np.mean(own_next)) if len(own_next) else np.nan,
                    "peer_median": pm,
                    "truth": float(row.people_true), "selected_prediction": float(row.prediction),
                    "reference_prediction": float(row.people_ref), "abs_error": pred_error,
                    "normalized_full_score_contribution": pred_error/(ref*n*8),
                    "task_reference_mae": ref,
                })

out_dir = ROOT / "reports" / "breakthrough_v2"
out_dir.mkdir(parents=True, exist_ok=True)
rows = pd.DataFrame(ROWS)
rows.to_csv(out_dir / "missing_target_diagnostics.csv", index=False)
dimensions = ["recording", "camera", "block_length", "gap_position", "flow_pattern"]
tables = []
for dim in dimensions:
    agg = rows.groupby(dim, dropna=False).agg(
        n=("id", "size"), mae=("abs_error", "mean"),
        score_contribution=("normalized_full_score_contribution", "sum"),
        mean_left_distance=("left_neighbor_distance", "mean"),
        mean_right_distance=("right_neighbor_distance", "mean"),
    ).reset_index().rename(columns={dim: "value"})
    agg.insert(0, "dimension", dim)
    tables.append(agg)
pd.concat(tables, ignore_index=True).sort_values("score_contribution", ascending=False).to_csv(
    out_dir / "missing_error_groups.csv", index=False)
print(f"rows={len(rows)} blocks={rows[['recording','camera','block_start']].drop_duplicates().shape[0]}")
