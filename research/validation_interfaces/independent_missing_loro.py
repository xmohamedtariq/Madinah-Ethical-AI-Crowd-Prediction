"""Independent recording-holdout Missing stress test on four earlier recordings."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "verified_605700"
sys.path.insert(0, str(ROOT / "src" / "final_missing_hidden_research"))
sys.path.insert(0, str(PKG / "cloud_research" / "selected"))
from candidate import predict_fold as champion
from missing_state_space import predict_missing

DATA = pd.read_csv(ROOT / "data" / "official" / "train.csv")
OUT = ROOT / "reports" / "breakthrough_v2"
OUT.mkdir(parents=True, exist_ok=True)
heldout = [r for r in sorted(DATA.recording.unique()) if r not in {"20260911_1100", "20260912_1100"}]
results, row_reports = [], []
for fold_i, recording in enumerate(heldout):
    train = DATA.loc[DATA.recording.ne(recording)].copy()
    full = DATA.loc[DATA.recording.eq(recording)].copy()
    rng = np.random.RandomState(6700 + fold_i)
    used = set(); target_rows = []; blocks_made = 0
    for attempt in range(5000):
        if len(target_rows) >= 70:
            break
        camera = rng.choice(sorted(full.unit.unique()))
        camera_rows = full.loc[full.unit.eq(camera)].sort_values("minute_of_day")
        times = camera_rows.minute_of_day.to_numpy(int)
        length = int(rng.randint(5, 19))
        starts = np.flatnonzero((times >= times.min()+3) & (times <= times.max()-length-3))
        if not len(starts):
            continue
        start_idx = int(rng.choice(starts)); block = times[start_idx:start_idx+length]
        if len(block) != length or np.any(np.diff(block) != 1):
            continue
        keys = {(camera, int(t)) for t in block}
        if keys & used:
            continue
        used |= keys
        block_rows = camera_rows.loc[camera_rows.minute_of_day.isin(block)].to_dict("records")
        for row in block_rows:
            row["block_start"] = int(block[0])
        target_rows.extend(block_rows)
        blocks_made += 1
    if len(target_rows) < 50:
        raise RuntimeError(f"Could not construct enough matched blocks for {recording}: {len(target_rows)}")
    gaps = pd.DataFrame(target_rows)
    ids = [f"{recording}_{r.unit}_{int(r.minute_of_day):04d}" for r in gaps.itertuples()]
    gaps["id"] = ids; gaps["kind"] = "missing"; gaps["now"] = np.nan
    gaps["now_minute_of_day"] = np.nan; gaps["horizon"] = np.nan
    targets = gaps[["id", "recording", "day", "weekday", "unit", "minute", "minute_of_day", "kind", "now", "now_minute_of_day", "horizon"]].copy()
    held = set(zip(gaps.unit, gaps.minute_of_day.astype(int)))
    released = full.loc[~full.apply(lambda r: (r.unit, int(r.minute_of_day)) in held, axis=1)].copy()
    refs = {}
    for camera, part in gaps.groupby("unit"):
        obs = released.loc[released.unit.eq(camera)].set_index("minute_of_day").people.sort_index()
        grid = obs.reindex(sorted(set(obs.index) | set(part.minute_of_day)))
        interp = grid.interpolate(method="linear", limit_area="inside")
        for row in part.itertuples():
            refs[row.id] = float(interp.loc[row.minute_of_day])
    baseline = pd.DataFrame({"id": targets.id, "people": targets.id.map(refs)})
    candidate_pred = champion(train, released, targets, baseline)
    state_pred = predict_missing(train, released, targets, baseline)
    truth = pd.DataFrame({"id": gaps.id, "truth": gaps.people.to_numpy(float)})
    scored = truth.merge(baseline.rename(columns={"people":"linear"}), on="id").merge(
        candidate_pred.rename(columns={"people":"champion"}), on="id").merge(
        state_pred.rename(columns={"people":"rts"}), on="id")
    ref_mae = float(np.mean(np.abs(scored.linear-scored.truth)))
    champion_mae = float(np.mean(np.abs(scored.champion-scored.truth)))
    rts_mae = float(np.mean(np.abs(scored.rts-scored.truth)))
    results.append({"recording": recording, "n": len(scored), "blocks": blocks_made,
                    "reference_mae": ref_mae, "champion_mae": champion_mae, "rts_mae": rts_mae,
                    "champion_ratio": champion_mae/ref_mae, "rts_ratio": rts_mae/ref_mae,
                    "incremental_full_score_delta_rts_vs_champion": (rts_mae-champion_mae)/ref_mae/4})
    scored["recording"] = recording
    scored.to_csv(OUT / f"independent_missing_{recording}_predictions.csv", index=False)
pd.DataFrame(results).to_csv(OUT / "independent_missing_loro.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
