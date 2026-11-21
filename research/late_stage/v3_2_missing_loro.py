#!/usr/bin/env python3
"""V3.2: recording-disjoint pseudo-gap stress-test for the exact 0.605700 champion.

Research ONLY: does not write Kaggle submissions, does not alter selected model,
never calls Kaggle or reads test.csv.  It tests a *narrow curvature correction*
on Missing targets from four earlier recordings, then optionally replays a
pre-specified correction on the two historically reused frozen folds.

Invocation on Windows from Madinah-AI-Fresh:
    python v3_2_missing_loro.py --root .

Smoke-test logic (substitutes linear interpolation, NOT the champion):
    python v3_2_missing_loro.py --root . --smoke-linear --skip-frozen

Requires numpy/pandas and the already selected champion source:
    verified_605700/cloud_research/selected/candidate.py
The candidate.py file must expose predict_fold(train, released, targets, baseline).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

CHAMPION_SCORE = 0.6057000521183546
TARGET_SCORE = 0.6000000000
HISTORIC_FROZEN = ("20260911_1100", "20260912_1100")
OLD_RECORDINGS = ("20260904_1105", "20260905_0426", "20260905_1124", "20260905_1900")
# cam_E is not present in *any* of the four older records; it cannot be externally validated.
VALIDATED_CAMERAS = ("cam_A", "cam_F")
UNVALIDATED_TRANSFER_CAMERA = "cam_E"
SEED = 32026
# These fixed strata/lengths are declared before any held-out-label scoring.
STRATA = (
    ("cam_A", "fast", (5, 8, 11, 14)),
    ("cam_F", "fast", (5, 8, 11, 14)),
    ("cam_A", "control", (7, 12)),
    ("cam_F", "control", (7, 12)),
    ("other", "all", (5, 9, 13, 15)),
)
ALPHAS = (-0.10, -0.05, 0.0, 0.05, 0.10)
MAX_DELTA_PEOPLE = 3.0
GUARD_PADDING_MINUTES = 4
MIN_SELECTED_BLOCKS = 6
MIN_SELECTED_FAST_A_F = 2
REQUIRED_COLUMNS = {"recording", "day", "weekday", "unit", "minute", "minute_of_day", "people"}
FEATURE_COLUMNS = (
    "id", "recording", "camera", "minute_of_day", "block_start", "block_length",
    "gap_position", "gap_position_fraction", "left_count", "right_count",
    "endpoint_slope_per_min", "flow_pattern", "local_prev_mean", "local_next_mean",
    "selected_prediction", "reference_prediction",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_output_dir_safe(root: Path, out: Path) -> None:
    root = root.resolve()
    out = out.resolve()
    if root not in out.parents:
        raise ValueError("Output directory must be within the selected project root")
    if any("submission" in part.lower() for part in out.relative_to(root).parts):
        raise ValueError("Refusing to create any artifacts inside a submission directory")
    if out == root:
        raise ValueError("Output directory cannot be project root")


def find_train(root: Path) -> Path:
    for rel in ("data/official/train.csv", "data/raw/train.csv"):
        path = root / rel
        if path.is_file():
            return path
    raise FileNotFoundError("No data/official/train.csv or data/raw/train.csv found in project")


def read_train(root: Path) -> tuple[pd.DataFrame, Path]:
    path = find_train(root)
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"train.csv lacks required columns: {sorted(missing)}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required train columns contain nulls")
    if data.duplicated(["recording", "unit", "minute_of_day"]).any():
        raise ValueError("Nonunique (recording,unit,minute_of_day) in train")
    data["minute_of_day"] = data.minute_of_day.astype(int)
    data["people"] = data.people.astype(float)
    data["recording"] = data.recording.astype(str)
    if not np.isfinite(data.people.to_numpy(float)).all():
        raise ValueError("Nonfinite counts in train")
    if not set(OLD_RECORDINGS).issubset(data.recording.unique()):
        raise ValueError("One or more of the four older recordings is unavailable")
    if UNVALIDATED_TRANSFER_CAMERA in set(data.loc[data.recording.isin(OLD_RECORDINGS), "unit"]):
        raise AssertionError("Camera E presence changed; rerun research-design audit")
    if not set(HISTORIC_FROZEN).issubset(data.recording.unique()):
        raise ValueError("Historical Friday/Saturday frozen recordings absent")
    return data, path


def load_champion(root: Path, smoke_linear: bool) -> tuple[Callable, dict[str, Any]]:
    if smoke_linear:
        def linear_stub(train: pd.DataFrame, released: pd.DataFrame,
                        targets: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
            return baseline[["id", "people"]].copy()
        return linear_stub, {"kind": "SMOKE_LINEAR_NOT_CHAMPION", "path": None}
    source = root / "verified_605700" / "cloud_research" / "selected" / "candidate.py"
    if not source.is_file():
        raise FileNotFoundError(
            "Verified champion source not found at 'verified_605700/cloud_research/selected/candidate.py'. "
            "Do not substitute best_missing_model.py or evaluate_full_solution.py."
        )
    # Avoid 'candidate' name collision with another project file.
    # Match module search paths used by existing independent_missing_loro.py.
    for search_root in (root, root / "src", root / "src" / "final_missing_hidden_research", source.parent):
        if str(search_root) not in sys.path:
            sys.path.insert(0, str(search_root))
    spec = importlib.util.spec_from_file_location("v3_2_verified_candidate", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import champion source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = getattr(module, "predict_fold", None)
    if not callable(predict):
        raise TypeError("Selected source lacks callable predict_fold")
    return predict, {"kind": "SELECTED_CHAMPION_CALLBACK", "path": str(source), "sha256": digest(source)}


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def flow_pattern(slope: float) -> str:
    if not np.isfinite(slope):
        return "one_sided_or_unbounded"
    if slope > 1.0:
        return "rising_fast"
    if slope > 0.15:
        return "rising_slow"
    if slope < -1.0:
        return "falling_fast"
    if slope < -0.15:
        return "falling_slow"
    return "flat_endpoints"


@dataclass(frozen=True)
class Block:
    camera: str
    start: int
    length: int
    slope: float
    stratum: str

    @property
    def end(self) -> int:
        return self.start + self.length - 1


def enumerate_candidate_blocks(full: pd.DataFrame, recording: str) -> dict[tuple[str, str, int], list[Block]]:
    """Only released-eligible endpoints used for FAST/CONTROL selection.

    Interior truth values are never read here (except available timestamps).
    The generated masked interior serves ONLY as gold later, after predictions.
    """
    result: dict[tuple[str, str, int], list[Block]] = {}
    for camera, grp in full.groupby("unit", sort=True):
        s = grp.sort_values("minute_of_day")
        times = s.minute_of_day.to_numpy(int)
        values = s.people.to_numpy(float)
        for length in range(5, 16):
            for i in range(3, len(times) - length - 3):
                # At least 3 fully consecutive observations on either side.
                # Endpoints at i-1 and i+length; all target timestamps contiguous.
                window = times[i - 3: i + length + 3]
                if len(window) != length + 6 or not (np.diff(window) == 1).all():
                    continue
                slope = (float(values[i + length]) - float(values[i - 1])) / (length + 1)
                stratum = "fast" if slope <= -2.0 else "control"
                b = Block(camera, int(times[i]), length, slope, stratum)
                result.setdefault((camera, stratum, length), []).append(b)
                result.setdefault(("other", "all", length), []).append(b)
    # In 'other' only permit cameras that are neither of the two intervention cameras.
    for k in list(result):
        if k[0] == "other":
            result[k] = [b for b in result[k] if b.camera not in VALIDATED_CAMERAS]
    return result


def choose_pseudo_blocks(full: pd.DataFrame, recording: str) -> tuple[list[Block], dict[str, int]]:
    pool = enumerate_candidate_blocks(full, recording)
    selected: list[Block] = []
    chosen_intervals: dict[str, list[tuple[int, int]]] = {}
    coverage = {}
    for cam, stratum, lengths in STRATA:
        count = 0
        for intended_length in lengths:
            candidates = pool.get((cam, stratum, intended_length), [])
            # No label-based prioritization: hashed ordering is strictly fixed.
            for candidate in sorted(candidates, key=lambda b: stable_order(
                    f"{recording}|{b.camera}|{b.start}|{b.length}")):
                blocked = chosen_intervals.get(candidate.camera, [])
                if any(not (candidate.end + GUARD_PADDING_MINUTES < lo or
                            candidate.start - GUARD_PADDING_MINUTES > hi) for lo, hi in blocked):
                    continue
                chosen_intervals.setdefault(candidate.camera, []).append((candidate.start, candidate.end))
                selected.append(candidate)
                count += 1
                break
        coverage[f"{cam}_{stratum}"] = count
    selected = sorted(selected, key=lambda b: (b.camera, b.start))
    if len(selected) < MIN_SELECTED_BLOCKS:
        raise RuntimeError(f"Insufficient independent synthetic blocks for {recording}: {len(selected)}")
    a_f_fast = sum(1 for b in selected if b.camera in VALIDATED_CAMERAS and b.stratum == "fast")
    if a_f_fast < MIN_SELECTED_FAST_A_F:
        raise RuntimeError(f"Too few fast A/F blocks in {recording}: {a_f_fast}")
    return selected, coverage


def make_pseudogap_inputs(full: pd.DataFrame, blocks: list[Block]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (released, metadata-only targets, true labels kept separately, legal baseline)."""
    recording = str(full.recording.iloc[0])
    to_mask: set[tuple[str, int]] = set()
    metadata_rows = []
    for b in blocks:
        for minute in range(b.start, b.end + 1):
            k = (b.camera, minute)
            if k in to_mask:
                raise AssertionError("Overlapping mask windows")
            to_mask.add(k)
            match = full.loc[full.unit.eq(b.camera) & full.minute_of_day.eq(minute)]
            if len(match) != 1:
                raise AssertionError(f"Expected one row for {recording}, {k}")
            row = match.iloc[0]
            metadata_rows.append({
                "id": f"{recording}_{b.camera}_{minute:04d}",
                "recording": recording,
                "day": row.day, "weekday": row.weekday,
                "unit": b.camera, "minute": row.minute,
                "minute_of_day": int(minute),
                "kind": "missing", "now": np.nan,
                "now_minute_of_day": np.nan, "horizon": np.nan,
                "block_start": int(b.start), "block_length": int(b.length),
                "stratum": b.stratum,
            })
    # First quarantine truth; it must never be in a callback's targets or baseline.
    mask = pd.Series([(u, int(m)) in to_mask for u, m in
                      zip(full.unit, full.minute_of_day)], index=full.index)
    gold = full.loc[mask, ["unit", "minute_of_day", "people"]].copy()
    gold["id"] = [f"{recording}_{u}_{int(m):04d}" for u, m in
                  zip(gold.unit, gold.minute_of_day)]
    gold = gold.rename(columns={"people": "truth"})[["id", "truth"]]
    released = full.loc[~mask].copy()
    target_full = pd.DataFrame(metadata_rows)
    if target_full.empty or target_full.id.duplicated().any():
        raise AssertionError("Invalid pseudo targets")
    # Only the same columns as original independent_missing_loro.py.
    public_cols = ["id", "recording", "day", "weekday", "unit", "minute",
                   "minute_of_day", "kind", "now", "now_minute_of_day", "horizon"]
    targets = target_full[public_cols].copy()
    assert "people" not in targets and "truth" not in targets and "people_true" not in targets
    assert "people_true" not in released
    assert not any((u, int(m)) in to_mask for u, m in zip(released.unit, released.minute_of_day))
    # Same legal, two-sided Missing reference interpolation as previous heldout script.
    refs: dict[str, float] = {}
    for camera, group in target_full.groupby("unit", sort=True):
        observed = (released.loc[released.unit.eq(camera)]
                    .set_index("minute_of_day").people.sort_index())
        grid = observed.reindex(sorted(set(observed.index) | set(group.minute_of_day)))
        interp = grid.interpolate(method="linear", limit_area="inside")
        for row in group.itertuples():
            refs[row.id] = float(interp.loc[row.minute_of_day])
    baseline = pd.DataFrame({"id": targets.id, "people": targets.id.map(refs)})
    if not np.isfinite(baseline.people.to_numpy(float)).all():
        raise AssertionError("Missing synthetic reference values")
    if set(gold.id) != set(targets.id) or set(baseline.id) != set(targets.id):
        raise AssertionError("Pseudo target/reference label identity mismatch")
    return released, targets, gold, baseline


def get_predictions(pred_fn: Callable, train: pd.DataFrame, released: pd.DataFrame,
                    targets: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Use exactly the selected champion's predict_fold callback."""
    # Keep metadata-only target columns and copy to prevent accidental data edits.
    raw = pred_fn(train.copy(), released.copy(), targets.copy(), baseline.copy())
    if not isinstance(raw, pd.DataFrame) or not {"id", "people"}.issubset(raw.columns):
        raise TypeError("Champion predict_fold must return a DataFrame with id and people")
    if raw.id.duplicated().any():
        raise ValueError("Champion predicted duplicate IDs")
    got = targets[["id"]].merge(raw[["id", "people"]], on="id", how="left", validate="one_to_one")
    got["people"] = pd.to_numeric(got.people, errors="coerce")
    if not np.isfinite(got.people.to_numpy(float)).all():
        raise ValueError("Champion generated missing or non-finite predictions")
    return got.rename(columns={"people": "selected_prediction"})


def build_features(released: pd.DataFrame, targets: pd.DataFrame,
                   baseline: pd.DataFrame, champ: pd.DataFrame) -> pd.DataFrame:
    meta = targets.merge(baseline.rename(columns={"people": "reference_prediction"}),
                         on="id", validate="one_to_one")
    meta = meta.merge(champ, on="id", validate="one_to_one")
    diagnostics = []
    for camera, own in meta.groupby("unit", sort=True):
        observed = released.loc[released.unit.eq(camera)].sort_values("minute_of_day")
        t_obs = observed.minute_of_day.to_numpy(int)
        y_obs = observed.people.to_numpy(float)
        minutes = np.sort(own.minute_of_day.to_numpy(int))
        runs: list[np.ndarray] = []
        start = 0
        for i in range(1, len(minutes) + 1):
            if i == len(minutes) or minutes[i] != minutes[i-1] + 1:
                runs.append(minutes[start:i]); start = i
        for run in runs:
            i_l = np.searchsorted(t_obs, run[0], side="left") - 1
            i_r = np.searchsorted(t_obs, run[-1], side="right")
            if i_l < 0 or i_r >= len(t_obs):
                raise AssertionError("Missing fully released two-sided endpoints")
            left_t, right_t = int(t_obs[i_l]), int(t_obs[i_r])
            left_y, right_y = float(y_obs[i_l]), float(y_obs[i_r])
            slope = (right_y-left_y)/(right_t-left_t)
            prev = float(np.mean(y_obs[max(0,i_l-2):i_l+1]))
            nxt = float(np.mean(y_obs[i_r:min(len(y_obs),i_r+3)]))
            for k, minute in enumerate(run, start=1):
                row = own.loc[own.minute_of_day.eq(minute)].iloc[0]
                diagnostics.append({
                    "id": row.id, "recording": row.recording, "camera": camera,
                    "minute_of_day": int(minute), "block_start": int(run[0]),
                    "block_length": int(len(run)), "gap_position": k,
                    "gap_position_fraction": k/(len(run)+1),
                    "left_count": left_y, "right_count": right_y,
                    "endpoint_slope_per_min": slope,
                    "flow_pattern": flow_pattern(slope),
                    "local_prev_mean": prev, "local_next_mean": nxt,
                    "selected_prediction": float(row.selected_prediction),
                    "reference_prediction": float(row.reference_prediction),
                })
    result = pd.DataFrame(diagnostics)
    if len(result) != len(targets):
        raise AssertionError("Feature extraction lost pseudo target rows")
    return result.sort_values("id").reset_index(drop=True)


def curvature_and_gate(feature_frame: pd.DataFrame, include_e: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent to the v3.1A label-free features; E transfer is never validated."""
    x = feature_frame.loc[:, FEATURE_COLUMNS]  # excludes truth and error by construction
    cams = (*VALIDATED_CAMERAS, UNVALIDATED_TRANSFER_CAMERA) if include_e else VALIDATED_CAMERAS
    p = x.selected_prediction.to_numpy(float)
    left = x.left_count.to_numpy(float)
    right = x.right_count.to_numpy(float)
    prev = x.local_prev_mean.to_numpy(float)
    nxt = x.local_next_mean.to_numpy(float)
    t = x.gap_position_fraction.to_numpy(float)
    d = x.block_length.to_numpy(float) + 1.0
    secant = (right-left)/d
    left_deriv = (left-prev)/2.0
    right_deriv = (nxt-right)/2.0
    curvature = d*((left_deriv-secant)*t*(1-t)**2 - (right_deriv-secant)*t*t*(1-t))
    gate = (
        x.camera.isin(cams).to_numpy()
        & x.flow_pattern.eq("falling_fast").to_numpy()
        & (x.endpoint_slope_per_min.to_numpy(float) <= -2.0)
        & (x.block_length.to_numpy(float) >= 5)
        & (x.block_length.to_numpy(float) <= 15)
        & np.isfinite(curvature) & np.isfinite(p)
        & (t > 0) & (t < 1)
    )
    return curvature, gate


def apply_fixed_alpha(features: pd.DataFrame, alpha: float, include_e: bool = False) -> tuple[np.ndarray, np.ndarray]:
    curv, gate = curvature_and_gate(features, include_e)
    base = features.selected_prediction.to_numpy(float)
    output = base.copy()
    correction = np.clip(alpha*curv, -MAX_DELTA_PEOPLE, MAX_DELTA_PEOPLE)
    output[gate] = np.maximum(0.0, base[gate]+correction[gate])
    assert np.array_equal(output[~gate], base[~gate]), "Non-gated row changed"
    assert np.isfinite(output).all(), "Non-finite candidate"
    return output, gate


def add_truth_after_prediction(feat: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    scored = feat.merge(gold, on="id", how="left", validate="one_to_one")
    if len(scored) != len(feat) or scored.truth.isna().any():
        raise AssertionError("Not all pseudo targets match gold")
    return scored


def ratios(scored: pd.DataFrame, candidate: np.ndarray) -> dict[str, float]:
    tr = scored.truth.to_numpy(float)
    r = scored.reference_prediction.to_numpy(float)
    p = scored.selected_prediction.to_numpy(float)
    ref_mae = float(np.mean(np.abs(r-tr)))
    if ref_mae <= 0:
        raise AssertionError("Non-positive reference MAE")
    champ_mae = float(np.mean(np.abs(p-tr)))
    cand_mae = float(np.mean(np.abs(candidate-tr)))
    return {"ref_mae": ref_mae, "champ_mae": champ_mae, "cand_mae": cand_mae,
            "champ_ratio": champ_mae/ref_mae, "cand_ratio": cand_mae/ref_mae,
            "missing_ratio_delta": (cand_mae-champ_mae)/ref_mae}


def choose_alpha_on_training(scored_by_recording: dict[str, pd.DataFrame], training_recordings: list[str]) -> tuple[float, list[dict]]:
    """Use only training recordings to calibrate alpha; never inspect holdout truth."""
    table = []
    for alpha in ALPHAS:
        deltas = []
        for rec in training_recordings:
            feat = scored_by_recording[rec]
            candidate, _ = apply_fixed_alpha(feat, alpha)
            deltas.append(ratios(feat, candidate)["missing_ratio_delta"])
        # Penalize magnitudes unless there is clear, consistent gain across recordings.
        mean_delta = float(np.mean(deltas))
        penalty = 0.001 * abs(alpha)/0.10
        eligible = (alpha == 0.0 or (sum(d < -1e-6 for d in deltas) >= max(2, len(deltas)-1)
                   and max(deltas) <= 0.005 and mean_delta < -0.002))
        table.append({"alpha": float(alpha), "train_recordings": training_recordings,
                      "mean_delta": mean_delta, "max_harm": float(max(deltas)),
                      "wins": int(sum(d < -1e-6 for d in deltas)),
                      "eligible": bool(eligible), "penalized_objective": mean_delta+penalty})
    ok = [row for row in table if row["eligible"]]
    best = min(ok, key=lambda row: (row["penalized_objective"], abs(row["alpha"]), row["alpha"]))
    return float(best["alpha"]), table


def aggregate_groups(rec: str, scored: pd.DataFrame, candidate: np.ndarray, alpha: float) -> list[dict]:
    raw = scored.copy()
    raw["new_pred"] = candidate
    raw["old_abs"] = np.abs(raw.selected_prediction-raw.truth)
    raw["new_abs"] = np.abs(raw.new_pred-raw.truth)
    raw["gap_bucket"] = pd.cut(raw.block_length, [0, 7, 11, 15, 1_000],
                              labels=["5-7", "8-11", "12-15", "16+"])
    table = []
    # Denominator is per-recording *all pseudo-gap rows*, not per subgroup.
    ref_mae = float(np.abs(raw.reference_prediction-raw.truth).mean())
    n = len(raw)
    for dim in ("camera", "gap_bucket", "flow_pattern", "block_length"):
        for value, subset in raw.groupby(dim, observed=True, dropna=False):
            table.append({"recording": rec, "alpha": alpha,
                          "dimension": dim, "value": str(value),
                          "n": int(len(subset)),
                          "old_mae": float(subset.old_abs.mean()),
                          "new_mae": float(subset.new_abs.mean()),
                          "missing_ratio_delta_contribution": float((subset.new_abs-subset.old_abs).sum()/(n*ref_mae)),
                          "changed_rows": int((subset.new_pred != subset.selected_prediction).sum())})
    return table


def historical_frozen_preview(root: Path, alpha: float,
                              transfer_e: bool = True) -> dict[str, Any]:
    """After alpha is frozen on older recordings: retrospective display ONLY."""
    input_dir = root / "reports" / "breakthrough_v2"
    missing_file = input_dir / "missing_target_diagnostics.csv"
    task_file = input_dir / "per_task_score_comparisons.csv"
    if not missing_file.is_file() or not task_file.is_file():
        return {"status": "SKIPPED_DIAGNOSTICS_UNAVAILABLE", "reason": "Missing frozen diagnostics/task comparison"}
    diagnostics = pd.read_csv(missing_file)
    comp = pd.read_csv(task_file)
    if len(diagnostics) != 358 or diagnostics.id.duplicated().any():
        raise AssertionError("Frozen Missing replay must contain exactly 358 unique targets")
    if set(diagnostics.recording.unique()) != set(HISTORIC_FROZEN):
        raise AssertionError("Frozen recording identities changed")
    ref_rows = comp.loc[comp.model.eq("verified_champion")].copy()
    if len(ref_rows) != 8 or ref_rows.duplicated(["recording", "task"]).any():
        raise AssertionError("Need exactly 8 champion task/fold ratios")
    # Audit legacy V2 consistency separately; these experiments reportedly leave
    # Forecast unchanged, yet their task-comparison rows show nonzero differences.
    legacy_integrity_warnings = []
    ref_forecast = ref_rows.loc[ref_rows.task.eq("forecast"), ["recording", "mae"]].set_index("recording").mae.to_dict()
    for candidate_model in ("rts_missing_only", "nearest_profile_hidden_only"):
        sub = comp.loc[comp.model.eq(candidate_model) & comp.task.eq("forecast")]
        for row in sub.itertuples():
            difference = float(row.mae-ref_forecast[row.recording])
            if abs(difference) > 1e-9:
                legacy_integrity_warnings.append({"model":candidate_model,
                    "recording":row.recording, "unexpected_forecast_mae_delta":difference})
    reconstructed = float(ref_rows.ratio.sum()/8.0)
    if abs(reconstructed-CHAMPION_SCORE) > 1e-12:
        raise AssertionError(f"Champion not exact: {reconstructed}")
    # verify reference denominators and the original model's normalized contributions.
    n_by_rec = diagnostics.groupby("recording").size().to_dict()
    scale = np.array([1/(8*n_by_rec[rec]*ref) for rec, ref in
                      zip(diagnostics.recording, diagnostics.task_reference_mae)],float)
    gold = diagnostics.truth.to_numpy(float)
    p0 = diagnostics.selected_prediction.to_numpy(float)
    orig = abs(gold-p0)*scale
    if np.max(np.abs(orig-diagnostics.normalized_full_score_contribution.to_numpy(float))) > 1e-10:
        raise AssertionError("Frozen contribution mismatch; do not trust preview")
    # No frozen labels passed to correction: only frozen FEATURE_COLUMNS.
    x = diagnostics[list(FEATURE_COLUMNS)].copy()
    # Gold/diagnostic decoys shouldn't change predictions.
    p_none, none_gate = apply_fixed_alpha(x, alpha, include_e=False)
    x_bait = x.copy(); x_bait["truth"] = np.linspace(-100000,100000,len(x))
    x_bait["abs_error"] = np.arange(len(x), dtype=float)
    q, _ = apply_fixed_alpha(x_bait, alpha, include_e=False)
    if not np.array_equal(p_none, q):
        raise AssertionError("Label invariance failed in frozen correction")
    previews = {}
    for label, is_transfer in [("AF_ONLY_OLD_RECORDING_COVERAGE", False),
                               ("AFE_TRANSFER_E_UNVALIDATED", True)]:
        if is_transfer and not transfer_e:
            continue
        new_pred, gate = apply_fixed_alpha(x, alpha, include_e=is_transfer)
        delta = (abs(gold-new_pred)-abs(gold-p0))*scale
        by_rec = {}
        by_camera = {}
        for rec in HISTORIC_FROZEN:
            mask = diagnostics.recording.eq(rec).to_numpy()
            by_rec[rec] = float(delta[mask].sum())
        for cam in sorted(diagnostics.camera.unique()):
            mask = diagnostics.camera.eq(cam).to_numpy()
            by_camera[cam] = {"n": int(mask.sum()),
                              "changed": int(np.sum((new_pred!=p0)&mask)),
                              "full_score_delta_contribution": float(delta[mask].sum())}
        previews[label] = {
            "status": "REUSED_FROZEN_DEVELOPMENT_ONLY_NOT_INDEPENDENT",
            "alpha_frozen_from_older_recordings": float(alpha),
            "score": float(reconstructed + delta.sum()),
            "score_delta_vs_champion": float(delta.sum()),
            "below_0_600_exploratory_only": bool(reconstructed+delta.sum() < TARGET_SCORE),
            "changed_rows": int(np.sum(new_pred!=p0)),
            "eligible_rows": int(gate.sum()),
            "fold_deltas": by_rec,
            "camera_contributions": by_camera,
            "verified_full_champion_reconstruction": float(reconstructed),
            "promotable": False,
        }
    return {"status": "AVAILABLE_REUSED_FROZEN_ONLY", "previews": previews,
            "v2_forecast_identity_warnings": legacy_integrity_warnings,
            "diagnostics_sha256": digest(missing_file), "task_comparisons_sha256": digest(task_file)}


def run(root: Path, output_dir: Path | None = None,
        smoke_linear: bool = False, skip_frozen: bool = False) -> dict:
    root = root.resolve()
    output_dir = output_dir or (root / "reports" / ("v3_2_smoke" if smoke_linear else "v3_2"))
    assert_output_dir_safe(root, output_dir)
    data, train_path = read_train(root)
    pred_fn, champion_identity = load_champion(root, smoke_linear)
    print("V3.2 RESEARCH ONLY — NO KAGGLE SUBMISSION", flush=True)
    print("CHAMPION:", champion_identity["kind"], flush=True)
    print("OLD RECORDINGS:", list(OLD_RECORDINGS), flush=True)
    print("No old recording contains cam_E — A/F validated; E transfer exploratory only.", flush=True)
    summaries: dict[str, Any] = {}
    scored_by_rec: dict[str, pd.DataFrame] = {}
    block_coverage = {}
    for rec in OLD_RECORDINGS:
        full = data.loc[data.recording.eq(rec)].copy()
        train_ctx = data.loc[~data.recording.eq(rec)].copy()
        blocks, coverage = choose_pseudo_blocks(full, rec)
        released, targets, gold, baseline = make_pseudogap_inputs(full, blocks)
        print(f"{rec}: {len(blocks)} blocks, {len(targets)} missing rows, calling champion...",flush=True)
        champ = get_predictions(pred_fn, train_ctx, released, targets, baseline)
        features = build_features(released, targets, baseline, champ)
        curv, gated = curvature_and_gate(features)
        if not np.isfinite(curv).all():
            raise ValueError("Nonfinite curvature features on released context")
        # Quarantine label join until all predictions & features are created.
        scored = add_truth_after_prediction(features, gold)
        scored_by_rec[rec] = scored
        base = ratios(scored, scored.selected_prediction.to_numpy(float))
        block_coverage[rec] = coverage
        summaries[rec] = {"blocks": int(len(blocks)), "rows": int(len(targets)),
                          "eligible_A_F_fast_rows": int(gated.sum()),
                          "baseline_missing_reference_mae": base["ref_mae"],
                          "champion_missing_mae": base["champ_mae"],
                          "champion_missing_ratio": base["champ_ratio"],
                          "source": "RECORDING_HELD_OUT_BUT_PREVIOUSLY_USED_IN_RTS_RESEARCH",
                          "block_length_distribution": {
                              str(k): int(v) for k,v in pd.Series([b.length for b in blocks]).value_counts().sort_index().items()},
                          "coverage_by_stratum": coverage}
    # Outer cross-fitting: outer fold's gold never inspected by choose_alpha_on_training.
    fold_results = []
    groups = []
    calibration_reports = []
    for holdout in OLD_RECORDINGS:
        others = [r for r in OLD_RECORDINGS if r != holdout]
        alpha, choices = choose_alpha_on_training(scored_by_rec, others)
        scored = scored_by_rec[holdout]
        new, gated = apply_fixed_alpha(scored, alpha)
        metrics = ratios(scored, new)
        fold_results.append({"recording": holdout, "alpha_chosen_without_holdout_gold": alpha,
                             "n": len(scored), "gated_n": int(gated.sum()),
                             "changed_n": int(np.sum(new!=scored.selected_prediction.to_numpy(float))),
                             **metrics})
        groups.extend(aggregate_groups(holdout, scored, new, alpha))
        calibration_reports.append({"outer_holdout": holdout,
                                    "training_recordings": others,
                                    "selection_grid": choices,
                                    "selected_alpha": alpha})
    # Pre-registered guardrail; not a claim of untouched independent confirmation.
    fold_deltas = [r["missing_ratio_delta"] for r in fold_results]
    mean_delta = float(np.mean(fold_deltas))
    wins = sum(d < -1e-6 for d in fold_deltas)
    max_harm = max(fold_deltas)
    supportive = bool(wins >= 3 and max_harm <= 0.01 and mean_delta <= -0.005)
    all_alpha, full_choice_grid = choose_alpha_on_training(scored_by_rec, list(OLD_RECORDINGS))
    frozen_config = {"alpha": all_alpha, "gate_cameras": list(VALIDATED_CAMERAS),
                     "max_change_people": MAX_DELTA_PEOPLE,
                     "selection_source": "FOUR_OLDER_RECORDINGS_ONLY",
                     "pass_outer_guardrail": supportive}
    # Freeze alpha & external verdict BEFORE loading or scoring frozen diagnostics.
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "calibrated_alpha_from_older_only.json"
    config_path.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")
    config_sha = digest(config_path)
    frozen = (historical_frozen_preview(root, all_alpha, transfer_e=True)
              if not skip_frozen and not smoke_linear else
              {"status": "SKIPPED_AS_REQUESTED_OR_SMOKE"})
    full = {
        "experiment": "V3_2_RECORDING_DISJOINT_HISTORICALLY_REUSED_MISSING_LOSO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RESEARCH_ONLY_CHAMPION_PRESERVED_NO_SUBMISSION",
        "champion_score_frozen_reference": CHAMPION_SCORE,
        "official_kaggle_best_user_report": 0.729,
        "target": TARGET_SCORE,
        "champion_source": champion_identity,
        "source_train": {"path": str(train_path), "sha256": digest(train_path),
                         "rows": len(data), "recordings": sorted(data.recording.unique().tolist())},
        "protocol": {"seed": SEED, "strata": [list(x[:2])+[list(x[2])] for x in STRATA],
                     "alpha_grid": ALPHAS, "outer_recordings": OLD_RECORDINGS,
                     "gate_cameras": VALIDATED_CAMERAS,
                     "cam_E_support": "ZERO_IN_ALL_FOUR_OLD_RECORDINGS",
                     "same_recording_hidden_truth_used_as_features": False,
                     "two_sided_missing_context_allowed": True},
        "holdout_recordings": summaries,
        "outer_results": fold_results,
        "mean_outer_missing_ratio_delta": mean_delta,
        "outer_wins": wins, "outer_max_harm": float(max_harm),
        "outer_guardrail_passed": supportive,
        "calibration": calibration_reports,
        "final_four_recording_grid": full_choice_grid,
        "frozen_alpha_config": frozen_config,
        "config_sha256_at_freeze": config_sha,
        "frozen_dev_preview": frozen,
        "decision": (
            "SMOKE_ONLY_NO_CHAMPION_COMPARISON" if smoke_linear else
            "NO_PROMOTION_HISTORIC_HOLDOUT_REUSE_AND_NO_CAM_E_SUPPORT"
        ),
        "source_caveats": [
            "All four older records have been examined in previous RTS research; these are recording-disjoint callback tests, not untouched confirmation.",
            "Camera E is absent in all older records; transferring A/F correction to E is unvalidated.",
            "The two main Friday/Saturday frozen folds have been repeatedly inspected; any preview score is retrospective only.",
            "The exact 0.605700 score cannot be inferred from pseudo-gap scores; it is only reconstructed from frozen diagnostics when present.",
            "No official Kaggle validation performed; no submission generated or uploaded.",
        ],
    }
    (output_dir / "v3_2_summary.json").write_text(json.dumps(full, indent=2, allow_nan=False),encoding="utf-8")
    pd.DataFrame(fold_results).to_csv(output_dir / "v3_2_outer_holdouts.csv", index=False)
    pd.DataFrame(groups).to_csv(output_dir / "v3_2_group_metrics.csv", index=False)
    lines = ["# V3.2 Missing correction — decision", "",
             "**Preserve Champion 0.6057000521183546. No Kaggle submission created.**", "",
             f"Source: `{champion_identity['kind']}`.",
             "", f"Outer held-out-recording wins: **{wins}/4**; mean Missing ratio delta **{mean_delta:+.6f}**; worst heldout ratio delta **{max_harm:+.6f}**.",
             f"Outer consistency gate passed: **{supportive}**. Final older-recording alpha: **{all_alpha:+.3f}**.",
             "", "## Recording-disjoint results (historically reused old recordings)", "",
             "| Recording | Targets | Gated | Alpha | Champion Missing ratio | Delta (negative=better) |", "|---|---:|---:|---:|---:|---:|"]
    for row in fold_results:
        lines.append(f"| {row['recording']} | {row['n']} | {row['gated_n']} | {row['alpha_chosen_without_holdout_gold']:+.2f} | {row['champ_ratio']:.5f} | {row['missing_ratio_delta']:+.5f} |")
    lines.extend(["", "## Limits", "",
        "- Older recordings were **already used** in other research, so the tests are not untouched holdouts.",
        "- No cam_E in older recordings: correction for E is an unvalidated transfer.",
        "- Frozen Friday/Saturday scores are retrospective; do not use them to choose alpha.",
        "- The legacy `evaluate_full_solution.py` is not the verified selected callback.",
        "- V2 Forecast identity mismatches require resolution before any future promotion.",
        "- No submitted CSV, model overwrite, Kaggle interaction, or selected-prediction change was produced.", ""])
    if frozen.get("previews"):
        lines.extend(["## Reused frozen-fold preview (NOT independent validation)", "",
                      "| Scenario | Score | Delta | Changed | Friday delta | Saturday delta |", "|---|---:|---:|---:|---:|---:|"])
        for name, row in frozen["previews"].items():
            fr = row["fold_deltas"][HISTORIC_FROZEN[0]]
            sa = row["fold_deltas"][HISTORIC_FROZEN[1]]
            lines.append(f"| {name} | {row['score']:.9f} | {row['score_delta_vs_champion']:+.9f} | {row['changed_rows']} | {fr:+.9f} | {sa:+.9f} |")
    lines += ["", "**Final: No change to the established Champion.**", ""]
    (output_dir / "v3_2_decision.md").write_text("\n".join(lines),encoding="utf-8")
    print("Outer LO(R)O mean Missing ratio change:", f"{mean_delta:+.6f}",flush=True)
    print("Outer improvement counts:", wins, "/4; guardrail", supportive,flush=True)
    print("Research reports:", output_dir,flush=True)
    print("CHAMPION_PRESERVED; SUBMISSION_GENERATED=False; OFFICIAL_SCORE_NEW=None",flush=True)
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-linear", action="store_true",
                        help="Use linear interpolation only to smoke test. This is NOT the champion.")
    parser.add_argument("--skip-frozen", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output_dir is None:
        out = None
    else:
        out = args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
    try:
        run(root, out, args.smoke_linear, args.skip_frozen)
    except Exception as exc:
        print(f"V3.2 STOPPED WITHOUT SUBMISSION: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
