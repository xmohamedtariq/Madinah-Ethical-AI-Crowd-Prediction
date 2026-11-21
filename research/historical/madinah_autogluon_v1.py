r"""Leakage-aware AutoGluon local experiment for Madinah Ethical AI.

Runs only LOCAL diagnostics / rolling-date validation. Never submits to Kaggle.
The displayed error is MAE in people counts, not necessarily the Kaggle score.

Usage from the project root, Windows PowerShell:
  .\.venv-ag\Scripts\python.exe .\madinah_autogluon_v1.py --mode audit
  .\.venv-ag\Scripts\python.exe .\madinah_autogluon_v1.py --mode dry-run
  .\.venv-ag\Scripts\python.exe .\madinah_autogluon_v1.py --mode validate --time-limit 300
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / "data" / "official"
REPORTS = ROOT / "reports"
REQUIRED = {"recording", "day", "weekday", "unit", "minute_of_day", "people"}
KINDS = {"forecast", "hidden", "hidden-future", "missing"}
MAX_CONTEXT_GAP = 240


def read_data():
    files = [OFFICIAL / name for name in ("train.csv", "test.csv", "sample_submission.csv")]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Official input files missing: {missing}")
    tr, te, sub = (pd.read_csv(f) for f in files)
    for name, frame in (("train", tr), ("test", te)):
        need = REQUIRED - set(frame.columns)
        if need:
            raise ValueError(f"{name}.csv is missing columns: {sorted(need)}")
        frame["minute_of_day"] = pd.to_numeric(frame["minute_of_day"], errors="coerce")
        frame["people"] = pd.to_numeric(frame["people"], errors="coerce")
        frame["day"] = frame["day"].astype("string")
        frame["unit"] = frame["unit"].astype("string")
        frame["weekday"] = frame["weekday"].astype("string")
        frame["recording"] = frame["recording"].astype("string")
    tr = tr.dropna(subset=list(REQUIRED)).copy()
    if not {"id", "people"}.issubset(sub.columns):
        raise ValueError("sample_submission.csv needs columns: id, people")
    if not {"id", "kind", "now_minute_of_day"}.issubset(te.columns):
        raise ValueError("test.csv needs id, kind, now_minute_of_day")
    ids = te["id"].notna() & te["id"].astype(str).str.strip().ne("")
    targets = te.loc[ids].copy()
    targets["kind"] = (
        targets["kind"].astype("string").str.lower().str.strip().str.replace("_", "-", regex=False)
    )
    bad_kinds = sorted(set(targets["kind"].dropna()) - KINDS)
    if bad_kinds or targets["kind"].isna().any():
        raise ValueError(f"Unknown test kinds: {bad_kinds}; null kinds: {int(targets['kind'].isna().sum())}")
    targets["now_minute_of_day"] = pd.to_numeric(targets["now_minute_of_day"], errors="coerce")
    targets["horizon"] = pd.to_numeric(targets.get("horizon", np.nan), errors="coerce")
    fh = targets["kind"].isin(["forecast", "hidden-future"])
    if targets.loc[fh, "now_minute_of_day"].isna().any():
        raise ValueError("Forecast/hidden-future rows without now_minute_of_day; cannot prevent leakage.")
    if (targets.loc[fh, "now_minute_of_day"] > targets.loc[fh, "minute_of_day"]).any():
        raise ValueError("A forecast target has now later than its target minute. Inspect data.")
    derived_horizon = targets["minute_of_day"] - targets["now_minute_of_day"]
    targets["horizon"] = targets["horizon"].fillna(derived_horizon)
    targets.loc[~fh, "horizon"] = 0.0
    targets.loc[~fh, "now_minute_of_day"] = np.nan
    submitted_ids = sub["id"].astype("string")
    target_ids = targets["id"].astype("string")
    if submitted_ids.duplicated().any() or target_ids.duplicated().any() or set(submitted_ids) != set(target_ids):
        raise ValueError("Sample-submission ID set does not match unknown test target ID set.")
    # Only released observation rows are used as contextual evidence.
    context = te.loc[~ids & te["people"].notna()].dropna(subset=list(REQUIRED)).copy()
    return tr, te, sub, targets, context


def audit(tr, te, sample, targets, context):
    print("\n=== OFFICIAL DATA AUDIT ===")
    print("Train labeled rows:", len(tr), "| Test rows:", len(te))
    print("Test target IDs:", len(targets), "| Released context rows:", len(context))
    print("Sample submission rows:", len(sample))
    print("Train days:", sorted(tr["day"].unique().tolist()))
    print("Test days:", sorted(te["day"].dropna().unique().tolist()))
    print("\nTest targets by kind:\n", targets.groupby("kind").size().to_string())
    print("\nTest targets by (weekday, kind):\n", targets.groupby(["weekday", "kind"]).size().to_string())
    print("\nUnique train cameras:", tr["unit"].nunique(), "| Test cameras:", te["unit"].nunique())
    print("Test target IDs verified against sample_submission.csv.")


def prepare_validation(val_labeled, test_targets):
    """Simulate test hole geometry on a completely held-out historical day."""
    keys = ["weekday", "unit", "minute_of_day"]
    schedule = test_targets[keys + ["kind", "now_minute_of_day", "horizon"]].copy()
    dup = int(schedule.duplicated(keys).sum())
    if dup:
        print(f"WARNING: {dup} duplicate weekday/camera/minute templates; taking first.")
        schedule = schedule.drop_duplicates(keys, keep="first")
    withheld = val_labeled.merge(schedule, on=keys, how="inner", validate="many_to_one")
    if withheld.empty:
        return withheld, val_labeled.copy(), 0, len(schedule)
    marked = withheld[["recording", "unit", "minute_of_day"]].drop_duplicates()
    marked["_withheld"] = 1
    visible = val_labeled.merge(
        marked, on=["recording", "unit", "minute_of_day"], how="left", validate="many_to_one"
    )
    visible = visible.loc[visible["_withheld"].isna()].drop(columns="_withheld")
    wday = str(val_labeled["weekday"].iloc[0])
    requested = int((schedule["weekday"] == wday).sum())
    matched = int(withheld[keys].drop_duplicates().shape[0])
    return withheld, visible, matched, requested


def generate_train_holes(labeled, test_targets, seed=42, frac=0.48):
    """Make synthetic holes without ever using a target's own label as input."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(labeled)) < frac
    if int(mask.sum()) < 100:
        raise ValueError("Too few synthetic training holes; need more released training rows.")
    withheld = labeled.loc[mask].copy()
    visible = labeled.loc[~mask].copy()
    probs = test_targets["kind"].value_counts(normalize=True)
    kinds = probs.index.to_numpy(dtype=str)
    withheld["kind"] = rng.choice(kinds, size=len(withheld), p=probs.to_numpy())
    withheld["now_minute_of_day"] = np.nan
    withheld["horizon"] = 0.0
    for kind in ["forecast", "hidden-future"]:
        selected = withheld["kind"] == kind
        if not selected.any():
            continue
        actual = test_targets.loc[test_targets["kind"] == kind, "horizon"].dropna()
        positive = actual.loc[actual > 0].to_numpy(dtype=float)
        if not len(positive):
            positive = np.arange(1, 16, dtype=float)
        horizons = rng.choice(positive, size=int(selected.sum()), replace=True)
        withheld.loc[selected, "horizon"] = horizons
        withheld.loc[selected, "now_minute_of_day"] = (
            withheld.loc[selected, "minute_of_day"].to_numpy() - horizons
        )
    return withheld, visible


def obs_index(visible):
    """Keyed only by observations explicitly available for this scenario."""
    index = defaultdict(dict)
    for (rec, cam), group in visible.groupby(["recording", "unit"], observed=True):
        group = group.sort_values("minute_of_day")
        index[str(rec)][str(cam)] = (
            group["minute_of_day"].to_numpy(dtype=float),
            group["people"].to_numpy(dtype=float),
        )
    return index


def before(times, values, pivot, strict=False):
    pos = int(np.searchsorted(times, pivot, side="left" if strict else "right")) - 1
    if pos < 0:
        return np.nan, np.nan, pos
    gap = float(pivot - times[pos])
    return (float(values[pos]), gap, pos) if gap <= MAX_CONTEXT_GAP else (np.nan, np.nan, pos)


def after(times, values, pivot):
    pos = int(np.searchsorted(times, pivot, side="right"))
    if pos >= len(times):
        return np.nan, np.nan
    gap = float(times[pos] - pivot)
    return (float(values[pos]), gap) if gap <= MAX_CONTEXT_GAP else (np.nan, np.nan)


def rec_start(rec):
    try:
        hhmm = rec.split("_")[1]
        return int(hhmm[:2]) * 60 + int(hhmm[2:4])
    except (IndexError, ValueError):
        return np.nan


def make_features(withheld, visible, cameras):
    index = obs_index(visible)
    out = []
    for row in withheld.itertuples(index=False):
        kind = str(row.kind)
        t = float(row.minute_of_day)
        forecast_like = kind in ("forecast", "hidden-future")
        now = float(row.now_minute_of_day) if forecast_like else np.nan
        if forecast_like and (not np.isfinite(now) or now > t):
            raise ValueError(f"Invalid now for {row.recording}/{row.unit}/{t}")
        ref = now if forecast_like else t
        cam_name = str(row.unit)
        day_start = rec_start(str(row.recording))
        f = {
            "unit": cam_name,
            "weekday": str(row.weekday),
            "kind": kind,
            "minute_of_day": t,
            "hour": float(math.floor(t / 60)),
            "time_sin": math.sin(2 * math.pi * t / 1440),
            "time_cos": math.cos(2 * math.pi * t / 1440),
            "recording_start": day_start,
            "minutes_from_recording_start": t - day_start if np.isfinite(day_start) else np.nan,
            "now_minute": now,
            "horizon": float(row.horizon) if forecast_like else 0.0,
        }
        available_last = []
        simultaneous = []
        for camera in cameras:
            # Hidden cameras cannot use any measurements of their own camera.
            hidden_own = camera == cam_name and kind in ("hidden", "hidden-future")
            pair = index.get(str(row.recording), {}).get(camera)
            last, age, next_value, next_age = np.nan, np.nan, np.nan, np.nan
            if pair is not None and not hidden_own:
                times, values = pair
                last, age, _ = before(times, values, ref, strict=(camera == cam_name and not forecast_like))
                if not forecast_like:
                    # Future observations are forbidden for forecast and hidden-future.
                    next_value, next_age = after(times, values, t)
            f[f"last_{camera}"] = last
            f[f"age_{camera}"] = age
            f[f"next_{camera}"] = next_value
            f[f"next_gap_{camera}"] = next_age
            if camera != cam_name and np.isfinite(last):
                available_last.append(last)
                if age == 0:
                    simultaneous.append(last)
        f["other_cameras_available"] = float(len(available_last))
        f["other_camera_mean"] = float(np.mean(available_last)) if available_last else np.nan
        f["other_camera_std"] = float(np.std(available_last)) if len(available_last) > 1 else np.nan
        f["simultaneous_cameras"] = float(len(simultaneous))
        f["simultaneous_mean"] = float(np.mean(simultaneous)) if simultaneous else np.nan
        out.append(f)
    return pd.DataFrame(out)


def historical_baseline(history, withheld):
    """Transparent historical same-camera/weekday-nearby-minute baseline."""
    source = {}
    for (weekday, cam), chunk in history.groupby(["weekday", "unit"], observed=True):
        source[(str(weekday), str(cam))] = (
            chunk["minute_of_day"].to_numpy(dtype=float), chunk["people"].to_numpy(dtype=float)
        )
    cam_mean = history.groupby("unit", observed=True)["people"].median().to_dict()
    fallback = float(history["people"].median())
    outputs = []
    for row in withheld.itertuples(index=False):
        key = (str(row.weekday), str(row.unit))
        ts, ys = source.get(key, (np.array([]), np.array([])))
        if len(ts):
            dist = np.abs(ts - float(row.minute_of_day))
            neighbors = ys[np.argsort(dist)[:min(12, len(ts))]]
            prediction = float(np.median(neighbors))
        else:
            prediction = float(cam_mean.get(str(row.unit), fallback))
        outputs.append(max(0.0, prediction))
    return np.asarray(outputs, dtype=float)


def get_folds(train, targets, max_folds):
    # Latest labeled day of each test weekday; strict training rows before that day.
    test_weekdays = set(targets["weekday"].dropna().astype(str))
    folds = []
    for weekday in sorted(test_weekdays):
        days = sorted(train.loc[train["weekday"] == weekday, "day"].dropna().unique().tolist())
        if len(days) < 2:
            print(f"WARNING: weekday {weekday} has fewer than 2 historical days; skip CV.")
            continue
        day = str(days[-1])
        prior = train.loc[train["day"] < day].copy()
        heldout = train.loc[train["day"] == day].copy()
        if len(prior) < 100 or len(heldout) < 50:
            print(f"WARNING: insufficient historical/heldout rows for {day}; skip CV.")
            continue
        folds.append((day, prior, heldout))
    return folds[:max_folds]


def run(args):
    tr, te, sample, test_targets, test_context = read_data()
    audit(tr, te, sample, test_targets, test_context)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        return
    folds = get_folds(tr, test_targets, args.max_folds)
    if not folds:
        raise RuntimeError("Cannot construct forward-date validation folds with the available history.")
    cameras = sorted(set(tr["unit"].astype(str)).union(set(te["unit"].dropna().astype(str))))
    summaries, detailed = [], []
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for day, hist, val in folds:
        print(f"\n=== FORWARD-DATE HOLDOUT: {day} ===")
        withheld, visible_val, matched, requested = prepare_validation(val, test_targets)
        print(f"Template targets matched: {matched}/{requested} ({100*matched/max(1,requested):.1f}%)")
        weekday = str(val["weekday"].iloc[0])
        reference_counts = test_targets.loc[test_targets["weekday"] == weekday, "kind"].value_counts()
        matched_counts = withheld["kind"].value_counts() if not withheld.empty else pd.Series(dtype="int64")
        for kind in sorted(set(reference_counts.index)):
            n_ref, n_got = int(reference_counts[kind]), int(matched_counts.get(kind, 0))
            print(f"  {kind}: matched {n_got}/{n_ref} ({100*n_got/max(1,n_ref):.1f}%)")
        print(f"Prior historical rows: {len(hist)} | held-out targets: {len(withheld)} | visible context: {len(visible_val)}")
        if len(withheld) < 30:
            print("SKIP: too few matched targets for meaningful validation.")
            continue
        pseudo_targets, pseudo_context = generate_train_holes(
            hist, test_targets, seed=args.seed + int(day[-2:]), frac=args.train_fraction
        )
        train_features = make_features(pseudo_targets, pseudo_context, cameras)
        validation_features = make_features(withheld, visible_val, cameras)
        ytrain = pseudo_targets["people"].to_numpy(dtype=float)
        yvalid = withheld["people"].to_numpy(dtype=float)
        baseline = historical_baseline(hist, withheld)
        base_mae = float(np.mean(np.abs(yvalid - baseline)))
        print(f"Train synthetic holes: {len(train_features)} | Features: {train_features.shape[1]}")
        print(f"Prior-days historical reference MAE: {base_mae:.6f} people")
        if args.mode == "dry-run":
            continue
        from autogluon.tabular import TabularPredictor
        train_features["people"] = ytrain
        model_dir = ROOT / "models" / "ag_strict_cv" / run_label / day
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        if model_dir.exists():
            raise FileExistsError(f"Refusing to silently reuse/overwrite existing model: {model_dir}")
        predictor = TabularPredictor(
            label="people",
            problem_type="regression",
            eval_metric="mean_absolute_error",
            path=str(model_dir),
        ).fit(
            train_data=train_features,
            presets="medium_quality",
            hyperparameters={"GBM": {}, "CAT": {}, "XGB": {}, "RF": {}},
            time_limit=args.time_limit,
            num_gpus=0,
        )
        prediction = np.clip(np.asarray(predictor.predict(validation_features), dtype=float), 0.0, None)
        mae = float(np.mean(np.abs(yvalid - prediction)))
        gain_pct = 100 * (base_mae - mae) / base_mae if base_mae > 0 else np.nan
        print(f"AutoGluon held-out MAE: {mae:.6f} people | vs historical baseline: {gain_pct:+.2f}%")
        model_board = predictor.leaderboard(silent=True)
        model_board.to_csv(REPORTS / f"ag_leaderboard_{day}.csv", index=False)
        result = withheld[["recording", "day", "weekday", "unit", "minute_of_day", "kind", "people"]].copy()
        result["prediction"] = prediction
        result["historical_prediction"] = baseline
        result["abs_error"] = np.abs(yvalid - prediction)
        result["baseline_abs_error"] = np.abs(yvalid - baseline)
        result.to_csv(REPORTS / f"ag_holdout_predictions_{day}.csv", index=False)
        detailed.append(result)
        summaries.append({
            "day": day,
            "n_val": len(withheld),
            "template_matched": matched,
            "template_requested": requested,
            "coverage_pct": round(100 * matched / max(1, requested), 2),
            "mae_people": mae,
            "historical_baseline_mae_people": base_mae,
            "relative_improvement_pct": gain_pct,
            "model_dir": str(model_dir),
        })
    if args.mode == "dry-run":
        print("\nDRY-RUN PASSED: no model trained, no Kaggle submission created.")
        return
    if not detailed:
        raise RuntimeError("No valid folds completed; no results or submission produced.")
    scored = pd.concat(detailed, ignore_index=True)
    per_kind = scored.groupby("kind", observed=True).agg(
        n=("people", "size"),
        model_mae_people=("abs_error", "mean"),
        baseline_mae_people=("baseline_abs_error", "mean"),
    )
    per_kind["gain_pct"] = 100 * (per_kind["baseline_mae_people"] - per_kind["model_mae_people"]) / per_kind["baseline_mae_people"]
    per_kind.to_csv(REPORTS / "ag_strict_per_kind.csv")
    summary = {
        "warning": "This is held-out synthetic target MAE in people counts, NOT an official Kaggle leaderboard score.",
        "rules": "Only earlier days used in supervised fit; forecast/hidden-future observations filtered at now; withheld validation rows unavailable in context.",
        "model_mae_people": float(scored["abs_error"].mean()),
        "historical_baseline_mae_people": float(scored["baseline_abs_error"].mean()),
        "folds": summaries,
        "per_kind": per_kind.reset_index().to_dict(orient="records"),
    }
    (REPORTS / "ag_strict_validation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== LOCAL VALIDATION RESULTS (MAE PEOPLE, NOT KAGGLE SCORE) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved: reports/ag_strict_validation.json and reports/ag_strict_per_kind.csv")
    print("No Kaggle submission file was created or uploaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "dry-run", "validate"], default="audit")
    parser.add_argument("--time-limit", type=int, default=300, help="seconds per held-out fold")
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.48)
    args = parser.parse_args()
    if not 0.1 <= args.train_fraction <= 0.9:
        raise ValueError("train-fraction should be between 0.1 and 0.9")
    run(args)
