from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"
VALIDATION_FILE = ROOT / "reports" / "v11_2_validation_predictions.csv"
REPORT_FILE = ROOT / "reports" / "v19_hidden_analog_day_oof.csv"
BASE_COL = "blend_extra_leaf3_a0.5"
TEST_TO_VALIDATION = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
TOP_K = 3
SOFTMAX_TEMPERATURE = 0.20


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    return pd.read_csv(path)


def normalize(frame: pd.DataFrame, name: str, needs_people: bool) -> pd.DataFrame:
    x = frame.copy()
    required = {"recording", "unit", "minute_of_day"}
    if needs_people:
        required.add("people")
    missing = required - set(x.columns)
    if missing:
        raise SystemExit(f"{name} missing columns: {sorted(missing)}")
    x["recording"] = x["recording"].astype(str)
    x["unit"] = x["unit"].astype(str)
    x["minute_of_day"] = pd.to_numeric(x["minute_of_day"], errors="coerce")
    if needs_people:
        x["people"] = pd.to_numeric(x["people"], errors="coerce")
    x = x.loc[x["minute_of_day"].notna()].copy()
    x["minute_of_day"] = x["minute_of_day"].astype(int)
    return x


def profile(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame.pivot_table(
        index="minute_of_day", columns="unit", values="people", aggfunc="first"
    ).sort_index()
    p.index = p.index.astype(int)
    return p


def interp_inside(series: pd.Series, minute: int) -> float:
    s = series.dropna().sort_index()
    if s.empty or minute < int(s.index.min()) or minute > int(s.index.max()):
        return np.nan
    return float(np.interp(float(minute), s.index.to_numpy(float), s.to_numpy(float)))


def build_stats(donor_profiles: dict[str, pd.DataFrame], units: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for unit in units:
        vals = []
        for p in donor_profiles.values():
            if unit in p:
                vals.extend(np.log1p(np.maximum(0.0, p[unit].dropna().to_numpy(float))).tolist())
        if vals:
            arr = np.asarray(vals, dtype=float)
            q25, q75 = np.quantile(arr, [0.25, 0.75])
            stats[unit] = (float(np.median(arr)), max(0.25, float(q75 - q25)))
    return stats


def profile_distance(
    current: pd.DataFrame,
    donor: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    exclude_unit: str,
) -> float:
    errors = []
    for unit in current.columns.intersection(donor.columns):
        if unit == exclude_unit or unit not in stats:
            continue
        pair = pd.concat(
            [current[unit].rename("cur"), donor[unit].rename("donor")], axis=1
        ).dropna()
        if len(pair) < 20:
            continue
        center, scale = stats[unit]
        a = (np.log1p(np.maximum(0.0, pair["cur"].to_numpy(float))) - center) / scale
        b = (np.log1p(np.maximum(0.0, pair["donor"].to_numpy(float))) - center) / scale
        errors.append(float(np.median(np.abs(a - b))))
    return float(np.median(errors)) if errors else np.inf


def candidate_prediction(
    current: pd.DataFrame,
    donor_profiles: dict[str, pd.DataFrame],
    distances: dict[str, float],
    target_unit: str,
    minute: int,
) -> tuple[float, float, list[tuple[str, float]]]:
    eligible = [
        (rec, dist) for rec, dist in distances.items()
        if np.isfinite(dist) and target_unit in donor_profiles[rec]
    ]
    eligible.sort(key=lambda x: x[1])
    eligible = eligible[:TOP_K]
    if not eligible:
        return np.nan, 1.0, []

    d0 = eligible[0][1]
    raw_w = np.asarray([np.exp(-(d - d0) / SOFTMAX_TEMPERATURE) for _, d in eligible], dtype=float)
    weights = raw_w / raw_w.sum()
    chosen = [(rec, float(w)) for (rec, _), w in zip(eligible, weights)]

    # Blend historical analog curves at the target minute.
    analog = 0.0
    valid_weight = 0.0
    for rec, w in chosen:
        value = interp_inside(donor_profiles[rec][target_unit], minute)
        if np.isfinite(value):
            analog += w * value
            valid_weight += w
    if valid_weight <= 0:
        return np.nan, 1.0, chosen
    analog /= valid_weight

    # Calibrate analog level to the target camera's released current-day cells.
    if target_unit not in current.columns:
        return float(analog), 1.0, chosen
    obs = current[target_unit].dropna()
    ratios = []
    for minute_obs, value_obs in obs.items():
        donor_at_obs = 0.0
        donor_weight = 0.0
        for rec, w in chosen:
            value = interp_inside(donor_profiles[rec][target_unit], int(minute_obs))
            if np.isfinite(value):
                donor_at_obs += w * value
                donor_weight += w
        if donor_weight and value_obs >= 0:
            donor_at_obs /= donor_weight
            ratios.append(np.log((float(value_obs) + 5.0) / (donor_at_obs + 5.0)))
    if len(ratios) >= 10:
        scale = float(np.exp(np.clip(np.median(ratios), np.log(0.70), np.log(1.30))))
    else:
        scale = 1.0
    calibrated = max(0.0, (analog + 5.0) * scale - 5.0)
    return calibrated, scale, chosen


def main() -> None:
    train = normalize(read(TRAIN_FILE), "train.csv", True)
    test = normalize(read(TEST_FILE), "test.csv", True)
    valid = normalize(read(VALIDATION_FILE), "v11_2 validation", False)
    if "truth" not in valid or BASE_COL not in valid:
        raise SystemExit("Validation report must contain truth and baseline columns")
    valid["truth"] = pd.to_numeric(valid["truth"], errors="coerce")
    valid[BASE_COL] = pd.to_numeric(valid[BASE_COL], errors="coerce")
    valid = valid.loc[valid["truth"].notna() & valid[BASE_COL].notna()].copy().reset_index(drop=True)
    train = train.loc[train["people"].notna()].copy()

    # Mask every test-target cell when constructing the pseudo-test current-day profile.
    blocked = test.loc[test["people"].isna(), ["recording", "unit", "minute_of_day"]].copy()
    blocked["recording"] = blocked["recording"].map(TEST_TO_VALIDATION)
    blocked = blocked.dropna(subset=["recording"])
    blocked_keys = set(zip(blocked["recording"], blocked["unit"], blocked["minute_of_day"]))

    validation_keys = set(zip(valid["recording"], valid["unit"], valid["minute_of_day"]))
    unmasked_validation = validation_keys - blocked_keys
    if unmasked_validation:
        raise SystemExit(
            f"Leakage guard failed: {len(unmasked_validation)} validation targets "
            "are not covered by the mapped test-target mask"
        )
    print(f"Leakage guard: all {len(validation_keys)} validation targets are masked from current-day profiles")

    train["key"] = list(zip(train["recording"], train["unit"], train["minute_of_day"]))
    current_rows = train.loc[train["key"].map(lambda k: k not in blocked_keys)].copy()
    train_profiles = {
        rec: profile(g)
        for rec, g in train.groupby("recording")
    }

    pred_analog = np.full(len(valid), np.nan)
    pred_unscaled = np.full(len(valid), np.nan)
    diagnostics = []

    for heldout in sorted(valid["recording"].unique()):
        date = datetime.strptime(heldout[:8], "%Y%m%d")
        weekday = date.weekday()
        donor_recs = [
            rec for rec in train_profiles
            if rec != heldout and datetime.strptime(rec[:8], "%Y%m%d").weekday() == weekday
        ]
        # If the training set has no same-weekday analogs, fall back to any prior recording.
        if not donor_recs:
            donor_recs = [rec for rec in train_profiles if rec != heldout]
        donor_profiles = {rec: train_profiles[rec] for rec in donor_recs}
        units = sorted(set().union(*(set(p.columns) for p in donor_profiles.values()))) if donor_profiles else []
        stats = build_stats(donor_profiles, units)
        cur = current_rows.loc[current_rows["recording"].eq(heldout)].copy()
        current_profile = profile(cur) if not cur.empty else pd.DataFrame()
        valid_mask = valid["recording"].eq(heldout).to_numpy()

        for unit in sorted(valid.loc[valid_mask, "unit"].unique()):
            distance_map = {
                rec: profile_distance(current_profile, p, stats, exclude_unit=unit)
                for rec, p in donor_profiles.items()
            }
            group_positions = np.flatnonzero(valid_mask & valid["unit"].eq(unit).to_numpy())
            for i in group_positions:
                m = int(valid.iloc[i]["minute_of_day"])
                calibrated, scale, chosen = candidate_prediction(
                    current_profile, donor_profiles, distance_map, unit, m
                )
                pred_analog[i] = calibrated
                # Recover uncalibrated analog for a diagnostic blend.
                chosen_series = []
                for rec, w in chosen:
                    value = interp_inside(donor_profiles[rec][unit], m)
                    if np.isfinite(value):
                        chosen_series.append((value, w))
                if chosen_series:
                    denom = sum(w for _, w in chosen_series)
                    pred_unscaled[i] = sum(v * w for v, w in chosen_series) / denom
                diagnostics.append({
                    "recording": heldout,
                    "unit": unit,
                    "minute_of_day": m,
                    "donors": ";".join(f"{r}:{w:.3f}" for r, w in chosen),
                    "camera_scale": scale,
                })
        print(f"Fold {heldout}: same-weekday donor recordings={donor_recs}; validation={int(valid_mask.sum())}")

    out = valid[["recording", "unit", "minute_of_day", "truth", BASE_COL]].copy()
    out = out.rename(columns={BASE_COL: "baseline"})
    out["analog"] = pred_unscaled
    out["analog_camera_scaled"] = pred_analog
    analog_ok = np.isfinite(pred_analog)
    out["analog_fallback"] = np.where(analog_ok, pred_analog, out["baseline"])
    print(
        f"Analog coverage: {int(analog_ok.sum())}/{len(out)} rows "
        f"({100.0 * analog_ok.mean():.1f}%); uncovered rows use baseline"
    )
    coverage = out.assign(analog_available=analog_ok).groupby(
        ["recording", "unit"]
    )["analog_available"].agg(["sum", "count", "mean"])
    coverage["coverage_pct"] = 100.0 * coverage["mean"]
    print("\n=== ANALOG COVERAGE BY CAMERA ===")
    print(coverage[["sum", "count", "coverage_pct"]].rename(
        columns={"sum": "analog_rows", "count": "total_rows"}
    ).round(1).to_string())
    for alpha in (0.25, 0.50, 0.75):
        out[f"blend_analog_a{alpha:.2f}"] = (
            (1 - alpha) * out["baseline"] + alpha * out["analog_fallback"]
        )

    y = out["truth"].to_numpy(float)
    records = sorted(out["recording"].unique())
    candidates = [
        "baseline", "analog", "analog_camera_scaled", "analog_fallback",
        "blend_analog_a0.25", "blend_analog_a0.50", "blend_analog_a0.75",
    ]
    summary = []
    for c in candidates:
        p = out[c].to_numpy(float)
        score_mask = np.isfinite(p)
        if not score_mask.any():
            continue
        row = {
            "candidate": c,
            "rows": int(score_mask.sum()),
            "coverage_pct": 100.0 * float(score_mask.mean()),
            "mae": float(np.mean(np.abs(y[score_mask] - p[score_mask]))),
        }
        gains = []
        for rec in records:
            mask = out["recording"].eq(rec).to_numpy() & score_mask
            if not mask.any():
                row[f"mae_{rec}"] = np.nan
                row[f"gain_{rec}_pct"] = np.nan
                gains.append(-np.inf)
                continue
            b = float(np.mean(np.abs(y[mask] - out.loc[mask, "baseline"].to_numpy(float))))
            v = float(np.mean(np.abs(y[mask] - p[mask])))
            row[f"mae_{rec}"] = v
            row[f"gain_{rec}_pct"] = 100 * (b - v) / b if b else 0.0
            gains.append(row[f"gain_{rec}_pct"])
        row["improves_both_recordings"] = bool(all(g > 0 for g in gains))
        summary.append(row)

    result = pd.DataFrame(summary).sort_values(["rows", "mae"], ascending=[False, True])
    print("\n=== HISTORICAL-ANALOG RESULTS ===")
    print(result.round(4).to_string(index=False))
    full_results = result.loc[result["rows"].eq(len(out))].sort_values("mae")
    best_full = full_results.iloc[0] if not full_results.empty else None
    if best_full is not None and best_full["candidate"] != "baseline":
        best_name = str(best_full["candidate"])
        group_rows = []
        for (rec, unit), group in out.groupby(["recording", "unit"]):
            yy = group["truth"].to_numpy(float)
            base_mae = float(np.mean(np.abs(yy - group["baseline"].to_numpy(float))))
            best_mae = float(np.mean(np.abs(yy - group[best_name].to_numpy(float))))
            group_rows.append({
                "recording": rec,
                "unit": unit,
                "rows": len(group),
                "baseline_mae": base_mae,
                "candidate_mae": best_mae,
                "gain_pct": 100 * (base_mae - best_mae) / base_mae if base_mae else 0.0,
            })
        print(f"\n=== BEST CANDIDATE BY CAMERA: {best_name} ===")
        print(pd.DataFrame(group_rows).round(4).to_string(index=False))
    if diagnostics:
        detail = pd.DataFrame(diagnostics)
        print("\n=== SELECTED DONORS / CAMERA-SCALE SUMMARY ===")
        print(detail.groupby(["recording", "unit"]).agg(
            donor_choices=("donors", "first"),
            camera_scale_median=("camera_scale", "median"),
            camera_scale_min=("camera_scale", "min"),
            camera_scale_max=("camera_scale", "max"),
        ).round(4).to_string())
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_FILE, index=False)
    print("\nSaved diagnostic:", REPORT_FILE)
    if best_full is not None and best_full["candidate"] != "baseline" and bool(best_full["improves_both_recordings"]):
        print("STATUS: PROMISING_ANALOG; inspect camera groups before any candidate.")
    else:
        print("STATUS: NO_ROBUST_ANALOG_GAIN; retain current hidden predictions.")
    print("STRICT HIDDEN VALIDATION — NO TEST CANDIDATE OR SUBMISSION CREATED")


if __name__ == "__main__":
    main()
