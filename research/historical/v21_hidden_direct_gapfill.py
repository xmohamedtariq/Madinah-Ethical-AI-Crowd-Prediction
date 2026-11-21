from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path.cwd()
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"
VALIDATION_FILE = ROOT / "reports" / "v11_2_validation_predictions.csv"
REPORT_FILE = ROOT / "reports" / "v21_hidden_direct_gapfill_oof.csv"
BASE_COL = "blend_extra_leaf3_a0.5"
TEST_TO_VALIDATION = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
LOCAL_RADII = (15, 30, 60, 120)


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    return pd.read_csv(path)


def normalize(frame: pd.DataFrame, name: str, label_col: str | None = None) -> pd.DataFrame:
    x = frame.copy()
    required = {"recording", "unit", "minute_of_day"}
    if label_col:
        required.add(label_col)
    absent = required - set(x.columns)
    if absent:
        raise SystemExit(f"{name} missing columns: {sorted(absent)}")
    x["recording"] = x["recording"].astype(str)
    x["unit"] = x["unit"].astype(str)
    x["minute_of_day"] = pd.to_numeric(x["minute_of_day"], errors="coerce")
    x = x.loc[x["minute_of_day"].notna()].copy()
    x["minute_of_day"] = x["minute_of_day"].astype(int)
    if label_col:
        x[label_col] = pd.to_numeric(x[label_col], errors="coerce")
    return x


def interpolate_linear(t: np.ndarray, y: np.ndarray, target: float) -> float:
    if len(t) < 2 or target < t[0] or target > t[-1]:
        return np.nan
    return float(np.interp(target, t, y))


def interpolate_pchip(t: np.ndarray, y: np.ndarray, target: float) -> float:
    if len(t) < 3 or target < t[0] or target > t[-1]:
        return np.nan
    return float(max(0.0, PchipInterpolator(t, y, extrapolate=False)(target)))


def local_average(t: np.ndarray, y: np.ndarray, target: float, radius: int) -> float:
    delta = np.abs(t - target)
    mask = delta <= radius
    if not mask.any():
        return np.nan
    # Triangular weights make the nearest observed values matter most.
    weights = np.maximum(0.0, 1.0 - delta[mask] / (radius + 1.0))
    if weights.sum() <= 0:
        return np.nan
    return float(np.average(y[mask], weights=weights))


def main() -> None:
    train = normalize(read_required(TRAIN_FILE), "train.csv", "people")
    test = normalize(read_required(TEST_FILE), "test.csv", "people")
    valid = normalize(read_required(VALIDATION_FILE), "v11_2 validation")
    for col in ("truth", BASE_COL):
        if col not in valid:
            raise SystemExit(f"Validation report missing column: {col}")
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.loc[valid["truth"].notna() & valid[BASE_COL].notna()].copy().reset_index(drop=True)
    if len(valid) != 843:
        raise SystemExit(f"Expected 843 hidden validation targets; found {len(valid)}")

    blocked = test.loc[test["people"].isna(), ["recording", "unit", "minute_of_day"]].copy()
    blocked["recording"] = blocked["recording"].map(TEST_TO_VALIDATION)
    blocked = blocked.dropna(subset=["recording"])
    blocked_keys = set(zip(blocked["recording"], blocked["unit"], blocked["minute_of_day"]))
    valid_keys = set(zip(valid["recording"], valid["unit"], valid["minute_of_day"]))
    unmasked = valid_keys - blocked_keys
    if unmasked:
        raise SystemExit(f"Leakage guard failed: {len(unmasked)} validation target keys are not masked")
    print(f"Leakage guard: all {len(valid_keys)} validation targets masked")

    released = train.loc[train["people"].notna()].copy()
    released["key"] = list(zip(released["recording"], released["unit"], released["minute_of_day"]))
    released = released.loc[~released["key"].isin(blocked_keys)].copy()
    released_map = {}
    for (rec, unit), g in released.groupby(["recording", "unit"]):
        g = g.sort_values("minute_of_day")
        released_map[(str(rec), str(unit))] = (
            g["minute_of_day"].to_numpy(float),
            g["people"].to_numpy(float),
        )

    out = valid[["recording", "unit", "minute_of_day", "truth", BASE_COL]].copy()
    out = out.rename(columns={BASE_COL: "baseline"})
    methods = ["linear", "pchip"] + [f"local_mean_{r}" for r in LOCAL_RADII]
    predictions = {name: np.full(len(out), np.nan) for name in methods}
    context_rows = []
    for (rec, unit), positions in out.groupby(["recording", "unit"]).groups.items():
        series = released_map.get((str(rec), str(unit)))
        if series is None:
            context_rows.append({"recording": rec, "unit": unit, "visible_context": 0})
            continue
        t, y = series
        context_rows.append({"recording": rec, "unit": unit, "visible_context": len(t)})
        for i in positions:
            minute = float(out.loc[i, "minute_of_day"])
            predictions["linear"][i] = interpolate_linear(t, y, minute)
            predictions["pchip"][i] = interpolate_pchip(t, y, minute)
            for radius in LOCAL_RADII:
                key = f"local_mean_{radius}"
                predictions[key][i] = local_average(t, y, minute, radius)

    truth = out["truth"].to_numpy(float)
    baseline = out["baseline"].to_numpy(float)
    summary = [{
        "method": "baseline",
        "rows_scored": len(out),
        "coverage_pct": 100.0,
        "mae_covered": float(np.mean(np.abs(truth - baseline))),
        "selective_replace_mae": float(np.mean(np.abs(truth - baseline))),
    }]
    for name, pred in predictions.items():
        ok = np.isfinite(pred)
        if not ok.any():
            continue
        # The standalone interpolation is scored on its supported rows.
        covered_mae = float(np.mean(np.abs(truth[ok] - pred[ok])))
        # Also report a valid full-coverage policy: replace only where the direct
        # method has support; preserve the baseline on unsupported rows.
        selective = baseline.copy()
        selective[ok] = pred[ok]
        summary.append({
            "method": name,
            "rows_scored": int(ok.sum()),
            "coverage_pct": 100.0 * float(ok.mean()),
            "mae_covered": covered_mae,
            "selective_replace_mae": float(np.mean(np.abs(truth - selective))),
        })
        out[name] = pred
        out[f"{name}_selective"] = selective

    result = pd.DataFrame(summary).sort_values("selective_replace_mae")
    print("\n=== DIRECT SAME-CAMERA GAPFILL ===")
    print(result.round(4).to_string(index=False))
    print("\n=== AVAILABLE SAME-CAMERA CONTEXT ===")
    print(pd.DataFrame(context_rows).to_string(index=False))

    best = result.iloc[0]["method"]
    if best != "baseline":
        policy = out[f"{best}_selective"].to_numpy(float)
        out["baseline_error"] = np.abs(out["truth"].to_numpy(float) - baseline)
        out["best_error"] = np.abs(out["truth"].to_numpy(float) - policy)
        groups = out.groupby(["recording", "unit"])[["baseline_error", "best_error"]].mean()
        groups["gain_pct"] = 100.0 * (groups["baseline_error"] - groups["best_error"]) / groups["baseline_error"]
        print(f"\n=== SELECTIVE REPLACEMENT BY CAMERA: {best} ===")
        print(groups.round(4).to_string())

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_FILE, index=False)
    print("\nSaved diagnostic:", REPORT_FILE)
    if best != "baseline":
        print("Best local selective method:", best)
    else:
        print("STATUS: NO_DIRECT_GAPFILL_GAIN; retain current hidden predictions.")
    print("NO SUBMISSION CANDIDATE CREATED OR UPLOADED")


if __name__ == "__main__":
    main()
