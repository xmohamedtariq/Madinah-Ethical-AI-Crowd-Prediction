from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"
VALIDATION_FILE = ROOT / "reports" / "v11_2_validation_predictions.csv"
REPORT_FILE = ROOT / "reports" / "v20_hidden_weekly_timewarp_oof.csv"
BASE_COL = "blend_extra_leaf3_a0.5"
TEST_TO_VALIDATION = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
SHIFTS = (-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90)
ALPHAS = (0.10, 0.25, 0.50, 0.75)


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


def start_minute(recording: str) -> int:
    suffix = str(recording).split("_")[-1]
    if len(suffix) != 4 or not suffix.isdigit():
        return 12 * 60
    return int(suffix[:2]) * 60 + int(suffix[2:])


def make_series_map(frame: pd.DataFrame, label_col: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    result = {}
    for (rec, unit), group in frame.groupby(["recording", "unit"]):
        group = group.loc[group[label_col].notna()].sort_values("minute_of_day")
        if group.empty:
            continue
        result[(str(rec), str(unit))] = (
            group["minute_of_day"].to_numpy(float),
            group[label_col].to_numpy(float),
        )
    return result


def interp(series: tuple[np.ndarray, np.ndarray] | None, minute: float) -> float:
    if series is None:
        return np.nan
    t, y = series
    if len(t) < 2 or minute < t[0] or minute > t[-1]:
        return np.nan
    return float(np.interp(minute, t, y))


def choose_shift(context: tuple[np.ndarray, np.ndarray], donor: tuple[np.ndarray, np.ndarray]) -> tuple[int, float, int, float]:
    now_t, now_y = context
    trials = []
    for shift in SHIFTS:
        prior = np.asarray([interp(donor, float(t + shift)) for t in now_t], dtype=float)
        ok = np.isfinite(prior) & np.isfinite(now_y)
        if int(ok.sum()) < max(10, int(0.30 * len(now_t))):
            continue
        log_ratios = np.log((now_y[ok] + 5.0) / (prior[ok] + 5.0))
        scale = float(np.exp(np.clip(np.median(log_ratios), np.log(0.70), np.log(1.30))))
        predicted = np.maximum(0.0, (prior[ok] + 5.0) * scale - 5.0)
        normalized_mae = float(np.mean(np.abs(now_y[ok] - predicted) / np.maximum(10.0, np.abs(now_y[ok]))))
        coverage_penalty = 0.03 * (1.0 - float(ok.mean()))
        trials.append((normalized_mae + coverage_penalty, shift, scale, int(ok.sum()), normalized_mae))
    if not trials:
        return 0, 1.0, 0, np.inf
    _, shift, scale, support, loss = min(trials, key=lambda r: r[0])
    return int(shift), float(scale), int(support), float(loss)


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
    missing_mask = valid_keys - blocked_keys
    if missing_mask:
        raise SystemExit(f"Leakage guard failed: {len(missing_mask)} validation target keys were not masked")
    print(f"Leakage guard: all {len(valid_keys)} validation targets masked")

    train = train.loc[train["people"].notna()].copy()
    train["key"] = list(zip(train["recording"], train["unit"], train["minute_of_day"]))
    released_current = train.loc[~train["key"].isin(blocked_keys)].copy()
    current_map = make_series_map(released_current, "people")
    history_map = make_series_map(train, "people")

    out = valid[["recording", "unit", "minute_of_day", "truth", BASE_COL]].copy()
    out = out.rename(columns={BASE_COL: "baseline"})
    n = len(out)
    weekly_exact = np.full(n, np.nan)
    weekly_aligned = np.full(n, np.nan)
    chosen_rows = []

    for (rec, unit), idx in out.groupby(["recording", "unit"]).groups.items():
        date = datetime.strptime(str(rec)[:8], "%Y%m%d")
        prior_date = (date - timedelta(days=7)).strftime("%Y%m%d")
        donor_recs = sorted({r for (r, u) in history_map if u == unit and r[:8] == prior_date})
        if not donor_recs:
            chosen_rows.append({"recording": rec, "unit": unit, "donor": "none", "shift": 0, "scale": 1.0, "support": 0, "context_loss": np.nan})
            continue
        donor_rec = min(donor_recs, key=lambda r: abs(start_minute(r) - start_minute(str(rec))))
        donor = history_map[(donor_rec, unit)]
        context = current_map.get((str(rec), str(unit)))
        if context is None:
            shift, scale, support, context_loss = 0, 1.0, 0, np.inf
        else:
            shift, scale, support, context_loss = choose_shift(context, donor)
        chosen_rows.append({
            "recording": rec, "unit": unit, "donor": donor_rec,
            "shift": shift, "scale": scale, "support": support,
            "context_loss": context_loss,
        })
        for i in idx:
            minute = float(out.loc[i, "minute_of_day"])
            weekly_exact[i] = interp(donor, minute)
            shifted = interp(donor, minute + shift)
            if np.isfinite(shifted):
                weekly_aligned[i] = max(0.0, (shifted + 5.0) * scale - 5.0)

    out["weekly_exact"] = weekly_exact
    out["weekly_aligned_scaled"] = weekly_aligned
    out["weekly_aligned_fallback"] = np.where(np.isfinite(weekly_aligned), weekly_aligned, out["baseline"])
    out["weekly_exact_fallback"] = np.where(np.isfinite(weekly_exact), weekly_exact, out["baseline"])
    for alpha in ALPHAS:
        out[f"blend_aligned_a{alpha:.2f}"] = (
            (1.0 - alpha) * out["baseline"] + alpha * out["weekly_aligned_fallback"]
        )
        out[f"blend_exact_a{alpha:.2f}"] = (
            (1.0 - alpha) * out["baseline"] + alpha * out["weekly_exact_fallback"]
        )

    candidates = ["baseline", "weekly_exact_fallback", "weekly_aligned_fallback"]
    candidates += [f"blend_aligned_a{a:.2f}" for a in ALPHAS]
    candidates += [f"blend_exact_a{a:.2f}" for a in ALPHAS]
    truth = out["truth"].to_numpy(float)
    records = sorted(out["recording"].unique())
    results = []
    for name in candidates:
        pred = out[name].to_numpy(float)
        row = {"candidate": name, "rows": n, "mae": float(np.mean(np.abs(truth - pred)))}
        gains = []
        for rec in records:
            mask = out["recording"].eq(rec).to_numpy()
            base_mae = float(np.mean(np.abs(truth[mask] - out.loc[mask, "baseline"].to_numpy(float))))
            mae = float(np.mean(np.abs(truth[mask] - pred[mask])))
            row[f"mae_{rec}"] = mae
            row[f"gain_{rec}_pct"] = 100.0 * (base_mae - mae) / base_mae if base_mae else 0.0
            gains.append(row[f"gain_{rec}_pct"])
        row["improves_both_recordings"] = bool(all(g > 0 for g in gains))
        results.append(row)
    result = pd.DataFrame(results).sort_values("mae")
    print("\n=== WEEKLY TIME-ALIGNMENT RESULTS ===")
    print(result.round(4).to_string(index=False))
    print("\n=== DONOR / ALIGNMENT CHOICES ===")
    print(pd.DataFrame(chosen_rows).round(4).to_string(index=False))

    best = result.iloc[0]["candidate"]
    if best != "baseline":
        out["baseline_error"] = np.abs(out["truth"] - out["baseline"])
        out["best_error"] = np.abs(out["truth"] - out[best])
        groups = out.groupby(["recording", "unit"])[["baseline_error", "best_error"]].mean()
        groups["gain_pct"] = 100.0 * (groups["baseline_error"] - groups["best_error"]) / groups["baseline_error"]
        print(f"\n=== BEST POLICY BY CAMERA: {best} ===")
        print(groups.round(4).to_string())

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_FILE, index=False)
    print("\nSaved diagnostic:", REPORT_FILE)
    if best != "baseline" and bool(result.iloc[0]["improves_both_recordings"]):
        print("STATUS: REVIEW_CANDIDATE; inspect per-camera gains before any test candidate.")
    else:
        print("STATUS: NO_ROBUST_GAIN; retain current hidden predictions.")
    print("STRICT HIDDEN VALIDATION ONLY — NO TEST CANDIDATE OR SUBMISSION CREATED")


if __name__ == "__main__":
    main()
