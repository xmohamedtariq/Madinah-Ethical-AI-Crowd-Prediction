from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
DATA = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
TRAIN_FILE = DATA / "train.csv"
TEST_FILE = DATA / "test.csv"
OUT_FILE = REPORTS / "v22_missing_peer_ridge_oof.csv"
MAPPING = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
MIN_FIT_ROWS = 50
MIN_PEERS = 3


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    return pd.read_csv(path)


def normalize(df: pd.DataFrame, label_required: bool) -> pd.DataFrame:
    x = df.copy()
    for col in ("recording", "unit"):
        if x[col].isna().any():
            raise SystemExit(f"Missing key: {col}")
        x[col] = x[col].astype(str)
    minute = pd.to_numeric(x["minute_of_day"], errors="coerce")
    x = x.loc[minute.notna()].copy()
    x["minute_of_day"] = minute.loc[x.index].astype(int)
    if label_required:
        if "people" not in x:
            raise SystemExit("Missing people column")
        x["people"] = pd.to_numeric(x["people"], errors="coerce")
    return x


def build_features(frame: pd.DataFrame, peer_units: list[str]) -> pd.DataFrame:
    # One row per recording/minute. Peer values are used only when released.
    wide = frame.pivot_table(
        index=["recording", "minute_of_day"],
        columns="unit",
        values="people",
        aggfunc="first",
    )
    wide = wide.reindex(columns=peer_units)
    wide.columns = [f"peer_{u}" for u in wide.columns]
    minute = wide.index.get_level_values("minute_of_day").to_numpy(float)
    wide["time_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    wide["time_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    vals = wide[[c for c in wide.columns if c.startswith("peer_")]].to_numpy(float)
    wide["peer_count"] = np.isfinite(vals).sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        wide["peer_median"] = np.nanmedian(vals, axis=1)
        wide["peer_mean"] = np.nanmean(vals, axis=1)
        wide["peer_std"] = np.nanstd(vals, axis=1)
    wide = wide.replace([np.inf, -np.inf], np.nan).reset_index()
    return wide


def linear_same_camera(source: pd.DataFrame, recording: str, unit: str, minute: int) -> float:
    g = source.loc[
        source["recording"].eq(recording) & source["unit"].eq(unit),
        ["minute_of_day", "people"],
    ].dropna().sort_values("minute_of_day")
    if g.empty:
        return np.nan
    t = g["minute_of_day"].to_numpy(float)
    y = g["people"].to_numpy(float)
    if len(t) == 1:
        return float(y[0])
    return float(np.interp(float(minute), t, y))


def main() -> None:
    train = normalize(read_required(TRAIN_FILE), True)
    test = normalize(read_required(TEST_FILE), True)
    needed_test = {"kind", "people"}
    if needed_test - set(test.columns):
        raise SystemExit(f"test.csv missing columns: {sorted(needed_test - set(test.columns))}")

    # Translate every unreleased test cell back to its validation recording and
    # remove it from the current-day peer/context panel before fitting or scoring.
    blocked = test.loc[test["people"].isna(), ["recording", "unit", "minute_of_day"]].copy()
    blocked["recording"] = blocked["recording"].map(MAPPING)
    blocked = blocked.dropna(subset=["recording"])
    blocked_keys = set(zip(blocked["recording"], blocked["unit"], blocked["minute_of_day"]))

    mapped = test.loc[test["kind"].eq("missing"), ["recording", "unit", "minute_of_day"]].copy()
    mapped["validation_recording"] = mapped["recording"].map(MAPPING)
    mapped = mapped.dropna(subset=["validation_recording"])
    mapped = mapped.rename(columns={"recording": "test_recording"})
    mapped["recording"] = mapped.pop("validation_recording")
    mapped = mapped.drop(columns=["test_recording"])
    mapped_keys = set(zip(mapped["recording"], mapped["unit"], mapped["minute_of_day"]))

    visible = test.loc[test["people"].notna(), ["recording", "unit", "minute_of_day"]].copy()
    visible["recording"] = visible["recording"].map(MAPPING)
    visible = visible.dropna(subset=["recording"])
    visible_keys = set(zip(visible["recording"], visible["unit"], visible["minute_of_day"]))

    train_idx = pd.MultiIndex.from_frame(train[["recording", "unit", "minute_of_day"]])
    target_mask = train_idx.isin(pd.MultiIndex.from_tuples(mapped_keys, names=["recording", "unit", "minute_of_day"]))
    gold = train.loc[target_mask, ["recording", "unit", "minute_of_day", "people"]].copy()
    gold = gold.rename(columns={"people": "truth"}).reset_index(drop=True)
    gold = gold.loc[gold["truth"].notna()].copy().reset_index(drop=True)
    gold_keys = set(zip(gold["recording"], gold["unit"], gold["minute_of_day"]))
    if gold_keys - blocked_keys:
        raise SystemExit(f"Leakage guard failed: {len(gold_keys - blocked_keys)} validation missing targets were not masked")
    if len(gold) != 358:
        raise SystemExit(f"Expected 358 recoverable validation missing targets; found {len(gold)}")
    print(f"Leakage guard: all {len(gold)} validation missing targets masked")

    valid_recordings = set(gold["recording"].unique())
    train["key"] = list(zip(train["recording"], train["unit"], train["minute_of_day"]))
    in_validation = train["recording"].isin(valid_recordings)
    keep_visible = train["key"].isin(visible_keys)
    release = train.loc[~in_validation | keep_visible].copy()
    release = release.loc[release["people"].notna()].copy()
    units = sorted(release["unit"].unique())
    print(f"Released context rows: {len(release)}; cameras: {len(units)}")

    rows = []
    for (rec, target_unit), group in gold.groupby(["recording", "unit"], sort=True):
        peer_units = [u for u in units if u != target_unit]
        camera_peer_cols = [f"peer_{u}" for u in peer_units]
        Xcols = [f"peer_{u}" for u in peer_units] + ["time_sin", "time_cos", "peer_count", "peer_median", "peer_mean", "peer_std"]
        feature_frame = build_features(release, peer_units)
        feature_lookup = feature_frame.set_index(["recording", "minute_of_day"])
        feature_lookup = feature_lookup.loc[
            feature_lookup.index.get_level_values("recording") == rec
        ]
        # Current recording's released target-camera rows provide local camera
        # loading/calibration; held-out target cells were removed above.
        fit_rows = release.loc[
            release["recording"].eq(rec) & release["unit"].eq(target_unit),
            ["minute_of_day", "people"],
        ].copy()
        fit_idx = pd.MultiIndex.from_arrays(
            [np.repeat(rec, len(fit_rows)), fit_rows["minute_of_day"].to_numpy(int)],
            names=["recording", "minute_of_day"],
        )
        Xfit = feature_lookup.reindex(fit_idx)[Xcols].copy()
        fit_peer_count = Xfit[camera_peer_cols].notna().sum(axis=1).to_numpy()
        usable = fit_peer_count >= MIN_PEERS
        Xfit = Xfit.loc[usable]
        yfit = np.log1p(np.maximum(0.0, fit_rows.loc[usable, "people"].to_numpy(float)))
        model = None
        if len(Xfit) >= MIN_FIT_ROWS:
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=20.0),
            )
            model.fit(Xfit.to_numpy(float), yfit)

        for target in group.itertuples(index=False):
            key = (rec, int(target.minute_of_day))
            if key in feature_lookup.index:
                xrow = feature_lookup.loc[key, Xcols]
                if isinstance(xrow, pd.DataFrame):
                    xrow = xrow.iloc[0]
                peer_count = int(xrow[camera_peer_cols].notna().sum())
            else:
                xrow = None
                peer_count = 0
            peer_pred = np.nan
            if model is not None and xrow is not None and peer_count >= MIN_PEERS:
                log_pred = float(model.predict(xrow.to_numpy(float).reshape(1, -1))[0])
                peer_pred = float(max(0.0, np.expm1(np.clip(log_pred, 0.0, 12.0))))
            linear_pred = linear_same_camera(release, rec, target_unit, int(target.minute_of_day))
            rows.append({
                "recording": rec,
                "unit": target_unit,
                "minute_of_day": int(target.minute_of_day),
                "truth": float(target.truth),
                "peer_count": peer_count,
                "fit_rows": len(Xfit),
                "linear": linear_pred,
                "peer_ridge": peer_pred,
            })

    out = pd.DataFrame(rows)
    if len(out) != len(gold):
        raise SystemExit(f"Prediction count mismatch: {len(out)} vs {len(gold)}")
    y = out["truth"].to_numpy(float)
    baseline = out["linear"].to_numpy(float)
    peer = out["peer_ridge"].to_numpy(float)
    supported = np.isfinite(peer)
    selective = baseline.copy()
    selective[supported] = peer[supported]
    out["peer_ridge_selective"] = selective
    out["baseline_abs_error"] = np.abs(y - baseline)
    out["peer_abs_error_on_support"] = np.where(supported, np.abs(y - peer), np.nan)
    out["selective_abs_error"] = np.abs(y - selective)

    summary = []
    summary.append({
        "method": "linear_baseline",
        "rows": len(out),
        "coverage_pct": 100.0,
        "mae_on_support": float(np.nanmean(np.abs(y - baseline))),
        "full_policy_mae": float(np.nanmean(np.abs(y - baseline))),
    })
    if supported.any():
        summary.append({
            "method": "peer_ridge_supported_only",
            "rows": int(supported.sum()),
            "coverage_pct": 100.0 * float(supported.mean()),
            "mae_on_support": float(np.mean(np.abs(y[supported] - peer[supported]))),
            "full_policy_mae": np.nan,
        })
        summary.append({
            "method": "peer_ridge_selective_replace",
            "rows": len(out),
            "coverage_pct": 100.0,
            "mae_on_support": float(np.mean(np.abs(y[supported] - peer[supported]))),
            "full_policy_mae": float(np.mean(np.abs(y - selective))),
        })
    print("\n=== MISSING: SAME-RECORDING PEER REGRESSION ===")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\n=== BY RECORDING / CAMERA ===")
    by_group = out.groupby(["recording", "unit"]).agg(
        rows=("truth", "size"),
        peer_coverage=("peer_ridge", lambda x: 100.0 * x.notna().mean()),
        linear_mae=("baseline_abs_error", "mean"),
        selective_mae=("selective_abs_error", "mean"),
        fit_rows=("fit_rows", "median"),
    )
    by_group["gain_pct"] = 100.0 * (by_group["linear_mae"] - by_group["selective_mae"]) / by_group["linear_mae"]
    print(by_group.round(4).to_string())

    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print("\nSaved diagnostic:", OUT_FILE)
    if supported.mean() >= 0.5 and np.mean(np.abs(y - selective)) < np.mean(np.abs(y - baseline)) * 0.95:
        print("STATUS: PROMISING_PEER_MODEL; inspect recording/camera groups before any test candidate.")
    else:
        print("STATUS: NO_STRONG_MISSING_GAIN; do not replace current missing predictions.")
    print("NO SUBMISSION CANDIDATE CREATED OR UPLOADED")


if __name__ == "__main__":
    main()
