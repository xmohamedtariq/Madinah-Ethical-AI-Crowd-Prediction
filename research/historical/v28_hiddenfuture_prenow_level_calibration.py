from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
SUBMISSIONS = ROOT / "submissions"
TRAIN_FILE = DATA / "train.csv"
TEST_FILE = DATA / "test.csv"
VALIDATION_FILE = REPORTS / "v8_1c_hidden_future_predictions.csv"
BASE_SUBMISSION = SUBMISSIONS / (
    "submission_v6_4_fixed_v5blend_v11_2_v8_future_"
    "no_v13_missing_v15_prayer_full_v27_missing_shrink075_candidate.csv"
)
OUT_VALIDATION = REPORTS / "v28_hiddenfuture_prenow_validation.csv"
OUT_SUMMARY = REPORTS / "v28_hiddenfuture_prenow_summary.csv"
OUT_CANDIDATE = SUBMISSIONS / (
    "submission_v6_4_fixed_v5blend_v11_2_v8_future_"
    "no_v13_missing_v15_prayer_full_v27_missing_shrink075_"
    "v28_hiddenfuture_prenow_candidate.csv"
)
OUT_AUDIT = REPORTS / "v28_hiddenfuture_prenow_test_audit.csv"

MAPPING = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
CAT = ["unit", "weekday"]
NUM = ["minute_of_day", "time_sin", "time_cos", "time_bin_15", "time_bin_30"]
FEATURES = CAT + NUM
WINDOWS = (60, 120, 240, 360)
POWERS = (0.25, 0.50, 0.75, 1.00)


def static_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    minute = frame["minute_of_day"].to_numpy(dtype=float)
    angle = 2.0 * np.pi * minute / 1440.0
    x["unit"] = frame["unit"].astype(str)
    x["weekday"] = frame["weekday"].astype(str)
    x["minute_of_day"] = minute
    x["time_sin"] = np.sin(angle)
    x["time_cos"] = np.cos(angle)
    x["time_bin_15"] = minute // 15
    x["time_bin_30"] = minute // 30
    return x[FEATURES]


def make_model() -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT,
            ),
            ("num", "passthrough", NUM),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", model)])


def fit_model(train: pd.DataFrame, heldout: str | None = None) -> Pipeline:
    fit = train.loc[train["people"].notna()].copy()
    if heldout is not None:
        fit = fit.loc[fit["recording"].astype(str).ne(str(heldout))].copy()
    if len(fit) < 500:
        raise ValueError(f"Too few rows to fit static model: {len(fit)}")
    model = make_model()
    model.fit(static_features(fit), fit["people"].to_numpy(dtype=float))
    return model


def robust_level_ratio(
    visible: pd.DataFrame,
    model: Pipeline,
    cutoff: int,
    window: int,
) -> tuple[float, int]:
    z = visible.loc[
        visible["minute_of_day"].le(cutoff)
        & visible["minute_of_day"].ge(cutoff - window + 1)
    ].copy()
    if z.empty:
        return 1.0, 0
    expected = model.predict(static_features(z))
    observed = z["people"].to_numpy(dtype=float)
    valid = np.isfinite(expected) & np.isfinite(observed) & (expected > 1.0)
    ratios = observed[valid] / expected[valid]
    if len(ratios) < 8:
        return 1.0, int(len(ratios))
    return float(np.clip(np.median(ratios), 0.60, 1.60)), int(len(ratios))


def minute_of_day_values(values: pd.Series) -> pd.Series:
    """Accept numeric minutes or clock strings such as HH:MM."""
    result = pd.to_numeric(values, errors="coerce")
    unresolved = result.isna()
    if unresolved.any():
        parts = values.loc[unresolved].astype(str).str.extract(
            r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$"
        )
        if parts.isna().any().any():
            raise ValueError("Could not parse now as minutes or HH:MM")
        hours = pd.to_numeric(parts[0], errors="raise")
        minutes = pd.to_numeric(parts[1], errors="raise")
        if (hours > 23).any() or (minutes > 59).any():
            raise ValueError("Invalid HH:MM value in now")
        result.loc[unresolved] = hours.to_numpy() * 60 + minutes.to_numpy()
    if (
        not np.isfinite(result.to_numpy(dtype=float)).all()
        or (result < 0).any()
        or (result >= 1440).any()
        or (result != np.floor(result)).any()
    ):
        raise ValueError("now values must be integer minutes within a day")
    return result.astype(int)


def validation_context(train: pd.DataFrame, rec: str, unit: str, cutoff: int):
    # Recreate the information available before `now` in the held-out
    # validation recording. Never borrow the corresponding test-day context.
    z = train.loc[
        train["people"].notna()
        & train["recording"].astype(str).eq(rec)
        & train["unit"].astype(str).eq(unit)
        & pd.to_numeric(train["minute_of_day"], errors="coerce").le(cutoff)
    ].copy()
    z["minute_of_day"] = pd.to_numeric(z["minute_of_day"], errors="raise").astype(int)
    z["people"] = pd.to_numeric(z["people"], errors="raise")
    return z


def test_context(test: pd.DataFrame, rec: str, unit: str, cutoff: int):
    z = test.loc[
        test["people"].notna()
        & test["recording"].astype(str).eq(rec)
        & test["unit"].astype(str).eq(unit)
        & pd.to_numeric(test["minute_of_day"], errors="coerce").le(cutoff)
    ].copy()
    z["minute_of_day"] = pd.to_numeric(z["minute_of_day"], errors="raise").astype(int)
    z["people"] = pd.to_numeric(z["people"], errors="raise")
    return z


def validation_predictions(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame):
    result = val.copy().reset_index(drop=True)
    if not {"recording", "unit", "minute_of_day", "now", "truth", "static_hist",
            "global_shrink_prediction"}.issubset(result.columns):
        raise SystemExit("V8.1c validation report lacks required columns")
    result["recording"] = result["recording"].astype(str)
    if "weekday" not in result.columns:
        weekday_rows = train[["recording", "weekday"]].copy()
        weekday_rows["recording"] = weekday_rows["recording"].astype(str)
        weekday_lookup = (
            weekday_rows.drop_duplicates("recording")
            .set_index("recording")["weekday"].astype(str).to_dict()
        )
        result["weekday"] = result["recording"].map(weekday_lookup)
        if result["weekday"].isna().any():
            raise ValueError("Could not recover original weekday categories from training data")
    for col in ("minute_of_day", "now", "truth", "static_hist", "global_shrink_prediction"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if result[["minute_of_day", "now", "truth", "static_hist", "global_shrink_prediction"]].isna().any().any():
        raise ValueError("V8.1c validation report has missing values in required fields")

    for rec in sorted(result["recording"].unique()):
        model = fit_model(train, heldout=rec)
        idx = result.index[result["recording"].eq(rec)]
        target_features = result.loc[idx].copy()
        model_prediction = model.predict(static_features(target_features))
        stored = result.loc[idx, "static_hist"].to_numpy(float)
        max_difference = float(np.max(np.abs(model_prediction - stored)))
        if max_difference > 1e-4:
            raise ValueError(
                f"Static baseline mismatch for {rec}: max abs diff={max_difference:.6g}"
            )
        print(f"Validation fold {rec}: static-baseline max difference={max_difference:.3g}")

        fold_rows = result.loc[idx]
        for (unit, now_value), group in fold_rows.groupby(["unit", "now"], sort=True):
            unit = str(unit)
            group_idx = group.index
            cutoff = int(now_value)
            visible = validation_context(train, rec, unit, cutoff)
            for window in WINDOWS:
                ratio, support = robust_level_ratio(visible, model, cutoff, window)
                result.loc[group_idx, f"ratio_w{window}"] = ratio
                result.loc[group_idx, f"support_w{window}"] = support
                for power in POWERS:
                    result.loc[group_idx, f"cal_w{window}_p{power:.2f}"] = (
                        result.loc[group_idx, "global_shrink_prediction"].to_numpy(float)
                        * ratio**power
                    )
    return result


def make_summary(predictions: pd.DataFrame):
    truth = predictions["truth"].to_numpy(float)
    baseline = predictions["global_shrink_prediction"].to_numpy(float)
    baseline_mae = float(np.mean(np.abs(truth - baseline)))
    names = [f"cal_w{w}_p{p:.2f}" for w in WINDOWS for p in POWERS]
    rows = []
    for name in names:
        error = np.abs(truth - predictions[name].to_numpy(float))
        row = {
            "candidate": name,
            "rows": len(predictions),
            "mae": float(error.mean()),
            "gain_vs_v8_1c_pct": 100.0 * (baseline_mae - float(error.mean())) / baseline_mae,
        }
        all_improve = True
        for rec, group in predictions.groupby("recording"):
            rec_error = float(np.mean(np.abs(group["truth"] - group[name])))
            rec_base = float(np.mean(np.abs(group["truth"] - group["global_shrink_prediction"])))
            gain = 100.0 * (rec_base - rec_error) / rec_base if rec_base > 0 else 0.0
            row[f"mae_{rec}"] = rec_error
            row[f"gain_{rec}_pct"] = gain
            all_improve &= gain > 0
        row["improves_both_recordings"] = bool(all_improve)
        by = predictions[["recording", "unit", "minute_of_day", "truth", "global_shrink_prediction", name]].copy()
        by["block"] = (by["minute_of_day"] // 180).astype(int)
        by["base_error"] = np.abs(by["truth"] - by["global_shrink_prediction"])
        by["candidate_error"] = np.abs(by["truth"] - by[name])
        blocks = by.groupby(["recording", "unit", "block"])[["base_error", "candidate_error"]].mean()
        row["block_win_pct"] = 100.0 * float(
            (blocks["candidate_error"] <= blocks["base_error"]).mean()
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mae")
    base_row = {
        "candidate": "global_shrink_prediction",
        "rows": len(predictions),
        "mae": baseline_mae,
        "gain_vs_v8_1c_pct": 0.0,
        "improves_both_recordings": False,
        "block_win_pct": 100.0,
    }
    for rec, group in predictions.groupby("recording"):
        base_row[f"mae_{rec}"] = float(np.mean(np.abs(group["truth"] - group["global_shrink_prediction"])))
        base_row[f"gain_{rec}_pct"] = 0.0
    return pd.concat([pd.DataFrame([base_row]), summary], ignore_index=True).sort_values("mae")


def main():
    for path in (TRAIN_FILE, TEST_FILE, VALIDATION_FILE, BASE_SUBMISSION):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    if OUT_CANDIDATE.exists() or OUT_AUDIT.exists():
        raise SystemExit("Refusing to overwrite an existing V28 candidate or test audit")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    val = pd.read_csv(VALIDATION_FILE)
    base_sub = pd.read_csv(BASE_SUBMISSION)
    if "weekday" not in train.columns:
        raise SystemExit("Training data lacks weekday, required to reproduce V8.1c features")
    weekday_rows = train[["recording", "weekday"]].copy()
    weekday_rows["recording"] = weekday_rows["recording"].astype(str)
    train_weekday_lookup = (
        weekday_rows.drop_duplicates("recording")
        .set_index("recording")["weekday"].astype(str).to_dict()
    )
    for frame in (train, test):
        frame["recording"] = frame["recording"].astype(str)
        frame["unit"] = frame["unit"].astype(str)
        frame["minute_of_day"] = pd.to_numeric(frame["minute_of_day"], errors="raise").astype(int)
        frame["people"] = pd.to_numeric(frame["people"], errors="raise")
        if "weekday" not in frame.columns:
            mapped_recording = frame["recording"].map(MAPPING).fillna(frame["recording"])
            frame["weekday"] = mapped_recording.map(train_weekday_lookup)
            if frame["weekday"].isna().any():
                raise ValueError("Could not map original weekday categories")
    train = train.loc[train["people"].notna()].copy()
    if not np.isfinite(train["people"].to_numpy(float)).all():
        raise ValueError("Training labels must be finite")

    validation = validation_predictions(train, test, val)
    summary = make_summary(validation)
    selected = summary.loc[
        summary["candidate"].ne("global_shrink_prediction")
        & summary["improves_both_recordings"].fillna(False)
        & summary["gain_vs_v8_1c_pct"].ge(1.0)
        & summary["block_win_pct"].ge(50.0)
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUT_VALIDATION, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print("\n=== V28 PRE-NOW LEVEL CALIBRATION ===")
    print(summary.head(12).round(5).to_string(index=False))
    print("\nSaved validation diagnostics:", OUT_VALIDATION, OUT_SUMMARY)

    if selected.empty:
        print("STATUS: NO_ROBUST_GAIN; no test candidate created.")
        print("LOCAL VALIDATION ONLY — NO KAGGLE UPLOAD")
        return

    best = selected.sort_values("mae").iloc[0]
    import re
    match = re.fullmatch(r"cal_w(\d+)_p([0-9.]+)", str(best["candidate"]))
    if not match:
        raise ValueError(f"Cannot parse selected candidate: {best['candidate']}")
    window = int(match.group(1))
    power = float(match.group(2))
    print(
        f"Selected {best['candidate']}: MAE={best['mae']:.5f}; "
        f"gain={best['gain_vs_v8_1c_pct']:.2f}%; "
        f"recording gains={best.get('gain_20260911_1100_pct', np.nan):.2f}% / "
        f"{best.get('gain_20260912_1100_pct', np.nan):.2f}%"
    )

    target_columns = ["id", "recording", "unit", "minute_of_day", "now"]
    if "now_minute_of_day" in test.columns:
        target_columns.append("now_minute_of_day")
    test_targets = test.loc[
        test["id"].notna()
        & test["people"].isna()
        & test["kind"].astype(str).eq("hidden-future"),
        target_columns,
    ].copy()
    test_targets["id"] = test_targets["id"].astype(str)
    test_targets["recording"] = test_targets["recording"].astype(str)
    test_targets["unit"] = test_targets["unit"].astype(str)
    test_targets["minute_of_day"] = pd.to_numeric(test_targets["minute_of_day"], errors="raise").astype(int)
    if "now_minute_of_day" in test_targets.columns:
        parsed_now = pd.to_numeric(test_targets["now_minute_of_day"], errors="coerce")
        missing_now = parsed_now.isna()
        if missing_now.any():
            parsed_now.loc[missing_now] = minute_of_day_values(
                test_targets.loc[missing_now, "now"]
            )
        test_targets["now"] = minute_of_day_values(parsed_now)
    else:
        test_targets["now"] = minute_of_day_values(test_targets["now"])
    base_sub["id"] = base_sub["id"].astype(str)
    base_sub["people"] = pd.to_numeric(base_sub["people"], errors="raise")
    if base_sub["id"].duplicated().any() or test_targets["id"].duplicated().any():
        raise ValueError("Duplicate IDs in test targets or base submission")
    if not set(test.loc[test["id"].notna(), "id"].astype(str)).issubset(set(base_sub["id"])):
        raise ValueError("Base submission does not cover test IDs")

    model = fit_model(train)
    ratio_lookup = {}
    audit_rows = []
    for (rec, unit, now_value), group in test_targets.groupby(
        ["recording", "unit", "now"], sort=True
    ):
        cutoff = int(now_value)
        visible = test_context(test, str(rec), str(unit), cutoff)
        ratio, support = robust_level_ratio(visible, model, cutoff, window)
        factor = ratio**power if support >= 8 else 1.0
        ratio_lookup.update({str(x): factor for x in group["id"]})
        audit_rows.append({
            "recording": rec,
            "unit": unit,
            "now": cutoff,
            "window_minutes": window,
            "ratio": ratio,
            "support_rows": support,
            "power": power,
            "scale_factor": factor,
            "target_rows": len(group),
        })

    old = base_sub["people"].to_numpy(float).copy()
    mask = base_sub["id"].isin(ratio_lookup)
    if int(mask.sum()) != len(ratio_lookup):
        raise ValueError("Hidden-future target ID mismatch")
    factors = base_sub.loc[mask, "id"].map(ratio_lookup).to_numpy(float)
    base_values = base_sub.loc[mask, "people"].to_numpy(float)
    base_sub.loc[mask, "people"] = np.maximum(0.0, base_values * factors)
    new = base_sub["people"].to_numpy(float)
    if not np.isfinite(new).all() or (new < 0).any():
        raise ValueError("Candidate has invalid predictions")
    if not np.array_equal(old[~mask.to_numpy()], new[~mask.to_numpy()]):
        raise ValueError("A non-hidden-future row changed")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    base_sub.to_csv(OUT_CANDIDATE, index=False)
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT, index=False)
    print(f"Test hidden-future rows adjusted: {int(mask.sum())}")
    print(f"Other rows changed: 0")
    print(f"Local candidate saved: {OUT_CANDIDATE}")
    print(f"Test calibration audit saved: {OUT_AUDIT}")
    print("NO KAGGLE UPLOAD PERFORMED")


if __name__ == "__main__":
    main()
