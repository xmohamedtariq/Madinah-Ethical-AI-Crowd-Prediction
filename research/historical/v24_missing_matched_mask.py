from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import v13_missing_reconstruction as v13
import v13_1_missing_robust as prior


KEYS = v13.KEYS
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"
REPORTS = ROOT / "reports"
OLD_PREDICTIONS = REPORTS / "v13_missing_predictions.csv"
OUT_PREDICTIONS = REPORTS / "v24_missing_matched_mask_predictions.csv"
OUT_SUMMARY = REPORTS / "v24_missing_matched_mask_summary.csv"


def contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = sorted(set(int(x) for x in values))
    if not values:
        return []
    result = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def target_mask(test: pd.DataFrame, gold: pd.DataFrame):
    mask = test.loc[
        test["people"].isna() & test["kind"].astype(str).eq("missing"),
        KEYS,
    ].copy()
    mask["recording"] = mask["recording"].map(v13.MAPPING)
    mask = mask.dropna(subset=["recording"])
    mask["recording"] = mask["recording"].astype(str)
    mask["unit"] = mask["unit"].astype(str)
    mask["minute_of_day"] = pd.to_numeric(
        mask["minute_of_day"], errors="raise"
    ).astype(int)
    gold_keys = set(zip(gold.recording, gold.unit, gold.minute_of_day))
    mask["_key"] = list(zip(mask.recording, mask.unit, mask.minute_of_day))
    mask = mask.loc[mask["_key"].isin(gold_keys)].drop(columns="_key")
    mask = mask.drop_duplicates(KEYS).reset_index(drop=True)
    if len(mask) != len(gold):
        raise ValueError(
            f"Mask/truth alignment failed: {len(mask)} mask rows vs {len(gold)} truths"
        )
    return mask


def make_templates(mask: pd.DataFrame):
    templates: dict[str, list[tuple[int, int, int]]] = {}
    run_lookup: dict[tuple[str, str, int], tuple[int, int]] = {}
    for (rec, unit), group in mask.groupby(["recording", "unit"], sort=True):
        minutes = group["minute_of_day"].to_numpy(int)
        for start, end in contiguous_runs(minutes):
            run_minutes = minutes[(minutes >= start) & (minutes <= end)]
            length = end - start + 1
            for minute in run_minutes:
                templates.setdefault(str(unit), []).append(
                    (int(minute), int(length), int(minute - start))
                )
                run_lookup[(str(rec), str(unit), int(minute))] = (start, end)
    return templates, run_lookup


def nearest_index(times: np.ndarray, minute: int) -> int:
    j = int(np.searchsorted(times, minute))
    candidates = [i for i in (j - 1, j) if 0 <= i < len(times)]
    return min(candidates, key=lambda i: abs(times[i] - minute))


def matched_training_data(release: pd.DataFrame, holdout: str, templates):
    builder = v13.GapFeatures(release)
    rng = np.random.default_rng(2401 + int(holdout[-4:]))
    features, labels, weights, bases, records = [], [], [], [], []

    for (rec, unit), (times, values) in builder.series.items():
        rec, unit = str(rec), str(unit)
        if rec == holdout or unit not in templates or not len(templates[unit]):
            continue

        # Match the held-out mask's camera, time-of-day, gap length and
        # position within each gap, while labels come only from other records.
        n_examples = min(320, max(160, 2 * len(times)))
        camera_templates = templates[unit]
        for _ in range(n_examples):
            target_minute, gap_length, offset = camera_templates[
                int(rng.integers(len(camera_templates)))
            ]
            i = nearest_index(times, target_minute)
            minute = int(times[i])
            if abs(minute - target_minute) > 15:
                continue
            start = minute - offset
            end = start + gap_length - 1
            if start < times[0] or end > times[-1]:
                continue
            row, base, scale = builder.make(
                rec, unit, minute, start=start, end=end
            )
            if row is None:
                continue
            features.append(row)
            labels.append((float(values[i]) - base) / scale)
            weights.append(scale)
            bases.append(base)
            records.append(rec)

    if len(features) < 500:
        raise ValueError(f"Too few matched synthetic examples: {len(features)}")
    X = pd.DataFrame(features, dtype=float)
    return builder, X, np.asarray(labels), np.asarray(weights), np.asarray(bases), np.asarray(records)


def main():
    for path in (TRAIN_FILE, TEST_FILE, OLD_PREDICTIONS):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    release, gold, total, extra = v13.prepare(train, test)
    mask = target_mask(test, gold)
    templates, run_lookup = make_templates(mask)

    old = v13.normalize(pd.read_csv(OLD_PREDICTIONS))
    if not {"truth", "linear", "spatial"}.issubset(old.columns):
        raise SystemExit("V13 predictions lack truth/linear/spatial columns")
    out = gold.merge(
        old[KEYS + ["truth", "linear", "spatial"]],
        on=KEYS,
        how="left",
        suffixes=("", "_old"),
        validate="one_to_one",
    )
    if out[["linear", "spatial", "truth_old"]].isna().any().any():
        raise SystemExit("V13 predictions do not cover all validation targets")
    if not np.allclose(out["truth"], out["truth_old"], rtol=0, atol=1e-6):
        raise SystemExit("V13 truth does not align with current validation labels")
    if not np.isfinite(out[["linear", "spatial"]].to_numpy(float)).all():
        raise SystemExit("V13 baseline candidates contain non-finite values")

    out["matched_mask"] = np.nan
    for holdout in sorted(out["recording"].astype(str).unique()):
        builder, X_train, y_train, sample_weight, _, train_records = matched_training_data(
            release, holdout, templates
        )
        fit_rows = int(len(X_train))
        print(
            f"Holdout={holdout}; matched synthetic rows={fit_rows}; "
            f"training recordings={len(np.unique(train_records))}",
            flush=True,
        )

        model = prior.new_model()
        model.fit(X_train, y_train, sample_weight=sample_weight)

        positions = out.index[out["recording"].astype(str).eq(holdout)].to_list()
        target_features, target_bases, target_scales, target_positions = [], [], [], []
        predictions = out.loc[positions, "linear"].to_numpy(float).copy()
        for local_i, row in enumerate(out.loc[positions].itertuples(index=False)):
            key = (str(row.recording), str(row.unit), int(row.minute_of_day))
            if key not in run_lookup:
                raise ValueError(f"Missing target mask run for {key}")
            start, end = run_lookup[key]
            feature_row, base, scale = builder.make(
                str(row.recording), str(row.unit), int(row.minute_of_day),
                start=start, end=end,
            )
            if feature_row is None:
                predictions[local_i] = max(0.0, float(base))
            else:
                target_features.append(feature_row)
                target_bases.append(base)
                target_scales.append(scale)
                target_positions.append(local_i)

        if target_features:
            X_target = pd.DataFrame(target_features, dtype=float).reindex(
                columns=X_train.columns
            )
            residual = model.predict(X_target)
            for local_i, base, scale, value in zip(
                target_positions, target_bases, target_scales, residual
            ):
                predictions[local_i] = max(0.0, base + scale * float(value))

        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite matched predictions for {holdout}")
        out.loc[positions, "matched_mask"] = predictions

    if out["matched_mask"].isna().any():
        raise ValueError("Some validation targets were not predicted")

    candidates = ["linear", "spatial", "matched_mask"]
    truth = out["truth"].to_numpy(float)
    summary_rows = []
    for candidate in candidates:
        row = {
            "candidate": candidate,
            "rows": len(out),
            "mae": float(np.mean(np.abs(truth - out[candidate].to_numpy(float)))),
        }
        for rec, group in out.groupby("recording"):
            row[f"mae_{rec}"] = float(
                np.mean(np.abs(group["truth"] - group[candidate]))
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("mae")

    group_rows = []
    for (rec, unit), group in out.groupby(["recording", "unit"]):
        item = {"recording": rec, "unit": unit, "rows": len(group)}
        for candidate in candidates:
            item[f"{candidate}_mae"] = float(
                np.mean(np.abs(group["truth"] - group[candidate]))
            )
        group_rows.append(item)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PREDICTIONS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Validation missing targets: {len(out)}/{total}; removed extra context={extra}")
    print("\n=== MASK-MATCHED MISSING RESULTS ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== BY RECORDING / CAMERA ===")
    print(pd.DataFrame(group_rows).round(5).to_string(index=False))
    print("\nSaved local diagnostic:", OUT_PREDICTIONS)
    print("NO SUBMISSION CREATED OR UPLOADED")


if __name__ == "__main__":
    main()
