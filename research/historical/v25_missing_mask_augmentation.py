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
import v13_2_missing_blocked as v132


KEYS = v13.KEYS
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"
REPORTS = ROOT / "reports"
OLD_PREDICTIONS = REPORTS / "v13_missing_predictions.csv"
OUT_PREDICTIONS = REPORTS / "v25_missing_mask_augmentation_predictions.csv"
OUT_SUMMARY = REPORTS / "v25_missing_mask_augmentation_summary.csv"


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


def target_mask(test: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    mask = test.loc[
        test["people"].isna() & test["kind"].astype(str).eq("missing"), KEYS
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
    rng = np.random.default_rng(2501 + int(holdout[-4:]))
    features, labels, weights, records = [], [], [], []

    for (rec, unit), (times, values) in builder.series.items():
        rec, unit = str(rec), str(unit)
        if rec == holdout or unit not in templates or not templates[unit]:
            continue
        camera_templates = templates[unit]
        n_examples = min(320, max(160, 2 * len(times)))
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
            records.append(rec)

    if len(features) < 500:
        raise ValueError(f"Too few matched synthetic examples: {len(features)}")
    return (
        pd.DataFrame(features, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(weights, dtype=float),
        np.asarray(records, dtype=str),
    )


def fit_and_predict(
    random_x: pd.DataFrame,
    random_y: np.ndarray,
    random_weight: np.ndarray,
    matched_x: pd.DataFrame,
    matched_y: np.ndarray,
    matched_weight: np.ndarray,
    target_x: pd.DataFrame,
    columns: list[str],
    matched_factor: float,
) -> np.ndarray:
    x_random = random_x.reindex(columns=columns)
    y_parts = [random_y]
    weight_parts = [random_weight]
    x_parts = [x_random]

    if matched_factor > 0:
        x_parts.append(matched_x.reindex(columns=columns))
        y_parts.append(matched_y)
        weight_parts.append(matched_weight * matched_factor)

    x_fit = pd.concat(x_parts, ignore_index=True)
    y_fit = np.concatenate(y_parts)
    weights = np.concatenate(weight_parts)
    if not np.isfinite(y_fit).all() or not np.isfinite(weights).all():
        raise ValueError("Non-finite synthetic labels or sample weights")

    model = prior.new_model()
    model.fit(x_fit, y_fit, sample_weight=weights)
    return model.predict(target_x.reindex(columns=columns))


def main():
    for path in (TRAIN_FILE, TEST_FILE, OLD_PREDICTIONS):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    if OUT_PREDICTIONS.exists() or OUT_SUMMARY.exists():
        raise SystemExit("Refusing to overwrite an existing V25 report")
    if not hasattr(prior, "_V13FoldSafeModel"):
        raise SystemExit("V13 fold-feature guard is not installed")

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
        suffixes=("", "_saved"),
        validate="one_to_one",
    )
    if out[["linear", "spatial", "truth_saved"]].isna().any().any():
        raise SystemExit("V13 predictions do not cover all validation targets")
    if not np.allclose(out["truth"], out["truth_saved"], rtol=0, atol=1e-6):
        raise SystemExit("V13 validation truth does not align with current labels")
    if not np.isfinite(out[["linear", "spatial"]].to_numpy(float)).all():
        raise SystemExit("V13 baseline predictions contain non-finite values")

    _, random_x, random_y, random_scale, _, random_meta = v132.training_data(release)
    random_x = pd.DataFrame(random_x, dtype=float)
    random_y = np.asarray(random_y, dtype=float)
    random_scale = np.asarray(random_scale, dtype=float)
    random_meta = random_meta.reset_index(drop=True)
    random_x = random_x.reset_index(drop=True)
    if not (len(random_x) == len(random_y) == len(random_scale) == len(random_meta)):
        raise ValueError("V13 random synthetic arrays are misaligned")

    candidate_factors = (0.0, 0.25, 0.5, 1.0)
    candidate_names = {
        0.0: "random_only",
        0.25: "augment_025",
        0.5: "augment_050",
        1.0: "augment_100",
    }
    for name in candidate_names.values():
        out[name] = np.nan

    for holdout in sorted(out["recording"].astype(str).unique()):
        random_fit = random_meta["recording"].astype(str).ne(holdout).to_numpy()
        if random_fit.sum() < 1000:
            raise ValueError(f"Too few random examples for {holdout}")
        matched_x, matched_y, matched_weight, matched_records = matched_training_data(
            release, holdout, templates
        )
        columns = sorted(set(random_x.columns) | set(matched_x.columns))
        random_x_fold = random_x.loc[random_fit].reset_index(drop=True)
        random_y_fold = random_y[random_fit]
        random_weight_fold = random_scale[random_fit]

        positions = out.index[out["recording"].astype(str).eq(holdout)].to_list()
        target_rows = out.loc[positions]
        builder = v13.GapFeatures(release)
        target_features, target_bases, target_scales, target_local_positions = [], [], [], []
        fallback = target_rows["linear"].to_numpy(float).copy()
        for local_i, row in enumerate(target_rows.itertuples(index=False)):
            key = (str(row.recording), str(row.unit), int(row.minute_of_day))
            if key not in run_lookup:
                raise ValueError(f"Missing actual-mask run for {key}")
            start, end = run_lookup[key]
            feature_row, base, scale = builder.make(
                str(row.recording), str(row.unit), int(row.minute_of_day),
                start=start, end=end,
            )
            if feature_row is None:
                continue
            target_features.append(feature_row)
            target_bases.append(base)
            target_scales.append(scale)
            target_local_positions.append(local_i)

        if not target_features:
            raise ValueError(f"No supported target features for {holdout}")
        target_x = pd.DataFrame(target_features, dtype=float)
        for factor in candidate_factors:
            residual = fit_and_predict(
                random_x_fold,
                random_y_fold,
                random_weight_fold,
                matched_x,
                matched_y,
                matched_weight,
                target_x,
                columns,
                factor,
            )
            prediction = fallback.copy()
            for local_i, base, scale, value in zip(
                target_local_positions, target_bases, target_scales, residual
            ):
                prediction[local_i] = max(0.0, float(base) + float(scale) * float(value))
            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite prediction for {holdout}, factor={factor}")
            out.loc[positions, candidate_names[factor]] = prediction

        print(
            f"Holdout={holdout}; random train={int(random_fit.sum())}; "
            f"matched added={len(matched_x)}; matched recordings="
            f"{len(np.unique(matched_records))}; target feature coverage="
            f"{len(target_features)}/{len(target_rows)}",
            flush=True,
        )

    candidates = ["linear", "spatial", *candidate_names.values()]
    if out[candidates].isna().any().any():
        raise ValueError("Incomplete V25 validation predictions")
    truth = out["truth"].to_numpy(float)
    summary_rows = []
    for candidate in candidates:
        error = np.abs(truth - out[candidate].to_numpy(float))
        row = {"candidate": candidate, "rows": len(out), "mae": float(error.mean())}
        for rec, positions in out.groupby("recording").groups.items():
            row[f"mae_{rec}"] = float(
                np.mean(np.abs(out.loc[positions, "truth"] - out.loc[positions, candidate]))
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
    print("\n=== V25 RANDOM + MASK-MATCHED AUGMENTATION ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== BY RECORDING / CAMERA ===")
    print(pd.DataFrame(group_rows).round(5).to_string(index=False))
    print("\nSaved local diagnostics:", OUT_PREDICTIONS, "and", OUT_SUMMARY)
    print("NO SUBMISSION CREATED OR UPLOADED")


if __name__ == "__main__":
    main()
