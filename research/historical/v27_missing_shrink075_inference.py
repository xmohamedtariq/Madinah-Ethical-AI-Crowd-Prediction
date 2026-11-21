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
DATA = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
SUBMISSIONS = ROOT / "submissions"
TRAIN_FILE = DATA / "train.csv"
TEST_FILE = DATA / "test.csv"
BASE_SUB = SUBMISSIONS / (
    "submission_v6_4_fixed_v5blend_v11_2_v8_future_"
    "no_v13_missing_v15_prayer_full.csv"
)
OUT_SUB = SUBMISSIONS / (
    "submission_v6_4_fixed_v5blend_v11_2_v8_future_"
    "no_v13_missing_v15_prayer_full_v27_missing_shrink075_candidate.csv"
)
OUT_AUDIT = REPORTS / "v27_missing_shrink075_test_audit.csv"

MATCHED_TRAIN_WEIGHT = 0.25
PREDICTION_SHRINK = 0.75


def contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = sorted(set(int(x) for x in values))
    if not values:
        return []
    runs = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def test_templates(test: pd.DataFrame, train_units: set[str]):
    targets = test.loc[
        test["id"].notna()
        & test["people"].isna()
        & test["kind"].astype(str).eq("missing"),
        ["id", "recording", "unit", "minute_of_day"],
    ].copy()
    targets["id"] = targets["id"].astype(str)
    targets["recording"] = targets["recording"].astype(str)
    targets["unit"] = targets["unit"].astype(str)
    targets["minute_of_day"] = pd.to_numeric(
        targets["minute_of_day"], errors="raise"
    ).astype(int)
    if targets["id"].duplicated().any():
        raise ValueError("Duplicate IDs among missing targets")

    templates: dict[str, list[tuple[int, int, int]]] = {}
    run_lookup: dict[tuple[str, str, int], tuple[int, int]] = {}
    for (recording, unit), group in targets.groupby(["recording", "unit"], sort=True):
        minutes = group["minute_of_day"].to_numpy(int)
        for start, end in contiguous_runs(minutes):
            length = end - start + 1
            run_minutes = minutes[(minutes >= start) & (minutes <= end)]
            for minute in run_minutes:
                templates.setdefault(unit, []).append(
                    (int(minute), int(length), int(minute - start))
                )
                run_lookup[(recording, unit, int(minute))] = (start, end)

    supported = targets.loc[targets["unit"].isin(train_units)].copy().reset_index(drop=True)
    return targets.reset_index(drop=True), supported, templates, run_lookup


def nearest_index(times: np.ndarray, minute: int) -> int:
    j = int(np.searchsorted(times, minute))
    candidates = [i for i in (j - 1, j) if 0 <= i < len(times)]
    return min(candidates, key=lambda i: abs(times[i] - minute))


def matched_training_data(train: pd.DataFrame, templates):
    builder = v13.GapFeatures(train)
    rng = np.random.default_rng(2701)
    features, labels, weights = [], [], []
    records = set()

    for (recording, unit), (times, values) in builder.series.items():
        recording, unit = str(recording), str(unit)
        camera_templates = templates.get(unit, [])
        if not camera_templates:
            continue
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
                recording, unit, minute, start=start, end=end
            )
            if row is None:
                continue
            features.append(row)
            labels.append((float(values[i]) - base) / scale)
            weights.append(scale)
            records.add(recording)

    if len(features) < 500:
        raise ValueError(f"Too few matched synthetic training rows: {len(features)}")
    return (
        pd.DataFrame(features, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(weights, dtype=float),
        len(records),
    )


def main():
    for path in (TRAIN_FILE, TEST_FILE, BASE_SUB):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    if OUT_SUB.exists() or OUT_AUDIT.exists():
        raise SystemExit("Refusing to overwrite an existing V27 output")
    if not hasattr(prior, "_V13FoldSafeModel"):
        raise SystemExit("V13 fold-feature guard is not installed")

    train = v13.normalize(pd.read_csv(TRAIN_FILE))
    train["people"] = pd.to_numeric(train["people"], errors="raise")
    if not np.isfinite(train["people"].to_numpy(float)).all():
        raise ValueError("Training labels must be finite")
    test = v13.normalize(pd.read_csv(TEST_FILE))
    test["people"] = pd.to_numeric(test["people"], errors="raise")
    train_units = set(train["unit"].astype(str).unique())

    all_targets, supported, templates, run_lookup = test_templates(test, train_units)
    if len(all_targets) != 464:
        raise ValueError(f"Expected 464 missing test targets; found {len(all_targets)}")
    if supported.empty:
        raise ValueError("No missing targets belong to cameras with training history")

    sub = pd.read_csv(BASE_SUB)
    if not {"id", "people"}.issubset(sub.columns):
        raise ValueError("Base submission must contain id and people")
    sub["id"] = sub["id"].astype(str)
    if sub["id"].duplicated().any():
        raise ValueError("Duplicate IDs in base submission")
    test_ids = test.loc[test["id"].notna(), "id"].astype(str)
    if test_ids.duplicated().any() or set(test_ids) != set(sub["id"]):
        raise ValueError("Base submission IDs do not exactly match test target IDs")
    sub["people"] = pd.to_numeric(sub["people"], errors="raise")
    original = sub["people"].to_numpy(float).copy()
    if not np.isfinite(original).all() or (original < 0).any():
        raise ValueError("Base submission contains invalid predictions")
    original_lookup = dict(zip(sub["id"], original))

    # Train synthetic residual models using labeled training data only.
    _, random_x, random_y, random_scale, _, random_meta = v132.training_data(train)
    random_x = pd.DataFrame(random_x, dtype=float).reset_index(drop=True)
    random_y = np.asarray(random_y, dtype=float)
    random_scale = np.asarray(random_scale, dtype=float)
    if not (len(random_x) == len(random_y) == len(random_scale) == len(random_meta)):
        raise ValueError("Random synthetic training arrays are misaligned")

    matched_x, matched_y, matched_scale, matched_recordings = matched_training_data(
        train, templates
    )
    columns = sorted(set(random_x.columns) | set(matched_x.columns))
    random_x = random_x.reindex(columns=columns)
    matched_x = matched_x.reindex(columns=columns)

    random_model = prior.new_model()
    random_model.fit(random_x, random_y, sample_weight=random_scale)

    aug_x = pd.concat([random_x, matched_x], ignore_index=True)
    aug_y = np.concatenate([random_y, matched_y])
    aug_weight = np.concatenate(
        [random_scale, matched_scale * MATCHED_TRAIN_WEIGHT]
    )
    augmented_model = prior.new_model()
    augmented_model.fit(aug_x, aug_y, sample_weight=aug_weight)

    # Test rows are used only as released context. New cameras without training
    # history and unsupported gaps retain their current base-submission values.
    visible = test.loc[test["people"].notna(), KEYS + ["people"]].copy()
    visible = visible.loc[visible["unit"].astype(str).isin(train_units)]
    context = pd.concat([train[KEYS + ["people"]], visible], ignore_index=True)
    context = v13.normalize(context)
    context["people"] = pd.to_numeric(context["people"], errors="raise")
    inference_builder = v13.GapFeatures(context)

    feature_rows, bases, scales, ids = [], [], [], []
    fallback_ids = []
    for row in supported.itertuples(index=False):
        key = (str(row.recording), str(row.unit), int(row.minute_of_day))
        if key not in run_lookup:
            raise ValueError(f"Missing test mask interval for {key}")
        start, end = run_lookup[key]
        feature, base, scale = inference_builder.make(
            str(row.recording), str(row.unit), int(row.minute_of_day),
            start=start, end=end,
        )
        if feature is None:
            fallback_ids.append(str(row.id))
            continue
        feature_rows.append(feature)
        bases.append(float(base))
        scales.append(float(scale))
        ids.append(str(row.id))

    if not feature_rows:
        raise ValueError("No missing targets had supported spatial features")
    target_x = pd.DataFrame(feature_rows, dtype=float).reindex(columns=columns)
    spatial_residual = random_model.predict(target_x)
    augmented_residual = augmented_model.predict(target_x)
    base_values = np.asarray(bases, dtype=float)
    scales = np.asarray(scales, dtype=float)
    spatial_prediction = np.maximum(0.0, base_values + scales * spatial_residual)
    augmented_prediction = np.maximum(0.0, base_values + scales * augmented_residual)
    prediction = spatial_prediction + PREDICTION_SHRINK * (
        augmented_prediction - spatial_prediction
    )
    prediction = np.maximum(0.0, prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("V27 generated non-finite predictions")

    prediction_lookup = dict(zip(ids, prediction))
    sub["people"] = sub["people"].astype(float)
    mask = sub["id"].isin(prediction_lookup)
    if int(mask.sum()) != len(prediction_lookup):
        raise ValueError("Updated missing IDs do not align with the base submission")
    sub.loc[mask, "people"] = sub.loc[mask, "id"].map(prediction_lookup).to_numpy(float)
    updated = sub["people"].to_numpy(float)
    untouched = ~mask.to_numpy()
    if not np.array_equal(original[untouched], updated[untouched]):
        raise ValueError("A non-updated submission row changed")
    if not np.isfinite(updated).all() or (updated < 0).any():
        raise ValueError("Candidate contains invalid predictions")

    audit = supported[["id", "recording", "unit", "minute_of_day"]].copy()
    audit["updated"] = audit["id"].astype(str).isin(prediction_lookup)
    audit["fallback_reason"] = np.where(
        audit["updated"], "", "unsupported_gap_keep_base"
    )
    audit["base_submission"] = audit["id"].astype(str).map(original_lookup)
    audit["prediction"] = audit["id"].astype(str).map(prediction_lookup)
    audit["prediction"] = audit["prediction"].fillna(
        audit["id"].astype(str).map(original_lookup)
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_SUB, index=False)
    audit.to_csv(OUT_AUDIT, index=False)

    print(f"Missing test targets: {len(all_targets)}")
    print(f"Targets with cameras represented in training: {len(supported)}")
    print(f"Targets updated with V27 model: {len(prediction_lookup)}")
    print(f"Unsupported gaps kept from base: {len(fallback_ids)}")
    print(f"No-training-history cameras kept from base: {len(all_targets)-len(supported)}")
    print(f"Random synthetic rows: {len(random_x)}")
    print(f"Mask-matched synthetic rows: {len(matched_x)} across {matched_recordings} recordings")
    print(f"Base candidate: {BASE_SUB}")
    print(f"Local candidate saved: {OUT_SUB}")
    print(f"Audit saved: {OUT_AUDIT}")
    print("Non-missing and unsupported rows preserved.")
    print("NO KAGGLE UPLOAD PERFORMED")


if __name__ == "__main__":
    main()
