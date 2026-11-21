"""Audit Madinah local validation coverage. Does not train or create submissions."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


KINDS = ("forecast", "missing", "hidden", "hidden-future")
KEYS = ["recording", "unit", "minute_of_day"]
TEST_TO_VAL = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}


def normalize_keys(frame):
    result = frame.copy()
    for name in ("recording", "unit"):
        result[name] = result[name].astype("string")
        if result[name].isna().any():
            raise ValueError(f"Null key: {name}")
    minutes = pd.to_numeric(result["minute_of_day"], errors="raise")
    if not np.isfinite(minutes.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite minute key")
    if (minutes != np.floor(minutes)).any():
        raise ValueError("Non-integer minute key")
    result["minute_of_day"] = minutes.astype("int64")
    return result


def strict_competition_score(frame, reference_col, pred_col="prediction", *, expected):
    """Return a LOCAL reference ratio only on the complete expected validation set.

    The supplied reference must be independently reconstructed and verified.
    This function does not certify causality, model provenance, or Kaggle score.
    """
    required = KEYS + ["kind", "truth", reference_col, pred_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    f = normalize_keys(frame)
    e = normalize_keys(expected)
    if f.duplicated(KEYS).any() or e.duplicated(KEYS).any():
        raise ValueError("Duplicate validation keys")
    if f["kind"].isna().any():
        raise ValueError("Null hole kind")
    labels = set(f["kind"])
    if labels != set(KINDS):
        raise ValueError(f"Need all four kinds. Found: {sorted(labels)}")
    left = pd.MultiIndex.from_frame(f[KEYS])
    right = pd.MultiIndex.from_frame(e[KEYS])
    if len(left.difference(right)) or len(right.difference(left)):
        raise ValueError("Partial or mismatched validation coverage")
    if not {"kind", "truth"}.issubset(e.columns):
        raise ValueError("Expected manifest must contain kind and truth")
    checked = f.merge(e[KEYS + ["kind", "truth"]], on=KEYS,
                      validate="one_to_one", suffixes=("", "_expected"))
    if (checked["kind"] != checked["kind_expected"]).any():
        raise ValueError("Kind differs from expected manifest")
    values = checked[["truth", "truth_expected", reference_col, pred_col]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite truth, reference, or prediction")
    if not np.allclose(values["truth"], values["truth_expected"], rtol=0, atol=1e-9):
        raise ValueError("Truth differs from expected manifest")
    ratios = []
    for kind in KINDS:
        mask = checked["kind"] == kind
        truth = values.loc[mask, "truth"].to_numpy(dtype=float)
        ref = values.loc[mask, reference_col].to_numpy(dtype=float)
        pred = values.loc[mask, pred_col].to_numpy(dtype=float)
        denominator = np.abs(truth - ref).mean()
        if denominator <= 0:
            raise ValueError(f"Zero reference MAE for {kind}")
        ratios.append(float(np.abs(truth - pred).mean() / denominator))
    return float(np.mean(ratios))


def run(root):
    raw = root / "data" / "raw"
    reports = root / "reports"
    for path in (raw / "train.csv", raw / "test.csv"):
        if not path.is_file():
            raise ValueError(f"Missing file: {path}")
    train = normalize_keys(pd.read_csv(raw / "train.csv"))
    test = normalize_keys(pd.read_csv(raw / "test.csv"))
    train["people"] = pd.to_numeric(train["people"], errors="raise")
    test["people"] = pd.to_numeric(test["people"], errors="raise")
    holes = test.loc[test["people"].isna()].copy()
    if holes.empty or holes["kind"].isna().any():
        raise ValueError("Empty targets or missing hole labels")
    if set(holes["kind"]) != set(KINDS):
        raise ValueError("Test targets do not contain exactly the four expected kinds")
    if holes.duplicated(KEYS).any():
        raise ValueError("Duplicate test target keys; inspect before proceeding")
    unknown = sorted(set(holes["recording"]) - set(TEST_TO_VAL))
    if unknown:
        raise ValueError(f"Unmapped test recordings: {unknown}")

    expected = holes[KEYS + ["kind"]].copy()
    expected["recording"] = expected["recording"].map(TEST_TO_VAL).astype("string")
    truth = train.groupby(KEYS, as_index=False)["people"].mean().rename(
        columns={"people": "truth"}
    )
    expected = expected.merge(truth, on=KEYS, how="left", validate="one_to_one")
    expected["has_truth"] = np.isfinite(expected["truth"].to_numpy(dtype=float))
    coverage = expected.groupby("kind").agg(
        test_targets=("kind", "size"), mapped_validation_truth=("has_truth", "sum")
    ).reindex(KINDS)
    coverage["coverage_pct"] = (
        100 * coverage["mapped_validation_truth"] / coverage["test_targets"]
    )
    print("\n=== TEST-LIKE VALIDATION COVERAGE ===")
    print("Mapping: Sep18 -> Sep11; Sep19 -> Sep12")
    print(coverage.round(2).to_string())
    print("Unlabelled mapped rows cannot be scored; they are not zero-error rows.")

    available = expected.loc[expected["has_truth"]].copy()
    detail = available.groupby(["recording", "kind"]).size().unstack(fill_value=0)
    print("\n=== LABELLED ROWS BY VALIDATION RECORDING ===")
    print(detail.reindex(columns=KINDS, fill_value=0).to_string())

    current_path = reports / "v11_5_validation_predictions.csv"
    if current_path.is_file():
        current = normalize_keys(pd.read_csv(current_path))
        hidden = available.loc[available["kind"] == "hidden"]
        if current.duplicated(KEYS).any():
            raise ValueError("V11.5 contains duplicate keys")
        matched = hidden.merge(current[KEYS], on=KEYS, how="inner", validate="one_to_one")
        print("\n=== V11.5 HIDDEN COVERAGE ===")
        print(f"Covered labelled Hidden rows: {len(matched)} / {len(hidden)}")
        print(f"Rows outside labelled Hidden manifest: {len(current) - len(matched)}")
        hidden_index = pd.MultiIndex.from_frame(hidden[KEYS])
        current_index = pd.MultiIndex.from_frame(current[KEYS])
        uncovered = hidden.loc[~hidden_index.isin(current_index)]
        print("Uncovered labelled Hidden rows by recording/unit:")
        if uncovered.empty:
            print("NONE")
        else:
            print(uncovered.groupby(["recording", "unit"]).size().to_string())

    print("\n=== EXISTING ROW-LEVEL VALIDATION REPORTS ===")
    inventory = []
    warnings = []
    if reports.is_dir():
        for path in reports.glob("*.csv"):
            try:
                columns = list(pd.read_csv(path, nrows=0).columns)
                if "truth" not in columns:
                    continue
                labels = "NO_KIND_COLUMN"
                four = False
                if "kind" in columns:
                    kinds = pd.read_csv(path, usecols=["kind"])["kind"]
                    present = set(kinds.dropna())
                    four = present == set(KINDS) and not kinds.isna().any()
                    labels = ",".join(sorted(map(str, present)))
                predictors = [c for c in columns if c not in KEYS + [
                    "truth", "kind", "id", "day", "time", "test_recording",
                    "now", "now_minute_of_day", "people", "has_truth",
                ]]
                inventory.append((four, path.stat().st_mtime, path.name, labels, predictors))
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError,
                    OSError, ValueError) as error:
                warnings.append(f"{path.name}: {type(error).__name__}")
    for four, _, name, labels, columns in sorted(inventory, key=lambda x: (x[0], x[1]), reverse=True)[:12]:
        print(f"\n{name}")
        print(f"  Kinds: {labels}")
        print("  Columns: " + ", ".join(columns[:14]) + (", ..." if len(columns) > 14 else ""))
    if not inventory:
        print("No CSV reports with a 'truth' column found.")
    if warnings:
        print("\nUnreadable CSV reports:")
        print("\n".join(warnings[:8]))
    print("\n=== STATUS ===")
    print("Full four-kind score: NOT ESTABLISHED BY THIS AUDIT")
    print("Header/kind coverage does not verify baseline or prediction provenance.")
    print("Strict scorer is available in this script; src/metric.py is unchanged.")
    print("READ ONLY - NO TRAINING - NO SUBMISSION CREATED OR UPLOADED")


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parent
    if default_root.name == "src":
        default_root = default_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()
    try:
        run(args.root.resolve())
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as error:
        raise SystemExit(f"AUDIT STOPPED: {error}") from error
