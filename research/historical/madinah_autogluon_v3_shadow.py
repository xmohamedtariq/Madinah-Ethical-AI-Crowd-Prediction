"""Madinah AutoGluon V3: chronological shadow-test of V2 blends.

Analyzes V2 *saved out-of-day predictions*; NEVER trains models, reads Kaggle
labels, writes a submission, or uploads anything. Uses the earlier validation
calendar day to choose a per-kind blend, then scores that frozen selection on
an independently later validation day. This is a diagnostic, not the Kaggle
metric and not a replacement for a truly untouched validation set.

PowerShell (project root):
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v3_shadow.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_COLS = ["recording", "day", "unit", "minute_of_day", "kind", "people"]
PRIMARY = "sp_rule_50"
PREDEFINED = ("pooled", "specialists", "sp_rule_25", "sp_rule_50", "rules_only", "kind_safe")
FUTURE_KINDS = {"forecast", "hidden-future"}


def compute_candidates(df: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = set(BASE_COLS + list(PREDEFINED)) - set(df.columns)
    if missing:
        raise ValueError(f"V2 CV data missing columns: {sorted(missing)}")
    result = {}
    for name in PREDEFINED:
        pred = pd.to_numeric(df[name], errors="raise").to_numpy(float)
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError(f"Invalid predictions: {name}")
        result[name] = pred
    anchor, specialist, pooled = result["rules_only"], result["specialists"], result["pooled"]
    # The grid is fixed before looking at any validation labels.
    for alpha in (0.25, 0.5, 0.75):
        label = f"sp_plus_rule_{alpha:.2f}"
        result[label] = alpha * specialist + (1-alpha) * anchor
        label = f"pool_plus_rule_{alpha:.2f}"
        result[label] = alpha * pooled + (1-alpha) * anchor
        label = f"pool_plus_sp_{alpha:.2f}"
        result[label] = alpha * pooled + (1-alpha) * specialist
    return result


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def report_frame(block: pd.DataFrame, pool: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    y = block["people"].to_numpy(float)[mask]
    return {name: mae(y, preds[mask]) for name, preds in pool.items()}


def analyze(path: Path, reports: Path, min_group: int, min_gain: float) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2 holdout predictions: {path}. Run V2 --mode validate first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("V2 predictions file is empty")
    keys = ["recording", "unit", "minute_of_day", "kind"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate V2 validation targets; cannot safely score")
    for field in ["day", "kind", "unit"]:
        if df[field].isna().any():
            raise ValueError(f"Missing values in {field}")
    y = pd.to_numeric(df["people"], errors="raise").to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Non-finite or negative validation truth")
    df = df.copy()
    df["people"] = y
    if "horizon" in df:
        h = pd.to_numeric(df["horizon"], errors="coerce").to_numpy(float)
        future = df["kind"].astype(str).isin(FUTURE_KINDS).to_numpy()
        if future.any() and (not np.isfinite(h[future]).all() or (h[future] < 0).any()):
            raise ValueError("Invalid horizon on future targets")
    dates = sorted(df["day"].astype(str).unique())
    if len(dates) != 2:
        raise ValueError(f"Need two chronological CV days; got {dates}")
    first = (df["day"].astype(str) == dates[0]).to_numpy()
    second = (df["day"].astype(str) == dates[1]).to_numpy()
    kinds = df["kind"].astype(str).to_numpy()
    pool = compute_candidates(df)
    primary = pool[PRIMARY]
    selected = {k: PRIMARY for k in sorted(set(kinds))}
    policy_details = []
    for kind in selected:
        mask = first & (kinds == kind)
        baseline_error = mae(y[mask], primary[mask]) if mask.any() else math.nan
        if int(mask.sum()) >= min_group and baseline_error > 0:
            scores = {name: mae(y[mask], pred[mask]) for name, pred in pool.items()}
            best = min(scores, key=scores.get)
            gain = 100.0*(baseline_error - scores[best]) / baseline_error
            # Predeclared conservative threshold; no label from later validation day used.
            if gain >= min_gain:
                selected[kind] = best
        policy_details.append({"kind": kind, "train_count": int(mask.sum()),
                               "early_baseline_mae": baseline_error,
                               "selected": selected[kind],
                               "early_selected_mae": mae(y[mask], pool[selected[kind]][mask]) if mask.any() else None})
    chosen = primary.copy()
    for kind, name in selected.items():
        chosen[kinds == kind] = pool[name][kinds == kind]
    early_base, early_policy = mae(y[first], primary[first]), mae(y[first], chosen[first])
    later_base, later_policy = mae(y[second], primary[second]), mae(y[second], chosen[second])
    total_base, total_policy = mae(y, primary), mae(y, chosen)
    rows = []
    for day in dates:
        day_mask = (df["day"].astype(str) == day).to_numpy()
        for kind in sorted(set(kinds)):
            mask = day_mask & (kinds == kind)
            if not mask.any():
                continue
            rows.append({"day":day,"kind":kind,"n":int(mask.sum()),
                         "v2_champion_mae":mae(y[mask],primary[mask]),
                         "v3_shadow_mae":mae(y[mask],chosen[mask]),
                         "frozen_candidate":selected[kind]})
    camera_rows = []
    for (kind, unit), subgroup in df.groupby(["kind", "unit"], sort=True):
        mask = ((kinds == kind) & (df["unit"].astype(str).to_numpy() == unit))
        if not mask.any():
            continue
        camera_rows.append({"kind":kind,"unit":unit,"n":int(mask.sum()),
                            "v2_champion_mae":mae(y[mask],primary[mask]),
                            "v3_shadow_mae":mae(y[mask],chosen[mask]),
                            "v2_total_abs_error":float(np.abs(y[mask]-primary[mask]).sum())})
    # Diagnostic gating uses the LATER day alone, without changing the frozen policy.
    improvement = 100.0 * (later_base - later_policy) / later_base if later_base > 0 else 0.0
    later_kind = [r for r in rows if r["day"] == dates[1]]
    severe_regression = any(r["n"] >= 40 and r["v3_shadow_mae"] > r["v2_champion_mae"] * 1.10 for r in later_kind)
    passed = bool(improvement >= 2.0 and not severe_regression)
    verdict = ("PASS_DIAGNOSTIC_ONLY" if passed else "KEEP_V2")
    summary = {
        "warning": "Held-out MAE is in people, NOT official Kaggle score. V2 already chose hyperparameters using both folds; this shadow test is NOT independent final validation.",
        "days": dates,"rows":len(df),"selection_train_rows":int(first.sum()),"selection_test_rows":int(second.sum()),
        "fixed_v2_champion":PRIMARY,"selected_per_kind":selected,"selection_details":policy_details,
        "early_mae": {"v2":early_base,"shadow":early_policy},
        "later_mae": {"v2":later_base,"shadow":later_policy},
        "combined_mae": {"v2":total_base,"shadow":total_policy},
        "later_gain_percent":improvement,"no_later_kind_regression_over_10pct":not severe_regression,
        "gate":"Later-day shadow improves V2 by >=2%, with no >=40-row kind worsening >10%.",
        "verdict":verdict,"competition_score_verified":False,
        "no_models_trained":True,"no_submission_created":True,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports/"ag_v3_shadow_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame(rows).to_csv(reports/"ag_v3_shadow_per_kind.csv", index=False)
    pd.DataFrame(camera_rows).sort_values("v2_total_abs_error",ascending=False).to_csv(
        reports/"ag_v3_shadow_by_camera.csv", index=False)
    print("\n=== MADINAH V3 CHRONOLOGICAL SHADOW DIAGNOSTIC ===")
    print(f"Early day: {dates[0]} ({int(first.sum())} rows) | Late day: {dates[1]} ({int(second.sum())} rows)")
    print("Selected on EARLY day only:")
    for kind, name in selected.items():
        print(f"  {kind:<16} {name}")
    print(f"Early-day V2 / shadow MAE: {early_base:.6f} / {early_policy:.6f} (selection set; NOT independent)")
    print(f"Later-day V2 / shadow MAE: {later_base:.6f} / {later_policy:.6f} (shadow holdout)")
    print(f"Later-day gain: {improvement:+.2f}%")
    print(f"Combined V2 / shadow MAE: {total_base:.6f} / {total_policy:.6f} (NOT unbiased)")
    print(f"Gate: {verdict}")
    print("Saved reports/ag_v3_shadow_summary.json, ag_v3_shadow_per_kind.csv, ag_v3_shadow_by_camera.csv")
    print("No Kaggle submission made. Do not claim 0.400 from these MAE values.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit V2 blends with chronological policy selection (no training/upload)")
    parser.add_argument("--input",default="reports/ag_v2_cv_predictions.csv")
    parser.add_argument("--reports",default="reports")
    parser.add_argument("--min-kind-rows",type=int,default=40)
    parser.add_argument("--min-training-gain",type=float,default=2.0)
    args=parser.parse_args()
    if args.min_kind_rows < 1 or args.min_training_gain < 0:
        parser.error("Invalid optimization thresholds")
    analyze(Path(args.input),Path(args.reports),args.min_kind_rows,args.min_training_gain)

if __name__ == "__main__":
    main()
