from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE = REPORTS / "v25_missing_mask_augmentation_predictions.csv"
OUT_PREDICTIONS = REPORTS / "v26_missing_blend_sweep_predictions.csv"
OUT_SUMMARY = REPORTS / "v26_missing_blend_sweep_summary.csv"
OUT_GROUPS = REPORTS / "v26_missing_blend_sweep_by_camera.csv"
OUT_BLOCKS = REPORTS / "v26_missing_blend_sweep_by_block.csv"


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Missing V25 validation predictions: {SOURCE}")
    if any(path.exists() for path in (OUT_PREDICTIONS, OUT_SUMMARY, OUT_GROUPS, OUT_BLOCKS)):
        raise SystemExit("Refusing to overwrite an existing V26 report")

    frame = pd.read_csv(SOURCE)
    required = {
        "recording", "unit", "minute_of_day", "truth", "spatial", "augment_025"
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"V25 report lacks columns: {sorted(missing)}")
    for column in ("minute_of_day", "truth", "spatial", "augment_025"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[["minute_of_day", "truth", "spatial", "augment_025"]]).all(axis=1)
    ].copy().reset_index(drop=True)
    if len(frame) != 358:
        raise SystemExit(f"Expected 358 matched validation targets; found {len(frame)}")

    baseline = frame["spatial"].to_numpy(float)
    truth = frame["truth"].to_numpy(float)
    augmented = frame["augment_025"].to_numpy(float)
    baseline_mae = float(np.mean(np.abs(truth - baseline)))
    if baseline_mae <= 0:
        raise SystemExit("Invalid spatial baseline MAE")

    blend_weights = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    candidates = []
    summary_rows = []
    group_rows = []
    block_rows = []

    for weight in blend_weights:
        name = "spatial" if weight == 0 else f"shrink_{weight:.2f}"
        pred = baseline + weight * (augmented - baseline)
        frame[name] = pred
        error = np.abs(truth - pred)
        rec_mae = {}
        rec_gain = {}
        for recording, indexes in frame.groupby("recording").groups.items():
            idx = np.asarray(list(indexes), dtype=int)
            rec_error = float(error[idx].mean())
            rec_base = float(np.abs(truth[idx] - baseline[idx]).mean())
            rec_mae[str(recording)] = rec_error
            rec_gain[str(recording)] = (
                100.0 * (rec_base - rec_error) / rec_base if rec_base > 0 else 0.0
            )

        summary_rows.append({
            "candidate": name,
            "blend_weight": weight,
            "rows": len(frame),
            "mae": float(error.mean()),
            "gain_vs_spatial_pct": 100.0 * (baseline_mae - float(error.mean())) / baseline_mae,
            "improves_both_recordings": bool(all(v > 0 for v in rec_gain.values())),
            **{f"gain_{k}_pct": v for k, v in rec_gain.items()},
            **{f"mae_{k}": v for k, v in rec_mae.items()},
        })

        candidates.append(name)
        for (recording, unit), indexes in frame.groupby(["recording", "unit"]).groups.items():
            idx = np.asarray(list(indexes), dtype=int)
            base_err = np.abs(truth[idx] - baseline[idx]).mean()
            cand_err = error[idx].mean()
            group_rows.append({
                "candidate": name,
                "blend_weight": weight,
                "recording": recording,
                "unit": unit,
                "rows": len(idx),
                "mae": float(cand_err),
                "gain_vs_spatial_pct": (
                    100.0 * (base_err - cand_err) / base_err if base_err > 0 else 0.0
                ),
            })

        block = frame[["recording", "minute_of_day"]].copy()
        block["block"] = (block["minute_of_day"] // 180).astype(int)
        block["base_error"] = np.abs(truth - baseline)
        block["candidate_error"] = error
        for (recording, block_id), group in block.groupby(["recording", "block"]):
            block_rows.append({
                "candidate": name,
                "blend_weight": weight,
                "recording": recording,
                "block": int(block_id),
                "rows": len(group),
                "base_mae": float(group["base_error"].mean()),
                "candidate_mae": float(group["candidate_error"].mean()),
            })

    summary = pd.DataFrame(summary_rows).sort_values("mae")
    groups = pd.DataFrame(group_rows)
    blocks = pd.DataFrame(block_rows)
    block_win = (
        blocks.assign(win=blocks["candidate_mae"].le(blocks["base_mae"]))
        .groupby("candidate")["win"].mean()
        .mul(100.0)
    )
    summary["block_win_pct"] = summary["candidate"].map(block_win)
    strict = summary.loc[
        summary["improves_both_recordings"]
        & summary["gain_vs_spatial_pct"].ge(1.0)
    ]
    if not strict.empty:
        selected = strict.sort_values("mae").iloc[0]
        status = f"PROMISING_BOTH_RECORDINGS: {selected['candidate']}"
    else:
        selected = summary.iloc[0]
        status = "NO_BLEND_PASSES_BOTH_RECORDINGS"

    REPORTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PREDICTIONS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    groups.to_csv(OUT_GROUPS, index=False)
    blocks.to_csv(OUT_BLOCKS, index=False)

    print(f"Matched validation targets: {len(frame)}")
    print(f"Spatial baseline MAE: {baseline_mae:.5f}")
    print("\n=== SHRUNK V25 BLENDS ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== CAMERA RESULTS FOR BEST POOLED CANDIDATE ===")
    print(groups.loc[groups["candidate"].eq(summary.iloc[0]["candidate"])]
          .round(5).to_string(index=False))
    print("\n=== STATUS ===")
    print(status)
    print("\nSaved local diagnostics:", OUT_PREDICTIONS, OUT_SUMMARY, OUT_GROUPS, OUT_BLOCKS)
    print("LOCAL VALIDATION ONLY — NO SUBMISSION CREATED OR UPLOADED")


if __name__ == "__main__":
    main()
