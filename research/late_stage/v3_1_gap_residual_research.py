#!/usr/bin/env python3
"""V3.1 non-submission research: small, label-free Missing gap corrections.

Run in Madinah-AI-Fresh:
    python v3_1_gap_residual_research.py --root .

Input: existing breakthrough_v2 diagnostic CSVs and optional phase0 audit.
Outputs: aggregate diagnostics + source interface probe, never predictions or a submission.
Caveat: the two frozen recordings are reused development folds, NOT independent evidence.
Requires: numpy, pandas. Python 3.10+.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CHAMPION_SCORE = 0.6057000521183546
TARGET_SCORE = 0.6
EXPECTED_MISSING_ROWS = 358
RECORDINGS = ("20260911_1100", "20260912_1100")
CAMERAS = ("cam_A", "cam_E", "cam_F")
REQUIRED = {
    "id", "recording", "camera", "minute_of_day", "block_start",
    "block_length", "gap_position", "gap_position_fraction",
    "left_count", "right_count", "endpoint_slope_per_min", "flow_pattern",
    "local_prev_mean", "local_next_mean", "truth", "selected_prediction",
    "abs_error", "normalized_full_score_contribution", "task_reference_mae",
}
# Gold is used solely AFTER prediction is determined. make_predictions() has no labels.
LEGAL_FEATURES = (
    "recording", "camera", "minute_of_day", "block_start", "block_length",
    "gap_position", "gap_position_fraction", "left_count", "right_count",
    "endpoint_slope_per_min", "flow_pattern", "local_prev_mean",
    "local_next_mean", "selected_prediction",
)
RESEARCH_LABEL = "REUSED_FROZEN_DEVELOPMENT_ONLY"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def safe_number(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def inspect_source_functions(root: Path) -> dict:
    paths = (
        "src/frozen_protocol.py", "src/evaluate_full_solution.py",
        "src/final_missing_hidden_research/best_missing_model.py",
        "src/final_missing_hidden_research/independent_missing_loro.py",
        "src/final_missing_hidden_research/diagnose_missing.py",
        "src/final_missing_hidden_research/run_missing_rts.py",
    )
    findings = {}
    for rel in paths:
        path = root / rel
        info = {"exists": path.is_file()}
        if not path.is_file():
            findings[rel] = info
            continue
        info["sha256"] = digest(path)
        try:
            module = ast.parse(path.read_text(encoding="utf-8-sig"))
            functions = []
            for item in module.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    positional = item.args.posonlyargs + item.args.args
                    args = [a.arg for a in positional]
                    if item.args.vararg:
                        args.append("*" + item.args.vararg.arg)
                    args.extend(a.arg for a in item.args.kwonlyargs)
                    if item.args.kwarg:
                        args.append("**" + item.args.kwarg.arg)
                    functions.append({"name": item.name, "parameters": args})
            info["functions"] = functions
        except Exception as exc:
            info["parse_error"] = f"{type(exc).__name__}: {exc}"
        findings[rel] = info
    for rel in ("data/raw/train.csv", "data/raw/test.csv"):
        path = root / rel
        info = {"exists": path.is_file()}
        if path.is_file():
            info["size_bytes"] = path.stat().st_size
            try:
                info["columns"] = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception as exc:
                info["parse_error"] = str(exc)
        findings[rel] = info
    return findings


def make_predictions(feature_frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fixed hypotheses only. NO held-out labels, errors, reference values, or test-data truth."""
    # Copy down only an explicit allowlist, ensuring no accidental label access.
    x = feature_frame.loc[:, LEGAL_FEATURES]
    n = len(x)
    p = x["selected_prediction"].to_numpy(float)
    left = x["left_count"].to_numpy(float)
    right = x["right_count"].to_numpy(float)
    before_mean = x["local_prev_mean"].to_numpy(float)
    after_mean = x["local_next_mean"].to_numpy(float)
    t = x["gap_position_fraction"].to_numpy(float)
    D = x["block_length"].to_numpy(float) + 1.0
    secant = (right-left) / D

    # Conservative gate: endpoints and camera are known from released context.
    # Neither target label nor error statistics are used for prediction.
    gate = (
        x["camera"].isin(CAMERAS).to_numpy()
        & x["flow_pattern"].eq("falling_fast").to_numpy()
        & (x["endpoint_slope_per_min"].to_numpy(float) <= -2.0)
        & (x["block_length"].to_numpy(float) >= 5.0)
        & (x["block_length"].to_numpy(float) <= 15.0)
        & np.isfinite(p) & np.isfinite(before_mean) & np.isfinite(after_mean)
        & np.isfinite(left) & np.isfinite(right)
        & (t > 0.0) & (t < 1.0)
    )

    # 3 predeclared hypotheses; all bounded and all preserve non-gated champion rows.
    # Local means approximate slopes over approximately two minutes; this is
    # a heuristic to test, not a statement about the actual data-generating model.
    m_left = (left-before_mean) / 2.0
    m_right = (after_mean-right) / 2.0
    curvature = D * ((m_left-secant)*t*(1.0-t)**2 - (m_right-secant)*t*t*(1.0-t))
    linear = left + (right-left)*t
    bridge_delta = linear - p
    allowed_adjustments = {
        "V3_1A_curvature_10pct_cap3": np.clip(0.10*curvature, -3.0, 3.0),
        "V3_1B_linear_pull_5pct_cap3": np.clip(0.05*bridge_delta, -3.0, 3.0),
        "V3_1C_curvature_7p5_bridge_2p5_cap3": np.clip(
            0.075*curvature + 0.025*bridge_delta, -3.0, 3.0
        ),
    }
    output = {"verified_champion_unchanged": p.copy()}
    for name, adjustment in allowed_adjustments.items():
        q = p.copy()
        q[gate] = np.maximum(0.0, p[gate] + adjustment[gate])
        if len(q) != n or not np.isfinite(q).all():
            raise AssertionError(f"Invalid prediction shape or nonfinite values: {name}")
        if not np.array_equal(p[~gate], q[~gate]):
            raise AssertionError(f"Ungated predictions changed: {name}")
        output[name] = q
    return output, gate


def run(root: Path, input_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    input_dir = input_dir or (root / "reports" / "breakthrough_v2")
    output_dir = output_dir or (root / "reports" / "v3_1")
    if "submission" in output_dir.as_posix().lower():
        raise ValueError("Refusing to write anywhere under a submission directory")
    for name in ("missing_target_diagnostics.csv", "per_task_score_comparisons.csv"):
        if not (input_dir / name).is_file():
            raise FileNotFoundError(f"Required input missing: {input_dir / name}")
    target_path = input_dir / "missing_target_diagnostics.csv"
    task_path = input_dir / "per_task_score_comparisons.csv"
    audit = root / "reports" / "v3_phase0" / "schema_audit.json"
    if audit.is_file():
        recorded = json.loads(audit.read_text(encoding="utf-8-sig"))
        for label, path in (("missing_targets", target_path), ("task_comparisons", task_path)):
            old = recorded.get("files", {}).get(label, {}).get("sha256")
            if old and digest(path) != old:
                raise ValueError(f"SHA256 mismatch with Phase0 audit: {path}")

    df = pd.read_csv(target_path)
    comp = pd.read_csv(task_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_MISSING_ROWS or df.id.duplicated().any():
        raise AssertionError("Expected 358 unique frozen Missing targets")
    if set(df.recording.unique()) != set(RECORDINGS):
        raise AssertionError("Unexpected recordings; stop and re-audit")
    if df[list(("truth", "selected_prediction", "task_reference_mae"))].isna().any().any():
        raise AssertionError("Missing numeric values in core columns")
    if (df.task_reference_mae <= 0).any():
        raise AssertionError("Reference MAE must be positive")

    gold = df.truth.to_numpy(dtype=float)  # ONLY consumed below, after make_predictions()
    features = df[list(LEGAL_FEATURES)].copy()
    predictions, gate = make_predictions(features)

    # Gold-free prediction invariance test: adding random label columns to
    # the feature frame must not alter any candidate result.
    bait = features.copy()
    bait["truth"] = np.linspace(-1e9, 1e9, len(bait))
    bait["abs_error"] = np.arange(len(bait))*23
    bait_pred, bait_gate = make_predictions(bait)
    if not np.array_equal(gate, bait_gate):
        raise AssertionError("Label invariance violation in gate")
    for key in predictions:
        if not np.array_equal(predictions[key], bait_pred[key]):
            raise AssertionError(f"Label invariance violation for {key}")

    reference_rows = comp.loc[comp.model.eq("verified_champion")].copy()
    if len(reference_rows) != 8 or reference_rows.duplicated(["recording", "task"]).any():
        raise AssertionError("Champion must have exactly eight task/recording components")
    baseline_score = float(reference_rows.ratio.sum()/8.0)
    if abs(baseline_score - CHAMPION_SCORE) > 1e-12:
        raise AssertionError(f"Frozen champion is not exact: {baseline_score} != {CHAMPION_SCORE}")

    per_recording = df.groupby("recording").size().to_dict()
    scale = np.asarray([1.0/(8*per_recording[rec]*r) for rec,r in
        zip(df.recording, df.task_reference_mae)], dtype=float)
    expected_contribution = scale * np.abs(gold-predictions["verified_champion_unchanged"])
    if np.max(np.abs(expected_contribution-df.normalized_full_score_contribution.to_numpy(float))) > 1e-10:
        raise AssertionError("Per-target contribution does not match frozen score formula")

    groups = []
    summary = []
    p0 = predictions["verified_champion_unchanged"]
    for name, p1 in predictions.items():
        contribution_delta = scale*(np.abs(gold-p1)-np.abs(gold-p0))
        by_rec = {rec: float(np.sum(contribution_delta[df.recording.eq(rec).to_numpy()]))
                  for rec in RECORDINGS}
        score = float(baseline_score+np.sum(contribution_delta))
        changed = (p1 != p0)
        per = {
            "model": name,
            "full_score_reused_frozen": score,
            "delta_vs_champion_negative_is_better": float(score-baseline_score),
            "modified_rows": int(changed.sum()),
            "eligible_gated_rows": int(gate.sum()),
            "modified_blocks": int(df.loc[changed,["recording","camera","block_start"]].drop_duplicates().shape[0]),
            "mean_abs_change_on_modified": float(np.abs(p1[changed]-p0[changed]).mean()) if changed.any() else 0.0,
            "recording_full_score_deltas": by_rec,
            "would_cross_0_600_on_reused_folds_only": bool(score<TARGET_SCORE),
            "accepted_into_champion": False,
            "acceptance_status": ("CHAMPION_PRESERVED" if name=="verified_champion_unchanged"
                                  else "INDEPENDENT_VALIDATION_UNAVAILABLE" if all(v<=1e-14 for v in by_rec.values())
                                  else "REJECT_REUSED_OR_UNSTABLE")
        }
        summary.append(per)
        group_defs = {
            "recording": ["recording"],
            "camera": ["camera"],
            "recording_camera": ["recording","camera"],
            "gap_length": ["block_length"],
            "flow_pattern": ["flow_pattern"],
            "recording_flow_pattern": ["recording","flow_pattern"],
        }
        for dim, keys in group_defs.items():
            for keyvals, g in df.groupby(keys, sort=True):
                if not isinstance(keyvals, tuple):
                    keyvals = (keyvals,)
                ix = g.index.to_numpy(int)
                groups.append({
                    "model": name,
                    "dimension": dim,
                    "value": "/".join(map(str, keyvals)),
                    "target_rows": len(ix),
                    "eligible_rows": int(gate[ix].sum()),
                    "changed_rows": int(changed[ix].sum()),
                    "champion_mae": float(np.abs(gold[ix]-p0[ix]).mean()),
                    "candidate_mae": float(np.abs(gold[ix]-p1[ix]).mean()),
                    "full_score_delta_negative_is_better": float(contribution_delta[ix].sum()),
                })

    # Detect prior source-report integrity mismatch; do not misattribute old score
    # comparisons to Missing-only predictions if Forecast was changed there.
    discrepancy = []
    for candidate in sorted(set(comp.model)-{"verified_champion"}):
        for rec in RECORDINGS:
            sub = comp.loc[(comp.model.eq(candidate)) & comp.recording.eq(rec) & comp.task.eq("forecast")]
            champ = reference_rows.loc[reference_rows.recording.eq(rec)&reference_rows.task.eq("forecast")]
            if len(sub) == len(champ) == 1:
                d = float(sub.mae.iloc[0]-champ.mae.iloc[0])
                if abs(d)>1e-10:
                    discrepancy.append({"candidate":candidate,"recording":rec,"forecast_mae_delta":d})
    result = {
        "experiment": "V3_1_GAP_RESIDUAL_REPLAY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": RESEARCH_LABEL,
        "champion_reconstructed_exact_score": baseline_score,
        "official_kaggle_best_unchanged_from_user_report": 0.729,
        "full_task_score_reconstructed": True,
        "task_components": 8,
        "missing_target_count": len(df),
        "missing_gap_count": int(df.groupby(["recording","camera","block_start"]).ngroups),
        "fast_falling_count": int(df.flow_pattern.eq("falling_fast").sum()),
        "fast_falling_AEF_count": int((df.flow_pattern.eq("falling_fast")&df.camera.isin(CAMERAS)).sum()),
        "fast_falling_AEF_score_contribution": float(df.loc[df.flow_pattern.eq("falling_fast")&df.camera.isin(CAMERAS),"normalized_full_score_contribution"].sum()),
        "all_missing_score_contribution": float(df.normalized_full_score_contribution.sum()),
        "label_invariance": "PASS",
        "candidate_specs_exploratory_reused_folds": [
            "A: 10% curvature from adjacent released mean slopes, capped +/-3 persons",
            "B: 5% pull towards endpoint interpolation, capped +/-3 persons",
            "C: 7.5% curvature + 2.5% endpoint pull, capped +/-3 persons",
            "All: only A/E/F, released endpoint slope <= -2, gap length 5-15, pattern falling_fast",
        ],
        "candidates": summary,
        "prior_v2_forecast_changed_despite_narrative": discrepancy,
        "production_integration_allowed": False,
        "submission_generated": False,
        "limitations": [
            "Reused 20260911 and 20260912 development folds; not blind or independent",
            "Full train/test, complete champion callback and frozen evaluator not included in diagnostic bundle",
            "Coefficient family and scope were informed by inspected development folds; no independent model selection claim",
            "The provided diagnostics hold labels; prediction function accesses only listed released-context features and the champion prediction",
            "The claim that local_prev_mean/local_next_mean use released context must be checked against diagnose_missing.py before integration",
            "Older four recording holdouts were already used for RTS research; cannot be called untouched",
            "Predictions not saved; score deltas calculated without changing any non-Missing prediction",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"v3_1_exploratory_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    pd.DataFrame(groups).to_csv(output_dir/"v3_1_group_comparisons.csv",index=False,float_format="%.12g")
    probe = inspect_source_functions(root)
    (output_dir/"v3_1_source_probe.json").write_text(json.dumps(probe,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    report = [
        "# V3.1 Missing correction — reused frozen-fold research ONLY",
        "",
        "**Decision: PRESERVE CHAMPION. No candidate promoted. No submission generated.**",
        f"- Reconstructed champion: `{baseline_score:.13f}` (exact match).",
        f"- Targets: {len(df)}, blocks: {result['missing_gap_count']}, fast-falling: {result['fast_falling_count']}.",
        f"- Gated A/E/F targets: {int(gate.sum())}.",
        "- No forecast/hidden/hidden-future predictions touched or created.",
        "",
        "| Candidate | Frozen score (not independent) | Full score change | Friday change | Saturday change | Changed rows |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for v in summary:
        report.append(f"| {v['model']} | {v['full_score_reused_frozen']:.9f} | {v['delta_vs_champion_negative_is_better']:+.9f} | {v['recording_full_score_deltas'][RECORDINGS[0]]:+.9f} | {v['recording_full_score_deltas'][RECORDINGS[1]]:+.9f} | {v['modified_rows']} |")
    report += [
        "",
        "These hypotheses are exploratory, not blind: the frozen diagnostics had already been inspected during research.",
        "These results were scored on repeatedly inspected validation folds, so they CANNOT justify replacing the champion.",
        "A candidate that helps Friday and hurts Saturday fails the consistency gate.",
        "",
        "## V2 report consistency alert",
    ]
    if discrepancy:
        report.append("The previous V2 task comparison contains nonzero Forecast MAE changes for supposed Missing/Hidden-only substitutions:")
        for row in discrepancy:
            report.append(f"- {row['candidate']} / {row['recording']}: Forecast MAE delta {row['forecast_mae_delta']:+.9f}.")
        report.append("Before any new integration, reconcile the original V2 baseline and re-check non-Missing exact identity.")
    else:
        report.append("No non-Missing mismatch found in the prior per-task table.")
    report += [
        "",
        "## Next integration gate",
        "Inspect `v3_1_source_probe.json`; obtain the real frozen callback/evaluator interface, and",
        "run pseudo-gap LOSO on truly independent recordings with frozen coefficients before",
        "any 0.600 claim. Previous four older holdouts were already used during RTS research.",
        "",
        "No standalone predictions, submission CSV or uploads were produced.",
    ]
    (output_dir/"v3_1_decision.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("V3.1 READ-ONLY CHAMPION; no submissions")
    print(f"Champion exact replay via components: {baseline_score:.13f}")
    print(f"Missing {len(df)} / gaps {result['missing_gap_count']} / eligible {int(gate.sum())}")
    for v in summary:
        print(f"{v['model']:<40s} score {v['full_score_reused_frozen']:.9f} "
              f"delta {v['delta_vs_champion_negative_is_better']:+.9f} "
              f"Fri {v['recording_full_score_deltas'][RECORDINGS[0]]:+.6f} "
              f"Sat {v['recording_full_score_deltas'][RECORDINGS[1]]:+.6f}")
    print(f"V2 baseline integrity warnings: {len(discrepancy)}")
    print(f"Reports saved to: {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    opts=parser.parse_args()
    root=opts.root.expanduser().resolve()
    inp=opts.input_dir.expanduser().resolve() if opts.input_dir else None
    out=opts.output_dir.expanduser().resolve() if opts.output_dir else None
    run(root, inp, out)


if __name__ == "__main__":
    main()
