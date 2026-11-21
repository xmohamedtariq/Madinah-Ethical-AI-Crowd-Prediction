"""V11: immutable, label-aware *diagnostic* of V10 forecast candidates.

No retraining, policy mutation, Kaggle file generation, or Kaggle upload.
Uses real saved local CV predictions and forbids treating the reused later day
as a clean holdout. The optional oracle uses target labels and is NOT deployable.

PowerShell:
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v11_candidate_forensics.py --mode audit
  .\\.venv-ag\\Scripts\\python.exe .\\madinah_autogluon_v11_candidate_forensics.py --mode report
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["recording", "day", "unit", "minute_of_day", "kind"]
FOCUS = ("cam_E", "cam_A", "cam_F")
REQUIRED = set(KEYS + ["people", "v7_accepted", "v10_proposed", "v10_accepted"])


def mae(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if not len(y):
        return math.nan
    return float(np.mean(np.abs(y-p)))


def pct(before, after):
    if not np.isfinite(before) or before <= 0 or not np.isfinite(after):
        return math.nan
    return 100.0*(before-after)/before


def json_number(x):
    return None if x is None or not np.isfinite(x) else float(x)


def check_table(df, name, need):
    missing = set(need)-set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if df.empty or df.duplicated(KEYS).any():
        raise ValueError(f"{name}: empty or duplicate key rows")
    for key in ("people", "v7_accepted"):
        vals = pd.to_numeric(df[key], errors="coerce").to_numpy(float)
        if (not np.isfinite(vals).all()) or np.any(vals < 0):
            raise ValueError(f"{name}: invalid {key}")
    return df


def load_inputs(reports):
    s = reports/"ag_v10_run_status.json"
    summary_file = reports/"ag_v10_summary.json"
    v10_file = reports/"ag_v10_cv_predictions.csv"
    v7_file = reports/"ag_v7_cv_predictions.csv"
    v7_sum_file = reports/"ag_v7_summary.json"
    for p in (s, summary_file, v10_file, v7_file, v7_sum_file):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}; V11 requires a successful V10 run and V7 checkpoint")
    status = json.loads(s.read_text(encoding="utf-8"))
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    v7_sum = json.loads(v7_sum_file.read_text(encoding="utf-8"))
    if status.get("state") != "SUCCESS" or status.get("mode") != "validate":
        raise ValueError("V10 run status is not SUCCESS validation; stale files cannot be analyzed")
    if status.get("result_file") != summary_file.name or summary.get("parameters", {}).get("mode") != "validate":
        raise ValueError("V10 status/summary mismatch: not a completed validation")
    df = check_table(pd.read_csv(v10_file, low_memory=False), "V10", REQUIRED)
    ref = check_table(pd.read_csv(v7_file, low_memory=False), "V7", REQUIRED - {"v10_proposed", "v10_accepted"})
    df["day"] = df.day.astype(str)
    ref["day"] = ref.day.astype(str)
    # The reference is locked to the stored V7 file by exact target keys and labels.
    aligned = df[KEYS + ["people", "v7_accepted"]].merge(
        ref[KEYS + ["people", "v7_accepted"]], on=KEYS, how="outer",
        validate="one_to_one", indicator=True, suffixes=("_v10", "_v7"), sort=False)
    if len(df) != len(ref) or not aligned["_merge"].eq("both").all():
        raise ValueError("V10 and V7 target sets differ")
    for field in ("people", "v7_accepted"):
        if not np.allclose(aligned[f"{field}_v10"], aligned[f"{field}_v7"], atol=1e-7, rtol=0):
            raise ValueError(f"V10 and V7 {field} values differ")
    if int(summary.get("n_rows", -1)) != len(df):
        raise ValueError("V10 summary row count differs")
    base_error = mae(df.people, df.v7_accepted)
    v7_mae = float(v7_sum.get("accepted_mae", math.nan))
    v10_mae = float(summary.get("reused_combined", {}).get("v7_mae", math.nan))
    if not np.isclose(base_error, v7_mae, atol=0.0002, rtol=0):
        raise ValueError(f"V7 baseline MAE mismatch: {base_error:.6f} != {v7_mae:.6f}")
    if not np.isclose(base_error, v10_mae, atol=0.0002, rtol=0):
        raise ValueError("V10 summary baseline MAE mismatch")
    candidates = sorted(c for c in df if c.startswith("candidate_"))
    if not candidates:
        raise ValueError("No V10 candidate prediction columns")
    for col in candidates + ["v10_proposed", "v10_accepted"]:
        a = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"Invalid/non-finite V10 candidate: {col}")
    days = sorted(df.day.astype(str).unique().tolist())
    if days != summary.get("days") or len(days) != 2:
        raise ValueError("V10 day order differs from summary")
    return df, summary, status, days, candidates


def block_stats(group, candidate, *, block_minutes=20):
    if group.empty:
        return {"blocks": 0, "winning_blocks": 0, "block_win_rate": None,
                "mean_block_gain_people": None}
    delta = np.abs(group.people.to_numpy(float)-group.v7_accepted.to_numpy(float)) - \
            np.abs(group.people.to_numpy(float)-group[candidate].to_numpy(float))
    block = np.floor(group.minute_of_day.to_numpy(float)/block_minutes).astype(int)
    diffs = [float(delta[block == b].mean()) for b in np.unique(block)
             if np.sum(block == b) >= 3]
    wins = sum(d > 0 for d in diffs)
    return {"blocks": len(diffs), "winning_blocks": wins,
            "block_win_rate": (wins/len(diffs) if diffs else None),
            "mean_block_gain_people": (float(np.mean(diffs)) if diffs else None)}


def eligible_first_gate(df, days, candidates, *, min_rows=60, min_gain=6.,
                        min_block_wins=4, win_ratio=.60, block_minutes=20):
    """Repair V10 decision ordering: filter eligible choices before ranking.

    V10 sorted by maximal total gain, then checked the winning block ratio
    of ONLY the top candidate. A runner-up could be stable across many more
    blocks but never be examined. Here all candidates and fixed alphas are
    tested against the SAME early day, and only eligible ones are ranked.

    Later day remains a REUSED shadow diagnostic. Never use its labels to
    select the winning candidate; never report it as unbiased validation.
    """
    base = df.v7_accepted.to_numpy(float)
    y = df.people.to_numpy(float)
    new = base.copy()
    cameras = df.unit.astype(str).to_numpy()
    kinds = df.kind.astype(str).to_numpy()
    d = df.day.astype(str).to_numpy()
    timeline = df.minute_of_day.to_numpy(float)
    decisions = []
    for cam in FOCUS:
        mask = (cameras == cam) & (kinds == 'forecast')
        early = mask & (d == days[0])
        item = {'camera': cam, 'n_early': int(early.sum()),
                'n_later': int(np.sum(mask & (d == days[1]))),
                'selected': 'v7_accepted', 'alpha': 0.,
                'eligible_candidates': 0}
        if int(early.sum()) < min_rows:
            item['reason'] = 'too_few_early_rows'
            decisions.append(item)
            continue
        truth = y[early]
        before = base[early]
        old_abs = np.abs(truth-before)
        er = float(old_abs.mean())
        b = np.floor(timeline[early]/block_minutes).astype(int)
        eligible = []
        all_choices = []
        for name in candidates:
            alt = df[name].to_numpy(float)
            for alpha in (.25, .50, .75, 1.):
                predicted = np.clip((1.-alpha)*before+alpha*alt[early],0.,100000.)
                improvement = old_abs-np.abs(truth-predicted)
                g = pct(er, mae(truth, predicted))
                block_diffs = [float(np.mean(improvement[b==bb]))
                               for bb in np.unique(b) if int(np.sum(b==bb))>=3]
                wins = sum(q>0 for q in block_diffs)
                nblocks = len(block_diffs)
                record = {'candidate': name, 'alpha': alpha, 'early_gain_pct': g,
                          'winning_blocks': wins, 'blocks': nblocks,
                          'block_win_ratio': wins/max(1,nblocks)}
                all_choices.append(record)
                if (g>=min_gain and nblocks>=4 and wins>=min_block_wins
                        and wins/max(1,nblocks)>=win_ratio):
                    eligible.append(record)
        all_choices.sort(key=lambda o:(o['early_gain_pct'],o['winning_blocks']), reverse=True)
        eligible.sort(key=lambda o:(o['early_gain_pct'],o['winning_blocks']), reverse=True)
        item['eligible_candidates']=len(eligible)
        if all_choices:
            item['top_unfiltered']=all_choices[0]
        if eligible:
            best=eligible[0]
            alt=df[best['candidate']].to_numpy(float)
            new[mask]=np.clip((1-best['alpha'])*base[mask]+best['alpha']*alt[mask],0.,100000.)
            item.update({'selected':best['candidate'],'alpha':best['alpha'],
                         'early_gain_pct':best['early_gain_pct'],
                         'winning_blocks':best['winning_blocks'],
                         'blocks':best['blocks']})
        else:
            item['reason']='no_candidate_passed_all_early_gates'
        decisions.append(item)
    late = d == days[1]
    later_gain = pct(mae(y[late],base[late]), mae(y[late],new[late]))
    severe = []
    for cam in FOCUS:
        m = late & (cameras==cam) & (kinds=='forecast')
        if int(m.sum()) < 40:
            continue
        old=mae(y[m],base[m]); adjusted=mae(y[m],new[m])
        if adjusted > 1.15*old:
            severe.append({'camera':cam,'n':int(m.sum()),'old_mae':old,'proposed_mae':adjusted})
    outcome={
        'policy': 'eligible-first repair of V10 gate; frozen from early day only',
        'decisions': decisions,
        'early_v7_mae': mae(y[d==days[0]],base[d==days[0]]),
        'early_v11_mae': mae(y[d==days[0]],new[d==days[0]]),
        'later_v7_mae_REUSED': mae(y[late],base[late]),
        'later_v11_mae_REUSED': mae(y[late],new[late]),
        'later_gain_pct_REUSED': later_gain,
        'combined_v11_mae_REUSED': mae(y,new),
        'severe_later_regressions_REUSED': severe,
        'shadow_diagnostic_only': True,
        'note': 'No final model accepted; later performance cannot be used as unbiased model selection.',
    }
    return new, outcome


def analyze(df, days, candidates, summary):
    early_day, late_day = days
    overall = mae(df.people, df.v7_accepted)
    abs_error = (df.people.to_numpy(float)-df.v7_accepted.to_numpy(float))
    focus = df.kind.astype(str).eq("forecast") & df.unit.astype(str).isin(FOCUS)
    focus_error_share = float(100 * np.abs(abs_error[focus.to_numpy()]).sum() /
                              max(np.abs(abs_error).sum(), 1e-12))
    target55 = overall * .45
    required_focus_reduction = 55.0 / (focus_error_share/100.0) if focus_error_share > 0 else math.inf
    rows = []
    for cam in FOCUS:
        sub = df.loc[df.kind.eq("forecast") & df.unit.eq(cam)]
        ea = sub.loc[sub.day.eq(early_day)]
        la = sub.loc[sub.day.eq(late_day)]
        for name in candidates:
            er = mae(ea.people, ea.v7_accepted)
            en = mae(ea.people, ea[name])
            lr = mae(la.people, la.v7_accepted)
            ln = mae(la.people, la[name])
            change_early = np.abs(ea[name].to_numpy(float) - ea.v7_accepted.to_numpy(float)) > 1e-7
            change_later = np.abs(la[name].to_numpy(float) - la.v7_accepted.to_numpy(float)) > 1e-7
            b = block_stats(ea, name)
            rows.append({
                "camera": cam, "candidate": name,
                "n_early": len(ea), "n_later": len(la),
                "changed_early": int(change_early.sum()),
                "changed_later": int(change_later.sum()),
                "coverage_early_pct": float(change_early.mean()*100) if len(ea) else 0.,
                "coverage_later_pct": float(change_later.mean()*100) if len(la) else 0.,
                "early_v7_mae": er, "early_candidate_mae": en, "early_gain_pct": pct(er, en),
                "later_v7_mae": lr, "later_candidate_mae": ln, "later_gain_pct": pct(lr, ln),
                "all_v7_mae": mae(sub.people, sub.v7_accepted),
                "all_candidate_mae": mae(sub.people, sub[name]),
                **b,
            })
    rankings = pd.DataFrame(rows)
    rankings.sort_values(["camera", "early_gain_pct", "block_win_rate"],
                         ascending=[True, False, False], inplace=True, na_position="last")
    rankings.reset_index(drop=True, inplace=True)
    # This nondeployable oracle looks at true labels to choose per row; it is
    # a *diagnostic ceiling*, NEVER a model selection/prediction or valid CV.
    focus_df = df.loc[focus]
    preds = focus_df[["v7_accepted"] + candidates].to_numpy(float)
    truths = focus_df.people.to_numpy(float)
    oracle_mae = float(np.min(np.abs(preds-truths[:, None]), axis=1).mean()) if len(focus_df) else None
    focus_v7_mae = mae(focus_df.people, focus_df.v7_accepted)
    oracle_gain = pct(focus_v7_mae, oracle_mae) if oracle_mae is not None else None
    frozen = summary.get("early_frozen_policy", [])
    notes = []
    for item in frozen:
        cam = item.get("camera", "")
        sub = rankings.loc[rankings.camera.eq(cam)]
        if sub.empty:
            continue
        best = sub.iloc[0]
        notes.append({
            "camera": cam,
            "existing_frozen_selected": item.get("selected"),
            "existing_alpha": item.get("alpha"),
            "strongest_early_candidate": str(best.candidate),
            "strongest_early_gain_pct": json_number(best.early_gain_pct),
            "strongest_early_block_win_rate": json_number(best.block_win_rate),
            "strongest_early_later_gain_pct_DIAGNOSTIC_ONLY": json_number(best.later_gain_pct),
            "strongest_early_change_coverage_pct": json_number(best.coverage_early_pct),
        })
    _, corrected_gate = eligible_first_gate(df, days, candidates)
    result = {
        "eligible_first_v10_policy_repair": corrected_gate,
        "version": "V11 candidate feasibility audit (no training or submission)",
        "source": "V10 SUCCESS validation candidates + locked V7 checkpoint",
        "days": days,
        "rows": int(len(df)),
        "v7_mae_people_reused": overall,
        "v10_existing_verdict": summary.get("verdict"),
        "v10_proposed_mae_reused": mae(df.people, df.v10_proposed),
        "target_55pct_from_v7_mae": target55,
        "forecast_focus_absolute_error_share_pct": focus_error_share,
        "required_reduction_of_focus_error_if_everything_else_is_unchanged_pct": required_focus_reduction,
        "candidate_columns": candidates,
        "early_frozen_policy_diagnostics": notes,
        "nondeployable_row_oracle": {
            "warning": "USES VALIDATION LABELS PER ROW; NOT A FORECAST, NOT CV, NOT A SUBMISSION",
            "focus_rows": int(len(focus_df)),
            "v7_focus_mae": focus_v7_mae,
            "best_possible_with_label_aware_selection_mae": oracle_mae,
            "label_oracle_focus_gain_pct": oracle_gain,
        },
        "limitations": [
            "Sep 11 was used for prior V7 policy selection and Sep 12 reused repeatedly",
            "Sep 11 (Friday) and Sep 12 (Saturday) are different regimes",
            "The CSV excludes 644 new-camera target IDs cam_I through cam_M",
            "Does not compute the official Kaggle custom metric",
            "No labels are read from official hidden test IDs",
        ],
        "submission_created": False,
        "models_trained": False,
    }
    return result, rankings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("audit", "report"), default="report")
    p.add_argument("--reports", type=Path, default=Path(__file__).resolve().parent/"reports")
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args()
    df, summary, status, days, candidates = load_inputs(args.reports)
    data, rankings = analyze(df, days, candidates, summary)
    print("=== V11 FROZEN-CANDIDATE FORENSICS ===")
    print(f"Run status: {status['state']} | V10 verdict: {data['v10_existing_verdict']}")
    print(f"V7 baseline: {data['v7_mae_people_reused']:.6f} MAE on {len(df)} reused CV rows")
    print(f"55% target from V7: {data['target_55pct_from_v7_mae']:.6f} MAE")
    print(f"Focus cameras: {data['forecast_focus_absolute_error_share_pct']:.2f}% of absolute error")
    print(f"Required focus reduction if other errors unchanged: {data['required_reduction_of_focus_error_if_everything_else_is_unchanged_pct']:.1f}%")
    for cam in FOCUS:
        s = rankings.loc[rankings.camera.eq(cam) & rankings.n_early.gt(0)].head(args.top)
        print(f"\n{cam}: best candidates sorted ONLY by EARLY (reused) performance")
        for r in s.itertuples(index=False):
            later = "N/A" if not np.isfinite(r.later_gain_pct) else f"{r.later_gain_pct:+.2f}%"
            print(f"  {r.candidate:31s} early={r.early_gain_pct:+7.2f}% "
                  f"later_diagnostic={later:>9} "
                  f"blocks={r.winning_blocks}/{r.blocks} "
                  f"changed_early={r.changed_early}/{r.n_early}")
    repaired = data['eligible_first_v10_policy_repair']
    print('\nV11 eligible-first gate (early frozen; later REUSED):')
    for item in repaired['decisions']:
        print(f"  {item['camera']}: {item['selected']} alpha={item['alpha']:.2f} "
              f"eligible={item['eligible_candidates']}")
    print(f"  Early V7={repaired['early_v7_mae']:.4f} V11={repaired['early_v11_mae']:.4f}")
    print(f"  Later V7={repaired['later_v7_mae_REUSED']:.4f} V11={repaired['later_v11_mae_REUSED']:.4f} "
          f"(shadow gain={repaired['later_gain_pct_REUSED']:+.2f}%)")
    print("\nWARNING: all comparison days reused; do NOT use later diagnostics for model selection.")
    if args.mode == "report":
        args.reports.mkdir(parents=True, exist_ok=True)
        out = args.reports/"ag_v11_candidate_feasibility.json"
        out.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        rankings.to_csv(args.reports/"ag_v11_candidate_rankings.csv", index=False)
        print("Saved", out.name, "and ag_v11_candidate_rankings.csv")
    else:
        print("Audit only; no files written")
    print("NO training. NO Kaggle submission. No score of 0.400 verified.")


if __name__ == "__main__":
    main()
