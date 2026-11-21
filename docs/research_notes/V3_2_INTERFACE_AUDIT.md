# V3.2 interface and validation audit

## Source cross-check

**Exact champion (reported):** `0.6057000521183546`, rebuilt from the eight original frozen reference components in `per_task_score_comparisons.csv`.

**Incorrect older entry point:** `src/evaluate_full_solution.py` imports LightGBM, older model modules, and `create_synthetic_validation_split`, rather than `verified_605700/cloud_research/selected/candidate.py`. Its numeric results must not be presented as a replay of the selected champion.

**Relevant current holdout entry point:** `src/final_missing_hidden_research/independent_missing_loro.py` imports `predict_fold` from `verified_605700/cloud_research/selected/candidate.py`. This script is adapted by V3.2 without overwriting anything in the selected source.

**Missing reference helper:** `best_missing_model.py` is a standalone linear interpolation helper and is not a drop-in substitute for the champion callback.

## Main statistical limitations

- Four earlier recordings have `cam_A` / `cam_F`, but zero rows for `cam_E`.
- All four earlier recordings were already used in past RTS research. New pseudo-gap masks are predeclared, but recording-level independence from historic model development cannot be claimed.
- The 358 frozen Friday/Saturday Missing rows are repeatedly reused development folds. Even a retrospective score below `0.600` on them is **not proof of generalization**.
- New pseudo Missing gap blocks deliberately oversample released fast-declining windows. Their Missing ratio cannot be converted directly into the global 4-task score.
- The preview uses an exact champion-score reconstruction for existing archived rows and leaves the seven non-Missing task/fold components unchanged arithmetically; it **does not** rerun the full selected inference pipeline.

## Previously observed V2 metric discrepancies

In `per_task_score_comparisons.csv`, both `rts_missing_only` and `nearest_profile_hidden_only` have Forecast MAE different from the `verified_champion` baseline:

| Recording | Forecast MAE delta for each candidate versus Champion |
|---|---:|
| 20260911_1100 | +0.012952776821443024 |
| 20260912_1100 | -0.034480641551517976 |

The difference should be reconciled before an integration decision; nonzero differences do not alone establish leakage.

## Reproducibility checks

1. Run `v3_2_missing_loro.py` with the real selected callback.
2. Inspect `v3_2_outer_holdouts.csv` and group metrics, including whether **both** A and F generalize.
3. Inspect `v3_2_summary.json` for selected alpha per outer fold, reused-fold preview, camera E warning, baseline and candidate checksums.
4. Keep Champion and all existing official submissions untouched. The script produces only `reports/v3_2` research artifacts.
