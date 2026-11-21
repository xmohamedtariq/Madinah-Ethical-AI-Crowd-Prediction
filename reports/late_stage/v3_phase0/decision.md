# Missing-First Reconstruction Decision

## Decision

Keep the verified champion at local `0.6057000521183546` and official Kaggle best `0.729`. Reject both new candidates and produce no submission file. Forecast and Hidden-future predictions stayed exactly on the champion for both candidate integrations.

The Missing-only RTS integration scored `0.685879588695` on the exact frozen masks, versus `0.605700052118` for the champion. Missing MAE increased Friday from `5.533` to `8.289` and Saturday from `3.986` to `5.782`. This is a decisive regression.

After Missing failed, a whole-recording nearest-profile Hidden transfer was evaluated. It scored `0.660281417712`; Hidden MAE increased Friday from `3.646` to `4.816` and Saturday from `4.166` to `5.660`. It is also rejected. This was profile retrieval with peer level adaptation, not the previous phase-blending or XGBoost experiments.

## Missing error map

All `358` frozen Missing targets were joined to verified champion predictions and frozen gold only after prediction. They form `46` contiguous blocks. Group contributions use the exact local score weighting, including the fold's pooled Missing reference denominator and the outer `1/8` weight (four tasks, two recordings).

- Fast-falling endpoint patterns account for `147` rows, MAE `6.591`, and `0.095875` score contribution, about 56% of Missing's `0.169872` contribution. The classification is based only on the nearest released observations outside the gap.
- Highest camera contributions are cam_A `0.043281`, cam_E `0.040328`, and cam_F `0.029548`.
- Highest duration contributions are 8-minute gaps `0.034619`, 9-minute gaps `0.031507`, and 11-minute gaps `0.020742`.
- Gap positions 6, 2, and 3 contribute `0.023552`, `0.022020`, and `0.018387`; interior positions 6–8 are the largest position groupings.
- The nearest released own-camera neighbor is typically one minute from a target. The longest gaps and high-error cameras do not alone explain the errors; sharp declines dominate the aggregate contribution.

Detailed row diagnostics, including gap length/position, neighbor distances and values, endpoint slope, peer median, selected prediction error, and normalized score contribution, are in `missing_target_diagnostics.csv`. Group summaries are in `missing_error_groups.csv`.

## Prior experiments and failure evidence

The original Madinah V13 script is Forecast-only; its README states Missing remains unchanged. The more recent research kit evaluated 20 Missing variants, including signed cross-camera residuals, local kernels, gap-shape regressors, event bridges, and complementary blends. Its best `complementary_additive` composite scored `0.615828`, still above the selected `0.605700` champion; variants were tuned on the same two reused recordings. The add-half OU combination scored `0.613848` under the older baseline wrapper and did not run its causality check. Neither result establishes a Missing-only gain over the selected branch.

The earlier Ridge residual correction directly regressed on both frozen Missing ratios (`1.49782` Friday, `1.04411` Saturday). Endpoint smoothing reduces local extremes and the historical event/kernel variants condition on limited, weakly aligned peer shape; their gains do not transfer to the selected branch. The new RTS model selected process noise using pseudo-gaps in other training recordings, but its extrapolated local trend over-smoothed abrupt CCTV count changes: Missing ratios rose to `0.964` and `1.038`. The `prior_missing_experiments.csv` file lists all 20 saved variants and their exact reused-fold composite scores.

## Recording-held-out check

Four earlier recordings, excluded from the frozen champion folds, were held out one at a time. In each, 70–78 Missing rows across 6–8 contiguous 5–18 minute blocks were removed from released context; model fitting and RTS noise selection used other recordings only. The champion beat the RTS candidate on two recordings and lost on two. It is a small, Missing-only generalization check, not a full four-task score; the direction is mixed and provides no reproducible basis to integrate RTS.

The exact frozen comparison remains authoritative for the full local objective: champion `0.605700`, RTS Missing-only `0.685880`, Hidden profile-only `0.660281`. See `per_task_score_comparisons.csv`, `per_recording_results.csv`, and `experiment_comparison.csv`.

## Integrity

The frozen callback gives models only `train`, `released`, target metadata, and baseline predictions; it rejects `people_true` and `people_ref` in callback inputs. The Missing smoother uses two-sided released observations, permitted for Missing. Hidden profile transfer uses only released peer counts and historical training recordings. Sampled future-row mutation checks passed, though they check Forecast cutoffs and are not exhaustive Missing/Hidden leakage proofs. No held-out labels were used for model fitting, configuration selection, or prediction. Gold was used by the frozen evaluator only for scoring and by the post-hoc diagnostic script.

The frozen validation itself is recording-disjoint in each fold but is reused and repeatedly tuned development data. The four earlier recording checks provide additional recording holdouts. No score on Kaggle was obtained or inferred. Existing best predictions and submission artifacts remain unchanged.

Reproduce diagnostics and candidate comparisons with `python src/final_missing_hidden_research/diagnose_missing.py`, `python src/final_missing_hidden_research/run_missing_rts.py`, `python src/final_missing_hidden_research/independent_missing_loro.py`, and `python src/final_missing_hidden_research/run_hidden_profile.py`.
