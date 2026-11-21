# V2 audited repair — 17 November 2026

**The local 0.600 target has not been reached. Kaggle public score is unknown. No upload or final-selection change was performed.**

## Measured scores

| Evaluation | Original frozen masks | Actual-test-mask replay |
|---|---:|---:|
| Nested date-held-out model-selection procedure | 0.799967763 | 0.834872552 |
| Retrospective replay of final selected configuration | 0.793102623 | 0.769424008 |

The retrospective row uses dates that also informed final configuration selection. It is **selection evidence**, not independent confirmation. The nested row chooses components inside each outer training set, excluding the outer date entirely. Available outer dates had already been inspected during project development, so this is not a pristine new confirmation cohort either.

The original frozen masks/reference values were preserved in `reports/legacy_frozen_snapshot`, with input/file hashes. The additional matched-mask protocol was frozen separately before evaluating the repaired candidates; it does not replace the old protocol. Both use equal recording weights and the mean of four local normalized MAE ratios. The exact official reference implementation and score aggregation were not included in the uploaded project, so these are **local reconstructed-reference scores**, not certified Kaggle score estimates.

## What failed in the uploaded code

1. The V2 generator did not calculate its literal `0.78496` CV value. Its proposed checkpoint would have said ready for manual upload regardless of whether the local target was met.
2. Running that generator on actual test inputs produced **91 NaN predictions** and failed its own CSV integrity assertion. Replays on the old synthetic folds failed with **123 Friday** and **9 Saturday** NaNs. Hidden reconstruction blended directly with missing reference-camera values.
3. The V1 evaluator and V1 submission generator were different algorithms: Ridge/LightGBM hidden blending in evaluation versus stabilized Ridge/reference in generation, plus different forecast sample strides and model parameters. Their reported scores are not interchangeable.
4. The missing-gap code grouped all gaps for one camera together and used the extrema across those separate gaps for residual corrections. The repair uses the nearest actual released endpoints for each queried missing interval.
5. A global pivot followed by `dropna()` removed all four early recordings from hidden-camera fitting because cam_E is absent there. The repair handles missing input features within each recording, without dropping all target observations for that reason.
6. Hidden-future reconstruction was created from the full recording before its history was cut at now. Bidirectional imputation could therefore contaminate past reconstructed features. The repaired future-task inference cuts rows **before** any interpolation, aggregation or reconstruction.
7. The old Saturday validation used Friday's now-times. Matched-mask replay reads the actual per-date, per-camera target/release positions and horizons from test metadata.
8. The old frozen scorer only printed results, did not persist them, accepted some invalid denominators with a fallback score, and did not persist its declared cache. The repaired path saves masks, denominators, predictions, configurations and hashes, and rejects incomplete or non-finite inputs.

## Preserved V1 and reproducibility limitation

The original V1 CSV is preserved at `v1_archive/submission_v1_independent.csv` with SHA-256:

`5a2f1c00468c4ec8a60f5a14b3cabee86a67edf711830994ad02b5ed60c1f3cb`

Replaying its generator source here on the old frozen masks scored 1.028892650. **That is a source-replay result, not a certified score for the archived trained artifact.** Retraining the same submitted generator source on actual inputs changed 3,143 archived CSV predictions (maximum change 22.839). The exact cause of the source/artifact mismatch remains unresolved; source/environment/training differences are possibilities. No original trained model or OOF artifacts were supplied. The accelerated feature implementation matched 150 original feature queries exactly; a separate 465-row training-matrix check also matched all features, targets, column order and dtypes exactly. These bounded checks do not establish complete historical artifact reproduction.

The literal V1 and V2 scores in the original metadata are retained only as historical source material in `original_src`; none is used as verified evidence.

## Coverage and references

| Task | Labelled replay rows / actual target rows | Coverage |
|---|---:|---:|
| forecast | 2,036 / 2,660 | 76.54% |
| hidden | 843 / 849 | 99.29% |
| hidden-future | 482 / 484 | 99.59% |
| missing | 358 / 464 | 77.16% |

Coverage describes target labels, not complete equality of every available historical context. Missing source/context observations are retained as missing information, not counted as validated data. No genuine cam_I–cam_M target labels were invented.

The disclosed sample forecast values reproduce exactly (2,660/2,660); missing values reproduce within 0.0005 (464/464). For hidden/hidden-future, direct reference-camera reconstruction matches where its source is released, but the direct source is absent on **91 hidden and 14 hidden-future rows**. The uploaded reference-reproduction script crashes on such cases. This audit does not establish the official fallback policy and makes no attempt to invert sample values into withheld target labels. The repaired predictor computes its own legal-context baseline; sample_submission supplies output IDs/order only.

A separate pseudo-new-camera experiment removed each historical target-camera identity from all training records, mapped its held-out labels into actual cam_I positions, and retained only the permitted release mask. Mean forecast ratios across those cohorts: damped baseline 0.873344, normalized 0.900278, transfer 0.826539. These are **pseudo-camera stress results**, not actual I–M validation. The final production choice was made by the declared date-held-out selection rule; it was not changed after observing this stress result.

## Final experimental artifact

- File: `submissions/submission_v2_independent.csv`
- SHA-256: `2fa4af7e3e8c1d56c5bf4ff8daab415e4f963a0cea10cd172495a17439da557d`
- Rows: 4,457; columns exactly id,people; exact sample ID order; zero duplicate IDs, missing/non-finite values or negative predictions.
- Prediction range: 2.534991 to 999.676144.
- Configuration: forecast=normalized_damped, hidden=trees, hidden-future=direct, missing=learned_half.
- Configuration selection: inner recording-date-out evaluation, equal date weighting, at least 1% improvement over the default. Four distinct historical dates informed the final fit. Production and validation use the same pipeline implementation and selection rule.
- All 3,144 actual future targets passed fixed-model counterfactual checks: changing every released count after now, and separately deleting all such rows, changed no prediction (maximum difference 0). This tests prediction-time causality with fitted parameters held fixed. Training uses released labelled rows as authorized in the rules supplied by the user.
- Eleven structural/regression checks passed, including masking/fit separation, rejection of invalid scores, non-pooling of distant missing gaps, and exclusion of synthetic masked target values from missing features.
- Huber and neighbor candidates were evaluated but not selected. Some Huber fits emitted convergence warnings; final generation does not fit or use those optional models.

Detailed predictions, per-recording MAEs/denominators, nested selections, 30-minute-block bootstrap diagnostics and logs are included in reports. Bootstrap intervals condition on fitted models and selected configuration; they do not cover selection uncertainty or forecast a public score.

## Reproduce and continue

See README.md for PowerShell commands. Verification recalculates scores from saved row predictions and checks source/input/mask/CSV hashes; it does not retrain models or call any LLM API. Generation and training need Python 3.12 and the pinned dependencies.

The current checkpoint is `reports/v2_checkpoint.json`. No background model job remains running. The next development decision should address the missing-task generalization gap and new-camera transfer, using predeclared tests and the saved protocols. Obtain the official reference/scorer documentation before claiming exact official metric parity. Additional authorized labelled recordings or a genuinely withheld cohort would improve confidence; unknown test labels must not be fabricated or sought through prohibited sources.
