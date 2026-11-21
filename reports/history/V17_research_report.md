# Madinah Ethical AI — V17 structural candidate

**Status: research candidate, NOT submitted to Kaggle.** Best verified official public score from user screenshots: **0.740** for V13/V14; V15 public **0.780**; V16/V17 have no official public or private scores. The target **0.500 has NOT been verified**.

## Reproducible files

- `train.csv`, `test.csv`, `sample_submission.csv`: official released data and reference template. All official withheld target `people` values are blank.
- `submission_v14_friday_hidden_guarded_missing.csv`: verified baseline public 0.740.
- `submission_v16_selective_saturday_forecast.csv`: Saturday causal forecast LightGBM candidate, unsubmitted.
- `v14_historical_oof.csv`: frozen reconstructed historical V14 predictions for 2 dates / 2 masks. All 7,353 historical `people_true` entries were verified against official `train.csv` exactly. **Development data, not an independent validation set.**
- `forecast_saturday_historical_oof.csv`: V16 Saturday forecast research predictions.
- `V17_build.py`: trains Hidden residual model on historical TRAIN labels and generates V17 CSV.
- `V17_replay_validation.py`: replays V14/V16/V17 scores, training Hidden models on one historic date and checking on the other.
- `submission_v17_structural_calibrated.csv`: **experimental candidate**, 4,457 rows in exact sample ID order.
- `V17_historical_results.json`, `V17_historical_cv_summary.csv`, `V17_historical_cv_by_kind.csv`, `V17_historical_cv_camera.csv`: detailed historical test results.
- `V17_integrity.json`: source hashes and causal/no-leakage audit.
- `V17_build_audit.log`, `V17_replay_audit.log`: build and test output.

## Installation and recreation (PowerShell)

```powershell
py -m pip install numpy pandas lightgbm
py .\V17_build.py
py .\V17_replay_validation.py
Get-FileHash .\submission_v17_structural_calibrated.csv -Algorithm SHA256
```

Expected SHA-256: `9b00a2413af14f85a9e52d5494090c73beea9a2875e8f8a5574b6a244065edbf`.

## Causal safety

- Friday Forecast A/E/F, 486 rows: `(1 + 0.15)*V16 - 0.15*(last known same-camera value <= now)`, clipped at zero. Official baseline reference was independently verified to equal the last causal observation across the 33 affected forecast origins.
- Hidden, 849 rows: additive relative residual correction with constrained L1 LightGBM trained on historical TRAIN-label masks. All released context is allowed for Hidden.
- Saturday forecast: retained from V16; same legal past-only features as before.
- Missing, Hidden-Future: byte-for-byte retained from V16/V14.
- No information from any held-out test `people` target was available or used.

## Historical surrogate scores

The four masks reuse only two days Sep11/Sep12, so do not regard them as independent or as a calibrated estimate of Kaggle public/private score.

| Historical mask / date | V14 | V16 | V17 |
|---|---:|---:|---:|
| Legacy Friday Sep 11 | 0.622078 | 0.622078 | **0.612940** |
| Matched Friday Sep 11 | 0.654289 | 0.654289 | **0.645502** |
| Legacy Saturday Sep 12 | 0.645367 | 0.643893 | **0.637905** |
| Matched Saturday Sep 12 | 0.618764 | 0.613111 | **0.609892** |
| **Mean (surrogate)** | **0.635124** | **0.633343** | **0.626560** |

From V14 to V17 average historical Forecast MAE: 17.65486 → 17.01498 (~3.62% better), Hidden MAE: 4.13044 → 4.02567 (~2.54% better); Missing and Hidden-Future unchanged. The combined mean improvement is ~1.35% compared with V14. Improvements were observed for both historical masked protocols and each of three selected Friday forecast cameras and three Hidden cameras, but historical folds have been repeatedly inspected and could overfit.

**Decision:** No remaining Kaggle attempt consumed. This candidate is verified and ready to run, but **not sufficiently strong to justify guaranteeing an improvement on Kaggle or the 0.500 target**. Preserve public-best 0.740 and both remaining submissions until more external evidence is available.
