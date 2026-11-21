# Madinah Ethical AI — V12 diagnostic research update

**Result:** No new candidate survived the cross-recording reliability gate; best preserved submission remains V11. The reported 0.400 target has **not** been achieved locally or on Kaggle.

## Baselines and experimental evidence

- The frozen V11 matched-mask development score is **0.645284723**, computed as an unweighted mean of four task-level MAE/reference-MAE ratios averaged over the two development recordings. It is not an official leaderboard estimate.
- V11 Friday matched aggregate: 0.667501795; V11 Saturday matched aggregate: 0.623067651.
- Forecast MAE of preserved V7/V11 forecasting component on matched masked rows: **16.905150**. Best tested direct local-history alternative (ewm3_sl0.7) produced MAE **30.889642**, substantially worse. This was only a diagnostic screen, not a production model.
- Cross-recording supervised residual correction: best tested overall forecast MAE **16.922336**, worse than baseline, with an adverse fold. No correction accepted.
- Missing reconstruction screen on 25 camera × recording × mask groups: best candidate (**pchip**, blend **0.10**) increased pooled MAE by **0.130646**. No candidate improved all four matched/legacy comparisons. Existing predictions retained.
- Additional cross-camera hidden and smoothing experiments were already documented as inconclusive or slightly negative in prior development tests.

## Important limits

The matched and legacy masks share underlying September 11 and 12 recordings, repeatedly used in model selection. This is **not independent out-of-time evaluation**. Some test-specific camera/horizon configurations are not represented. A public Kaggle score can differ greatly from the locally reconstructed metric.

## Decision

No V12 submission is generated: uploading an unvalidated or objectively worse candidate would risk an available leaderboard submission. Preserve `submission_v11_hidden_secondpass.csv` (SHA-256 `1c4c537079cd11273f95612acfa7af7765a39e00c8a96726f2b6e7d3cc17c9e0`) as the latest development best, and preserve the highest officially scored prior public submission (0.749). Neither constitutes a 0.400 result.

## Reproducible screens

- `exp/v12_forecast_baseline_screen.csv` and `experiment_forecast_v12.py`
- `exp/v12_crossday_correction.csv` and `exp/v12_crossday_correction.py`
- `exp/v12_missing_interp_groups.csv`, `exp/v12_missing_interp_summary.csv`, and `exp/v12_missing_interpolation_screen.py`

One additional, genuinely held-out test under a frozen model-selection protocol would be worth more than more fine-tuning on the same two dates.
