# V3.4 -- causal shadow-backtest Forecast residuals (research-only)

## Source and motivation

The uploaded V3.3 results rejected camera-E residual correction: Friday pseudo
ratio -0.002115734 improved, Saturday +0.000237679 worsened, and frozen replay
rose to 0.6057086999660036. Preserve champion at 0.6057000521183546.

We now test a **different, online Forecast calibration** with no use of future
measurements. For a main forecast at time `now`, run the unchanged champion once
at `now` for horizons 1..15, and once at `now-12` for horizons 3, 6, 9, 12. The
shadow targets (now-9, now-6, now-3, now) are ALREADY OBSERVED by the main
`now`. Their errors estimate recent champion bias for each visible camera. A
small, clipped correction to main forecasts decays with forecast horizon.

Correction: `p_new = max(0, p_champion + clip(alpha * weighted_shadow_bias *
exp(-horizon/18), -5, +5))`. Alpha is chosen from `0, 0.10, 0.20, 0.35` using
ONLY two older recordings with prior-date training. The chosen coefficient is
saved BEFORE any 2026-09-11 or 2026-09-12 pseudo test is evaluated. If old
recording evidence is not consistent, alpha=0.

The independent-vs-reused limitation remains: all available recordings have
previously been used in model research. No new independent Kaggle score exists.

## Safety and validation

- Original `verified_605700/cloud_research/selected/candidate.py` is imported,
  never edited. Must match SHA-256 `836f54e615089b67099d7b6d2d685765bc999b96bd14f2599982c095fbffa930`.
- Training `data/official/train.csv` or `data/raw/train.csv` must match SHA-256
  `3210a0b33212a60a888977848ec09f5f89bc0954bfd3b88e61fb9d840537f6f6`.
- Requires `v3_2_missing_loro.py` from earlier research, plus Python, numpy,
  pandas and all the champion's own dependencies.
- For every pseudo `now`, the holdout's released rows are cut off at `now`;
  model training comes exclusively from earlier recordings. The shadow call
  separately cuts off released data at `now-12`.
- Historically hidden cameras are excluded from 2026-09-11/12 pseudo released
  context. Camera E is not corrected because no older training recording has it.
- A future-value poison audit on one selected origin per later recording
  verifies predictions do not change when values AFTER that origin are altered.
- Selected alpha is frozen before evaluating the two previously used long
  recordings. Their pseudo-mask design differs from the original frozen suite.
- No Kaggle submission, submission CSV, code integration, source overwrite,
  GitHub push, or Kaggle API interaction.

## Run on Windows / PowerShell

Download `V3_4_research_kit.zip` into the Downloads folder. From the project:

```powershell
Set-Location "C:\Users\USER\Desktop\Madinah-AI-Fresh"

Expand-Archive `
  -Path "$env:USERPROFILE\Downloads\V3_4_research_kit.zip" `
  -DestinationPath "." -Force

Test-Path ".\v3_2_missing_loro.py"
Test-Path ".\verified_605700\cloud_research\selected\candidate.py"

.\.venv\Scripts\python.exe .\v3_4_shadow_forecast.py --root .
```

Both `Test-Path` checks must be `True`; do not substitute an older candidate.

Reports are written to `reports\v3_4`:

- `v3_4_summary.json`: selected alpha, strict cutoff checks, predeclared
  candidate grid on old recordings, and later pseudo results.
- `v3_4_per_recording.csv`: exact per-recording forecast-pseudo scores.
- `v3_4_group_metrics.csv`: results by camera, horizon band and shadow bias.
- `v3_4_frozen_config.json`: frozen correction strength; explicitly NOT deployed.
- `v3_4_decision.md`: pass/reject/abstain and evidence table.

Compress and upload ONLY research reports:

```powershell
Compress-Archive `
  -Path ".\reports\v3_4\*" `
  -DestinationPath ".\V3_4_results.zip" -Force
```

## Target interpretation

An improvement of `0.0057000521183546` is needed to go below 0.600 from
0.6057000521183546. Because the local full score is a mean across 8
recording/task ratios, if Forecast is the ONLY changed task then the sum of
Friday+Saturday Forecast ratio deltas must be < -0.045600416947, equivalent
on average to a drop of ~0.0228002 ratio per recording. The V3.4 pseudo ratios
are not interchangeable with these exact frozen components.

## Author's software QA (NOT real champion performance)

The script was compiled and run on an unmodified local copy of the 13,443-row
training file using the `--smoke-persistence` dummy baseline, NOT the champion.
Four valid anchors per research recording were evaluated. Future-value poison
checks on the two later recordings changed 3,501 and 3,765 future count rows,
respectively; predictions remained identical under the smoke stub. Alpha=0 in
the software smoke run. This is a code-only check and **not** a performance
result, a real champion causality proof, or an official Kaggle result.

## Operational decision rule

A nonzero alpha passing these pseudo guardrails is evidence to study further,
NOT an automatic promotion. The selected model remains untouched. The proper
next step would be exact multi-task frozen replay with the original verified
predictor and a genuinely disjoint validation set, if available. Even the
historically reused long dates cannot certify generalization on their own.
