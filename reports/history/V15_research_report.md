# Madinah Ethical AI — V15 Weekly Causal Challenger

Date: November 19, 2026

## Honest result status

- User-confirmed official Kaggle public result for V13 and V14: **0.740** (tie), private not yet available.
- Last reconstructed paired historical **V14** local four-mask average: **0.635124441**, based on *two repeatedly inspected recording days* (September 11 and September 12), **not independent validation**.
- A **comparable V15 all-task local score is not established**. The new weekly source method was measured on pseudo-masked **test released observations**, where prior-week analog data is available. The old historical fold could not offer an equivalent full-week predecessor.
- **Neither local 0.500 nor public 0.500 has been achieved or verified.** Do not advertise this file as scoring 0.500.
- V15 is a **riskier experimental challenger**; the officially scored V14 remains the proven fallback.

## Main production changes

| Kind | Total | Changed vs V14 | Method |
|---|---:|---:|---|
| forecast | 2660 | 1538 | 983 old-camera forecasts gated by *strictly prior* pseudo-window errors, 555 new-camera forecasts shrunk toward their last causally available reading |
| hidden | 849 | 849 | 50% V14 and 50% previous-week same-weekday camera prior dynamically calibrated with full released cross-camera context; full released context is allowed for hidden |
| hidden-future | 484 | 484 | 60% V14 + 40% historical prior calibrated *only with peer observations up to now* |
| missing | 464 | 0 | V14 untouched because alternate imputation did not show reliable improvement |

In total, **2,871** predictions changed. All **4,457** submission IDs match the original sample order, are unique, and have finite nonnegative values. SHA-256 `bdc81f110458a6a6519ca83af02ff0df9cd3b9732893cc1859e2acbcd0ed6d19`.

## Pseudo-label studies (NOT Kaggle score estimates)

1. Causal forecast study on released TEST observations: 1,571 origins met the past-only gate-evaluation history requirement. The weekly analog was activated at 793 origins; among their **10,166** future released pseudo labels the weekly analog MAE was **16.0920**, while last-observation MAE was **20.5044** (21.52% reduction). These overlapping pseudo windows are highly dependent. **V14 model predictions were not generated for the pseudo origins**, so this does *not* show an improvement against V14.
2. Noncausal pseudo-hidden study: 3,283 released observations of 13 known cameras were simulated as target-hidden cameras, and *excluded* as features. Historic previous-week smoothed prior MAE **22.545**, peer-calibrated weekly prior **17.014** (24.53% reduction). This comparison was not against V14 and isn't representative of actual hidden B/G/D labels.
3. Causal pseudo-hidden-future study: 5,868 held-out released labels in 455 synthetic origin windows; previous-week prior MAE **22.685**, prior-peer calibrated **20.405** (10.05% reduction). All peer observations used in each model query were at or before its own origin.

None of these three compares overall all-four-task reconstructed Kaggle local score to 0.500. None uses withheld TEST `people` values.

## Data rules audited

- All 4,457 official target rows have `people` missing in the provided official `test.csv`; no hidden target labels were loaded.
- Historical fitted patterns use released `train.csv` from Sept 11/12 as the prior reference for Sept 18/19 predictions.
- `forecast` and `hidden-future` only use current-recording released observations `minute_of_day <= now` and fixed earlier training data. Forecast gate is built from released pseudo-label windows with each future pseudo-label time **<= the real forecast origin now**.
- `hidden` may use full released same-recording observations; hidden target camera is never read as an input.
- `missing` is unchanged, preserving V14's released-only imputation.

## Reproduce

Unzip the research package, open a shell in that folder, and run:

```bash
python -m pip install numpy pandas scipy scikit-learn
python V15_build_source.py
```

The `forecast_analog_pseudo.csv` shipped in this package includes the prior released-only pseudo-window backtest needed to recreate the gate. To rebuild it from official data instead, run `python forecast_analog_study.py` *before* `V15_build_source.py`.

The V15 CSV has been recreated byte-for-byte in a separate clean directory from the copied source, official data and pseudo-window CSV; the SHA-256 matched exactly. Re-running `V15_build_source.py` writes `V15_integrity.json` and `V15_gating_audit.csv`.

## Kaggle upload (Windows PowerShell)

After saving CSV to Downloads:

```powershell
Set-Location "C:\Users\USER\Desktop\Madinah-AI-Fresh"
New-Item -ItemType Directory -Force .\submissions | Out-Null
Copy-Item "$HOME\Downloads\submission_v15_weekly_causal_challenger.csv" .\submissions\ -Force
& ".\.venv\Scripts\kaggle.exe" competitions submit `
  -c "MDA-Ethical-AI-KAUST-Academy" `
  -f ".\submissions\submission_v15_weekly_causal_challenger.csv" `
  -m "V15 weekly causal challenger (experimental)"
```

`kaggle.exe` is the installed CLI in the project's virtual environment; if absent, run `& ".\.venv\Scripts\python.exe" -m pip install kaggle` first. Authentication is required. Submission is manual and was **not performed** by this package.
