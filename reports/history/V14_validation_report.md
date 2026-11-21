# Madinah Ethical AI — V14 experimental candidate

Date: 19 November 2026 (Yemen time)

## 1. Official situation

- Verified via the user's Kaggle screenshot: V13 **public score 0.740** (improved from 0.749). **This is the only verified official result in this package.**
- V14 has **not** been submitted. Its public/private scores are **unknown**.
- The requested public score **0.500 or lower has not yet been reached**.

## 2. What was actually tested

Test-side released-only **pseudo-gap experiments**:
- 282 synthetic masked segments, 2,287 known values removed from input before prediction.
- Pure linear interpolation MAE 10.43393.
- Local peers correction (0.5 amplitude) MAE 9.58576, an 8.13% relative reduction in this synthetic evaluation.
- Historical train released-style pseudo-gaps: 378 groups; linear 9.3499; peers 8.5697, 8.34% relative reduction.

But matched-mask V13 comparisons showed that a **wholesale replacement** of V13's missing imputer would hurt. On matched masks, V13 missing MAE was 5.01315, versus 5.02833 for even a 15% peer blend; thus this replacement was **rejected**. The only accepted missing edit is a small, guarded, opposite-sign correction on Friday, affecting 12 real test rows.

Forecast transfer experiments on released observations:
- 13,803 pseudo-forecast labelled pairs from TEST released context; historical TRAIN fit only.
- Simple last-observation MAE 18.77065; causal damped local trend MAE 18.27979.
- LightGBM 150 estimators/depth4 MAE 30.88346 (worse). **Rejected**.
- Historical matched-mask V13 forecast MAE was 22.14615 on Sep 11 and 12.43556 on Sep 12, far better than local-only methods (~42.02 and ~19.59 respectively). **V13 forecast preserved**.
- Pseudo-test forecast future released values were used only for research evaluation and **never as inputs to V14 forecasts**.

Hidden-camera evaluation:
- New Friday reference weight: 20% historical paired-camera reference / 80% frozen V11, up from V13's 10% / 90%.
- Saturday hidden remains byte-identical to V13.
- Reference on historic validation is built from TRAIN with the validation recording removed and the full masked validation target set removed from published context; it does not read withheld labels.

## 3. Audited local scores (surrogate, not certified Kaggle metric)

| Mask protocol | Recording | Reconstructed V13 | New V14 | Difference (negative is improvement) |
|---|---|---:|---:|---:|
| Matched | Sep 11, Friday | 0.658953 | **0.654289** | -0.004664 |
| Matched | Sep 12, Saturday | 0.618764 | **0.618764** | 0 |
| Legacy | Sep 11, Friday | 0.623942 | **0.622078** | -0.001865 |
| Legacy | Sep 12, Saturday | 0.645367 | **0.645367** | 0 |

- Matched mean: V13 0.638858412 -> V14 **0.636526557**.
- Legacy mean: V13 0.634654696 -> V14 **0.633722326**.
- Four-mask arithmetic mean: V13 0.636756554 -> V14 **0.635124441** (relative improvement ~0.256%).
- These are **only two independent recording dates**, each repeatedly evaluated using two masks. They are not four independent holdouts and are prone to repeated-selection bias.
- The scorer is a local reconstruction (four taskwise MAE / reference MAE ratios); actual public score remains unknown.

## 4. Candidate integrity

- Output: `submission_v14_friday_hidden_guarded_missing.csv`
- SHA-256: `ceec4f57451f6182e88213aa8a2984fac66d42a9fe60acf93d4ebcbe878ac6cd`
- Exactly 4,457 rows in official sample order; columns `id,people`, all IDs unique, all predictions finite and nonnegative.
- Changed precisely 564 Friday `hidden` predictions and 12 guarded Friday `missing` predictions, total 576.
- Preserved 3,881 source lines byte-for-byte. **All 2,660 forecast rows and all 484 hidden-future rows are untouched**, and all Saturday task rows are untouched.
- For hidden/missing changes, all input observations are legally released data (plus original TRAIN and fixed baselines). No withheld test targets read.

## 5. Promotion decision

**Experimental candidate only.** It passes the local holdout-replay integrity checks, but expected gains are modest and not independently validated. Keep the proven V13 file (public 0.740) as the safe fallback; submit V14 at most once to get an authentic public measurement. If V14 is worse, retain V13. The private 48% can rank differently.

### Manual Kaggle upload (PowerShell)

```powershell
Set-Location "C:\Users\USER\Desktop\Madinah-AI-Fresh"
New-Item -ItemType Directory -Force -Path .\submissions | Out-Null
Copy-Item "$HOME\Downloads\submission_v14_friday_hidden_guarded_missing.csv" ".\submissions\" -Force
& ".\.venv\Scripts\kaggle.exe" competitions submit `
  -c "MDA-Ethical-AI-KAUST-Academy" `
  -f ".\submissions\submission_v14_friday_hidden_guarded_missing.csv" `
  -m "V14 masked-validation Friday hidden + guarded missing"
```

To reproduce locally from the research package:

```powershell
python -m pip install numpy pandas scipy
python .\V14_build_source.py
python .\V14_validate_source.py
Get-FileHash .\submission_v14_friday_hidden_guarded_missing.csv -Algorithm SHA256
```

## 6. Next substantive research

The immediate route toward a much lower public score is not blind postprocessing. It is an independently validated structural update for low-coverage camera/time windows, especially new-camera I–M forecast and the hidden-camera scenario, using a genuinely held-out recording or a higher-fidelity causal simulator. Those changes were **not** promoted here because the current released data did not support honest, decisive improvement over V13. A public 0.500 outcome cannot be promised from the present evidence.
