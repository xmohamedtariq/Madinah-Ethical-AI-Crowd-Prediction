# V3.3 — Camera E residual calibration (research only)

**Decision from V3.2:** REJECT. Four historically reused older-recording holdouts gave 0/4 improvements, mean Missing ratio delta +0.006388 (worse), and final alpha=0. No frozen prediction changes. Keep the selected champion at 0.6057000521183546. The user-reported official Kaggle best is 0.729. **No Kaggle submissions will be generated or uploaded by this script.**

## Why this distinct experiment?

V3.2 was limited to cameras A and F because all four older dates lack camera E. Camera E contributes 0.040328 to the frozen full score from 55 fast-falling Missing targets. In the reused Friday/Saturday diagnostics, the mean signed E error differs substantially by date. A fixed A/F curvature coefficient does not justify a transfer to E.

V3.3 tests an entirely different, conservatively regularized *conditional residual* on E alone. It does **not** tune on the 358 frozen labels:

1. Read only the IDs, dates, cameras, and timestamps of the 358 frozen Missing targets to remove them from pseudo-gap targets and same-date released context. No frozen gold or champion residual columns are loaded for model fitting.
2. Predeclare 12 disjoint E pseudo-gaps per date (4 clock quarters × two fast-decline lengths of 8/12 minutes plus one control length of 7), excluding the 358 known frozen Missing rows with a 7-minute buffer. Endpoints and timestamps, not hidden interval truths, determine eligibility.
3. Divide these synthetic gaps into two disjoint panels per date. For each panel, call the exact **verified selected champion** `predict_fold(train, released, targets, baseline)` with pseudo labels removed from `released` and **not** supplied in targets or baseline.
4. After inference, join pseudo-gap truth only for learning/scoring. Train one fixed clipped weighted ridge residual correction with six **label-free** features from the other date only; compare against Champion on the held-out date. Swap dates (two LOSO folds). No per-date alpha hunting.
5. Aggregate camera/gap/flow/panel results. A strict two-date pseudo-gap gate requires both held-out dates to improve, with mean normalized Missing ratio improvement of at least 0.005. Even a pass is **research evidence only**, not an automatic promotion.
6. Freeze a descriptive coefficient JSON before reading the full frozen diagnostics for a strictly retrospective preview. No Kaggle upload, submission CSV, or edits to selected source or champion predictions.

**Important limitation:** These two E-containing dates have already been examined repeatedly in prior model development. The masks exclude the 358 frozen Missing cells, but cannot reproduce all other original Forecast/Hidden/Hidden-future validation masks. This is not untouched independent confirmation. The corrected estimate on the old 358 targets is **not a new validated model score**. Neither a positive nor a negative result can be equated with a new public/private Kaggle score.

## Project requirements

- Existing Windows project: `C:\Users\USER\Desktop\Madinah-AI-Fresh`
- Python 3.10+; `numpy`, `pandas`
- `v3_2_missing_loro.py` still present in the project root from the previous research kit
- Selected champion source at `verified_605700\cloud_research\selected\candidate.py`
- Official train: `data\official\train.csv` (unchanged SHA-256 expected from V3.2)
- Frozen V2 files: `reports\breakthrough_v2\missing_target_diagnostics.csv` and `reports\breakthrough_v2\per_task_score_comparisons.csv`

V3.3 checks the model and dataset SHA-256 against the successful V3.2 run. If a source changed, it **stops**, rather than substituting an older model.

## Run (PowerShell)

Download `V3_3_research_kit.zip`, then run:

```powershell
Set-Location "C:\Users\USER\Desktop\Madinah-AI-Fresh"
Expand-Archive -Path "$env:USERPROFILE\Downloads\V3_3_research_kit.zip" -DestinationPath "." -Force

Test-Path ".\v3_2_missing_loro.py"
Test-Path ".\verified_605700\cloud_research\selected\candidate.py"

.\.venv\Scripts\python.exe .\v3_3_e_residual_calibration.py --root .
```

If either `Test-Path` returns `False`, stop and use the original files; do not rename another model to `candidate.py`.

Artifacts, all **research only**, will be written to `reports\v3_3\`:

- `v3_3_decision.md` — human-readable verdict
- `v3_3_summary.json` — exact source identity, validation metadata, and fold scores
- `v3_3_outer_dates.csv` — Friday-versus-Saturday pseudo holdouts
- `v3_3_group_results.csv` — error changes by flow/gap/panel/window
- `v3_3_pseudogap_coverage.csv` — deterministic gap selection
- `v3_3_frozen_config.json` — *non-deployed* coefficients, frozen before retrospective replay

Zip the output for review (no Kaggle submission):

```powershell
Compress-Archive -Path ".\reports\v3_3\*" -DestinationPath ".\V3_3_results.zip" -Force
```

Upload `V3_3_results.zip` to the chat for analysis.

## QA and scientific status

The author ran a local code smoke-test on the real train rows with a **linear-interpolation stub**, not with the user's selected `candidate.py`. The code generated 12 nonoverlapping E gaps on each date, 216 pseudo targets in four panels, trained/tested ridge without errors, and verified that adding fake truth columns does not affect predictions. A zero-coefficient replay reconstructs the frozen 0.6057000521183546 score exactly. These are **software tests only, not evidence of V3.3 accuracy**.

The original final Kaggle submission remains unchanged. A newly observed retrospective score below 0.600 would require additional legitimate, genuinely independent testing before any production decision.
