# Champion Final Audit — Madinah Ethical AI

**Date:** 2026-11-21 (Yemen local).  **Decision:** FORMAT READY, PERFORMANCE UNVERIFIED — submission is optional and requires manual user approval.

## Source package verified

- 51 ZIP entries; extracted source modules and train/test/sample CSVs without altering original archive.
- `selected/candidate.py` and its 19 additional Python files match SHA-256 values recorded by both `selected_composite` and `verified_rerun` archived run metadata (20/20 source files).
- Both saved validation prediction files are byte-identical. The two archived validation reports agree.
- Independently reconstructed all 3,719 labelled validation targets against the actual train.csv labels and recomputed the eight MAE/reference ratios; aggregate equals **0.6057000521183546** exactly.
- The reference denominators were reconstructed, and the development folds were repeatedly used; **0.6057000521183546 is NOT an official Kaggle score or generalization guarantee.** Previous user-reported public best: **0.729**.

## Production compatibility and policy

- Test file: 8,611 rows, including exactly 4,457 withheld targets.
- Submission template: 4,457 unique IDs, order verified.
- Historical cameras A–H; 644 test targets from previously unseen cameras I–M (555 Forecast, 89 Missing).
- The candidate expects callback `targets` without the withheld `people` column. Passing the raw test dataframe directly causes a merge/key error; the production adapter strips it before calling the unchanged `predict_fold`.
- The candidate requires a full **baseline prediction** input: the official `sample_submission.csv` is NOT interchangeable with the archived V13 production reference. We used `submission_v13_causal_reference_calibrated.csv` (SHA-256 `268a031687db92e2968f4477fa216d32709658231fd14447eadf6aa31a27e991`), which is the archived production reference in the user's earlier project, to preserve its fallback predictions.
- Supported historical-camera targets were processed using `selected/candidate.py` **unchanged**, with only historical-camera released context. For unseen-camera target IDs, we preserved the V13 baseline exactly. Hidden remains unchanged, as in the selected model design.

## Measured production dry-run

| Task | Target IDs | Modified from V13 | Unseen-camera fallback |
|---|---:|---:|---:|
| Forecast | 2,660 | 2,105 | 555 |
| Hidden | 849 | 0 | 0 |
| Hidden-future | 484 | 312 | 0 |
| Missing | 464 | 375 | 89 |
| **All** | **4,457** | **2,792** | **644** |

The remaining 1,665 values are unchanged. Predictions preserve the original sample ID order, have no missing/duplicate identifiers or nonfinite/negative counts, and retain the archived V13 values numerically exactly for all 644 unsupported targets and all 849 hidden targets. The chosen production adaptation is not an independently validated Kaggle configuration.

Dry-run source output:
- `Champion_Production_DryRun_605700.csv` (4,457 rows; exact columns `id,people`)
- `Champion_Production_DryRun_Audit.json` (input hashes, count/coverage and modification details)
- `build_champion_production.py` (reproducible production adapter)

The dry-run emitted several `Mean of empty slice` NumPy warnings from the existing `missing/ou_component.py`. They were not fatal and the final CSV passed finite/nonnegative validation, but they are a limitation of this execution rather than proof of robustness.

## What is NOT verified

1. No official Kaggle score for the newly generated production CSV. No Kaggle upload performed and no submission attempt consumed.
2. The V13 production reference may not be the user's public-best **0.729** submission. The 0.729-file itself was not supplied for comparison.
3. Performance on the 644 new-camera targets has never been measured on labelled new-camera holdouts; the fallback comes from archived V13, not this model.
4. Full production causality was not exhaustively audited here; two saved sampled checks passed in historical development only.

## Reproduction in the user's Windows project

The following uses the current project tree and the files already present on the user's machine; choose fresh output paths so it does not overwrite existing files:

```powershell
Set-Location 'C:\Users\USER\Desktop\Madinah-AI-Fresh'
.\.venv\Scripts\python.exe "$env:USERPROFILE\Downloads\build_champion_production.py" `
  --source '.\verified_605700\cloud_research\selected' `
  --train '.\verified_605700\official_data\train.csv' `
  --test '.\verified_605700\official_data\test.csv' `
  --sample '.\verified_605700\official_data\sample_submission.csv' `
  --baseline '.\verified_605700\research_kit\production_reference\submission_v13_causal_reference_calibrated.csv' `
  --output '.\champion_production_dryrun_new.csv' `
  --audit '.\champion_production_dryrun_new_audit.json'
```

**Decision gate:** Consider manually testing this candidate in one official submission only if the user's remaining quota and closing time permit, understanding its local score is not an official estimate. Preserve the public-best 0.729 submission file and the other submission slots. Never automatically upload.
