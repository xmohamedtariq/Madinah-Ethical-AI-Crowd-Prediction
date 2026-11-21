# V3.2 — Conservative Missing correction (research only)

**Baseline preserved:** local champion `0.6057000521183546`; user-reported best Kaggle public score `0.729`. **No Kaggle submission, upload, production-candidate promotion, or overwrite.**

## What I inspected

The user-supplied `V3_2_interface_bundle.zip` contains:

- `frozen_protocol.py` — an older frozen synthetic validation/multitask metric implementation.
- `evaluate_full_solution.py` — an older end-to-end model; **not** the selected 0.605700 champion.
- `independent_missing_loro.py` — the relevant callback invocation: `verified_605700/cloud_research/selected/candidate.py`, function `predict_fold(train, released, targets, baseline)`.
- `diagnose_missing.py` — released-context gap features and post-hoc gold comparison.
- `best_missing_model.py` — a linear interpolation helper; **not** the selected champion model.

The previously uploaded frozen diagnostics contain 358 Missing rows from two repeatedly examined recordings. The actual selected `candidate.py` and all four official benchmark task predictions were *not* part of the interface upload, so an authentic champion-based V3.2 score has not been run in this sandbox.

The available train.csv has 13,443 rows from six recordings. All four older recordings contain `cam_A` and `cam_F`, but **none contain `cam_E`**, making a standalone independent correction calibration for camera E impossible using these four recordings alone. Moreover, these older recordings were already examined in the RTS experiment: they are recording-disjoint in the new callback tests but **not untouched final holdouts**.

## What the script does

1. Checks dataset schema and calls the **actual selected** `candidate.predict_fold` (without silently substituting an older model).
2. Builds predeclared, non-overlapping 5–15-minute pseudo Missing gaps from four older recordings. Gap geometry uses timestamps and released endpoints; hidden interiors are kept strictly for later gold scoring.
3. Passes only legal inputs (`train`, `released`, metadata-only `targets`, linear `baseline`) to champion.
4. Computes the same **label-free** curvature feature used by V3.1A, narrowly gated on fast-falling `cam_A` / `cam_F` blocks.
5. Cross-fits **one** small correction coefficient across the four recording holdouts. For each holdout it calibrates on the other three, never on the held-out recording's labels.
6. Reports all recording/camera/gap-length/flow-regime changes. Chooses a conservative coefficient using only older-recording data.
7. *After freezing that coefficient*, optionally shows a **retrospective only** preview on the 358 reused frozen Missing diagnostics. A `cam_E` transfer is shown separately and **cannot be promoted**.
8. Preserves Champion, reports a negative/positive score change without generating any Kaggle submission file.

### Run on your Windows project

Download and extract `V3_2_research_kit.zip` into:

`C:\Users\USER\Desktop\Madinah-AI-Fresh`

Open PowerShell:

```powershell
Set-Location "C:\Users\USER\Desktop\Madinah-AI-Fresh"
Test-Path ".\verified_605700\cloud_research\selected\candidate.py"
.\.venv\Scripts\python.exe .\v3_2_missing_loro.py --root .
```

`Test-Path` must show `True`. If it prints `False`, the program will **stop** without substituting any older baseline; share the location of the actual selected champion source rather than renaming an unrelated model.

Result files are created in **`reports\v3_2\`**:

- `v3_2_decision.md` — human-readable decision and per-recording scores.
- `v3_2_summary.json` — frozen coefficient, guardrail checks, file hashes, limitations, and retrospective score preview.
- `v3_2_outer_holdouts.csv` — the four recording-disjoint Missing comparisons.
- `v3_2_group_metrics.csv` — camera/gap-length/flow-regime comparisons.
- `calibrated_alpha_from_older_only.json` — coefficient frozen **before** reading reused Friday/Saturday diagnostics.

To share these **research results** (not a Kaggle submission):

```powershell
Compress-Archive -Path ".\reports\v3_2\*" -DestinationPath ".\V3_2_results.zip" -Force
```

### QA already run in this sandbox

- Original train dataset read and validated (13,443 rows, six recordings).
- Synthetic gap mask generator built 16 / 16 / 14 / 16 blocks, total **596** pseudo target rows.
- End-to-end logic ran with an explicit **linear-stub smoke test**, not a real champion evaluation. The smoke-test 0 improvement is not an ML result.
- Callback interface exercised with an isolated **mock** candidate.py; that mock is not the selected model and is not included in the zip.
- Retrospective frozen replay with the preexisting V3.1A alpha `+0.10`, including A/E/F, exactly reproduces **`0.6044515190612298`** from V3.1. This is a consistency check, not a new validation success.
- Existing V2 task comparisons show Forecast MAE changes in supposedly Missing-only/Hidden-only candidate runs; this is flagged as a protocol integrity issue. It does not demonstrate leakage by itself.

## What counts as success

A recording-disjoint improvement is necessary but not sufficient: these four previously inspected recordings do not provide **untouched** independent confirmation, and no older recording contains camera E. Reused frozen previews, even below `0.600`, cannot justify a champion replacement on their own. An actually untouched recording / independent benchmark or an official permissible final evaluation would be required for stronger claims. The user requested no new Kaggle submission, and none is generated.
