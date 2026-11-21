# V4 continuation — exact public target still pending

**لم نصل إلى 0.600.** أقرب تجربة محلية محفوظة الآن هي **0.702275**، وأفضل نتيجة Kaggle عامة مؤكدة الآن **0.802** لملف V4 fused، وفق نتيجة Kaggle التي أرسلها المستخدم. تحسن السكور العام من **0.811** بمقدار **0.009**. سكور مرشح phase/bridge الآخر غير معروف. المرشح الأقرب **بحثي وغير معتمد تلقائيًا**؛ لم يجتز بوابة التحسن الإضافي 3%، رغم تحسن المقارنات الأربع.

## Current objective

The latest user instruction is to approach a **displayed Kaggle PUBLIC score of exactly 0.600**, without going below it. Local candidate selection is constrained to aggregate score **>=0.600**. This supersedes the earlier <=0.600 criterion. Hidden test labels mean the public result, including whether it lands below 0.600, cannot be guaranteed in advance. No local value establishes the public or private score.

**Goal: PENDING. The user manually uploaded fused V4 and reported PUBLIC 0.802. No assistant upload or final-selection change was performed.** The public evidence and original internal gate result are preserved. No background experimentation is running while this review is being reported.

## Frozen files and status

| File | Matched local estimate | Kaggle public | Status |
| --- | ---: | ---: | --- |
| submission_v3_new_camera_transfer.csv | 0.769424 | **0.811**, user-provided result | Preserved scored baseline |
| submission_v4_phase_bridge_experimental.csv | 0.720650 | Unknown | Earlier experimental candidate; its original development gate passed |
| submission_v4_fused_forecast_research.csv | **0.702275** | **0.802**, user-provided result | Best measured public; original incremental promotion gate failed |

The local table uses identical frozen masks, reference definitions and denominators. Local estimates are not public-score predictions. The full dates were repeatedly inspected during development; they are not fresh independent confirmation.

Research CSV SHA-256:

`2195ad5d9cbc256a3bc16dd65d273b0eb25bcdacd38cca35b96a5e8800813521`

Preserved phase/bridge CSV SHA-256:

`b531a65c6e2323ba3640449bf702dd3cf830e16bfbd124df0d424656205ddbee`

Preserved V3 SHA-256:

`159aa8d53b78120e167b5bb0d8f865c810f542330c6753f877212f03c96c4713`

The displayed PUBLIC 0.802 comes from the user's Kaggle recent-submission card naming the fused V4 CSV and its new-personal-best message. The local file hash was verified; the remote upload bytes were not independently downloaded. No account/API refresh was performed. The submission reference, absolute timestamp and private score remain unknown. See `reports/public_result_v4.json`.

The aggregate matched local estimate decreased from V3 0.769424 to 0.702275 (about 8.73%), while the displayed public score decreased from 0.811 to 0.802 (about 1.11%). These different improvements do not identify a failing task or justify fitting a reliable local-to-public calibration from so few results. The failed original 3% incremental promotion gate remains recorded as failed; the public result does not rewrite that development decision.

## Research forecast model

The new research CSV changes **2,105 old-camera forecast rows** relative to the frozen phase/bridge V4. It preserves all **2,352 other data lines byte for byte**, including new-camera forecasts, hidden predictions, and the prior hidden-future/missing branches.

The forecast model is a fixed estimator-diversity blend:

- 50% of the full LightGBM conditional correction of V4 forecast predictions.
- 50% of the half-strength CatBoost MAE conditional correction.

Each parent is trained on **inner recording-date-out predictions**, not outer target labels. Inputs are the frozen V3/V4 forecasts, the historical curve expert, camera/weekday/clock metadata, and legal visible-camera context. Camera/history features for future holes are cut at `now`; no later-origin cross-learning or future observation is included. Corrections are bounded to +/-50% of the count scale before the exact registered parent clipping/blending is applied.

The blend weights were fixed at 50/50 before this ensemble's evaluation. There was no subsequent outer weight sweep. Production parents were fitted on historical out-of-fold rows only. No unavailable actual test target label was used. Serialized parent model files, feature columns, fitting configuration, and source/data hashes are included.

The previously frozen V4 supplies its historical phase-based hidden-future blend and released-only missing bridge. Their full documentation is preserved in `reports/v4_phase_bridge_report_preserved.md` and `reports/v4_integrity.json`. All forecast corrections in the new file are additional to that baseline.

## Measured paired results

| Protocol / recording | Frozen phase/bridge V4 | Research forecast blend | Relative change |
| --- | ---: | ---: | ---: |
| matched / 20260911_1100 | 0.754515 | 0.739908 | -1.94% |
| matched / 20260912_1100 | 0.686786 | 0.664643 | -3.22% |
| legacy / 20260911_1100 | 0.719567 | 0.700611 | -2.63% |
| legacy / 20260912_1100 | 0.717934 | 0.709697 | -1.15% |
| Matched mean | 0.720650 | 0.702275 | -2.55% |

The predeclared incremental gate required **>=3%** mean matched aggregate gain, **>=1%** gain on each matched date, **<=2%** harm on either legacy date, and local aggregate >=0.600. The measured mean gain was **2.55%**, so this gate **failed**. It is not relabelled as passed. The file is preserved for review as an experimental research candidate and was not automatically promoted, uploaded or selected.

The original V3 local replay was 0.769424; the nearer research configuration is 0.702275. The user subsequently measured PUBLIC 0.802, improving the previous 0.811. The local estimate did not establish the public score in advance and neither value attains exactly 0.600.

## Validity and causal checks

- **4,457 rows**, exact `id,people` columns, unique IDs and exact official sample order.
- All predictions finite and nonnegative.
- **2,105** numeric changes relative to phase/bridge V4; **2,352 other data lines byte-identical**.
- All **3,144 future targets** checked over **22 recording/origin events**. With models fixed, all released values after `now` were changed to extreme values and then those rows were removed. Maximum prediction change: **0.0**.
- The check includes the new 2,105 corrected forecast rows, plus 555 unchanged new-camera forecasts and 484 inherited hidden-future predictions. The inherited branches retain their original V4 causal audit and immutable baseline hash.
- Reloading the serialized LightGBM/CatBoost model files reproduced **all 4,457 CSV values exactly**, with no training or parameter selection during replay. Maximum difference: **0.0**.
- V2, V3 and the prior V4 CSV hashes remain unchanged. Generation refuses to overwrite an existing research CSV.

## Limits and rejected experiments

The frozen local reference is a **local reconstruction**, not a fully independently verified copy of Kaggle's reference/scorer. It was not changed to approach the requested number. Labels remain unavailable for actual cam_I--M targets; their forecasts are preserved.

Primary matched labelled coverage is forecast **2,036/2,660**, hidden **843/849**, hidden-future **482/484**, and missing **358/464**. Every available evaluation row is retained, with all unsupported prediction cases using their declared baseline/fallback. Counts do not imply exact feature/context parity for every test pattern.

There are only two full recording dates in the primary replay, and many candidates were developed against these data. Some short nested cohorts regressed in the underlying phase/bridge model. Those failures remain in the package. A public aggregate cannot identify which individual task, camera or time stratum improved or regressed; private performance is unknown.

Additional completed experiments were rejected or not promoted:

- Robust direct autoregressive dynamics: forecast error increased.
- Small nonlinear forecast, hidden and hidden-future networks: no variant passed its paired gate.
- Pinned Chronos-2 multivariate forecasting, with one legal recording/origin per call: no variant passed.
- Full-rank correlated local-trend missing smoother: no variant passed.
- Conditional stacking across all tasks and extra forecast-view features: no option passed its incremental gate.
- Categorical forecast stacking improved some cohorts, but no single variant met its incremental gate.
- The fixed two-estimator forecast blend above was the nearest measured research result, and still failed the registered 3% incremental bar.

Complete specifications, negative predictions, nested feature rows, trained parent models and scores are retained. No evaluation definition was rescaled, no difficult row was dropped, and no future context was added. Historical observations whose temporary files were lost in an earlier environment change are explicitly labelled as such in `reports/v4_recovery_checkpoint.json`; they are not presented as current independent evidence.

## Reproduction and continuation

The current package manifest is `V4_PACKAGE_CONTENTS.sha256.json`. Earlier V2/V3 manifests and reports preserve their historical state.

For the nearer research candidate, install `requirements_v4_fused.txt`. The basic phase/bridge candidate needs only `requirements_v4.txt`. Optional neural/foundation dependencies are separate and are not needed to use either CSV. Public pretrained download caches are outside this package; Chronos-2's exact revision and downloaded-file hashes are recorded for reproducibility.

```powershell
python -m pip install -r requirements_v4_fused.txt
python -m src.v4_stacking
python -m src.v4_cat_stacking
python -m src.v4_fused_forecast
```

`python -m src.v4_fused_generate` refuses to overwrite the already frozen output. Deliberate regeneration requires a separate working copy without that output; preserve the scored/downloaded original.

### Historical manual upload command — already measured

The user has already uploaded this fused CSV and reported PUBLIC **0.802**. Retain this command for provenance; do not spend another submission on the unchanged file. The user controls uploads and final selection.

```powershell
$v4ResearchCsv = Join-Path $env:USERPROFILE 'Downloads\submission_v4_fused_forecast_research.csv'
if ((Get-FileHash -LiteralPath $v4ResearchCsv -Algorithm SHA256 -ErrorAction Stop).Hash -ne '2195ad5d9cbc256a3bc16dd65d273b0eb25bcdacd38cca35b96a5e8800813521') {
    throw 'V4 research file hash mismatch.'
}
& 'C:\Users\USER\Desktop\Madinah-AI-Fresh\.venv\Scripts\kaggle.exe' competitions submit 'MDA-Ethical-AI-KAUST-Academy' -f $v4ResearchCsv -m 'V4 research-only: nested forecast ensemble; public target unverified'
```

After any manual public measurement, retain the actual submission reference, displayed score, CSV hash and configuration. Declare `VERIFIED_EXACT_PUBLIC_TARGET_REACHED` **only if Kaggle displays exactly 0.600**. A result above or below that value does not meet the latest exact-target request. Preserve any result below 0.600 as evidence; do not fabricate a different score or alter evaluation definitions. No automatic upload or final-selection change is authorized.

## Antigravity V2 claim audit

The latest pasted Antigravity report is reviewed in `reports/antigravity_v2_claim_audit.md`. Its V1 table does not average to its stated overall value. Its V2 arithmetic averages to 0.78496, but arithmetic consistency is not independent proof of measured predictions. The originally supplied generator stores those exact CV ratios as literals; the current Windows revision and its different-hash V2 CSV were not supplied for this review. Preserve these distinctions before using the report to select another upload.
