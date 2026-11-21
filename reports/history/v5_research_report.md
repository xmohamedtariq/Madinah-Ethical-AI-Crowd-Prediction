# V5 — PUBLIC 0.766 recorded; V6 continues toward 0.400

**جهزنا V5 بملف جديد قابل لإعادة الإنتاج.** السكور المحلي المختار انخفض من **0.702275 إلى 0.689865**، أي تحسن نسبي **1.77%**. أفضل نتيجة Kaggle عامة مؤكدة الآن **V5 = 0.766**، وفق نتيجة المستخدم، بتحسن **0.036** عن V4 = 0.802. لم نصل إلى 0.500؛ النتيجة الخاصة غير معروفة. الملف حقق أفضل نتيجة عامة لدينا؛ بوابة التحسن الإجمالي المحلية الأصلية 3% بقيت غير متحققة.

The user changed the objective to making the next upload closer to **0.500**. This supersedes the former exact-0.600 objective and local floor. It does not change any evaluation mask, denominator, label or baseline prediction. The user subsequently uploaded V5 and reported PUBLIC **0.766**. The latest V6 objective is PUBLIC **0.400**, superseding 0.500. Goal remains pending. No assistant upload or final-selection change occurred. This local work round is finished; no experiment is claimed to keep running in the background.

## Frozen submission

- File: `submission_v5_adaptive_gap_gp.csv`
- SHA-256: `161a541062cdf6d59f5895d484d81e124279a0a481933fb6498f74b77f7c113d`
- 4,457 rows; exact official `id,people` columns, unique IDs and sample order.
- All values finite and nonnegative; minimum 2.410525, maximum 969.918521.
- Exactly **464 missing-task predictions** changed. All **3,993 other CSV data lines** remain byte-identical to scored V4.
- All **3,144 future predictions** are inherited unchanged from V4, whose fixed-model 22-event future mutation/deletion checks passed. The new component predicts only missing holes, where the supplied competition rule permits the full released table.
- Saved GP parameters/models were reloaded in a separate process and reproduced all 464 changed values **exactly**, with no parameter selection or fitting during verification.
- Scored V2, V3 and V4 hashes remain unchanged.

Baseline: `submission_v4_fused_forecast_research.csv`; SHA-256 `2195ad5d9cbc256a3bc16dd65d273b0eb25bcdacd38cca35b96a5e8800813521`. Its PUBLIC 0.802 is evidenced by the user's Kaggle card, not a fresh API read. Submission ID, absolute upload timestamp and private score remain unknown.

## Predictor and label separation

For every missing row, V5 uses **50% frozen V4 prediction + 50% released-only adaptive Gaussian-process prediction**. The GP combines a Matern-3/2 temporal covariance, a constant prior and an optional linear peer-camera covariance. Peer columns use a fixed schema determined from released support; the target camera is excluded from peer features.

Within each released recording/camera, candidate length scales 3/10/30 minutes, noise ratios 0.03/0.15/0.5, peer covariance strengths 0/3 and internal interpolation blends 0.5/1.0 are selected using deterministic self-masked blocks of 1/3/5/8/12 minutes. The exact block-deletion identity removes every count in each calibration block from its conditional prediction. This was checked against actual refitting: maximum discrepancy **2.16e-11** across four blocks/24 held observations. Selection uses count MAE on legally released self-masks; unavailable actual test labels are never read. If the GP does not improve self-masked linear interpolation by at least 5%, it falls back to linear interpolation. This internal calibration is not independent test evidence.

The external 50/50 blend was declared before outer scores. The original count GP was chosen over the transformed-count follow-up because it had the slightly better matched result and the completed nested/proxy stress evidence. Lower raw-GP exploratory scores were not used: raw GP worsened matched Saturday by about 1.26%.

## Paired local results

The protocols and their reference predictions are unchanged. Scores are the mean of recording scores, each a mean of the four task MAE/reference-MAE ratios. References are locally reconstructed; this report does not certify a complete copy of Kaggle's official scorer/reference.

| Protocol / recording | V4 missing MAE | V5 missing MAE | Missing error change | V4 aggregate | V5 aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched / 20260911_1100 | 7.205156 | 6.490131 | -9.92% | 0.739908 | 0.719125 |
| matched / 20260912_1100 | 4.441521 | 4.351555 | -2.03% | 0.664643 | 0.660604 |
| legacy / 20260911_1100 | 9.149324 | 8.232888 | -10.02% | 0.700611 | 0.681509 |
| legacy / 20260912_1100 | 7.771834 | 7.123235 | -8.35% | 0.709697 | 0.695303 |

Matched mean: **0.702275258 → 0.689864519**. Legacy mean: **0.705154345 → 0.688405813**.

The missing-task component gate passed: at least 5% average paired matched MAE gain, at least 1% on each matched date and no more than 2% legacy harm. The system gate did **not** pass: mean aggregate gain was only 1.77%, below 3%, and matched Saturday aggregate gain was below 1%. This has not been relabelled as success. V5 was originally preserved as an experimental candidate and subsequently measured PUBLIC 0.766. Do not reupload unchanged bytes.

Primary matched labelled replay coverage remains forecast **2,036/2,660**, hidden **843/849**, hidden-future **482/484**, missing **358/464**: **3,719** matched labelled target positions, not 4,457 independently labelled test cases. All available evaluation rows are retained. Unsupported genuine new-camera targets do not acquire invented truths.

## New-camera missing proxy study

For each historical donor camera, its entire identity was removed from historical training and old-camera released peers. Only the selected actual I--M camera's release/target mask was mapped onto that donor. The comparison baseline uses the preserved V2/V3 `learned_half` missing algorithm fitted without the donor's historical labels, matching the branch still used by scored V4 for genuinely new cameras. Candidate weights were fixed at 50/50 before this study.

| Held date | Donor/template cohorts | Labelled proxy target positions | Mean paired MAE change | Winning cohorts |
| --- | ---: | ---: | ---: | ---: |
| 2026-09-11 | 37 | 320 | -5.19% | 24/37 |
| 2026-09-12 | 40 | 353 | -5.24% | 27/40 |

This covers **77 cohorts and 673 proxy target positions**, with labels reused across camera templates; they are not 673 independent observations. Three nominal cohorts had insufficient released support and were recorded through the coverage difference rather than invented. Actual I--M target labels remain unavailable. Some cohorts worsened; the worst ratio was approximately 1.884 on Friday. All ten date/template average comparisons favored the half-GP rule, but individual improvement is not guaranteed. Full cohort predictions and counts are included in `reports/v5_gp_stress/`.

## Important negative evidence and uncertainty

Shorter earlier inner cohorts did not uniformly improve. Missing MAE changed **2.58731 → 3.28157** on the early Friday recording, **2.14555 → 2.31549** on a short Saturday midday recording and **6.22032 → 5.68144** on the evening recording. These retained failures limit generalization confidence; they are not dropped from the saved study.

The primary dates have already been inspected in many experiments. Local scores are development estimates subject to selection bias, not fresh independent confirmation. Conditional paired 30-minute block bootstraps, shared across cameras, give the following missing-MAE differences; negative favors V5:

| Comparison | Observed difference | Conditional 95% interval |
| --- | ---: | --- |
| matched_20260911_1100 | -0.715025 | [-1.328740, -0.310587] |
| matched_20260912_1100 | -0.089966 | [-0.332984, +0.112486] |
| legacy_20260911_1100 | -0.916436 | [-1.656305, -0.422797] |
| legacy_20260912_1100 | -0.648599 | [-1.303255, -0.169501] |

Matched Saturday still includes possible harm. These intervals are conditional on the inspected dates and cannot measure unseen-day or private-leaderboard uncertainty. The aggregate PUBLIC score alone cannot identify task/camera contributions. We did not fit a local-to-public conversion from the few leaderboard observations, alter the metric, retrieve withheld labels or invert sample values into target labels.

## Completed experiments

Eight experiment families and 54 declared outer variants were evaluated. The table reports each family's lowest exploratory matched aggregate, which can include a rejected variant; it is not a recommended combined configuration.

| Family | Outer variants | Lowest exploratory matched score | Component-gate passes | Final use |
| --- | ---: | ---: | --- | --- |
| v5_spatial | 12 | 0.699258 | None | Not used |
| v5_masked_forecast | 6 | 0.702626 | None | Not used |
| v5_gap_gp | 2 | 0.686820 | 0.5 | Half GP selected |
| v5_lagged_spatial | 12 | 0.700712 | None | Not used |
| v5_quantile_spatial | 12 | 0.704326 | None | Not used |
| v5_warped_gp | 2 | 0.687242 | 0.5 | Not used |
| v5_camera_forecast | 6 | 0.699610 | None | Not used |
| v5_causal_gp_forecast | 2 | 0.730212 | None | Not used |

The camera-relation, lagged/masked spatial, regularized median, mask-augmented forecast, camera-aware forecast and causal-GP forecast families were rejected because gains were inconsistent or error worsened. The transformed-count GP passed the component gate at half strength, but did not beat the selected count GP and lacks that rule's completed proxy stress study. Interrupted index-check runs and their repair notes remain preserved; no frozen evaluation data were changed to repair them.

## Reproduce without uploading

Extract the archive into its own new directory, preserve the scored project and downloaded CSV, then use Python 3.12:

```powershell
Set-Location 'C:\Users\USER\Desktop\Madinah_V5_research'
python -m pip install -r requirements_v5.txt
python -m src.v5_finalize verify
```

Verification loads saved GP parameters and checks exact CSV reproduction, format and immutable V4 hash. It does not retrain or upload. `python -m src.v5_finalize generate` deliberately refuses to overwrite the frozen V5 CSV. Deliberate regeneration should use an isolated copy without the V5 output. `V5_PACKAGE_CONTENTS.sha256.json` is the current package manifest; earlier reports/manifests are historical.

## Historical manual command — V5 already scored

V5 has already been manually scored at **0.766**. Do not reupload the unchanged CSV. The following command is retained for provenance:

```powershell
$v5File = Join-Path $env:USERPROFILE 'Downloads\submission_v5_adaptive_gap_gp.csv'
if ((Get-FileHash -LiteralPath $v5File -Algorithm SHA256 -ErrorAction Stop).Hash -ne '161a541062cdf6d59f5895d484d81e124279a0a481933fb6498f74b77f7c113d') {
    throw 'V5 file hash mismatch.'
}
& 'C:\Users\USER\Desktop\Madinah-AI-Fresh\.venv\Scripts\kaggle.exe' competitions submit MDA-Ethical-AI-KAUST-Academy -f $v5File -m 'V5 experimental: released-only adaptive gap GP on scored V4'
```

After the user supplies the actual result, record the submission reference, displayed score, file/configuration/hash and any available timestamp. Preserve V4 if V5 is worse. Continue development from the best measured public candidate toward 0.500. A local estimate or successful file validation does not satisfy the public target. No final selections are changed automatically.

## User-confirmed public result

The user supplied Kaggle's recent-submission card naming `submission_v5_adaptive_gap_gp.csv`, PUBLIC **0.766**, and the new-personal-best message versus **0.802**. The local file SHA-256 remains `161a541062cdf6d59f5895d484d81e124279a0a481933fb6498f74b77f7c113d`. Remote bytes, submission ID, absolute timestamp and private score were not independently retrieved. See `reports/public_result_v5.json`. The development gates remain recorded as failed; the public observation does not retroactively change them. The user next requested V6 toward 0.400.
