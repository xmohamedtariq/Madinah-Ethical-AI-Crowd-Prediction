# V6 PUBLIC 0.749 recorded — V7 target 0.400 pending

**V6 subsequently achieved user-confirmed Kaggle PUBLIC 0.749**, improving V5 0.766 by **0.017**. V7 continues toward 0.400.

Historical baseline: **V5 achieved user-confirmed Kaggle PUBLIC 0.766**, improving V4 0.802 by **0.036**, or about **4.49%**. That scored file and hash are preserved. The user now requests V6 toward **0.400**. No 0.400 result has been reached or guaranteed.

**V6 is a frozen experimental CSV**, with paired matched development score **0.682947036**, versus V5 **0.689864519**: approximately **1.00%** local aggregate improvement. The original component and system promotion gates **failed**. This is a modest experimental improvement for optional public measurement, not a promoted candidate or a prediction of 0.400. No assistant upload or final-selection change occurred. No experiment is claimed to keep running after this checkpoint.

## File and reproduction

- `submission_v6_dual_kernel_gp.csv`; SHA-256 **`301a58589c5bc38ae466857434118af2878fec6f9f08aceba38bd68c8e6ebcac`**.
- 4,457 rows; exact sample columns, unique IDs and order; finite nonnegative values.
- 464 numerical changes, all within the 464 missing rows.
- All **3,993 other data lines** remain byte-identical to scored V5, including all **3,144 future predictions**. Future-context checks are inherited through unchanged V5/V4 artifacts; the V6 component only reconstructs missing holes with permitted full released context.
- Both fitted GP families were serialized and reloaded. Every changed production value was reproduced exactly in a separate process without fitting or hyperparameter reselection.
- Scored V5 SHA-256: `161a541062cdf6d59f5895d484d81e124279a0a481933fb6498f74b77f7c113d`. Remote submitted bytes, submission reference, exact timestamp and private score were not independently retrieved. PUBLIC 0.766 is based on the user's Kaggle card.

## Structural experiments

A geometry audit measured actual missing endpoint spans of **6--20 minutes**, typically 10--12 minutes. V6 tested nonlinear peer-state kernels, locally conditioned GPs, robust nonlinear kernels for hidden/hidden-future cameras, and alternative temporal smoothness. All masks, labels, reference predictions, denominators and retained evaluation rows remained unchanged.

The final single fixed ensemble is:

**0.50 V5 + 0.25 two-scale scene GP + 0.25 smoothness-selected GP**.

The scene GP combines Matern-3/2 temporal components at a selected 3/10/30-minute scale plus a fixed 60-minute scale, a constant prior, and a nonlinear peer-state covariance. The second GP selects among Matern-3/2, Matern-5/2, squared-exponential and exponential temporal kernels, with optional linear peer covariance. Both use released-only exact block-deletion calibration; geometry-matched self-mask lengths are 6/10/16 minutes. Each averages its best three fixed-grid calibration configurations and falls back to interpolation if calibration gain is below 5%. Peer features exclude the target camera.

The two 50/50 parent blends were chosen for complementary covariance behavior. Their equal ensemble weights were frozen before evaluating this ensemble, with no subsequent weight sweep. Parents were screened on the same already inspected outer dates, so the final score is selection-biased development evidence. No actual withheld test labels were used.

The direct nonlinear hidden-camera models and localized GP conditioning were rejected for worse or inconsistent paired error. The source, fitted parameters, all negative results and specifications are included. The four families evaluated 26 declared variants; one additional fixed ensemble was then evaluated. No past gate was silently changed.

## Paired results against scored V5

| Protocol / recording | V5 missing MAE | V6 missing MAE | Change | V5 aggregate | V6 aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched / 20260911_1100 | 6.490131 | 6.063432 | -6.57% | 0.719125 | 0.706722 |
| matched / 20260912_1100 | 4.351555 | 4.319645 | -0.73% | 0.660604 | 0.659172 |
| legacy / 20260911_1100 | 8.232888 | 7.818968 | -5.03% | 0.681509 | 0.672881 |
| legacy / 20260912_1100 | 7.123235 | 6.686002 | -6.14% | 0.695303 | 0.685599 |

The missing-task gate required average matched MAE improvement >=5%, each matched date >=1%, and legacy harm <=2%. The actual matched gains were **6.57% Friday and 0.73% Saturday**, so the gate failed. The system gate required >=3% average aggregate improvement and >=1% on each matched date; it also failed. The CSV is preserved explicitly as **EXPERIMENTAL_RESEARCH_ONLY_NOT_PROMOTED**.

These scores use the unchanged local reconstruction of reference predictions. They do not certify a complete copy of Kaggle's official scorer/reference. Matched labelled coverage stays forecast **2,036/2,660**, hidden **843/849**, hidden-future **482/484**, missing **358/464**: 3,719 matched labelled positions. File validation on 4,457 output rows is a different claim.

## Unseen-camera proxy study and limitations

The same 77 pseudo-new-camera cohorts used for V5 were replayed without changing cohort selection or baselines. The donor camera is absent from historical fitting and old-camera released peers; only the actual new-camera release mask is mapped onto its known historical labels. The V5 proxy predictions are preserved and compared row by row.

| Date | Cohorts | Proxy target positions | Mean paired MAE change vs V5 | Wins |
| --- | ---: | ---: | ---: | ---: |
| 2026-09-11 | 37 | 320 | -5.39% | 26/37 |
| 2026-09-12 | 40 | 353 | -3.04% | 24/40 |

These 673 proxy positions reuse labels across templates and are not independent observations. Actual I--M target labels remain unavailable. Some cohorts regressed; the worst proxy ratios were approximately 1.160 and 1.178. Improvement on each actual test camera is not established.

Earlier short inner cohorts also regressed: approximately **+5.74%** missing MAE on early Friday and **+1.23%** on short Saturday midday; evening improved about **1.70%**. The failures remain in `reports/v6_hybrid_stress/`. The original V5 nested regressions are also preserved.

Conditional 30-minute block bootstrap intervals, shared across cameras, for missing-MAE change (negative favors V6):

| Comparison | Observed MAE change | Conditional 95% interval |
| --- | ---: | --- |
| matched_20260911_1100 | -0.426699 | [-0.734655, -0.217215] |
| matched_20260912_1100 | -0.031910 | [-0.181037, +0.109357] |
| legacy_20260911_1100 | -0.413919 | [-1.021256, +0.042461] |
| legacy_20260912_1100 | -0.437233 | [-0.762963, -0.186683] |

These intervals condition on repeatedly inspected dates and are not independent confirmation, a future-day guarantee or a private-score estimate. A few public aggregate scores cannot support reliable task attribution or a local-to-public calibration. The result does not establish a large move toward 0.400.

## Reproduce without submitting

Extract into a new folder and preserve all scored CSVs. Use Python 3.12 and:

```powershell
python -m pip install -r requirements_v6.txt
python -m src.v6_finalize verify
```

This loads saved GP parameters and exactly reproduces the CSV. Generation refuses to overwrite the frozen output. Deliberate regeneration requires a separate working copy without the V6 output. `V6_PACKAGE_CONTENTS.sha256.json` is current; older reports and manifests preserve their historical state.

## Historical manual command — V6 already scored

The user has already uploaded this exact-named V6 file and reported PUBLIC **0.749**. Do not reupload unchanged V6. The command is preserved for provenance; the original promotion gates remain failed. No 0.400 or private result is established.

```powershell
$v6File = Join-Path $env:USERPROFILE 'Downloads\submission_v6_dual_kernel_gp.csv'
if ((Get-FileHash -LiteralPath $v6File -Algorithm SHA256 -ErrorAction Stop).Hash -ne '301a58589c5bc38ae466857434118af2878fec6f9f08aceba38bd68c8e6ebcac') {
    throw 'V6 file hash mismatch.'
}
& 'C:\Users\USER\Desktop\Madinah-AI-Fresh\.venv\Scripts\kaggle.exe' competitions submit MDA-Ethical-AI-KAUST-Academy -f $v6File -m 'V6 experimental: dual-kernel released-only gap reconstruction'
```

Record the actual Kaggle result, uploaded file hash, submission ID and configuration. Keep V5 if V6 is worse. The 0.400 goal remains PENDING; uploads and final selections remain manual.

## Recorded user result

See `reports/public_result_v6.json`: user-provided Kaggle recent-submission card reports **0.749**, previous best **0.766**. Local artifact hash was verified unchanged. Remote bytes, exact submission reference/time and private score were not fetched. The latest V7 round did not produce a recommended new submission and preserved both user-reported remaining attempts.
