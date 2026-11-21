# V2 public result and V3 continuation — 17 November 2026

## Actual public result

The user supplied Kaggle leaderboard text showing **submission_v2_independent.csv scored 0.812**, a new personal best compared with 0.886. The displayed improvement is **0.074**, approximately **8.35%** relative to the previous displayed best. The remaining displayed gap to 0.600 is **0.212**. The user snapshot showed rank 195 and 18 entries; these are snapshot values, not a live leaderboard refresh.

The source is the user's Kaggle platform result. The account was not refreshed via API. A numeric submission reference and exact submission timestamp were not provided. Private score is unknown. V2's scored file is preserved with SHA-256 `2fa4af7e3e8c1d56c5bf4ff8daab415e4f963a0cea10cd172495a17439da557d`.

**Best confirmed public result in this project is now 0.812. The <=0.600 goal remains PENDING.**

The aggregate public result does not identify which tasks or strata caused the improvement. It was not used to change masks, reference denominators, evaluation definitions, or model weights, and was not used to invent a public-score prediction.

## V3 experiments

### New-camera-only forecast transfer

The model removes the ordinal camera identifier and normalizes level/motion features. It is applied only to forecast queries whose camera had no rows in the original historical train table. As in V2, training may include legally released rows; this is not a claim of zero labelled release context.

The paired stress test removes each mapped historical camera identity from the fit history, then simulates cam_I's actual release/target schedule. V2's exact horizon damping was reconstructed from its saved normalized-model predictions and legal persistence reference. Candidate and baseline use identical labelled target rows.

| Labelled proxy date | V2 mean normalized MAE | Transfer mean normalized MAE | Relative change |
|---|---:|---:|---:|
| 2026-09-11 | 0.918813 | 0.763465 | -16.91% |
| 2026-09-12 | 0.900499 | 0.889614 | -1.21% |

Transfer won **7 of 10** camera/date cohorts. The registered screening rule required at least 1% mean gain on each date and at least 60% cohort wins; it passed.

These are reused labelled historical proxy cohorts, not independent confirmation. Some individual cohorts worsened, including Friday cam_E and Saturday cam_A/cam_C. The stress uses cam_I geometry; it does not fully establish equivalence to every actual cam_L/cam_M context. Actual I--M target labels remain unavailable. Consequently no V3 public gain is established.

### Released-label spatial adaptation

Fitting hidden and hidden-future parameters on additional released labels improved a separate joint-date holdout stress. Both long recording dates were removed from historical training, and only their masked released rows were admitted to fitting. All their target positions stayed excluded. The new joint snapshot was frozen before scoring; existing matched and legacy masks/reference values were retained.

| Protocol | V2 local score | Released adaptation | Combined variant |
|---|---:|---:|---:|
| Existing matched masks | 0.769424008 | 0.770402652 | 0.770402652 |
| Existing legacy masks | 0.793102623 | 0.792198702 | 0.792198702 |
| Additional joint-date stress | 0.889974420 | 0.856121169 | 0.857471188 |

Released adaptation passed the registered permissive screen: at least 1% aggregate gain on both joint recordings, no greater than 2% aggregate harm or 5% changed-task harm on either primary recording. It was **deferred despite passing that screen**, because the matched-mask aggregate worsened on both dates and the joint stress has much less historical support than the actual deployment. The next candidate isolates the new-camera change. This is a promotion decision, not a retroactive rewrite of the screening rule.

New-camera-only primary matched and legacy predictions remain exactly V2-identical: those target rows all have historical camera support. Their unchanged scores are not evidence that actual new-camera performance improved. In the additional joint stress cam_E becomes a pseudo-new camera; the transfer-only aggregate there changed from 0.889974420 to 0.891324439, a small worsening. This is another reason confidence in a public gain is limited.

## V3 experimental CSV

- File: `submissions/submission_v3_new_camera_transfer.csv`
- SHA-256: `159aa8d53b78120e167b5bb0d8f865c810f542330c6753f877212f03c96c4713`
- Valid rows: 4,457; exact id,people columns, exact sample order, no duplicate IDs, non-finite values or negative counts.
- **Exactly 555 predictions change**, all forecast rows for cameras I--M. The other **3902 CSV data lines are byte-identical to the scored V2 file**.
- Friday new-camera rows: 255. Saturday new-camera rows: 300.
- Hidden, hidden-future, missing and historical-camera forecast predictions retain the scored V2 values.
- Replaying the V2 model before switching the new-camera branch reproduced every scored V2 prediction exactly (maximum difference 0 on both recordings).
- All 3,144 actual future targets passed counterfactual tests with fitted models fixed: changing future released counts or removing all rows after now changed no predictions.
- No automatic upload or final-selection change was performed. V3 public score is unknown.

Current scripts use relative project paths. `src/v3_trials.py` contains the registered screening plan, paired evaluations and the additional joint stress. `src/v3_generate.py` generates the scoped candidate and checks exact V2 reproduction/scope/causality. The V2 source files and scored submission are preserved.

The historical V2 documents describe their state before the user upload. This report and `reports/public_result_v2.json` supersede their earlier public-score-unknown statement. They do not change V2's CSV bytes or historical local measurements. Local metric/reference parity limitations documented in V2 still apply.

## Manual next step

A manual V3 upload would be an **experimental measurement**, not a submission with a guaranteed score. Its validation rationale is limited to the stated camera-transfer proxy evidence. There is no claim that this incremental candidate reaches 0.600.

After a user-provided public result, record the numeric submission reference when available, score, filename and hash. If the public score is above 0.600, continue research with the best confirmed public file preserved. Only an actual qualifying public result can satisfy the public target; private score remains a separate unknown.

Checkpoint: `reports/v3_checkpoint.json`. No background model job remains running. Exact manual PowerShell upload instructions are in `README_V3.md`.
