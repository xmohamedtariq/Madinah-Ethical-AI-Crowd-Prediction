# Madinah Ethical AI — V8 experimental submission audit

**File:** `submission_v8_hidden_median17.csv`  
**SHA-256:** `85eb544cbb1886b973604094d08b82686857f56819caa4fc4de4b24068903158`  
**Submission format:** `id,people` — 4,457 rows, official sample order, exact IDs, unique IDs, all values finite and nonnegative.

## What was changed

We recovered the official `train.csv`, `test.csv`, `sample_submission.csv` and the saved V7 research artifacts from the user's file library. We verified the V7 input and all three dataset SHA-256 hashes. Only the **hidden** target class was postprocessed: a centered 17-sample median on each camera's contiguous prediction run (camera and recording isolated; no gap-crossing). The test labels were never used. This operation is legally restricted to the noncausal `hidden` task; all other rows are copied byte-for-byte from verified V7.

- Hidden: 849 output rows, 362 numeric changes.
- Forecast: 2,660 rows, **0 changes**.
- Hidden-future: 484 rows, **0 changes**.
- Missing: 464 rows, **0 changes**.
- Unchanged nonhidden source rows: 3,608, byte-identical.

## Local validation (reconstructed reference, not official Kaggle score)

| Protocol | Historical recording | V7 | V8 | Delta (V8–V7) |
|---|---|---:|---:|---:|
| matched | 20260911_1100 | 0.670869 | 0.669002 | -0.001867 |
| matched | 20260912_1100 | 0.627509 | 0.625668 | -0.001841 |
| legacy | 20260911_1100 | 0.639237 | 0.635823 | -0.003414 |
| legacy | 20260912_1100 | 0.655217 | 0.654222 | -0.000995 |

Mean **matched-mask V8 local score: 0.647335364** vs V7 **0.649188995** and historical V6 **0.682947036**. The V8 vs V6 matched improvement is **5.214%**. These dates were used during the preceding development; they are not independent validation. Smoothing width was selected after inspecting these same dates, so this gain can overstate generalization.

### Limits

The submitted user's best historically reported Kaggle public score was **0.749** for V6. **V8 has not been submitted and has no certified public or private Kaggle result.** The user goal of **0.400 or below has not been verified or achieved locally**. Neither the private ground truth nor the official scoring denominator is available. The local score uses a historically reconstructed reference; the actual public score could be better or worse. The 52% public leaderboard may differ from final 48% private evaluation.

**Data-use integrity:** The only changed predictions are hidden-camera outputs, for which released full-recording context is permitted. Forecast and hidden-future outputs inherited from the V7 audited causal model remain byte-identical. No unknown test labels were inferred or leaked.

**Recommendation:** This is a single, formatted experimental candidate, not a guaranteed improvement over the 0.749 public best. Preserve the original V6 file for final submission selection if this V8 attempt performs worse.
