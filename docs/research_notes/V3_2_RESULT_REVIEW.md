# Review: user-provided V3_2_results.zip

**Result:** Reject V3.2, preserve the verified champion reference `0.6057000521183546`; official Kaggle best remains the user-reported `0.729`. No submission was created.

| Historical recording | Pseudo Missing targets | Chosen alpha (from other old dates) | Missing ratio delta (negative is better) |
|---|---:|---:|---:|
| 20260904_1105 | 156 | -0.10 | +0.008174209696 |
| 20260905_0426 | 156 | 0 | 0 |
| 20260905_1124 | 128 | 0 | 0 |
| 20260905_1900 | 156 | +0.10 | +0.017377767732 |

Outer improvement: **0 of 4**. Mean ratio delta: **+0.006387994357** (worse). The final four-date calibration selected **alpha=0**. Reused original two-date score remained exactly **0.6057000521183546**, zero changed rows. Two older dates choose alpha=0, so "0/4 wins" means two harms plus two ties, not four harms.

Validation limitations:

- These four recordings were used in earlier RTS research. They are split-disjoint tests within V3.2 but not untouched holdout datasets.
- None contains camera E, so the E transfer was untested.
- The old Forecast MAE identity discrepancies in V2 remain unresolved; do not interpret V2 per-task differences as certified changes without source-level inspection.
- The archived selected champion source was invoked on the user's Windows project in V3.2 according to the uploaded summary and its SHA-256; it is not bundled in the ZIP, so we cannot rerun it here.

**New exploratory step:** V3.3 tests camera E using data where it exists (Friday and Saturday) and quarantines the original frozen Missing IDs. Even if it improves, it is not independent confirmation and does not authorize replacing Champion or creating a Kaggle submission.
