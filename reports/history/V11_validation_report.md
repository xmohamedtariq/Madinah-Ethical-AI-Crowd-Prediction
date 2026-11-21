# Madinah Ethical AI — V11 validation and submission audit

**Status:** Experimental Kaggle CSV ready for **manual public evaluation**. Local results are development-only and do not establish the private or public Kaggle score. No result at or below 0.400 has been verified.

## 1. Measured changes

| Revision | Matched-mask development score | Relative reduction against V6 |
|---|---:|---:|
| V6 | 0.682947036 | baseline |
| V7 | 0.649188995 | 4.943% |
| V8 | 0.647335364 | 5.214% |
| V9 | 0.646893026 | 5.279% |
| V10 | 0.646259908 | 5.372% |
| **V11** | **0.645284723** | **5.515%** |

V11 vs V10: absolute change **-0.000975186** (smaller is better), **0.151% relative improvement**. Compared with the Kaggle target 0.400, the difference remains **0.245285** on this **local, non-officially verified** metric.

### Per-fold paired comparisons

| Protocol | Recording | V10 | V11 | Change |
|---|---|---:|---:|---:|
| Matched test-like mask | 20260911_1100 | 0.668012320 | 0.667501795 | -0.000510525 |
| Matched test-like mask | 20260912_1100 | 0.624507497 | 0.623067651 | -0.001439846 |
| Legacy mask | 20260911_1100 | 0.635011921 | 0.634802933 | -0.000208988 |
| Legacy mask | 20260912_1100 | 0.653522302 | 0.653414942 | -0.000107360 |

All four comparisons improved. But matched vs legacy reuse much of the **same target days and labels**. They are *not* four independent out-of-time tests. The only matchable labels cover 3,719 of 4,457 official test-mask positions; the rest are not validated locally. The local reference-denominator reconstruction has **not been verified identical to the private Kaggle scorer**.

### Hidden-camera slices

| Protocol | Camera | Rows | V10 hidden MAE | V11 hidden MAE |
|---|---|---:|---:|---:|
| Matched Friday | cam_B | 284 | 4.837018 | 4.821493 |
| Matched Friday | cam_G | 276 | 2.873491 | 2.869249 |
| Matched Saturday | cam_D | 283 | 4.306954 | 4.263509 |
| Legacy Friday | cam_B | 285 | 4.753970 | 4.750715 |
| Legacy Friday | cam_G | 281 | 2.862898 | 2.857001 |
| Legacy Saturday | cam_D | 283 | 4.947742 | 4.944310 |

A conditional 30-minute block bootstrap on the reused matched days gives an approximate 95% interval of **[-0.001807, -0.000261]** for the V11–V10 score delta, **holding denominator values fixed**. This does **not** measure independent/generalizable forecasting uncertainty, and it is optimistic after selecting the method on those same days.

## 2. Research attempts and negative evidence

Experiments run in this continuation:

- 105 time-aware, causal forecast correction candidates using prior observed levels, robust trends, and decays: the best lower mean score increased errors on Friday and/or legacy folds, so no forecast correction was promoted.
- Cross-camera hidden reconstruction via historical-only Ridge / LightGBM (7- and 15-leaf models), fitted excluding the evaluated recording date. Full replacement regressed on hidden errors; blending harmed at least one evaluated date. No extra historical model is included in V11.
- 100+ hidden time/run smoother candidates (real-time kernel regression, medians, Gaussian smoothing, Savitzky–Golay). Only a subset consistently beat V10.
- Repeated polynomial smoothing (varying width, blend and passes) revealed that **two Savitzky–Golay passes** on V10 hidden-only estimates consistently improve all four reused development comparisons. Additional passes had negligible added gain and were rejected as unnecessary complexity.

These screens involved repeated inspection of the same two days; the lower local score must be treated as **selection-biased**.

## 3. Exact competition artifact

- **File:** `submission_v11_hidden_secondpass.csv`
- **SHA-256:** `1c4c537079cd11273f95612acfa7af7765a39e00c8a96726f2b6e7d3cc17c9e0`
- **Rows / columns:** 4,457 data rows, `id,people`, exactly official sample IDs and order.
- **Finite nonnegative values:** all pass; unique IDs: pass.
- **Source file:** `submission_v10_temporal_ensemble.csv`; SHA-256 `9e1237ac258cca66b3ee4ad5cd3916a41c12c163dcf81c878aa0be9701171efc`.
- **Changes:** 843 predictions in `hidden`. Six other hidden predictions remain unchanged. All **3,608 non-hidden lines** are byte-identical to V10. All **2,660 forecast** and **484 hidden-future** predictions stay unchanged, as do the **464 missing** predictions.
- **New method:** two polynomial-order-2 Savitzky–Golay smoothing passes on each contiguous hidden target run from V10, window capped at 31; uses **only pre-existing predictions of the same camera**. No future observed rows or withheld test labels enter the new processing.

Generation: `python V11_build_source.py`. Validation: `python V11_validate_source.py`. Both scripts verify checksums of the frozen V10 file and/or official data. Neither invokes Kaggle or reads test target labels.

## 4. Decision

**Manual Kaggle experiment recommended only if a submission attempt is available.** Preserve the earlier public best (0.749) and both eligible final-submission slots until new public evidence supports changing them. No Kaggle upload was executed. V11 is a modest experimental step toward, *not an achievement of*, the target 0.400. The main bottleneck remains forecast error on camera A/E/F and unseen-camera coverage; this smoothing change does not address those tasks.

**Research links:** Kaggle competition documentation (https://www.kaggle.com/docs/competitions), short-term forecast evaluation (https://pmc.ncbi.nlm.nih.gov/articles/PMC5870978/). These are methodological guidance, not external validation of this submission.
