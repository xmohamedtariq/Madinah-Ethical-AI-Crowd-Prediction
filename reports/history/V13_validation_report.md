# Madinah Ethical AI — V13 Research, Validation & Submission Audit

**Status:** Complete experimental CSV ready for **manual Kaggle submission**. No Kaggle upload has been made. No official score has been verified for V13. The target **0.400 or lower remains unmet**.

## 1. Fixed development results

| Evaluation protocol | Recording | Frozen V11 | New V13 | Absolute score change |
|---|---|---:|---:|---:|
| matched | 20260911_1100 | 0.667501795 | **0.658949159** | -0.008552635 |
| matched | 20260912_1100 | 0.623067651 | **0.618783982** | -0.004283669 |
| legacy | 20260911_1100 | 0.634802933 | **0.624332292** | -0.010470642 |
| legacy | 20260912_1100 | 0.653414942 | **0.645649979** | -0.007764963 |

**Matched mean:** V11 `0.645284723` → V13 **`0.638866571`**, a **0.995%** relative reduction, improvement `0.006418152`. Legacy mean: `0.644108938` → `0.634991136`. All four development comparisons improved.

**Critical caveat:** The matched and legacy folds are two masks applied to September 11 and 12. They are NOT four independent recording days. Hyperparameters were selected by extensive prior reuse of those days. The scorer is a reconstructed local average of four MAE/reference-MAE ratios and has not been officially certified as Kaggle-equivalent. This score cannot be guaranteed on the actual test data.

### Per-task development ratios (matched folds)

| Kind | V11 mean ratio | V13 mean ratio | Change |
|---|---:|---:|---:|
| forecast | 0.562180 | 0.562039 | -0.000140 |
| hidden | 0.678100 | 0.667322 | -0.010778 |
| hidden-future | 0.600596 | 0.600596 | +0.000000 |
| missing | 0.740263 | 0.725509 | -0.014754 |

## 2. Exact modifications relative to V11

- **Forecast (2,660 rows):** A 10% blend of a Savitzky–Golay polynomial smoothing pass of the 15-horizon trajectory, individually within each camera and forecast origin. Only already-computed predictions from the SAME `now` are combined. No later observations are accessed.
- **Hidden (849 rows):** 90% V11 plus 10% a legally generated historical matched-camera reference derived only from TRAIN and the legally released observations. Full released context is used exclusively for `hidden` (not `hidden-future`).
- **Missing (464 rows):** `max(0, 1.25 × V11 - 0.25 × official sample_submission reference)`; missing is permitted to use all legally released values. The official sample for `missing` agrees with released-only linear interpolation within 0.0005 (rounding).
- **Hidden-future (484 rows):** Byte-identical to V11. This is a deliberate safety boundary.

## 3. Additional self-mask diagnostics on released TEST data

After the initial development screening, 59 artificial gaps were constructed by deliberately concealing **568 already released observations** from Sep 18 and Sep 19 across available camera units. Every prediction was made after removing its synthetic labels from inputs. Actual unknown missing-target labels remained untouched and unread.

- Pooled surrogate V6 missing MAE: **7.29763**; calibrated extrapolation `.25`: **7.22467**.
- Artificial gap groups improved: **37/59**. Some camera/groups degraded; this is not a universal gain.
- This test uses a V6-style surrogate with exact original V4 pair-bridge for historical cameras A–H, and a linear V4 proxy for new cameras I–M. It is informative but **not a direct evaluation of V13 on unreleased target positions**, and should not be interpreted as an official score.
- A per-camera coefficient selected using leave-one-artificial-gap-out validation had MAE **7.48732**, worse than using the fixed 0.25 coefficient (**7.30927**). We therefore REJECTED adaptive per-camera tuning and retained the fixed, conservative formulation.

## 4. Submission artifact audit

- **CSV:** `submission_v13_causal_reference_calibrated.csv`
- **SHA-256:** `268a031687db92e2968f4477fa216d32709658231fd14447eadf6aa31a27e991`
- **Rows:** 4,457 predictions; columns exactly `id,people` in official sample order
- **Unique identifiers:** PASS
- **All values finite and nonnegative:** PASS
- **No withheld test target labels used:** PASS
- **Hidden-future source rows preserved byte-for-byte:** 484
- **Legal input classes:** frozen V11, original TRAIN, legally released TEST observations, official sample
- **Automatic Kaggle submission:** NO

## 5. Limitations and decision

- **0.400 not reached** locally or on Kaggle. The remaining local distance is `0.238867` points. The best previously confirmed public score supplied by the user is **0.749**, not the local V13 figure.
- The same two historical days have been repeatedly inspected. There is selection bias; only Kaggle can measure the actual public score, and the private 48% remains unknown.
- New-camera forecast targets I–M cannot be directly label-validated using the matched historical fold; these predictions inherit V11 with only same-origin smoothing.
- The release-based pseudo-mask study is a diagnostic and depends on its synthetic gap geometry, not a true official target measurement.
- The appropriate next decision is **manual** public evaluation of V13 if a submission is available, preserving the verified 0.749 candidate as fallback. Do not treat the reduction as a guaranteed public gain.

## 6. Reproduction

```powershell
python -m pip install numpy pandas scipy
python .\V13_build_source.py
python .\V13_validate_source.py
Get-FileHash .\submission_v13_causal_reference_calibrated.csv -Algorithm SHA256
```

The V13 source includes strict SHA-256 assertions for original inputs, and the validation source reads the supplied saved V7 development fold snapshots. It does not contact Kaggle.
