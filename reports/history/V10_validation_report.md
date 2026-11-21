# Madinah Ethical AI — V10 local optimization and integrity review

## Outcome
- V6 matched-mask development: **0.682947036**.
- V7 matched-mask development: **0.649188995**.
- V8 matched-mask development: **0.647335364**.
- V9 matched-mask development: **0.646893026**.
- V10 matched-mask development: **0.646259908**.
- Relative reduction V10 vs V6: **5.3719%**.
- Relative reduction V10 vs V8: **0.1661%**.
- 99% reduction vs V6 would require at most **0.006829470**; this goal **WAS NOT ACHIEVED**.

## Matched-mask & legacy fold scores

| Protocol | Date | V6 | V7 | V8 | V9 | V10 |
|---|---|---:|---:|---:|---:|---:|
| matched | 20260911_1100 | 0.706722 | 0.670869 | 0.669002 | 0.668847 | 0.668012 |
| matched | 20260912_1100 | 0.659172 | 0.627509 | 0.625668 | 0.624939 | 0.624507 |
| legacy | 20260911_1100 | 0.672881 | 0.639237 | 0.635823 | 0.635821 | 0.635012 |
| legacy | 20260912_1100 | 0.685599 | 0.655217 | 0.654222 | 0.654124 | 0.653522 |

Bootstrap 95% paired block-change interval for V10–V8: [-0.001916, -0.000260]. This is conditional on the previously reused validation dates and does **not** establish statistical generalization or independent improvement.

## Test-file integrity
- CSV: `submission_v10_temporal_ensemble.csv`, **4,457 rows**, exactly official sample order and columns `id,people`, unique IDs, finite nonnegative values.
- SHA-256: `9e1237ac258cca66b3ee4ad5cd3916a41c12c163dcf81c878aa0be9701171efc`.
- Future forecast rows (2,660) and missing rows (464) are byte-identical to verified V7.
- Hidden (849): 75% median17 + 25% Savitzky–Golay31 on pre-existing V7 hidden estimates; same camera, no gap-crossing.
- Hidden-future (484): Savitzky–Golay up to window 15, using only the pre-existing V7 predictions from the **same camera and same now**. No observed future test row is read.
- No actual hidden test labels, future target values, leaderboard/private labels or modified denominators were used.

## Experiments
- Screened contiguous-run median, Gaussian, Savitzky–Golay, weighted mean and Wiener; time-distance median, Gaussian and local linear regression; and constrained blend candidates on fixed development rows.
- Evaluated two-sided missing-value interpolation and PCHIP/Akima against V7/V6 GP: all were worse; missing predictions were not changed.
- Examined causal horizon-only smoothing for forecast and hidden-future; the modest hidden-future smoothing candidate improved all four reused development comparisons.

## Critical limitations and recommendation
**Development-set selection bias:** only two dated recordings with stored matching historical mask evaluations. These same dates were used in earlier candidate design and this search; the reported improvement may be optimistic. The public score has not been measured and cannot be predicted reliably from this reconstruction. The leaderboard is a different dataset split. **Do not present local 0.646 as a Kaggle score or as 99% accuracy.**
Preserve the historical best Kaggle submission (0.749) until a new public evaluation confirms an improvement. Only the CSV belongs in the Kaggle submission form; the Python files and reports are supporting evidence.

## Reproduction

`python V10_build_source.py` then `python V10_validate_source.py` from the folder containing verified V7 and the official-data directories. Original datasets and baseline hashes are checked before output.

## Research basis

- Daamen et al. (2026), short-term crowd forecasting: adaptive hybrid estimation with filter-based short-term error correction (https://doi.org/10.14311/APP.2026.57.0052).
- Cerqueira et al. (2022), *Forecast evaluation for data scientists: common pitfalls and best practices*: rolling-origin and future-data leakage cautions (https://link.springer.com/article/10.1007/s10618-022-00894-5).
- These are methodological references, not external validation of V10.
