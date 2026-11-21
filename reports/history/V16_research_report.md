# Madinah Ethical AI | V16 selective Saturday forecast challenger

**Outcome:** V16 historical reconstructed four-mask development mean **0.633342802**, improved from V14 **0.635124441**. This is a **0.001781639** score change, not a breakthrough to 0.500. These are two previously studied recording dates (Friday Sep 11, Saturday Sep 12), reused under two masking protocols; do not mistake four masks for independent recording days. The only verified official Kaggle public score is **V13/V14 = 0.740**; V15 scored **0.780**. V16 has **not** been submitted.

## What changed

- V14 stays the base model and is kept byte-for-byte for **Friday, both days' hidden, hidden-future, missing, and Saturday cam_A and new cameras I–M**.
- Exactly **966 Saturday forecast predictions** for cameras **B, C, E, F, G, H** changed. The remaining **3,491** source lines are identical to V14.
- One historical LightGBM model (`num_leaves=40`, `objective=l1`, 300 boosting iterations) fits **6,713 historical causal synthetic forecast samples**. It learns a robust *relative correction to the last available count* using only historical labelled observations and strictly past features. The predictions are blended with V14 at B/C/G/H=30%, E/F=10%.
- Selection of these six cameras and mixing weights used two repeatedly assessed historical recording dates and **can overfit**. Do not treat the estimated gain as independent or guaranteed.

## Comparable score replay

| Mask protocol | Day | V14 local | V16 local |
|---|---|---:|---:|
| Matched | Fri Sep 11 | 0.654289 | **0.654289** unchanged |
| Matched | Sat Sep 12 | 0.618764 | **0.613111** |
| Legacy | Fri Sep 11 | 0.622078 | **0.622078** unchanged |
| Legacy | Sat Sep 12 | 0.645367 | **0.643893** |
| **Mean** | Reused development | **0.635124** | **0.633343** |

All scores above are a *reconstructed local surrogate score*, based on the V14 prior report's task reference normalization. They do **not** predict the Kaggle public/private scores directly.

## Additional independent published-label pseudo-check

Using 1,838 causal synthetic forecast targets from TEST *released* rows (not official withheld target values), the historical LightGBM model was compared to persistence: Friday model MAE 19.639 vs last-count MAE 20.379; Saturday model MAE **13.594** vs last-count **18.245**, improving each of 7 old Saturday camera groups against persistence. **This does not compare against V14 at pseudo-origins**, so it cannot establish that V16 will beat 0.740 on Kaggle.

Four more studies tested and rejected unpromising variants: cross-day hidden imputation, all-task V15 weekly blending (already scored 0.780), synthetic missing LightGBM, and multi-scale temporal smoothing. Full research CSV evidence is included.

## Causality and integrity

- Official 4,457 masked test `people` values are all missing, never read.
- Prediction features are created from the **current Saturday recording's released observations at minute <= each prediction's `now` only**; the implementation constructs a separate truncated context for every origin.
- Training uses only originally published historical `train.csv`, never official masked TEST labels.
- 4,457 sample IDs reproduced exactly in original order, no duplicates, finite non-negative predictions.
- Output hash SHA256: `2b03299d4396443f9be23f3b564ae2cfce66419ef2a5dd7ed63f6f4b9464cceb`.

## Reproduction

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm
python V16_build.py
python V16_cv_replay.py
```

`V16_build.py` regenerates the final CSV from the official input files and the frozen V14 predictions. Results have been re-generated in a clean directory with the same SHA256.

## Submission policy

**Do NOT submit automatically.** The user reports having only two Kaggle attempts left. V16 is a measured, strictly-causal **experimental candidate**, not verified to reach 0.500 or even surpass V14 on unseen Kaggle labels. The current proven fallback is V14 at **0.740**. If the user decides to spend one submission, upload the V16 CSV once and wait for the official score; do not spend both attempts on small variants.
