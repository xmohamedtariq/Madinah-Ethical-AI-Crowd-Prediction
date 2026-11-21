# Madinah Ethical AI — V20 causal research audit

**Status (2026-11-19): RESEARCH ONLY — No new Kaggle submission created.** Target `<= 0.400` on a credible local test has **not** been reached. The best user-confirmed public score remains **0.740** from V13/V14. The previous V18 historical surrogate **0.623765** is not an independent estimate of public or private Kaggle performance.

## Why prior local improvements failed to transfer

The local historical test reuses 11 and 12 September and does not include labels for new-camera I–M targets. Earlier V7 `matched_protocol.build_matched_folds` correctly uses **only** the released rows from the official test template: none of the fully hidden cameras B/G/D appear in the released context. Several later explorations constructed a looser synthetic release by taking training rows minus marked target rows, leaving 159–171 rows of would-be hidden cameras that are unavailable in official tests. These looser views must **not** be interpreted as equivalent to the original V7 matched protocol or as proof that the earlier Kaggle gap was caused by leakage.

## Research streams tested today

### Weekly phase alignment — old cameras

A causal previous-week shape alignment algorithm fitted shift and amplitude based only on released observations up to each forecast origin; historic previous week observations are available in official `train.csv`.

- On current-week released-only retrospective rows, the anchored 50-minute algorithm achieved **16.529 MAE** against **22.099 MAE** for last value (23,922 paired rows; **25.21% gain**). This does **not** compare against V14.
- On comparable historical masked OOF rows, the unblended 50-minute algorithm achieved **16.349 MAE** against **13.064 MAE** for **V14** (2,086 rows; **25.15% worse**). Small blend weights also failed the combined historical comparison.
- **Decision:** reject; do not risk one of two Kaggle submissions based on a last-value benchmark alone.

### Analog-motif transfer — new cameras

A new cross-camera analog pattern library learned normalized 1–15-minute changes from prior released training dates. Fully strict pseudo-new-camera validation excludes **both the held-out recording and all occurrences of the simulated new camera** from its historical neighbors.

- On 415 pseudo-new-camera historical predictions, **last-value MAE 19.466**, motif MAE **18.179**, 50% blended MAE **18.125** (~6.9% vs last value).
- On 2,102 released-only current-week new-camera pseudo rows shared by all experts, motif MAE **11.584** vs last **11.637** (~0.46% gain). No official future-hole labels are known.
- **Decision:** small, inconsistent improvement; research only.

### Cross-camera LightGBM — new cameras

The count-ratio LightGBM trained on **86,355** rolling training examples, using causal 24-minute history, time, horizon, level and recent volatility. In the strict pseudo-new-camera validation, it excludes every training example from the held-out recording and simulated target camera.

- **Historical pseudo-new-camera CV:** last **19.466** vs LightGBM **17.079** MAE across 415 paired targets, a **12.27% improvement** over last.
- **Later-week released-only pseudo test:** last **11.624** vs LightGBM **11.988** across 2,252 targets, a **3.14% regression** on the true later-week distribution.
- **Decision:** reject. Its earlier gain does not generalize to later-week released data.

### Causal online expert aggregation

An exploratory online blend of last observation, KNN motif and LightGBM uses only already revealed pseudo-target outcomes at or before each origin for its weighting calculation. Its raw test labels for scoring are published released rows, **not** withheld Kaggle targets.

- On 2,102 shared new-camera pseudo rows, last-value MAE **11.6374**, online blend **11.4362**: a **1.73% improvement** over last, with mixed camera outcomes.
- This retrospective same-week test has already been inspected for model selection; it is not a fresh final holdout.
- **Decision:** insufficient support for full deployment; no promotion to a submission.

## Coverage and limits

The historical V7 matched-score coverage from the earlier archive was **3,719/4,457** target positions, with major unsupported portions including **555 new-camera forecast predictions**. Even a strong result on the covered slice cannot establish a full-test score at 0.400. The official Kaggle public/private split differs from our reconstructed local denominator/mask coverage. There is no reliable formula to transform the reused local score into a future public or private score.

All official `test.csv` target `people` values remain blank. None were imputed from private answers or used in training. Forecast pseudotests use source observations up to the origin and score only against already released later observations. No upload was made and V14 remains byte-identical (SHA-256 `ceec4f57451f6182e88213aa8a2984fac66d42a9fe60acf93d4ebcbe878ac6cd`).

**Promotion gate:** require a credible and stable >=0.400 local objective across all four task kinds plus substantially more convincing temporal/generalization evidence, while preserving the two remaining Kaggle submissions. This research does not pass that gate.

## Contents and replay

The separate ZIP archive contains research scripts, pseudo test result CSVs, log evidence, and `V20_integrity.json`. It deliberately contains **no new submission CSV**. All scripts use existing official training/test files from the prior research workspace; absolute paths in the prototypes point to `/mnt/data/madinah_v18_research/` and can be edited to match the local source folder.

Key scripts:

- `weekly_alignment_cv.py`, `historical_alignment_cv2.py`, `hist_blend_analysis.py`: weekly alignment testing
- `motif_forecast.py`, `new_cam_strict_replay.py`: cross-camera nearest-neighbor motifs
- `gradient_new_camera_cv.py`, `gradient_current_unseen_test.py`: LightGBM strict held-out and later-week transfer
- `online_experts_eval.py`: online combination, past-outcome-only weighting
- `audit_hidden_mask.py`, `audit_v20.py`: mask and provenance audit

The historical and current-week pseudo scores above use different target sets. Do not compare them as if they were a single official metric, and do not infer a Kaggle score of 0.400 from any one of these numbers.
