# Engineering retrospective

The project began with forecasts and gap filling for count time series from multiple urban cameras. The team (single participant) explored temporal statistics, ensemble methods, camera-to-camera transfer, hybrid modeling, missing-data interpolation, feature engineering, and increasingly strict validation.

### A few things that were learned

1. **Task-specific availability matters**: future observations may be legal for some reconstruction tasks but illegal for causal forecasting and hidden-future tasks.
2. **Validation must reflect test masks**: the exact camera, recording, origin, horizon, and visibility mask can change the evaluation distribution.
3. **Local improvement is not a guarantee**: historical development folds were reused extensively, which increases overfitting risk. The best saved local score was 0.605700, while the best officially displayed public score was 0.729.
4. **Unseen cameras require fallback design**: 644 test targets were on cameras absent from historical training, requiring a documented support gate.
5. **A negative result is a result**: several elaborate postprocessors, profile-transfer approaches, RTS smoothing, and residual calibration variants were rejected because they worsened validation or generalized inconsistently.

### What this archive represents

This repository is a curated portion of the documented experiments, including selected source and rejected hypotheses. It is **not** a complete snapshot of every original local run, and the presence of a research script does not mean it is independently deployable or was used in the official submission.
