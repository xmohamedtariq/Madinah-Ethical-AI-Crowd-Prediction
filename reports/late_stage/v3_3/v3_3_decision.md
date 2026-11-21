# V3.3 E-camera residual calibration — research-only decision

**Champion preserved at 0.6057000521183546. No submission generated.**

Champion source: `SELECTED_CHAMPION_CALLBACK`. The two-date pseudo-gap guardrail passed: **False**.
Mean E pseudo Missing ratio delta: **-0.000939** (negative is improvement).

| Held-out date | Pseudo rows | Gated | Changed | Champion MAE | Candidate MAE | Delta ratio |
|---|---:|---:|---:|---:|---:|---:|
| 20260911_1100 | 108 | 80 | 80 | 10.88334 | 10.81978 | -0.002116 |
| 20260912_1100 | 108 | 80 | 80 | 7.74422 | 7.74762 | +0.000238 |

## Frozen replay (development only)

- Status: `RETROSPECTIVE_REUSED_FROZEN_NOT_INDEPENDENT_NOT_PROMOTABLE`
- Estimated full local score: `0.6057086999660036`
- Frozen score delta: `8.647847648933141e-06`

**No promotion**: camera E has only two historically reused labeled recording dates; this test is neither untouched nor official.
The historical original frozen-task masks are only partially represented by the Missing ID exclusions.
Never treat the retrospective score as an independently verified improvement or as a Kaggle result.
