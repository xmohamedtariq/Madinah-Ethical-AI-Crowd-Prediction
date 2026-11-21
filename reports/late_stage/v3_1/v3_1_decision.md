# V3.1 Missing correction — reused frozen-fold research ONLY

**Decision: PRESERVE CHAMPION. No candidate promoted. No submission generated.**
- Reconstructed champion: `0.6057000521184` (exact match).
- Targets: 358, blocks: 46, fast-falling: 147.
- Gated A/E/F targets: 115.
- No forecast/hidden/hidden-future predictions touched or created.

| Candidate | Frozen score (not independent) | Full score change | Friday change | Saturday change | Changed rows |
|:--|--:|--:|--:|--:|--:|
| verified_champion_unchanged | 0.605700052 | +0.000000000 | +0.000000000 | +0.000000000 | 0 |
| V3_1A_curvature_10pct_cap3 | 0.604451519 | -0.001248533 | -0.002369950 | +0.001121417 | 115 |
| V3_1B_linear_pull_5pct_cap3 | 0.606661476 | +0.000961423 | +0.000425607 | +0.000535817 | 115 |
| V3_1C_curvature_7p5_bridge_2p5_cap3 | 0.604705914 | -0.000994138 | -0.002115867 | +0.001121729 | 115 |

These hypotheses are exploratory, not blind: the frozen diagnostics had already been inspected during research.
These results were scored on repeatedly inspected validation folds, so they CANNOT justify replacing the champion.
A candidate that helps Friday and hurts Saturday fails the consistency gate.

## V2 report consistency alert
The previous V2 task comparison contains nonzero Forecast MAE changes for supposed Missing/Hidden-only substitutions:
- nearest_profile_hidden_only / 20260911_1100: Forecast MAE delta +0.012952777.
- nearest_profile_hidden_only / 20260912_1100: Forecast MAE delta -0.034480642.
- rts_missing_only / 20260911_1100: Forecast MAE delta +0.012952777.
- rts_missing_only / 20260912_1100: Forecast MAE delta -0.034480642.
Before any new integration, reconcile the original V2 baseline and re-check non-Missing exact identity.

## Next integration gate
Inspect `v3_1_source_probe.json`; obtain the real frozen callback/evaluator interface, and
run pseudo-gap LOSO on truly independent recordings with frozen coefficients before
any 0.600 claim. Previous four older holdouts were already used during RTS research.

No standalone predictions, submission CSV or uploads were produced.
