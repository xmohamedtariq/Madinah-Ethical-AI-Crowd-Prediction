# V3.3 actual-results audit -- 21 November 2026

## Decision

**REJECT V3.3. Preserve selected champion at 0.6057000521183546 (local).**
The user's last reported Kaggle public best is 0.729; it was not updated here.
No submission generated or uploaded.

Reviewed exact input bundle `V3_3_results.zip` supplied by the user:
`v3_3_decision.md`, `v3_3_frozen_config.json`, `v3_3_group_results.csv`,
`v3_3_outer_dates.csv`, `v3_3_pseudogap_coverage.csv`, `v3_3_summary.json`.

| Pseudo holdout | Rows | Gated | Champion MAE | Candidate MAE | Delta ratio (negative helps) |
|---|---:|---:|---:|---:|---:|
| 20260911_1100 | 108 | 80 | 10.883336 | 10.819783 | -0.0021157340 |
| 20260912_1100 | 108 | 80 | 7.744224 | 7.747622 | +0.0002376795 |

Mean E pseudo Missing ratio delta: -0.0009390273. This is smaller than the
predeclared requirement and the Saturday holdout worsened. Gate false (1/2 wins).

The retrospectively reused frozen development score was **0.6057086999660036**,
versus **0.6057000521183546** for the selected champion. The candidate worsened
by **+0.00000864784765**. It changed 55 frozen E targets. This is NOT independent
validation and was NOT an official Kaggle score.

Additional diagnostics: Friday long (10-15 minute) pseudo gaps improved by
-0.0021130 normalized ratio contribution; Saturday long gaps worsened by
+0.0027502. In contrast, Saturday short (5-9 minute) gaps improved -0.0025125.
These patterns are exploratory and must not be used to tune a new E correction
on the same two dates.

## Why switch tasks for V3.4?

V3.1-V3.3 Missing corrections lack consistent recording-level gains. Camera E
occurs in only the two main historical recordings, and the four short older
recordings do not contain it. We stop tuning E using those same labels.

V3.4 investigates a different, **causal Forecast** mechanism: whether recent,
already-observed champion forecast errors can be used to modestly calibrate the
next 1-15 minutes. Shadow predictions are made from data <= an earlier origin,
scored only when their targets become part of the observed past, and applied
only to later forecasts. Old chronological recordings select the correction
strength; the two longer historical dates are used only for research checks.

**No promotion** follows from a pseudo test. If a nonzero correction looks
promising, an exact eight-component, two-recording frozen replay and independent
evidence are needed before integration. The old 0.605700 frozen score must not
be algebraically adjusted using V3.4 proxy ratios.
