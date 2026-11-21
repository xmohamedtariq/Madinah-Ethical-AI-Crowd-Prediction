# V3.2 Missing correction — decision

**Preserve Champion 0.6057000521183546. No Kaggle submission created.**

Source: `SELECTED_CHAMPION_CALLBACK`.

Outer held-out-recording wins: **0/4**; mean Missing ratio delta **+0.006388**; worst heldout ratio delta **+0.017378**.
Outer consistency gate passed: **False**. Final older-recording alpha: **+0.000**.

## Recording-disjoint results (historically reused old recordings)

| Recording | Targets | Gated | Alpha | Champion Missing ratio | Delta (negative=better) |
|---|---:|---:|---:|---:|---:|
| 20260904_1105 | 156 | 76 | -0.10 | 0.98162 | +0.00817 |
| 20260905_0426 | 156 | 76 | +0.00 | 0.98095 | +0.00000 |
| 20260905_1124 | 128 | 48 | +0.00 | 0.87471 | +0.00000 |
| 20260905_1900 | 156 | 76 | +0.10 | 0.88148 | +0.01738 |

## Limits

- Older recordings were **already used** in other research, so the tests are not untouched holdouts.
- No cam_E in older recordings: correction for E is an unvalidated transfer.
- Frozen Friday/Saturday scores are retrospective; do not use them to choose alpha.
- The legacy `evaluate_full_solution.py` is not the verified selected callback.
- V2 Forecast identity mismatches require resolution before any future promotion.
- No submitted CSV, model overwrite, Kaggle interaction, or selected-prediction change was produced.

## Reused frozen-fold preview (NOT independent validation)

| Scenario | Score | Delta | Changed | Friday delta | Saturday delta |
|---|---:|---:|---:|---:|---:|
| AF_ONLY_OLD_RECORDING_COVERAGE | 0.605700052 | +0.000000000 | 0 | +0.000000000 | +0.000000000 |
| AFE_TRANSFER_E_UNVALIDATED | 0.605700052 | +0.000000000 | 0 | +0.000000000 | +0.000000000 |

**Final: No change to the established Champion.**
