# Madinah V3.1 — Missing gap residual investigation

This research kit contains a single standalone Python script and the aggregate
results obtained from the supplied V3_phase0_bundle.zip.

## On Windows

Copy `v3_1_gap_residual_research.py` into the root of `Madinah-AI-Fresh`, then:

```powershell
Set-Location "C:\Users\USER\Desktop\Madinah-AI-Fresh"
.\.venv\Scripts\python.exe .\v3_1_gap_residual_research.py --root .
```

Outputs under `reports/v3_1`:
- v3_1_decision.md
- v3_1_exploratory_summary.json
- v3_1_group_comparisons.csv (aggregate only)
- v3_1_source_probe.json (source interfaces without executing them)

No champion source, prediction, training data, model, or Kaggle submissions are changed.
The coefficients are research-only and the development dates are reused.

## Next dependency for independent experiment

For the next step, inspect the generated `v3_1_source_probe.json`. The full
source for `src/frozen_protocol.py`, `src/evaluate_full_solution.py`,
`src/final_missing_hidden_research/best_missing_model.py`, and
`src/final_missing_hidden_research/independent_missing_loro.py` is required
to connect this proposal to the original champion and mask generator. The
provided diagnostic zip does not include them.

Previous four short recordings are not untouched; they were used for RTS
research. Until new independent validations are obtained, preserve champion.
