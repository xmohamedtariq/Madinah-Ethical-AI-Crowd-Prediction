Madinah local validation candidate

Run from the extracted Madinah_Codex_Cloud_Handoff directory.

  python cloud_research/selected/run_validation.py --kit research_kit --output cloud_research/selected/runs/replay

Choose a fresh output directory for each run. The runner records source hashes,
dependency versions, elapsed time and both sampled causal checks. The two saved
runs are selected_composite and verified_rerun.

candidate.predict_fold(train, released, targets, baseline) is the model entry
point. The wrapper partitions by recording, aligns outputs by target ID and
verifies finite, nonnegative counts.

The forecast model generates masks for historical training from actual released
prefix availability and the supplied target schedule. It uses only current
recording observations at or before each query origin. The forecast projection
uses the predicted same-origin path and estimates count ratios from the prefix.
Supply the complete known forecast target schedule for the measured model.

The hidden-future model gates peer innovations using completed pseudo-forecasts
from visible cameras. Each pseudo prediction masks its own entire camera and
cuts other cameras at its origin. Only completed pseudo labels at or before the
query origin can affect the correction. Fully hidden uses the frozen baseline.

The missing model starts from the archived missing branch, then adds a peer
bridge calibrated on pseudo-gaps with similar peer outages and a weak event
curvature correction. Missing uses allowed observations on both sides of a gap.

The supplied baseline is fixed throughout the timing checks. The measured local
score uses two reused recordings, reconstructed reference denominators and
3,719 matched rows. The official test has 4,457 targets, including 644 targets
from historically unseen cameras. A future production run would need an
explicit per-ID support gate retaining the archived production fallback.

Score, coverage, timing and comparisons are in RESULTS.json. Individual branch
reports are in the neighboring forecast, hidden and missing research folders.
