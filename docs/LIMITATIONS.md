# Validation and reproducibility limitations

- The saved `0.6057000521183546` metric uses **reused** development recordings and a reconstructed local reference, not an independent Kaggle-compatible evaluation.
- The public score of `0.729` is a **Kaggle-displayed value rounded to 3 decimal places**, not a measured private score or a claim of final rank.
- Numerous proposals were evaluated on the same development data; reported promising deltas may be optimistic due to repeated model selection.
- The official test contained 644 targets from novel cameras I–M. The production adapter retained the V13 archived reference on these IDs and did not validate their ground truth.
- The selected historical validation runner expects an original `research_kit/input/benchmark.py` not part of the shared champion archive. Raw validation prediction files and private target data are deliberately not published here.
- Several historical research scripts require source files, features, or dependencies not included in this repository; they are retained as an **archive**, not as a turnkey pipeline.
- No `0.400` local or Kaggle score was achieved or independently verified, and no competition win is claimed.
