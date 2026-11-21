# V7 continuation — measured local improvement, no verified public target

**V7 وصل محليًا إلى 0.649188995، مقابل 0.682947036 لـV6، بتحسن 4.943%. أفضل سكور Kaggle عام مؤكد في المحادثة ما زال V6 = 0.749.** نتيجة V7 العامة غير معروفة؛ لم نرفع شيئًا أو نغيّر الاختيارات النهائية. هدف 0.400 العام، وكذلك 0.600، لم يتحقق كنتيجة عامة جديدة.

الملف المختار للمراجعة هو `submission_v7_masked_foundation_temporal.csv`. هذه نسخة **تجريبية غير مُرقّاة**: بوابة التحسن الكلي المعلنة كانت 5%، والنتيجة 4.943%، ولذلك بقي الحكم **FAILED**. لم نخفّض الشرط ولم نعدّل التقييم أو نبحث عن أوزان لتعويض الفرق. يوجد مرشح وسيط محفوظ لإعادة الإنتاج؛ ليس طلب رفع ثانٍ.

## Public evidence and immutable baselines

- V2: public 0.812; V3: 0.811; V4: 0.802; V5: 0.766; V6: 0.749, according to the user's Kaggle cards.
- Latest scored baseline: `submission_v6_dual_kernel_gp.csv`, SHA-256 `301a58589c5bc38ae466857434118af2878fec6f9f08aceba38bd68c8e6ebcac`.
- The local hashes of all five scored files are unchanged. Remote uploaded bytes, submission IDs/timestamps, account history and remaining quota were not refreshed in this environment.
- Two remaining attempts are user-reported. No attempt was consumed by this work. No future public or private score is guaranteed.

## Frozen formula and measured comparisons

Forecast on old cameras uses a two-member TabICLv2 count-error correction of the historical V4 phase expert, trained only on inner date-out expert rows. Hidden uses half scored V6 plus half a two-member robust/quantile-normalized TabICL count model, trained on historical **sparse release-mask views only**. A fixed five-minute centered median suppresses isolated hidden-output spikes when at least three nearby predictions exist. Hidden-future uses half V6 plus half an eight-member count model trained on historical dense and masked views. Missing and actual new-camera forecasts retain V6 values.

The hidden temporal rule is legal full-context reconstruction and is **never applied to forecast or hidden-future**. There was one declared five-minute width; it was not searched until a threshold passed. The sparse-view-only HF experiment was inferior, so the stronger original HF path remains. Direct trajectory prediction and local context retrieval also failed to beat the retained components.

Selection occurred after inspecting development component results. The final formulas were frozen before their joint evaluations, but this does **not** make those reused dates independent confirmation.

| Protocol | Date | V6 | V7 | Relative change | Paired score-change interval |
| --- | --- | ---: | ---: | ---: | --- |
| matched | 20260911 | 0.706721986 | 0.670868799 | -5.07% | [-0.05518, -0.01777] |
| matched | 20260912 | 0.659172086 | 0.627509192 | -4.80% | [-0.06377, -0.00177] |
| legacy | 20260911 | 0.672881104 | 0.639236990 | -5.00% | [-0.05372, -0.01300] |
| legacy | 20260912 | 0.685598948 | 0.655217065 | -4.43% | [-0.06502, +0.00416] |

Primary mean: **0.649188995**, versus V6 **0.682947036**. The unchanged 5% gate required at most **0.648799684**. Each primary-date 3% gate and the no-legacy-point-harm gate passed; the mean 5% gate failed.

Intervals use a paired 20-minute camera-block bootstrap with 2,000 draws and frozen reference denominators. They describe conditional variation on these already inspected development rows. They are not public-score confidence intervals or independent statistical confirmation. The legacy Saturday interval still includes possible harm; the number of distinct full recording dates is only two.

### Task MAE on unchanged primary rows

| Task | Friday V6 | Friday V7 | Saturday V6 | Saturday V7 |
| --- | ---: | ---: | ---: | ---: |
| forecast | 23.84505 | 22.15757 | 12.81353 | 12.43582 |
| hidden | 4.23172 | 3.90874 | 5.03409 | 4.38450 |
| hidden-future | 4.00255 | 3.78592 | 5.65266 | 5.43309 |
| missing | 6.06343 | 6.06343 | 4.31965 | 4.31965 |

Local scoring remains the mean of per-recording, equally weighted four-task normalized MAE ratios. The local reference is a reconstruction; it has not become a fully independently certified copy of Kaggle's unavailable scorer/reference. No denominator, hard row, label, mask or score definition was changed.

Primary coverage stays forecast **2,036/2,660**, hidden **843/849**, hidden-future **482/484**, missing **358/464**: **3,719/4,457** labelled positions. Actual new-camera target labels do not exist locally; the 555 new-camera forecast outputs remain unchanged. The extra boundary/unsupported positions are not claimed as validated. Wider deployment history differs from outer fitting history, so observed local improvement need not transfer to Kaggle.

## Experiments and negative evidence

The continuation rebuilt the promising foundation-model observations from the continuation notes because their unsaved scratch files were absent. All numerical claims here come from the newly saved predictions and results, not those notes. The earlier V7 report is preserved as `reports/v7_round1_research_report.md`.

23 saved experiment screens contain 56 declared options/formulas across the original round and this continuation. These include task-only alternatives and joint formulas, not 56 independent experiments. The table shows the lowest development option in each screen; this is optimistically selected evidence and is not an upload recommendation. Task-only options preserve V6 on other tasks. Complete failures and all predictions remain packaged.

| Screen | Options | Lowest option | Matched aggregate | All four comparisons improve? |
| --- | ---: | --- | ---: | --- |
| v7_combined_foundation | 1 | fixed formula | 0.653904954 | yes |
| v7_direct_foundation | 4 | count_delta_0.5 | 0.689106603 | no / not established |
| v7_diverse_foundation | 1 | fixed formula | 0.654535602 | yes |
| v7_foundation_composition_hidden | 2 | 0.5 | 0.669543115 | yes |
| v7_foundation_composition_hidden-future | 2 | 0.5 | 0.683378186 | no / not established |
| v7_foundation_forecast | 2 | 1.0 | 0.675703058 | yes |
| v7_foundation_hf_stream | 2 | 0.5 | 0.675471847 | yes |
| v7_foundation_hidden | 2 | 0.5 | 0.672445558 | yes |
| v7_foundation_norm_forecast | 2 | 0.5 | 0.683100475 | no / not established |
| v7_foundation_norm_hf | 2 | 0.5 | 0.679090418 | yes |
| v7_foundation_norm_hidden | 2 | 0.5 | 0.668624121 | yes |
| v7_foundation_reference_hidden | 2 | 0.5 | 0.671920925 | yes |
| v7_foundation_support_hidden | 2 | 0.5 | 0.677459599 | no / not established |
| v7_foundation_support_hidden-future | 2 | 0.5 | 0.682193626 | no / not established |
| v7_foundation_uncertainty | 2 | uncertainty_shrink | 0.669080679 | yes |
| v7_hidden_temporal | 1 | fixed formula | 0.649188995 | yes |
| v7_joint_gp | 6 | missing_0.5 | 0.697901513 | no / not established |
| v7_mask_context_hf | 2 | 0.5 | 0.680473820 | yes |
| v7_mask_context_hidden | 2 | 0.5 | 0.665774948 | yes |
| v7_online_forecast | 3 | level | 0.685533079 | no / not established |
| v7_retrieval_foundation | 2 | 0.5 | 0.673681829 | yes |
| v7_selfmasked_reconstruction | 6 | lgb_0.5 | 0.686065415 | no / not established |
| v7_variance_gp | 4 | sqrt_0.5 | 0.683960349 | no / not established |

One eight-member HF attempt exited 137 before a score was written; a representation-cache replay was then interrupted after the equivalent one-camera-at-a-time KV implementation completed all four comparisons. These folders are marked incomplete and excluded from results. One temporal-assembly run stopped before scoring because a pandas Boolean Series aligned unlike row indices; it was corrected to a positional mask while retaining ID-order assertions. The corrected replay is the source of the reported result. No incomplete output is counted as a successful experiment.

## Final file, lineage and reproducibility

- File: `submissions/submission_v7_masked_foundation_temporal.csv`.
- SHA-256: `1c6984fec7d3ec3674fa0135b8b08e3e01394ecf5bacdf319ea22faaa6a998e8`.
- 4,457 rows; exact official sample `id,people` columns, IDs and order; unique IDs; finite nonnegative predictions.
- 849 hidden rows were replaced relative to the intermediate parent; the other **3,608 parent rows are byte-identical**, including all **3,144 future targets**. Missing and new-camera forecast lines are inherited unchanged from scored V6.
- The verified intermediate parent explicitly truncated released inputs before future feature extraction. Altering and deleting post-`now` values left those legal input tables identical for **22 origins covering 3,144 future rows**. Each future prediction call grouped only one origin. Historical training contexts had no actual test-target labels. This new hidden-only replacement introduces no new future input path.
- Frozen contexts, features, outputs and configuration are saved. The public checkpoint is shared once under `models/`: `jingang/TabICL`, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, checkpoint SHA-256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`.
- Replaying the parent from frozen contexts/checkpoint had maximum raw CSV difference **2.57324218e-06** counts (tolerance 1e-5, including float32 text serialization). The new full **4,457-row** composition replay had maximum difference **0** counts. No configuration reselection or pretrained weight optimization occurred during replay. Zero full-file difference was measured on this CPU/runtime; bitwise equality on different hardware is not assumed.
- TabICL 2.2.0, CPU torch 2.5.1, Python 3.12. Dependency pins, portable checkpoint handling and third-party BSD-3-Clause license are included. No API key is needed for local replay with the included weights.

Run `python -m src.v7_verify` and `python -m src.v7_verify_temporal` after installing `requirements_v7.txt`. The generators intentionally refuse to overwrite frozen files; see `README_V7_FOUNDATION.md`. All current package files are hashed in `V7_PACKAGE_CONTENTS.sha256.json`.

Official implementation/model sources: https://github.com/soda-inria/tabicl ; https://tabicl.readthedocs.io/en/latest/api.html ; https://huggingface.co/jingang/TabICL . These document the method and weights, not our measured competition scores.

## Decision and resumption

Status: **PENDING_RESEARCH_EXPERIMENTAL_V7_NOT_PROMOTED**. The final file is available for review as an experimental candidate; it is not certified to reach 0.600 or 0.400. Preserve the two manual attempts unless the user chooses to measure it publicly. Never auto-upload or change final selections.

This research execution has finished; no training or polling task remains active after the checkpoint. The goal has not been marked successful or blocked by invented missing resources. Further work can resume from the saved contexts/results, or from an actual manual Kaggle measurement if the user chooses one. Keep the current development failures and any returned public result; do not tune the scoring protocol to match them.
