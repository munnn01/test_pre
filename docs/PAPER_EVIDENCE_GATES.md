# AR paper evidence gates (artifact audit, 2026-09-25)

This is a status ledger for the frozen V2-C selector, not a claim that all
experiments below have passed. Five QPs, six candidates, and multiple
analyzers are paired measurements of each clip, not independent replicates.
The existing 1,000-clip bootstrap resamples clip records because source IDs
were not audited; it cannot certify independence of source videos. A future
new-source holdout must use whole-source clusters as its resampling unit.
The existing TEST was previously inspected during V1.

## Outcome amendment (2026-09-25)

The project no longer requires BD-rate below −15%. For future work the
**directional objective** is Top-1 BD-rate < 0 and BD-accuracy > 0 for both
`r2plus1d_18` and `r3d_18` on at least one codec, with the existing same-QP
guard reported. This amendment was made **after** seeing the 1,000-clip V2
results: its application to those clips is exploratory, not a retroactive
prespecified success. The historical <−15% decision and raw artifacts remain
unchanged. For the not-yet-measured `mc3_18` transfer test, report point
estimates and paired bootstrap intervals; negative BD-rate with an interval
entirely below zero would be stronger evidence than a negative point alone.

| Gate | Evidence in this checkout | Decision | What would close it |
|---|---|---|---|
| Unseen analyzer | `ops/paper_heldout_mc3.py` implements a frozen-choice `mc3_18` test, but no merged `mc3_18` result is committed. | **Not measured.** Neither `r2plus1d_18` nor `r3d_18` is an independent analyzer for V2 policy development. | Run both codec shards, merge all 1,000 paired videos per codec, publish curves, BD-rate/BD-accuracy, intervals, and provenance. Never let `mc3_18` predictions change the chosen stream. This tests one new model on old data, not new-source generalization. |
| New-source holdout | The Kinetics-cleaned TEST and all its subdivisions come from the source used during development. `ops/check_ar_source_holdout.py` is only a preflight. | **Not available.** Existing TEST is a paired replication, not a pristine holdout. | Obtain a genuinely new video source with auditable `source_id`, Kinetics-400-compatible labels and rights to evaluate; compare against all historical source IDs before opening outcomes, freeze code/policy/analysis, and bootstrap whole sources. |
| Historical prespecified <−15% target | [`results/dual_codec_search_v2_confirm_1000/`](../results/dual_codec_search_v2_confirm_1000/README.md) contains the 1,000-clip paired results. | **Failed.** H.264 `r3d_18` Top-1 BD-rate is −14.47%, missing that historical strict target by 0.53 percentage points. | Keep the historical decision visible; it is not the current optimization requirement. |
| Revised directional objective | The same V2 JSON reports negative Top-1 BD-rate and positive BD-accuracy for both development-involved analyzers on both codecs. | **Met descriptively on old TEST**, but post hoc relative to these data; no independent confirmation. | Evaluate the frozen policy on an unseen analyzer and a genuinely new source. Do not retune using their outcomes and then call the same run confirmatory. |
| Search compute | [`results/paper_runtime_v2_qp40_20/`](../results/paper_runtime_v2_qp40_20/README.md) contains 20 paired clips per codec and raw per-clip JSON. | **Pilot measured, deployment cost not settled.** At QP 40, median per-clip overhead is 6.22× H.264 and 6.15× H.265 versus one codec encode+decode without encoder-side analyzers. Different workers ran the two codecs. | Repeat on more preselected clips/workers, all relevant QPs, and measure production encoder-only latency, memory and energy if claimed. Count all six trial encodes and analyzer passes. |
| OD alongside AR | [`results/paper_od_pilot_100/`](../results/paper_od_pilot_100/README.md) is a separate COCO background-suppression pilot with a different intervention, metric, and sample size. | **No unified-method claim.** The 100-image pilot does not establish that V2-C improves detection; H.265 uncertainty includes zero. | Either present OD as a separate exploratory study or run the *same specified intervention* with adequately powered OD evaluation and a prespecified joint claim. Do not pool OD mAP with AR Top-1. |

The runtime pilot closes only the literal statement “no cost numbers exist.” It
does not close the model-transfer, data-transfer, or performance gates. The
`mc3_18` runner, even if it later succeeds, will remain a within-dataset
transfer diagnostic. A strong generalization claim requires the new-source
experiment independently of the third-model experiment.

For a paper using only current evidence, describe V2-C as a six-candidate
encoder-side search policy that improved two *development-involved* analyzers
on a previously inspected TEST sample, met the revised *exploratory*
directional objective but missed its historical strict prespecified target,
and incurred the measured pilot compute overhead. Report OD separately.
