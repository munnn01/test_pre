# V2 paper-validation plan (exploratory reanalysis)

This plan specifies the analysis implemented in `ops/paper_validation.py`.
It reuses the already collected 1,000 paired TEST clips per codec; it is
**not** a new holdout and must not be described as independent confirmation.

## Question and unit

Does frozen dual-analyzer mode C improve the rate--accuracy trade-off over
simple fixed transforms and the frozen A/B policies? The independent sampling
unit recorded in the existing artifact is a clip. All five QPs, six candidates,
policies, and analyzers are repeated measurements of that clip, not new samples.
The historical index lacks audited `source_id`, so clip-level resampling is
descriptive and is **not** proof that distinct clips are source-independent.

## Fixed comparisons

For each codec separately, use the same 1,000 video IDs and existing real-codec
candidate measurements. Compare the unmodified `identity128` anchor with all
five non-identity fixed transforms, V1, and the pilot-frozen A, B, C policies.
Report every arm, including failures; do not choose a winning fixed transform
from TEST and then call it a prespecified comparator. Also compare C directly
against B with paired whole-video bootstrap resampling.

Report Top-1 BD-rate, BD-accuracy (percentage points), each QP's bitrate and
Top-1 accuracy, the minimum same-QP accuracy gap, choice counts, and 95%
bootstrap intervals. Bootstrap whole clip records with all their QPs paired.
The bootstrap is descriptive because this TEST set influenced earlier V1
research and source-level independence was not audited.

## Independent experiments still required

1. Evaluate a frozen third analyzer (`mc3_18`) that never enters policy fitting,
   calibration, or selection. Decode the bitstream actually chosen by the frozen
   policy; do not select using the third model or its labels.
2. Repeat on a genuinely new source-disjoint holdout/dataset. The existing
   `qktttttttttt/kineticscleaned` dataset supplied for these experiments is
   **not** a new data source; its unused clips can support only a labeled
   within-dataset diagnostic unless historical inspection and source identity
   can be audited. Require explicit source IDs and exclude sources used in
   pilot fit, calibration, development, and prior inspection before making a
   source-disjoint claim.
3. Extend the measured QP-40 compute pilot to encoder-only latency, peak
   memory, relevant QPs, more clips and repeated workers. Include every trial
   encoding in total cost.

Items 1 and 2 have **no completed results**. Item 3 has a completed 20-clip
encode+decode wall-time pilot, but its deployment-cost extension is pending.
The existing two-analyzer JSONL records cannot supply the missing experiments.
See the [five evidence gates](PAPER_EVIDENCE_GATES.md) for decisions and limits.

## Follow-up runners and measurement status

`ops/paper_heldout_mc3.py` runs the frozen C selector from the cached
two-analyzer measurements, then **re-encodes only the identity and chosen
streams** and evaluates `mc3_18`. It never lets mc3 predictions or labels
affect selection. Run two 500-clip shards per codec and merge their records;
the shard BD-rates are diagnostic only. Both prior V2 cache directories and
the Kinetics index/videos must be mounted in the runtime environment.
No completed `mc3_18` result is claimed here. The private Kaggle launcher
[`ops/push_paper_heldout_mc3.py`](../ops/push_paper_heldout_mc3.py) prepares
one commit-pinned codec/shard notebook at a time, provided the account can
access both Kinetics-cleaned and the private V2 cache dataset. Preparing or
pushing a notebook is not a completed evaluation. `--cpu` disables the Kaggle
GPU requirement when that account has no accelerator quota; record hardware
separately for every shard, and never compare its timing with GPU shards.

```powershell
python -m ops.paper_heldout_mc3 evaluate --index <kinetics-index.json> `
  --codec h264 --shard 0 `
  --cache-dir <h264-original-shard-0> --cache-dir <h264-original-shard-1> `
  --out-dir <mc3-h264-shard-0-output>
python -m ops.paper_heldout_mc3 merge --codec h264 `
  --shard-dir <mc3-h264-shard-0-output> `
  --shard-dir <mc3-h264-shard-1-output> `
  --out <mc3-h264-merged.json> --bootstrap 2000
```

Repeat with shard 1 and H.265. A mode-C gain on mc3 would support transfer
to one unseen analyzer, not universal model independence. The sample remains
the previously inspected 1,000 TEST clips.

`ops/paper_runtime.py` benchmarks the *complete* six-candidate selector
against a codec-only arm on the same predetermined source clips. The baseline
does not run an AR analyzer at the encoder; the full selector does. The default
is one codec at QP 40 (six versus one encode/decode call per clip). A separate
five-QP RD sweep is 30 versus five calls per clip and codec. Both arms measure
FFmpeg encode **and decode**, not production encode-only latency.
The clip is the paired block; arm order is balanced by a fixed hash. It times
all trial encodes, the two feature analyzers and the selector, excluding model
startup/download. Report both absolute wall time and overhead ratio.
The 20-clip QP-40 pilot on both codecs is complete; see
[`results/paper_runtime_v2_qp40_20/`](../results/paper_runtime_v2_qp40_20/README.md).
It does not replace a larger or encode-only deployment benchmark.

```powershell
python -m ops.paper_runtime --index <kinetics-index.json> `
  --codec h264 --split val --clips 20 --qps 40 `
  --out-dir <runtime-h264-output>
```

The CLI examples describe how to run the follow-ups. Do not infer a third-model
or new-source result from the completed runtime pilot.

## Paired visual and image-dataset checks

The qualitative Kinetics gallery is generated by `ops/paper_ar_visual.py`.
It chooses eight video IDs by a frozen hash of the ID alone, before looking at
labels, correctness, bitrate, or appearance. For each source video, codec and
QP 40, it shows the same frames 4/8/12: source, identity-coded anchor, and the
frozen V2 C-selected coded stream. Some selected streams have native 96 or
112-pixel resolution rather than 128; panels enlarge each to 256 pixels by
nearest-neighbor display scaling only, with native sizes stated on the panels.
The accompanying manifest records chosen IDs, choices, cached and re-encoded
bpp, and both analyzers' correctness. These eight clips are illustrations,
**not** a new Top-1 estimate. The complete 1,000-video metrics remain primary.

The distinct COCO val2017 OD check uses `ops/probe_background_suppression.py`
at a fixed 320-pixel letterbox resolution: 100 images sampled by seed
20260924, H.264/H.265 QPs 35/40/45, and the historical `blur4` ROI background
arm versus codec-only. The mask teacher is frozen MobileNet Faster R-CNN;
the held-out evaluator is frozen ResNet50 Faster R-CNN. Its panel IDs are the
smallest eight IDs in that preselected 100-image subset, never selected on
mAP. Each panel shows source, both reconstructions, protected mask, and
absolute-error maps on a common 0--64 RGB-level scale. Quantitative COCO
mAP@[.5:.95] and BD-rate use all 100 images and a paired image bootstrap.
This is an exploratory OD pilot with a different intervention from V2 C; it
does not establish that the AR selector improves OD, and its mAP/BD-rate must
not be compared numerically with Kinetics Top-1/BD-rate.

The deployed notebooks are private and commit-pinned. The Kinetics notebook
uses the private derived cache dataset `qktttttttttt/v2-paper-cache-1000-20260924`
and the public source videos; the COCO notebook uses
`awsaf49/coco-2017-dataset`. Both write PNG panels, a selection manifest,
and a compressed output artifact. A notebook being queued/running is not a
completed scientific result; inspect its final status and artifacts first.
