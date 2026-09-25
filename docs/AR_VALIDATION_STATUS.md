# AR V2-C: evidence and validation status

This is a code-and-artifact status note, not a fresh experiment or an independent
peer-review report. The primary paired TEST result remains the previously
inspected 1,000-video Kinetics sample in
[`results/dual_codec_search_v2_confirm_1000/`](../results/dual_codec_search_v2_confirm_1000/README.md).

## Verified artifacts

The SHA-256 values of both committed aggregate JSON files match the values in
their results README. Check this mechanically after every change:

```bash
python -m ops.verify_ar_result_hashes
```

Matching hashes establish consistency between these repository files and their
documented checksums. They do **not** by themselves re-run the Kaggle notebooks
or prove the original per-video records; those are separate provenance checks.

## What the selector costs

V2-C is a **search-and-selection policy over fixed pixel transforms**, not a
single-pass learned preprocessor. At one operating point for one codec it
encodes/decodes six candidates, evaluates both frozen AR analyzers on each
reconstruction, and evaluates both analyzers on the clean source before making
one choice. The codec-only encoder processes one stream. For a five-QP RD sweep,
this is 30 versus five codec encode/decode calls per clip and codec. Numbers
summed across both codecs describe a two-codec benchmark, not the cost of
deploying one selected codec at one QP. Only the chosen stream's bytes enter the
reported BD-rate; search compute is a separate cost.

[`ops/paper_runtime.py`](../ops/paper_runtime.py) measures this paired wall-time
comparison, with clip as the paired unit and balanced arm order. Its codec-only
baseline does not run task analyzers; its full arm includes candidate creation,
FFmpeg encode/decode, both analyzers, and selection. It reports the one-QP
deployment case by default (`--qps 40`) and accepts all five QPs for an RD
sweep. These are **encode+decode search timings**, not production encoder-only
latency. Model download/startup is excluded. A completed **20-clip QP 40
pilot** is recorded with manifests and artifact hashes in
[`results/paper_runtime_v2_qp40_20/`](../results/paper_runtime_v2_qp40_20/README.md).

## Data/model transfer are separate questions

- The V2 risk model and policy were fitted/tuned before the 1,000-video TEST
  comparison, but that TEST sample had already been examined in V1. Its paired
  bootstrap intervals are descriptive for that sample, not selection-adjusted
  evidence from a pristine holdout.
- [`ops/paper_heldout_mc3.py`](../ops/paper_heldout_mc3.py) can test transfer to
  an analyzer not used in V2 policy development. It uses the same previously
  inspected 1,000 clips, so it does **not** test transfer to new data.
- Reusing `qktttttttttt/kineticscleaned`, including a previously unused subset,
  is at most a **within-dataset diagnostic**. A source-disjoint holdout claim
  requires a genuinely new video source plus auditable source IDs; exclude all
  sources used in pilot fit/calibration/development and historical inspection,
  freeze the policy and analysis rule before seeing outcomes, then measure both
  analyzers with concurrent codec-only controls and whole-source bootstrap.
  The prospective [`ops/check_ar_source_holdout.py`](../ops/check_ar_source_holdout.py)
  preflight requires explicit `source_id` in both new and historical indices
  and rejects overlaps. The existing Kinetics index does not carry those IDs;
  a pass cannot be manufactured by assuming file names prove source identity.

## Repository lineage

Some older Kaggle templates still clone `munnn01/pre_updated` or an even older
repository. Keep them labeled as **historical runners** rather than silently
rewriting their code lineage. The V2 confirmation template
[`kaggle/dual_codec_search_confirm_1000_cell.sh`](../kaggle/dual_codec_search_confirm_1000_cell.sh)
clones `munnn01/pre_updated_v2` at the full commit SHA supplied by its push
helper; it does **not** run this `test_pre` checkout merely because the file
also exists here. The paper-visual templates clone `munnn01/test_pre` at a
pinned SHA. Reproduction should start from each exact notebook manifest, not
from whichever repository branch is currently `main`.

## Remaining gates

1. Extend the completed 20-clip, one-QP runtime pilot to more clips, repeated
   workers, all predeclared QPs where relevant, and production encode-only
   latency if a deployment-cost claim is desired.
2. Obtain a **new source-disjoint** dataset with stable source IDs and compatible
   Kinetics-400 labels; do not relabel the existing cleaned dataset as new.
3. Run the frozen policy on the new source with both original analyzers and
   concurrent controls; report all predeclared outcomes, including failures.
4. Keep a third-analyzer test separate from the new-data test. Passing either
   one alone is not evidence of universal AR generalization.
