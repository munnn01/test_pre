# V5: V2-anchored sparse MC3 repair (development experiment)

## Hypothesis and scope

Keep the frozen V2-C choice as the default shared H.264/H.265 bitstream.
At QP30/35/40 only, consider replacing that choice with one of the same six
already encoded candidates when a TRAIN-fitted, label-free pairwise model
predicts that doing so will rescue `mc3_18` without harming the two primary
analyzers too often. At QP45/50 and whenever V2 chooses identity, V5 is
**exactly V2**. The intervention does not alter the standard codecs.

`mc3_18` labels on TRAIN are used to fit and calibrate V5; MC3 is therefore
**not an unseen independent analyzer** for this experiment. Inference uses
only bitrate, QP, source-video motion features, and the existing V2 signals
from `r2plus1d_18`/`r3d_18`; it never receives a ground-truth label or MC3
prediction. All six candidate encodes are still required, so a selected
stream's BD-rate does not include encoder search compute.

## Locked fit/calibration protocol

- Fit: 400 V2 TRAIN-fit videos per codec, joined by sequence ID to V4 MC3
  labels; fit pairwise logistic heads for MC3 rescue/loss and for each primary
  analyzer's loss relative to the *frozen V2 selection*.
- Calibrate: 200 disjoint TRAIN-calibration videos per codec. Candidate switches
  are limited to at most 20% more bits on the switched clip. The selected
  policy must have at most 2% aggregate bpp increase at each measured QP and
  at most one additional primary-model error per 200 videos at each QP.
  Among eligible policies, maximize net MC3 correct count; otherwise keep V2.
- This 2% bpp limit is a development budget, **not** a guarantee on five-QP
  BD-rate. On evaluation, compare V5 minus V2 with paired whole-video
  bootstrap. Declare noninferiority only if a separately specified BD-rate
  margin is supported by that interval; no such claim follows from TRAIN-cal.

The 1% budget was inspected first and yielded no positive-MC3 eligible
policy; the 2% budget was then adopted on TRAIN-calibration. This is
**exploratory calibration**, locked only before VAL, not preregistration.

Frozen TRAIN-calibration diagnostics (not independent results):

| Codec | Net MC3 correct gain / 600 QP-video points | Switches / 600 | bpp change QP30 / 35 / 40 | Primary Top-1 gap |
|---|---:|---:|---|---|
| H.264 | +5 | 72 | +0.97% / +1.57% / +1.72% | r2: 0/0/0 pp; r3: 0/+0.5/+0.5 pp |
| H.265 | +10 | 61 | +1.34% / +1.18% / +0.87% | r2: 0/0/0 pp; r3: -0.5/0/+0.5 pp |

The old VAL clips were inspected during earlier development. The Kaggle
five-QP comparison here is therefore a **development replication**, not a
new source-disjoint holdout or a paper-level generalization result.

## Reproduce and falsify

1. Run `python -m ops.dual_codec_v5_repair` separately for H.264/H.265 with
   the V2 pilot archive and merged V4 TRAIN cache. Check the archive/manifest
   hashes and commit `configs/dual_codec_v5_frozen/{codec}`.
2. Push `kaggle/v5_repair_val_cell.sh` with `ops/push_v5_repair_val.py` on two
   shards per codec, then merge with `ops.dual_codec_v5_eval merge`. The
   notebook is pinned to a full Git commit and the private V2 pilot dataset.
3. Report both primary BD-rate curves and the MC3 curve, all five QPs, paired
   V5-minus-V2 bootstrap intervals, choice counts, and encoder cost. If the
   two primary BD-rates regress materially or MC3 paired improvement is
   inconclusive, reject this candidate and keep V2. Do not retune on VAL.
4. For a generalization claim, acquire source-disjoint video, freeze the
   policy before seeing it, and evaluate an additional unseen analyzer.

The prior oracle check was label-aware and served only to test feasibility;
its gains are not deployable performance.
