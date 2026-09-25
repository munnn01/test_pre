# V4: source-motion-conditioned candidate regret, stage 1

Status: pre-measurement design for Kaggle TRAIN-label collection. No V4 policy
has been fitted, selected or evaluated yet.

The V3 development replay showed that blocking `area96` at QP30/35 improves
`mc3_18` on H.264 VAL but leaves a 7-point same-QP loss at H.264 QP40, while
H.265 still loses 5.5 points at QP30. Thus a codec-specific, QP-conditioned
candidate-risk model is the next *hypothesis*, not an established result.

## Stage-1 layout (frozen before its Kaggle runs)

- The experimental unit is a whole source video; QPs and transforms are paired
  technical measurements on that video, not independent replicates.
- Use only the V2 pilot's 400 TRAIN-fit and disjoint 200 TRAIN-calibration source
  IDs per codec. Split each stage by alternating position into two 200-fit /
  100-calibration shards. The four jobs occupy the same four Kaggle accounts
  used for V3, one shard per account.
- At QP 30, 35 and 40, encode all six *unchanged* V2 candidates using the
  standard H.264/H.265 codec. The archive's cached bpp must match every new
  encode. Collect frozen `mc3_18` correctness as a TRAIN label, not as a
  selector input. Keep the old V2 `r2plus1d_18`/`r3d_18` measurements frozen.
- Record cheap, source-only temporal difference and edge statistics. No target
  class, correctness or `mc3_18` output may appear in deployed features.
- Preserve the original class mapping, 16 frames, 128-pixel input, preset
  `medium`, all QP values and all candidate definitions. Record commit, split
  fingerprints, archive hash, index hash and per-clip outcomes.

## Planned stages after collection

1. Merge the two shards per codec and verify all 400 fit and 200 calibration
   sources, all 3 QPs x 6 candidates, bpp agreement and source disjointness.
2. Fit a regularized risk/regret predictor on TRAIN-fit only. Calibrate its
   codec-QP constraints on TRAIN-calibration only. The V2 policy is the fixed
   control; freeze one V4 policy before any further evaluation.
3. Audit all source IDs previously used in project experiments before choosing
   an untouched evaluation subset. If fully untouched sources cannot be
   demonstrated, describe the run as further development replication. Once
   `mc3_18` supplies V4 training labels it is no longer an independent analyzer;
   reserve a fourth compatible frozen model and untouched clips for transfer.
4. Compare all analyzers at the same five QPs, with BD-rate Top-1,
   BD-accuracy Top-1, every same-QP Top-1 gap, and full encoder-side wall time.
   Paired bootstrap resamples whole source videos, not QP observations.

Stage 1 alone cannot demonstrate better BD-rate. It only creates the missing
training labels for a testable V4 policy; there is no V4 result until the later
stages have been run and audited.
