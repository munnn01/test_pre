# V4 TRAIN calibration (not a BD-rate result)

Four private Kaggle shards completed with `rc=0` and were downloaded, checked,
and merged. For **each codec**, the cache contains 400 source-disjoint
TRAIN-fit videos and 200 disjoint TRAIN-calibration videos at QP30/35/40 with
all six frozen V2 candidates. The two shards per codec agree on the pilot
archive hash, split fingerprints, code commit and candidate order. Every
new bitrate matched the cached V2 bitrate at `1e-9` tolerance; a mismatch
would have failed the Kaggle job. TRAIN-fit and TRAIN-calibration are the
same source sets for H.264 and H.265, enabling a paired codec comparison.

The collector was pinned to commit `62d986fbb719f75e3d0eb4bd8f40fdc99c7c6ed0`.
The merged TRAIN-fit fingerprint is
`4c12443af5c0ef44dc5143217ab00b146e0d9090607c49393487d5decd4e4bb1`;
TRAIN-calibration is
`ce5acf9334d4f683fdc7757d53b0d5be0ce80cc5a6fd4a4340e30c9357e79721`.

## Frozen selection

The preregistered fitting/calibration procedure is in
[`docs/DUAL_CODEC_V4_DESIGN.md`](../../docs/DUAL_CODEC_V4_DESIGN.md).
Model coefficients, complete calibration grid and frozen policies are in
[`configs/dual_codec_v4_frozen`](../../configs/dual_codec_v4_frozen).
They were committed in `ac2c6eb` **before** launching any V4 VAL notebook.

| Codec | QP30/35/40 frozen choice | Mean bitrate saving on TRAIN-calibration | Eligible V4 arms |
|---|---|---:|---:|
| H.264 | Identity fallback (all 600 codec/QP points) | 0.00% | 0/24 |
| H.265 | Predicted harm ≤0.04 at low QP and QP40, gain weight 0 | 3.73% | 3/24 |

H.265 selected identity for 508/600 codec/QP points and a lower-bitrate
candidate for 92/600. At each QP, the selected-versus-identity Top-1 gaps
on TRAIN-calibration were:

| Analyzer | H.265 QP30 | H.265 QP35 | H.265 QP40 |
|---|---:|---:|---:|
| `r2plus1d_18` | +0.5 pp | +1.5 pp | 0.0 pp |
| `r3d_18` | 0.0 pp | +0.5 pp | +0.5 pp |
| `mc3_18` | +1.0 pp | −0.5 pp | +0.5 pp |

For comparison, the *fixed* V2 selector saved 19.58% H.264 and 14.59%
H.265 mean bpp at these QPs, but lost `mc3_18` Top-1 on H.264 by
7.0/6.5/2.5 pp and on H.265 by 5.5/5.5 pp at QP30/35 (QP40 gained 1.0 pp).
This explains why the three-analyzer V4 constraint is conservative. It does
**not** establish that H.265 V4 improves full-curve BD-rate or that H.264
will improve at all. QP45/50 still use the frozen V2 selector, and are not
included in this three-QP calibration table.

## Next measurement

Four private [H.264 s0](https://www.kaggle.com/code/nguyenhoanglan1232/dual-v4-val-h264-s0),
[H.264 s1](https://www.kaggle.com/code/vtk269/dual-v4-val-h264-s1),
[H.265 s0](https://www.kaggle.com/code/wagur124705/dual-v4-val-h265-s0) and
[H.265 s1](https://www.kaggle.com/code/hieusunday0412/dual-v4-val-h265-s1)
notebooks are evaluating the frozen policies on the old 200-video V2/V3 VAL
subset at all five QPs. That VAL set has already influenced earlier
development, so its result is **development replication, not a new holdout**.
Because `mc3_18` provided V4 TRAIN labels, it is no longer an independent
analyzer for V4 even on VAL.
