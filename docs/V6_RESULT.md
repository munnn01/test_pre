# V6 matched-rate resolution × QP: merged TRAIN diagnostic

The four Kaggle shards completed and both codecs merged with identical
frozen 30-video TRAIN-fit source fingerprints. This is **not** an achieved
BD-rate or a validation/test result. It is a label-aware oracle ceiling on
three QP points and a very small development subset.

| Codec | QP | V2 MC3 errors / 30 | Strict-rate matched / 30 | Safe oracle rescues / 30 | Search encodes |
|---|---:|---:|---:|---:|---:|
| H.264 | 30 | 13 | 30 | 3 | 795 |
| H.264 | 35 | 13 | 30 | 1 | 786 |
| H.264 | 40 | 18 | 30 | 1 | 597 |
| H.265 | 30 | 15 | 30 | 2 | 804 |
| H.265 | 35 | 17 | 30 | 3 | 795 |
| H.265 | 40 | 19 | 30 | 3 | 618 |

“Safe” means a matched-rate candidate changed an MC3 error to a correct
prediction without converting a correct r2plus1d or r3d prediction to an
error at that clip/QP. The union is only 5/90 H.264 and 8/90 H.265 points.
In H.264, respectively 7/8/8 of the MC3-wrong clips at QP30/35/40 are
also wrong on *all* clean 96/112/128 spatial representations. In H.265,
the counts are 8/8/7. A spatial-only codec adjustment cannot generally
correct those errors. The V6 family is therefore not promoted to a policy.

Each codec used over 2,100 search encodes for 90 QP-video points. That cost
is a pilot measurement, not the cost of an optimized deployment, but it makes
blind per-clip resolution/QP search unattractive. The next experiment is
[V7 temporal feasibility](V7_TEMPORAL_DESIGN.md), not a claim that V6
improved any of the three networks.

Source Kaggle notebooks:

- H.264: `nguyenhoanglan1232/dual-v6-rateqp-h264-s0`,
  `vtk269/dual-v6-rateqp-h264-s1`.
- H.265: `wagur124705/dual-v6-rateqp-h265-s0`,
  `hieusunday0412/dual-v6-rateqp-h265-s1`.

The merge manifests and summary numbers are preserved in
[H.264 JSON](../results/v6_rate_matched_train_30/h264_merged.json) and
[H.265 JSON](../results/v6_rate_matched_train_30/h265_merged.json).
