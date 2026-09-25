# Dual-analyzer AR V2: 1,000-clip paired evaluation

Four Kaggle jobs finished and were merged as two disjoint 500-clip shards per
codec. Each codec has exactly 1,000 unique TEST clips, the same fingerprint
`aae3888f3ae34d08`, five QPs (30, 35, 40, 45, 50), and six encoded
candidates per clip/QP. BD-rate was computed from the *merged* curves, not by
averaging shard BD-rates. The reported intervals use 2,000 paired whole-video
bootstrap resamples. The aggregate JSONs retain the policy, shard IDs,
fingerprints, full curves, and V1 concurrent controls.

| Codec | Analyzer | V2 Top-1 BD-rate | 95% bootstrap CI | V2 BD-accuracy (pp) | Concurrent V1 BD-rate |
|---|---|---:|---:|---:|---:|
| H.264 | `r2plus1d_18` | **-22.01%** | [-23.85%, -20.24%] | +10.40 | -24.88% |
| H.264 | `r3d_18` | -14.47% | [-15.69%, -13.21%] | +6.09 | -1.03% |
| H.265 | `r2plus1d_18` | -14.03% | [-15.39%, -12.77%] | +9.92 | -16.09% |
| H.265 | `r3d_18` | -8.72% | [-9.62%, -7.82%] | +5.53 | -0.99% |

**Decision:** The dual-analyzer improvement and same-QP guards pass, but the
strict point-estimate target of BD-rate below -15% on *both* analyzers for at
least one codec fails. H.264 misses on `r3d_18` by 0.53 percentage points;
the CI crossing -15% does not change the point-estimate decision. V2 trades
some `r2plus1d_18` gain against V1 for much stronger `r3d_18` transfer.
Neither analyzer is independent of V2 policy development. Moreover, the
same TEST set was inspected in earlier V1 work: this is a paired replication,
not a new untouched holdout or a generalization claim.

The Kaggle runs are [H.264 shard 0](https://www.kaggle.com/code/trnhlng/dual-ar-v2-confirm-h264-s0),
[H.264 shard 1](https://www.kaggle.com/code/huolgggnuyen/dual-ar-v2-confirm-h264-s1),
[H.265 shard 0](https://www.kaggle.com/code/baoancut/dual-ar-v2-confirm-h265-s0),
and [H.265 shard 1](https://www.kaggle.com/code/shungg05/dual-ar-v2-confirm-h265-s1).
The raw downloaded shard archives and per-clip records remain in the local
`proxy_v3/kaggle_results/dual_codec_search_v2_confirm_1000/` workspace;
only the compact aggregate JSONs are committed here.

SHA-256 of committed aggregates:

- `h264_result.json`: `52aabb9a63464b39842e394479e301b1842df05b6a1ccb4111c68a5c677c1286`
- `h265_result.json`: `c0fbb81131abeb79916b2ddb0196eafbe0694596fc6c0eef6380f791c716e72e`

Verify both checksums against these exact committed files with
`python -m ops.verify_ar_result_hashes` from the repository root. A matching
checksum is a repository-integrity check, not an independent re-run.
