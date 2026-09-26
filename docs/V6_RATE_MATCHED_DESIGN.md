# V6 TRAIN feasibility: matched-rate resolution × QP

The existing V2 candidate family (identity, 96/112 resizing, an upsampled
112-pixel variant, and two blur levels) is encoded at the same QP for each
rate-distortion point. V6 asks a
narrower mechanistic question: when a 96/112/128 representation is encoded at
a *different* QP but uses nearly the same measured bytes as frozen V2, can it
correct MC3 without changing a correct r2plus1d/r3d prediction into an error?

## Committed pilot protocol

- Two codecs, H.264 and H.265, with unmodified ffmpeg x264/x265 at preset
  `medium`, 16 frames, original 128-pixel rate denominator, strict decode.
- First 30 source IDs from the frozen V2 TRAIN-fit manifest, split into two
  15-video Kaggle shards per codec. This is development data, not a holdout.
- QP30/35/40. Candidate resolutions 96/112/128; QP offsets
  `{-4,0,+4,+8,+12,+16}` within the legal [0,51] range. When measured rates
  bracket the V2 target, the intervening integer QPs are also measured.
- Every candidate byte count comes from the actual codec. Analyzer inference
  runs on at most two shortlisted QPs per resolution: the closest point at
  or below 102% of V2's bytes, plus the overall closest point if it is no
  more than 105% of V2's bytes. The strict matched-rate analysis uses ratios
  in [0.95,1.02].
- Run all three frozen analyzers on the decoded shortlisted candidate. Also
  run them on uncompressed 96/112/128 source clips to separate resize effects
  from codec effects. Re-encode the frozen V2 selection and verify its byte
  count exactly against the cache.
- Report match coverage, MC3 errors in V2, and a label-aware oracle that
  switches only to a matched candidate correcting MC3 without causing a new
  r2plus1d or r3d error on that QP-video point. Report the actual bpp change
  and search compute. This oracle reads labels and is **not deployable**.

The experiment tests a bounded candidate set and small TRAIN subset. A zero
rescue count would argue against learning a policy from this candidate family;
a positive count would justify a separate TRAIN-fit/validation policy study.
Neither outcome is a five-QP BD-rate or a generalization result. MC3 is a
development target and cannot then be described as independent.

The code is `ops/v6_rate_matched_pilot.py`, its Kaggle cell is
`kaggle/v6_rate_matched_cell.sh`, and the commit-pinned pusher is
`ops/push_v6_rate_matched.py`. Merge both shards per codec using
`python -m ops.v6_rate_matched_pilot merge` and preserve the source manifests.
