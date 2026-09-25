# V3 low-QP development ablation

Status: development experiment. No V3 policy is frozen or claimed superior yet.

The old 1,000-video TEST was inspected during V1/V2 research, and the later
`mc3_18` measurement on the same TEST motivated this design. Consequently neither
the old TEST nor the VAL run below can establish new-source generalization. The
independent unit for paired resampling is a source video, not a frame or QP.

## Question and fixed arms

Does restricting the frequent `area96` choice at QP 30/35 reduce same-QP
recognition loss without forfeiting the rate–accuracy gain? The same cached
V2 pilot clips, candidate bitstreams, frozen risk model and frozen C policy are
replayed in every arm. No correctness field is provided to selection.

| Arm | Change at QP 30 and 35 | QP 40/45/50 |
| --- | --- | --- |
| `v2_frozen` | None | Frozen V2-C |
| `no_area96_low` | Exclude `area96`; choose the next feasible candidate | Frozen V2-C |
| `guard_area96_low` | Permit `area96` only when its rate saving is at least 20% and both V2 risk scores are at most half the original low-QP threshold | Frozen V2-C |

The constants above were set before the new Kaggle runs. This is a deliberately
small ablation, not a threshold search on old TEST. The cached TRAIN-calibration
and VAL-dev samples (200 distinct videos each, per codec) are reported separately.
The old `r2plus1d_18` and `r3d_18` outcomes come directly from the pilot cache;
the `mc3_18` outcomes require fresh encode/decode and inference on the same VAL
videos. Its old TEST result has already influenced the V3 hypothesis, so `mc3_18`
must be described as an exploratory transfer check here, not pristine held-out
confirmation. The two 100-video shards must be merged before reporting metrics.

Primary descriptive metrics per codec and analyzer are BD-rate Top-1,
BD-accuracy Top-1, and the minimum same-QP Top-1 gap over QP 30–50. Compute
cost is not estimated by this replay: a production selector still measures all
six encoded candidates before deciding. The `mc3_18` timing records only the
anchor and the union of selected streams needed for independent evaluation.

The next confirmatory study must freeze one policy and use videos with source
identifiers not used in V1/V2/V3 policy development. It should then measure
full encoder-side latency and memory, as well as all three analyzers. Do not
substitute another shard of the same Kinetics source pool for a new-source test.

## Reproduction

The Kaggle notebook template is `kaggle/dual_codec_lowqp_v3_cell.sh`. It pins a
Git commit, attaches a private copy of the original V2 pilot archive, builds the
same Kinetics index, replays the three arms, and evaluates one `mc3_18` VAL
shard. The archive loader checks frozen policy, risk state, candidate list,
split IDs and cache keys before producing results.

After downloading both shards for a codec, merge with:

```powershell
python -m ops.paper_lowqp_v3_mc3 merge --codec h264 `
  --shard-dir <h264-shard-0-mc3> --shard-dir <h264-shard-1-mc3> `
  --out <h264-merged-json> --bootstrap 2000
```

Repeat with `--codec h265`. The `replay_report.json` copies from the two shards
should have the same source archive, pilot fingerprints and arm definitions.
