# Paper-validation reanalysis on the existing 1,000-clip TEST set

This is an **exploratory reanalysis** of the previously inspected V1/V2 TEST
sample, not a new independent holdout. No policy was fitted or tuned using these
records. The pilot-frozen A/B/C parameters and the same 1,000 paired clips per
codec were used throughout. All five non-identity fixed transforms are reported
in [H.264](h264_paper_validation.md) and [H.265](h265_paper_validation.md),
along with V1 and A/B/C. JSON files retain full curves, choices, exact source
SHA-256 hashes and 2,000 whole-video bootstrap resamples.

| Codec | Arm | `r2plus1d_18` BD-rate | `r3d_18` BD-rate | Worst analyzer |
|---|---|---:|---:|---:|
| H.264 | best fixed transform for worst-analyzer BD-rate (`area112`) | -3.38% | -0.65% | -0.65% |
| H.264 | V1 | -24.88% | -1.03% | -1.03% |
| H.264 | A | -12.88% | -9.34% | -9.34% |
| H.264 | B | -16.33% | -11.64% | -11.64% |
| H.264 | C | **-22.01%** | **-14.47%** | **-14.47%** |
| H.265 | best fixed transform for worst-analyzer BD-rate (`area112`) | -0.34% | +0.40% | +0.40% |
| H.265 | V1 | -16.09% | -0.99% | -0.99% |
| H.265 | A | -8.86% | -6.14% | -6.14% |
| H.265 | B | -11.02% | -7.77% | -7.77% |
| H.265 | C | **-14.03%** | **-8.72%** | **-8.72%** |

The *post-hoc descriptive* C-vs-B paired BD-rate is -6.61%/-3.24% on H.264
and -3.36%/-1.10% on H.265 for `r2plus1d_18`/`r3d_18`, respectively.
These are direct C-vs-B BD-rates, not arithmetic differences between columns.
The comparison favors C, but because the TEST was seen during earlier research,
the intervals should not be presented as pristine confirmatory evidence.

The fixed-transform row is selected **only for display in this overview**;
the complete per-transform tables show every outcome. Do not claim that the
displayed fixed transform was prospectively selected. A 20-clip, QP-40
[runtime pilot](../paper_runtime_v2_qp40_20/README.md) has since measured
encode+decode search wall time; production encoder-only latency and memory,
a third unseen analyzer, and a genuinely new source-disjoint holdout remain to be
measured; see [the validation plan](../../docs/PAPER_VALIDATION_PLAN.md).

To reproduce from the downloaded per-clip records:

```powershell
python -m ops.paper_validation --codec h264 `
  --shard-dir <h264-shard-0> --shard-dir <h264-shard-1> `
  --out-dir results/paper_validation_1000 --bootstrap 2000
python -m ops.paper_validation --codec h265 `
  --shard-dir <h265-shard-0> --shard-dir <h265-shard-1> `
  --out-dir results/paper_validation_1000 --bootstrap 2000
```
