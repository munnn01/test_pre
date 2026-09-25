# V2-C runtime pilot: 20 paired Kinetics clips at QP 40

Two private Kaggle notebooks completed on 2026-09-25 using commit
`b25f75d767f24163a1edc6f976c52c7051981057`: [H.264 on baoancut](https://www.kaggle.com/code/baoancut/paper-runtime-v2-h264-qp40)
and [H.265 on trnhlng](https://www.kaggle.com/code/trnhlng/paper-runtime-v2-h265-qp40).
Both used the same 20 preselected Kinetics-cleaned validation clip IDs,
the same index SHA-256, 16 decoded frames at 128 px, QP 40, FFmpeg preset
`medium`, two frozen TorchVision analyzers, and a Tesla T4 with four logical
CPUs. Different Kaggle workers ran the codecs; codec-to-codec speed rankings
are therefore not a controlled comparison.

| Codec | Codec-only median / p95 | Full selector median / p95 | Paired overhead-ratio median / p95 |
|---|---:|---:|---:|
| H.264 | 0.443 / 0.660 s | 2.672 / 3.057 s | 6.22× / 6.88× |
| H.265 | 0.449 / 0.703 s | 2.612 / 2.939 s | 6.15× / 6.93× |

The codec-only arm performs **one encode+decode and no encoder-side analyzer
pass** per clip. The full arm generates six candidates, performs **six
encode+decode calls**, runs both analyzers on the clean source and each
reconstruction (14 analyzer passes), and selects one stream. Source-video
decode is shared between arms and included equally. Model download/startup
is excluded. These are Python/FFmpeg **encode+decode search wall times**, not
production encoder-only latency, end-to-end streaming latency, or energy use.
The ratio is computed per paired clip before taking its median; it is not the
quotient of the two displayed medians. With just 20 clips, these are
descriptive pilot timings, not a confidence interval or a general hardware
benchmark.

The full downloadable Kaggle outputs include per-clip timing records and
manifests. Locally audited `runtime_result.json` SHA-256:

- H.264: `3d3afc54975b8c38aaed83a1c191a9d321c8d4390ce606e7f2c3ddf10df616d2`
- H.265: `183421076eccf0882122bb7e22f2f3c6bee42b4a4a1fec02f232e4a9f3ccdcbcb`

This pilot measures compute overhead only. It neither changes the previously
reported Top-1 BD-rate nor supplies a new source-disjoint holdout.
