# V12 semantic: CAL-only pilot

Preregistration commit `225e06509ebdd72744ae82c6b75a40ff6f2c16d1`; [locked protocol](PREREGISTRATION_V12_SEMANTIC_LOWQP.md), [source/config hashes](../configs/v12_lowqp/protocol.json). No DEV, TEST, holdout or MC3 scoring is included. All outcomes are CHƯA ĐO until all four raw archives pass validation. Repo legacy pipelines/results are retained unchanged.

The spatial direction can replace V6 with identity directly at QP 30–40. The semantic direction reranks the existing six streams with frozen ImageNet ResNet18 layer2/layer3 distance, also at QP 30–40. QP 45/50 retain V6. The same 200 reused CAL sources support paired comparisons between hypotheses; they are not independent holdout replications. Primary CAL outcomes are a separate hash-locked historical index, read only by local calibration after proxy extraction. MC3 is excluded from all CAL selection.

Four private notebooks (50 sources each), CPU for spatial and GPU for semantic. Every worker pins code/preregistration/config/source hashes, records normalized coded bytes, all trial encode/decode counts, decoded-pixel SHA-256, proxy/timing/environment; semantic also checks full official checkpoint/state hashes. No trial is omitted from pilot counts. This timing excludes the previously frozen V6 selector implementation and is not total selector overhead.

```powershell
& $py -B -X utf8 -m research.v12_lowqp.study --prereg-commit 225e06509ebdd72744ae82c6b75a40ff6f2c16d1 preflight
& $py -B -X utf8 -m research.v12_lowqp.push --commit $codeRevision --prereg-commit 225e06509ebdd72744ae82c6b75a40ff6f2c16d1 --account $account --slug $uniqueSlug --shard $shard --payload-dir $freshPayload
```

Outputs: `v12_semantic_cal_shard0.tgz` through `3.tgz`, each with exact prefix `outputs/v12_semantic_cal/shardN/`, members manifest.json, manifest.sha256, shard_records.jsonl, run.log. Preserve archives and verify safe paths/types/sizes before reading, then:

```powershell
& $py -B -u -X utf8 -m research.v12_lowqp.study --prereg-commit 225e06509ebdd72744ae82c6b75a40ff6f2c16d1 calibrate --shard-dir $s0 --shard-dir $s1 --shard-dir $s2 --shard-dir $s3 --out results/v12_lowqp/calibration_result.json
```

The complete 24-point grid uses only primary CAL correctness and the locked proxy. NO-GO stops before DEV. A selected policy and all CAL choices must be committed before developing a separate registered fresh DEV assessment. Selected-policy CIs use unchanged V8 curves/compare/summarize verbatim, original unchanged BD metric, 2,000 paired source-video draws (seed 20261009). CAL intervals are descriptive after selection, not confirmatory, and MC3 outcomes remain CHƯA ĐO. Future engineering and original −15% research gates remain unchanged.

## Deployment

Four private notebooks are submitted, pinned to worker commit `f75e6231ff7a29396d9eb66ee5591fdb1107f956`. [Launch record, account assignment, dated API status and hashes](V12_KAGGLE_LAUNCH.md). Outcomes remain CHƯA ĐO.
