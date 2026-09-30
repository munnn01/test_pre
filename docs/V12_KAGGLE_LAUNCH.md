# V12 semantic: Kaggle launch record

Worker commit `f75e6231ff7a29396d9eb66ee5591fdb1107f956`; preregistration commit `225e06509ebdd72744ae82c6b75a40ff6f2c16d1`. Four private notebooks were successfully pushed and the authenticated API reported RUNNING for all four at `2026-09-30T11:59:37.095308+00:00`. This is a dated status snapshot, not a completion claim. See [submission receipt](../results/v12_lowqp_launch/launch_receipt.json) and [status snapshot](../results/v12_lowqp_launch/status_snapshot.json).

| Shard | Account | Notebook | Snapshot |
|---|---|---|---|
| 0 | trmnguyn111 | [50 sources](https://www.kaggle.com/code/trmnguyn111/v12b-semantic-cal-s0-f75e623) | RUNNING |
| 1 | shungg05 | [50 sources](https://www.kaggle.com/code/shungg05/v12b-semantic-cal-s1-f75e623) | RUNNING |
| 2 | baoancut | [50 sources](https://www.kaggle.com/code/baoancut/v12b-semantic-cal-s2-f75e623) | RUNNING |
| 3 | trnhlng | [50 sources](https://www.kaggle.com/code/trnhlng/v12b-semantic-cal-s3-f75e623) | RUNNING |

Exactly 200 reused CAL source videos, 50 per shard, fingerprint `ce5acf9334d4f683fdc7757d53b0d5be0ce80cc5a6fd4a4340e30c9357e79721`. Planned unique encode/decode trials: 6000; actual completed trials: CHƯA ĐO. Spatial uses CPU; semantic requires GPU and verifies frozen ResNet18 checkpoint/state hashes. All notebook payloads and submission logs are committed under `results/v12_lowqp_launch/` with SHA-256 in its artifact manifest.

Two initial attempts stopped before `kernels push`: nonexistent-slug status lookup and malformed owner-list metadata. Failed payloads/logs/receipts are preserved in the spatial repo. The six verified accounts run eight notebooks in total; existing successful jobs were not resubmitted. See [technical amendment](V12_PUBLISHER_TECHNICAL_AMENDMENT.md).

Tests: original implementation whole suite 463 passed / 4 warnings; after publisher repair, all 19 V12 tests passed. The whole suite was not rerun after that isolated repair. [Evidence](../results/v12_lowqp_launch/testing/test_results.json).

No CAL results are merged yet. BD-rate, BD-accuracy, same-QP gaps, CIs, MC3, fresh DEV, holdout and total encoder overhead are **CHƯA ĐO**. No gate success is claimed. After four raw archives validate, local calibration uses the locked primary CAL outcomes, unchanged BD/paired-bootstrap functions, 2,000 source-video draws and seed 20261009. CAL intervals are descriptive after selection. NO-GO stops this direction; a chosen policy must be committed before a separately registered fresh DEV study. Both alternatives share sources and are not independent replications. No holdout evaluation or MC3-based selection has been launched.
