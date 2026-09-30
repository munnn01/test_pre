# V12 frozen MC3 CAL audit

Four exact-version v1 Kaggle archives and raw records, 200 reused CAL source videos, 2264 unique trials.
No policy tuning, no fresh DEV or holdout. Both frozen policies and three controls are reported.

[Results and limitations](../../docs/RESULTS_V12_MC3_CAL.md). `mc3_result.json` contains curves, direct comparisons, paired 2000-draw CIs and provenance.
`assessment.json` contains per-QP correctness/flip counts, copied primary evidence and diagnostic timers.
`raw/shardN/` preserves original archives, manifests, records, run/server logs and exact-version download receipts.
`launch/shardN/` preserves submitted notebooks, private-GPU metadata and push logs.
`pytest_full_receipt.json`/`pytest_full.log` preserve the first full test run (42 temp-directory permission errors).
`pytest_full_retry_receipt.json`/`pytest_full_retry.log` preserve the retry with a fresh workspace basetemp (one order-dependent test-fixture failure).
`pytest_full_fixed_fixture_receipt.json`/`pytest_full_fixed_fixture.log` preserve the final passing full suite after fixing only the fixture order.
`artifacts_manifest.json` pins SHA-256 of every file in this directory except itself and its own sidecar.

Scorer and merge from preeee commit 43f62f3aff682b557bc00bbb4707479abbe268b7; preregistration commit fda9e069643e084d715c6f71a0cdefc9000f0204.
The test_pre copy has the same scorer provenance and is not a second experiment.
