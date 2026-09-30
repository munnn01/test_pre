# V12 publisher repair before any measurement

The first attempted submission, `shungg05/v12a-spatial-cal-s0-4107661`, stopped before `kernels push`: Kaggle's status endpoint returned `kernels.get` permission denied for a new notebook slug. A complete authenticated `mine=True` listing subsequently verified 38 owned notebooks, including existing V11 notebooks, and confirmed that this destination did not exist. No worker was submitted and no video measurement occurred in that attempt. Its payload, failure log and receipt are preserved with the launch artifacts.

The publisher now requires a complete authenticated owner listing, including all pagination tokens. It verifies the requested owner, rejects any existing destination (including completed notebooks), rejects missing or incomplete responses, and publishes only a fresh slug. Tests cover second-page collisions, owner mismatch, empty/missing responses and repeated cursors. Worker code is pinned to a new commit after this repair.

This is a submission repair only. Both preregistrations, source/config hashes, candidate grid, selection rules, seed, bootstrap implementation and gates remain unchanged. No CAL proxy measurements, primary outcomes, MC3 inference or holdout evaluations informed the repair. CAL proxy results, MC3 and holdout metrics are CHƯA ĐO.
