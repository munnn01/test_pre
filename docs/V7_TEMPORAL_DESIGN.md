# V7 TRAIN feasibility: temporal residual before H.264/H.265

V6 showed sparse label-aware rescue from joint resolution/QP choice at
matched rate. V7 tests a different failure mode: temporal details may be
damaged even when a spatial rate match exists. This is a **feasibility gate**,
not a trained or deployable selector.

- Freeze V2's six-candidate decision and analyze the same first 30 TRAIN-fit
  clip IDs in two shards per codec. Do not revisit VAL or TEST while choosing
  the candidate family.
- On the V2-selected uncompressed representation, try two deterministic
  temporal residual transforms: a neighboring-frame denoise and a mild
  temporal unsharp. At a detected hard cut, do not mix across the boundary.
  Both transforms preserve frame count, dimensions, and temporal order.
- Encode with the real x264/x265 codec at QP offsets
  `{-4,-2,0,+2,+4}` around QP30/35/40, interpolate integer QPs where rates
  bracket the V2 rate, and run three frozen analyzers only on the nearest
  measured candidates. The matched analysis requires a real-bitstream
  bpp ratio in `[0.95,1.02]` versus the frozen V2 output. Verify re-encoded
  V2 bpp against the archived candidate cache exactly.
- Report strict matched-rate coverage; a diagnostic *label-aware* MC3 rescue
  ceiling that forbids new r2plus1d/r3d errors; MC3 regressions that would
  occur if one selected poorly; and measured search compute.

An oracle rescue is not a practical result: it reads the true action label.
MC3 is now a development objective, not an independent analyzer. Only if the
TRAIN gate shows a nontrivial, consistent opportunity should a separate
label-free selector be fit on TRAIN-fit, calibrated on disjoint TRAIN-cal,
then evaluated once on VAL with clip-level uncertainty and runtime.
