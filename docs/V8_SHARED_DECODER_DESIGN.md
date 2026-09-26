# V8 shared decoder restoration: locked H.264 development protocol

## Goal and interpretation

The measured frozen V2 H.264 Top-1 BD-rates on the historical 1,000 clips
are -22.01% (`r2plus1d_18`), -14.47% (`r3d_18`) and -3.88%
(`mc3_18`). V8 tests whether a **single restored video** after V2 H.264
decoding can retain the first two gains and improve transfer to MC3. The
target is BD-rate < -10% for all three models, with BD-accuracy > 0. This is
a prospective target, not a claim that V8 has achieved it.

MC3 has influenced previous V4-V7 studies. It is excluded from this V8 fit
and checkpoint selection, but its later readout must not be described as the
first-ever unseen analyzer. The historical TEST has also been inspected.

## Frozen training and calibration

- The codec and V2 policy remain frozen. An H.264 pilot archive is validated
  against committed V2 artifacts; for each clip/QP the archived policy picks
  one representation and its real x264 encode/decode is re-created. Measured
  bpp must match the archive exactly. No extra transmitted metadata is used.
- The 400 V2 TRAIN-fit clips fit one small decoder restorer. Each clip is
  visited for three epochs, with a rotating QP from 30/35/40/45/50. Two
  pixel/temporal epochs produce `pixel.pth`; one subsequent epoch adds equal
  layer-2 feature losses from frozen `r2plus1d_18` and `r3d_18`, producing
  `semantic.pth`. All 16 decoded frames share one restorer.
- The separate 200 V2 TRAIN-calibration clips score both checkpoints and the
  unmodified V2 output at QP30/35/40 on the two allowed analyzers. A
  checkpoint may lose at most two correct clips out of 200 for each
  analyzer/QP. Among qualifying checkpoints with lower mean cross-entropy
  than V2, choose the lowest mean cross-entropy; otherwise keep V2. This
  decision is saved in `calibration.json` before any MC3 evaluation.
- Both checkpoints and the calibration report remain available even if the
  fallback is V2. The selector never reads MC3 weights, labels or logits.

## Required paired evaluation

After Stage 1 completes, evaluate the locked choice at all five QPs on the
same 1,000 old TEST clips, with an identical decoded stream supplied to all
three analyzers. Compare codec-only, V2, codec-only plus the selected
restorer, and V2 plus the selected restorer. Use real coded bytes; report
curves, Top-1 BD-rate, BD-accuracy, paired whole-clip bootstrap intervals,
selection provenance, decoder latency and memory. The 1,000 historical
clips provide a paired development replication, not a new-source holdout.

The restorer changes decoded pixels but not V2 transmitted bytes. This
prediction is an implementation property; it does not imply a BD-rate gain,
which depends on measured accuracy at all QPs. Any external data or model
claim requires a separate source-audited study.
