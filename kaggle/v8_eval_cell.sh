set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
SHARD="__SHARD__"
CHECKPOINT_SLUG="__CHECKPOINT_SLUG__"
REPO=/kaggle/working/test_pre
INDEX=/kaggle/working/kinetics_hash_split.json
CACHE=/kaggle/working/paper_cache
OUT="/kaggle/working/outputs/v8_restorer/h264/test/shard_${SHARD}"
mkdir -p "$OUT" "$CACHE"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "v8_restorer_h264_test_s${SHARD}.tgz" \
    "outputs/v8_restorer/h264/test/shard_${SHARD}" 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/v8_restorer_h264_test_s${SHARD}.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/test_pre.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available())'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265'
KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
test -n "$KIN_ROOT" || { echo 'Missing Kinetics dataset' >&2; exit 2; }
CACHE_SOURCE=""
for candidate in /kaggle/input/v2-paper-cache-1000-20260924 /kaggle/input/datasets/qktttttttttt/v2-paper-cache-1000-20260924; do
  if [ -d "$candidate" ]; then CACHE_SOURCE="$candidate"; break; fi
done
test -n "$CACHE_SOURCE" || { echo 'Missing private V2 TEST cache' >&2; exit 2; }
if [ -d "$CACHE_SOURCE/h264/h264/shard_0" ]; then
  cp -r "$CACHE_SOURCE/h264/h264" "$CACHE/"
elif [ -d "$CACHE_SOURCE/h264/shard_0" ]; then
  cp -r "$CACHE_SOURCE/h264" "$CACHE/"
elif [ -f "$CACHE_SOURCE/h264.zip" ]; then
  unzip -q "$CACHE_SOURCE/h264.zip" -d "$CACHE"
else
  echo 'Missing cached H.264 TEST shard directories' >&2
  exit 2
fi
for s in 0 1; do
  test -s "$CACHE/h264/shard_${s}/shard_records.jsonl"
  test -s "$CACHE/h264/shard_${s}/manifest.json"
  test -s "$CACHE/h264/shard_${s}/shard_result.json"
done
CHECKPOINT=""
for candidate in \
  "/kaggle/input/${CHECKPOINT_SLUG}/v8_restorer_h264_train.tgz" \
  "/kaggle/input/datasets/qktttttttttt/${CHECKPOINT_SLUG}/v8_restorer_h264_train.tgz"; do
  if [ -s "$candidate" ]; then CHECKPOINT="$candidate"; break; fi
done
test -n "$CHECKPOINT" || { echo 'Missing private V8 checkpoint dataset' >&2; exit 2; }
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
python -m ops.v8_eval evaluate --index "$INDEX" \
  --checkpoint-archive "$CHECKPOINT" --shard "$SHARD" \
  --cache-dir "$CACHE/h264/shard_0" \
  --cache-dir "$CACHE/h264/shard_1" --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
test -s "$OUT/shard_records.jsonl"
test -s "$OUT/shard_result.json"
echo "[done] V8 H.264 TEST shard=$SHARD; merge both shards before reporting"
