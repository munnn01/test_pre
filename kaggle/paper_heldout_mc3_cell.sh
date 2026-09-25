set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
CODEC="__CODEC__"
SHARD="__SHARD__"
REPO=/kaggle/working/test_pre
INDEX=/kaggle/working/kinetics_hash_split.json
CACHE=/kaggle/working/paper_cache
OUT="/kaggle/working/outputs/paper_heldout_mc3/${CODEC}/shard_${SHARD}"
mkdir -p "$OUT" "$CACHE"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "paper_heldout_mc3_${CODEC}_s${SHARD}.tgz" \
    "outputs/paper_heldout_mc3/${CODEC}/shard_${SHARD}" 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/paper_heldout_mc3_${CODEC}_s${SHARD}.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/test_pre.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, cv2; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available())'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265'
KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
test -n "$KIN_ROOT" || { echo 'Missing qktttttttttt/kineticscleaned' >&2; exit 2; }
CACHE_SOURCE=""
for candidate in /kaggle/input/v2-paper-cache-1000-20260924 /kaggle/input/datasets/qktttttttttt/v2-paper-cache-1000-20260924; do
  if [ -d "$candidate" ]; then CACHE_SOURCE="$candidate"; break; fi
done
test -n "$CACHE_SOURCE" || { echo 'Missing private V2 cache dataset' >&2; exit 2; }
if [ -d "$CACHE_SOURCE/$CODEC/$CODEC/shard_0" ]; then
  cp -r "$CACHE_SOURCE/$CODEC/$CODEC" "$CACHE/"
elif [ -d "$CACHE_SOURCE/$CODEC/shard_0" ]; then
  cp -r "$CACHE_SOURCE/$CODEC" "$CACHE/"
elif [ -f "$CACHE_SOURCE/$CODEC.zip" ]; then
  unzip -q "$CACHE_SOURCE/$CODEC.zip" -d "$CACHE"
else
  echo "Missing cached $CODEC shard directories" >&2
  exit 2
fi
for s in 0 1; do
  test -s "$CACHE/$CODEC/shard_${s}/shard_records.jsonl"
  test -s "$CACHE/$CODEC/shard_${s}/manifest.json"
  test -s "$CACHE/$CODEC/shard_${s}/shard_result.json"
done
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
python -m ops.paper_heldout_mc3 evaluate --index "$INDEX" \
  --codec "$CODEC" --shard "$SHARD" \
  --cache-dir "$CACHE/$CODEC/shard_0" \
  --cache-dir "$CACHE/$CODEC/shard_1" --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
test -s "$OUT/shard_records.jsonl"
test -s "$OUT/shard_result.json"
echo "[done] mc3 codec=$CODEC shard=$SHARD; merge both shards before reporting"
