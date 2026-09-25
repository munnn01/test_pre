set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
CODEC="__CODEC__"
SHARD="__SHARD__"
DATASET_SLUG="__DATASET_SLUG__"
ACCOUNT="__ACCOUNT__"
REPO=/kaggle/working/test_pre
INDEX=/kaggle/working/kinetics_hash_split.json
OUT="/kaggle/working/outputs/dual_v4_mc3_fit/${CODEC}/shard_${SHARD}"
mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "dual_v4_mc3_fit_${CODEC}_s${SHARD}.tgz" \
    "outputs/dual_v4_mc3_fit/${CODEC}/shard_${SHARD}" 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/dual_v4_mc3_fit_${CODEC}_s${SHARD}.tgz"
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
test -n "$KIN_ROOT" || { echo 'Missing Kinetics dataset' >&2; exit 2; }
CACHE_ROOT=""
for candidate in "/kaggle/input/${DATASET_SLUG}" "/kaggle/input/datasets/${ACCOUNT}/${DATASET_SLUG}"; do
  if [ -d "$candidate" ]; then CACHE_ROOT="$candidate"; break; fi
done
test -n "$CACHE_ROOT" || { echo 'Missing private pilot-cache dataset' >&2; exit 2; }
ARCHIVE="$CACHE_ROOT/dual_codec_search_v2_${CODEC}.tgz"
test -s "$ARCHIVE" || { echo "Missing pilot archive: $ARCHIVE" >&2; exit 2; }
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
python -m ops.v4_mc3_fit_cache evaluate --archive "$ARCHIVE" \
  --index "$INDEX" --codec "$CODEC" --shard "$SHARD" \
  --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
test -s "$OUT/fit_records.jsonl"
test -s "$OUT/calibration_records.jsonl"
test -s "$OUT/shard_result.json"
echo "[done] codec=$CODEC shard=$SHARD V4 TRAIN labels only; merge both shards"
