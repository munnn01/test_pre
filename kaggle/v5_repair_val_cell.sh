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
OUT="/kaggle/working/outputs/dual_v5_repair_val/${CODEC}/shard_${SHARD}"
mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "dual_v5_repair_val_${CODEC}_s${SHARD}.tgz" \
    "outputs/dual_v5_repair_val/${CODEC}/shard_${SHARD}" 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/dual_v5_repair_val_${CODEC}_s${SHARD}.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/test_pre.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
test -s "configs/dual_codec_v5_frozen/${CODEC}/frozen_policy.json" || {
  echo 'Missing committed, frozen V5 policy' >&2; exit 2;
}
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
python -m ops.dual_codec_v5_eval evaluate --archive "$ARCHIVE" \
  --v5-dir "configs/dual_codec_v5_frozen/${CODEC}" --index "$INDEX" \
  --codec "$CODEC" --shard "$SHARD" --out-dir "$OUT" \
  --bootstrap 100 2>&1 | tee "$OUT/run.log"
test -s "$OUT/shard_records.jsonl"
test -s "$OUT/shard_result.json"
echo "[done] codec=$CODEC shard=$SHARD V5 sparse-repair VAL development comparison only"
