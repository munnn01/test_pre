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
OUT="/kaggle/working/outputs/v6_rate_matched/${CODEC}/shard_${SHARD}"
mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "v6_rate_matched_${CODEC}_s${SHARD}.tgz" \
    "outputs/v6_rate_matched/${CODEC}/shard_${SHARD}" 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/v6_rate_matched_${CODEC}_s${SHARD}.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/test_pre.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
test -n "$KIN_ROOT" || { echo 'Missing Kinetics dataset' >&2; exit 2; }
CACHE_ROOT=""
for candidate in "/kaggle/input/${DATASET_SLUG}" "/kaggle/input/datasets/${ACCOUNT}/${DATASET_SLUG}"; do
  if [ -d "$candidate" ]; then CACHE_ROOT="$candidate"; break; fi
done
test -n "$CACHE_ROOT" || { echo 'Missing private V2 pilot-cache dataset' >&2; exit 2; }
ARCHIVE="$CACHE_ROOT/dual_codec_search_v2_${CODEC}.tgz"
test -s "$ARCHIVE" || { echo "Missing pilot archive: $ARCHIVE" >&2; exit 2; }
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
python -m ops.v6_rate_matched_pilot evaluate --archive "$ARCHIVE" \
  --index "$INDEX" --codec "$CODEC" --shard "$SHARD" \
  --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
test -s "$OUT/shard_records.jsonl"
test -s "$OUT/shard_result.json"
echo "[done] codec=$CODEC shard=$SHARD V6 TRAIN feasibility only"
