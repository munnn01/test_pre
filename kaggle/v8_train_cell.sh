set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
DATASET_SLUG="__DATASET_SLUG__"
ACCOUNT="__ACCOUNT__"
REPO=/kaggle/working/test_pre
INDEX=/kaggle/working/kinetics_hash_split.json
OUT=/kaggle/working/outputs/v8_restorer/h264/train
mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf v8_restorer_h264_train.tgz outputs/v8_restorer/h264/train 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/v8_restorer_h264_train.tgz"
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
test -n "$CACHE_ROOT" || { echo 'Missing private V2 pilot cache' >&2; exit 2; }
ARCHIVE="$CACHE_ROOT/dual_codec_search_v2_h264.tgz"
test -s "$ARCHIVE" || { echo "Missing pilot archive: $ARCHIVE" >&2; exit 2; }
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
python -m ops.v8_train --archive "$ARCHIVE" --index "$INDEX" \
  --codec h264 --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
test -s "$OUT/pixel.pth"
test -s "$OUT/semantic.pth"
test -s "$OUT/calibration.json"
echo '[done] H264 V8 TRAIN fit and calibration complete; MC3 was not loaded'
