%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
PREREG="__PREREG__"
SHARD="__SHARD__"
DIRECTION="__DIRECTION__"
REPO=/kaggle/working/v12_repository
OUT="/kaggle/working/outputs/v12_${DIRECTION}_cal/shard${SHARD}"
LOG="/kaggle/working/v12_${DIRECTION}_cal_shard${SHARD}.log"
finish() {
  rc=$?
  trap - EXIT
  set +e
  if [ -d "$OUT" ]; then
    cp "$LOG" "$OUT/run.log"
    cd /kaggle/working
    tar -czf "v12_${DIRECTION}_cal_shard${SHARD}.tgz" "outputs/v12_${DIRECTION}_cal/shard${SHARD}"
  fi
  echo "[exit] rc=$rc direction=$DIRECTION shard=$SHARD"
  exit "$rc"
}
trap finish EXIT
git clone -q __GITHUB__ "$REPO"
git -C "$REPO" checkout -q "$REF"
test "$(git -C "$REPO" rev-parse HEAD)" = "$REF"
cd "$REPO"
python -m research.v12_lowqp.study --prereg-commit "$PREREG" preflight
ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then ROOT="$candidate"; break; fi
done
test -n "$ROOT" || { echo 'Locked CAL dataset missing' >&2; exit 2; }
python -m research.v12_lowqp.study --prereg-commit "$PREREG" shard --source-root "$ROOT" --shard "$SHARD" --out-dir "$OUT" 2>&1 | tee "$LOG"
test -s "$OUT/manifest.json"
test -s "$OUT/shard_records.jsonl"
echo "[done] V12 $DIRECTION CAL shard=$SHARD"
