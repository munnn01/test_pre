set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
CODEC="__CODEC__"
QPS="__QPS__"
CLIPS="__CLIPS__"
REPO=/kaggle/working/test_pre
INDEX=/kaggle/working/kinetics_hash_split.json
OUT="/kaggle/working/outputs/paper_runtime/${CODEC}_qp${QPS//,/x}"
mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "paper_runtime_${CODEC}_qp${QPS//,/x}.tgz" \
    "outputs/paper_runtime/${CODEC}_qp${QPS//,/x}" 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/paper_runtime_${CODEC}_qp${QPS//,/x}.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/test_pre.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, cv2; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available()); assert torch.cuda.is_available(), "GPU is required"'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265'
KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
test -n "$KIN_ROOT" || { echo 'Missing qktttttttttt/kineticscleaned' >&2; exit 2; }
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
python -m ops.paper_runtime --index "$INDEX" --codec "$CODEC" \
  --split val --clips "$CLIPS" --qps "$QPS" --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
test -s "$OUT/runtime_result.json"
echo "[done] codec=$CODEC qps=$QPS clips=$CLIPS"
