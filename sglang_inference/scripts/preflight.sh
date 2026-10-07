#!/usr/bin/env bash
# Section 2 pre-flight. Usage: bash preflight.sh [PEER_IP] ; writes to stdout.
PEER=${1:-}
PY=${PYTHON:-$HOME/venv/sglang/bin/python}
echo "### host: $(hostname)  $(hostname -I)  date: $(date -u +%FT%TZ)"
echo "--- A. software"
if [ -x "$PY" ]; then
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 $PY - <<'PY'
import importlib
for m in ["sglang","torch","torch_tpu","libtpu","jax","transformers","tpu_sync","sglang_router"]:
    try:
        mod=importlib.import_module(m); print(f"{m:14s} {getattr(mod,'__version__','?')}  {getattr(mod,'__file__','')}")
    except Exception as e: print(f"{m:14s} MISSING ({type(e).__name__}: {str(e)[:80]})")
PY
  $PY -m pip list 2>/dev/null | awk '/libtpu|sglang|torch|jax|transformers|tpu-sync/'
  (cd ~/sglang-torchtpu && echo "sglang-torchtpu commit: $(git rev-parse HEAD) $(git log -1 --format=%ci)"; git status --short | head -5)
else
  echo "venv python not present: $PY"
fi
echo "--- B. flags present in this build (from --help)"
if [ -x "$PY" ]; then
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=tpu $PY -m sglang.launch_server --help > /tmp/sglang_help_$$.txt 2>&1
  for f in --disaggregation-mode --disaggregation-transfer-backend --disaggregation-bootstrap-port \
           --disaggregation-decode-enable-radix-cache --disaggregation-decode-enable-offload-kvcache --disaggregation-decode-retraction-backup \
           --enable-hierarchical-cache --hicache-ratio --hicache-write-policy --hicache-mem-layout --hicache-io-backend \
           --speculative-algorithm --speculative-draft-model-path --enable-torch-compile --torch-compile-max-bs \
           --tpu-compile-tokens --tpu-compile-bs --tpu-compile-tracer --attention-backend --ep-size --enable-metrics \
           --watchdog-timeout --kv-cache-dtype --page-size --swa-full-tokens-ratio --disable-hybrid-swa-memory --enable-cache-report; do
    grep -q -- "$f" /tmp/sglang_help_$$.txt && echo "OK      $f" || echo "MISSING $f"
  done
  rm -f /tmp/sglang_help_$$.txt
fi
echo "--- C. hardware / OS"
echo "vfio devices: $(ls /dev/vfio/ 2>/dev/null | grep -cE '^[0-9]+$')"
free -g | awk '/Mem/{print "host RAM GiB total/free:", $2, $4}'
df -h / /dev/shm /mnt/data 2>/dev/null | tail -3
echo "ulimit -l: $(ulimit -l)  ulimit -n: $(ulimit -n)  nproc: $(nproc)"
echo "TPU in use by: $(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | tr '\n' ' ')"
echo "sglang procs: $(pgrep -fc 'sglang.launch_server|sglang::scheduler')"
echo "--- D. network"
if [ -n "$PEER" ]; then ping -c 3 -W 2 $PEER | tail -1; for p in 30000 30001 8998 31000; do (timeout 2 bash -c "</dev/tcp/$PEER/$p" 2>/dev/null && echo "peer port $p OPEN") || echo "peer port $p closed"; done; fi
echo "--- E. checkpoints"
H=${HF_HOME:-/mnt/data/tt/hf-cache}
for m in models--openai--gpt-oss-120b models--Qwen--Qwen3-VL-30B-A3B-Instruct models--lmsys--EAGLE3-gpt-oss-120b-bf16; do
  s=$(ls -d $H/hub/$m/snapshots/*/ 2>/dev/null | head -1); echo "$m -> ${s:-MISSING} ($(ls $s 2>/dev/null | grep -c safetensors) safetensors)"
done
