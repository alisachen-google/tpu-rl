#!/usr/bin/env bash
# Single-host isolation + static serving env for 1P1D on wyzhang-tpu7x-16 (2x2x2 slice, one engine per host).
# Source this on BOTH hosts before launching anything. Does NOT kill processes (see cleanup.sh).
# Adapted from sglang-torchtpu benchmark/tpu/common_env.sh + benchmark/tpu/launch_server.py single-host handling.
export SGLANG_ROOT=${SGLANG_ROOT:-$HOME/sglang-torchtpu}
export PYTHON=${PYTHON:-$HOME/venv/sglang/bin/python}
export PYTHONPATH="$SGLANG_ROOT/python:$SGLANG_ROOT/sgl-model-gateway/bindings/python/src:$SGLANG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# torch_tpu registers the dynamo "tpu" backend via autoload; client tools disable it, servers must not inherit that.
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
# Hosts
export PREFILL_IP=${PREFILL_IP:-10.128.15.217}   # wyzhang-tpu7x-16-a / t1v-n-eb299eb1-w-0
export DECODE_IP=${DECODE_IP:-10.128.0.90}       # wyzhang-tpu7x-16-b / t1v-n-eb299eb1-w-1
export SERVER_PORT=${SERVER_PORT:-30000}
export BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
export LB_PORT=${LB_PORT:-8080}
# Weights (NFS mirror of the HF cache; identical path on both hosts)
export HF_HOME=${HF_HOME:-/mnt/data/tt/hf-cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# --- single-host isolation: each engine owns only its own host (4 chips / 8 cores) ---
unset TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_TPU_TOPOLOGY TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT TPU_MULTIHOST_BACKEND
export TPU_WORKER_HOSTNAMES=$(hostname) TPU_WORKER_ID=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29500}
# --- static serving knobs (common_env.sh) ---
source "$SGLANG_ROOT/benchmark/tpu/common_env.sh"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCH_LOGS=${TORCH_LOGS:-+recompiles}
# --- disaggregation / raiden ---
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-7200}
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-7200}
export TPU_RAIDEN_CONTROL_PLANE_BACKEND=${TPU_RAIDEN_CONTROL_PLANE_BACKEND:-grpc}
export SGLANG_RAIDEN_KV_PORT=${SGLANG_RAIDEN_KV_PORT:-31000}
export SGLANG_RAIDEN_TIMEOUT_S=${SGLANG_RAIDEN_TIMEOUT_S:-600}      # raise for 1M transfers after iperf3 measurement
export SGLANG_RAIDEN_HOST_POOL_GB=${SGLANG_RAIDEN_HOST_POOL_GB:-128}
# GPT-OSS: SparseCore MoE gather path unsupported
export SGLANG_TPU_FUSED_MOE_SC_GATHER=${SGLANG_TPU_FUSED_MOE_SC_GATHER:-0}
# --- compile cache: ON for dev iterations, set ENABLE_COMPILE_CACHE=0 for TC-0 / TC-5 ---
export ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1}
# Compile caches live on the RAM-backed tmpfs (473 GiB): a compiled MoE decoder writes ~20 GB of inductor cache + ~6 GB tier-3 per launch and filled the 96 GB root disk twice.
export COMPILE_CACHE_ROOT=${COMPILE_CACHE_ROOT:-/dev/shm/alisachen_cache/tier3}
# Inductor cache grows ~4-5 GB per launch and is never pruned (filled a 96 GB disk in one day); keep it under the campaign dir.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/alisachen_cache/inductor}
mkdir -p "$COMPILE_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
echo "[env] host=$(hostname) ip=$(hostname -I | awk '{print $1}') TPU_WORKER_HOSTNAMES=$TPU_WORKER_HOSTNAMES python=$PYTHON sglang=$(git -C $SGLANG_ROOT rev-parse --short HEAD 2>/dev/null)"
