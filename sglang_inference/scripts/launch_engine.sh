#!/usr/bin/env bash
# Launch ONE P/D engine (TP-4, chip subset) on THIS host from a plain engine JSON (no router, no group).
#   launch_engine.sh <engine.json> <RUN_NAME> <engine-name>
# Logs: ~/pd_disagg/results/runs/<RUN_NAME>/<engine-name>/server.log
set -euo pipefail
recipe=${1:?engine.json}; run=${2:?RUN_NAME}; name=${3:?engine-name}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env_singlehost_v7x.sh"
RUN_DIR="$HOME/pd_disagg/results/runs/$run/$name"; mkdir -p "$RUN_DIR"; export RUN_DIR
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if (( free_gb < 20 )); then echo "[launch] root disk low (${free_gb} GB free): pruning inductor caches"; rm -rf "$HOME/.cache/sglang/inductor_dp1_dprank0_tp8" "$HOME/.cache/sglang/inductor_dp1_dprank0_tp4" 2>/dev/null || true; fi
holders=$(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | grep -v "^$(id -un)$" || true)
if [[ -n "$holders" ]]; then echo "[launch] REFUSING: TPU held by: $holders" >&2; exit 3; fi
chips=$(python3 -c "import json,sys; print(json.load(open('$recipe'))['runtime_envs'].get('TPU_VISIBLE_CHIPS','all'))")
export TORCH_TPU_TIER2_NAME="pd2_${name}_tp4_chips${chips//,/}_$(basename "$recipe" .json)"
export TORCH_TPU_TIER3_ROOT="$COMPILE_CACHE_ROOT/2p2d/$(basename "$recipe" .json)/$name"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/alisachen_cache/inductor_2p2d_$name"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_TPU_TIER3_ROOT"
export TPU_LOG_DIR="$RUN_DIR/tpu_logs"; mkdir -p "$TPU_LOG_DIR"
unset SGLANG_LAUNCHER_HOSTNAMES; export SGLANG_LAUNCHER_RANK=0
cd "$SGLANG_ROOT"
{ echo "[launch] $(date -u +%FT%TZ) host=$(hostname) engine=$name run=$run recipe=$recipe sglang=$(git rev-parse --short HEAD) chips=$chips";
  "$PYTHON" -m benchmark.tpu.launch_server --config "@$recipe" --dry-run 2>&1 | head -c 20000; } > "$RUN_DIR/launch_plan.txt" 2>&1
exec "$PYTHON" -u -m benchmark.tpu.launch_server --config "@$recipe" --shutdown-grace 60 >> "$RUN_DIR/server.log" 2>&1
