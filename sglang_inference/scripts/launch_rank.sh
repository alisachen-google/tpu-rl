#!/usr/bin/env bash
# Launch one engine of an N-host P/D group on THIS host via the repo launcher.
#   PD_HOSTS=ip0,ip1,... launch_rank.sh <recipe.json> <rank> <RUN_NAME>
# Ranks below group.prefill_engines are prefills; rank 0 also starts the mini-LB on LB_PORT.
# Logs: ~/pd_disagg/results/runs/<RUN_NAME>/<label>/server.log (label p1,p2,d1,d2 ...).
set -euo pipefail
recipe=${1:?recipe.json}; rank=${2:?rank}; run=${3:?RUN_NAME}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env_singlehost_v7x.sh"
: "${PD_HOSTS:?comma list of engine host IPs in rank order}"
IFS=, read -r -a hosts <<< "$PD_HOSTS"
nprefill=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['group']['prefill_engines'])" "$recipe")
if (( rank < nprefill )); then role=prefill; label="p$((rank+1))"; else role=decode; label="d$((rank-nprefill+1))"; fi
RUN_DIR="$HOME/pd_disagg/results/runs/$run/$label"; mkdir -p "$RUN_DIR"; export RUN_DIR
shm_used_gb=$(df -BG --output=used /dev/shm | tail -1 | tr -dc 0-9)
if (( shm_used_gb > 300 )); then echo "[launch] /dev/shm at ${shm_used_gb} GB: pruning inductor cache"; rm -rf "$TORCHINDUCTOR_CACHE_DIR"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"; fi
holders=$(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | grep -v "^$(id -un)$" || true)
if [[ -n "$holders" ]]; then echo "[launch] REFUSING: TPU held by: $holders" >&2; exit 3; fi
# Tier-2 cache namespace follows the 1P/1D recipe name so the seeded caches hit (strip the _2p2d suffix).
base=$(basename "$recipe" .json); base=${base%_2p2d}
export TORCH_TPU_TIER2_NAME="pd_${role}_${base}"
export TORCH_TPU_TIER3_ROOT="$COMPILE_CACHE_ROOT/$base/$role"
export TPU_LOG_DIR="$RUN_DIR/tpu_logs"; mkdir -p "$TPU_LOG_DIR"
export SGLANG_LAUNCHER_HOSTNAMES="$PD_HOSTS" SGLANG_LAUNCHER_RANK=$rank SGLANG_HOST_IP=${hosts[$rank]}
cd "$SGLANG_ROOT"
{
  echo "[launch] $(date -u +%FT%TZ) host=$(hostname) rank=$rank label=$label role=$role run=$run recipe=$recipe sglang=$(git rev-parse --short HEAD)"
  echo "[launch] env: SGLANG_LAUNCHER_HOSTNAMES=$SGLANG_LAUNCHER_HOSTNAMES RANK=$SGLANG_LAUNCHER_RANK HOST_IP=$SGLANG_HOST_IP LB_PORT=$LB_PORT"
  "$PYTHON" -m benchmark.tpu.launch_server --config "@$recipe" --dry-run 2>&1 | head -c 20000
} > "$RUN_DIR/launch_plan.txt" 2>&1
exec "$PYTHON" -u -m benchmark.tpu.launch_server --config "@$recipe" --shutdown-grace 60 >> "$RUN_DIR/server.log" 2>&1
