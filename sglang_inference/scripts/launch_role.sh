#!/usr/bin/env bash
# Launch one 1P1D role (or the monolithic golden) on THIS host via the repo's Python launcher.
#   launch_role.sh <recipe.json> prefill|decode|golden <RUN_NAME>
# Rank 0 (prefill host) also starts the mini-LB on LB_PORT once both engines are healthy.
# Logs: ~/pd_disagg/results/runs/<RUN_NAME>/<role>/server.log ; launch_meta.json in the same dir.
set -euo pipefail
recipe=${1:?recipe.json}; role=${2:?prefill|decode|golden}; run=${3:?RUN_NAME}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env_singlehost_v7x.sh"
RUN_DIR="$HOME/pd_disagg/results/runs/$run/$role"; mkdir -p "$RUN_DIR"
export RUN_DIR
# Disk guard: the inductor cache (~/.cache/sglang, $TORCHINDUCTOR_CACHE_DIR) grows 4-5 GB per launch; prune when the root disk is low.
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if (( free_gb < 20 )); then echo "[launch] root disk low (${free_gb} GB free): pruning inductor caches"; rm -rf "$HOME/.cache/sglang/inductor_dp1_dprank0_tp8" "$COMPILE_CACHE_ROOT/inductor"; df -h / | tail -1; fi
# tmpfs guard: compile caches live in /dev/shm; prune the inductor cache when the tmpfs is above 300 GB used (a compiled MoE decoder adds ~65 GB).
shm_used_gb=$(df -BG --output=used /dev/shm | tail -1 | tr -dc 0-9)
if (( shm_used_gb > 300 )); then echo "[launch] /dev/shm at ${shm_used_gb} GB: pruning inductor cache"; rm -rf /dev/shm/alisachen_cache/inductor_dp1_dprank0_tp8 "$TORCHINDUCTOR_CACHE_DIR"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"; df -h /dev/shm | tail -1; fi
# Refuse to start if another user's engine holds the TPU.
holders=$(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | grep -v "^$(id -un)$" || true)
if [[ -n "$holders" ]]; then echo "[launch] REFUSING: TPU held by: $holders" >&2; exit 3; fi
# Compile caches (tier-2 name per role so prefill/decode never share a bucket set; tier-3 on local disk).
export TORCH_TPU_TIER2_NAME="pd_${role}_$(basename "$recipe" .json)"
export TORCH_TPU_TIER3_ROOT="$COMPILE_CACHE_ROOT/$(basename "$recipe" .json)/$role"
export TPU_LOG_DIR="$RUN_DIR/tpu_logs"; mkdir -p "$TPU_LOG_DIR"
export SGLANG_LAUNCHER_HOSTNAMES="$PREFILL_IP,$DECODE_IP"
case "$role" in
  prefill) export SGLANG_LAUNCHER_RANK=0 SGLANG_HOST_IP=$PREFILL_IP ;;
  decode)  export SGLANG_LAUNCHER_RANK=1 SGLANG_HOST_IP=$DECODE_IP ;;
  golden)  unset SGLANG_LAUNCHER_HOSTNAMES; export SGLANG_LAUNCHER_RANK=0 ;;
  *) echo "bad role $role" >&2; exit 2 ;;
esac
cd "$SGLANG_ROOT"
{
  echo "[launch] $(date -u +%FT%TZ) host=$(hostname) role=$role run=$run recipe=$recipe sglang=$(git rev-parse --short HEAD) compile_cache=$ENABLE_COMPILE_CACHE"
  echo "[launch] env: SGLANG_LAUNCHER_HOSTNAMES=${SGLANG_LAUNCHER_HOSTNAMES:-} RANK=$SGLANG_LAUNCHER_RANK LB_PORT=$LB_PORT RAIDEN_TIMEOUT=$SGLANG_RAIDEN_TIMEOUT_S"
  "$PYTHON" -m benchmark.tpu.launch_server --config "@$recipe" --dry-run 2>&1 | head -c 20000
} > "$RUN_DIR/launch_plan.txt" 2>&1
echo "[launch] plan written to $RUN_DIR/launch_plan.txt; starting server (log: $RUN_DIR/server.log)"
exec "$PYTHON" -u -m benchmark.tpu.launch_server --config "@$recipe" --shutdown-grace 60 >> "$RUN_DIR/server.log" 2>&1
