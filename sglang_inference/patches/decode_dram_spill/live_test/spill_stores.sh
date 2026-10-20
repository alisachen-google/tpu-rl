#!/usr/bin/env bash
# Decode-host side of the DRAM spill feature: Raiden global registry + per-rank/pool host-store
# processes on THIS host (b, 10.128.0.90). Prefill (host a) writes one manifest per rank/pool at
# startup; this script copies them over and launches the stores from them.
#
#   spill_stores.sh registry-start | registry-stop
#   spill_stores.sh stores-start <RUN> [budget_bytes_per_store]   # default 16 GiB per store
#   spill_stores.sh stores-stop
#   spill_stores.sh status
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ME=alisachen_google_com
VENV=$HOME/venv/sglang
SPILL_ROOT=$HOME/sglang-spill
PREFILL_IP=10.128.15.217; DECODE_IP=10.128.0.90
REGISTRY_PORT=${REGISTRY_PORT:-50051}
MANIFEST_DIR_A=/home/$ME/pd_disagg/spill/manifests          # written by the prefill engine on host a
LOGS=$HERE/logs; mkdir -p "$LOGS" "$HERE/pids"
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 sa_101411256518727260534@$PREFILL_IP"
log() { echo "$(date -u '+%F %T') $*"; }

stop_pidfile() { # name
  local f=$HERE/pids/$1.pid; [[ -f $f ]] || { log "$1: not running"; return 0; }
  local pid; pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"; fi
  rm -f "$f"; log "$1: stopped (pid $pid)"
}

case "${1:-}" in
  registry-start)
    if [[ -f $HERE/pids/registry.pid ]] && kill -0 "$(cat "$HERE/pids/registry.pid")" 2>/dev/null; then log "registry already running"; exit 0; fi
    nohup "$VENV/bin/global_registry_server" --port="$REGISTRY_PORT" > "$LOGS/registry.log" 2>&1 < /dev/null &
    echo $! > "$HERE/pids/registry.pid"; sleep 1
    kill -0 "$(cat "$HERE/pids/registry.pid")" 2>/dev/null && log "registry up on $DECODE_IP:$REGISTRY_PORT (pid $(cat "$HERE/pids/registry.pid"))" || { log "registry FAILED"; tail -5 "$LOGS/registry.log"; exit 1; } ;;
  registry-stop) stop_pidfile registry ;;
  stores-start)
    run=${2:?RUN}; budget=${3:-17179869184}
    dest=$HERE/manifests_b/$run; rm -rf "$dest"; mkdir -p "$dest"
    # Manifests appear when the prefill scheduler attaches the storage backend (early in startup).
    for _ in $(seq 1 120); do
      n=$($SSH_A "sudo find $MANIFEST_DIR_A -name '*.json' 2>/dev/null | wc -l" 2>/dev/null | tr -dc 0-9); n=${n:-0}
      (( n > 0 )) && break; sleep 5
    done
    (( n > 0 )) || { log "no manifests on host a after 10 min"; exit 1; }
    sleep 10  # let all ranks/pools finish writing
    rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' \
      "sa_101411256518727260534@$PREFILL_IP:$MANIFEST_DIR_A/" "$dest/" 2>&1 | grep -v -i 'permanently added'
    log "manifests: $(ls "$dest"/*.json | wc -l) files -> $dest"
    export SGLANG_ROOT=$SPILL_ROOT PYTHONPATH=$SPILL_ROOT/python:$SPILL_ROOT PATH=$VENV/bin:$PATH
    nohup "$VENV/bin/python" -m sglang_tpu.raiden_dram.launch_stores --manifest "$dest"/*.json --replica 0 \
      --store-server-ip "$DECODE_IP" --dram-budget-bytes-per-store "$budget" --binary "$VENV/bin/kv_cache_host_store_node_main" \
      > "$LOGS/stores_$run.log" 2>&1 < /dev/null &
    echo $! > "$HERE/pids/stores.pid"; sleep 5
    kill -0 "$(cat "$HERE/pids/stores.pid")" 2>/dev/null && log "stores launched (pid $(cat "$HERE/pids/stores.pid"), budget $budget B each, log $LOGS/stores_$run.log)" || { log "stores FAILED"; tail -20 "$LOGS/stores_$run.log"; exit 1; } ;;
  stores-stop) stop_pidfile stores ;;
  status)
    echo "registry: $(pgrep -u $ME -f 'global_registry_server' | tr '\n' ' ')"
    echo "store nodes: $(pgrep -u $ME -c -f 'kv_cache_host_store_node_main') procs; RSS total $(ps -u $ME -o rss=,comm= | awk '$2 ~ /kv_cache_host/ {s+=$1} END {printf "%.1f GiB", s/1048576}')"
    free -g | sed -n 2p ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac
