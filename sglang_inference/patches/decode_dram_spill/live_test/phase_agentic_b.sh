#!/usr/bin/env bash
# Agentic variant B: 12 sessions x 10 turns, 16k shared prefix + 112k private context, 2k tool tokens, 256 out.
# Store half runs in buffer_only host-memory mode (host tier = staging to L3) so recalls can land.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--sessions 12 --turns 10 --shared 16384 --prefix 114688 --turn-new 2048 --turn-out 256 --think-min 1 --think-max 4 --stagger 2"
launch_dram() { # recipe run
  bash $D/pd_spill.sh stop all; bash $D/spill_stores.sh stores-stop; sleep 15
  ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 'sudo rm -f /home/alisachen_google_com/pd_disagg/spill/manifests/*.json'
  bash $D/spill_stores.sh registry-start
  ( bash $D/spill_stores.sh stores-start $2 17179869184 > $D/logs/stores_start_$2.log 2>&1 ) &
  bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/$1 $2 > $R/$2.pd_start.log 2>&1
  curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "$2 LB not healthy"; return 1; }
  for _ in $(seq 1 40); do n=$(pgrep -u $(id -un) -c -f '^[^ ]*kv_cache_host_store_node_main'); (( n >= 16 )) && break; sleep 5; done
  bash $D/spill_stores.sh status
}
launch_base() { # recipe run
  bash $D/pd_spill.sh stop all; bash $D/spill_stores.sh stores-stop; sleep 15
  bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/$1 $2 > $R/$2.pd_start.log 2>&1
  curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "$2 LB not healthy"; return 1; }
}
log "launch dram_bo for B"; launch_dram gpt-oss_C_spill_dram_bo.json agenticB_dram_bo && {
  log "agentic B on dram_bo"; $PY $D/spill_agentic.py $AG --run agenticB_dram_bo --label recall --out $R/agenticB_dram_bo.jsonl 2>&1 | grep -E '^SUMMARY|error' | cut -c1-700
  curl -s -m 10 http://10.128.15.217:30000/metrics | grep -E '^sglang:(backuped_tokens_total|prefetched_tokens_total|storage_prefetch_hit_tokens_total|storage_prefetch_unfulfilled_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "%s %.0f\n", k, a[k]}' | sort
  bash $D/spill_stores.sh status; }
log "launch base_r05 for B"; launch_base gpt-oss_C_spill_base_r05.json agenticB_base_r05 && {
  log "agentic B on base_r05"; $PY $D/spill_agentic.py $AG --run agenticB_base_r05 --label recompute --out $R/agenticB_base_r05.jsonl 2>&1 | grep -E '^SUMMARY|error' | cut -c1-700; }
log "agentic B done"
