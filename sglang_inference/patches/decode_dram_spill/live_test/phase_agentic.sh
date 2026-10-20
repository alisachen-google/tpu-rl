#!/usr/bin/env bash
# Agentic workload: 8 sessions x 6 turns, 128k context each, 4k tool tokens per turn, 512 out.
# Runs on a half-size host tier (hicache_ratio 0.5) first without, then with the decode-host store.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--sessions 8 --turns 6 --prefix 131072 --turn-new 4096 --turn-out 512"
# --- baseline (no store) ---
log "stop current pair"; bash $D/pd_spill.sh stop all; bash $D/kill_stores_start.sh; bash $D/spill_stores.sh stores-stop
sleep 15
log "launch base_r05"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_base_r05.json agentic_base_r05 > $R/agentic_base_r05.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "base_r05 LB not healthy"; exit 1; }
log "agentic on base_r05"; $PY $D/spill_agentic.py $AG --run agentic_base_r05 --label recompute --out $R/agentic_base_r05.jsonl 2>&1 | grep -E '^SUMMARY|error' | cut -c1-600
# --- with the decode-host store ---
log "stop base_r05"; bash $D/pd_spill.sh stop all; sleep 15
ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 'sudo rm -f /home/alisachen_google_com/pd_disagg/spill/manifests/*.json'
( bash $D/spill_stores.sh stores-start agentic_dram_r05 17179869184 > $D/logs/stores_start_agentic_dram_r05.log 2>&1 ) &
log "launch dram_r05"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_dram_r05.json agentic_dram_r05 > $R/agentic_dram_r05.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "dram_r05 LB not healthy"; exit 1; }
for _ in $(seq 1 30); do n=$(pgrep -u $(id -un) -c -f kv_cache_host_store_node_main); (( n >= 16 )) && break; sleep 10; done
bash $D/spill_stores.sh status
log "agentic on dram_r05"; $PY $D/spill_agentic.py $AG --run agentic_dram_r05 --label recall --out $R/agentic_dram_r05.jsonl 2>&1 | grep -E '^SUMMARY|error' | cut -c1-600
bash $D/spill_stores.sh status
log "agentic phase done"
