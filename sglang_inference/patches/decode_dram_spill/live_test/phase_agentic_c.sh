#!/usr/bin/env bash
# Agentic variant A again, store pair in buffer_only mode (comparable to agentic_base_r05).
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--sessions 8 --turns 6 --prefix 131072 --turn-new 4096 --turn-out 512"
bash $D/pd_spill.sh stop all; bash $D/spill_stores.sh stores-stop; sleep 15
ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 'sudo rm -f /home/alisachen_google_com/pd_disagg/spill/manifests/*.json'
bash $D/spill_stores.sh registry-start
( bash $D/spill_stores.sh stores-start agenticA_dram_bo 17179869184 > $D/logs/stores_start_agenticA_dram_bo.log 2>&1 ) &
log "launch dram_bo for A"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_dram_bo.json agenticA_dram_bo > $R/agenticA_dram_bo.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "LB not healthy"; exit 1; }
for _ in $(seq 1 40); do n=$(pgrep -u $(id -un) -c -f '^[^ ]*kv_cache_host_store_node_main'); (( n >= 16 )) && break; sleep 5; done
bash $D/spill_stores.sh status
log "agentic A on dram_bo"; $PY $D/spill_agentic.py $AG --run agenticA_dram_bo --label recall --out $R/agenticA_dram_bo.jsonl 2>&1 | grep -E '^SUMMARY|error' | cut -c1-700
curl -s -m 10 http://10.128.15.217:30000/metrics | grep -E '^sglang:(backuped_tokens_total|prefetched_tokens_total|storage_prefetch_hit_tokens_total|storage_prefetch_unfulfilled_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "%s %.0f\n", k, a[k]}' | sort
bash $D/spill_stores.sh status
log "agentic A bo done"
