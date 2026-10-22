#!/usr/bin/env bash
# Agentic variant B (fixed prompt file): on the buffer_only store pair left up by phase C, then on the baseline.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--sessions 12 --turns 10 --shared 16384 --prefix 114688 --turn-new 2048 --turn-out 256 --think-min 1 --think-max 4 --stagger 2"
if curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null; then
  bash $D/spill_stores.sh stores-stop; sleep 3
  bash $D/spill_stores.sh stores-start agenticA_dram_bo 17179869184 || log "stores restart failed"
  for _ in $(seq 1 30); do n=$(pgrep -u $(id -un) -c -f '^[^ ]*kv_cache_host_store_node_main'); (( n >= 16 )) && break; sleep 5; done
  bash $D/spill_stores.sh status
  log "agentic B on dram_bo"; $PY $D/spill_agentic.py $AG --run agenticB_dram_bo --label recall --out $R/agenticB_dram_bo.jsonl 2>&1 | grep -E '^SUMMARY|Error|error' | cut -c1-700
  curl -s -m 10 http://10.128.15.217:30000/metrics | grep -E '^sglang:(backuped_tokens_total|prefetched_tokens_total|storage_prefetch_hit_tokens_total|storage_prefetch_unfulfilled_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "%s %.0f\n", k, a[k]}' | sort
  bash $D/spill_stores.sh status
else
  log "dram_bo pair not up; skipping B on dram"
fi
log "launch base_r05 for B"; bash $D/pd_spill.sh stop all; bash $D/spill_stores.sh stores-stop; sleep 15
bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_base_r05.json agenticB_base_r05 > $R/agenticB_base_r05.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "base_r05 LB not healthy"; exit 1; }
log "agentic B on base_r05"; $PY $D/spill_agentic.py $AG --run agenticB_base_r05 --label recompute --out $R/agenticB_base_r05.jsonl 2>&1 | grep -E '^SUMMARY|Error|error' | cut -c1-700
log "agentic B done"
