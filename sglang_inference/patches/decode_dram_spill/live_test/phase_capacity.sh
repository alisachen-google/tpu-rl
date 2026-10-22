#!/usr/bin/env bash
# Capacity ladder: throughput vs working set. Sessions 4/8/12/16/20 x 128k context, 4 turns each.
# Baseline = HBM + full local host tier (cache mode, ratio 2). Store pair = buffer_only + decode-host stores (28 GiB each).
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
LADDER="4 8 12 16 20"
AG="--turns 4 --prefix 131072 --turn-new 4096 --turn-out 256 --think-min 1 --think-max 3 --stagger 2"
flush() { for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do curl -s -m 120 -X POST "$u/flush_cache" >/dev/null; done; }
metrics() { curl -s -m 10 http://10.128.15.217:30000/metrics | grep -E '^sglang:(backuped_tokens_total|prefetched_tokens_total|storage_prefetch_hit_tokens_total|storage_prefetch_unfulfilled_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "%s %.0f\n", k, a[k]}' | sort; }
# ---- baseline: HBM + local host tier ratio 2 ----
bash $D/pd_spill.sh stop all; bash $D/spill_stores.sh stores-stop; sleep 15
log "launch baseline (ratio 2, cache mode)"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_base.json cap_base > $R/cap_base.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "baseline LB not healthy"; exit 1; }
for n in $LADDER; do
  flush; sleep 5
  log "baseline sessions=$n"; $PY $D/spill_agentic.py $AG --sessions $n --run cap_base --label "base_n$n" --out $R/cap_base_n$n.jsonl 2>&1 | grep -E '^SUMMARY|rror' | cut -c1-600
done
metrics
# ---- store pair: buffer_only + decode-host stores ----
bash $D/pd_spill.sh stop all; sleep 15
ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 'sudo rm -f /home/alisachen_google_com/pd_disagg/spill/manifests/*.json'
bash $D/spill_stores.sh registry-start
( bash $D/spill_stores.sh stores-start cap_dram 30064771072 > $D/logs/stores_start_cap_dram.log 2>&1 ) &
log "launch store pair (buffer_only)"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_dram_bo.json cap_dram > $R/cap_dram.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "store LB not healthy"; exit 1; }
for _ in $(seq 1 60); do c=$(pgrep -u $(id -un) -c -f '^[^ ]*kv_cache_host_store_node_main'); (( c >= 16 )) && break; sleep 5; done
bash $D/spill_stores.sh status
for n in $LADDER; do
  if [[ $n != 4 ]]; then bash $D/spill_stores.sh stores-stop; sleep 3; bash $D/spill_stores.sh stores-start cap_dram 30064771072; for _ in $(seq 1 60); do c=$(pgrep -u $(id -un) -c -f '^[^ ]*kv_cache_host_store_node_main'); (( c >= 16 )) && break; sleep 5; done; fi
  flush; sleep 5
  log "store sessions=$n"; $PY $D/spill_agentic.py $AG --sessions $n --run cap_dram --label "store_n$n" --out $R/cap_dram_n$n.jsonl 2>&1 | grep -E '^SUMMARY|rror' | cut -c1-600
  metrics; bash $D/spill_stores.sh status | head -2
done
log "capacity ladder done"
