#!/usr/bin/env bash
# After the agentic phase: rerun the store_pause and registry stress phases on the dram_r05 pair (fixed script).
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -q 'agentic phase done' $D/logs/phase_agentic.log 2>/dev/null || ! pgrep -f 'phase_agenti[c].sh' >/dev/null; do sleep 60; done
if curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null && (( $(pgrep -u $(id -un) -c -f '^[^ ]*kv_cache_host_store_node_main') >= 16 )); then
  echo "$(date -u '+%F %T') stress rerun (store_pause, registry) on agentic_dram_r05"
  $PY $D/spill_stress.py --phases store_pause,registry --prefix 32768 --new 8192 --spill-wait 60 --out $R/agentic_dram_r05.stress2.json 2>&1 | grep -E '^\{' | cut -c1-400
else
  echo "$(date -u '+%F %T') dram_r05 pair or stores not up; skipping stress rerun"
fi
echo "[chain-stress2] done"
