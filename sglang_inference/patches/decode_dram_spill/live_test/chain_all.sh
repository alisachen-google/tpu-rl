#!/usr/bin/env bash
D=$HOME/pd_disagg/spill
log() { echo "[chain-all] $(date -u '+%F %T') $*"; }
until grep -q 'base pair start returned' $D/logs/chain_base.log 2>/dev/null; do sleep 30; done
if ! curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null; then log "LB not healthy after base start; see chain_base.log"; fi
log "baseline bench start"; bash $D/phase_base_bench.sh spill_base_1 > $D/logs/phase_base_bench.log 2>&1; log "baseline bench exit $?"
log "dram phase start"; bash $D/phase_dram.sh spill_dram_1 > $D/logs/phase_dram.log 2>&1; log "dram phase exit $?"
log "ALL DONE"
