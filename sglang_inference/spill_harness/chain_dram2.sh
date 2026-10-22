#!/usr/bin/env bash
D=$HOME/pd_disagg/spill; RUN=${1:-spill_dram_2}
log() { echo "[chain-dram] $(date -u '+%F %T') $*"; }
bash $D/spill_stores.sh registry-start
( bash $D/spill_stores.sh stores-start $RUN 17179869184 > $D/logs/stores_start_$RUN.log 2>&1 ) &
log "launching pair $RUN"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_dram.json $RUN > $D/results/$RUN.pd_start.log 2>&1
log "pair start returned $?: $(grep -E 'ready' $D/results/$RUN.pd_start.log | tr '\n' ';' | cut -c1-200)"
if ! curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null; then log "LB not healthy; aborting"; exit 1; fi
sleep 20; bash $D/spill_stores.sh status; tail -3 $D/logs/stores_start_$RUN.log
log "resume phase"; bash $D/phase_dram_resume.sh $RUN > $D/logs/phase_dram_resume_$RUN.log 2>&1; log "resume exit $?"
log "ALL DONE"
