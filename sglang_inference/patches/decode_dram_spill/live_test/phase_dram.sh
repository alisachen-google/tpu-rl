#!/usr/bin/env bash
# DRAM-spill pair: registry + stores on b, prefill with RaidenDramStorage; recall + local numbers, then stress phases.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; mkdir -p $R
PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=${1:-spill_dram_1}; BUDGET=${BUDGET:-17179869184}
log() { echo "$(date -u '+%F %T') $*"; }
log "stopping baseline pair"; bash $D/pd_spill.sh stop all
sleep 20
log "registry start"; bash $D/spill_stores.sh registry-start || exit 1
# launch the pair in the background so the stores can start as soon as the manifests exist
( bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_dram.json $RUN > $R/$RUN.pd_start.log 2>&1 ) &
PD=$!
log "stores start (waits for manifests)"; bash $D/spill_stores.sh stores-start $RUN $BUDGET || { log "stores failed"; }
wait $PD; log "pair start returned $?"; grep -E 'ready|NOT ready' $R/$RUN.pd_start.log | tail -4
bash $D/spill_stores.sh status
sleep 30; log "stores log head:"; tail -5 $D/logs/stores_$RUN.log
for combo in "131072 8192" "131072 16384" "262144 8192" "262144 16384"; do
  set -- $combo; P=$1; N=$2
  log "dram recall P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario flush --label recall --spill-wait 120 --repeat 1 --run $RUN --out $R/$RUN.jsonl
  log "dram local  P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario local --label local --repeat 1 --run $RUN --out $R/$RUN.jsonl
done
bash $D/spill_stores.sh status
log "stress phases"; $PY $D/spill_stress.py --prefix 32768 --new 8192 --sessions 4 --spill-wait 60 --out $R/$RUN.stress.json 2>&1 | tail -20
bash $D/spill_stores.sh status
log "dram phase done"
