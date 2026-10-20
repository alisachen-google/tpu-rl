#!/usr/bin/env bash
# Resume: pair + stores already up. Recall + local benchmark, then stress phases.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; mkdir -p $R
PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=${1:-spill_dram_1}
log() { echo "$(date -u '+%F %T') $*"; }
bash $D/spill_stores.sh status
for combo in "131072 8192" "131072 16384" "262144 8192" "262144 16384"; do
  set -- $combo; P=$1; N=$2
  log "dram recall P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario flush --label recall --spill-wait 120 --repeat 1 --run $RUN --out $R/$RUN.jsonl
  log "dram local  P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario local --label local --repeat 1 --run $RUN --out $R/$RUN.jsonl
done
bash $D/spill_stores.sh status
log "stress phases"; $PY $D/spill_stress.py --prefix 32768 --new 8192 --sessions 4 --spill-wait 60 --out $R/$RUN.stress.json 2>&1 | tail -20
bash $D/spill_stores.sh status
log "dram phase done"
