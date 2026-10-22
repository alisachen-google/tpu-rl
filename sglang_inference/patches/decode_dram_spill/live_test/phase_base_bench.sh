#!/usr/bin/env bash
# Baseline pair (no storage backend): local-hit and recompute numbers for the 4 workload combos.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; mkdir -p $R
PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=${1:-spill_base_1}
log() { echo "$(date -u '+%F %T') $*"; }
for combo in "131072 8192" "131072 16384" "262144 8192" "262144 16384"; do
  set -- $combo; P=$1; N=$2
  log "base local  P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario local --label local --repeat 1 --run $RUN --out $R/$RUN.jsonl
  log "base flush  P=$P N=$N (recompute)"; $PY $D/spill_bench.py --prefix $P --new $N --scenario flush --label recompute --repeat 1 --run $RUN --out $R/$RUN.jsonl
done
log "base bench done"
