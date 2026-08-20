#!/bin/bash
# Read a finished/running run's validity + perf gates from its GCS artifacts.
# Usage: tools/run_gates.sh <run-name>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?usage: run_gates.sh <run-name>}
BASE=gs://ubench-logs/alisachen-tpu7x/runs/$RUN
TMP=$(mktemp -d)
echo "== $RUN"
echo "-- LENDIST (truncation / lengths)"
gsutil ls "$BASE/lendist/" 2>/dev/null | tail -3 | while read -r f; do
  gsutil cat "$f" 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin); o=d["osl"]
print("   step %s n=%d cap=%d trunc=%.1f%% osl_mean=%.0f" % (
  d["step"], d["n"], d["cap"], 100.0*d["truncated_at_cap"]/max(1,len(o)), sum(o)/max(1,len(o))))'
done
echo "-- PHASETIME (per-step phases, s)"
gsutil ls "$BASE/phasetime/" 2>/dev/null | tail -5 | while read -r f; do
  gsutil cat "$f" 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("   step %s global=%.1f gen=%.1f rewards=%.1f train=%.1f" % (
  d.get("step"), d.get("run_global_step",0), d.get("engine_generate",0),
  d.get("compute_rewards",0), d.get("micro_batch_steps",0)))'
done
echo "-- TensorBoard scalars (rewards / perf / kl)"
F=$(gsutil ls "$BASE/tensorboard/" 2>/dev/null | grep tfevents | head -1)
if [ -n "$F" ]; then
  gsutil -q cp "$F" "$TMP/ev" && python3 tools/tb_scalars.py "$TMP/ev" \
    'rewards/train/(mean|match_format_exactly)|actor/train/(kl|pg_clipfrac)|perf/train/(global_step_time|weight_sync_time|mean/rollout_time|sum/actor_train_time)' \
    | sort -k1,1n -k2,2 | awk '{printf "   step%-4s %-40s %s\n",$1,$2,$3}'
else
  echo "   (no tensorboard events yet)"
fi
rm -rf "$TMP"
