#!/usr/bin/env bash
# MT-Bench (80 questions x 2 turns) through four gpt-oss pairs: A (radix off), B (radix on) for the prefix-cache benefit;
# C_bf16 (no speculation) vs D (EAGLE3) for the speculative-decoding benefit. c=1 for clean latency, c=4 for throughput.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; Q=$HOME/pd_disagg/prompts/mt_bench/question.jsonl
SNAP=$(ls -d /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/* | head -1); OUT=$R/mtbench; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [mtbench] $*"; }
until grep -q 'phase 1m BC done' $R/phase_1m_BC.log 2>/dev/null; do sleep 120; done; log "1m phase done"
for CFG in A B C_bf16 D; do
  case $CFG in C_bf16) REC=$HOME/pd_disagg/recipes/gpt-oss_C_128k_bf16.json;; *) REC=$HOME/pd_disagg/recipes/gpt-oss_${CFG}_128k.json;; esac
  RUN=mtbench_${CFG}; mkdir -p $OUT/$CFG
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  log "=== $CFG: start $REC"
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $REC $RUN 2>&1 | tee $R/runs/$RUN.driver.log | grep -E 'ready|NOT'
  if ! curl -sf -m 30 $LB/health_generate >/dev/null; then log "$CFG: pair did not come up"; bash $S/pd.sh stop >/dev/null 2>&1; continue; fi
  for c in 1 4; do
    $PY $S/mtbench_runner.py --url $LB --questions $Q --snapshot $SNAP --n 80 --concurrency $c --max-new-tokens 512 --out $OUT/$CFG/mtbench_c$c.jsonl --label "$CFG" 2>&1 | grep -E '^\[' | tee -a $OUT/$CFG/summary.txt
    echo "   prefill cache_hit_rate after c=$c: $(curl -s -m 10 $PRE/metrics | grep -E '^sglang:cache_hit_rate' | sed -E 's/\{[^}]*\}//' | head -1)"
  done
  bash $S/pd.sh stop >/dev/null 2>&1; log "=== $CFG done"
done
log "phase mtbench done"
