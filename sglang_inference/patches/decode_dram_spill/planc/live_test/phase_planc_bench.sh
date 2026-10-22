#!/usr/bin/env bash
# Plan C benchmarks on an already running pair (prefill run planc_1 on a; decode on b).
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--turns 4 --prefix 131072 --turn-new 4096 --turn-out 256 --think-min 1 --think-max 3 --stagger 2"
flush_both() { for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do curl -s -m 120 -X POST "$u/flush_cache" >/dev/null; done; }
dec_metrics() { curl -s -m 10 http://10.128.0.90:30000/metrics | grep -E '^sglang:(backuped_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "decode %s %.0f\n", k, a[k]}' | sort; }
RUN=${RUN:-planc_1}
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "LB not healthy"; exit 1; }
for combo in "131072 8192" "131072 16384" "262144 8192" "262144 16384"; do
  set -- $combo; P=$1; N=$2
  log "W1 local  P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario local --label local --repeat 1 --run $RUN --out $R/$RUN.jsonl 2>&1 | grep -o '"turn2": {"ttft_s": [0-9.]*' | head -1
  log "W1 recall P=$P N=$N (flush prefill only)"; $PY $D/spill_bench.py --prefix $P --new $N --scenario flush --flush-targets prefill --spill-wait 20 --label recall_planc --repeat 1 --run $RUN --out $R/$RUN.jsonl 2>&1 | grep -o '"turn2": {"ttft_s": [0-9.]*\|"cached_tokens_details": {[^}]*}' | head -2 | tr '\n' ' '; echo
  bash $D/peer_stats.sh $RUN
done
flush_both; sleep 5
log "agentic A"; $PY $D/spill_agentic.py --sessions 8 --turns 6 --prefix 131072 --turn-new 4096 --turn-out 512 --think-min 2 --think-max 8 --stagger 3 --run $RUN --label agenticA_planc --out $R/agenticA_planc.jsonl 2>&1 | grep -E '^SUMMARY|rror"' | grep -v '"error": null' | cut -c1-600
bash $D/peer_stats.sh $RUN; dec_metrics
for n in 4 8 12 16 20; do
  flush_both; sleep 5
  log "ladder sessions=$n"; $PY $D/spill_agentic.py $AG --sessions $n --run $RUN --label "planc_n$n" --out $R/${RUN}_n$n.jsonl 2>&1 | grep -E '^SUMMARY' | cut -c1-600
  bash $D/peer_stats.sh $RUN; dec_metrics
done
log "PLANC BENCH DONE"
