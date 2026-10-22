#!/usr/bin/env bash
# Full Plan C validation on a running pair: W1 (prefill-only flush), ladder 12/16/20, stress phases.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
RUN=${RUN:-planc_8}; LADDER=${LADDER:-"12 16 20"}
AG="--turns 4 --prefix 131072 --turn-new 4096 --turn-out 256 --think-min 1 --think-max 3 --stagger 2"
flush_both() { for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do curl -s -m 120 -X POST "$u/flush_cache" >/dev/null; done; }
dec_metrics() { curl -s -m 10 http://10.128.0.90:30000/metrics | grep -E '^sglang:(backuped_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "decode %s %.0f\n", k, a[k]}' | sort; }
miss_reasons() { grep -h "peer prefix miss" $HOME/pd_disagg/results/runs/$RUN/decode/server.log | grep TP0 | sed 's/.*reason=//' | sort | uniq -c | sort -rn | head -4 | sed 's/^/  decode miss: /'; }
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "LB not healthy"; exit 1; }
for combo in "131072 8192" "262144 8192"; do
  set -- $combo; P=$1; N=$2
  log "W1 recall P=$P N=$N (flush prefill only)"; $PY $D/spill_bench.py --prefix $P --new $N --scenario flush --flush-targets prefill --spill-wait 20 --label recall_planc --repeat 1 --run $RUN --out $R/$RUN.jsonl 2>&1 | grep -o '"turn2": {"ttft_s": [0-9.]*\|"cached_tokens_details": {[^}]*}' | head -2 | tr '\n' ' '; echo
  bash $D/peer_stats.sh $RUN
done
for n in $LADDER; do
  flush_both; sleep 5
  log "ladder sessions=$n"; timeout 1500 $PY $D/spill_agentic.py $AG --sessions $n --timeout 600 --run $RUN --label "planc_n$n" --out $R/${RUN}_n$n.jsonl 2>&1 | grep -E '^SUMMARY|"error": "[^n]' | cut -c1-600
  bash $D/peer_stats.sh $RUN; dec_metrics; miss_reasons
done
flush_both; sleep 5
log "stress"; $PY $D/spill_stress.py --phases concurrent,abort,decode_flush --flush-targets prefill --prefix 32768 --new 8192 --sessions 4 --spill-wait 5 --out $R/$RUN.stress.json 2>&1 | grep -E '"phase"|health_samples'
bash $D/peer_stats.sh $RUN
log "PLANC FULL DONE"
