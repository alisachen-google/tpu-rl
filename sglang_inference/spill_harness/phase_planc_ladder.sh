#!/usr/bin/env bash
# Plan C ladder points on a running pair (prefill run $RUN on a). Points from $LADDER (default 12 16 20).
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--turns 4 --prefix 131072 --turn-new 4096 --turn-out 256 --think-min 1 --think-max 3 --stagger 2"
flush_both() { for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do curl -s -m 120 -X POST "$u/flush_cache" >/dev/null; done; }
dec_metrics() { curl -s -m 10 http://10.128.0.90:30000/metrics | grep -E '^sglang:(backuped_tokens_total|cached_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "decode %s %.0f\n", k, a[k]}' | sort; }
RUN=${RUN:-planc_4}; LADDER=${LADDER:-"12 16 20"}
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "LB not healthy"; exit 1; }
for n in $LADDER; do
  flush_both; sleep 5
  log "ladder sessions=$n"; timeout 1500 $PY $D/spill_agentic.py $AG --sessions $n --timeout 600 --run $RUN --label "planc_n$n" --out $R/${RUN}_n$n.jsonl 2>&1 | grep -E '^SUMMARY|"error": "[^n]' | cut -c1-600
  bash $D/peer_stats.sh $RUN; dec_metrics
done
log "PLANC LADDER DONE"
