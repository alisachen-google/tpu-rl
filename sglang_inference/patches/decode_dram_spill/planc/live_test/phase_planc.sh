#!/usr/bin/env bash
# Plan C campaign. Step 1 on the running planc_s1 pair (decode HiCache on, no reverse pull):
#   ladder 4/12/16 to see decode write-through cost. Then relaunch both engines with the Plan C
#   tree + SGLANG_TPU_PEER_PREFIX=1 and run W1 (flush prefill only), agentic A, and the ladder.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
AG="--turns 4 --prefix 131072 --turn-new 4096 --turn-out 256 --think-min 1 --think-max 3 --stagger 2"
flush_both() { for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do curl -s -m 120 -X POST "$u/flush_cache" >/dev/null; done; }
dec_metrics() { curl -s -m 10 http://10.128.0.90:30000/metrics | grep -E '^sglang:(backuped_tokens_total|cached_tokens_total|prefetched_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "decode %s %.0f\n", k, a[k]}' | sort; }
pre_metrics() { curl -s -m 10 http://10.128.15.217:30000/metrics | grep -E '^sglang:(backuped_tokens_total|cached_tokens_total|prefetched_tokens_total)' | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END {for (k in a) printf "prefill %s %.0f\n", k, a[k]}' | sort; }
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"

if [[ ${SKIP_S1:-0} != 1 ]]; then
  curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "step-1 pair not healthy"; exit 1; }
  for n in 4 12 16; do
    flush_both; sleep 5
    log "step1 decode-hicache sessions=$n"; $PY $D/spill_agentic.py $AG --sessions $n --run planc_s1 --label "s1_n$n" --out $R/planc_s1_n$n.jsonl 2>&1 | grep -E '^SUMMARY|rror' | cut -c1-600
    dec_metrics
  done
  log "step 1 done"
fi

# ---- Plan C pair ----
bash $D/pd_spill.sh stop all; sleep 15
log "sync tree + recipe to a"; bash $D/sync_spill_to_a.sh
cat $HOME/pd_disagg/recipes/gpt-oss_C_planc.json | $SSH_A "sudo -n bash -c 'cat > /home/alisachen_google_com/pd_disagg/recipes/gpt-oss_C_planc.json && chown alisachen_google_com:alisachen_google_com /home/alisachen_google_com/pd_disagg/recipes/gpt-oss_C_planc.json'"
log "launch Plan C pair"; bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_planc.json planc_1 > $R/planc_1.pd_start.log 2>&1
curl -sf -m 5 http://10.128.15.217:8080/health >/dev/null || { log "Plan C LB not healthy"; exit 1; }
# W1: turn 2 after flushing the prefill only (the decode keeps its cache)
for combo in "131072 8192" "131072 16384" "262144 8192" "262144 16384"; do
  set -- $combo; P=$1; N=$2
  log "planc W1 local  P=$P N=$N"; $PY $D/spill_bench.py --prefix $P --new $N --scenario local --label local --repeat 1 --run planc_1 --out $R/planc_1.jsonl 2>&1 | grep -E 'ttft|rror' | cut -c1-300
  log "planc W1 recall P=$P N=$N (flush prefill only)"; $PY $D/spill_bench.py --prefix $P --new $N --scenario flush --flush-targets prefill --spill-wait 20 --label recall_planc --repeat 1 --run planc_1 --out $R/planc_1.jsonl 2>&1 | grep -E 'ttft|rror' | cut -c1-300
  bash $D/peer_stats.sh planc_1
done
# W2: agentic A
flush_both; sleep 5
log "planc agentic A"; $PY $D/spill_agentic.py --sessions 8 --turns 6 --prefix 131072 --turn-new 4096 --turn-out 512 --think-min 2 --think-max 8 --stagger 3 --run planc_1 --label agenticA_planc --out $R/agenticA_planc.jsonl 2>&1 | grep -E '^SUMMARY|rror' | cut -c1-600
bash $D/peer_stats.sh planc_1; pre_metrics; dec_metrics
# W6: ladder
for n in 4 8 12 16 20; do
  flush_both; sleep 5
  log "planc ladder sessions=$n"; $PY $D/spill_agentic.py $AG --sessions $n --run planc_1 --label "planc_n$n" --out $R/planc_1_n$n.jsonl 2>&1 | grep -E '^SUMMARY|rror' | cut -c1-600
  bash $D/peer_stats.sh planc_1; dec_metrics
done
log "PLANC ALL DONE"
