#!/usr/bin/env bash
# Instrumented 12-session point: stack dumps of both schedulers while it runs, then the peer logs.
set -uo pipefail
D=$HOME/pd_disagg/spill; R=$D/results; PY=$HOME/venv/sglang/bin/python; O=$R/probe_${RUN:-planc_5}; mkdir -p $O
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%F %T') $*"; }
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
RUN=${RUN:-planc_5}; N=${N:-12}
dump() { # tag
  $SSH_A 'sudo -n bash -c "P=\$(pgrep -u alisachen_google_com -f sglang::scheduler_TP0 | head -1); timeout 60 /home/alisachen_google_com/venv/sglang/bin/py-spy dump --pid \$P 2>&1 | head -30"' > $O/prefill_tp0_$1.txt 2>&1
  P=$(pgrep -u $(id -un) -f 'sglang::scheduler_TP0' | head -1); timeout 60 $HOME/venv/sglang/bin/py-spy dump --pid $P > $O/decode_tp0_$1.txt 2>&1
  for h in 10.128.15.217 10.128.0.90; do curl -s -m 5 http://$h:30000/metrics | grep -E "^sglang:(num_queue_reqs|num_running_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs)" | sed 's/{[^}]*}//' | sort -u | sed "s/^/$h /"; done > $O/queues_$1.txt
  log "dump $1 done"
}
for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do curl -s -m 120 -X POST "$u/flush_cache" >/dev/null; done; sleep 5
log "probe sessions=$N"
( sleep 90; dump t90; sleep 150; dump t240; sleep 240; dump t480 ) &
timeout 1500 $PY $D/spill_agentic.py --turns 4 --prefix 131072 --turn-new 4096 --turn-out 256 --think-min 1 --think-max 3 --stagger 2 --sessions $N --timeout 600 --run $RUN --label "probe_n$N" --out $R/${RUN}_probe_n$N.jsonl 2>&1 | grep -E '^SUMMARY|"error": "[^n]' | cut -c1-500
wait
grep -h "peer prefix" $HOME/pd_disagg/results/runs/$RUN/decode/server.log | grep -v installed | grep TP0 | cut -c1-200 > $O/decode_peer.log
$SSH_A "sudo -n grep -h 'peer prefix' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | grep -v installed | grep TP0 | cut -c1-200" > $O/prefill_peer.log 2>&1
log "probe done: $(wc -l < $O/prefill_peer.log) prefill peer lines, $(wc -l < $O/decode_peer.log) decode peer lines"
