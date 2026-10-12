#!/usr/bin/env bash
# 2P/2D TP-8, phase 4: partial failure. Kill p2 mid-prefill and d2 mid-decode; the other pair must keep serving; restart the dead engine and rejoin; abort storm across pairs.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p4; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
RUN=${RUN:-2p2d_tp8_bringup}; P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p4] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_engine(){ on $1 "pkill -9 -u $ME -f 'sglang::scheduler'; pkill -9 -u $ME -f 'sglang.launch_server'; pkill -9 -u $ME -f 'benchmark.tpu.launch_server'; sleep 2; echo killed on \$(hostname)"; }
burst(){ # $1 label $2 n $3 prompt-file -> HTTP code histogram and p50 latency
  for i in $(seq 1 $2); do ( t0=$(date +%s.%N); c=$(curl -s -m 600 $LB/generate -H 'Content-Type: application/json' -d "$(python3 -c "import json,sys; t=json.loads(open('$3').readline())['text']; print(json.dumps({'text':t,'sampling_params':{'max_new_tokens':32,'temperature':0}}))")" -o /dev/null -w '%{http_code}'); echo "$c $(python3 -c "print(round($(date +%s.%N)-$t0,1))")" ) & done | tee $OUT/$1.txt | awk '{h[$1]++} END{for(k in h) printf "   HTTP %s x%d\n", k, h[k]}'; }
PROMPT8K=$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl; PROMPT1K=$HOME/pd_disagg/prompts/gpt-oss/1024.jsonl
log "=== baseline burst of 8 x 1k through the router"; burst baseline 8 $PROMPT1K
log "=== kill p2 while 6 x 8k prefills are in flight"; ( burst during_p2_kill 6 $PROMPT8K ) & sleep 4; kill_engine $P2; wait
log "   p2 health now: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$P2:30000/health); router health: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $LB/health)"
log "=== with p2 dead: 8 x 1k must still be served by p1 (router may or may not skip the dead P: #1274 class)"; burst p2_dead 8 $PROMPT1K
log "=== restart p2 and wait for health"; on $P2 "cd /tmp && exec >/dev/null 2>&1 </dev/null; PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/gpt-oss_A_128k_2p2d.json 1 ${RUN}_p2restart &"
t0=$(date +%s); until curl -sf -m 5 http://$P2:30000/health >/dev/null || (( $(date +%s)-t0 > 1800 )); do sleep 15; done; log "   p2 back after $(( $(date +%s)-t0 )) s (health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$P2:30000/health))"
log "=== after p2 rejoin: 8 x 1k"; burst p2_back 8 $PROMPT1K
log "=== kill d2 while 6 x 8k requests are decoding"; ( burst during_d2_kill 6 $PROMPT8K ) & sleep 25; kill_engine $D2; wait
log "   d2 health now: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D2:30000/health)"
log "=== with d2 dead: 8 x 1k (requests routed to d2 fail; d1 ones must succeed)"; burst d2_dead 8 $PROMPT1K
log "=== restart d2"; on $D2 "cd /tmp && exec >/dev/null 2>&1 </dev/null; PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/gpt-oss_A_128k_2p2d.json 3 ${RUN}_d2restart &"
t0=$(date +%s); until curl -sf -m 5 http://$D2:30000/health >/dev/null || (( $(date +%s)-t0 > 1800 )); do sleep 15; done; log "   d2 back after $(( $(date +%s)-t0 )) s"
log "=== after d2 rejoin: 8 x 1k"; burst d2_back 8 $PROMPT1K
log "=== abort storm: 40 aborts in 10 s across both pairs, then follow-ups"; $PY $S/abort_test.py --url $LB --prompt-file $PROMPT8K --phase decode --abort-after 0.25 --repeat 40 --concurrent 4 --server-info http://$D1:30000 --server-info http://$D2:30000 --settle 90 2>&1 | tail -n 8 | sed 's/^/   /'
log "=== prefill bootstrap queues and decode inflight after storm"; for u in http://$P1:30000 http://$P2:30000 http://$D1:30000 http://$D2:30000; do echo "   $u: $(curl -s -m 5 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_prefill_prealloc_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs)\{[^}]*tp_rank="0"' | sed -E 's/\{[^}]*\}//' | tr '\n' ' ')"; done
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p4 done"
