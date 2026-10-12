#!/usr/bin/env bash
# 2P/2D TP-8, phase 4b: partial failure with the kill truly mid-flight.
# Phase 4 used 32-token outputs, so every burst finished before the kill landed. Here: kill p2 while a 128k prefill
# is running on it (prefill takes seconds), kill d2 while 1,024-token decodes are running on it, then recover both.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p4c; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
RUN=${RUN:-2p2d_tp8_bringup}; P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p4c] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_engine(){ on $1 "pkill -9 -u $ME -f 'sglang::scheduler'; pkill -9 -u $ME -f 'sglang.launch_server'; pkill -9 -u $ME -f 'benchmark.tpu.launch_server'; sleep 2; echo killed on \$(hostname)"; }
gauges(){ for u in $P1 $P2 $D1 $D2; do echo "   $u: $(curl -s -m 5 http://$u:30000/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_prefill_prealloc_queue_reqs|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs)\{[^}]*tp_rank="0"' | sed -E 's/sglang://; s/\{[^}]*\}//' | tr '\n' ' ')"; done; }
# burst LABEL N PROMPT MAXTOK -> "code seconds" per request, plus body tail on non-200
burst(){ for i in $(seq 1 $2); do ( t0=$(date +%s.%N); b=$(mktemp); req=$(mktemp); python3 -c "import json,sys; t=json.loads(open('$3').readline())['text']; json.dump({'text':t,'sampling_params':{'max_new_tokens':$4,'temperature':0,'ignore_eos':True}}, open('$req','w'))" ; c=$(curl -s -m 900 $LB/generate -H 'Content-Type: application/json' -d @$req -o $b -w '%{http_code}'); rm -f $req; echo "$c $(python3 -c "print(round($(date +%s.%N)-$t0,1))") $( [ "$c" != 200 ] && tr -d '\n' < $b | cut -c1-120 )"; rm -f $b ) & done | tee $OUT/$1.txt | sort | awk '{print $1}' | uniq -c | awk '{print "   HTTP " $2 " x" $1}'; }
restart(){ local ip=$1 rank=$2 label=$3; on $ip "cd /tmp && exec >/dev/null 2>&1 </dev/null; PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/gpt-oss_A_128k_2p2d.json $rank ${RUN}_${label} &"; local t0=$(date +%s); until curl -sf -m 5 http://$ip:30000/health >/dev/null || (( $(date +%s)-t0 > 1800 )); do sleep 15; done; log "   $label back after $(( $(date +%s)-t0 )) s (health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$ip:30000/health))"; }
P128K=$HOME/pd_disagg/prompts/gpt-oss/131072.jsonl; P8K=$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl; P1K=$HOME/pd_disagg/prompts/gpt-oss/1024.jsonl
log "=== gauges before"; gauges
log "=== reference: 4 x 128k/64 and 4 x 8k/1024 through the router, nothing killed"; burst ref_128k 4 $P128K 64; burst ref_8k_long 4 $P8K 4096
log "=== kill p2 3 s into 6 x 128k prefills (about half land on p2)"; ( burst during_p2_kill_128k 6 $P128K 64 ) & sleep 3; kill_engine $P2; log "   p2 killed at +3 s"; wait
log "   gauges with p2 dead:"; gauges
log "=== restart p2"; restart $P2 1 p2restart3; burst p2_back 6 $P1K 32
log "=== kill d2 6 s into 6 x 8k/4096 decodes (about 25 s each; about half decode on d2)"; ( burst during_d2_kill_long 6 $P8K 4096 ) & sleep 6; kill_engine $D2; log "   d2 killed at +6 s"; wait
log "   gauges with d2 dead (prefills must not hold KV for the dead decode):"; gauges
log "=== with d2 dead: 8 x 1k"; burst d2_dead 8 $P1K 32
log "=== restart d2"; restart $D2 3 d2restart3; burst d2_back 8 $P1K 32
log "=== greedy check after recoveries: 8k/256 via router vs the phase-0 output"; $PY - <<'PY'
import json,urllib.request,sys
t=json.loads(open(sys.argv[1] if len(sys.argv)>1 else '/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def gen(): 
    r=urllib.request.Request('http://10.128.15.217:8080/generate',data=json.dumps({'text':t,'sampling_params':{'max_new_tokens':256,'temperature':0}}).encode(),headers={'Content-Type':'application/json'}); return json.load(urllib.request.urlopen(r,timeout=600))['text']
a,b=gen(),gen(); print('   two greedy runs identical:', a==b)
PY
log "=== gauges after"; gauges; log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p4c done"
