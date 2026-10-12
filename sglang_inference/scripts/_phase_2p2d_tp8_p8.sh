#!/usr/bin/env bash
# 2P/2D TP-8, phase 8: deliberate mixed-config pair. On the config-B group, replace d2 with (a) a decode whose page_size is 128
# while the prefills use 64 -- the bootstrap sanity check must refuse it loudly -- and (b) a decode whose KV pool layout differs
# (pallas_tpu_rpa_batched + SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1 vs the prefills' token-major layout): #1077 predicts the handshake
# does not compare layouts, so requests paired with d2 would return garbage with HTTP 200. Each variant: restart d2, send 12 greedy
# 8k/256 requests through the router, attribute each to d1/d2 by the decode's num_requests_total delta, compare the text with the
# golden; then restore d2 with the normal recipe and check 8/8.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p8; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
RUN=${RUN:-2p2d_tp8_B}; P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p8] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_d2(){ on $D2 "pkill -9 -u $ME -f 'sglang::scheduler'; pkill -9 -u $ME -f 'sglang.launch_server'; pkill -9 -u $ME -f 'benchmark.tpu.launch_server'; sleep 3; echo killed d2 on \$(hostname)"; }
start_d2(){ # $1 recipe basename $2 run label
  on $D2 "cd /tmp && exec >/dev/null 2>&1 </dev/null; PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$1 3 $2 &"
  local t0=$(date +%s); until curl -sf -m 5 http://$D2:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   d2 ($1) health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D2:30000/health) after $(( $(date +%s)-t0 )) s; server_info: $(curl -s -m 10 http://$D2:30000/get_server_info | $PY -c "import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ('page_size','attention_backend','kv_cache_dtype')})" 2>/dev/null)"; }
d2log(){ $SSH sa_101411256518727260534@$D2 "sudo cat /home/$ME/pd_disagg/results/runs/$1/d2/server.log" 2>/dev/null; }
reqs(){ curl -s -m 5 http://$1:30000/metrics | grep -E '^sglang:num_requests_total\{[^}]*tp_rank="0"' | sed -E 's/.* //'; }
# probe LABEL N: sequential greedy 8k/256 requests through the router; attribute by d1/d2 request-counter deltas; compare with the golden
probe(){ local label=$1 n=$2; : > $OUT/$label.txt
  for i in $(seq 1 $n); do local a1=$(reqs $D1) a2=$(reqs $D2) t0=$(date +%s.%N) body=$(mktemp) req=$(mktemp)
    $PY -c "import json; t=json.loads(open('$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']; json.dump({'text':t,'sampling_params':{'max_new_tokens':256,'temperature':0}}, open('$req','w'))"
    local code=$(curl -s -m 600 $LB/generate -H 'Content-Type: application/json' -d @$req -o $body -w '%{http_code}'); local dt=$(python3 -c "print(round($(date +%s.%N)-$t0,1))")
    sleep 1; local b1=$(reqs $D1) b2=$(reqs $D2); local who=unknown; [[ "$b2" != "$a2" ]] && who=d2; [[ "$b1" != "$a1" ]] && who=d1
    $PY - $body $code $dt $who >> $OUT/$label.txt <<'PY'
import json,sys
body,code,dt,who=sys.argv[1:5]; g=json.loads(open('/home/alisachen_google_com/pd_disagg/results/golden/gpt-oss_fp8_rep2_c1/8192.jsonl').readline())['text']
try: d=json.load(open(body)); t=d.get('text',''); n=d.get('meta_info',{}).get('completion_tokens')
except Exception: t=open(body,errors='replace').read()[:200]; n=None
# agreement over the first 256 characters: identical / numeric drift (long common prefix) / garbage (short or none)
pref=0
for a,b in zip(t,g):
    if a!=b: break
    pref+=1
kind='identical' if t==g[:len(t)] and len(t)>50 else ('NUM' if pref>=40 else 'GARBAGE')
print(f"{code} {dt}s {who} tokens={n} common_prefix_chars={pref} {kind} :: {t[:70]!r}")
PY
    rm -f $body $req; done
  sed 's/^/   /' $OUT/$label.txt; echo "   summary: $(awk '{print $1, $3, $6}' $OUT/$label.txt | sort | uniq -c | awk '{printf "%s x%s(%s,%s) ", $2, $1, $3, $4}')"; }
log "=== make sure the config-B group is live"; bash $S/pd4.sh status | head -n 5 | sed 's/^/   /'
log "=== baseline: 6 requests, both decodes healthy"; probe baseline 6
log "=== (a) d2 with page_size 128 (prefills 64): expect the bootstrap sanity check to refuse loudly"
kill_d2; start_d2 gpt-oss_B_128k_2p2d_d2page128.json ${RUN}_d2page128
probe d2_page128 12
log "   d2 log: page-size / handshake / exception lines:"; d2log ${RUN}_d2page128 | grep -iE 'page size mismatch|mismatch|RuntimeError|Scheduler hit|Traceback|bootstrap' | grep -vE 'bootstrap-req' | head -n 8 | cut -c1-220 | sed 's/^/   /'
log "   d2 health now: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D2:30000/health); gauges:"; for u in $P1 $P2 $D1 $D2; do echo "   $u: $(curl -s -m 5 http://$u:30000/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs)\{[^}]*tp_rank="0"' | sed -E 's/sglang://; s/\{[^}]*\}//' | tr '\n' ' ')"; done
log "=== (b) d2 with rpa_batched + SEQ_ALONG_LANE KV layout (prefills token-major): #1077 predicts no handshake error"
kill_d2; start_d2 gpt-oss_B_128k_2p2d_d2lane.json ${RUN}_d2lane
probe d2_lane 12
log "   d2 log: layout / exception lines:"; d2log ${RUN}_d2lane | grep -iE 'SEQ_ALONG_LANE|layout|mismatch|RuntimeError|Scheduler hit|Traceback|NotImplemented' | head -n 8 | cut -c1-220 | sed 's/^/   /'
log "   d2 health now: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D2:30000/health)"
log "=== restore d2 with the normal recipe"; kill_d2; start_d2 gpt-oss_B_128k_2p2d.json ${RUN}_d2restore; probe restored 8
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p8 done"
