#!/usr/bin/env bash
# 2P/2D TP-8, issue #1276 repro: decode /health_generate stays 200 while every KV transfer fails.
# d2 is relaunched with the test hook (hooks/1276/sitecustomize.py via PYTHONPATH; RaidenKVManager.is_recv_done always False)
# and SGLANG_DISAGGREGATION_WAITING_TIMEOUT=60, so every request paired with d2 fails at the waiting timeout. While 8 greedy
# 8k/256 requests go through the router, d2's /health_generate (the generation probe; /health is unconditional 200 here because
# SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0) is polled every 5 s. Then d2 is restored with the standard recipe.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/i1276; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
RUN=${RUN:-2p2d_tp8_i1276b}; P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [i1276] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_d2(){ on $D2 "pkill -9 -u $ME -f 'sglang::scheduler'; pkill -9 -u $ME -f 'sglang.launch_server'; pkill -9 -u $ME -f 'benchmark.tpu.launch_server'; sleep 3; echo killed d2 on \$(hostname)"
  # wait until the old server stops answering, or the start-up wait below sees the dying one as healthy
  local t0=$(date +%s); while curl -sf -m 3 http://$D2:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d2 down after $(( $(date +%s)-t0 )) s (health $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://$D2:30000/health))"; }
start_d2(){ # $1 recipe basename $2 run label
  on $D2 "cd /tmp && exec >/dev/null 2>&1 </dev/null; PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$1 3 $2 &"
  local t0=$(date +%s); until curl -sf -m 5 http://$D2:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   d2 ($1) health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D2:30000/health) after $(( $(date +%s)-t0 )) s"; }
d2log(){ $SSH sa_101411256518727260534@$D2 "sudo cat /home/$ME/pd_disagg/results/runs/$1/d2/server.log" 2>/dev/null; }
reqs(){ curl -s -m 5 http://$1:30000/metrics | grep -E '^sglang:num_requests_total\{' | head -n 1 | sed -E 's/.* //'; }
tfail(){ curl -s -m 5 http://$1:30000/metrics | grep -E '^sglang:num_transfer_failed_reqs_total\{' | head -n 1 | sed -E 's/.* //'; }
hg(){ curl -s -m 30 -o /dev/null -w '%{http_code} %{time_total}' http://$1:30000/health_generate; }
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
pref=0
for a,b in zip(t,g):
    if a!=b: break
    pref+=1
kind='identical' if t==g[:len(t)] and len(t)>50 else ('NUM' if pref>=40 else 'ERR/GARBAGE')
print(f"{code} {dt}s {who} tokens={n} common_prefix_chars={pref} {kind} :: {t[:70]!r}")
PY
    rm -f $body $req; done
  sed 's/^/   /' $OUT/$label.txt; echo "   summary: $(awk '{print $1, $3, $6}' $OUT/$label.txt | sort | uniq -c | awk '{printf "%s x%s(%s,%s) ", $2, $1, $3, $4}')"; }

log "=== issue #1276 repro on the config-B group; d2 -> hooked decode (is_recv_done always False, waiting timeout 60 s)"
log "   before: d2 /health_generate $(hg $D2); d1 $(hg $D1); transfer_failed d2=$(tfail $D2) d1=$(tfail $D1)"
kill_d2; start_d2 gpt-oss_B_128k_2p2d_d2hook1276.json $RUN
log "   hook lines in d2 log:"; d2log $RUN | grep -a 'HOOK-1276' | head -n 4 | cut -c1-200 | sed 's/^/   /'
log "   d2 env check: $(d2log $RUN | grep -a -o 'SGLANG_DISAGGREGATION_WAITING_TIMEOUT[^ ,]*' | head -1)"
# health_generate poller on d2 every 5 s
: > $OUT/health_d2.csv; ( while [[ ! -f $OUT/.stop ]]; do echo "$(date -u +%FT%TZ),$(hg $D2 | tr ' ' ',')" >> $OUT/health_d2.csv; sleep 5; done ) & POLL=$!
log "=== 8 greedy 8k/256 requests through the router (those paired with d2 should fail at the 60 s waiting timeout)"
probe hooked 8
log "   transfer_failed d2=$(tfail $D2) d1=$(tfail $D1); d2 requests total $(reqs $D2); d2 /health_generate now $(hg $D2); /health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D2:30000/health)"
log "   d2 log: hook / failure lines:"; d2log $RUN | grep -a -E 'HOOK-1276: is_recv_done|Decode transfer failed|waiting timeout|WAITING_TIMEOUT' | head -n 6 | cut -c1-230 | sed 's/^/   /'
touch $OUT/.stop; wait $POLL 2>/dev/null; rm -f $OUT/.stop
log "   d2 /health_generate during the probes: $(awk -F, '{print $2}' $OUT/health_d2.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') (n=$(wc -l < $OUT/health_d2.csv), max latency $(awk -F, '{if($3>m)m=$3} END {print m}' $OUT/health_d2.csv) s)"
log "=== restore d2 with the standard recipe"; kill_d2; start_d2 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3
probe restored 6
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase i1276 done"
