#!/usr/bin/env bash
# Live validation of PR #1340 (decode readiness from real transfer outcomes) on the 2P/2D TP-8 group. d1 (this host)
# runs the branch with N=3, T=60 s, P=60 s and 60 s bootstrap/waiting timeouts so results arrive in minutes.
#  A. baseline: d1 on the branch, traffic OK, /v1/loads health state ready, /health_generate 200.
#  B. one prefill dead (p2 killed): requests paired with p2 fail on d1 after the 60 s bootstrap timeout, requests paired
#     with p1 succeed -> d1 stays 200, /v1/loads reports degraded with p2 named. p2 restored.
#  C. the #1276 hook on d1 (receive completion never signalled): every transfer into d1 fails -> 503 with a reason body.
#  D. probation: no traffic for T -> 200 again with state probation; one more failing request -> unhealthy, wait doubled.
#  E. restore d1 to main with the standard recipe.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr13c; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-peers
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [pr13c] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_engine(){ local ip=$1 label=$2; on $ip "pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; if [[ $ip == $P1 ]]; then pkill -9 -u $ME -f 'sglang::[r]outer'; pkill -9 -u $ME -f 'sglang_router.launch_[r]outer'; fi; sleep 3; true"
  local t0=$(date +%s); while curl -sf -m 3 http://$ip:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   $label down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local ip=$1 rank=$2 recipe=$3 run=$4 root=${5:-/home/$ME/sglang-torchtpu}
  on $ip "cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$recipe $rank $run &"; }
wait_up(){ local ip=$1 label=$2 t0=$(date +%s); until curl -sf -m 5 http://$ip:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   $label health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$ip:30000/health) after $(( $(date +%s)-t0 )) s"; }
hg(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$1:30000/health_generate | tr -d '\n' | cut -c1-300; }
health_json(){ curl -s -m 5 "http://$1:30000/v1/loads?include=disagg" | $PY -c "import sys,json; d=json.load(sys.stdin); h=d['loads'][0]['disaggregation'].get('health_json') or ''; print(h if h else '(no health_json)')"; }
traffic(){ # N requests with concurrency C through the router; prints per-request 'code seconds decode'
  local n=$1 c=$2 file=$3; $PY - $n $c $LB $D1 $D2 > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,lb,d1,d2=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def reqs(ip):
    try:
        m=urllib.request.urlopen(f'http://{ip}:30000/metrics',timeout=5).read().decode()
        return float([l for l in m.splitlines() if l.startswith('sglang:num_requests_total{')][0].split()[-1])
    except Exception: return float('nan')
def one(i):
    body=json.dumps({'text':text,'sampling_params':{'max_new_tokens':128,'temperature':0}}).encode()
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(lb+'/generate',data=body,headers={'Content-Type':'application/json'}),timeout=75); code=r.status; r.read()
    except urllib.error.HTTPError as e: code=e.code
    except Exception as e: code=f'ERR:{type(e).__name__}'
    return f'{code} {time.time()-t0:.1f}s'
with cf.ThreadPoolExecutor(c) as ex:
    for line in ex.map(one, range(n)): print(line, flush=True)
PY
  awk '{print $1}' $file | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }

hgp(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$1:30000/health_generate | tr -d '\n' | cut -c1-600; }
log "=== per-peer rule (PR #1393) live validation, client timeout 75 s so traffic keeps flowing; branch $(git -C $WT rev-parse --short HEAD); T=60 s, cold 120 s; peer streak 2, ratio 50%"
log "=== X-A. d1 on the peers branch, p1 with the sender fault, p2 healthy: d1 must report p1 failing (1 of 2) while its engine rule stays ready"
kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d_d1peers.json 2p2d_tp8_pr13d $WT; wait_up $D1 d1
log "   baseline traffic: $(traffic 8 4 $OUT/XA0_traffic.txt)"; log "   d1 probe: $(hgp $D1 | cut -c1-120)"
kill_engine $P1 p1; start_engine $P1 0 gpt-oss_B_128k_2p2d_p1http_hook1276p.json 2p2d_tp8_pr13p /home/$ME/wt-health-http2; wait_up $P1 p1; log "   LB health: $(curl -s -m 10 -o /dev/null -w '%{http_code}' http://$P1:8080/health)"
log "   hook installs on p1: $(on $P1 "grep -a -c 'HOOK-1276P.*installed' $R/runs/2p2d_tp8_pr13p/p1/server.log")"
: > $OUT/XA_health.csv; ( while [[ ! -f $OUT/.stop ]]; do echo "$(date -u +%T),$(hgp $D1 | sed -E 's/^(.*) ([0-9]{3}) ([0-9.]+)$/\2,\3,\1/' | cut -c1-400)" >> $OUT/XA_health.csv; sleep 5; done ) & POLL=$!
T_START=$(date +%s)
for i in 1 2 3; do log "   batch $i (12 req, c=4, LB alternates prefills): $(traffic 12 4 $OUT/XA_traffic_$i.txt)"; log "   +$(( $(date +%s) - T_START )) s: $(hgp $D1 | cut -c1-520)"; done
sleep 6; touch $OUT/.stop; wait $POLL 2>/dev/null; rm -f $OUT/.stop
log "   d1 /health_generate during: $(awk -F, '{print $2}' $OUT/XA_health.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') (n=$(wc -l < $OUT/XA_health.csv)); first 503 at: $(awk -F, '$2==503 {print $1; exit}' $OUT/XA_health.csv) (traffic started $(date -u -d @$T_START +%T))"
log "   d2 (main) probe for comparison: $(curl -s -m 30 -o /dev/null -w '%{http_code} %{time_total}' http://$D2:30000/health_generate)"
log "=== X-E. restore p1 and d1 to main"
kill_engine $P1 p1; start_engine $P1 0 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up $P1 p1
kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up $D1 d1; log "   traffic: $(traffic 6 3 $OUT/XE_traffic.txt)"
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr13c done"
