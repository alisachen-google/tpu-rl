#!/usr/bin/env bash
# pr14b: the busy path of the idle fallthrough. d1 on the branch; long generations keep requests in flight on d1 while
# /health_generate is polled: the tracker must answer 200 in ms (no probe) while requests are held, ~1 s (probe) once idle.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr14b; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-http2
log(){ echo "$(date -u +%FT%TZ) [pr14b] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
hgp(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$D1:30000/health_generate | tr -d '\n' | cut -c1-300; }
nreq(){ curl -s -m 5 http://$D1:30000/metrics | grep -E '^sglang:num_requests_total\{' | head -1 | awk '{print $NF}'; }
traffic(){ local n=$1 c=$2 tok=$3 file=$4; $PY - $n $c $tok $LB > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,tok,lb=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':f'request {i}: '+text,'sampling_params':{'max_new_tokens':tok,'temperature':0,'ignore_eos':True}}).encode()
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(lb+'/generate',data=body,headers={'Content-Type':'application/json'}),timeout=600); code=r.status; r.read()
    except urllib.error.HTTPError as e: code=e.code
    except Exception as e: code=f'ERR:{type(e).__name__}'
    return f'{code} {time.time()-t0:.1f}s'
with cf.ThreadPoolExecutor(c) as ex:
    for line in ex.map(one, range(n)): print(line, flush=True)
PY
  awk '{print $1}' $file | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }
log "=== busy path of the idle fallthrough; branch $(git -C $WT rev-parse --short HEAD)"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug.json 2p2d_tp8_pr14b $WT; wait_up
log "   idle after start: $(hgp)   (probe path, ~1 s)"
N0=$(nreq); log "   d1 requests before: $N0"
( traffic 8 4 4096 $OUT/traffic.txt > $OUT/traffic.summary 2>&1 & ); sleep 8
: > $OUT/during.csv; for i in $(seq 1 12); do echo "$(date -u +%T),$(nreq),$(hgp | sed -E 's/^ *(.*) ([0-9]{3}) ([0-9.]+)$/\2,\3,\1/')" >> $OUT/during.csv; sleep 2; done
log "   during traffic (time, d1 requests so far, code, latency):"; sed 's/^/      /' $OUT/during.csv
log "   latencies during: max $(awk -F, '{print $4}' $OUT/during.csv | sort -g | tail -1) s, min $(awk -F, '{print $4}' $OUT/during.csv | sort -g | head -1) s; d1 requests after: $(nreq)"
t0=$(date +%s); until [[ -s $OUT/traffic.summary ]] || (( $(date +%s)-t0 > 600 )); do sleep 5; done; log "   traffic result: $(cat $OUT/traffic.summary)"
log "   idle again: $(hgp)   (probe path, ~1 s)"
log "=== restore d1 to main"; kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic: $(traffic 4 2 128 $OUT/C_traffic.txt)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr14b done"
