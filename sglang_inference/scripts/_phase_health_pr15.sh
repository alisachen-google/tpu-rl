#!/usr/bin/env bash
# pr15: live smoke of the review fixes on d1 (branch pd-health-progress-http). The decode's transfers never complete
# (hook 1276): every request aborts with a 5xx at the 60 s transport timeout -> 503 with a JSON body at T_cold; the
# body must still carry role/state/reason/pd_health (JSONResponse + to_builtins path). Then a quiet gap with no probes
# followed by one request must read 200, not an immediate 503 (observe() fix). Restore to main at the end.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr15; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-http2
log(){ echo "$(date -u +%FT%TZ) [pr15] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
hgp(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$D1:30000/health_generate | tr -d '\n' | cut -c1-420; }
traffic(){ local n=$1 c=$2 file=$3; $PY - $n $c $LB > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,lb=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':f'request {i}: '+text,'sampling_params':{'max_new_tokens':64,'temperature':0}}).encode()
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(lb+'/generate',data=body,headers={'Content-Type':'application/json'}),timeout=400); code=r.status; r.read()
    except urllib.error.HTTPError as e: code=e.code
    except Exception as e: code=f'ERR:{type(e).__name__}'
    return f'{code} {time.time()-t0:.1f}s'
with cf.ThreadPoolExecutor(c) as ex:
    for line in ex.map(one, range(n)): print(line, flush=True)
PY
  awk '{print $1}' $file | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }
log "=== review-fix smoke; branch $(git -C $WT rev-parse --short HEAD); T=60 s, cold 120 s, transport timeouts 60 s"
log "=== A. d1 on the branch with the 1276 hook (receive never completes)"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276.json 2p2d_tp8_pr15 $WT; wait_up
log "   hook installs: $(grep -a -c 'HOOK-1276.*installed' $R/runs/2p2d_tp8_pr15/d1/server.log); tracebacks: $(grep -a -c Traceback $R/runs/2p2d_tp8_pr15/d1/server.log)"
log "   idle after start: $(hgp)"
T0=$(date +%s); ( traffic 6 3 $OUT/A_traffic.txt > $OUT/A.summary 2>&1 & )
sleep 20; log "   +20 s (requests held): $(hgp)"
sleep 115; log "   +135 s: $(hgp)"
until [[ -s $OUT/A.summary ]] || (( $(date +%s)-T0 > 400 )); do sleep 5; done; log "   traffic: $(cat $OUT/A.summary)"
log "   abort status codes seen by the engine: $(grep -a -o 'status_code[^,}]*' $R/runs/2p2d_tp8_pr15/d1/server.log | sort | uniq -c | tr '\n' ' ' | cut -c1-200)"
log "   +$(( $(date +%s)-T0 )) s, all aborted: $(hgp)"
log "=== B. quiet gap with NO probes, then one request: must not read 503 at once"
sleep 150
log "   one request sent (it will abort at 60 s); probe 3 s after send:"; ( traffic 1 1 $OUT/B_traffic.txt > $OUT/B.summary 2>&1 & ); sleep 3; log "   $(hgp)"
sleep 20; log "   +23 s: $(hgp)"
until [[ -s $OUT/B.summary ]] || (( $(date +%s)-T0 > 900 )); do sleep 5; done; log "   request result: $(cat $OUT/B.summary)"
log "=== C. restore d1 to main"; kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic: $(traffic 4 2 $OUT/C_traffic.txt)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr15 done"
