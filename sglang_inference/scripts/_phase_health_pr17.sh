#!/usr/bin/env bash
# pr17: live smoke of head 3fac03df28 on d1 (new failure-streak rule, decode progress signal via the detokenizer, /ready coupling).
#  A. hook 1276 (transfers never complete): 503 with body once the failure streak spans the window; /ready must agree; recovery after a quiet window.
#  B. buffered progress: non-streaming 4096-token generations; probes during must read 200 in ms (tracker) well past T; the progress signal count in the log.
#  C. hang while idle (file-triggered): 503 at 20 s via the core probe; /ready 503 too.
#  D. restore to main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr17; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-http2
log(){ echo "$(date -u +%FT%TZ) [pr17] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; pkill -9 -u $ME -f 'sglang::[d]etokenizer'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
hgp(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$D1:30000/health_generate | tr -d '\n' | cut -c1-330; }
code_lat(){ hgp | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1 \2/'; }
ready(){ curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/ready; }
traffic(){ local n=$1 c=$2 tok=$3 file=$4; $PY - $n $c $tok $LB > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,tok,lb=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':f'request {i} {time.time():.3f}: '+text,'sampling_params':{'max_new_tokens':tok,'temperature':0,'ignore_eos':True},'stream':False}).encode()
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
FLAG=$R/2p2d_tp8/pr14/hang.flag; rm -f $FLAG
log "=== smoke of head $(git -C $WT rev-parse --short HEAD); T=60 s, cold 120 s, transport timeouts 60 s"
log "=== A. transfers never complete (hook 1276)"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276.json 2p2d_tp8_pr17a $WT; wait_up
log "   tracebacks: $(grep -a -c Traceback $R/runs/2p2d_tp8_pr17a/d1/server.log); idle: $(code_lat); /ready $(ready)"
T0=$(date +%s); : > $OUT/A.csv; ( for i in 1 2 3 4; do traffic 3 3 16 $OUT/A_b$i.txt > /dev/null 2>&1; sleep 5; done & ) 
for i in $(seq 1 44); do echo "$(( $(date +%s)-T0 )),$(code_lat),$(ready)" >> $OUT/A.csv; sleep 5; done
log "   probes: $(awk -F, '{print $2}' $OUT/A.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'); first 503 at +$(awk -F, '$2==503{print $1; exit}' $OUT/A.csv) s; /ready at that moment: $(awk -F, '$2==503{print $4; exit}' $OUT/A.csv)"
log "   body at first 503: $(hgp | cut -c1-330)"
log "   clients: $(cat $OUT/A_b*.txt | awk '{print $1}' | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
log "   recovery: +70 s quiet $(sleep 70; code_lat) /ready $(ready); +130 s quiet $(sleep 60; code_lat) /ready $(ready)   (expected 200 once the last failure is a window old)"
log "=== B. buffered (non-streaming) long generations: progress must still be seen"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug.json 2p2d_tp8_pr17b $WT; wait_up
( traffic 4 4 4096 $OUT/B_traffic.txt > $OUT/B.summary 2>&1 & ); sleep 10
: > $OUT/B.csv; T0=$(date +%s); while ! [[ -s $OUT/B.summary ]] && (( $(date +%s)-T0 < 300 )); do echo "$(( $(date +%s)-T0 )),$(code_lat)" >> $OUT/B.csv; sleep 5; done
log "   traffic: $(cat $OUT/B.summary); probes during (code x n, max latency): $(awk -F, '{print $2}' $OUT/B.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') max $(awk -F, '{print $3}' $OUT/B.csv | sort -g | tail -1) s over $(tail -1 $OUT/B.csv | cut -d, -f1) s"
log "   progress signals seen by the HTTP process are internal; detokenizer forwarded HealthCheckOutput: $(grep -a -c 'HealthCheckOutput' $R/runs/2p2d_tp8_pr17b/d1/server.log) log lines; tracebacks: $(grep -a -c Traceback $R/runs/2p2d_tp8_pr17b/d1/server.log)"
log "=== C. hang while idle"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hangfile.json 2p2d_tp8_pr17c $WT; wait_up
log "   warm traffic: $(traffic 2 2 32 $OUT/C_warm.txt); idle: $(code_lat)"; touch $FLAG; sleep 3; log "   hung ranks: $(grep -a -c 'hanging the scheduler loop' $R/runs/2p2d_tp8_pr17c/d1/server.log)"
for i in 1 2; do log "   probe: $(code_lat)  /ready $(ready)   (expected 503 after ~20 s, /ready 503)"; done
rm -f $FLAG
log "=== D. restore d1 to main"; kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic: $(traffic 4 2 32 $OUT/D_traffic.txt)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr17 done"
