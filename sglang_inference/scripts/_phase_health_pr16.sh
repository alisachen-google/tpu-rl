#!/usr/bin/env bash
# pr16: live checks for the review findings on d1 (branch pd-health-progress-http). T=60 s, cold 120 s.
#  A. 4xx aborts only (hook forces the decode's real BAD_REQUEST abort path): one bad request every 10 s for 150 s with no
#     other traffic -> /health_generate must stay 200 (before the fix each 400 counted as a failure: 503 at the window).
#  B. scheduler-side running-timeout aborts (SGLANG_REQ_RUNNING_TIMEOUT=15): long generations are aborted by the
#     scheduler with 503 through AbortReq -> _handle_abort_req; then idle 100 s -> 200 via the core probe throughout
#     (before the fifth wrapper the rid stayed in flight; the old handler then read 503 at T).
#  C. duty cycle of a permanently broken decode (hook 1276): poll every 5 s for 8 min, send a burst whenever the engine
#     reads 200 and nothing is in flight (what a router would do); report the share of time at 200.
#  D. tokenizer_worker_num=2: the startup warning must appear and /health_generate must behave as core (probe path).
#  E. restore d1 to main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr16; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-http2
log(){ echo "$(date -u +%FT%TZ) [pr16] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; pkill -9 -u $ME -f 'sglang::[d]etokenizer'; pkill -9 -u $ME -f '[g]ranian'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local limit=${1:-1500} t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > limit )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
hgp(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$D1:30000/health_generate | tr -d '\n' | cut -c1-300; }
code_lat(){ hgp | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1 \2/'; }
traffic(){ local n=$1 c=$2 tok=$3 file=$4; $PY - $n $c $tok $LB > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,tok,lb=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':f'request {i} {time.time():.3f}: '+text,'sampling_params':{'max_new_tokens':tok,'temperature':0,'ignore_eos':True}}).encode()
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
d1_reqs(){ curl -s -m 5 http://$D1:30000/metrics | grep -E '^sglang:num_requests_total\{' | head -1 | awk '{print $NF}'; }
FLAG=$OUT/abort400.flag; rm -f $FLAG
log "=== review-finding live checks; branch $(git -C $WT rev-parse --short HEAD)"
log "=== A. 4xx aborts only"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_abort400.json 2p2d_tp8_pr16 $WT; wait_up
log "   hook installs: $(grep -a -c 'HOOK-ABORT400.*installed' $R/runs/2p2d_tp8_pr16/d1/server.log)"
touch $FLAG; : > $OUT/A.csv; T0=$(date +%s)
for i in $(seq 1 15); do ( traffic 1 1 16 $OUT/A_req_$i.txt > $OUT/A_req_$i.summary 2>&1 & ); sleep 10; echo "$(( $(date +%s)-T0 )),$(code_lat),$(cat $OUT/A_req_$i.summary 2>/dev/null | tr -d '\n')" >> $OUT/A.csv; done
rm -f $FLAG; sleep 3
log "   client codes: $(cat $OUT/A_req_*.txt | awk '{print $1}' | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'); hook rejections on d1: $(grep -a -c 'rejecting' $R/runs/2p2d_tp8_pr16/d1/server.log)"
log "   /health_generate over 150 s of 4xx-only traffic: $(awk -F, '{print $2}' $OUT/A.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') (expected: 200 x15; before the fix 503 from ~120 s)"
log "   last line: $(tail -1 $OUT/A.csv)"
log "=== B. scheduler-side running-timeout aborts (15 s), then idle"
N0=$(d1_reqs); ( traffic 2 2 4096 $OUT/B_traffic.txt > $OUT/B.summary 2>&1 & ); sleep 8; log "   +8 s (running): $(code_lat)"
until [[ -s $OUT/B.summary ]] || (( $(date +%s)-T0 > 900 )); do sleep 2; done
log "   client result: $(cat $OUT/B.summary) $(grep -m1 -o 'running timeout reached' $R/runs/2p2d_tp8_pr16/d1/server.log | head -1); d1 requests: $N0 -> $(d1_reqs)"
log "   right after the aborts: $(code_lat)"
for s in 30 60 100; do sleep $(( s==30 ? 30 : s==60 ? 30 : 40 )); log "   idle +$s s: $(code_lat)   (expected 200 ~1 s via the core probe; the old handler read 503 at 60 s)"; done
log "=== C. duty cycle of a permanently broken decode (hook 1276), 8 min"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276.json 2p2d_tp8_pr16c $WT; wait_up
: > $OUT/C.csv; T0=$(date +%s); BURST=""
while (( $(date +%s)-T0 < 480 )); do
  r=$(code_lat); c=${r%% *}
  if [[ $c == 200 ]] && { [[ -z $BURST ]] || ! kill -0 $BURST 2>/dev/null; }; then ( traffic 3 3 16 $OUT/C_burst_$(date +%s).txt > /dev/null 2>&1 ) & BURST=$!; sent=1; else sent=0; fi
  echo "$(( $(date +%s)-T0 )),$c,$sent" >> $OUT/C.csv; sleep 5
done
log "   samples: $(awk -F, '{print $2}' $OUT/C.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'); bursts sent: $(awk -F, '$3==1' $OUT/C.csv | wc -l); share of time at 200: $(awk -F, '{n++; if($2==200)k++} END{printf "%.0f%%", 100*k/n}' $OUT/C.csv)"
log "   timeline (s,code): $(awk -F, 'NR==1||$2!=p{printf "%s:%s ",$1,$2} {p=$2}' $OUT/C.csv)"
log "=== D. tokenizer_worker_num=2"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_workers2.json 2p2d_tp8_pr16d $WT; wait_up 900
log "   warning lines: $(grep -a -c 'not installed in the 2 tokenizer worker' $R/runs/2p2d_tp8_pr16d/d1/server.log); tracebacks: $(grep -a -c Traceback $R/runs/2p2d_tp8_pr16d/d1/server.log)"
if curl -sf -m 5 http://$D1:30000/health >/dev/null; then
  ( traffic 4 4 1024 $OUT/D_traffic.txt > $OUT/D.summary 2>&1 & ); sleep 6; log "   during traffic: $(code_lat)   (expected ~1 s: the worker keeps the core probe)"; log "   again: $(code_lat)"
  until [[ -s $OUT/D.summary ]] || (( $(date +%s)-T0 > 1500 )); do sleep 2; done; log "   traffic: $(cat $OUT/D.summary)"
else log "   d1 did not come up with two tokenizer workers within the time box; last log lines:"; tail -5 $R/runs/2p2d_tp8_pr16d/d1/server.log | cut -c1-200 | sed 's/^/      /'; fi
log "=== E. restore d1 to main"; kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic: $(traffic 4 2 64 $OUT/E_traffic.txt)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr16 done"
