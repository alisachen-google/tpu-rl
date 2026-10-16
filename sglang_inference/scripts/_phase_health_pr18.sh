#!/usr/bin/env bash
# #1392 rewrite (TM probe + request table): live cases on d1 of the 2P/2D group.
#  A. transfers never complete (hook 1276): 503 "stalled" once the oldest waiter passes the window (48 s); 200 again as soon as the 60 s transport aborts drain the table.
#  B. hang while idle: 503 "probe" after ~20 s, /ready 503.
#  C. 4xx-only rejections: 200 throughout.  D. busy long generations: 200 in ms; idle 200 ~1 s.
#  E. window >= transport timeout: launcher refuses to start.  F. tokenizer_worker_num=2: probe 200, no plugin warning needed.  G. restore main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; ME=alisachen_google_com
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; WT=$HOME/wt-health-http2
OUT=$R/2p2d_tp8/pr18; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [pr18] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; pkill -9 -u $ME -f 'sglang::[d]etokenizer'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
hg(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$D1:30000/health_generate | tr -d '\n' | cut -c1-260; }
code_lat(){ hg | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1 \2/'; }
ready(){ curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/ready; }
traffic(){ local n=$1 c=$2 tok=$3 file=$4 target=${5:-$LB}; $PY - $n $c $tok $target > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,tok,lb=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':f'request {i} {time.time():.3f}: '+text,'sampling_params':{'max_new_tokens':tok,'temperature':0,'ignore_eos':True},'stream':False}).encode()
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
summ(){ awk -F, '{print $2}' $1 | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }

log "=== branch $(git -C $WT rev-parse --short HEAD); window 48 s, transport timeouts 60 s, probe timeout 20 s"
log "=== A. transfers never complete (hook 1276): bursts every 20 s, probe every 4 s for 150 s"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276.json 2p2d_tp8_pr18a $WT; wait_up
log "   idle: $(code_lat) /ready $(ready); tracebacks: $(grep -a -c Traceback $R/runs/2p2d_tp8_pr18a/d1/server.log)"
T0=$(date +%s); : > $OUT/A.csv
( for i in 1 2 3 4 5 6 7; do traffic 6 6 16 $OUT/A_b$i.txt > /dev/null 2>&1 & sleep 20; done ) &
for i in $(seq 1 38); do r=$(hg); echo "$(( $(date +%s)-T0 )),$(echo "$r" | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1/'),$(ready),$(echo "$r" | grep -o '"reason":"[^"]*"' | cut -c1-120)" >> $OUT/A.csv; sleep 4; done
wait
log "   codes: $(summ $OUT/A.csv); first 503 at +$(awk -F, '$2==503{print $1; exit}' $OUT/A.csv) s (expected ~48-60 s after the first burst)"
log "   timeline (s:code/ready): $(awk -F, '{printf "%s:%s/%s ", $1, $2, $3}' $OUT/A.csv)"
log "   first 503 reason: $(awk -F, '$2==503{print $4; exit}' $OUT/A.csv)"
log "   clients (expected 5xx on d1 after 60 s, 200 via d2): $(cat $OUT/A_b*.txt | awk '{print $1}' | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
sleep 65; log "   quiet 65 s later (table drained): $(code_lat) /ready $(ready)   (expected 200 at once, no window to age out)"
log "=== B. hang while idle"
FLAG=$R/2p2d_tp8/pr14/hang.flag; rm -f $FLAG
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hangfile.json 2p2d_tp8_pr18b $WT; wait_up
log "   warm traffic: $(traffic 2 2 32 $OUT/B_warm.txt); idle: $(code_lat)"; touch $FLAG; sleep 3; log "   hung ranks: $(grep -a -c 'hanging the scheduler loop' $R/runs/2p2d_tp8_pr18b/d1/server.log)"
log "   probe: $(hg | cut -c1-200)"; log "   /ready $(ready)   (expected 503 after ~20 s with reason 'probe: ...', /ready 503)"
rm -f $FLAG
log "=== C. 4xx-only rejections for 100 s"
FLAG=$R/2p2d_tp8/pr16/abort400.flag; touch $FLAG
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_abort400.json 2p2d_tp8_pr18c $WT; wait_up
T0=$(date +%s); : > $OUT/C.csv
for i in $(seq 1 10); do ( traffic 1 1 16 $OUT/C_req_$i.txt > /dev/null 2>&1 & ); sleep 10; echo "$(( $(date +%s)-T0 )),$(code_lat | cut -d' ' -f1)" >> $OUT/C.csv; done
log "   probes: $(summ $OUT/C.csv)  (expected 200 x10); hook rejections: $(grep -a -c 'abort400' $R/runs/2p2d_tp8_pr18c/d1/server.log)"
rm -f $FLAG
log "=== D. busy, healthy: 4 x 2048-token generations"
( traffic 4 4 2048 $OUT/D_traffic.txt > $OUT/D.summary 2>&1 & ); sleep 6
for i in 1 2 3; do log "   during traffic: $(code_lat)   (expected 200 in ms)"; sleep 3; done
wait; sleep 20; log "   traffic: $(cat $OUT/D.summary); idle after: $(code_lat)   (expected 200 ~1 s)"
log "=== E. window 60 s with 60 s transport timeouts: launcher must refuse"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_badwindow.json 2p2d_tp8_pr18e $WT; sleep 90
log "   ValueError lines: $(grep -a -c 'SGLANG_HEALTH_GENERATE_STALL_WINDOW_S=60.0 must be' $R/runs/2p2d_tp8_pr18e/d1/server.log); launcher alive: $(pgrep -u $ME -fc 'benchmark.tpu.launch_[s]erver'); health $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://$D1:30000/health)   (expected 1 / 0 / 000)"
log "=== F. tokenizer_worker_num=2"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_workers2.json 2p2d_tp8_pr18f $WT; wait_up
( traffic 4 4 512 $OUT/F_traffic.txt > $OUT/F.summary 2>&1 & ); sleep 5
log "   during traffic: $(code_lat); again: $(code_lat)"; wait; log "   traffic: $(cat $OUT/F.summary); idle: $(code_lat); tracebacks: $(grep -a -c Traceback $R/runs/2p2d_tp8_pr18f/d1/server.log)"
log "=== G. restore d1 to main"
kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic: $(traffic 4 2 32 $OUT/G_traffic.txt)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr18 done"
