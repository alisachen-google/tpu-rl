#!/usr/bin/env bash
# Live validation of the idle fallthrough in the HTTP-side /health_generate (branch pd-health-progress-http) on d1.
#  A. d1 on the branch: with requests in flight the tracker answers 200 in ms (no probe); idle, the core probe answers 200 in ~1 s.
#  B. the scheduler loop is hung while idle (hook triggered by a marker file): /health_generate must read 503 after the core 20 s probe timeout.
#  C. restore d1 to main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr14; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-http2
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [pr14] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_engine(){ local ip=$1 label=$2; on $ip "pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; sleep 3; true"
  local t0=$(date +%s); while curl -sf -m 3 http://$ip:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   $label down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local ip=$1 rank=$2 recipe=$3 run=$4 root=${5:-/home/$ME/sglang-torchtpu}
  on $ip "cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$recipe $rank $run &"; }
wait_up(){ local ip=$1 label=$2 t0=$(date +%s); until curl -sf -m 5 http://$ip:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   $label health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$ip:30000/health) after $(( $(date +%s)-t0 )) s"; }
hgp(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$1:30000/health_generate | tr -d '\n' | cut -c1-300; }
traffic(){ local n=$1 c=$2 file=$3; $PY - $n $c $LB > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,lb=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':text,'sampling_params':{'max_new_tokens':128,'temperature':0}}).encode()
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
FLAG=$OUT/hang.flag; rm -f $FLAG
log "=== idle fallthrough live validation; branch $(git -C $WT rev-parse --short HEAD); T=60 s, cold 120 s; hang hook armed on $FLAG"
log "=== A. d1 on the branch"
kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d_d1plug_hangfile.json 2p2d_tp8_pr14 $WT; wait_up $D1 d1
log "   hook installs: $(grep -a -c 'HOOK-HANG.*installed' $R/runs/2p2d_tp8_pr14/d1/server.log); patch errors: $(grep -a -c -i 'Traceback' $R/runs/2p2d_tp8_pr14/d1/server.log)"
log "   idle right after start (cold, no traffic yet): probe path expected, ~1 s: $(hgp $D1)"
log "   traffic (8 req, c=4): $(traffic 8 4 $OUT/A_traffic.txt)"
log "   idle right after traffic: probe path expected: $(hgp $D1)"
( traffic 6 3 $OUT/A2_traffic.txt > /dev/null 2>&1 & ); sleep 4
for i in 1 2 3; do log "   during traffic #$i (tracker path expected, ms): $(hgp $D1)"; sleep 3; done
sleep 60; log "   traffic finished: $(awk '{print $1}' $OUT/A2_traffic.txt | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
log "   idle 90 s: $(sleep 90; hgp $D1)"
log "=== B. hang the scheduler loop while idle"
touch $FLAG; T_HANG=$(date +%s); sleep 3
log "   hung ranks: $(grep -a -c 'hanging the scheduler loop' $R/runs/2p2d_tp8_pr14/d1/server.log) at $(date -u -d @$T_HANG +%T)"
for i in 1 2 3; do log "   +$(( $(date +%s)-T_HANG )) s: /health_generate $(hgp $D1 | sed 's/^ *//')  (expected: 503 after ~20 s, empty body)"; done
log "   /health (no probe on this recipe): $(curl -s -m 10 -o /dev/null -w '%{http_code} %{time_total}' http://$D1:30000/health)"
log "   core log lines: $(grep -a -c 'Health check failed' $R/runs/2p2d_tp8_pr14/d1/server.log)"
log "=== C. restore d1 to main"
rm -f $FLAG; kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up $D1 d1
log "   traffic: $(traffic 6 3 $OUT/C_traffic.txt)"; log "   d1 /health_generate: $(hgp $D1)"
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr14 done"
