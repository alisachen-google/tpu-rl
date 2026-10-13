#!/usr/bin/env bash
# Live validation of PR #1342 (readiness state on /metrics): d1 on the stacked branch with the #1276 hook and N=3, T=60 s.
# Expect sglang:pd_health_state{state="unhealthy"} 1 while the gate is open and state="probation" after the quiet period.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr5; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-metrics
log(){ echo "$(date -u +%FT%TZ) [pr5] $*"; }
kill_d1(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_d1(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}
  ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & )
  local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   d1 ($recipe) health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
pdm(){ curl -s -m 5 http://$D1:30000/metrics | grep -E '^sglang:pd_health_(state|consecutive_failures|probation_wait_seconds)\{' | sed -E 's/\{[^}]*(state="[a-z]+")?[^}]*\}/{\1}/' | paste -sd' '; }
hg(){ curl -s -m 40 -o /dev/null -w '%{http_code} %{time_total}' http://$D1:30000/health_generate; }
traffic(){ local n=$1 c=$2; $PY - $n $c $LB <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,lb=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':text,'sampling_params':{'max_new_tokens':64,'temperature':0}}).encode()
    try: r=urllib.request.urlopen(urllib.request.Request(lb+'/generate',data=body,headers={'Content-Type':'application/json'}),timeout=400); r.read(); return r.status
    except urllib.error.HTTPError as e: return e.code
    except Exception as e: return type(e).__name__
with cf.ThreadPoolExecutor(c) as ex: codes=list(ex.map(one, range(n)))
from collections import Counter; print(' '.join(f'{k} x{v}' for k,v in sorted(Counter(map(str,codes)).items())))
PY
}
log "=== PR #1342 live validation; branch $(git -C $WT rev-parse --short HEAD) on d1 with the #1276 hook, N=3, T=60 s"
kill_d1; start_d1 gpt-oss_B_128k_2p2d_d1metrics_hook1276.json 2p2d_tp8_pr5 $WT
log "   metrics at start: $(pdm)"
log "   traffic (10 req, c=2; d1-paired ones fail after 60 s): $(traffic 10 2)"; sleep 5
log "   metrics after failures: $(pdm)"; log "   /health_generate: $(hg)"
sleep 70; log "   metrics after 70 s quiet: $(pdm)"; log "   /health_generate: $(hg)"
log "=== restore d1 to main"; kill_d1; start_d1 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; log "   d1 /health_generate: $(hg)"; log "   traffic: $(traffic 4 2)"
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr5 done"
