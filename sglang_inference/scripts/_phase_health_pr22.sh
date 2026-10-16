#!/usr/bin/env bash
# #1392 head with crash-only rejection: probe timeouts and stalls are diagnostics (200 + JSON body); only a confirmed worker crash is 503.
#  A (d1, W=48 s, hook 1276): transfers never complete -> every probe 200; from ~48 s the body is JSON with a "stalled" reason; /ready stays 200; clients 5xx from d1.
#  Rerun of phase A only: with abort outputs excluded from progress the "stalled" diagnostic must recur for every burst past 48 s, not just the first.
#  D restore d1 to main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; ME=alisachen_google_com
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; WT=$HOME/wt-health-http2
OUT=$R/2p2d_tp8/pr22; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [pr22] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; pkill -9 -u $ME -f 'sglang::[d]etokenizer'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
hg(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$D1:30000/health_generate | tr -d '\n' | cut -c1-320; }
code_of(){ hg | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1/'; }
ready(){ curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/ready; }
summ(){ awk -F, -v c=$2 '{print $c}' $1 | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }

log "=== branch $(git -C $WT rev-parse --short HEAD); W = 48 s, transport timeouts 60 s, probe timeout 20 s; crash-only rejection; abort outputs no longer count as progress"
log "=== A. transfers never complete (hook 1276): bursts every 20 s, probe every 4 s for 120 s"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276.json 2p2d_tp8_pr22a $WT; wait_up
log "   idle: $(hg)   /ready $(ready)"
T0=$(date +%s); : > $OUT/A.csv
( for i in 1 2 3 4 5 6; do $PY $S/router_traffic.py 6 6 16 > $OUT/A_b$i.txt 2>&1 & sleep 20; done; wait ) &
for i in $(seq 1 30); do r=$(hg); echo "$(( $(date +%s)-T0 )),$(echo "$r" | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1/'),$(ready),$(echo "$r" | grep -o '"stalled":[a-z]*'),$(echo "$r" | grep -o '"reason":"[^"]*"' | cut -c1-110)" >> $OUT/A.csv; sleep 4; done; wait
log "   codes: $(summ $OUT/A.csv 2); /ready: $(summ $OUT/A.csv 3); stalled field: $(summ $OUT/A.csv 4)"
log "   first stalled diagnostic at +$(awk -F, '$4=="\"stalled\":true"{print $1; exit}' $OUT/A.csv) s: $(awk -F, '$4=="\"stalled\":true"{print $5; exit}' $OUT/A.csv)   (expected: all 200, /ready 200, stalled:true from ~48 s)"
log "   clients: $(cat $OUT/A_b*.txt | tr ' ' '\n' | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')   (expected 5xx from d1 after the 60 s aborts)"
log "   diagnostics logged on d1: $(grep -a -c 'P/D health diagnostics' $R/runs/2p2d_tp8_pr22a/d1/server.log)"
log "=== D. restore d1 to main"
kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic: $($PY $S/router_traffic.py 4 2 32)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr22 done"
