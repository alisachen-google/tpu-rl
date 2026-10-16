#!/usr/bin/env bash
# #1392: end-to-end with the real gateway (sgl-model-gateway on 127.0.0.1:8081, --health-check-endpoint /health_generate,
# interval 10 s, timeout 30 s, 3 failures remove, 2 successes restore). Every phase samples the engine verdict, the gateway's
# view of d1 (is_healthy) and client outcomes every 5 s.
#  P1 hang while busy, then release (resumable hook, W=48): d1 must go 503 after ~48 s of no progress, the gateway must drop d1,
#     clients must keep succeeding via d2, and after the flag is removed d1 must return and the gateway must take it back.
#  P2 transfers never complete (hook 1276, W=48): 503 spans only 48..60 s per burst (12 s), which is under 3 polls x 10 s, so the
#     gateway is expected NOT to remove d1. P2b repeats with W=30 (30 s span): the gateway must remove d1.
#  P3 restore d1 to main: the gateway must mark d1 healthy again and route to it.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; ME=alisachen_google_com
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; GW=http://127.0.0.1:8081; WT=$HOME/wt-health-http2
OUT=$R/2p2d_tp8/pr24; mkdir -p $OUT; FLAG=$OUT/hang.flag; rm -f $FLAG
log(){ echo "$(date -u +%FT%TZ) [pr24] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; pkill -9 -u $ME -f 'sglang::[d]etokenizer'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
wait_gw_healthy(){ local t0=$(date +%s); until [[ $(gw_d1) == true ]] || (( $(date +%s)-t0 > 300 )); do sleep 5; done; log "   gateway sees d1 healthy=$(gw_d1) after $(( $(date +%s)-t0 )) s"; }
code(){ curl -s -m 40 -o /dev/null -w '%{http_code}' http://$D1:30000/health_generate; }
gw_d1(){ curl -s -m 5 $GW/workers | $PY -c "import json,sys; print([w['is_healthy'] for w in json.load(sys.stdin)['workers'] if w['url'].startswith('http://$D1')][0])" 2>/dev/null || echo "?"; }
# traffic_loop <seconds> <tokens> <client timeout> <out>: one request every 4 s through the gateway, "t,code" per line
traffic_loop(){ local secs=$1 tok=$2 tmo=$3 out=$4; : > $out; local t0=$(date +%s); while (( $(date +%s)-t0 < secs )); do ( s=$(( $(date +%s)-t0 )); c=$($PY $S/router_traffic.py 1 1 $tok $GW 2>/dev/null); echo "$s,$c" >> $out ) & sleep 4; done; wait; }
sample_loop(){ local secs=$1 csv=$2; : > $csv; local t0=$(date +%s); while (( $(date +%s)-t0 < secs )); do echo "$(( $(date +%s)-t0 )),$(code),$(gw_d1)" >> $csv; sleep 5; done; }
first(){ awk -F, -v col=$2 -v val="$3" '$col==val{print $1; exit}' $1; }
tl(){ awk -F, '{printf "%s:%s/%s ", $1, $2, $3}' $1; }

log "=== branch $(git -C $WT rev-parse --short HEAD); gateway $GW health=$(curl -s -m 3 -o /dev/null -w '%{http_code}' $GW/health), d1 healthy=$(gw_d1)"
log "=== P1. hang while busy (resumable hook), then release; W = 48 s"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hangresume.json 2p2d_tp8_pr24a $WT; wait_up; wait_gw_healthy
traffic_loop 300 128 150 $OUT/P1_clients.csv & TL=$!
( sleep 20; touch $FLAG; echo "$(date -u +%T) flag set" >> $OUT/P1_events.txt; sleep 140; rm -f $FLAG; echo "$(date -u +%T) flag removed" >> $OUT/P1_events.txt ) &
sample_loop 300 $OUT/P1.csv; wait $TL
log "   events: $(tr '\n' ';' < $OUT/P1_events.txt)"
log "   engine first 503 at +$(first $OUT/P1.csv 2 503) s (flag at +20: expected ~+70); gateway first unhealthy at +$(first $OUT/P1.csv 3 false) s (expected <= 30 s later)"
log "   after release at +160: engine first 200 at +$(awk -F, '$1>160 && $2==200{print $1; exit}' $OUT/P1.csv) s; gateway healthy again at +$(awk -F, '$1>160 && $3=="true"{print $1; exit}' $OUT/P1.csv) s"
log "   timeline (s:engine/gateway): $(tl $OUT/P1.csv)"
log "   clients by phase: before flag $(awk -F, '$1<20{print $2}' $OUT/P1_clients.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') | flag..removal $(awk -F, '$1>=20 && $1<160{print $2}' $OUT/P1_clients.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') | after $(awk -F, '$1>=160{print $2}' $OUT/P1_clients.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
log "   hung/resumed log lines on d1: $(grep -a -c 'hanging the scheduler loop' $R/runs/2p2d_tp8_pr24a/d1/server.log) / $(grep -a -c 'resuming the scheduler loop' $R/runs/2p2d_tp8_pr24a/d1/server.log)"
log "=== P2. transfers never complete, W = 48 s against 60 s aborts: 12 s of 503 per burst, under 3 polls"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276.json 2p2d_tp8_pr24b $WT; wait_up; wait_gw_healthy
( for i in 1 2 3 4 5 6 7; do $PY $S/router_traffic.py 6 6 16 $GW > $OUT/P2_b$i.txt 2>&1 & sleep 20; done; wait ) &
sample_loop 160 $OUT/P2.csv; wait
log "   engine codes: $(awk -F, '{print $2}' $OUT/P2.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'); gateway d1 healthy values: $(awk -F, '{print $3}' $OUT/P2.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')   (expected: 503 bursts of ~12 s, gateway stays true)"
log "   timeline: $(tl $OUT/P2.csv)"
log "   clients: $(cat $OUT/P2_b*.txt | tr ' ' '\n' | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
log "=== P2b. same with W = 30 s: 30 s of 503 per burst, 3 polls"
kill_engine; start_engine gpt-oss_B_128k_2p2d_d1plug_hook1276_w30.json 2p2d_tp8_pr24c $WT; wait_up; wait_gw_healthy
( for i in 1 2 3 4 5 6 7 8; do $PY $S/router_traffic.py 6 6 16 $GW > $OUT/P2b_b$i.txt 2>&1 & sleep 20; done; wait ) &
sample_loop 180 $OUT/P2b.csv; wait
log "   engine codes: $(awk -F, '{print $2}' $OUT/P2b.csv | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'); gateway first unhealthy at +$(first $OUT/P2b.csv 3 false) s   (expected: removed once three 10 s polls land in a 30 s 503 span)"
log "   timeline: $(tl $OUT/P2b.csv)"
log "   clients: $(cat $OUT/P2b_b*.txt | tr ' ' '\n' | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
log "=== P3. restore d1 to main; the gateway must take it back"
kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up; wait_gw_healthy
log "   traffic via gateway: $($PY $S/router_traffic.py 8 4 32 $GW)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr24 done"
