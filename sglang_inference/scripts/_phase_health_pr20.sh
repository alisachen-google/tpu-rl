#!/usr/bin/env bash
# #1392 plugin-only head: the cases the old suites covered that the new unit suites leave to documentation.
#  A (d1, W=20 s): A1 one 128K-token prompt, non-streaming: 503 "stalled" from ~21 s until the first token, then 200 (the TTFT constraint on W).
#     A2 one 8192-token non-streaming generation: 503 from ~21 s until it finishes. A3 the same request streaming: 200 throughout.
#  B (d1, W=48 s, SGLANG_REQ_RUNNING_TIMEOUT=15): B1 engine aborts (5xx) drain the table -> probe 200 (accepted false negative). B2 client disconnect -> 200.
#  C (p1 on the branch, hook 1276p): the prefill side's own verdict while its KV never reaches the decode.
#  D restore p1 and d1 to main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr20; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-http2
PROMPTS=$HOME/pd_disagg/prompts/gpt-oss
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [pr20] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
kill_engine(){ local ip=$1 label=$2; on $ip "pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; if [[ $ip == $P1 ]]; then pkill -9 -u $ME -f 'sglang::[r]outer'; pkill -9 -u $ME -f 'sglang_router.launch_[r]outer'; fi; sleep 3; true"
  local t0=$(date +%s); while curl -sf -m 3 http://$ip:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   $label down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local ip=$1 rank=$2 recipe=$3 run=$4 root=${5:-/home/$ME/sglang-torchtpu}
  on $ip "cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$recipe $rank $run &"; }
wait_up(){ local ip=$1 label=$2 t0=$(date +%s); until curl -sf -m 5 http://$ip:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   $label health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$ip:30000/health) after $(( $(date +%s)-t0 )) s"; }
hg(){ curl -s -m 40 -w ' %{http_code} %{time_total}' http://$1:30000/health_generate | tr -d '\n' | cut -c1-300; }
code_of(){ hg $1 | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1/'; }
pair(){ $PY $S/pd_pair_request.py "$1" "$2" "$3" "$4" $P1 $D1 > "$5" 2>&1; }
# probe_during <host> <csv> <pid>: probe every 3 s until the process exits; rows are "t,code,reason"
probe_during(){ local host=$1 csv=$2 pid=$3 t0=$(date +%s); : > $csv; while kill -0 $pid 2>/dev/null; do r=$(hg $host); echo "$(( $(date +%s)-t0 )),$(echo "$r" | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1/'),$(echo "$r" | grep -o '"reason":"[^"]*"' | cut -c1-90)" >> $csv; sleep 3; done; }
tl(){ awk -F, '{printf "%s:%s ", $1, $2}' $1; }
first503(){ awk -F, '$2==503{print $1; exit}' $1; }
last503(){ awk -F, '$2==503{l=$1} END{print l}' $1; }
counts(){ awk -F, -v c=$2 '{print $c}' $1 | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }

log "=== branch $(git -C $WT rev-parse --short HEAD); probe timeout 20 s; transport timeouts 60 s"
log "=== A. d1 with W = 20 s"
kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d_d1plug_w20.json 2p2d_tp8_pr20a $WT; wait_up $D1 d1
log "   idle: $(code_of $D1)"
log "--- A1. one 127999-token prompt, 32 tokens, non-streaming (TTFT above W)"
pair $PROMPTS/127999.jsonl 32 0 400 $OUT/A1_req.txt & PID=$!; probe_during $D1 $OUT/A1.csv $PID; wait $PID
log "   request (code total_s first_byte_s): $(tail -1 $OUT/A1_req.txt); probes: $(tl $OUT/A1.csv)"
log "   first 503 at +$(first503 $OUT/A1.csv) s, last 503 at +$(last503 $OUT/A1.csv) s; reason: $(awk -F, '$2==503{print $3; exit}' $OUT/A1.csv)   (expected 503 from ~21 s until the first token, then 200)"
sleep 5; log "   after: $(code_of $D1)"
log "--- A2. one 8192-token non-streaming generation (duration above W)"
pair $PROMPTS/8192.jsonl 8192 0 400 $OUT/A2_req.txt & PID=$!; probe_during $D1 $OUT/A2.csv $PID; wait $PID
log "   request: $(tail -1 $OUT/A2_req.txt); probes: $(tl $OUT/A2.csv)"
log "   first 503 at +$(first503 $OUT/A2.csv) s, last 503 at +$(last503 $OUT/A2.csv) s   (expected 503 from ~21 s until the request finishes)"
sleep 5; log "   after: $(code_of $D1)"
log "--- A3. the same 8192-token generation, streaming"
pair $PROMPTS/8192.jsonl 8192 1 400 $OUT/A3_req.txt & PID=$!; probe_during $D1 $OUT/A3.csv $PID; wait $PID
log "   request: $(tail -1 $OUT/A3_req.txt); probes: $(counts $OUT/A3.csv 2)   (expected 200 throughout: every chunk advances last_time)"
log "=== B. d1 with W = 48 s and a 15 s running timeout (hook inert without its flag)"
rm -f $R/2p2d_tp8/pr16/abort400.flag
kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d_d1plug_abort400.json 2p2d_tp8_pr20b $WT; wait_up $D1 d1
LOGB=$R/runs/2p2d_tp8_pr20b/d1/server.log
log "--- B1. two 4096-token requests the engine aborts at 15 s"
N0=$(grep -a -c -i "abort" $LOGB)
pair $PROMPTS/8192.jsonl 4096 0 400 $OUT/B1a.txt & PA=$!; pair $PROMPTS/8192.jsonl 4096 0 400 $OUT/B1b.txt & PB=$!
probe_during $D1 $OUT/B1.csv $PA; wait $PA $PB
log "   clients: $(tail -1 $OUT/B1a.txt) | $(tail -1 $OUT/B1b.txt); probes: $(tl $OUT/B1.csv); abort lines +$(( $(grep -a -c -i "abort" $LOGB) - N0 ))   (expected clients 5xx at ~15 s, probes 200 throughout)"
for i in 1 2; do sleep 5; log "   after the aborts: $(code_of $D1)   (expected 200: the table drained)"; done
log "--- B2. client disconnects after 5 s from a streaming 4096-token request"
N0=$(grep -a -c -i "abort" $LOGB)
pair $PROMPTS/8192.jsonl 4096 1 5 $OUT/B2.txt; log "   client: $(tail -1 $OUT/B2.txt)"
for i in 1 2 3; do sleep 4; log "   +$((i*4)) s: $(code_of $D1)"; done
log "   abort lines after the disconnect: +$(( $(grep -a -c -i "abort" $LOGB) - N0 ))   (expected 200 throughout; the entry leaves when the scheduler answers the abort)"
log "=== C. prefill p1 on the branch with hook 1276p (its KV never reaches the decode)"
kill_engine $P1 p1; start_engine $P1 0 gpt-oss_B_128k_2p2d_p1plug_hook1276p.json 2p2d_tp8_pr20c $WT; wait_up $P1 p1
until curl -sf -m 3 $LB/health >/dev/null; do sleep 5; done; log "   router back"
T0=$(date +%s); : > $OUT/C.csv
( for i in 1 2 3; do $PY $S/router_traffic.py 4 4 16 > $OUT/C_traffic_$i.txt 2>&1 & sleep 25; done; wait ) &
for i in $(seq 1 24); do r=$(hg $P1); echo "$(( $(date +%s)-T0 )),$(echo "$r" | sed -E 's/^.* ([0-9]{3}) ([0-9.]+)$/\1/'),$(echo "$r" | grep -o '"reason":"[^"]*"' | cut -c1-100),$(code_of $D1)" >> $OUT/C.csv; sleep 4; done; wait
log "   p1 probes: $(counts $OUT/C.csv 2); first p1 503 at +$(first503 $OUT/C.csv) s; reason: $(awk -F, '$2==503{print $3; exit}' $OUT/C.csv)"
log "   timeline p1 (s:code): $(tl $OUT/C.csv)"
log "   d1 probes meanwhile: $(counts $OUT/C.csv 4); clients: $(cat $OUT/C_traffic_*.txt | tr '\n' ' ')"
log "=== D. restore p1 and d1 to main"
kill_engine $P1 p1; start_engine $P1 0 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up $P1 p1; wait_up $D1 d1
until curl -sf -m 3 $LB/health >/dev/null; do sleep 5; done
log "   router traffic: $($PY $S/router_traffic.py 4 2 32)"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr20 done"
