#!/usr/bin/env bash
# Live validation of PR #1341 (/health liveness from loop heartbeats) on the 2P/2D TP-8 group. d1 (this host) runs the
# branch with SGLANG_HEALTH_HEARTBEAT_MAX_AGE_S=20.
#  A. idle 3 min: /health stays 200; the board shows 8 scheduler slots + the detokenizer, all fresh.
#  B. under load (bench 8k/256 c=8): /health stays 200.
#  C. SIGSTOP one scheduler rank: /health -> 503 within 20 s naming the stuck rank(s); SIGCONT -> 200 again; traffic OK.
#  D. restore d1 to main.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr3; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-beat
log(){ echo "$(date -u +%FT%TZ) [pr3] $*"; }
kill_d1(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 http://$D1:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   d1 down after $(( $(date +%s)-t0 )) s"; }
start_d1(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}
  ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & )
  local t0=$(date +%s); until curl -sf -m 5 http://$D1:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   d1 ($recipe) health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$D1:30000/health) after $(( $(date +%s)-t0 )) s"; }
h(){ curl -s -m 10 -w ' %{http_code} %{time_total}' http://$D1:30000/health | tr -d '\n' | cut -c1-400; }
board(){ PYTHONPATH=$WT/python:$WT $PY - <<'PY' 2>/dev/null | tail -n 12
import glob, time
from sglang.srt.managers.heartbeat import HeartbeatReader
for p in sorted(glob.glob('/dev/shm/sglang_heartbeat_*.shm'), key=lambda f: -__import__('os').path.getmtime(f))[:1]:
    r = HeartbeatReader(p); slots = r.read()
    print(f"board {p}: {len(slots)} slots; " + ", ".join(f"{s.role}[{s.rank}]={s.age_s:.1f}s{'(paused)' if s.paused else ''}" for s in sorted(slots, key=lambda s:(s.role,s.rank))))
PY
}
poll(){ local secs=$1 every=$2 file=$3; local end=$(( $(date +%s) + secs )); while (( $(date +%s) < end )); do echo "$(date -u +%T) $(h)" >> $file; sleep $every; done; }
codes(){ awk '{print $(NF-1)}' $1 | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }

log "=== PR #1341 live validation; branch $(git -C $WT rev-parse --short HEAD) on d1, heartbeat max age 20 s"
log "=== A. relaunch d1 on the branch; idle 3 min"
kill_d1; start_d1 gpt-oss_B_128k_2p2d_d1beat.json 2p2d_tp8_pr3 $WT
log "   $(board)"
: > $OUT/A_idle.txt; poll 180 5 $OUT/A_idle.txt; log "   idle /health over 3 min: $(codes $OUT/A_idle.txt) (n=$(wc -l < $OUT/A_idle.txt))"; log "   $(board)"

log "=== B. under load: bench_serving 8k/256 c=8 n=48 through the router, /health every 2 s"
: > $OUT/B_busy.txt
( cd $WT && $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 8192 --random-output-len 256 --random-range-ratio 1 --num-prompts 48 --max-concurrency 8 --tokenizer /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/$(ls /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/ | head -1) --disable-tqdm > $OUT/B_bench.txt 2>&1 ) & BENCH=$!
while kill -0 $BENCH 2>/dev/null; do echo "$(date -u +%T) $(h)" >> $OUT/B_busy.txt; sleep 2; done
log "   bench: $(grep -E 'Successful requests|Request throughput' $OUT/B_bench.txt | tr -s ' ' | paste -sd';')"; log "   /health under load: $(codes $OUT/B_busy.txt) (n=$(wc -l < $OUT/B_busy.txt)); $(board)"

log "=== C. SIGSTOP scheduler rank TP3 on d1; /health must turn 503 within 20 s, naming the stalled ranks"
pid=$(pgrep -u $ME -f 'sglang::scheduler_TP3' | head -1); log "   TP3 pid $pid"; kill -STOP $pid; t0=$(date +%s)
: > $OUT/C_stop.txt; first503=""; while (( $(date +%s)-t0 < 70 )); do line="$(date -u +%T) $(h)"; echo "$line" >> $OUT/C_stop.txt; code=$(echo "$line" | awk '{print $(NF-1)}'); if [[ $code == 503 && -z $first503 ]]; then first503=$(( $(date +%s)-t0 )); fi; sleep 2; done
log "   first 503 after ${first503:-none} s; codes: $(codes $OUT/C_stop.txt)"; log "   last body: $(tail -n 1 $OUT/C_stop.txt | cut -c10-330)"; log "   $(board)"
kill -CONT $pid; t1=$(date +%s); : > $OUT/C_cont.txt; first200=""; while (( $(date +%s)-t1 < 40 )); do line="$(date -u +%T) $(h)"; echo "$line" >> $OUT/C_cont.txt; code=$(echo "$line" | awk '{print $(NF-1)}'); if [[ $code == 200 && -z $first200 ]]; then first200=$(( $(date +%s)-t1 )); fi; sleep 2; done
log "   SIGCONT: first 200 after ${first200:-none} s; codes: $(codes $OUT/C_cont.txt); $(board)"
req=$(mktemp); $PY -c "import json; t=json.loads(open('$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']; json.dump({'text':t,'sampling_params':{'max_new_tokens':64,'temperature':0}}, open('$req','w'))"
log "   traffic after recovery: $(for i in 1 2 3 4; do curl -s -m 300 $LB/generate -H 'Content-Type: application/json' -d @$req -o /dev/null -w '%{http_code} '; done)"; rm -f $req

log "=== D. restore d1 to main with the standard recipe"
kill_d1; start_d1 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3
log "   d1 /health: $(h)"; log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr3 done"
