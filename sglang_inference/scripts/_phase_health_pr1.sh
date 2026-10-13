#!/usr/bin/env bash
# Live validation of PR #1339 (/health_generate answered only by the probe's own request) on the 2P/2D TP-8 group.
#  A. d1 on the branch (no hook), d2 on main: idle probes and probes under load on both -> all 200 (behaviour unchanged).
#  B. d1 branch+starve hook, d2 main+starve hook, real traffic flowing: the hook admits the probe and never schedules it.
#     main (d2) must still answer 200 from other requests' messages (the bug); the branch (d1) must answer 503 (the fix).
#  C. restore both decodes to main with the standard recipe.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/pr1; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com; WT=/home/$ME/wt-health-rid
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [pr1] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
# patterns use [s] so the pkill command line never matches itself
kill_engine(){ local ip=$1 label=$2; on $ip "pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; sleep 3; true"
  local t0=$(date +%s); while curl -sf -m 3 http://$ip:30000/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   $label down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ # ip rank recipe run [sglang_root]
  local ip=$1 rank=$2 recipe=$3 run=$4 root=${5:-/home/$ME/sglang-torchtpu}
  on $ip "cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$recipe $rank $run &"; }
wait_up(){ local ip=$1 label=$2 t0=$(date +%s); until curl -sf -m 5 http://$ip:30000/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done
  log "   $label health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$ip:30000/health) after $(( $(date +%s)-t0 )) s"; }
elog(){ local ip=$1 run=$2 label=$3; if [[ $ip == $D1 ]]; then cat $R/runs/$run/$label/server.log; else $SSH sa_101411256518727260534@$ip "sudo cat /home/$ME/pd_disagg/results/runs/$run/$label/server.log" 2>/dev/null; fi; }
hg(){ curl -s -m 40 -o /dev/null -w '%{http_code} %{time_total}' http://$1:30000/health_generate; }
hg_series(){ local ip=$1 label=$2 n=$3 file=$4; for i in $(seq 1 $n); do echo "$(date -u +%T),$label,$(hg $ip | tr ' ' ',')" | tee -a $file; done; }
one_request(){ local req=$(mktemp); $PY -c "import json; t=json.loads(open('$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']; json.dump({'text':t,'sampling_params':{'max_new_tokens':256,'temperature':0}}, open('$req','w'))"
  curl -s -m 300 $LB/generate -H 'Content-Type: application/json' -d @$req -o /dev/null -w '%{http_code} %{time_total}\n'; rm -f $req; }
summarize(){ awk -F, -v l=$2 '$2==l {c[$3]++; if($4>m[$3])m[$3]=$4} END {for(k in c) printf "%s x%d (max %.1fs) ", k, c[k], m[k]}' $1; }

log "=== PR #1339 live validation; branch $(git -C $WT rev-parse --short HEAD) on d1, main $(git -C /home/$ME/sglang-torchtpu rev-parse --short HEAD) on d2"
log "=== A. relaunch d1 on the branch (no hook)"
kill_engine $D1 d1; start_engine $D1 2 gpt-oss_B_128k_2p2d_d1branch.json 2p2d_tp8_pr1 $WT; wait_up $D1 d1 & wait_up $D2 d2 & wait
log "   d1 sglang import path: $(elog $D1 2p2d_tp8_pr1 d1 | grep -a -o 'sglang=[0-9a-f]*' | head -1)"
: > $OUT/A_idle.csv; log "   idle probes"; hg_series $D1 d1-branch 3 $OUT/A_idle.csv | sed 's/^/      /'; hg_series $D2 d2-main 3 $OUT/A_idle.csv | sed 's/^/      /'
log "   under load: bench_serving 8k/256 c=8 n=48 through the router, probes every 2 s"
: > $OUT/A_busy.csv
( cd $WT && $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 8192 --random-output-len 256 --random-range-ratio 1 --num-prompts 48 --max-concurrency 8 --tokenizer /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/$(ls /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/ | head -1) --disable-tqdm > $OUT/A_bench.txt 2>&1 ) & BENCH=$!
while kill -0 $BENCH 2>/dev/null; do echo "$(date -u +%T),d1-branch,$(hg $D1 | tr ' ' ',')" >> $OUT/A_busy.csv; echo "$(date -u +%T),d2-main,$(hg $D2 | tr ' ' ',')" >> $OUT/A_busy.csv; sleep 2; done
log "   bench: $(grep -E 'Successful requests|Request throughput' $OUT/A_bench.txt | tr -s ' ' | paste -sd';')"
log "   busy probes: d1-branch $(summarize $OUT/A_busy.csv d1-branch); d2-main $(summarize $OUT/A_busy.csv d2-main)"
log "   d1 running/queue after: $(curl -s -m 5 http://$D1:30000/metrics | grep -E '^sglang:num_(running|queue)_reqs\{[^}]*tp_rank=\"0\"' | sed -E 's/\{[^}]*\}//' | paste -sd' ')"

log "=== B. starve the probe on both decodes (hook), d1 on the branch, d2 on main; real traffic every 3 s"
kill_engine $D1 d1 & kill_engine $D2 d2 & wait
start_engine $D1 2 gpt-oss_B_128k_2p2d_d1branch_starve.json 2p2d_tp8_pr1s $WT; start_engine $D2 3 gpt-oss_B_128k_2p2d_d2main_starve.json 2p2d_tp8_pr1s
wait_up $D1 d1 & wait_up $D2 d2 & wait
log "   hook lines: d1 $(elog $D1 2p2d_tp8_pr1s d1 | grep -a -c 'HOOK-STARVE.*installed') installs, d2 $(elog $D2 2p2d_tp8_pr1s d2 | grep -a -c 'HOOK-STARVE.*installed') installs"
: > $OUT/B_traffic.txt; ( end=$(( $(date +%s) + 240 )); while (( $(date +%s) < end )); do one_request >> $OUT/B_traffic.txt; sleep 3; done ) & TRAFFIC=$!
sleep 6
: > $OUT/B_starved.csv; log "   probes (each up to the 20 s health timeout):"
for i in 1 2 3; do hg_series $D2 d2-main 1 $OUT/B_starved.csv | sed 's/^/      /'; hg_series $D1 d1-branch 1 $OUT/B_starved.csv | sed 's/^/      /'; done
wait $TRAFFIC 2>/dev/null
log "   traffic during the test: $(awk '{print $1}' $OUT/B_traffic.txt | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}') ($(wc -l < $OUT/B_traffic.txt) requests)"
log "   starved probes: d2-main $(summarize $OUT/B_starved.csv d2-main); d1-branch $(summarize $OUT/B_starved.csv d1-branch)"
log "   d1 handler log: $(elog $D1 2p2d_tp8_pr1s d1 | grep -a -c 'got no answer of its own') x 'no answer of its own'; dropped probes: d1 $(elog $D1 2p2d_tp8_pr1s d1 | grep -a -c 'dropped probe'), d2 $(elog $D2 2p2d_tp8_pr1s d2 | grep -a -c 'dropped probe')"
log "   d2 (main) handler log: $(elog $D2 2p2d_tp8_pr1s d2 | grep -a -c 'Health check failed') x 'Health check failed'"

log "=== C. restore both decodes to main with the standard recipe"
kill_engine $D1 d1 & kill_engine $D2 d2 & wait
start_engine $D1 2 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; start_engine $D2 3 gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3
wait_up $D1 d1 & wait_up $D2 d2 & wait
: > $OUT/C_restored.txt; for i in 1 2 3 4; do one_request >> $OUT/C_restored.txt; done; log "   restored traffic: $(awk '{print $1}' $OUT/C_restored.txt | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')"
log "   restored probes: d1 $(hg $D1); d2 $(hg $D2)"
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase pr1 done"
