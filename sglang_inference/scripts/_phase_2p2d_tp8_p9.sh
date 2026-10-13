#!/usr/bin/env bash
# 2P/2D TP-8, phase 9: soak. DURATION_MIN (default 120) of mixed traffic through the router -- bench_serving random 8k/256 at c=8,
# MT-Bench slices at c=4, decode-phase abort bursts -- with queue gauges, failure counters and host RAM (RSS of all sglang
# processes) sampled per engine every round. Pass: gauges drain to 0 every round, bootstrap/transfer failure counters flat,
# RSS growth per host under RSS_GROWTH_LIMIT_GB (default 4) between the first and last round, all engines healthy at the end.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p9; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080; ME=alisachen_google_com
DURATION_MIN=${DURATION_MIN:-120}; RSS_GROWTH_LIMIT_GB=${RSS_GROWTH_LIMIT_GB:-4}
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p9] $*"; }
on(){ local ip=$1; shift; if [[ $ip == $D1 ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
rss_gb(){ # host RSS of the engine tree in GB; the remote path goes straight over ssh as the SA user (ps -u works without sudo), because the sudo -i / bash -lc layers in on() swallow the awk $1
  local cmd="ps -u $ME -o rss= -p \$(pgrep -u $ME -d, -f sglang) 2>/dev/null | awk '{s+=\$1} END {printf \"%.1f\", s/1048576}'"
  if [[ $1 == $D1 ]]; then bash -c "$cmd"; else $SSH sa_101411256518727260534@$1 "$cmd"; fi; }
sample(){ # one CSV row per engine: ts,engine,running,queue,prealloc,transfer,bootstrap_failed,transfer_failed,rss_gb,health
  local ts=$(date -u +%FT%TZ); for e in "$P1 p1" "$P2 p2" "$D1 d1" "$D2 d2"; do set -- $e
    local m=$(curl -s -m 5 http://$1:30000/metrics); local g=$(echo "$m" | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs|num_bootstrap_failed_reqs_total|num_transfer_failed_reqs_total)\{[^}]*tp_rank="0"' | sed -E 's/\{[^}]*\}//' | awk '{a[$1]=$2} END {printf "%s,%s,%s,%s,%s,%s", a["sglang:num_running_reqs"]+0, a["sglang:num_queue_reqs"]+0, a["sglang:num_decode_prealloc_queue_reqs"]+0, a["sglang:num_decode_transfer_queue_reqs"]+0, a["sglang:num_bootstrap_failed_reqs_total"]+0, a["sglang:num_transfer_failed_reqs_total"]+0}')
    echo "$ts,$2,$g,$(rss_gb $1),$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$1:30000/health)" | tee -a $OUT/samples.csv | sed 's/^/   /'; done; }
SNAP=$(ls -d /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/* | head -1); Q=$HOME/pd_disagg/prompts/mt_bench/question.jsonl; P8K=$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl
echo "ts,engine,running,queue,prealloc,transfer,bootstrap_failed,transfer_failed,rss_gb,health" > $OUT/samples.csv
log "=== soak start: $DURATION_MIN min; sample 0"; sample
t_end=$(( $(date +%s) + DURATION_MIN*60 )); round=0
while (( $(date +%s) < t_end )); do round=$((round+1)); log "--- round $round"
  $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 8192 --random-output-len 256 --random-range-ratio 1.0 --num-prompts 32 --max-concurrency 8 --disable-tqdm 2>&1 | grep -E 'Successful requests|Failed requests|Request throughput' | sed -E 's/ +/ /g; s/^/   bench: /'
  $PY $S/mtbench_runner.py --url $LB --questions $Q --snapshot $SNAP --n 20 --concurrency 4 --max-new-tokens 256 --out $OUT/mtbench_r$round.jsonl --label soak_r$round 2>&1 | grep -E '^\[' | head -n 1 | sed 's/^/   /'
  $PY $S/abort_test.py --url $LB --prompt-file $P8K --phase decode --abort-after 0.5 --repeat 4 --concurrent 2 --server-info http://$D1:30000 --server-info http://$D2:30000 --settle 20 2>&1 | grep -E 'PASS|FAIL' | sed 's/^/   /'
  sleep 15; sample; done
log "=== soak end after $round rounds; verdict"
$PY - $OUT/samples.csv $RSS_GROWTH_LIMIT_GB <<'PY'
import csv,sys,collections
rows=list(csv.DictReader(open(sys.argv[1]))); lim=float(sys.argv[2]); by=collections.defaultdict(list)
for r in rows: by[r['engine']].append(r)
ok=True
for e,rs in by.items():
    first,last=rs[0],rs[-1]; rss=[float(r['rss_gb']) for r in rs if r['rss_gb']]  # tolerate rows whose sampler returned nothing
    nz=[r['ts'] for r in rs if any(float(r[k])>0 for k in ('running','queue','prealloc','transfer'))]
    grow=(rss[-1]-rss[0]) if len(rss)>1 else 0.0; bf=float(last['bootstrap_failed'])-float(first['bootstrap_failed']); tf=float(last['transfer_failed'])-float(first['transfer_failed'])
    bad=[]; 
    if nz: bad.append(f'gauges non-zero at {len(nz)} sample(s)')
    if bf or tf: bad.append(f'bootstrap_failed +{bf:.0f} transfer_failed +{tf:.0f}')
    if grow>lim: bad.append(f'RSS grew {grow:.1f} GB')
    if len(rss)<len(rs): bad.append(f'rss missing at {len(rs)-len(rss)} sample(s)')
    if last['health']!='200': bad.append(f'health {last["health"]}')
    print(f"   {e}: samples {len(rs)}, RSS {rss[0] if rss else '?'} -> {rss[-1] if rss else '?'} GB ({grow:+.1f}), failures +{bf:.0f}/+{tf:.0f}, {'PASS' if not bad else 'FAIL: '+'; '.join(bad)}"); ok&=not bad
print('   SOAK', 'PASS' if ok else 'FAIL')
PY
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p9 done"
