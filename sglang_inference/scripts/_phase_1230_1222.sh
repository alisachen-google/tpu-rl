#!/usr/bin/env bash
# Live validation of two teammate PRs: #1230 (decode handshake bounded by the bootstrap timeout) and #1222 (--kv-canary rejected at TPU arg validation).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_1230_1222; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [1230/1222] $*"; }
excs(){ echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log 2>/dev/null") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/$1/decode/server.log 2>/dev/null); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); prefill inflight $(curl -s -m 5 $PRE/metrics | grep -E '^sglang:num_prefill_inflight_queue_reqs' | head -n1 | sed -E 's/.* //')"; }
deploy(){ # $1 ref (origin/<branch>)  $2 local branch name
  cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin ${1#origin/} && git bundle create /tmp/v.bundle $1 2>/dev/null
  scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/v.bundle sa_101411256518727260534@10.128.15.217:/tmp/v_$2.bundle
  $SSH_A "sudo -n chown alisachen_google_com /tmp/v_$2.bundle; $GA fetch -q /tmp/v_$2.bundle +$1:$2 && $GA checkout -q --detach $2 && echo '   host a on' \$($GA log -1 --format=%h)"
  git checkout -q --detach $1 && log "local on $(git log -1 --format='%h %s' | cut -c1-80)"
}
restore(){ $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $1 $2 2>&1 | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
while pgrep -f '1191_boundar''y.sh' >/dev/null; do sleep 60; done
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/

########## PR #1230
deploy origin/amandaliang-raiden-handshake-timeout pr1230
log "=== 1230/A orphans: Config A 8 slots, BOOTSTRAP_TIMEOUT 60"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_bt60.json v1230_orphans; then
  $PY $S/handshake_timeout_probe.py --lb $LB --decode $DEC --prefill $PRE --prefill-host 10.128.15.217 --timeout 60 --mode orphans > $OUT/1230_orphans.txt 2>&1; sed 's/^/   /' $OUT/1230_orphans.txt | cut -c1-300; excs v1230_orphans
  echo "   decode log, timeout messages: $(grep -cE 'timed out after .* before its KV metadata' $R/runs/v1230_orphans/decode/server.log); sample: $(grep -m1 -oE 'Request [0-9]+ timed out after [0-9.]+s in KVPoll\.[A-Za-z]+ before its KV metadata was sent to the prefill' $R/runs/v1230_orphans/decode/server.log)"
else log "1230/A pair did not come up"; fi
log "=== 1230/B KV pressure: decode pool 262k, 8 x 131k, BOOTSTRAP_TIMEOUT 60"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_smallpool_bt60.json v1230_pressure; then
  $PY $S/handshake_timeout_probe.py --lb $LB --decode $DEC --prefill $PRE --prefill-host 10.128.15.217 --timeout 60 --mode pressure > $OUT/1230_pressure.txt 2>&1; sed 's/^/   /' $OUT/1230_pressure.txt | cut -c1-300; excs v1230_pressure
  echo "   decode log, timeout messages: $(grep -cE 'timed out after .* before its KV metadata' $R/runs/v1230_pressure/decode/server.log)"
else log "1230/B pair did not come up"; fi
log "=== 1230/C the F-12 trigger: single slot 128k, BOOTSTRAP/WAITING 300 (before: request hung until the client gave up)"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_mr1_t300.json v1230_f12; then
  t0=$(date +%s); code=$(curl -s -m 900 -o $OUT/1230_f12_first.json -w '%{http_code}' $LB/generate -H 'Content-Type: application/json' -d "$(python3 -c "import json; print(json.dumps({'text': json.loads(open('$HOME/pd_disagg/prompts/gpt-oss/8192.jsonl').readline()).get('text'), 'sampling_params': {'temperature':0,'max_new_tokens':64,'ignore_eos':True}}))")"); echo "   first 8k request: HTTP $code after $(( $(date +%s)-t0 ))s; body: $(head -c 240 $OUT/1230_f12_first.json | tr '\n' ' ')"
  sleep 5; excs v1230_f12
  echo "   decode timeout messages: $(grep -cE 'timed out after .* before its KV metadata' $R/runs/v1230_f12/decode/server.log); prefill 'timed out .* waiting for the decode': $($SSH_A "sudo -n grep -cE 'timed out after .* waiting for the decode' /home/alisachen_google_com/pd_disagg/results/runs/v1230_f12/prefill/server.log")"
  t0=$(date +%s); code=$(curl -s -m 400 -o $OUT/1230_f12_second.json -w '%{http_code}' $LB/generate -H 'Content-Type: application/json' -d '{"text":"Second request after the deadline.","sampling_params":{"temperature":0,"max_new_tokens":16,"ignore_eos":true}}'); echo "   second request: HTTP $code after $(( $(date +%s)-t0 ))s; body: $(head -c 160 $OUT/1230_f12_second.json | tr '\n' ' ')"; excs v1230_f12
else log "1230/C pair did not come up"; fi
log "=== 1230/D control on main: one orphan room, BOOTSTRAP_TIMEOUT 60 (expect: no deadline, client gives up at 240 s)"
restore
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_bt60.json v1230_main; then
  t0=$(date +%s); code=$(curl -s -m 240 -o $OUT/1230_main_orphan.json -w '%{http_code}' $DEC/generate -H 'Content-Type: application/json' -d '{"text":"Orphan room on main.","sampling_params":{"temperature":0,"max_new_tokens":8,"ignore_eos":true},"bootstrap_host":"10.128.15.217","bootstrap_port":8998,"bootstrap_room":123456789012}'); echo "   main orphan: HTTP ${code:-000} after $(( $(date +%s)-t0 ))s (000 = client timed out); body: $(head -c 160 $OUT/1230_main_orphan.json 2>/dev/null | tr '\n' ' ')"
  sleep 3; echo "   decode prealloc queue after client gave up: $(curl -s -m 5 $DEC/metrics | grep -E '^sglang:num_decode_prealloc_queue_reqs' | head -n1 | sed -E 's/.* //'); timeout messages in decode log: $(grep -cE 'timed out after .* before its KV metadata' $R/runs/v1230_main/decode/server.log)"; excs v1230_main
else log "1230/D pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1

########## PR #1222
deploy origin/amandaliang-kv-canary-tpu pr1222
canary_launch(){ # $1 tag  -> launches monolithic on b, reports exit time and message
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 60; done
  t0=$(date +%s); env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 900 bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_canary_$1.json v1222_$1 b > $OUT/1222_$1.driver.log 2>&1; rc=$?; dt=$(( $(date +%s)-t0 ))
  L=$R/runs/v1222_$1/golden/server.log
  echo "   kv_canary=$1: launcher rc=$rc after ${dt}s; health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); message: $(grep -m1 -oE 'kv-canary [a-z]+ is not supported on TPU[^\"]{0,80}|invalid choice[^\n]{0,80}|NotImplementedError[^\n]{0,80}|k_buffer[^\n]{0,60}' $L | head -n 1); weights loaded before exit: $(grep -cE 'Load weight end|Memory pool end' $L); leftover sglang procs: $(pgrep -fc 'sglang::scheduler')"
}
log "=== 1222/A monolithic launches on the PR head"
for tag in log raise bogus; do canary_launch $tag; bash $S/pd.sh stop >/dev/null 2>&1; done
canary_launch none; if curl -sf -m 10 $DEC/health >/dev/null; then echo "   kv_canary=none serves: HTTP $(curl -s -m 120 -o /dev/null -w '%{http_code}' $DEC/generate -H 'Content-Type: application/json' -d '{"text":"The three primary colors are","sampling_params":{"temperature":0,"max_new_tokens":8}}')"; fi; bash $S/pd.sh stop >/dev/null 2>&1
log "=== 1222/B pair with --kv-canary log on the decode only"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 60; done
t0=$(date +%s); env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 900 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_canary_decode_only.json v1222_pair 2>&1 | grep -E 'ready|NOT' | sed 's/^/   /'; echo "   pair launcher returned after $(( $(date +%s)-t0 ))s; decode message: $(grep -m1 -oE 'kv-canary [a-z]+ is not supported on TPU[^\"]{0,80}' $R/runs/v1222_pair/decode/server.log); prefill health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) (prefill ran with kv_canary none)"
bash $S/pd.sh stop >/dev/null 2>&1
log "=== 1222/C control on main: --kv-canary log (before: crash after weights load)"
restore; canary_launch log; bash $S/pd.sh stop >/dev/null 2>&1
log "phase 1230/1222 done"
