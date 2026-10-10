#!/usr/bin/env bash
# #1230 follow-up: orphan rooms that pass the handshake wait in the KV window, bounded by SGLANG_DISAGGREGATION_WAITING_TIMEOUT (here 90 s).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_1230_1222
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [1230wt] $*"; }
while pgrep -f '1230_122''2.sh|1191_base_reru''n.sh' >/dev/null; do sleep 60; done
cd $HOME/sglang-torchtpu && git checkout -q main && $SSH_A "$GA checkout -q --detach pr1230 && echo '   host a on' \$($GA log -1 --format=%h)"; git checkout -q --detach origin/amandaliang-raiden-handshake-timeout && log "local on $(git log -1 --format=%h)"
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/
log "=== 1230/E orphans with WAITING_TIMEOUT 90 (BOOTSTRAP 60): the post-metadata window"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_bt60_wt90.json v1230_wt90 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/handshake_timeout_probe.py --lb $LB --decode $DEC --prefill $PRE --prefill-host 10.128.15.217 --timeout 90 --message "timed out" --mode orphans > $OUT/1230_orphans_wt90.txt 2>&1; sed 's/^/   /' $OUT/1230_orphans_wt90.txt | cut -c1-300
  echo "   decode log: handshake-window messages $(grep -cE 'before its KV metadata' $R/runs/v1230_wt90/decode/server.log); waiting-window messages $(grep -cE 'timed out' $R/runs/v1230_wt90/decode/server.log); sample: $(grep -m1 -oE 'Request [0-9]+ timed out[^\"]{0,120}' $R/runs/v1230_wt90/decode/server.log)"
  echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/v1230_wt90/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/v1230_wt90/decode/server.log); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
fi
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1230 wt done"
