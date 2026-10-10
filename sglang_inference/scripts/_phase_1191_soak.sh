#!/usr/bin/env bash
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_1191_boundary
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [1191soak] $*"; }
while pgrep -f '1191_base_reru''n.sh|1230_w''t.sh' >/dev/null; do sleep 60; done
cd $HOME/sglang-torchtpu && git checkout -q main && $SSH_A "$GA checkout -q --detach pd-queue-limit-prefill && echo '   host a on' \$($GA log -1 --format=%h)"; git checkout -q --detach origin/pd-queue-limit-prefill && log "local on $(git log -1 --format=%h)"
log "=== retract rerun on the PR head (stalled once at 10:05 UTC; main control passed): recipe gpt-oss_A_128k_smallpool_ps_q"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_smallpool_ps_q.json b1191_retract2 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode retract --slots 8 > $OUT/retract2.txt 2>&1; sed 's/^/   /' $OUT/retract2.txt | cut -c1-300
  echo "   decode: last batch $(grep -E 'Decode batch' $R/runs/b1191_retract2/decode/server.log | tail -n 1 | grep -oE '^\[[^]]+\]|#running-req: [0-9]+|#prealloc-req: [0-9]+' | tr '\n' ' '); fence: $(grep -E 'KV free fence' $R/runs/b1191_retract2/decode/server.log | tail -n 1 | grep -oE 'pages held [0-9]+')"
  echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/b1191_retract2/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/b1191_retract2/decode/server.log)"
fi
log "=== soak: recipe gpt-oss_C_128k_ps_q"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q.json b1191_soak 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode soak --slots 8 > $OUT/soak.txt 2>&1; sed 's/^/   /' $OUT/soak.txt | cut -c1-300
  echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/b1191_soak/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/b1191_soak/decode/server.log); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); 503s on decode: $(grep -cE '" 503 ' $R/runs/b1191_soak/decode/server.log)"
fi
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1191 soak done"
