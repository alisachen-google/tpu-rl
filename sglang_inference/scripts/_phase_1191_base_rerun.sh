#!/usr/bin/env bash
# Rerun the base configuration of the #1191 matrix with the redesigned Q7/Q10 cases (after the other phases).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_1191_boundary
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [1191b2] $*"; }
while pgrep -f '1191_boundar''y.sh|1230_122''2.sh' >/dev/null; do sleep 60; done
cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin pd-queue-limit-prefill && git bundle create /tmp/q1191c.bundle origin/pd-queue-limit-prefill 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/q1191c.bundle sa_101411256518727260534@10.128.15.217:/tmp/q1191c.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/q1191c.bundle; $GA fetch -q /tmp/q1191c.bundle +origin/pd-queue-limit-prefill:pd-queue-limit-prefill && $GA checkout -q --detach pd-queue-limit-prefill && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q --detach origin/pd-queue-limit-prefill && log "local on $(git log -1 --format=%h)"
log "=== base rerun (Q7/Q10 redesigned): recipe gpt-oss_C_128k_ps_q"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q.json b1191_base2 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode base --slots 8 > $OUT/base2.txt 2>&1; sed 's/^/   /' $OUT/base2.txt | cut -c1-260
  echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/b1191_base2/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/b1191_base2/decode/server.log); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
fi
log "=== q0 rerun (empty-queue eviction guard): recipe gpt-oss_C_128k_ps_q0"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q0.json b1191_q0b 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode q0 --slots 8 > $OUT/q0b.txt 2>&1; sed 's/^/   /' $OUT/q0b.txt | cut -c1-260
  echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/b1191_q0b/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/b1191_q0b/decode/server.log); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
fi
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1191 base rerun done"
