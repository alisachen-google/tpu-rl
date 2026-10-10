#!/usr/bin/env bash
# PR #1191 head on both hosts: queue-limit boundary matrix on the pair (gpt-oss), plus the monolithic reference.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_1191_boundary; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
BR=pd-queue-limit-prefill
log(){ echo "$(date -u +%FT%TZ) [1191b] $*"; }
excs(){ echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log 2>/dev/null") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/$1/decode/server.log 2>/dev/null); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); 503s on decode: $(grep -cE '" 503 ' $R/runs/$1/decode/server.log 2>/dev/null)"; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $1 $2 2>&1 | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }

cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin $BR && git bundle create /tmp/q1191b.bundle origin/$BR 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/q1191b.bundle sa_101411256518727260534@10.128.15.217:/tmp/q1191b.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/q1191b.bundle; $GA fetch -q /tmp/q1191b.bundle +origin/$BR:$BR && $GA checkout -q --detach $BR && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q --detach origin/$BR && log "local on $(git log -1 --format=%h)"
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/

run(){ # $1 recipe  $2 run-name  $3 probe mode  $4 slots
  log "=== $3: recipe $1"
  if start $HOME/pd_disagg/recipes/$1.json $2; then
    $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode $3 --slots ${4:-8} > $OUT/$3.txt 2>&1; sed 's/^/   /' $OUT/$3.txt | cut -c1-260; excs $2
  else log "$3: pair did not come up"; fi
}
run gpt-oss_C_128k_ps_q        b1191_base    base
run gpt-oss_C_128k_ps_q_low    b1191_low     low
run gpt-oss_C_128k_ps_q0       b1191_q0      q0
run gpt-oss_C_128k_ps_q1       b1191_q1      q1
run gpt-oss_C_128k_q_nops      b1191_nops    nops
run gpt-oss_C_128k_ps_np       b1191_nolimit nolimit
run gpt-oss_A_128k_smallpool_ps_q b1191_retract retract
log "=== monolithic reference (G, 4 slots, limit 4): base cases Q3-Q5"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_ps_q.json b1191_mono b 2>&1 | grep -E 'ready|NOT'
$PY $S/queue_limit_probe.py --url $DEC --decode $DEC --mode base --slots 4 > $OUT/mono.txt 2>&1; grep -E '^\[Q[1-5]|^\[Q-end' $OUT/mono.txt | sed 's/^/   /' | cut -c1-260
echo "   mono exceptions: $(grep -cE 'Scheduler hit an exception' $R/runs/b1191_mono/golden/server.log); health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1191 boundary done"
