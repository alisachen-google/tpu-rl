#!/bin/bash
# #1299 extended live validation (reworked head): door, nops, low, q0, q1, retract.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/fix_validate_door4; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; BR=pd-decode-door-admission
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [door4] $*"; }
while pgrep -f '_phase_doo''r2.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin $BR && git bundle create /tmp/door4.bundle origin/$BR 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/door4.bundle sa_101411256518727260534@10.128.15.217:/tmp/door4.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/door4.bundle; $GA fetch -q /tmp/door4.bundle +origin/$BR:$BR && $GA checkout -q --detach $BR && $GA log -1 --format='prefill host on %h'"
git checkout -q --detach origin/$BR && log "decode host on $(git log -1 --format=%h)"
run_case(){ # $1 mode $2 recipe $3 name $4 limit
  log "=== $1: recipe $2 (limit $4)"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/$2 $3 2>&1 | grep -E 'ready|NOT'
  if curl -sf -m 30 $LB/health_generate >/dev/null; then
    $PY $S/door_probe.py --url $LB --decode $DEC --prefill $PRE --mode $1 --slots 8 --limit $4 > $OUT/$1.txt 2>&1; sed 's/^/   /' $OUT/$1.txt | cut -c1-260
    echo "   prefill log: big prefill batches (>=4000 new tokens, 8k refusals would show here)=$($SSH_A "sudo -n grep -rhE 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/$3/prefill/ 2>/dev/null | grep -cE 'new-token: ([4-9][0-9]{3}|[0-9]{5,})'")"
    echo "   prefill log: refused rooms logged=$($SSH_A "sudo -n grep -rhc 'decode refused room' /home/alisachen_google_com/pd_disagg/results/runs/$3/prefill/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'") bootstrap-failed warnings=$($SSH_A "sudo -n grep -rhc 'Prefill bootstrap failed' /home/alisachen_google_com/pd_disagg/results/runs/$3/prefill/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'") door rids seen=$($SSH_A "sudo -n grep -rhoE 'door-[a-z]+(-[0-9]+)*' /home/alisachen_google_com/pd_disagg/results/runs/$3/prefill/ 2>/dev/null | sort -u | wc -l"); exceptions: prefill $($SSH_A "sudo -n grep -rhcE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$3/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'") decode $(grep -rhcE 'Scheduler hit an exception|AssertionError' $R/runs/$3/decode/ 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  else echo "   server not ready, skipped"; fi
}
run_case q0      gpt-oss_C_128k_ps_q0.json           door4_q0      0
run_case q1      gpt-oss_C_128k_ps_q1.json           door4_q1      1
run_case retract gpt-oss_A_128k_smallpool_ps_q.json  door4_retract 4
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main; log "phase door2 done"
