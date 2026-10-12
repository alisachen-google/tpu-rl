#!/bin/bash
# #1299 diagnostic rerun: door case only, head with install/refusal logs.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/fix_validate_door3; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; BR=pd-decode-door-admission
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [door3] $*"; }
cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin $BR && git bundle create /tmp/door3.bundle origin/$BR 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/door3.bundle sa_101411256518727260534@10.128.15.217:/tmp/door3.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/door3.bundle; $GA fetch -q /tmp/door3.bundle +origin/$BR:$BR && $GA checkout -q --detach $BR && $GA log -1 --format='prefill host on %h'"
git checkout -q --detach origin/$BR && log "decode host on $(git log -1 --format=%h)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q.json door3_door 2>&1 | grep -E 'ready|NOT'
echo "   decode: 'door admission installed' lines = $(grep -rhc 'door admission installed' $R/runs/door3_door/decode/ 2>/dev/null | awk '{s+=$1} END{print s+0}')"
$PY $S/door_probe.py --url $LB --decode $DEC --prefill $PRE --mode door --slots 8 --limit 4 > $OUT/door.txt 2>&1; sed 's/^/   /' $OUT/door.txt | cut -c1-260
echo "   decode: refusal log lines = $(grep -rhc 'door refusal' $R/runs/door3_door/decode/ 2>/dev/null | awk '{s+=$1} END{print s+0}'); prefill: refused rooms logged=$($SSH_A "sudo -n grep -rhc 'decode refused room' /home/alisachen_google_com/pd_disagg/results/runs/door3_door/prefill/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'") bootstrap-failed=$($SSH_A "sudo -n grep -rhc 'Prefill bootstrap failed' /home/alisachen_google_com/pd_disagg/results/runs/door3_door/prefill/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'")"
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main; log "phase door3 done"
