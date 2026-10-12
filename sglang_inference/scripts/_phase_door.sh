#!/bin/bash
# Option 2 (door admission) live validation: PD pair on branch pd-decode-door-admission.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/fix_validate_door; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; DEC=http://10.128.0.90:30000; BR=pd-decode-door-admission
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [door] $*"; }
while pgrep -f '_phase_1191_option1''b.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin $BR && git bundle create /tmp/door.bundle origin/$BR 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/door.bundle sa_101411256518727260534@10.128.15.217:/tmp/door.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/door.bundle; $GA fetch -q /tmp/door.bundle +origin/$BR:$BR && $GA checkout -q --detach $BR && $GA log -1 --format='prefill host on %h'"
git checkout -q --detach origin/$BR && log "decode host on $(git log -1 --format=%h)"
run_case(){ # $1 mode  $2 recipe  $3 name
  log "=== $1: recipe $2"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/$2 $3 2>&1 | grep -E 'ready|NOT'
  if curl -sf -m 30 $LB/health_generate >/dev/null; then
    $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode $1 --slots 8 > $OUT/$1.txt 2>&1; sed 's/^/   /' $OUT/$1.txt | cut -c1-240
    # wasted-work check: fill the slots, send 3 newcomers with known rids, expect 503; the prefill must only ever log them as bootstrap failures
    $PY - <<PY
import json, threading, time, requests
LB="$LB"; H={"Content-Type":"application/json"}
def gen(text, rid=None, max_new=4000, prio=None):
    body={"text":text,"sampling_params":{"max_new_tokens":max_new,"temperature":0}}
    if rid: body["rid"]=rid
    if prio is not None: body["priority"]=prio
    try: r=requests.post(LB+"/generate",json=body,timeout=600); return r.status_code
    except Exception as e: return str(e)[:40]
th=[threading.Thread(target=gen,args=("hold "*8,None,3000)) for _ in range(12)]
[t.start() for t in th]; time.sleep(6)
codes=[gen("door newcomer %d"%i, rid="door-rid-%d"%i, max_new=8) for i in range(3)]
print("   door newcomers door-rid-0..2:", codes)
[t.join(900) for t in th]
PY
    sleep 3; echo "   prefill log: bootstrap failures for door rids = $($SSH_A "sudo -n grep -rhoE 'door-rid-[0-9]' /home/alisachen_google_com/pd_disagg/results/runs/$3/ 2>/dev/null | sort -u | wc -l"); prefill batches mentioning door rids = $($SSH_A "sudo -n grep -rhE 'door-rid-[0-9]' /home/alisachen_google_com/pd_disagg/results/runs/$3/ 2>/dev/null | grep -vciE 'bootstrap|abort'"); scheduler exceptions: decode $(grep -rhcE 'Scheduler hit an exception|AssertionError' $R/runs/$3/ 2>/dev/null | awk '{s+=$1} END{print s+0}'), prefill $($SSH_A "sudo -n grep -rhcE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$3/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'")"
  else echo "   server not ready, skipped"; fi
}
run_case nops gpt-oss_C_128k_q_nops.json  door_nops
run_case q0   gpt-oss_C_128k_ps_q0.json   door_q0
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main; log "phase door done"
